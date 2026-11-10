"""Phase 3 final experiment: classical baselines vs SASRec and its ablations (synthetic only).

Gates, in this order, before ANY dataset is written (a failure stops the run):
  1. default test suite (`pytest`)       2. CUDA validation run (`pytest -m cuda`, must run)
  3. every pinned manifest + the frozen SASRec configuration hash
  4. require_device("cuda:0")              5. `phase3_final` seed namespace
  6. pipeline versions and runtime fingerprint equal to those recorded at the freeze;
     output-directory and existing-lock checks.

Per final dataset (3 profiles x 10 seeds, 500 customers, 30 days, split 16/7/7):
- Generate (generator 0.3.0), validate, consent-aware temporal split.
- Classical models (Popularity, Personal Frequency, ItemKNN self/no-self, MF) with
  the unchanged Phase 2.3 grids: suites `phase3` (window) and `phase3-next-event`.
  Random = the multi-seed random floor of those runs.
- SASRec full / no_position / shuffled / meanpool / session_boundary with the frozen
  configuration (trial 21) on CUDA: fit on train next-event cases, early stopping on
  val, then ONE scoring pass on test. The same model is applied to the window protocol
  (its next-item scores rank the set target; no window-specific selection).
- Every model is scored on the same cases, candidates and relevance with the shared
  evaluator and the shared tie-break (train popularity, then item id); case ids are
  asserted identical across model families.
- Paired bootstrap differences within each seed (customer clusters for next-event,
  customer-task cases for window), and next-event metrics split by the evaluator's
  within-/cross-session label (never a model input).

Across seeds: mean, SD and Student-t 95% interval over the 10 seeds per profile (see
bri.experiments.robustness). Within-seed and across-seed intervals are never pooled.
SASRec uses one training seed everywhere: across-seed variation is DATASET variation,
not model-initialization variation.

Usage:
    python -m bri.experiments.phase3 run --output-dir artifacts/phase3/final
    python -m bri.experiments.phase3 lock --output-dir artifacts/phase3/final   (once)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bri.experiments import integrity as I
from bri.experiments.baselines import (FROZEN_MARKER, RUNNER_VERSION, experiment_config, load_processed,
                                       protocol_config, run_baselines, run_next_event)
from bri.experiments.robustness import across_seed_stats
from bri.tasks import SYNTHETIC_LABEL, TASKS

PHASE3_VERSION = "0.1.0"
NAMESPACE = "phase3_final"
WINDOW_SUITE, NEXT_SUITE, SASREC_SUITE = "phase3", "phase3-next-event", "phase3-sasrec"
LOCK_MANIFEST = "phase3"
KS = (1, 3)
N_BOOT, BOOT_SEED, RANDOM_SEEDS, MODEL_SEED = 1000, 42, 5, 42
CLASSICAL = ("popularity", "personal_frequency", "itemknn_self", "itemknn_noself", "mf")
SASREC = ("sasrec_full", "sasrec_no_position", "sasrec_shuffled", "sasrec_meanpool", "sasrec_session_boundary")
MODELS = CLASSICAL + SASREC
COMPARISONS = [
    ("sasrec_full", "personal_frequency"), ("sasrec_no_position", "personal_frequency"),
    ("sasrec_shuffled", "personal_frequency"), ("sasrec_meanpool", "personal_frequency"),
    ("sasrec_session_boundary", "personal_frequency"),
    ("sasrec_full", "sasrec_no_position"), ("sasrec_full", "sasrec_shuffled"), ("sasrec_full", "sasrec_meanpool"),
    ("sasrec_session_boundary", "sasrec_full"),
    ("sasrec_full", "itemknn_self"), ("sasrec_full", "mf"), ("sasrec_full", "popularity"),
    ("itemknn_self", "personal_frequency"), ("itemknn_noself", "personal_frequency"),
    ("mf", "personal_frequency"), ("popularity", "personal_frequency"),
]
METRICS = ("precision", "recall", "ndcg", "map", "coverage", "exposure_diversity")
RANK = ("precision", "recall", "ndcg", "map")


class GateError(RuntimeError):
    pass


@dataclass
class Plan:
    spec: dict                      # seed-namespace spec (seeds, profiles, customers, days, split_days)
    frozen: dict                    # frozen_config block of the config manifest
    config_manifest: dict
    manifests: dict
    device: str
    runtime: dict
    profiles: tuple = ()
    seeds: tuple = ()
    gate_log: list = field(default_factory=list)

    def run_config(self) -> dict:
        """Canonical configuration of this run (its hash guards output dirs and the lock)."""
        from bri.eval.next_event import NEXT_EVENT_VERSION
        from bri.seq.data import SEQ_DATA_VERSION
        from bri.seq.sasrec import MODEL_VERSION
        from bri.seq.train import TRAIN_VERSION
        return {
            "phase3_version": PHASE3_VERSION, "namespace": NAMESPACE,
            "profiles": list(self.profiles), "seeds": list(self.seeds),
            "customers": self.spec["customers"], "days": self.spec["days"], "split_days": self.spec["split_days"],
            "frozen_config_sha256": self.config_manifest["frozen_config_sha256"],
            "classical_configs_sha256": {e: I.canonical_hash(c) for e, c in classical_configs().items()},
            "sasrec": {"variants": list(SASREC), "comparisons": [list(c) for c in COMPARISONS], "ks": list(KS),
                       "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                       "window": "same model as next-event, next-item scores rank the window set"},
            "versions": {"model": MODEL_VERSION, "train": TRAIN_VERSION, "seq_data": SEQ_DATA_VERSION,
                         "next_event": NEXT_EVENT_VERSION, "runner": RUNNER_VERSION},
            "runtime_sha256": self.runtime["runtime_sha256"],
        }


def classical_configs() -> dict:
    return {"window": experiment_config(WINDOW_SUITE, "window", protocol_config(
                "window", KS, "ndcg", 3, N_BOOT, MODEL_SEED, RANDOM_SEEDS, 0.95)),
            "next_event": experiment_config(NEXT_SUITE, "next_event", protocol_config(
                "next_event", KS, "ndcg", 3, N_BOOT, MODEL_SEED, RANDOM_SEEDS, 0.95))}


def sasrec_lock_config(frozen_sha: str) -> dict:
    return {"suite": SASREC_SUITE, "evaluation": "sasrec",
            "models": {v: {"family": "sasrec", "grid": [{"frozen_config_sha256": frozen_sha}]} for v in SASREC},
            "key_comparisons": [list(c) for c in COMPARISONS],
            "protocol": {"ks": list(KS), "n_boot": N_BOOT, "boot_seed": BOOT_SEED}}


# --- gates ---------------------------------------------------------------------------

def _pytest(args: list[str], must_not_skip: bool) -> str:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", *args], capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    if r.returncode != 0:
        raise GateError(f"test gate failed ({' '.join(args) or 'default'}): {last}")
    if must_not_skip and ("skipped" in last or "passed" not in last):
        raise GateError(f"CUDA validation did not actually run: {last}")
    return last


def gates(run_tests: bool = True, manifests: dict | None = None, expected_device: str = "cuda:0",
          profiles=None, seeds=None, log=print) -> Plan:
    from bri.eval.next_event import NEXT_EVENT_VERSION  # noqa: F401
    from bri.seq import runtime as R
    from bri.seq.data import SEQ_DATA_VERSION
    from bri.seq.sasrec import MODEL_VERSION
    from bri.seq.train import TRAIN_VERSION
    glog = []
    if run_tests:  # 1-2
        glog.append({"gate": "pytest", "result": _pytest([], False)})
        log(f"gate 1 pytest: {glog[-1]['result']}")
        glog.append({"gate": "pytest -m cuda", "result": _pytest(["-m", "cuda"], True)})
        log(f"gate 2 pytest -m cuda: {glog[-1]['result']}")
    manifests = I.load_manifests() if manifests is None else manifests  # 3
    I.verify_frozen_reports(manifests)
    cfgm = manifests.get("phase3-config")
    if cfgm is None or I.canonical_hash(cfgm["frozen_config"]) != cfgm["frozen_config_sha256"]:
        raise GateError("frozen SASRec configuration missing or inconsistent with its hash")
    glog.append({"gate": "manifests", "result": sorted(manifests)})
    R.configure_determinism()  # before CUDA initialization in this process
    choice = R.select_device("cuda" if expected_device.startswith("cuda") else "cpu")
    R.require_device(choice, expected_device)  # 4
    glog.append({"gate": "device", "result": choice.device})
    spec = next((m["seed_namespaces"][NAMESPACE] for m in manifests.values()
                 if NAMESPACE in m.get("seed_namespaces", {})), None)
    if spec is None:
        raise GateError(f"seed namespace '{NAMESPACE}' not registered")
    profiles = tuple(profiles or spec["profiles"])
    seeds = tuple(seeds or spec["seeds"])
    I.check_seed_namespace(seeds, NAMESPACE, manifests)  # 5
    if not set(profiles) <= set(spec["profiles"]):
        raise GateError(f"profiles {profiles} not in the namespace spec")
    glog.append({"gate": "namespace", "result": {"profiles": profiles, "seeds": seeds}})
    want = {"model": MODEL_VERSION, "train": TRAIN_VERSION}  # 6
    have = {k: cfgm["versions"][k] for k in want}
    if want != have or cfgm["tuning_data"]["seq_data_version"] != SEQ_DATA_VERSION:
        raise GateError(f"pipeline versions differ from the frozen configuration: code {want}/{SEQ_DATA_VERSION}, "
                        f"frozen {have}/{cfgm['tuning_data']['seq_data_version']}")
    runtime = R.runtime_fingerprint(choice.device)
    if runtime["runtime_sha256"] != cfgm["runtime"]["runtime_sha256"]:
        raise GateError("runtime fingerprint differs from the one recorded when the configuration was frozen: "
                        f"{runtime['runtime']} vs {cfgm['runtime']['runtime']}")
    glog.append({"gate": "versions+runtime", "result": runtime["runtime_sha256"]})
    log("gates 3-6: manifests, frozen config, device, namespace, versions and runtime verified")
    return Plan(spec, cfgm["frozen_config"], cfgm, manifests, choice.device, runtime, profiles, seeds, glog)


def check_outputs(plan: Plan, output_dir: Path) -> None:
    """Refuse frozen dirs, reports of another configuration, and non-reproduction of a lock."""
    cfg_hash = I.canonical_hash(plan.run_config())
    if (output_dir / FROZEN_MARKER).exists():
        raise I.LockedTestSplitError(f"{output_dir} is frozen")
    agg = output_dir / "phase3_report.json"
    if agg.exists() and json.loads(agg.read_text(encoding="utf-8")).get("run_config_sha256") != cfg_hash:
        raise I.LockedTestSplitError(f"{agg} was produced by another configuration; use another --output-dir")
    lock = plan.manifests.get(LOCK_MANIFEST)
    if lock is not None:
        locked = {(e["behavior_profile"], e["seed"]) for e in lock["datasets"]}
        planned = {(p, s) for p in plan.profiles for s in plan.seeds}
        if not planned <= locked:
            raise I.LockedTestSplitError("planned datasets are not all part of the Phase 3 lock")
        rc = lock["phase3_run_config"]
        same = {k: v for k, v in plan.run_config().items() if k not in ("profiles", "seeds")}
        if same != {k: v for k, v in rc.items() if k not in ("profiles", "seeds")}:
            raise I.BenchmarkIntegrityError("configuration differs from the locked Phase 3 run; "
                                            "only exact (subset) reproduction is allowed")


# --- per dataset ---------------------------------------------------------------------------

def _round(d: dict) -> dict:
    return {k: round(float(v), 6) for k, v in d.items()}


def _metrics(cases, targets, recs, candidates, task) -> dict:
    from bri.eval.evaluate import evaluate
    return evaluate(cases, targets, recs, candidates, KS)[task]["metrics"]


def _per_case(cases, targets, recs, candidates) -> dict:
    from bri.eval.evaluate import per_case_metrics
    return {k: per_case_metrics(cases, targets, recs, candidates, k).set_index("case_id") for k in KS}


def run_sasrec(plan: Plan, data: dict, proc: Path, ckpt_dir: Path, log=print) -> dict:
    """Train all variants per task; score next-event test and window test once each."""
    from bri.models.common import to_recommendations
    from bri.prep.split import compute_cutoffs
    from bri.seq.data import build_seq_cases, encode_cases, history_steps
    from bri.seq.sasrec import SASRecConfig
    from bri.seq.train import (TrainConfig, checkpoint_metadata, fit, save_checkpoint, score, targets_frame,
                               train_popularity)
    m, inter, cands = data["meta"], data["interactions"], data["candidates"]
    c, p = m["cutoffs"], m["params"]
    cut = compute_cutoffs(c["window_start_utc"], c["window_end_utc"], p["val_days"], p["test_days"])
    max_len = plan.frozen["model"]["max_len"]
    seqs = build_seq_cases(inter, cut, cands, ("train", "val", "test"), max_len)
    steps = history_steps(inter, cands)
    out = {}
    for task, items in cands.items():
        tr, va, te = seqs[(task, "train")], seqs[(task, "val")], seqs[(task, "test")]
        wcases = data["cases"][(data["cases"]["split"] == "test") & (data["cases"]["task"] == task)]
        wseq = encode_cases(wcases, data["targets"], steps, task, "test", items, max_len)
        tb = train_popularity(inter, task, items)
        tcfg = TrainConfig(**plan.frozen["train"])
        res_t = {"tiebreak": tb, "next_event": {"seq": te}, "window": {"seq": wseq,
                 "empty_history_cases": int((wseq.length == 0).sum())}, "models": {}}
        for name in SASREC:
            variant = name.removeprefix("sasrec_")
            res = fit(SASRecConfig(n_items=len(items), variant=variant, **plan.frozen["model"]), tcfg, tr, va, tb,
                      plan.device)
            t0 = time.perf_counter()
            s_ne = score(res.model, te, plan.device)  # the single test scoring pass
            ne_ms = 1000 * (time.perf_counter() - t0) / max(len(te.cases), 1)
            s_w = score(res.model, wseq, plan.device)
            size = save_checkpoint(ckpt_dir / f"{task}_{variant}.pt", res.model,
                                   checkpoint_metadata(res, tcfg, tr, va, plan.device))
            rec = {}
            for proto, seq, sc in (("next_event", te, s_ne), ("window", wseq, s_w)):
                recs = to_recommendations(seq.cases["case_id"], sc, items, tb)
                tg = targets_frame(seq)
                rec[proto] = {"recs": recs, "targets": tg, "cases": seq.cases}
            res_t["models"][name] = {"rec": rec, "runtime": {
                **{k: res.runtime[k] for k in ("train_seconds", "epochs_run", "seconds_per_epoch",
                                               "peak_gpu_memory_mib", "parameters")},
                "best_epoch": res.best_epoch, "best_val_ndcg@3": res.best_val,
                "test_inference_ms_per_case": round(ne_ms, 5), "checkpoint_bytes": size}}
            log(f"    {task} {name}: val {res.best_val:.4f} epoch {res.best_epoch}/{len(res.history)}")
        out[task] = res_t
    return out


def _paired(a: pd.Series, b: pd.Series, clusters: pd.Series | None) -> dict:
    from bri.eval.bootstrap import paired_bootstrap_diff, paired_cluster_bootstrap_diff
    if clusters is None:
        return _round(paired_bootstrap_diff(a, b, N_BOOT, BOOT_SEED, 0.95))
    return _round(paired_cluster_bootstrap_diff(a, b, clusters, N_BOOT, BOOT_SEED, 0.95))


def run_dataset(plan: Plan, profile: str, seed: int, data_root: Path, output_dir: Path, log=print) -> dict:
    from bri.data.generate_synthetic import write_dataset
    from bri.data.validate_dataset import validate
    from bri.prep.build import build
    base = data_root / NAMESPACE / profile / f"seed{seed}"
    src, proc = base / "synthetic", base / "processed"
    train_d, val_d, test_d = plan.spec["split_days"]
    write_dataset(src, plan.spec["customers"], plan.spec["days"], seed, profile)
    errors = validate(src)
    if errors:
        raise RuntimeError(f"{profile}/seed{seed} failed validation: {errors[:3]}")
    prep = build(src, proc, val_d, test_d, min_history=1)
    common = dict(ks=KS, select_metric="ndcg", select_k=3, n_boot=N_BOOT, seed=MODEL_SEED,
                  random_seeds=RANDOM_SEEDS, confidence=0.95, manifests=plan.manifests)
    w_rep, w_pc = run_baselines(proc, suite=WINDOW_SUITE, return_per_case=True, **common)
    n_rep, n_pc, within = run_next_event(proc, suite=NEXT_SUITE, return_per_case=True, **common)
    run_dir = output_dir / "runs" / profile / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, rep in (("window_report.json", w_rep), ("next_event_report.json", n_rep)):
        (run_dir / name).write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8", newline="\n")
    data = load_processed(proc)
    sas = run_sasrec(plan, data, proc, output_dir / "checkpoints" / profile / f"seed{seed}", log)

    record = {"profile": profile, "seed": seed, "validation": "passed",
              "fingerprints": {"source_composite": I.composite_fingerprint(prep["source_files_sha256"]),
                               "processed_composite": I.processed_fingerprints(proc)["composite"],
                               "prep": I.prep_fingerprint(prep)},
              "evaluations": {}}
    for proto, rep, pc in (("next_event", n_rep, n_pc), ("window", w_rep, w_pc)):
        ev = {"tasks": {}}
        if proto == "next_event":
            ev["cold_start"] = {s: {k: v[k] for k in ("eligible_target_events", "insufficient_history",
                                                       "tied_timestamp_merges", "single_target_sessions")}
                                for s, v in rep["case_stats"].items()}
        else:
            st = prep["counts"]["eval_stats"]
            ev["cold_start"] = {s: {"customers_with_targets": st[s]["customers_with_targets"],
                                    "insufficient_history": st[s]["insufficient_history"]} for s in st}
        for task in TASKS:
            e = rep["tasks"][task]
            per = {name: pc[task][name] for name in CLASSICAL}
            metrics = {name: e["models"][name]["test"] for name in CLASSICAL}
            ref_ids = list(per["personal_frequency"][KS[0]].index)
            st_ = sas[task]
            for name in SASREC:
                r = st_["models"][name]["rec"][proto]
                cand = data["candidates"]  # identical candidate sets to the classical runs
                if cand[task] != e["candidates"]:
                    raise RuntimeError(f"{task}: candidate sets differ between model families")
                metrics[name] = _metrics(r["cases"], r["targets"], r["recs"], cand, task)
                per[name] = _per_case(r["cases"], r["targets"], r["recs"], cand)
                if list(per[name][KS[0]].index) != ref_ids:
                    raise RuntimeError(f"{proto}/{task}: SASRec cases differ from the classical cases")
            clusters = None
            if proto == "next_event":
                cases = st_["next_event"]["seq"].cases.set_index("case_id")
                clusters = cases["customer_token"].reindex(ref_ids)
            t = {"n_test_cases": len(ref_ids), "models": {n: metrics[n] for n in MODELS},
                 "random_floor": {k: {mm: v["mean"] for mm, v in d.items()}
                                  for k, d in e["random_floor"]["metrics"].items()},
                 "comparisons": {f"{a} - {b}": {f"@{k}": {mm: _paired(per[a][k][mm], per[b][k][mm], clusters)
                                                           for mm in ("ndcg", "map")} for k in KS}
                                 for a, b in COMPARISONS},
                 "selected_params": {n: e["models"][n]["selected_params"] for n in CLASSICAL}}
            if proto == "next_event":
                w = within.reindex(ref_ids).astype(bool).to_numpy()
                t["strata"] = {n: {f"@{k}": {
                    "within_session": {"n": int(w.sum()), "ndcg": round(float(per[n][k]["ndcg"].to_numpy()[w].mean()), 6)
                                       if w.any() else None},
                    "cross_session": {"n": int((~w).sum()), "ndcg": round(float(per[n][k]["ndcg"].to_numpy()[~w].mean()), 6)
                                      if (~w).any() else None}} for k in KS} for n in MODELS}
                t["sasrec_runtime"] = {n: st_["models"][n]["runtime"] for n in SASREC}
            else:
                t["sasrec_empty_history_cases"] = st_["window"]["empty_history_cases"]
            ev["tasks"][task] = t
        record["evaluations"][proto] = ev
    (run_dir / "phase3_dataset_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8",
                                                        newline="\n")
    return record


# --- summary ------------------------------------------------------------------------------

def summarize(plan: Plan, records: list[dict]) -> dict:
    profiles = {}
    for profile in plan.profiles:
        recs = sorted((r for r in records if r["profile"] == profile), key=lambda r: r["seed"])
        evs = {}
        for proto in ("next_event", "window"):
            tasks = {}
            for task in TASKS:
                ps = [r["evaluations"][proto]["tasks"][task] for r in recs]
                t = {"test_cases_per_seed": [p["n_test_cases"] for p in ps],
                     "models": {n: {f"@{k}": {mm: across_seed_stats([p["models"][n][f"@{k}"][mm] for p in ps])
                                              for mm in METRICS} for k in KS} for n in MODELS},
                     "random_floor": {f"@{k}": {mm: across_seed_stats([p["random_floor"][f"@{k}"][mm] for p in ps])
                                                for mm in ("ndcg", "map")} for k in KS},
                     "comparisons": {}, "per_seed_ndcg": {
                         str(r["seed"]): {n: {f"@{k}": p["models"][n][f"@{k}"]["ndcg"] for k in KS} for n in MODELS}
                         for r, p in zip(recs, ps)}}
                for comp in ps[0]["comparisons"]:
                    t["comparisons"][comp] = {}
                    for k in KS:
                        for mm in ("ndcg", "map"):
                            d = [p["comparisons"][comp][f"@{k}"][mm] for p in ps]
                            s = across_seed_stats([x["mean_diff"] for x in d])
                            s["seeds_ci_above_0"] = sum(x["ci_low"] > 0 for x in d)
                            s["seeds_ci_below_0"] = sum(x["ci_high"] < 0 for x in d)
                            t["comparisons"][comp][f"@{k}/{mm}"] = s
                if proto == "next_event":
                    t["strata"] = {n: {f"@{k}": {s: across_seed_stats([p["strata"][n][f"@{k}"][s]["ndcg"] for p in ps
                                                                       if p["strata"][n][f"@{k}"][s]["ndcg"] is not None])
                                                 for s in ("within_session", "cross_session")} for k in KS}
                                   for n in MODELS}
                    t["strata_n_per_seed"] = {s: [p["strata"]["personal_frequency"]["@1"][s]["n"] for p in ps]
                                              for s in ("within_session", "cross_session")}
                    t["sasrec_runtime"] = {n: {key: across_seed_stats([p["sasrec_runtime"][n][key] for p in ps])
                                               for key in ("train_seconds", "epochs_run", "best_epoch",
                                                           "test_inference_ms_per_case", "peak_gpu_memory_mib")}
                                           | {"parameters": ps[0]["sasrec_runtime"][n]["parameters"],
                                              "checkpoint_bytes": ps[0]["sasrec_runtime"][n]["checkpoint_bytes"]}
                                           for n in SASREC}
                else:
                    t["sasrec_empty_history_cases_per_seed"] = [p["sasrec_empty_history_cases"] for p in ps]
                t["classical_selection_counts"] = {
                    n: dict(sorted(pd.Series([", ".join(f"{a}={b}" for a, b in p["selected_params"][n].items())
                                              for p in ps]).value_counts().items())) for n in CLASSICAL}
                tasks[task] = t
            cold = {s: {k: int(sum(r["evaluations"][proto]["cold_start"][s][k] for r in recs))
                        for k in recs[0]["evaluations"][proto]["cold_start"][s]} for s in ("val", "test")}
            evs[proto] = {"tasks": tasks, "cold_start_totals": cold}
        profiles[profile] = evs
    cfg = plan.run_config()
    return {
        "label": SYNTHETIC_LABEL,
        "experiment": "Phase 3 sequential recommendation (SASRec + ablations) vs classical baselines",
        "run_config": cfg, "run_config_sha256": I.canonical_hash(cfg),
        "frozen_config": plan.frozen, "runtime": plan.runtime, "gates": plan.gate_log,
        "statistical_design": {
            "within_seed": ("paired percentile bootstrap, n_boot=1000: customer clusters (next-event), "
                            "customer-task cases (window); variation inside one synthetic dataset"),
            "across_seed": ("each seed's test value or paired difference = one observation; mean, sample SD, "
                            "Student-t 95% interval over seeds, per profile; never pooled with within-seed CIs"),
            "not_covered": ("SASRec initialization variation (one training seed, 1, everywhere), other profiles, "
                            "sizes, windows, generator versions, any real population"),
        },
        "protocol_notes": [
            "Primary protocol: next-event. Secondary: 7-day window.",
            "SASRec window scores are next-item predictions from history before the window start; the window "
            "target is a set of items, which is not SASRec's training objective.",
            "Within/cross-session strata use the evaluator's label only; no model receives it.",
            "Ablations reuse the Full configuration (trial 21): they measure component contributions under one "
            "controlled configuration, not each architecture's independently tuned best.",
        ],
        "profiles": profiles,
        "datasets": [{"profile": r["profile"], "seed": r["seed"], "validation": r["validation"],
                      "fingerprints": r["fingerprints"]} for r in records],
    }


def to_markdown(s: dict) -> str:
    def ms(x):
        if x["mean"] is None:
            return "-"
        return f"{x['mean']:.3f}" if x["sd"] is None else f"{x['mean']:.3f} +- {x['sd']:.3f} [{x['ci_low']:.3f}, {x['ci_high']:.3f}]"

    def md(x):
        return f"{x['mean']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]" if x["sd"] is not None else f"{x['mean']:+.3f}"

    cfg = s["run_config"]
    lines = ["# Phase 3: SASRec and ablations vs classical baselines", "", f"> **{s['label']}**", "",
             f"- {len(cfg['profiles'])} profiles x {len(cfg['seeds'])} seeds ({cfg['seeds'][0]}..{cfg['seeds'][-1]}), "
             f"{cfg['customers']} customers, {cfg['days']} days, split {cfg['split_days']}.",
             f"- Frozen SASRec configuration sha256 {cfg['frozen_config_sha256'][:12]}...: {s['frozen_config']['model']}, "
             f"lr {s['frozen_config']['train']['lr']:.3g}, batch {s['frozen_config']['train']['batch_size']}; "
             f"device {s['runtime']['runtime']['device']} ({s['runtime']['runtime']['device_name']}), FP32, deterministic.",
             f"- Across-seed cells: mean +- SD [Student-t 95% over seeds]. {s['statistical_design']['across_seed']}.",
             f"- Within-seed: {s['statistical_design']['within_seed']}. 'CI>0/CI<0' = seeds whose within-seed CI "
             "excludes 0 in that direction.", f"- Not covered: {s['statistical_design']['not_covered']}.", ""]
    lines += [f"- {n}" for n in s["protocol_notes"]] + [""]
    for profile, evs in s["profiles"].items():
        lines += [f"## Profile `{profile}`", ""]
        for proto in ("next_event", "window"):
            lines += [f"### {proto.replace('_', '-')} ({'primary' if proto == 'next_event' else 'secondary'})", ""]
            for task, t in evs[proto]["tasks"].items():
                tc = t["test_cases_per_seed"]
                lines += [f"#### {task.upper()} (test cases per seed {min(tc)}-{max(tc)})", "",
                          "| Model | NDCG@1 | NDCG@3 | P@3 | R@3 | MAP@3 | Cov@3 | ExpDiv@3 |", "|---|---|---|---|---|---|---|---|",
                          f"| random floor | {ms(t['random_floor']['@1']['ndcg'])} | {ms(t['random_floor']['@3']['ndcg'])} "
                          f"| - | - | {ms(t['random_floor']['@3']['map'])} | - | - |"]
                for n, m in t["models"].items():
                    k3 = m["@3"]
                    lines.append(f"| {n} | {ms(m['@1']['ndcg'])} | {ms(k3['ndcg'])} | {k3['precision']['mean']:.3f} "
                                 f"| {k3['recall']['mean']:.3f} | {k3['map']['mean']:.3f} | {k3['coverage']['mean']:.3f} "
                                 f"| {k3['exposure_diversity']['mean']:.3f} |")
                lines += ["", "| Paired difference | NDCG@1 [across-seed CI] | CI>0/CI<0 | NDCG@3 [across-seed CI] | CI>0/CI<0 |",
                          "|---|---|---|---|---|"]
                for comp, c in t["comparisons"].items():
                    a, b = c["@1/ndcg"], c["@3/ndcg"]
                    lines.append(f"| {comp} | {md(a)} | {a['seeds_ci_above_0']}/{a['seeds_ci_below_0']} | {md(b)} "
                                 f"| {b['seeds_ci_above_0']}/{b['seeds_ci_below_0']} |")
                if proto == "next_event":
                    n_w, n_c = t["strata_n_per_seed"]["within_session"], t["strata_n_per_seed"]["cross_session"]
                    lines += ["", f"Session strata, NDCG@1 (within-session cases per seed {min(n_w)}-{max(n_w)}, "
                              f"cross-session {min(n_c)}-{max(n_c)}):", "",
                              "| Model | within-session | cross-session |", "|---|---|---|"]
                    for n, st in t["strata"].items():
                        lines.append(f"| {n} | {ms(st['@1']['within_session'])} | {ms(st['@1']['cross_session'])} |")
                    lines += ["", "| SASRec variant | train s | epochs | best epoch | test ms/case | peak GPU MiB | params | ckpt KiB |",
                              "|---|---|---|---|---|---|---|---|"]
                    for n, rt in t["sasrec_runtime"].items():
                        lines.append(f"| {n} | {rt['train_seconds']['mean']:.1f} | {rt['epochs_run']['mean']:.1f} "
                                     f"| {rt['best_epoch']['mean']:.1f} | {rt['test_inference_ms_per_case']['mean']:.4f} "
                                     f"| {rt['peak_gpu_memory_mib']['mean']:.0f} | {rt['parameters']} "
                                     f"| {rt['checkpoint_bytes'] / 1024:.0f} |")
                lines.append("")
            ct = evs[proto]["cold_start_totals"]["test"]
            lines += ["Cold start / exclusions (test, total over seeds): " + "; ".join(f"{k} = {v}" for k, v in ct.items()), ""]
    lines += ["Notes: exposure diversity measures catalog exposure evenness, not per-customer semantic diversity. "
              "All results are synthetic-data experiments and say nothing about real customers.", ""]
    return "\n".join(lines)


def run(plan: Plan, data_root: Path, output_dir: Path, log=print) -> dict:
    check_outputs(plan, output_dir)
    records = []
    for profile in plan.profiles:
        for seed in plan.seeds:
            t0 = time.perf_counter()
            log(f"  {profile} seed {seed}: start")
            records.append(run_dataset(plan, profile, seed, data_root, output_dir, log))
            log(f"  {profile} seed {seed}: done in {time.perf_counter() - t0:.0f}s")
    summary = summarize(plan, records)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase3_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output_dir / "phase3_report.md").write_text(to_markdown(summary), encoding="utf-8", newline="\n")
    return summary


def lock(output_dir: Path, data_root: Path, frozen_dir: Path, manifest_path: Path | None = None,
         manifests: dict | None = None) -> str:
    """Freeze the aggregate report and write the immutable Phase 3 lock manifest (once)."""
    manifests = I.load_manifests() if manifests is None else manifests
    if LOCK_MANIFEST in manifests:
        raise I.LockedTestSplitError("Phase 3 is already locked")
    summary = json.loads((output_dir / "phase3_report.json").read_text(encoding="utf-8"))
    cfg = summary["run_config"]
    frozen_dir.mkdir(parents=True, exist_ok=False)
    for name in ("phase3_report.json", "phase3_report.md"):
        shutil.copyfile(output_dir / name, frozen_dir / name)
    (frozen_dir / FROZEN_MARKER).write_text(
        "Frozen Phase 3 report (synthetic-data experiment). Do not regenerate or edit.\n"
        f"Locked datasets and configuration: benchmarks/manifests/{LOCK_MANIFEST}.json\n", encoding="utf-8", newline="\n")
    datasets = []
    for d in summary["datasets"]:
        proc = data_root / NAMESPACE / d["profile"] / f"seed{d['seed']}" / "processed"
        meta = json.loads((proc / "prep_metadata.json").read_text(encoding="utf-8"))
        e = I.dataset_entry(None, proc, meta, f"{d['profile']}/seed{d['seed']}", ["window", "next_event", "sasrec"])
        e["seed_namespace"] = NAMESPACE
        datasets.append(e)
    configs = {**classical_configs(), "sasrec": sasrec_lock_config(cfg["frozen_config_sha256"])}
    try:
        rel = frozen_dir.resolve().relative_to(I.REPO_ROOT).as_posix()
    except ValueError:  # outside the repository (tests)
        rel = frozen_dir.resolve().as_posix()
    m = I.build_manifest(WINDOW_SUITE, configs, datasets, {**cfg["versions"]}, rel,
                         "Phase 3 final evaluation; every listed test split used once, after the SASRec configuration "
                         "was frozen.", extra={"phase3_run_config": cfg, "phase3_run_config_sha256": summary["run_config_sha256"],
                                                "runtime": summary["runtime"]})
    return I.write_manifest(m, manifest_path or (I.MANIFEST_DIR / f"{LOCK_MANIFEST}.json"))


SUMMARY_DIR = Path("benchmarks/phase3-summary")
SUMMARY_MODELS = ("popularity", "personal_frequency", "itemknn_self", "mf", "sasrec_full", "sasrec_no_position",
                  "sasrec_shuffled", "sasrec_meanpool", "sasrec_session_boundary")
SUMMARY_COMPARISONS = ("sasrec_full - personal_frequency", "sasrec_full - sasrec_shuffled",
                       "sasrec_full - sasrec_meanpool", "sasrec_full - sasrec_no_position",
                       "sasrec_session_boundary - sasrec_full")


def public_summary(frozen_report: Path | None = None, manifests: dict | None = None) -> dict:
    """Compact, DERIVED view of the frozen Phase 3 report for public consumption.

    Never modifies the frozen report. Refuses unless the source file's sha256 equals
    the hash pinned in the Phase 3 lock manifest, and records that hash.
    """
    manifests = I.load_manifests() if manifests is None else manifests
    lock_m = manifests[LOCK_MANIFEST]
    src = Path(frozen_report or I.REPO_ROOT / lock_m["frozen_report_dir"] / "phase3_report.json")
    digest = I.sha256_file(src)
    if digest != lock_m["frozen_report_sha256"]["phase3_report.json"]:
        raise I.BenchmarkIntegrityError(f"{src} does not match the frozen report hash in the Phase 3 manifest")
    s = json.loads(src.read_text(encoding="utf-8"))

    def stat(x):
        return {k: x[k] for k in ("mean", "ci_low", "ci_high")}

    groups = {}
    for profile, evs in s["profiles"].items():
        for proto, ev in evs.items():
            for task, t in ev["tasks"].items():
                g = {"test_cases_per_seed": [min(t["test_cases_per_seed"]), max(t["test_cases_per_seed"])],
                     "random_floor_ndcg@1": t["random_floor"]["@1"]["ndcg"]["mean"],
                     "ndcg": {m: {f"@{k}": stat(t["models"][m][f"@{k}"]["ndcg"]) for k in KS} for m in SUMMARY_MODELS},
                     "paired_ndcg@1": {c: {**stat(t["comparisons"][c]["@1/ndcg"]),
                                           "seeds_ci_above_0": t["comparisons"][c]["@1/ndcg"]["seeds_ci_above_0"],
                                           "seeds_ci_below_0": t["comparisons"][c]["@1/ndcg"]["seeds_ci_below_0"]}
                                       for c in SUMMARY_COMPARISONS}}
                if proto == "next_event":
                    g["strata_ndcg@1"] = {m: {w: stat(t["strata"][m]["@1"][w]) for w in ("within_session", "cross_session")}
                                          for m in ("personal_frequency", "sasrec_full")}
                    g["strata_cases_per_seed"] = {w: [min(v), max(v)] for w, v in t["strata_n_per_seed"].items()}
                groups[f"{profile}/{proto}/{task}"] = g
    return {
        "label": s["label"],
        "derived_from": {"file": "benchmarks/phase3/phase3_report.json", "sha256": digest,
                         "manifest": f"benchmarks/manifests/{LOCK_MANIFEST}.json",
                         "run_config_sha256": s["run_config_sha256"]},
        "note": ("Derived summary of the frozen Phase 3 report (synthetic-data experiment). The frozen report and "
                 "manifests remain the authoritative record. Brackets are across-seed Student-t 95% intervals over "
                 "10 seeds; seed counts are within-seed paired-bootstrap CIs excluding 0."),
        "limitations": [
            "Synthetic data only; no evidence about real customers or banking systems.",
            "One SASRec training seed: initialization variation not measured; 10 seeds are 10 datasets.",
            "Small NBP training sets and few within-session NBP cases.",
            "Training histories are shorter than validation/test histories (temporal shift).",
            "Bit-identical reproduction shown only on the same runtime (GPU, CUDA/torch build, threads).",
        ],
        "frozen_sasrec_config": s["frozen_config"],
        "groups": groups,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "lock", "summary"))
    parser.add_argument("--data-root", type=Path, default=Path("data/phase3"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase3/final"))
    parser.add_argument("--frozen-dir", type=Path, default=Path("benchmarks/phase3"))
    parser.add_argument("--profiles", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            print(SYNTHETIC_LABEL, flush=True)
            plan = gates(profiles=args.profiles, seeds=args.seeds, log=lambda x: print(x, flush=True))
            run(plan, args.data_root, args.output_dir, log=lambda x: print(x, flush=True))
            print(f"Reports written to {args.output_dir}")
        elif args.command == "summary":
            out = SUMMARY_DIR / "phase3_summary.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(public_summary(), indent=2) + "\n", encoding="utf-8", newline="\n")
            print(f"Derived summary written to {out}")
        else:
            digest = lock(args.output_dir, args.data_root, args.frozen_dir)
            print(f"Phase 3 locked; pin sha256 {digest} in bri.experiments.integrity.MANIFEST_SHA256")
    except (GateError, I.LockedTestSplitError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # device mismatch etc.
        from bri.seq.runtime import RuntimeConfigurationError
        if isinstance(exc, RuntimeConfigurationError):
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
