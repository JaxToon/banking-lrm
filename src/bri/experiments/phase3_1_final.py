"""Phase 3.1 final evaluation: PF vs SASRec Full vs the frozen static hybrid (synthetic only).

Implements the approved Checkpoint 6 plan. Nothing here selects anything on test data.

Gates, in this order, before ANY final dataset is read (a failure stops the run):
  1. default test suite (`pytest`)       2. CUDA validation run (`pytest -m cuda`, must run)
  3. pinned manifests: prereg, seeds, phase3-config, phase3_1-config (static fusion, hash,
     references), phase3_1-final-data; no `phase3_1` lock yet
  4. require_device("cuda:0")              5. `phase3_1_final` seed namespace (30 datasets)
  6. SASRec/sequence/hybrid versions and the runtime fingerprint equal to the frozen ones.
Only a Plan returned by `gates` can open a final dataset (`run_dataset` refuses otherwise),
so tests and pre-flight code never build or read a final test split.

Per final dataset x task:
- Fingerprints re-verified against phase3_1-final-data before the data is used.
- Personal Frequency: Phase 2.3 grids selected on VALIDATION only (next-event: signal
  `target`; 7-day window: `target` / `item_event`, NDCG@3, ties keep the earlier entry).
  P_PF = Laplace distribution of the selected signal's history counts.
- SASRec Full, frozen Phase 3 configuration, initialization seeds {1,2,3}: train split
  only, early stopping on validation only, then ONE scoring pass on next-event test and
  one on window test.
- Static hybrid: (1 - alpha) P_PF + alpha P_SASRec with the pinned alpha per task
  (no alpha search). Oracle router: diagnostic only (next-event), never a candidate.
- Shared evaluator and tie-break (train popularity, then item id); identical sorted
  candidate sets and identical case order are asserted.

Statistics (three separate sources of variation, never pooled):
- within seed: paired percentile bootstrap (n_boot=1000) on per-case means over the 3
  initializations; customer clusters (next-event), customer-task cases (window);
- across seeds: Student-t 95% interval over the 10 dataset seeds per profile x task;
- initialization: per-dataset SD over the 3 initializations, summarized separately.
RQ1/RQ2 use the across-seed interval of NDCG@1 (next-event). RQ3 is not evaluable under
the preregistered protocol (no gated configuration was selectable on tuning validation;
no OOF, gate fitting or gated evaluation on final data).

Usage (only after the Checkpoint 6 run is approved):
    python -m bri.experiments.phase3_1_final run --confirm-final-evaluation
    python -m bri.experiments.phase3_1_final lock          (once, after review)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bri.experiments import integrity as I
from bri.experiments.phase3_1 import (CONFIG_MANIFEST, DATA_ROOT, FINAL_DATA_MANIFEST, FINAL_NS, P3_CONFIG, PREREG,
                                      SEEDS_NAME, Phase31Refused)
from bri.experiments.robustness import across_seed_stats
from bri.seq import hybrid as H
from bri.tasks import SYNTHETIC_LABEL, TASKS

FINAL_VERSION = "0.1.0"
LOCK_MANIFEST = "phase3_1"
OUTPUT_DIR = Path("artifacts/phase3_1/final")
FROZEN_DIR = Path("benchmarks/phase3_1")
KS = (1, 3)
RANK = ("precision", "recall", "ndcg", "map")
N_BOOT, BOOT_SEED = 1000, 42
PF_SELECT = ("ndcg", 3)  # Phase 2.3 classical selection metric
MODELS = ("personal_frequency", "sasrec_full", "static_hybrid")
DIAGNOSTICS = ("oracle_router",)
COMPARISONS = (("static_hybrid", "personal_frequency"), ("static_hybrid", "sasrec_full"),
               ("sasrec_full", "personal_frequency"))
RQ1_GROUPS = (("habit_dominant", "nbs"), ("habit_dominant", "nbp"), ("balanced", "nbs"), ("balanced", "nbp"))
RQ2_GROUP = ("transition_dominant", "nbs")
MARGIN = 0.01
RQ3_REASON = ("no gated configuration was selectable on tuning validation (phase3_1-config: gated not selected "
              "under the preregistered rule, subset and C unavailable for both tasks); by the owner's Checkpoint 6 "
              "decision there is no OOF cross-fitting, gate fitting or gated evaluation on final data. This is not "
              "evidence that adaptive fusion is worse than static fusion.")
_ISSUED: list = []  # Plans returned by a successful gates() in this process (strong refs: ids never reused)


class FinalRunRefused(Phase31Refused):
    pass


@dataclass(frozen=True)
class Plan:
    manifests: dict
    sasrec: dict           # phase3-config frozen_config
    alpha: dict            # pinned static alpha per task
    init_seeds: tuple
    entries: tuple         # phase3_1-final-data final_datasets
    device: str
    runtime: dict
    gate_log: list = field(default_factory=list)

    def run_config(self) -> dict:
        from bri.seq.data import SEQ_DATA_VERSION
        from bri.seq.sasrec import MODEL_VERSION
        from bri.seq.train import TRAIN_VERSION
        return {"namespace": FINAL_NS, "final_version": FINAL_VERSION, "alpha": self.alpha,
                "init_seeds": list(self.init_seeds), "ks": list(KS), "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                "pf_selection": f"validation {PF_SELECT[0]}@{PF_SELECT[1]}",
                "config_sha256": I.MANIFEST_SHA256[CONFIG_MANIFEST],
                "final_data_sha256": I.MANIFEST_SHA256[FINAL_DATA_MANIFEST],
                "sasrec_frozen_config_sha256": self.manifests[P3_CONFIG]["frozen_config_sha256"],
                "versions": {"model": MODEL_VERSION, "train": TRAIN_VERSION, "seq_data": SEQ_DATA_VERSION,
                             "hybrid": H.HYBRID_VERSION, "final": FINAL_VERSION},
                "runtime_sha256": self.runtime["runtime_sha256"], "device": self.device}


# --- gates ---------------------------------------------------------------------------------

def check_manifests(manifests: dict) -> dict:
    """Gate 3 (pure): pinned state required for the final run. Returns the frozen parts."""
    for needed in (PREREG, SEEDS_NAME, P3_CONFIG, CONFIG_MANIFEST, FINAL_DATA_MANIFEST):
        if needed not in manifests:
            raise FinalRunRefused(f"manifest '{needed}' is not pinned")
    if LOCK_MANIFEST in manifests:
        raise FinalRunRefused("Phase 3.1 is already locked; the final test splits were used once")
    cfg, fd, pre, p3 = (manifests[n] for n in (CONFIG_MANIFEST, FINAL_DATA_MANIFEST, PREREG, P3_CONFIG))
    f = cfg["frozen_config"]
    if I.canonical_hash(f) != cfg["frozen_config_sha256"]:
        raise FinalRunRefused("phase3_1-config frozen_config does not match its hash")
    if f["primary_hybrid"] != "static_fusion" or set(f["alpha"]) != set(TASKS) or \
            any(a not in H.ALPHA_GRID for a in f["alpha"].values()):
        raise FinalRunRefused("phase3_1-config is not a static-fusion configuration on the preregistered grid")
    if not (f["sasrec"]["frozen_config_sha256"] == pre["references"]["phase3_frozen_config_sha256"]
            == p3["frozen_config_sha256"] == I.canonical_hash(p3["frozen_config"])):
        raise FinalRunRefused("SASRec configuration references disagree")
    if fd["references"].get("config_sha256") != I.MANIFEST_SHA256[CONFIG_MANIFEST] or \
            cfg["references"]["prereg_sha256"] != I.MANIFEST_SHA256[PREREG]:
        raise FinalRunRefused("final-data or config manifest references other pins")
    spec = manifests[SEEDS_NAME]["seed_namespaces"][FINAL_NS]
    entries = fd["final_datasets"]
    if sorted((e["behavior_profile"], e["seed"]) for e in entries) != \
            sorted((p, s) for p in spec["profiles"] for s in spec["seeds"]):
        raise FinalRunRefused("final datasets differ from the reserved profiles x seeds")
    I.check_seed_namespace([e["seed"] for e in entries], FINAL_NS, manifests)  # gate 5
    return {"sasrec": p3["frozen_config"], "alpha": dict(f["alpha"]),
            "init_seeds": tuple(f["sasrec"]["initialization_seeds"]), "entries": tuple(entries)}


def gates(run_tests: bool = True, manifests: dict | None = None, expected_device: str = "cuda:0",
          log=print) -> Plan:
    from bri.experiments.phase3 import _pytest
    from bri.seq import runtime as R
    from bri.seq.data import SEQ_DATA_VERSION
    from bri.seq.sasrec import MODEL_VERSION
    from bri.seq.train import TRAIN_VERSION
    glog = []
    if run_tests:
        glog.append({"gate": "pytest", "result": _pytest([], False)})
        log(f"gate 1 pytest: {glog[-1]['result']}")
        glog.append({"gate": "pytest -m cuda", "result": _pytest(["-m", "cuda"], True)})
        log(f"gate 2 pytest -m cuda: {glog[-1]['result']}")
    manifests = I.load_manifests() if manifests is None else manifests
    I.verify_frozen_reports(manifests)
    frozen = check_manifests(manifests)
    glog.append({"gate": "manifests+namespace", "result": sorted(manifests)})
    R.configure_determinism()  # before CUDA initialization in this process
    choice = R.select_device("cuda" if expected_device.startswith("cuda") else "cpu")
    R.require_device(choice, expected_device)  # 4
    glog.append({"gate": "device", "result": choice.device})
    p3, cfg = manifests[P3_CONFIG], manifests[CONFIG_MANIFEST]
    if {"model": MODEL_VERSION, "train": TRAIN_VERSION} != {k: p3["versions"][k] for k in ("model", "train")} or \
            p3["tuning_data"]["seq_data_version"] != SEQ_DATA_VERSION or \
            cfg["tuning_report"]["hybrid_version"] != H.HYBRID_VERSION:
        raise FinalRunRefused("pipeline versions differ from the frozen configurations")
    runtime = R.runtime_fingerprint(choice.device)
    if runtime["runtime_sha256"] != p3["runtime"]["runtime_sha256"] or \
            runtime["runtime_sha256"] != cfg["tuning_report"]["runtime_sha256"]:
        raise FinalRunRefused("runtime fingerprint differs from the frozen one")
    glog.append({"gate": "versions+runtime", "result": runtime["runtime_sha256"]})
    log("gates 3-6: manifests, namespace, device, versions and runtime verified")
    plan = Plan(manifests, frozen["sasrec"], frozen["alpha"], frozen["init_seeds"], frozen["entries"],
                choice.device, runtime, glog)
    _ISSUED.append(plan)
    return plan


def require_gated_plan(plan: Plan) -> None:
    """Only the exact Plan object issued by gates() opens final data, and it must still match the pins.

    Identity (not type or field equality) is checked, so a Plan built or copied by a caller is
    refused; the pinned manifests are re-read so that no field of the Plan can override them.
    This guards against accidental use; it is not a security boundary against deliberate tampering.
    """
    if not any(p is plan for p in _ISSUED):
        raise FinalRunRefused("final datasets are opened only by a run that passed every gate")
    fresh = check_manifests(I.load_manifests())
    if (plan.alpha, plan.sasrec, plan.init_seeds, plan.entries) !=             (fresh["alpha"], fresh["sasrec"], fresh["init_seeds"], fresh["entries"]):
        raise FinalRunRefused("the gated plan no longer matches the pinned manifests")


# --- one task (pure w.r.t. where the data comes from; used by tests on small data) ------------

def _per_case(td, split: str, scores: np.ndarray) -> dict:
    from bri.eval.evaluate import per_case_metrics
    from bri.models.common import to_recommendations
    recs = to_recommendations(td.cases[split]["case_id"], scores, td.items, td.tiebreak)
    return {k: per_case_metrics(td.cases[split], td.targets[split], recs, td.candidates, k).set_index("case_id")
            for k in KS}


def _mean_frames(frames: list[dict]) -> dict:
    """Per-case mean over initializations (same cases, same order)."""
    return {k: sum(f[k][list(RANK)] for f in frames) / len(frames) for k in KS}


def _means(pc: dict) -> dict:
    return {f"@{k}": {m: round(float(pc[k][m].mean()), 6) for m in RANK} for k in KS}


def _paired(a: pd.Series, b: pd.Series, clusters: pd.Series | None) -> dict:
    from bri.eval.bootstrap import paired_bootstrap_diff, paired_cluster_bootstrap_diff
    d = (paired_bootstrap_diff(a, b, N_BOOT, BOOT_SEED, 0.95) if clusters is None
         else paired_cluster_bootstrap_diff(a, b, clusters, N_BOOT, BOOT_SEED, 0.95))
    return {k: round(float(v), 6) for k, v in d.items()}


def evaluate_task(task: str, ne_data: dict, w_data: dict, seqs: dict, w_test, alpha: float, sasrec: dict,
                  init_seeds, device: str, log=None) -> dict:
    """One dataset x task. `seqs`: {(task, split): SeqCases} for train/val/test; `w_test`: window test SeqCases."""
    from bri.eval.next_event import history_counts
    from bri.experiments.baselines import SUITES, _select, _TaskData
    from bri.models.common import history_matrix
    from bri.seq.sasrec import SASRecConfig
    from bri.seq.train import TrainConfig, fit, score
    td = {"next_event": _TaskData(ne_data, task, history_counts), "window": _TaskData(w_data, task, history_matrix)}
    grid = {"next_event": SUITES["phase2.3-next-event"]["personal_frequency"][1],
            "window": SUITES["phase2.3"]["personal_frequency"][1]}
    tr, va, te = (seqs[(task, s)] for s in ("train", "val", "test"))
    H.check_candidates(td["next_event"].items, td["window"].items, tr.items_vocab, va.items_vocab, te.items_vocab,
                       w_test.items_vocab)
    seq_of = {"next_event": te, "window": w_test}
    for proto, s in seq_of.items():
        if list(s.cases["case_id"]) != list(td[proto].cases["test"]["case_id"]):
            raise RuntimeError(f"{task}/{proto}: SASRec and PF test cases differ")

    # PF: selection on validation only, then the selected signal's test counts.
    pf, pf_sel = {}, {}
    for proto, t in td.items():
        params, trials = _select("personal_frequency", grid[proto], t, *PF_SELECT, BOOT_SEED)
        pf_sel[proto] = {"selected": params, "validation_trials": trials}
        pf[proto] = H.pf_distribution(t.history("test", params["signal"]))

    # SASRec Full per initialization: train only, val early stopping, ONE scoring pass per protocol.
    model_cfg = SASRecConfig(n_items=len(tr.items_vocab), variant="full", **sasrec["model"])
    sas, runtime = {"next_event": [], "window": []}, []
    for init in init_seeds:
        res = fit(model_cfg, TrainConfig(**{**sasrec["train"], "seed": init}), tr, va, td["next_event"].tiebreak,
                  device)
        rt = {"init_seed": init, "best_epoch": res.best_epoch, "best_val_ndcg@3": res.best_val,
              **{k: res.runtime[k] for k in ("train_seconds", "epochs_run", "peak_gpu_memory_mib", "parameters")}}
        for proto, s in seq_of.items():
            t0 = time.perf_counter()
            sas[proto].append(H.softmax(score(res.model, s, device)))
            rt[f"{proto}_test_inference_ms_per_case"] = round(1000 * (time.perf_counter() - t0) / max(len(s.cases), 1), 5)
        runtime.append(rt)
        if log:
            log(f"      {task} init {init}: best epoch {res.best_epoch}, val ndcg@3 {res.best_val:.4f}")

    out = {"alpha": alpha, "pf_selection": pf_sel, "sasrec_runtime": runtime, "protocols": {}}
    for proto, t in td.items():
        t_eval = time.perf_counter()
        frames = {"personal_frequency": [_per_case(t, "test", pf[proto])],
                  "sasrec_full": [_per_case(t, "test", p) for p in sas[proto]],
                  "static_hybrid": [_per_case(t, "test", H.static_fusion(pf[proto], p, alpha)) for p in sas[proto]]}
        if proto == "next_event":
            w = t.cases["test"]["within_session"].to_numpy(dtype=bool)
            frames["oracle_router"] = [_per_case(t, "test", H.oracle_router(pf[proto], p, w)) for p in sas[proto]]
        mean = {n: _mean_frames(f) for n, f in frames.items()}
        clusters = (t.cases["test"].set_index("case_id")["customer_token"].reindex(mean["personal_frequency"][1].index)
                    if proto == "next_event" else None)
        p = {"n_test_cases": len(t.cases["test"]),
             "models": {n: _means(mean[n]) for n in MODELS},
             "diagnostics": {n: _means(mean[n]) for n in DIAGNOSTICS if n in mean},
             "initialization": {n: {f"@{k}": {"ndcg_per_init": [round(float(f[k]["ndcg"].mean()), 6) for f in frames[n]],
                                              "ndcg_sd": round(float(np.std([f[k]["ndcg"].mean() for f in frames[n]],
                                                                            ddof=1)), 6) if len(frames[n]) > 1 else None}
                                    for k in KS} for n in ("sasrec_full", "static_hybrid")},
             "comparisons": {f"{a} - {b}": {f"@{k}": {m: _paired(mean[a][k][m], mean[b][k][m], clusters)
                                                       for m in ("ndcg", "map")} for k in KS}
                             for a, b in COMPARISONS}}
        if proto == "next_event":
            wi = t.cases["test"].set_index("case_id")["within_session"].reindex(mean["personal_frequency"][1].index)
            wi = wi.astype(bool).to_numpy()
            p["strata"] = {n: {f"@{k}": {
                "within_session": {"n": int(wi.sum()), "ndcg": round(float(mean[n][k]["ndcg"].to_numpy()[wi].mean()), 6)
                                   if wi.any() else None},
                "cross_session": {"n": int((~wi).sum()), "ndcg": round(float(mean[n][k]["ndcg"].to_numpy()[~wi].mean()), 6)
                                  if (~wi).any() else None}} for k in KS} for n in (*MODELS, *DIAGNOSTICS)}
        else:
            p["sasrec_empty_history_cases"] = int((w_test.length == 0).sum())
        p["evaluation_seconds"] = round(time.perf_counter() - t_eval, 3)
        out["protocols"][proto] = p
    return out


# --- one final dataset -------------------------------------------------------------------------

def run_dataset(plan: Plan, entry: dict, root: Path, output_dir: Path, log=print) -> dict:
    """Opens one FINAL dataset; refuses unless `plan` came from `gates`."""
    require_gated_plan(plan)
    from bri.experiments.baselines import load_processed, next_event_data
    from bri.prep.split import compute_cutoffs
    from bri.seq.data import build_seq_cases, encode_cases, history_steps
    t0 = time.perf_counter()
    proc = root / FINAL_NS / entry["behavior_profile"] / f"seed{entry['seed']}" / "processed"
    meta = json.loads((proc / "prep_metadata.json").read_text(encoding="utf-8"))
    if I.verify_processed_dir(proc, meta)["files"] != entry["processed_files_sha256"] or \
            I.prep_fingerprint(meta) != entry["prep_fingerprint"]:
        raise FinalRunRefused(f"{entry['dataset_id']} differs from the pinned final-data fingerprints")
    data = load_processed(proc)
    ne_data, stats = next_event_data(data)
    c, p = meta["cutoffs"], meta["params"]
    cut = compute_cutoffs(c["window_start_utc"], c["window_end_utc"], p["val_days"], p["test_days"])
    L = plan.sasrec["model"]["max_len"]
    seqs = build_seq_cases(data["interactions"], cut, data["candidates"], ("train", "val", "test"), L)
    steps = history_steps(data["interactions"], data["candidates"])
    rec = {"dataset_id": entry["dataset_id"], "profile": entry["behavior_profile"], "seed": entry["seed"],
           "fingerprints": {"processed_composite": I.processed_fingerprints(proc)["composite"],
                            "prep": entry["prep_fingerprint"], "source_composite": entry["source_composite_sha256"]},
           "cold_start_next_event": {s: {k: stats[s][k] for k in ("eligible_target_events", "insufficient_history")}
                                     for s in ("val", "test")},
           "tasks": {}}
    for task in TASKS:
        wtest = encode_cases(data["cases"][(data["cases"]["split"] == "test") & (data["cases"]["task"] == task)],
                             data["targets"], steps, task, "test", data["candidates"][task], L)
        rec["tasks"][task] = evaluate_task(task, ne_data, data, seqs, wtest, plan.alpha[task], plan.sasrec,
                                           plan.init_seeds, plan.device, log)
    rec["wall_seconds"] = round(time.perf_counter() - t0, 3)
    run_dir = output_dir / "runs" / entry["behavior_profile"] / f"seed{entry['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dataset_record.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8", newline="\n")
    return rec


# --- summary and preregistered criteria --------------------------------------------------------

def _mean_or_none(values: list) -> float | None:
    return None if not values or any(v is None for v in values) else round(float(np.mean(values)), 6)


def _group(records: list[dict], profile: str, task: str, proto: str) -> list[dict]:
    return [r["tasks"][task]["protocols"][proto] for r in sorted(records, key=lambda r: r["seed"])
            if r["profile"] == profile]


def summarize(plan_config: dict, records: list[dict], profiles: list[str], interpretation: dict) -> dict:
    groups = {}
    for profile in profiles:
        for task in TASKS:
            for proto in ("next_event", "window"):
                ps = _group(records, profile, task, proto)
                g = {"n_seeds": len(ps), "test_cases_per_seed": [p["n_test_cases"] for p in ps],
                     "models": {n: {f"@{k}": {m: across_seed_stats([p["models"][n][f"@{k}"][m] for p in ps])
                                              for m in RANK} for k in KS} for n in MODELS},
                     "comparisons": {}, "initialization_sd_mean": {
                         n: {f"@{k}": _mean_or_none([p["initialization"][n][f"@{k}"]["ndcg_sd"] for p in ps])
                             for k in KS} for n in ("sasrec_full", "static_hybrid")}}
                for comp in ps[0]["comparisons"]:
                    g["comparisons"][comp] = {}
                    for k in KS:
                        for m in ("ndcg", "map"):
                            d = [p["comparisons"][comp][f"@{k}"][m] for p in ps]
                            s = across_seed_stats([x["mean_diff"] for x in d])
                            s["seeds_ci_above_0"] = sum(x["ci_low"] > 0 for x in d)
                            s["seeds_ci_below_0"] = sum(x["ci_high"] < 0 for x in d)
                            g["comparisons"][comp][f"@{k}/{m}"] = s
                if proto == "next_event":
                    g["diagnostics"] = {n: {f"@{k}": across_seed_stats([p["diagnostics"][n][f"@{k}"]["ndcg"] for p in ps])
                                            for k in KS} for n in DIAGNOSTICS}
                    g["strata"] = {n: {f"@{k}": {s: across_seed_stats([p["strata"][n][f"@{k}"][s]["ndcg"] for p in ps
                                                                       if p["strata"][n][f"@{k}"][s]["ndcg"] is not None])
                                                 for s in ("within_session", "cross_session")} for k in KS}
                                   for n in (*MODELS, *DIAGNOSTICS)}
                groups[f"{profile}/{task}/{proto}"] = g
    return {"label": SYNTHETIC_LABEL,
            "experiment": "Phase 3.1 final evaluation: Personal Frequency vs SASRec Full vs frozen static hybrid",
            "run_config": plan_config, "run_config_sha256": I.canonical_hash(plan_config),
            "statistical_design": {
                "within_seed": "paired percentile bootstrap n_boot=1000 on per-case means over the 3 initializations; "
                               "customer clusters (next-event), customer-task cases (window)",
                "across_seed": "Student-t 95% interval over the dataset seeds of one profile x task; never pooled",
                "initialization": "per-dataset SD over the 3 initializations, averaged; never merged into intervals"},
            "groups": groups, "criteria": criteria(groups, interpretation),
            "cost": cost(records),
            "notes": ["Primary: next-event NDCG@1. Window is secondary; nothing was selected on window metrics.",
                      "Oracle router reads the evaluator's within/cross label: diagnostic upper bound only.",
                      "alpha is the pinned phase3_1-config value; no alpha search on final data."],
            "datasets": [{"dataset_id": r["dataset_id"], "profile": r["profile"], "seed": r["seed"],
                          "fingerprints": r["fingerprints"]} for r in records]}


def criteria(groups: dict, interpretation: dict) -> dict:
    def comp(profile, task, name):
        return groups[f"{profile}/{task}/next_event"]["comparisons"][name]["@1/ndcg"]
    rq1 = {f"{p}/{t}": {"hybrid_minus_pf": comp(p, t, "static_hybrid - personal_frequency")} for p, t in RQ1_GROUPS}
    for v in rq1.values():
        lo = v["hybrid_minus_pf"]["ci_low"]
        v["met"] = None if lo is None else bool(lo > -MARGIN)
    p, t = RQ2_GROUP
    vs_pf, vs_sas = comp(p, t, "static_hybrid - personal_frequency"), comp(p, t, "static_hybrid - sasrec_full")
    rq2_met = None if None in (vs_pf["ci_low"], vs_sas["ci_low"]) else bool(vs_pf["ci_low"] > 0 and vs_sas["ci_low"] > -MARGIN)
    rq1_met = None if any(v["met"] is None for v in rq1.values()) else all(v["met"] for v in rq1.values())
    if rq1_met and rq2_met:
        outcome = interpretation["RQ1+RQ2"]
    elif rq2_met:
        outcome = interpretation["RQ2 only"]
    elif rq1_met:
        outcome = interpretation["RQ1 only"]
    else:
        outcome = "neither RQ1 nor RQ2 met"
    return {"margin_ndcg@1": MARGIN, "interval": "across-seed Student-t 95%, next-event NDCG@1",
            "RQ1": {"groups": rq1, "met": rq1_met},
            "RQ2": {"group": f"{p}/{t}", "hybrid_minus_pf": vs_pf, "hybrid_minus_sasrec": vs_sas, "met": rq2_met},
            "RQ3": {"status": H.NOT_EVALUABLE, "reason": RQ3_REASON,
                    "interpretation": interpretation["RQ3 failed or not satisfiable"]},
            "RQ4": "see cost", "outcome": outcome}


def cost(records: list[dict]) -> dict:
    fits = [rt for r in records for t in r["tasks"].values() for rt in t["sasrec_runtime"]]
    return {"datasets": len(records), "sasrec_fits": len(fits),
            "sasrec_train_seconds_total": round(sum(f["train_seconds"] for f in fits), 3),
            "dataset_wall_seconds_total": round(sum(r["wall_seconds"] for r in records), 3),
            "peak_gpu_memory_mib_max": max((f["peak_gpu_memory_mib"] or 0) for f in fits) if fits else None,
            "parameters": sorted({f["parameters"] for f in fits}),
            "test_inference_ms_per_case_mean": {p: round(float(np.mean([f[f"{p}_test_inference_ms_per_case"]
                                                                         for f in fits])), 5) if fits else None
                                                for p in ("next_event", "window")},
            "not_measured": ["OOF cross-fitting and gate fitting (not run on final data)"]}


def to_markdown(s: dict) -> str:
    def iv(x):
        return "n/a" if x["mean"] is None else f"{x['mean']:+.4f} [{x['ci_low']:+.4f}, {x['ci_high']:+.4f}]"
    c = s["criteria"]
    lines = ["# Phase 3.1 final evaluation", "", f"> **{s['label']}**", "",
             f"- Static hybrid alpha (pinned): {s['run_config']['alpha']}; init seeds {s['run_config']['init_seeds']}.",
             "- Brackets: across-seed Student-t 95% intervals (next-event NDCG@1).", "",
             "## Preregistered criteria", "", "| Criterion | Group | Difference | Met |", "|---|---|---|---|"]
    for g, v in c["RQ1"]["groups"].items():
        lines.append(f"| RQ1 hybrid - PF > -0.01 | {g} | {iv(v['hybrid_minus_pf'])} | {v['met']} |")
    lines += [f"| RQ2 hybrid - PF > 0 | {c['RQ2']['group']} | {iv(c['RQ2']['hybrid_minus_pf'])} | {c['RQ2']['met']} |",
              f"| RQ2 hybrid - SASRec > -0.01 | {c['RQ2']['group']} | {iv(c['RQ2']['hybrid_minus_sasrec'])} | |",
              f"| RQ3 | all | {c['RQ3']['status']} | - |", "",
              f"Outcome: {c['outcome']}. RQ3: {c['RQ3']['reason']}", "", "## Models (next-event NDCG@1)", "",
              "| Group | PF | SASRec Full | Static hybrid | Oracle (diagnostic) |", "|---|---|---|---|---|"]
    for key, g in s["groups"].items():
        if key.endswith("/next_event"):
            m = {n: g["models"][n]["@1"]["ndcg"]["mean"] for n in MODELS}
            o = g["diagnostics"]["oracle_router"]["@1"]["mean"]
            lines.append(f"| {key.removesuffix('/next_event')} | {m['personal_frequency']:.4f} | {m['sasrec_full']:.4f} "
                         f"| {m['static_hybrid']:.4f} | {o:.4f} |")
    return "\n".join(lines) + "\n"


# --- run, lock, CLI ------------------------------------------------------------------------------

def run(plan: Plan, root: Path = DATA_ROOT, output_dir: Path = OUTPUT_DIR, log=print) -> dict:
    if (output_dir / "phase3_1_report.json").exists():
        raise FinalRunRefused(f"{output_dir} already holds a final report; the final run happens once")
    records = []
    for e in plan.entries:
        log(f"  {e['dataset_id']}")
        records.append(run_dataset(plan, e, root, output_dir, log))
    profiles = plan.manifests[SEEDS_NAME]["seed_namespaces"][FINAL_NS]["profiles"]
    s = summarize(plan.run_config() | {"gates": plan.gate_log}, records, profiles,
                  plan.manifests[PREREG]["evaluation"]["interpretation"])
    s["runtime"] = plan.runtime
    (output_dir / "phase3_1_report.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output_dir / "phase3_1_report.md").write_text(to_markdown(s), encoding="utf-8", newline="\n")
    return s


def lock(output_dir: Path = OUTPUT_DIR, root: Path = DATA_ROOT, frozen_dir: Path = FROZEN_DIR,
         manifest_path: Path | None = None, manifests: dict | None = None) -> str:
    """Freeze the report and write the immutable `phase3_1` lock manifest (once)."""
    from bri.experiments.baselines import FROZEN_MARKER
    manifests = I.load_manifests() if manifests is None else manifests
    if LOCK_MANIFEST in manifests:
        raise FinalRunRefused("Phase 3.1 is already locked")
    s = json.loads((output_dir / "phase3_1_report.json").read_text(encoding="utf-8"))
    frozen_dir.mkdir(parents=True, exist_ok=False)
    for name in ("phase3_1_report.json", "phase3_1_report.md"):
        shutil.copyfile(output_dir / name, frozen_dir / name)
    (frozen_dir / FROZEN_MARKER).write_text(
        "Frozen Phase 3.1 report (synthetic-data experiment). Do not regenerate or edit.\n"
        f"Locked datasets and configuration: benchmarks/manifests/{LOCK_MANIFEST}.json\n", encoding="utf-8", newline="\n")
    datasets = []
    for d in s["datasets"]:
        proc = root / FINAL_NS / d["profile"] / f"seed{d['seed']}" / "processed"
        meta = json.loads((proc / "prep_metadata.json").read_text(encoding="utf-8"))
        e = I.dataset_entry(None, proc, meta, d["dataset_id"], ["next_event", "window", "phase3_1_hybrid"])
        e["seed_namespace"] = FINAL_NS
        datasets.append(e)
    try:
        rel = frozen_dir.resolve().relative_to(I.REPO_ROOT).as_posix()
    except ValueError:  # outside the repository (tests)
        rel = frozen_dir.resolve().as_posix()
    m = I.build_manifest(LOCK_MANIFEST, {}, datasets, dict(s["run_config"]["versions"]), rel,
                         "Phase 3.1 final evaluation; every listed test split used once, after phase3_1-config was "
                         "pinned.", extra={"phase3_1_run_config": s["run_config"],
                                           "phase3_1_run_config_sha256": s["run_config_sha256"],
                                           "runtime": s.get("runtime")})
    return I.write_manifest(m, manifest_path or (I.MANIFEST_DIR / f"{LOCK_MANIFEST}.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "lock"))
    parser.add_argument("--confirm-final-evaluation", action="store_true",
                        help="required for `run`: the Checkpoint 6 final run was approved")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if not args.confirm_final_evaluation:
                raise FinalRunRefused("`run` opens the final test splits once; pass --confirm-final-evaluation "
                                      "only after the Checkpoint 6 run is approved")
            s = run(gates())
            print(f"outcome: {s['criteria']['outcome']}; report {OUTPUT_DIR / 'phase3_1_report.json'}")
        else:
            digest = lock()
            print(f"{LOCK_MANIFEST} written; pin sha256 {digest} in bri.experiments.integrity.MANIFEST_SHA256")
    except I.LockedTestSplitError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
