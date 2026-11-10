"""Phase 2.3 multi-seed behavioral robustness experiment (synthetic-data experiments only).

For every behavior profile and every dataset seed:
  generate (generator 0.3.0) -> validate -> consent-aware temporal split ->
  window evaluation (suite phase2.3) and next-event evaluation (suite
  phase2.3-next-event), each: fit on train, select on validation, evaluate the selected
  configuration once on test, with Popularity, Personal Frequency (mandatory reference),
  ItemKNN self=True, ItemKNN self=False and implicit MF.

Statistical design (two levels: dataset seeds, then customers within a dataset):
- Within one seed: paired bootstrap CIs over cases (window) or customers (next-event).
  They describe case/customer variation inside that one synthetic dataset.
- Across seeds: each seed's test metric (or paired difference) is ONE observation;
  mean, sample SD and a Student-t 95% interval over seeds describe variation across
  independent synthetic datasets drawn from the same profile at the same size
  (customers, days). Within-seed CIs are never pooled across seeds, and no interval is
  computed across profiles (profiles are different experimental conditions).

Usage:
    python -m bri.experiments.robustness --output-dir artifacts/phase2_3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from bri.data.generate_synthetic import PROFILE_GENERATOR_VERSION, PROFILES, write_dataset
from bri.data.validate_dataset import validate
from bri.eval.next_event import NEXT_EVENT_VERSION
from bri.experiments import integrity as I
from bri.experiments.baselines import (FROZEN_MARKER, RUNNER_VERSION, experiment_config, protocol_config,
                                       run_baselines, run_next_event)
from bri.models.baselines import KEY_COMPARISONS, SUITES
from bri.prep.build import PREP_VERSION, build
from bri.tasks import SYNTHETIC_LABEL, TASKS

WINDOW_SUITE = "phase2.3"
NEXT_SUITE = "phase2.3-next-event"
SUITE_OF = {"window": WINDOW_SUITE, "next_event": NEXT_SUITE}
SUITE_MODELS = {e: tuple(SUITES[s]) for e, s in SUITE_OF.items()}
DEFAULT_SEEDS = tuple(range(20261110, 20261120))  # 10 seeds; none reused from locked benchmarks
METRICS = ("precision", "recall", "ndcg", "map", "coverage", "exposure_diversity")
COMPARISON_METRICS = ("ndcg", "map", "precision", "recall")
# Two-sided 95% Student-t critical values by degrees of freedom (avoids a SciPy dependency).
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
       10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
       18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
       26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


@dataclass(frozen=True)
class RobustnessConfig:
    profiles: tuple[str, ...] = tuple(PROFILES)
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    customers: int = 500
    days: int = 30
    val_days: int = 7
    test_days: int = 7
    min_history: int = 1
    ks: tuple[int, ...] = (1, 3)
    select_metric: str = "ndcg"
    select_k: int = 3
    n_boot: int = 1000
    random_seeds: int = 5
    model_seed: int = 42
    confidence: float = 0.95
    evaluations: tuple[str, ...] = field(default=("window", "next_event"))

    def __post_init__(self) -> None:
        unknown = set(self.profiles) - set(PROFILES)
        if unknown:
            raise ValueError(f"unknown profile(s) {sorted(unknown)}")
        if len(set(self.seeds)) != len(self.seeds) or not self.seeds:
            raise ValueError("seeds must be non-empty and distinct")
        if self.confidence != 0.95:
            raise ValueError("only confidence=0.95 is supported (fixed t table)")
        if len(self.seeds) - 1 > max(T95):
            raise ValueError(f"at most {max(T95) + 1} seeds are supported")

    def protocol(self, evaluation: str) -> dict:
        return protocol_config(evaluation, self.ks, self.select_metric, self.select_k, self.n_boot,
                               self.model_seed, self.random_seeds, self.confidence)

    def experiment_configs(self) -> dict[str, dict]:
        return {e: experiment_config(SUITE_OF[e], e, self.protocol(e)) for e in self.evaluations}

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}


# --- statistics --------------------------------------------------------------------

def across_seed_stats(values: list[float]) -> dict:
    """Mean, sample SD and Student-t 95% CI over seeds (each seed = one observation)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return {"n_seeds": 0, "mean": None, "sd": None, "ci_low": None, "ci_high": None}
    mean = float(v.mean())
    if n < 2:
        return {"n_seeds": n, "mean": round(mean, 6), "sd": None, "ci_low": None, "ci_high": None}
    sd = float(v.std(ddof=1))
    half = T95[n - 1] * sd / math.sqrt(n)
    return {"n_seeds": n, "mean": round(mean, 6), "sd": round(sd, 6),
            "ci_low": round(mean - half, 6), "ci_high": round(mean + half, 6)}


# --- per-dataset run -----------------------------------------------------------------

def _compact(report: dict) -> dict:
    """Per-seed numbers kept for the summary (full reports are written separately)."""
    out = {}
    for task, e in report["tasks"].items():
        t = {"n_val_cases": e["n_val_cases"], "n_test_cases": e["n_test_cases"], "models": {}, "comparisons": {}}
        if "note" in e:
            t["note"] = e["note"]
            out[task] = t
            continue
        for name, m in e["models"].items():
            t["models"][name] = {"selected_params": m["selected_params"], "test": m["test"]}
        for c in e["key_comparisons"]:
            t["comparisons"][c["comparison"]] = c["results"]
        t["random_floor"] = {k: {m: v["mean"] for m, v in mm.items()} for k, mm in e["random_floor"]["metrics"].items()}
        out[task] = t
    return out


def _strata(per_case: dict, within, ks) -> dict:
    """Next-event metric means split by whether the previous target was in the same session."""
    out = {}
    for task, models in per_case.items():
        out[task] = {}
        for name, by_k in models.items():
            frame = by_k[ks[0]]
            w = within.reindex(frame.index).astype(bool).to_numpy()
            out[task][name] = {
                f"@{k}": {
                    "within_session": {"n": int(w.sum()),
                                       "ndcg": round(float(by_k[k]["ndcg"].to_numpy()[w].mean()), 6) if w.any() else None},
                    "cross_session": {"n": int((~w).sum()),
                                      "ndcg": round(float(by_k[k]["ndcg"].to_numpy()[~w].mean()), 6) if (~w).any() else None},
                } for k in ks}
    return out


def run_dataset(cfg: RobustnessConfig, profile: str, seed: int, data_root: Path, runs_dir: Path,
                manifests: dict | None = None) -> dict:
    base = data_root / profile / f"seed{seed}"
    src, proc = base / "synthetic", base / "processed"
    gen_meta = write_dataset(src, cfg.customers, cfg.days, seed, profile)
    errors = validate(src)
    if errors:
        raise RuntimeError(f"{profile}/seed{seed} failed validation: {errors[:3]}")
    prep = build(src, proc, cfg.val_days, cfg.test_days, cfg.min_history)
    common = dict(ks=cfg.ks, select_metric=cfg.select_metric, select_k=cfg.select_k, n_boot=cfg.n_boot,
                  seed=cfg.model_seed, random_seeds=cfg.random_seeds, confidence=cfg.confidence,
                  manifests=manifests)
    out_dir = runs_dir / profile / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "profile": profile, "seed": seed,
        "validation": "passed",
        "fingerprints": {
            "source_composite": I.composite_fingerprint(prep["source_files_sha256"]),
            "processed_composite": I.processed_fingerprints(proc)["composite"],
            "prep_fingerprint": I.prep_fingerprint(prep),
        },
        "row_counts": gen_meta["row_counts"],
        "consent_filter": {k: prep["consent_filter"][k] for k in ("input_events", "kept_events")},
        "evaluations": {},
    }
    if "window" in cfg.evaluations:
        w = run_baselines(proc, suite=WINDOW_SUITE, **common)
        _write(out_dir / "window_report.json", w)
        st = prep["counts"]["eval_stats"]
        record["evaluations"]["window"] = {
            "cold_start": {s: {"customers_with_targets": st[s]["customers_with_targets"],
                               "insufficient_history_excluded": st[s]["insufficient_history"]} for s in st},
            "tasks": _compact(w)}
    if "next_event" in cfg.evaluations:
        n, per_case, within = run_next_event(proc, suite=NEXT_SUITE, return_per_case=True, **common)
        _write(out_dir / "next_event_report.json", n)
        record["evaluations"]["next_event"] = {
            "cold_start": {s: {"eligible_target_events": v["eligible_target_events"],
                               "insufficient_history_excluded": v["insufficient_history"],
                               "tied_timestamp_merges": v["tied_timestamp_merges"],
                               "single_target_sessions": v["single_target_sessions"]}
                           for s, v in n["case_stats"].items()},
            "case_stats_by_task": n["case_stats_by_task"],
            "tasks": _compact(n),
            "session_strata": _strata(per_case, within, cfg.ks)}
    return record


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")


# --- summary -------------------------------------------------------------------------

def summarize(cfg: RobustnessConfig, records: list[dict]) -> dict:
    by_profile = {}
    for profile in cfg.profiles:
        recs = sorted((r for r in records if r["profile"] == profile), key=lambda r: r["seed"])
        prof = {}
        for ev in cfg.evaluations:
            tasks = {}
            for task in TASKS:
                per_seed = [r["evaluations"][ev]["tasks"][task] for r in recs]
                usable = [(r["seed"], p) for r, p in zip(recs, per_seed) if "note" not in p]
                models = list(SUITE_MODELS[ev])
                t = {"n_seeds_with_test_cases": len(usable),
                     "test_cases_per_seed": [p["n_test_cases"] for p in per_seed],
                     "models": {}, "comparisons": {}, "per_seed": {}}
                for name in models:
                    t["models"][name] = {
                        f"@{k}": {m: across_seed_stats([p["models"][name]["test"][f"@{k}"][m] for _, p in usable])
                                  for m in METRICS} for k in cfg.ks}
                    sel = {}
                    for _, p in usable:
                        key = ", ".join(f"{a}={b}" for a, b in p["models"][name]["selected_params"].items())
                        sel[key] = sel.get(key, 0) + 1
                    t["models"][name]["selection_counts"] = dict(sorted(sel.items()))
                t["random_floor"] = {f"@{k}": {m: across_seed_stats([p["random_floor"][f"@{k}"][m] for _, p in usable])
                                               for m in ("ndcg", "map")} for k in cfg.ks}
                for a, b, _ in KEY_COMPARISONS[SUITE_OF[ev]]:
                    comp = f"{a} - {b}"
                    t["comparisons"][comp] = {}
                    for k in cfg.ks:
                        for m in COMPARISON_METRICS:
                            diffs = [p["comparisons"][comp][f"@{k}"][m] for _, p in usable]
                            s = across_seed_stats([d["mean_diff"] for d in diffs])
                            s["seeds_ci_above_0"] = sum(d["ci_low"] > 0 for d in diffs)
                            s["seeds_ci_below_0"] = sum(d["ci_high"] < 0 for d in diffs)
                            s["seeds_ci_includes_0"] = len(diffs) - s["seeds_ci_above_0"] - s["seeds_ci_below_0"]
                            t["comparisons"][comp][f"@{k}/{m}"] = s
                t["per_seed"] = {
                    str(seed): {
                        "n_test_cases": p["n_test_cases"],
                        "models": {name: {f"@{k}": p["models"][name]["test"][f"@{k}"] for k in cfg.ks}
                                   for name in models},
                        "comparisons": {c: {f"@{k}": {m: p["comparisons"][c][f"@{k}"][m] for m in ("ndcg", "map")}
                                            for k in cfg.ks} for c in p["comparisons"]},
                    } for seed, p in usable}
                tasks[task] = t
            cold = {}
            for split in ("val", "test"):
                keys = recs[0]["evaluations"][ev]["cold_start"][split].keys()
                cold[split] = {key: {"per_seed": [r["evaluations"][ev]["cold_start"][split][key] for r in recs],
                                     "total": int(sum(r["evaluations"][ev]["cold_start"][split][key] for r in recs))}
                               for key in keys}
            entry = {"tasks": tasks, "cold_start": cold}
            if ev == "next_event":
                entry["session_strata"] = {
                    task: {name: {f"@{k}": {s: across_seed_stats(
                        [r["evaluations"][ev]["session_strata"][task][name][f"@{k}"][s]["ndcg"] for r in recs
                         if r["evaluations"][ev]["session_strata"].get(task, {}).get(name, {}).get(f"@{k}", {})
                         .get(s, {}).get("ndcg") is not None])
                        for s in ("within_session", "cross_session")} for k in cfg.ks}
                        for name in SUITE_MODELS[ev]}
                    for task in TASKS}
                entry["cases_by_task"] = {task: {"test_cases_per_seed": [
                    r["evaluations"][ev]["case_stats_by_task"]["test"][task]["cases"] for r in recs],
                    "within_session_per_seed": [
                    r["evaluations"][ev]["case_stats_by_task"]["test"][task]["within_session_cases"] for r in recs]}
                    for task in TASKS}
            prof[ev] = entry
        by_profile[profile] = {"behavior_params": PROFILES[profile].to_dict(), "evaluations": prof}
    return {
        "label": SYNTHETIC_LABEL,
        "experiment": "phase2.3 multi-seed behavioral robustness",
        "config": cfg.to_dict(),
        "config_sha256": I.canonical_hash(cfg.to_dict()),
        "experiment_configs_sha256": {e: I.canonical_hash(c) for e, c in cfg.experiment_configs().items()},
        "versions": {"generator": PROFILE_GENERATOR_VERSION, "prep": PREP_VERSION, "runner": RUNNER_VERSION,
                     "next_event": NEXT_EVENT_VERSION},
        "statistical_design": {
            "within_seed": ("paired percentile bootstrap; resampling unit = customer-task case (window) or "
                            "customer (next-event, cluster bootstrap); describes variation inside one dataset"),
            "across_seed": ("each seed's test mean or paired difference is one observation; mean, sample SD, "
                            "Student-t 95% interval over seeds; describes variation across independent "
                            "synthetic datasets of one profile at fixed size; within-seed CIs are not pooled"),
            "not_covered": ("variation across profiles, population sizes, window lengths, generator "
                            "versions, or any real population; profiles are not compared with intervals"),
            "seed_sharing": ("the same seed list is used for every profile, so customer attributes and "
                             "consent are identical across profiles for a given seed; only behavior differs"),
        },
        "population_note": ("Results cover only customers / target events with sufficient consent-permitted "
                            f"same-task history (min_history={cfg.min_history}). Cold start is excluded and "
                            "counted, not evaluated."),
        "profiles": by_profile,
        "datasets": [{"profile": r["profile"], "seed": r["seed"], "validation": r["validation"],
                      "fingerprints": r["fingerprints"], "row_counts": r["row_counts"],
                      "consent_filter": r["consent_filter"]} for r in records],
    }


def preflight_lock_check(cfg: RobustnessConfig, manifests: dict) -> None:
    """Refuse BEFORE generating or writing anything if a planned dataset reuses a locked
    (generator, seed, profile) with different data parameters or configuration.

    Customer streams do not depend on population size, so reusing a locked seed with a
    different customer count or window would re-expose locked test customers; it is
    refused rather than treated as a fresh dataset.
    """
    configs = {e: I.canonical_hash(c) for e, c in cfg.experiment_configs().items()}
    for m in manifests.values():
        rc = m.get("robustness_config") or {}
        for entry in m["datasets"]:
            if (entry["generator_version"] != PROFILE_GENERATOR_VERSION or entry["seed"] not in cfg.seeds
                    or entry["behavior_profile"] not in cfg.profiles):
                continue
            ds = f"{entry['behavior_profile']}/seed{entry['seed']}"
            same_data = ((entry["customers"], entry["days"]) == (cfg.customers, cfg.days)
                         and all(rc.get(k) == getattr(cfg, k) for k in ("val_days", "test_days", "min_history")))
            if not same_data:
                raise I.LockedTestSplitError(
                    f"{ds} is locked by benchmark '{m['suite']}' with different data parameters "
                    "(customers/days/split); reusing a locked seed would re-expose its test customers. "
                    "Use fresh seeds.")
            for e, h in configs.items():
                if e not in entry["evaluations"] or m["configs"].get(e, {}).get("config_sha256") != h:
                    raise I.BenchmarkIntegrityError(
                        f"configuration differs from the one registered for locked dataset {ds} "
                        f"(benchmark '{m['suite']}', {e}); only the original configuration may be re-run")


def check_output_dir(output_dir: Path, cfg: RobustnessConfig) -> None:
    """Never replace an aggregate report produced by a different (e.g. partial) configuration."""
    if (output_dir / FROZEN_MARKER).exists():
        raise I.LockedTestSplitError(f"{output_dir} is a frozen benchmark; choose another --output-dir")
    existing = output_dir / "robustness_report.json"
    if existing.exists():
        old = json.loads(existing.read_text(encoding="utf-8")).get("config_sha256")
        if old != I.canonical_hash(cfg.to_dict()):
            raise I.LockedTestSplitError(
                f"{existing} was produced by a different configuration (e.g. the full run); a partial or "
                "changed run must use another --output-dir")


def run_robustness(cfg: RobustnessConfig, data_root: Path, output_dir: Path, manifests: dict | None = None,
                   log=print) -> dict:
    manifests = I.load_manifests() if manifests is None else manifests
    check_output_dir(output_dir, cfg)
    preflight_lock_check(cfg, manifests)
    runs_dir = output_dir / "runs"
    records = []
    for profile in cfg.profiles:
        for seed in cfg.seeds:
            t0 = time.perf_counter()
            records.append(run_dataset(cfg, profile, seed, data_root, runs_dir, manifests))
            log(f"  {profile} seed {seed}: done in {time.perf_counter() - t0:.1f}s")
    return summarize(cfg, records)


def lock_manifest(cfg: RobustnessConfig, data_root: Path, summary: dict, path: Path,
                  frozen_report_dir: str | None) -> str:
    """Register every dataset of this run as a locked benchmark (call once, after the
    final evaluation). Returns the manifest sha256 to pin in integrity.MANIFEST_SHA256."""
    datasets = []
    for profile in cfg.profiles:
        for seed in cfg.seeds:
            proc = data_root / profile / f"seed{seed}" / "processed"
            meta = json.loads((proc / "prep_metadata.json").read_text(encoding="utf-8"))
            datasets.append(I.dataset_entry(None, proc, meta, f"{profile}/seed{seed}", list(cfg.evaluations)))
    versions = dict(summary["versions"])
    manifest = I.build_manifest(WINDOW_SUITE, cfg.experiment_configs(), datasets, versions, frozen_report_dir,
                                "Phase 2.3 final evaluation; every listed test split used once.",
                                extra={"robustness_config": cfg.to_dict(),
                                       "robustness_config_sha256": I.canonical_hash(cfg.to_dict())})
    return I.write_manifest(manifest, path)


# --- markdown --------------------------------------------------------------------------

def _ms(s: dict, digits: int = 3) -> str:
    if s["sd"] is None:
        return f"{s['mean']:.{digits}f}"
    return f"{s['mean']:.{digits}f} +- {s['sd']:.{digits}f} [{s['ci_low']:.{digits}f}, {s['ci_high']:.{digits}f}]"


def _md(s: dict) -> str:
    if s["sd"] is None:
        return f"{s['mean']:+.3f}"
    return f"{s['mean']:+.3f} +- {s['sd']:.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"


def to_markdown(summary: dict) -> str:
    cfg = summary["config"]
    ks = cfg["ks"]
    lines = [
        "# Phase 2.3 multi-seed behavioral robustness", "",
        f"> **{summary['label']}**", "",
        f"- Generator {summary['versions']['generator']} profiles {', '.join(cfg['profiles'])}; "
        f"{len(cfg['seeds'])} seeds per profile ({cfg['seeds'][0]}..{cfg['seeds'][-1]}); "
        f"{cfg['customers']} customers, {cfg['days']} days, split "
        f"{cfg['days'] - cfg['val_days'] - cfg['test_days']}/{cfg['val_days']}/{cfg['test_days']}.",
        "- Per dataset: fit on train, select on validation "
        f"({cfg['select_metric']}@{cfg['select_k']}), evaluate the selected configuration once on test.",
        f"- Across-seed cells: mean +- SD [Student-t 95% interval over seeds]. {summary['statistical_design']['across_seed']}.",
        f"- Within-seed CIs: {summary['statistical_design']['within_seed']}. 'CI>0 / CI<0' counts seeds whose "
        "within-seed CI excludes 0 in that direction.",
        f"- Not covered: {summary['statistical_design']['not_covered']}.",
        f"- {summary['population_note']}",
        "- Behavior parameters are invented experimental assumptions, not real banking behavior.", "",
    ]
    for profile, pdata in summary["profiles"].items():
        bp = pdata["behavior_params"]
        lines += [f"## Profile `{profile}`", "",
                  f"Dirichlet concentration service/product = {bp['service_concentration']}/{bp['product_concentration']}; "
                  f"transition probability service/product = {bp['service_transition_p']}/{bp['product_transition_p']}.", ""]
        for ev, edata in pdata["evaluations"].items():
            lines += [f"### {ev.replace('_', '-')} evaluation", ""]
            for task, t in edata["tasks"].items():
                lines += [f"#### {task.upper()} (test cases per seed: {min(t['test_cases_per_seed'])}-"
                          f"{max(t['test_cases_per_seed'])})", "",
                          "| Model | NDCG@1 | NDCG@3 | P@3 | R@3 | MAP@3 | Coverage@3 | Exposure div.@3 |",
                          "|---|---|---|---|---|---|---|---|"]
                rf = t["random_floor"]
                lines.append(f"| random floor | {_ms(rf['@1']['ndcg'])} | {_ms(rf['@3']['ndcg'])} | - | - "
                             f"| {_ms(rf['@3']['map'])} | - | - |")
                for name, m in t["models"].items():
                    k3 = m["@3"]
                    lines.append(f"| {name} | {_ms(m['@1']['ndcg'])} | {_ms(k3['ndcg'])} | {k3['precision']['mean']:.3f} "
                                 f"| {k3['recall']['mean']:.3f} | {k3['map']['mean']:.3f} | {k3['coverage']['mean']:.3f} "
                                 f"| {k3['exposure_diversity']['mean']:.3f} |")
                lines += ["", "| Paired difference (within seed, then across seeds) | NDCG@1 | CI>0 / CI<0 | NDCG@3 | CI>0 / CI<0 |",
                          "|---|---|---|---|---|"]
                for comp, c in t["comparisons"].items():
                    a, b = c["@1/ndcg"], c[f"@{ks[-1]}/ndcg"]
                    lines.append(f"| {comp} | {_md(a)} | {a['seeds_ci_above_0']} / {a['seeds_ci_below_0']} "
                                 f"| {_md(b)} | {b['seeds_ci_above_0']} / {b['seeds_ci_below_0']} |")
                lines += ["", "Per-seed test NDCG@1 / NDCG@3:", "",
                          "| Seed | " + " | ".join(t["models"]) + " |", "|---|" + "---|" * len(t["models"])]
                for seed, ps in t["per_seed"].items():
                    lines.append(f"| {seed} | " + " | ".join(
                        f"{ps['models'][n]['@1']['ndcg']:.3f} / {ps['models'][n][f'@{ks[-1]}']['ndcg']:.3f}"
                        for n in t["models"]) + " |")
                sel = "; ".join(f"{n}: " + ", ".join(f"{k} x{v}" for k, v in m["selection_counts"].items())
                                for n, m in t["models"].items() if len(m["selection_counts"]) > 0)
                lines += ["", f"Selected configurations (seeds): {sel}", ""]
            if ev == "next_event":
                lines += ["Next-event NDCG@1 by session stratum (across-seed mean): within-session = previous "
                          "target in the same session; cross-session = first target of a session.", "",
                          "| Task | Model | within-session | cross-session |", "|---|---|---|---|"]
                def fmt(x):
                    return "-" if x is None else f"{x:.3f}"
                for task, models in edata["session_strata"].items():
                    for name, s in models.items():
                        lines.append(f"| {task} | {name} | {fmt(s['@1']['within_session']['mean'])} "
                                     f"| {fmt(s['@1']['cross_session']['mean'])} |")
                lines.append("")
            cs = edata["cold_start"]["test"]
            lines += ["Cold start / exclusions (test, total over seeds): " + "; ".join(
                f"{k} = {v['total']}" for k, v in cs.items()), ""]
    lines += ["Notes: exposure diversity measures how evenly recommendations spread across the catalog, "
              "not per-customer semantic diversity. All numbers are synthetic-data experiments and say "
              "nothing about real customers or real-world effectiveness.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILES), default=list(PROFILES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--customers", type=int, default=500)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--val-days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=Path("data/phase2_3"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase2_3"))
    parser.add_argument("--lock-manifest", type=Path, default=None,
                        help="after the final run, write an immutable manifest locking every test split")
    parser.add_argument("--frozen-report-dir", default=None,
                        help="repo-relative frozen report directory recorded in the lock manifest")
    args = parser.parse_args(argv)
    try:
        cfg = RobustnessConfig(profiles=tuple(args.profiles), seeds=tuple(args.seeds), customers=args.customers,
                               days=args.days, val_days=args.val_days, test_days=args.test_days,
                               n_boot=args.n_boot, random_seeds=args.random_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        check_output_dir(args.output_dir, cfg)  # refuse before printing or generating anything
        print(SYNTHETIC_LABEL)
        summary = run_robustness(cfg, args.data_root, args.output_dir)
    except I.LockedTestSplitError as exc:
        print(f"EXPERIMENT REFUSED: {exc}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "robustness_report.json", summary)
    md = to_markdown(summary)
    (args.output_dir / "robustness_report.md").write_text(md, encoding="utf-8", newline="\n")
    print(f"Reports written to {args.output_dir}")
    if args.lock_manifest:
        digest = lock_manifest(cfg, args.data_root, summary, args.lock_manifest, args.frozen_report_dir)
        print(f"Lock manifest written to {args.lock_manifest}; pin sha256 {digest} in "
              "bri.experiments.integrity.MANIFEST_SHA256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
