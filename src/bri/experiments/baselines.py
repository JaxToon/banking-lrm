"""Phase 2.1 classical baseline experiment (synthetic-data experiment only).

Protocol, applied to NBS and NBP separately:
1. Fit every grid configuration on the TRAIN split.
2. Select one configuration per model family by the selection metric on VALIDATION.
3. Evaluate only the selected configurations once on TEST.
4. Report paired bootstrap CIs (customer/case resampling) and a multi-seed random floor.

Test data is never read before step 3 and never influences selection.

Usage:
    python -m bri.experiments.baselines --processed-dir data/processed --output-dir artifacts/baselines
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bri.eval.bootstrap import bootstrap_ci, paired_bootstrap_diff
from bri.eval.evaluate import RANK_METRICS, evaluate, per_case_metrics, sanity_recommendations
from bri.models.baselines import GRIDS, Popularity, make_model
from bri.models.common import history_matrix, to_recommendations, user_item_matrix
from bri.tasks import SYNTHETIC_LABEL, TASKS

REFERENCE_MODEL = "popularity"


def load_processed(processed_dir: Path) -> dict:
    def read(name):
        return pd.read_csv(processed_dir / name, dtype=str, keep_default_na=False)

    meta = json.loads((processed_dir / "prep_metadata.json").read_text(encoding="utf-8"))
    inter = read("interactions.csv")
    inter["event_timestamp"] = pd.to_datetime(inter["event_timestamp"], utc=True, format="ISO8601")
    inter["is_target"] = inter["is_target"] == "True"
    cases = read("eval_cases.csv")
    cases["cutoff_utc"] = pd.to_datetime(cases["cutoff_utc"], utc=True, format="ISO8601")
    return {
        "meta": meta,
        "interactions": inter,
        "cases": cases,
        "targets": read("eval_targets.csv"),
        "candidates": {name: t["candidates"] for name, t in meta["tasks"].items()},
    }


class _TaskData:
    """Caches train matrices and case histories per signal for one task."""

    def __init__(self, data: dict, task: str):
        self.task, self.items = task, data["candidates"][task]
        self.candidates = data["candidates"]
        inter = data["interactions"]
        self.inter = inter[inter["task"] == task]
        self.train = self.inter[self.inter["split"] == "train"]
        cases = data["cases"]
        self.cases = {s: cases[(cases["task"] == task) & (cases["split"] == s)].sort_values("case_id")
                      for s in ("val", "test")}
        t = data["targets"]
        self.targets = {s: t[t["case_id"].isin(c["case_id"])] for s, c in self.cases.items()}
        self._train_m, self._hist = {}, {}
        # Shared deterministic tie-break for every model: train popularity (target signal).
        self.tiebreak = Popularity().fit(self.train_matrix("target")).popularity_

    def train_matrix(self, signal: str) -> np.ndarray:
        if signal not in self._train_m:
            self._train_m[signal] = user_item_matrix(self.train, self.items, signal)
        return self._train_m[signal]

    def history(self, split: str, signal: str) -> np.ndarray:
        if (split, signal) not in self._hist:
            self._hist[(split, signal)] = history_matrix(self.inter, self.cases[split], self.items, signal)
        return self._hist[(split, signal)]

    def recommend(self, model, split: str) -> pd.DataFrame:
        scores = model.score(self.history(split, model.signal))
        return to_recommendations(self.cases[split]["case_id"], scores, self.items, self.tiebreak)


def _fit(family: str, params: dict, td: _TaskData, seed: int):
    p = {**params, "seed": seed} if family == "mf" else dict(params)
    return make_model(family, p).fit(td.train_matrix(p["signal"]))


def _select(family: str, td: _TaskData, metric: str, k: int, seed: int) -> tuple[dict, list[dict]]:
    """Grid search on VALIDATION only. Ties keep the earlier grid entry."""
    trials, best = [], None
    for params in GRIDS[family]:
        if td.cases["val"].empty:
            score = float("nan")
        else:
            recs = td.recommend(_fit(family, params, td, seed), "val")
            pc = per_case_metrics(td.cases["val"], td.targets["val"], recs, td.candidates, k)
            score = round(float(pc[metric].mean()), 6)
        trials.append({"params": params, f"val_{metric}@{k}": score})
        if best is None or (not np.isnan(score) and (np.isnan(best[1]) or score > best[1])):
            best = (params, score)
    return best[0], trials


def run_baselines(
    processed_dir: Path,
    ks: tuple[int, ...] = (1, 3),
    select_metric: str = "ndcg",
    select_k: int = 3,
    n_boot: int = 1000,
    seed: int = 42,
    random_seeds: int = 5,
    confidence: float = 0.95,
) -> dict:
    if select_metric not in RANK_METRICS:
        raise ValueError(f"select_metric must be one of {RANK_METRICS}")
    if select_k not in ks:
        raise ValueError("select_k must be one of ks")
    if random_seeds < 1:
        raise ValueError("random_seeds must be >= 1")
    data = load_processed(processed_dir)
    meta = data["meta"]
    tasks_out = {}
    for task in TASKS:
        td = _TaskData(data, task)
        test_c, test_t = td.cases["test"], td.targets["test"]
        entry = {"n_val_cases": len(td.cases["val"]), "n_test_cases": len(test_c),
                 "candidates": td.items, "models": {}}

        # Steps 1-2: fit on train, select on validation. Test is not touched here.
        selected = {}
        for family in GRIDS:
            params, trials = _select(family, td, select_metric, select_k, seed)
            selected[family] = params
            entry["models"][family] = {"selected_params": params, "validation_trials": trials}

        if test_c.empty:
            entry["note"] = "no test cases; test metrics undefined"
            tasks_out[task] = entry
            continue

        # Step 3: evaluate the selected configurations once on test.
        per_case = {}
        for family, params in selected.items():
            recs = td.recommend(_fit(family, params, td, seed), "test")
            entry["models"][family]["test"] = evaluate(test_c, test_t, recs, td.candidates, ks)[task]["metrics"]
            per_case[family] = {k: per_case_metrics(test_c, test_t, recs, td.candidates, k).set_index("case_id")
                                for k in ks}

        # Step 4a: bootstrap CI per model, and paired difference vs the reference model.
        for family in selected:
            cis, diffs = {}, {}
            for k in ks:
                pc, ref = per_case[family][k], per_case[REFERENCE_MODEL][k]
                cis[f"@{k}"] = {m: _round(bootstrap_ci(pc[m], n_boot, seed, confidence)) for m in RANK_METRICS}
                if family != REFERENCE_MODEL:
                    diffs[f"@{k}"] = {m: _round(paired_bootstrap_diff(pc[m], ref[m], n_boot, seed, confidence))
                                      for m in RANK_METRICS}
            entry["models"][family]["test_bootstrap_ci"] = cis
            if diffs:
                entry["models"][family][f"test_paired_diff_vs_{REFERENCE_MODEL}"] = diffs

        # Step 4b: random floor over several seeds.
        runs = [evaluate(test_c, test_t,
                         sanity_recommendations(test_c, test_t, td.candidates, "random", seed + s),
                         td.candidates, ks)[task]["metrics"] for s in range(random_seeds)]
        entry["random_floor"] = {
            "seeds": [seed + s for s in range(random_seeds)],
            "metrics": {f"@{k}": {m: {"mean": round(float(np.mean([r[f"@{k}"][m] for r in runs])), 6),
                                      "std": round(float(np.std([r[f"@{k}"][m] for r in runs])), 6)}
                                  for m in runs[0][f"@{k}"]} for k in ks},
        }
        tasks_out[task] = entry

    stats = meta["counts"]["eval_stats"]["test"]
    return {
        "label": SYNTHETIC_LABEL,
        "protocol": {
            "fit": "train split", "selection": f"validation {select_metric}@{select_k}",
            "final_evaluation": "test split, selected configurations only",
            "bootstrap": f"paired percentile bootstrap over cases, n_boot={n_boot}, confidence={confidence}",
            "reference_model": REFERENCE_MODEL, "seed": seed, "ks": list(ks),
        },
        "population_note": (
            "Results cover only customers with consent-permitted, sufficient same-task history "
            f"before the cutoff (min_history={meta['params']['min_history']}). "
            f"{stats['insufficient_history']} test customer-task pairs with targets were excluded as "
            "cold start. Cold start is not evaluated."
        ),
        "data": {"source": meta["source"], "cutoffs": meta["cutoffs"], "prep_sha256": meta["sha256"]},
        "tasks": tasks_out,
    }


def _round(d: dict) -> dict:
    return {k: round(v, 6) for k, v in d.items()}


def _fmt_ci(ci: dict) -> str:
    return f"{ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"


def to_markdown(report: dict) -> str:
    p = report["protocol"]
    lines = [
        "# Baseline comparison (Phase 2.1)", "",
        f"> **{report['label']}**", "",
        f"- Fit: {p['fit']}; selection: {p['selection']}; final: {p['final_evaluation']}.",
        f"- Uncertainty: {p['bootstrap']}. Brackets are 95% CIs unless stated otherwise.",
        f"- {report['population_note']}", "",
    ]
    for task, e in report["tasks"].items():
        lines += [f"## {task.upper()} (test cases: {e['n_test_cases']}, candidates: {len(e['candidates'])})", ""]
        if "note" in e:
            lines += [e["note"], ""]
            continue
        for k in p["ks"]:
            key = f"@{k}"
            lines += [
                f"### K = {k}", "",
                f"| Model | Selected config | P{key} | R{key} | NDCG{key} [CI] | MAP{key} | Coverage{key} | Exposure diversity{key} | Diff NDCG{key} vs popularity [CI] |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            rf = e["random_floor"]["metrics"][key]
            lines.append(
                f"| random floor ({len(e['random_floor']['seeds'])} seeds, mean+-sd) | - "
                f"| {rf['precision']['mean']:.3f}+-{rf['precision']['std']:.3f} "
                f"| {rf['recall']['mean']:.3f}+-{rf['recall']['std']:.3f} "
                f"| {rf['ndcg']['mean']:.3f}+-{rf['ndcg']['std']:.3f} "
                f"| {rf['map']['mean']:.3f}+-{rf['map']['std']:.3f} "
                f"| {rf['coverage']['mean']:.3f} | {rf['exposure_diversity']['mean']:.3f} | - |"
            )
            for family, m in e["models"].items():
                t, ci = m["test"][key], m["test_bootstrap_ci"][key]
                diff = m.get(f"test_paired_diff_vs_{REFERENCE_MODEL}", {}).get(key)
                d = (f"{diff['ndcg']['mean_diff']:+.3f} [{diff['ndcg']['ci_low']:+.3f}, {diff['ndcg']['ci_high']:+.3f}]"
                     if diff else "(reference)")
                cfg = ", ".join(f"{a}={b}" for a, b in m["selected_params"].items())
                lines.append(
                    f"| {family} | {cfg} | {t['precision']:.3f} | {t['recall']:.3f} | {_fmt_ci(ci['ndcg'])} "
                    f"| {t['map']:.3f} | {t['coverage']:.3f} | {t['exposure_diversity']:.3f} | {d} |"
                )
            lines.append("")
    lines += [
        "Notes: exposure diversity measures how evenly recommendations spread across the catalog, "
        "not semantic diversity within one customer's list. A difference whose CI includes 0 is not "
        "a reliable improvement on this synthetic data.", "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--select-metric", choices=RANK_METRICS, default="ndcg")
    parser.add_argument("--select-k", type=int, default=3)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        report = run_baselines(args.processed_dir, tuple(args.k), args.select_metric, args.select_k,
                               args.n_boot, args.seed, args.random_seeds)
    except FileNotFoundError as exc:
        print(f"EXPERIMENT FAILED: {exc} (run `python -m bri.prep.build` first)", file=sys.stderr)
        return 1
    except ValueError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baseline_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    md = to_markdown(report)
    (args.output_dir / "baseline_report.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)
    print(f"Reports written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
