"""Classical baseline experiments (synthetic-data experiments only).

Protocol, applied to NBS and NBP separately, for every model of an experiment suite:
1. Fit every grid configuration on the TRAIN split.
2. Select one configuration per model by the selection metric on VALIDATION.
3. Evaluate only the selected configurations once on TEST.
4. Report paired bootstrap CIs (customer/case resampling), all pairwise paired
   differences, the suite's key comparisons, and a multi-seed random floor.

Test data is never read before step 3 and never influences selection. Datasets whose
test split has been used are registered in LOCKED_DATASETS and can only be re-run with
the suite that originally used them (reproduction, not model selection).

Usage:
    python -m bri.experiments.baselines --suite phase2.2 \
        --processed-dir data/processed/seed20261109 --output-dir artifacts/phase2_2
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
from bri.models.baselines import KEY_COMPARISONS, SUITES, Popularity, make_model
from bri.models.common import history_matrix, to_recommendations, user_item_matrix
from bri.tasks import SYNTHETIC_LABEL, TASKS

REFERENCE_MODEL = "popularity"
DEFAULT_SUITE = "phase2.2"
FROZEN_MARKER = "FROZEN"

# events.csv sha256 of datasets whose test split has been used -> the only suite allowed.
LOCKED_DATASETS: dict[str, str] = {
    # generator 0.2.0, seed 42, 500 customers, 30 days (benchmark-v1, Phase 2.1)
    "ca53375dd030f13627ae3b7748ccdf209f32f2007657a3d918d84c3682a34fa8": "benchmark-v1",
    # generator 0.2.0, seed 20261109, 500 customers, 30 days (Phase 2.2 final evaluation)
    "9d68d4145a7c251b749ea38464a780126545b02934b07d80f01c145d20a6927b": "phase2.2",
}


class LockedTestSplitError(RuntimeError):
    pass


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


def check_lock(meta: dict, suite: str) -> None:
    owner = LOCKED_DATASETS.get(meta["source"].get("events_sha256"))
    if owner is not None and owner != suite:
        raise LockedTestSplitError(
            f"the test split of this dataset is locked by suite '{owner}'. It may only be "
            f"re-run with --suite {owner} for reproducibility; use a fresh dataset seed for "
            f"suite '{suite}'."
        )


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


def _select(family: str, grid: list[dict], td: _TaskData, metric: str, k: int, seed: int):
    """Grid search on VALIDATION only. Ties keep the earlier grid entry."""
    trials, best = [], None
    for params in grid:
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


def _round(d: dict) -> dict:
    return {k: round(v, 6) for k, v in d.items()}


def run_baselines(
    processed_dir: Path,
    ks: tuple[int, ...] = (1, 3),
    select_metric: str = "ndcg",
    select_k: int = 3,
    n_boot: int = 1000,
    seed: int = 42,
    random_seeds: int = 5,
    confidence: float = 0.95,
    suite: str = DEFAULT_SUITE,
) -> dict:
    if suite not in SUITES:
        raise ValueError(f"suite must be one of {sorted(SUITES)}")
    if select_metric not in RANK_METRICS:
        raise ValueError(f"select_metric must be one of {RANK_METRICS}")
    if select_k not in ks:
        raise ValueError("select_k must be one of ks")
    if random_seeds < 1:
        raise ValueError("random_seeds must be >= 1")
    data = load_processed(processed_dir)
    meta = data["meta"]
    check_lock(meta, suite)
    models = SUITES[suite]
    tasks_out = {}
    for task in TASKS:
        td = _TaskData(data, task)
        test_c, test_t = td.cases["test"], td.targets["test"]
        entry = {"n_val_cases": len(td.cases["val"]), "n_test_cases": len(test_c),
                 "candidates": td.items, "models": {}}

        # Steps 1-2: fit on train, select on validation. Test is not touched here.
        selected = {}
        for name, (family, grid) in models.items():
            params, trials = _select(family, grid, td, select_metric, select_k, seed)
            selected[name] = (family, params)
            entry["models"][name] = {"family": family, "selected_params": params, "validation_trials": trials}

        if test_c.empty:
            entry["note"] = "no test cases; test metrics undefined"
            tasks_out[task] = entry
            continue

        # Step 3: evaluate the selected configurations once on test.
        per_case = {}
        for name, (family, params) in selected.items():
            recs = td.recommend(_fit(family, params, td, seed), "test")
            entry["models"][name]["test"] = evaluate(test_c, test_t, recs, td.candidates, ks)[task]["metrics"]
            per_case[name] = {k: per_case_metrics(test_c, test_t, recs, td.candidates, k).set_index("case_id")
                              for k in ks}

        def diff(a, b, k, metric):
            return _round(paired_bootstrap_diff(per_case[a][k][metric], per_case[b][k][metric],
                                                n_boot, seed, confidence))

        # Step 4a: bootstrap CI per model, and paired difference vs the reference model.
        for name in selected:
            entry["models"][name]["test_bootstrap_ci"] = {
                f"@{k}": {m: _round(bootstrap_ci(per_case[name][k][m], n_boot, seed, confidence))
                          for m in RANK_METRICS} for k in ks}
            if name != REFERENCE_MODEL:
                entry["models"][name][f"test_paired_diff_vs_{REFERENCE_MODEL}"] = {
                    f"@{k}": {m: diff(name, REFERENCE_MODEL, k, m) for m in RANK_METRICS} for k in ks}

        # Step 4b: every model pair on the selection metric, at every K.
        names = list(selected)
        entry["pairwise_paired_diff"] = {
            f"@{k}": {f"{a} - {b}": diff(a, b, k, select_metric)
                      for i, a in enumerate(names) for b in names[i + 1:]}
            for k in ks}

        # Step 4c: the suite's key comparisons, on every rank metric.
        entry["key_comparisons"] = [
            {"comparison": f"{a} - {b}", "question": q,
             "results": {f"@{k}": {m: diff(a, b, k, m) for m in RANK_METRICS} for k in ks}}
            for a, b, q in KEY_COMPARISONS[suite]]

        # Step 4d: random floor over several seeds.
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
        "suite": suite,
        "protocol": {
            "fit": "train split", "selection": f"validation {select_metric}@{select_k}",
            "final_evaluation": "test split, selected configurations only",
            "bootstrap": f"paired percentile bootstrap over cases, n_boot={n_boot}, confidence={confidence}",
            "ci_scope": "case-to-case variation within this one dataset seed only",
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


def _fmt_ci(ci: dict) -> str:
    return f"{ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"


def _fmt_diff(d: dict) -> str:
    return f"{d['mean_diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


def to_markdown(report: dict) -> str:
    p = report["protocol"]
    src = report["data"]["source"]
    sel = p["selection"].split()[-1]
    lines = [
        f"# Baseline comparison (suite {report.get('suite', 'benchmark-v1')})", "",
        f"> **{report['label']}**", "",
        f"- Dataset: generator {src.get('generator_version')}, seed {src.get('seed')}.",
        f"- Fit: {p['fit']}; selection: {p['selection']}; final: {p['final_evaluation']}.",
        f"- Uncertainty: {p['bootstrap']}. Brackets are 95% CIs. "
        f"CIs reflect {p.get('ci_scope', 'case-to-case variation within one dataset')}.",
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
            for name, m in e["models"].items():
                t, ci = m["test"][key], m["test_bootstrap_ci"][key]
                d = m.get(f"test_paired_diff_vs_{REFERENCE_MODEL}", {}).get(key)
                cfg = ", ".join(f"{a}={b}" for a, b in m["selected_params"].items())
                lines.append(
                    f"| {name} | {cfg} | {t['precision']:.3f} | {t['recall']:.3f} | {_fmt_ci(ci['ndcg'])} "
                    f"| {t['map']:.3f} | {t['coverage']:.3f} | {t['exposure_diversity']:.3f} "
                    f"| {_fmt_diff(d['ndcg']) if d else '(reference)'} |"
                )
            lines.append("")
        if e["key_comparisons"]:
            lines += ["### Key comparisons (paired bootstrap, test)", "",
                      "| Comparison | Question | " + " | ".join(f"NDCG@{k} diff [CI]" for k in p["ks"])
                      + " | " + " | ".join(f"MAP@{k} diff [CI]" for k in p["ks"]) + " |",
                      "|---|---|" + "---|" * (2 * len(p["ks"]))]
            for c in e["key_comparisons"]:
                r = c["results"]
                lines.append(f"| {c['comparison']} | {c['question']} | "
                             + " | ".join(_fmt_diff(r[f"@{k}"]["ndcg"]) for k in p["ks"]) + " | "
                             + " | ".join(_fmt_diff(r[f"@{k}"]["map"]) for k in p["ks"]) + " |")
            lines.append("")
        lines += [f"### All pairwise paired differences ({sel.split('@')[0]}, test)", "",
                  "| Comparison | " + " | ".join(f"@{k} diff [CI]" for k in p["ks"]) + " |",
                  "|---|" + "---|" * len(p["ks"])]
        pairs = e["pairwise_paired_diff"][f"@{p['ks'][0]}"]
        for pair in pairs:
            lines.append(f"| {pair} | " + " | ".join(
                _fmt_diff(e["pairwise_paired_diff"][f"@{k}"][pair]) for k in p["ks"]) + " |")
        lines.append("")
    lines += [
        "Notes: exposure diversity measures how evenly recommendations spread across the catalog, "
        "not semantic diversity within one customer's list. A difference whose CI includes 0 is not "
        "a reliable difference on this synthetic data.", "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=sorted(SUITES), default=DEFAULT_SUITE)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--select-metric", choices=RANK_METRICS, default="ndcg")
    parser.add_argument("--select-k", type=int, default=3)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if (args.output_dir / FROZEN_MARKER).exists():
        print(f"EXPERIMENT REFUSED: {args.output_dir} is a frozen benchmark; choose another --output-dir",
              file=sys.stderr)
        return 1
    try:
        report = run_baselines(args.processed_dir, tuple(args.k), args.select_metric, args.select_k,
                               args.n_boot, args.seed, args.random_seeds, suite=args.suite)
    except FileNotFoundError as exc:
        print(f"EXPERIMENT FAILED: {exc} (run `python -m bri.prep.build` first)", file=sys.stderr)
        return 1
    except LockedTestSplitError as exc:
        print(f"EXPERIMENT REFUSED: {exc}", file=sys.stderr)
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
