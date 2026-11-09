"""Shared offline evaluation for NBS and NBP (synthetic-data experiments only).

Recommendations are a table with columns `case_id`, `item_id`, `rank` (1 = best).
Every recommended item must belong to the case's task candidate set. Each task is
scored separately over all of its cases; a case with no recommendations scores 0.

Usage:
    python -m bri.eval.evaluate --processed-dir data/processed --split val --ranker random
    python -m bri.eval.evaluate --processed-dir data/processed --split val --recommendations recs.csv

The built-in `random` and `oracle` rankers are evaluation sanity checks (floor and
ceiling), not recommendation models. `oracle` reads the answers and must never be
reported as a model result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bri.eval import metrics as M
from bri.tasks import SYNTHETIC_LABEL, TASKS

REC_COLUMNS = ["case_id", "item_id", "rank"]
DEFAULT_KS = (1, 3)
SANITY_RANKERS = ("random", "oracle")


class InvalidRecommendationsError(ValueError):
    pass


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_recommendations(
    recs: pd.DataFrame, cases: pd.DataFrame, candidates: dict[str, list[str]]
) -> list[str]:
    """Return a list of problems; empty means the recommendations are well-formed."""
    missing = [c for c in REC_COLUMNS if c not in recs.columns]
    if missing:
        return [f"recommendations missing columns {missing}"]
    errors = []
    r = recs[REC_COLUMNS].astype(str)
    if (r == "").any().any() or recs[REC_COLUMNS].isna().any().any():
        errors.append("recommendations contain empty values")
    task_of = dict(zip(cases["case_id"], cases["task"]))
    unknown = ~r["case_id"].isin(task_of)
    if unknown.any():
        errors.append(f"{int(unknown.sum())} row(s) reference unknown case_id (wrong split?), "
                      f"e.g. {r.loc[unknown, 'case_id'].iloc[0]}")
    known = r[~unknown]
    allowed = {t: set(c) for t, c in candidates.items()}
    bad_item = [item not in allowed[task_of[case]] for case, item in zip(known["case_id"], known["item_id"])]
    if any(bad_item):
        sample = known[bad_item].iloc[0]
        errors.append(f"{sum(bad_item)} recommended item(s) not in the task candidate set, "
                      f"e.g. {sample['item_id']} for {sample['case_id']}")
    if r.duplicated(["case_id", "item_id"]).any():
        errors.append("duplicate item within a recommendation list")
    rank = pd.to_numeric(r["rank"], errors="coerce")
    if rank.isna().any() or (rank != rank.round()).any() or (rank < 1).any():
        errors.append("rank must be an integer >= 1")
    else:
        expected = r.assign(_rank=rank.astype(int)).groupby("case_id")["_rank"].apply(
            lambda s: sorted(s) == list(range(1, len(s) + 1)))
        if not expected.all():
            errors.append("ranks must be unique and contiguous from 1 within each case")
    return errors


def evaluate(
    cases: pd.DataFrame,
    targets: pd.DataFrame,
    recs: pd.DataFrame,
    candidates: dict[str, list[str]],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict:
    """Score recommendations per task. Raises InvalidRecommendationsError on bad input."""
    if not ks or any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("ks must be positive integers")
    errors = validate_recommendations(recs, cases, candidates)
    if errors:
        raise InvalidRecommendationsError("; ".join(errors))

    relevant = targets.groupby("case_id")["item_id"].apply(set).to_dict()
    ordered = recs.assign(_rank=pd.to_numeric(recs["rank"]).astype(int)).sort_values(["case_id", "_rank"])
    lists = ordered.groupby("case_id")["item_id"].apply(list).to_dict()

    results = {}
    for name in TASKS:
        task_cases = sorted(cases.loc[cases["task"] == name, "case_id"])
        cands = candidates[name]
        entry = {
            "n_cases": len(task_cases),
            "candidate_count": len(cands),
            "cases_without_recommendations": sum(c not in lists for c in task_cases),
            "metrics": {},
        }
        if any(k >= len(cands) for k in ks):
            entry["note"] = "K >= candidate_count makes Recall/coverage trivial at that K."
        for k in ks:
            if not task_cases:
                entry["metrics"][f"@{k}"] = None  # undefined, not zero
                continue
            per_case = {"precision": [], "recall": [], "ndcg": [], "map": []}
            for case in task_cases:
                ranked, rel = lists.get(case, []), relevant[case]
                per_case["precision"].append(M.precision_at_k(ranked, rel, k))
                per_case["recall"].append(M.recall_at_k(ranked, rel, k))
                per_case["ndcg"].append(M.ndcg_at_k(ranked, rel, k))
                per_case["map"].append(M.average_precision_at_k(ranked, rel, k))
            case_lists = [lists.get(c, []) for c in task_cases]
            entry["metrics"][f"@{k}"] = {
                **{m: round(float(np.mean(v)), 6) for m, v in per_case.items()},
                "coverage": round(M.catalog_coverage(case_lists, cands, k), 6),
                "diversity": round(M.exposure_diversity(case_lists, cands, k), 6),
            }
        results[name] = entry
    return results


def sanity_recommendations(
    cases: pd.DataFrame, targets: pd.DataFrame, candidates: dict[str, list[str]], ranker: str, seed: int
) -> pd.DataFrame:
    """Full-catalog rankings from a sanity-check ranker ('random' floor or 'oracle' ceiling)."""
    if ranker not in SANITY_RANKERS:
        raise ValueError(f"ranker must be one of {SANITY_RANKERS}")
    rng = np.random.default_rng(seed)
    relevant = targets.groupby("case_id")["item_id"].apply(set).to_dict()
    rows = []
    for case, task in sorted(zip(cases["case_id"], cases["task"])):
        cands = candidates[task]
        if ranker == "random":
            order = [cands[i] for i in rng.permutation(len(cands))]
        else:
            rel = relevant.get(case, set())
            order = sorted(cands, key=lambda c: (c not in rel, c))
        rows += [(case, item, r) for r, item in enumerate(order, start=1)]
    return pd.DataFrame(rows, columns=REC_COLUMNS)


def run(processed_dir: Path, split: str, ks: tuple[int, ...], ranker: str | None,
        recommendations: Path | None, seed: int) -> dict:
    meta = json.loads((processed_dir / "prep_metadata.json").read_text(encoding="utf-8"))
    cases = _read(processed_dir / "eval_cases.csv")
    cases = cases[cases["split"] == split]
    targets = _read(processed_dir / "eval_targets.csv")
    targets = targets[targets["case_id"].isin(cases["case_id"])]
    candidates = {name: t["candidates"] for name, t in meta["tasks"].items()}
    if recommendations is not None:
        recs, source = _read(recommendations), str(recommendations.name)
    else:
        recs, source = sanity_recommendations(cases, targets, candidates, ranker, seed), f"sanity:{ranker}"
    return {
        "label": SYNTHETIC_LABEL,
        "split": split,
        "recommendations": source,
        "seed": seed if recommendations is None else None,
        "ks": list(ks),
        "protocol": "full-catalog ranking over active task candidates; binary relevance; macro-average over cases",
        "data": {"source": meta["source"], "cutoffs": meta["cutoffs"], "prep_sha256": meta["sha256"]},
        "results": evaluate(cases, targets, recs, candidates, ks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ranker", choices=SANITY_RANKERS, help="built-in sanity-check ranker")
    src.add_argument("--recommendations", type=Path, help="CSV with case_id,item_id,rank")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    args = parser.parse_args(argv)
    try:
        report = run(args.processed_dir, args.split, tuple(args.k), args.ranker, args.recommendations, args.seed)
    except InvalidRecommendationsError as exc:
        print(f"EVALUATION FAILED: invalid recommendations: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"EVALUATION FAILED: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")
        print(f"{SYNTHETIC_LABEL}\nReport written to {args.output}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
