"""Generate Phase 3 TUNING datasets and report history-length statistics.

Only the `phase3_tuning` seed namespace is accepted: final-evaluation datasets must not
inform any hyperparameter decision. Statistics that drive the max_seq_len search space
use TRAIN-window cases only; VAL-window statistics are reported as a sanity check. Test
events are dropped before sequences are built, so no test-window information can enter.

Every check (manifests, seed namespace, data parameters, output directory) runs before
anything is generated or written.

Usage:
    python -m bri.seq.prepare --data-root data/phase3 --report-dir artifacts/phase3/seq_lengths
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bri.data.generate_synthetic import PROFILE_GENERATOR_VERSION, write_dataset
from bri.data.validate_dataset import validate
from bri.experiments import integrity as I
from bri.experiments.baselines import FROZEN_MARKER
from bri.prep.build import build
from bri.prep.split import compute_cutoffs
from bri.seq.data import SEQ_DATA_VERSION, build_seq_cases, length_stats
from bri.tasks import SYNTHETIC_LABEL, TASKS

NAMESPACE = "phase3_tuning"
DEFAULT_LENS = (5, 10, 20, 30, 50)


def preflight(namespace: str, manifests: dict, report_dir: Path) -> dict:
    """Validate everything before writing. Returns the namespace spec."""
    if namespace != NAMESPACE:
        raise I.LockedTestSplitError(f"sequence-length analysis may only use '{NAMESPACE}' datasets; "
                                     f"'{namespace}' must not inform hyperparameters")
    spec = next((m["seed_namespaces"][namespace] for m in manifests.values()
                 if namespace in m.get("seed_namespaces", {})), None)
    if spec is None:
        raise I.LockedTestSplitError(f"seed namespace '{namespace}' is not registered")
    I.check_seed_namespace(spec["seeds"], namespace, manifests)
    if spec["generator_version"] != PROFILE_GENERATOR_VERSION:
        raise I.BenchmarkIntegrityError(f"namespace requires generator {spec['generator_version']}, "
                                        f"code provides {PROFILE_GENERATOR_VERSION}")
    if (report_dir / FROZEN_MARKER).exists():
        raise I.LockedTestSplitError(f"{report_dir} is frozen")
    return spec


def load_interactions(processed_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    meta = json.loads((processed_dir / "prep_metadata.json").read_text(encoding="utf-8"))
    inter = pd.read_csv(processed_dir / "interactions.csv", dtype=str, keep_default_na=False)
    inter["event_timestamp"] = pd.to_datetime(inter["event_timestamp"], utc=True, format="ISO8601")
    inter["is_target"] = inter["is_target"] == "True"
    return inter, meta, {n: t["candidates"] for n, t in meta["tasks"].items()}


def no_test_events(inter: pd.DataFrame) -> pd.DataFrame:
    return inter[inter["split"] != "test"]


def run(data_root: Path, report_dir: Path, lens=DEFAULT_LENS, manifests: dict | None = None,
        namespace: str = NAMESPACE, log=print) -> dict:
    manifests = I.load_manifests() if manifests is None else manifests
    spec = preflight(namespace, manifests, report_dir)
    train_days, val_days, test_days = spec["split_days"]
    pooled = {(p, t, s): [] for p in spec["profiles"] for t in TASKS for s in ("train", "val")}
    datasets = []
    for profile in spec["profiles"]:
        for seed in spec["seeds"]:
            base = data_root / namespace / profile / f"seed{seed}"
            src, proc = base / "synthetic", base / "processed"
            write_dataset(src, spec["customers"], spec["days"], seed, profile)
            errors = validate(src)
            if errors:
                raise RuntimeError(f"{profile}/seed{seed} failed validation: {errors[:3]}")
            prep = build(src, proc, val_days, test_days, min_history=1)
            inter, meta, candidates = load_interactions(proc)
            c = meta["cutoffs"]
            cutoffs = compute_cutoffs(c["window_start_utc"], c["window_end_utc"], val_days, test_days)
            seq = build_seq_cases(no_test_events(inter), cutoffs, candidates, ("train", "val"), max(lens))
            entry = {"profile": profile, "seed": seed, "seed_namespace": namespace,
                     "source_composite_sha256": I.composite_fingerprint(prep["source_files_sha256"]),
                     "processed_composite_sha256": I.processed_fingerprints(proc)["composite"],
                     "sequence_fingerprints": {f"{t}/{s}": v.fingerprint() for (t, s), v in seq.items()},
                     "per_task": {}}
            for (task, split), v in seq.items():
                pooled[(profile, task, split)].append(v.full_length)
                entry["per_task"][f"{task}/{split}"] = length_stats(v.full_length, lens)
            datasets.append(entry)
            log(f"  {profile} seed {seed}: done")
    groups = {f"{p}/{t}/{s}": length_stats(np.concatenate(v), lens) for (p, t, s), v in pooled.items()}
    for s in ("train", "val"):
        allx = np.concatenate([np.concatenate(v) for (p, t, sp), v in pooled.items() if sp == s])
        groups[f"all/all/{s}"] = length_stats(allx, lens)
    report = {
        "label": SYNTHETIC_LABEL,
        "namespace": namespace,
        "seeds": spec["seeds"],
        "profiles": spec["profiles"],
        "seq_data_version": SEQ_DATA_VERSION,
        "unit": "history steps (distinct first-target timestamps, same task, strictly before the target)",
        "decision_basis": "train-window cases only",
        "val_role": "sanity check of the chosen truncation only",
        "test_role": "test-window events dropped before building sequences; no test statistics computed",
        "candidate_lens": list(lens),
        "groups": groups,
        "datasets": datasets,
    }
    return report


def to_markdown(r: dict) -> str:
    lens = r["candidate_lens"]
    lines = [f"# Phase 3 history-length report ({r['namespace']})", "", f"> **{r['label']}**", "",
             f"- Seeds {r['seeds']}; profiles {', '.join(r['profiles'])}.",
             f"- Unit: {r['unit']}.",
             f"- Decision basis: {r['decision_basis']}. Validation: {r['val_role']}. Test: {r['test_role']}.", "",
             "| Group (profile/task/split) | cases | P50 | P75 | P90 | P95 | max | mean | "
             + " | ".join(f">{L}" for L in lens) + " | " + " | ".join(f"steps cut @{L}" for L in lens) + " |",
             "|---|---|---|---|---|---|---|---|" + "---|" * (2 * len(lens))]
    for name, s in r["groups"].items():
        lines.append(f"| {name} | {s['n_cases']} | {s['p50']:.0f} | {s['p75']:.0f} | {s['p90']:.0f} | {s['p95']:.0f} "
                     f"| {s['max']} | {s['mean']:.1f} | "
                     + " | ".join(f"{100 * s['share_exceeding'][str(L)]:.1f}%" for L in lens) + " | "
                     + " | ".join(f"{100 * s['steps_truncated'][str(L)]:.1f}%" for L in lens) + " |")
    lines += ["", "'>L' = share of cases whose history exceeds L steps (case truncation rate at max_seq_len=L). "
              "'steps cut @L' = share of all history steps dropped at L (most recent steps are kept).", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--data-root", type=Path, default=Path("data/phase3"))
    parser.add_argument("--report-dir", type=Path, default=Path("artifacts/phase3/seq_lengths"))
    parser.add_argument("--lens", type=int, nargs="+", default=list(DEFAULT_LENS))
    args = parser.parse_args(argv)
    try:
        report = run(args.data_root, args.report_dir, tuple(args.lens), namespace=args.namespace)
    except I.LockedTestSplitError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "seq_length_report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                             encoding="utf-8", newline="\n")
    md = to_markdown(report)
    (args.report_dir / "seq_length_report.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
