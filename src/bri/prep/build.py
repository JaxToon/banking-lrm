"""Build consent-aware, temporally split training examples and evaluation cases.

Usage:
    python -m bri.prep.build --data-dir data/synthetic --output-dir data/processed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from bri.data import schema as S
from bri.data.validate_dataset import validate
from bri.prep.consent import filter_by_consent
from bri.prep.examples import build_eval_cases, build_interactions, build_train_examples, candidate_sets
from bri.prep.split import compute_cutoffs
from bri.tasks import SYNTHETIC_LABEL, TASKS

PREP_VERSION = "0.2.0"  # 0.2.0 adds source fingerprints; processed CSVs unchanged
PREP_METADATA_FILE = "prep_metadata.json"


class InvalidSourceDataError(RuntimeError):
    pass


def load_dataset(data_dir: Path) -> dict:
    """Validate (fail closed) and load the Phase 1 dataset with parsed UTC timestamps."""
    errors = validate(data_dir)
    if errors:
        raise InvalidSourceDataError("source dataset failed validation:\n- " + "\n- ".join(errors))

    def read(name):
        return pd.read_csv(data_dir / name, dtype=str, keep_default_na=False)

    events = read("events.csv")
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"], utc=True, format="ISO8601")
    consent = read("consent_preferences.csv")
    consent["captured_at"] = pd.to_datetime(consent["captured_at"], utc=True, format="ISO8601")
    metadata = json.loads((data_dir / S.METADATA_FILE).read_text(encoding="utf-8"))
    return {"events": events, "consent": consent, "catalog": read("catalog.csv"), "metadata": metadata}


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime(S.TIMESTAMP_FORMAT)
    out.to_csv(path, index=False, lineterminator="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(data_dir: Path, output_dir: Path, val_days: int, test_days: int, min_history: int) -> dict:
    data = load_dataset(data_dir)
    meta = data["metadata"]
    # Hash the source files actually read (not the checksums claimed in their metadata).
    source_files = {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in S.TABLES}
    cutoffs = compute_cutoffs(meta["window_start_utc"], meta["window_end_utc"], val_days, test_days)
    candidates = candidate_sets(data["catalog"])

    permitted, consent_report = filter_by_consent(data["events"], data["consent"])
    interactions = build_interactions(permitted, cutoffs)
    examples, ex_insufficient = build_train_examples(interactions, min_history)
    val_cases, val_targets, val_stats = build_eval_cases(interactions, cutoffs, "val", candidates, min_history)
    test_cases, test_targets, test_stats = build_eval_cases(interactions, cutoffs, "test", candidates, min_history)
    cases = pd.concat([val_cases, test_cases], ignore_index=True)
    targets = pd.concat([val_targets, test_targets], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "interactions.csv": interactions,
        "train_examples.csv": examples,
        "eval_cases.csv": cases,
        "eval_targets.csv": targets,
    }
    checksums = {name: _write_csv(df, output_dir / name) for name, df in outputs.items()}

    def count(df, by):
        return {f"{k[0]}/{k[1]}" if isinstance(k, tuple) else k: int(v)
                for k, v in df.groupby(by).size().items()} if len(df) else {}

    prep_meta = {
        "label": SYNTHETIC_LABEL,
        "prep_version": PREP_VERSION,
        "source": {
            "generator_version": meta.get("generator_version"),
            "schema_version": meta.get("schema_version"),
            "seed": meta.get("seed"),
            "events_sha256": meta.get("sha256", {}).get("events.csv"),
        },
        "source_files_sha256": source_files,
        "source_generation": {
            "customers": meta.get("customers"),
            "days": meta.get("days"),
            "behavior_profile": meta.get("behavior_profile"),
        },
        "params": {"val_days": val_days, "test_days": test_days, "min_history": min_history},
        "cutoffs": cutoffs.to_dict(),
        "tasks": {n: {"purpose": t.purpose, "item_type": t.item_type,
                      "target_event_types": sorted(t.target_event_types),
                      "candidates": candidates[n]} for n, t in TASKS.items()},
        "consent_filter": consent_report,
        "consent_note": "Consent table is a snapshot, not historical consent evidence.",
        "counts": {
            "interactions_by_task_split": count(interactions, ["task", "split"]),
            "train_examples_by_task": count(examples, "task"),
            "train_targets_insufficient_history": ex_insufficient,
            "eval_cases_by_split_task": count(cases, ["split", "task"]),
            "eval_stats": {"val": val_stats, "test": test_stats},
        },
        "sha256": checksums,
    }
    (output_dir / PREP_METADATA_FILE).write_text(
        json.dumps(prep_meta, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return prep_meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--min-history", type=int, default=1,
                        help="minimum same-task interactions before a target/cutoff")
    args = parser.parse_args(argv)
    try:
        meta = build(args.data_dir, args.output_dir, args.val_days, args.test_days, args.min_history)
    except InvalidSourceDataError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        parser.error(str(exc))
    print(SYNTHETIC_LABEL)
    print(f"Processed data written to {args.output_dir}")
    print(f"  cutoffs: {json.dumps(meta['cutoffs'])}")
    c = meta["consent_filter"]
    print(f"  consent filter: kept {c['kept_events']:,} of {c['input_events']:,} events; excluded {c['excluded']}")
    for key, value in meta["counts"].items():
        if key != "eval_stats":
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
