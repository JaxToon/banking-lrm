"""Validate a generated synthetic dataset: schema, integrity, ordering, privacy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from bri.data import schema as S

MAX_EXAMPLES = 3


def _examples(values) -> str:
    vals = list(dict.fromkeys(str(v) for v in values))
    more = f" (+{len(vals) - MAX_EXAMPLES} more)" if len(vals) > MAX_EXAMPLES else ""
    return ", ".join(vals[:MAX_EXAMPLES]) + more


def _prohibited(column: str) -> list[str]:
    col = column.lower()
    tokens = set(col.split("_"))
    return [k for k in S.PROHIBITED_COLUMN_KEYWORDS if (k in col if "_" in k else k in tokens)]


def _parse_utc(series: pd.Series) -> pd.Series:
    """Parse timestamps that explicitly declare UTC ("Z" or "+00:00"); others -> NaT."""
    explicit_utc = series.str.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|\+00:00)")
    parsed = pd.to_datetime(series.where(explicit_utc), utc=True, errors="coerce", format="ISO8601")
    return parsed


def _check_columns(name: str, df: pd.DataFrame, errors: list[str]) -> dict[str, pd.Series]:
    """Type/enum/null checks. Returns parsed timestamp columns."""
    parsed: dict[str, pd.Series] = {}
    for col in S.TABLES[name]:
        s = df[col.name]
        empty = s == ""
        if not col.nullable and empty.any():
            errors.append(f"{name}: {empty.sum()} null value(s) in non-nullable column '{col.name}'")
        present = s[~empty]
        bad = pd.Series(False, index=present.index)
        if col.kind == "id":
            bad = ~present.str.fullmatch(col.pattern)
        elif col.kind == "enum":
            bad = ~present.isin(col.values)
        elif col.kind == "bool":
            bad = ~present.isin(["True", "False"])
        elif col.kind == "int":
            bad = ~present.isin([str(v) for v in col.values])
        elif col.kind == "timestamp":
            ts = _parse_utc(present)
            bad = ts.isna()
            parsed[col.name] = ts
        if bad.any():
            errors.append(
                f"{name}: {bad.sum()} invalid value(s) in '{col.name}' "
                f"[{col.kind}] e.g. {_examples(present[bad])}"
            )
    return parsed


def _check_metadata(data_dir: Path, frames: dict[str, pd.DataFrame], errors: list[str]) -> dict | None:
    path = data_dir / S.METADATA_FILE
    if not path.exists():
        errors.append(f"Missing file: {S.METADATA_FILE}")
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{S.METADATA_FILE}: invalid JSON ({exc})")
        return None
    if meta.get("synthetic_only") is not True:
        errors.append(f"{S.METADATA_FILE}: synthetic_only must be true")
    for name, df in frames.items():
        expected = meta.get("row_counts", {}).get(name)
        if expected != len(df):
            errors.append(f"{S.METADATA_FILE}: row count for {name} is {expected}, file has {len(df)}")
    return meta


def validate(data_dir: Path) -> list[str]:
    """Return a list of human-readable errors; empty list means the dataset is valid."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return [f"Data directory not found: {data_dir}"]

    errors: list[str] = []
    frames: dict[str, pd.DataFrame] = {}

    # 1. Structure: files exist, exact column allowlist, prohibited-name scan.
    for name in S.TABLES:
        path = data_dir / name
        if not path.exists():
            errors.append(f"Missing file: {name}")
            continue
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            errors.append(f"{name}: cannot be parsed ({exc})")
            continue
        expected = S.columns(name)
        missing = [c for c in expected if c not in df.columns]
        unexpected = [c for c in df.columns if c not in expected]
        if missing:
            errors.append(f"{name}: missing columns {missing}")
        if unexpected:
            errors.append(f"{name}: unexpected columns {unexpected} (schema is an allowlist)")
        for c in df.columns:
            if hits := _prohibited(c):
                errors.append(f"{name}: prohibited column '{c}' (matches {hits})")
        frames[name] = df
    if errors:
        return errors  # later checks assume the structure is correct

    customers = frames["customers.csv"]
    catalog = frames["catalog.csv"]
    consent = frames["consent_preferences.csv"]
    events = frames["events.csv"]

    # 2. Column types, enums, nulls, ID formats, UTC timestamps.
    ts: dict[str, pd.Series] = {}
    for name, df in frames.items():
        for col, parsed in _check_columns(name, df, errors).items():
            ts[f"{name}:{col}"] = parsed
    meta = _check_metadata(data_dir, frames, errors)
    if errors:
        return errors  # semantic checks assume clean types

    # 3. Uniqueness.
    for name, key in (
        ("customers.csv", ["customer_token"]),
        ("catalog.csv", ["item_id"]),
        ("events.csv", ["event_id"]),
        ("consent_preferences.csv", ["customer_token", "purpose"]),
    ):
        dup = frames[name].duplicated(key, keep=False)
        if dup.any():
            sample = frames[name].loc[dup, key].astype(str).agg("/".join, axis=1)
            errors.append(f"{name}: duplicate {'/'.join(key)}: {_examples(sample)}")

    # 4. Referential integrity.
    customer_ids = set(customers["customer_token"])
    for name, df in (("events.csv", events), ("consent_preferences.csv", consent)):
        unknown = ~df["customer_token"].isin(customer_ids)
        if unknown.any():
            errors.append(f"{name}: unknown customer_token: {_examples(df.loc[unknown, 'customer_token'])}")
    item_type = dict(zip(catalog["item_id"], catalog["item_type"]))
    has_item = events["item_id"] != ""
    unknown_item = has_item & ~events["item_id"].isin(item_type)
    if unknown_item.any():
        errors.append(f"events.csv: unknown item_id: {_examples(events.loc[unknown_item, 'item_id'])}")
    session_owners = events.groupby("session_id")["customer_token"].nunique()
    if (session_owners > 1).any():
        errors.append(f"events.csv: session_id shared across customers: {_examples(session_owners[session_owners > 1].index)}")

    # 5. Event semantics: surface, purpose, and item type follow event_type.
    expected_surface = events["event_type"].map(S.EVENT_SURFACE)
    bad = events["surface"] != expected_surface
    if bad.any():
        errors.append(f"events.csv: surface incompatible with event_type: {_examples(events.loc[bad, 'event_id'])}")
    bad = events["purpose"] != expected_surface.map(S.SURFACE_PURPOSE)
    if bad.any():
        errors.append(
            f"events.csv: purpose inconsistent with event_type (consent-purpose mismatch): "
            f"{_examples(events.loc[bad, 'event_id'])}"
        )
    required_type = expected_surface.map(S.SURFACE_ITEM_TYPE)
    actual_type = events["item_id"].map(item_type)
    bad = (required_type.isna() & has_item) | (required_type.notna() & (actual_type != required_type))
    if bad.any():
        errors.append(f"events.csv: item_id missing or wrong item_type for surface: {_examples(events.loc[bad, 'event_id'])}")

    # 6. Consent coverage: every customer has exactly one row per purpose (fail closed).
    coverage = consent.groupby("customer_token")["purpose"].nunique()
    incomplete = customer_ids - set(coverage[coverage == len(S.PURPOSES)].index)
    if incomplete:
        errors.append(f"consent_preferences.csv: customers missing a purpose preference: {_examples(sorted(incomplete))}")

    # 7. Temporal rules.
    ev_ts = ts["events.csv:event_timestamp"]
    created = (
        customers.drop_duplicates("customer_token")  # duplicates already reported above
        .set_index("customer_token")["created_at"].pipe(_parse_utc)
    )
    before_created = ev_ts < events["customer_token"].map(created)
    if before_created.any():
        errors.append(f"events.csv: events before customer created_at: {_examples(events.loc[before_created, 'event_id'])}")
    cons_ts = ts["consent_preferences.csv:captured_at"]
    bad = cons_ts < consent["customer_token"].map(created)
    if bad.any():
        errors.append(f"consent_preferences.csv: captured_at before created_at: {_examples(consent.loc[bad, 'customer_token'])}")
    if meta and {"window_start_utc", "window_end_utc"} <= meta.keys():
        lo, hi = pd.Timestamp(meta["window_start_utc"]), pd.Timestamp(meta["window_end_utc"])
        outside = (ev_ts < lo) | (ev_ts >= hi)
        if outside.any():
            errors.append(f"events.csv: events outside metadata window: {_examples(events.loc[outside, 'event_id'])}")
    # Stored order must be grouped by customer and non-decreasing in time.
    same_customer = events["customer_token"].eq(events["customer_token"].shift())
    went_back = same_customer & (ev_ts < ev_ts.shift())
    if went_back.any():
        errors.append(f"events.csv: timestamps not ordered within customer: {_examples(events.loc[went_back, 'event_id'])}")
    regrouped = ~same_customer & events["customer_token"].duplicated()
    if regrouped.any():
        errors.append(f"events.csv: rows for a customer are not contiguous: {_examples(events.loc[regrouped, 'customer_token'])}")
    # Funnel: terminal events need an earlier start for the same session and item.
    pos = pd.Series(range(len(events)), index=events.index)
    for terminal, start in S.FUNNEL_PREREQUISITES.items():
        starts = events[events["event_type"] == start]
        first_start = pos[starts.index].groupby([starts["session_id"], starts["item_id"]]).min()
        terms = events[events["event_type"] == terminal]
        if terms.empty:
            continue
        keys = pd.MultiIndex.from_arrays([terms["session_id"], terms["item_id"]])
        prior = pd.Series(first_start.reindex(keys).to_numpy(), index=terms.index)
        orphan = prior.isna() | (prior > pos[terms.index])
        if orphan.any():
            errors.append(f"events.csv: '{terminal}' without earlier '{start}' in session: {_examples(terms.loc[orphan, 'event_id'])}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args(argv)
    errors = validate(args.data_dir)
    if errors:
        print(f"DATASET VALIDATION FAILED ({len(errors)} issue(s))")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"DATASET VALIDATION PASSED ({args.data_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
