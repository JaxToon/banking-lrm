"""Consent-aware event filtering (fail closed).

An event is kept only when ALL of the following hold:
1. Its purpose is a known processing purpose.
2. Exactly one consent record exists for (customer_token, purpose).
3. That record's `allowed` value is exactly True.
4. The record has a valid `captured_at` and the event is at or after it.

Anything else (missing, unknown, ambiguous, denied, malformed, or earlier than the
recorded preference) excludes the event. Absence of a "no" is never treated as "yes".

IMPORTANT: the Phase 1 consent table is a *snapshot* (one row per customer and purpose).
It is not historical consent evidence: it cannot show withdrawals or the preference
state at the time of each event. Rule 4 is a conservative approximation, not a
substitute for a real consent history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bri.tasks import PURPOSE_TO_TASK

# Ordered by precedence: the first matching reason is reported.
EXCLUSION_REASONS = (
    "unknown_purpose",
    "no_consent_record",
    "ambiguous_consent",
    "consent_not_granted",
    "event_before_consent_capture",
)
_KEY = ["customer_token", "purpose"]


def _consent_index(consent: pd.DataFrame) -> pd.DataFrame:
    """One row per (customer, purpose) with n_records, granted flag, and captured_at."""
    c = consent[_KEY].copy()
    # Only the literal boolean True (or its CSV form "True") grants permission.
    c["granted"] = consent["allowed"].astype(str) == "True"
    c["captured_at"] = pd.to_datetime(consent["captured_at"], utc=True, errors="coerce", format="ISO8601")
    return c.groupby(_KEY, sort=False).agg(
        n_records=("granted", "size"),
        granted=("granted", "all"),
        captured_at=("captured_at", "max"),
        captured_missing=("captured_at", lambda s: s.isna().any()),
    ).reset_index()


def filter_by_consent(events: pd.DataFrame, consent: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (permitted events, report). `events.event_timestamp` must be UTC datetimes."""
    report = {"input_events": int(len(events)), "kept_events": 0,
              "excluded": {r: 0 for r in EXCLUSION_REASONS}}
    if events.empty:
        return events.copy(), report

    merged = events.merge(_consent_index(consent), on=_KEY, how="left", validate="many_to_one")
    reason = np.select(
        [
            ~merged["purpose"].isin(list(PURPOSE_TO_TASK)),
            merged["n_records"].isna(),
            (merged["n_records"] > 1) | merged["captured_missing"].eq(True),
            ~merged["granted"].eq(True),
            merged["event_timestamp"] < merged["captured_at"],
        ],
        list(EXCLUSION_REASONS),
        default="",
    )
    for r in EXCLUSION_REASONS:
        report["excluded"][r] = int((reason == r).sum())
    kept = events.loc[reason == ""].reset_index(drop=True)
    report["kept_events"] = int(len(kept))
    return kept, report
