"""Generate a reproducible synthetic mobile-banking behavioral dataset.

All entities, behaviors, and rules are fictional. The behavior process is a simple
session/journey simulator designed to produce learnable, internally consistent
sequences; it does not model any real institution or observed customer behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bri.data import schema as S

GENERATOR_VERSION = "0.2.0"
WINDOW_END = datetime(2025, 1, 1, tzinfo=timezone.utc)  # fixed so output is reproducible
LOCAL_UTC_OFFSET_HOURS = 7  # fictional app operates in UTC+7; storage stays UTC
MAX_CUSTOMERS = 200_000
MAX_DAYS = 730

# (item_id, item_type, category, item_name, description)
CATALOG = [
    ("SVC_TRANSFER", "service", "payments", "Transfer", "Generic transfer service information"),
    ("SVC_BILLPAY", "service", "payments", "Bill payment", "Generic bill payment service information"),
    ("SVC_QR", "service", "payments", "QR payment", "Generic QR payment service information"),
    ("SVC_TOPUP", "service", "payments", "Mobile top-up", "Generic mobile top-up service information"),
    ("SVC_SCHEDULED", "service", "payments", "Scheduled payment", "Generic scheduled payment service information"),
    ("PRD_SAVINGS", "product", "savings", "Savings information", "Generic savings product information"),
    ("PRD_DEPOSIT", "product", "savings", "Time deposit information", "Generic time deposit information"),
    ("PRD_DEBIT", "product", "cards", "Debit card information", "Generic debit card information"),
    ("PRD_CREDIT", "product", "cards", "Credit card information", "Generic credit card information"),
    ("PRD_HOMEFIN", "product", "financing", "Home financing information", "Generic home financing information"),
    ("PRD_PERSONALFIN", "product", "financing", "Personal financing information", "Generic personal financing information"),
]
SERVICE_IDS = [c[0] for c in CATALOG if c[1] == "service"]
PRODUCT_IDS = [c[0] for c in CATALOG if c[1] == "product"]
CALCULATOR_CATEGORIES = {"savings", "financing"}
ITEM_CATEGORY = {c[0]: c[2] for c in CATALOG}

# --- Fictional behavior parameters (synthetic; not business rules) -------------
SEGMENT_P = [0.35, 0.30, 0.20, 0.15]
TENURE_DAYS = {"under_6m": (14, 180), "6_to_24m": (180, 730), "over_24m": (730, 2555)}
SESSIONS_PER_DAY = {"low": 0.10, "medium": 0.35, "high": 0.80}
# Probability that a session journey is service-oriented vs product-oriented.
SERVICE_JOURNEY_P = {"everyday": 0.80, "digital_active": 0.75, "emerging": 0.70, "established": 0.55}
# Segment-level tilt over PRODUCT_IDS (multiplied into the customer's Dirichlet affinity).
PRODUCT_TILT = {
    "everyday": [3, 1, 2, 1, 1, 1],
    "digital_active": [2, 1, 2, 2, 1, 1],
    "emerging": [3, 1, 2, 1, 1, 2],
    "established": [1, 3, 1, 2, 2, 1],
}
# Sequential signal: next service in the same session tends to follow this map.
SERVICE_TRANSITION = {
    "SVC_TRANSFER": "SVC_BILLPAY",
    "SVC_BILLPAY": "SVC_TOPUP",
    "SVC_QR": "SVC_TOPUP",
    "SVC_TOPUP": "SVC_QR",
    "SVC_SCHEDULED": "SVC_TRANSFER",
}
# Diurnal profile in local time (hours 0..23).
HOUR_WEIGHTS = np.array([1, 0.5, 0.3, 0.3, 0.5, 1.5, 3, 5, 6, 6, 5, 5,
                         6, 5, 4, 4, 5, 6, 7, 8, 8, 6, 4, 2], dtype=float)
HOUR_P = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()
SESSION_GAP = timedelta(minutes=30)


def fmt_ts(ts: datetime) -> str:
    return ts.strftime(S.TIMESTAMP_FORMAT)


def build_catalog() -> pd.DataFrame:
    df = pd.DataFrame(CATALOG, columns=S.columns("catalog.csv")[:-1])
    df["active_status"] = True
    return df[S.columns("catalog.csv")]


class _Session:
    """Accumulates the events of one session with monotonically increasing time."""

    def __init__(self, rng: np.random.Generator, start: datetime):
        self.rng = rng
        self.t = start
        self.events: list[tuple[datetime, str, str | None]] = []

    def add(self, event_type: str, item_id: str | None = None) -> None:
        if self.events:
            self.t += timedelta(seconds=int(3 + self.rng.exponential(25)))
        self.events.append((self.t, event_type, item_id))


def _service_journey(s: _Session, rng: np.random.Generator, item: str) -> None:
    if rng.random() < 0.3:
        s.add("menu_view")
    s.add("service_view", item)
    if rng.random() < 0.75:
        s.add("service_start", item)
        s.add("service_complete" if rng.random() < 0.85 else "service_cancel", item)


def _product_journey(s: _Session, rng: np.random.Generator, item: str) -> None:
    if rng.random() < 0.25:
        s.add("search_open")
        s.add("search_result_view")
    s.add("product_view", item)
    if rng.random() < 0.55:
        s.add("product_detail_view", item)
        if ITEM_CATEGORY[item] in CALCULATOR_CATEGORIES and rng.random() < 0.4:
            s.add("product_calculator_use", item)
        if rng.random() < 0.10:
            s.add("product_application_start", item)
            if rng.random() < 0.5:
                s.add("product_application_complete", item)


def _customer_events(
    rng: np.random.Generator,
    segment: str,
    engagement: str,
    window_start: datetime,
    days: int,
) -> list[list[tuple[datetime, str, str | None]]]:
    """Return a list of sessions; each session is a list of (ts, event_type, item_id)."""
    service_aff = rng.dirichlet(np.full(len(SERVICE_IDS), 0.6))
    product_aff = rng.dirichlet(np.full(len(PRODUCT_IDS), 0.6)) * PRODUCT_TILT[segment]
    product_aff /= product_aff.sum()
    p_service = SERVICE_JOURNEY_P[segment]

    n_sessions = int(rng.poisson(SESSIONS_PER_DAY[engagement] * days))
    day_idx = np.sort(rng.integers(0, days, size=n_sessions))
    hours = rng.choice(24, size=n_sessions, p=HOUR_P)
    minutes = rng.integers(0, 60 * 60, size=n_sessions)  # seconds within the hour

    sessions = []
    window_end = window_start + timedelta(days=days)
    prev_end: datetime | None = None
    starts = sorted(
        window_start
        + timedelta(days=int(d), hours=int(h) - LOCAL_UTC_OFFSET_HOURS, seconds=int(m))
        for d, h, m in zip(day_idx, hours, minutes)
    )
    for start in starts:
        start = max(start, window_start)
        if prev_end is not None and start < prev_end + SESSION_GAP:
            start = prev_end + SESSION_GAP
        s = _Session(rng, start)
        s.add("app_open")
        s.add("home_view")
        last_service: str | None = None
        for _ in range(int(rng.integers(1, 4))):  # 1-3 intents per session
            if rng.random() < p_service:
                if last_service and rng.random() < 0.4:
                    item = SERVICE_TRANSITION[last_service]
                else:
                    item = SERVICE_IDS[int(rng.choice(len(SERVICE_IDS), p=service_aff))]
                _service_journey(s, rng, item)
                last_service = item
            else:
                item = PRODUCT_IDS[int(rng.choice(len(PRODUCT_IDS), p=product_aff))]
                _product_journey(s, rng, item)
        if s.t >= window_end:
            break  # session would spill past the window; drop it and later ones
        sessions.append(s.events)
        prev_end = s.t
    return sessions


def generate_dataset(
    customers: int, days: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (customers, catalog, consent_preferences, events) DataFrames."""
    if not 1 <= customers <= MAX_CUSTOMERS:
        raise ValueError(f"customers must be between 1 and {MAX_CUSTOMERS}")
    if not 1 <= days <= MAX_DAYS:
        raise ValueError(f"days must be between 1 and {MAX_DAYS}")

    window_start = WINDOW_END - timedelta(days=days)
    # One independent random stream per customer: customer i's data does not depend
    # on how many customers are generated.
    child_seeds = np.random.SeedSequence(seed).spawn(customers)

    cust_cols: dict[str, list] = {c: [] for c in S.columns("customers.csv")}
    cons_cols: dict[str, list] = {c: [] for c in S.columns("consent_preferences.csv")}
    ev: dict[str, list] = {c: [] for c in S.columns("events.csv") if c != "event_id"}

    for i, child in enumerate(child_seeds):
        rng = np.random.default_rng(child)
        token = f"SYN_CUST_{i + 1:06d}"
        segment = S.CUSTOMER_SEGMENTS[int(rng.choice(4, p=SEGMENT_P))]
        tenure = S.TENURE_BUCKETS[int(rng.integers(3))]
        engagement = S.ENGAGEMENT_SEGMENTS[int(rng.choice(3, p=[0.25, 0.5, 0.25]))]
        lo, hi = TENURE_DAYS[tenure]
        # Tenure is measured at window start, so created_at always precedes events.
        created_at = window_start - timedelta(
            days=int(rng.integers(lo, hi)), seconds=int(rng.integers(0, 86_400))
        )
        cust_cols["customer_token"].append(token)
        cust_cols["customer_segment"].append(segment)
        cust_cols["tenure_bucket"].append(tenure)
        cust_cols["engagement_segment"].append(engagement)
        cust_cols["preferred_language"].append("id" if rng.random() < 0.9 else "en")
        cust_cols["created_at"].append(fmt_ts(created_at))

        # Consent captured between onboarding and window start (one row per purpose).
        span = int((window_start - created_at).total_seconds())
        for purpose, p_allow in ((S.SERVICE_PURPOSE, 0.96), (S.MARKETING_PURPOSE, 0.72)):
            captured = created_at + timedelta(seconds=int(rng.integers(0, max(span, 1))))
            cons_cols["customer_token"].append(token)
            cons_cols["purpose"].append(purpose)
            cons_cols["allowed"].append(bool(rng.random() < p_allow))
            cons_cols["captured_at"].append(fmt_ts(captured))
            cons_cols["policy_version"].append("SYN_POLICY_1")

        sessions = _customer_events(rng, segment, engagement, window_start, days)
        for s_num, events in enumerate(sessions, start=1):
            session_id = f"SYN_SESS_{i + 1:06d}_{s_num:04d}"
            for ts, event_type, item_id in events:
                surface = S.EVENT_SURFACE[event_type]
                ev["customer_token"].append(token)
                ev["session_id"].append(session_id)
                ev["event_timestamp"].append(fmt_ts(ts))
                ev["event_type"].append(event_type)
                ev["surface"].append(surface)
                ev["item_id"].append(item_id)
                ev["purpose"].append(S.SURFACE_PURPOSE[surface])
                ev["channel"].append("mobile_app")
                ev["event_value"].append(int(rng.choice(3, p=[0.5, 0.35, 0.15])) + 1)

    events_df = pd.DataFrame(ev)
    # Rows are already grouped by customer and in time order; IDs follow that order.
    events_df.insert(0, "event_id", [f"SYN_EVT_{k + 1:09d}" for k in range(len(events_df))])
    events_df = events_df[S.columns("events.csv")]
    return (
        pd.DataFrame(cust_cols),
        build_catalog(),
        pd.DataFrame(cons_cols),
        events_df,
    )


def write_dataset(
    output_dir: Path, customers: int, days: int, seed: int
) -> dict:
    """Generate and write all files. Returns the metadata dict."""
    tables = dict(zip(S.TABLES, generate_dataset(customers, days, seed)))
    output_dir.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for filename, df in tables.items():
        path = output_dir / filename
        # Fixed "\n" line endings so output is byte-identical on Windows and Linux.
        df.to_csv(path, index=False, lineterminator="\n")
        checksums[filename] = hashlib.sha256(path.read_bytes()).hexdigest()

    window_start = WINDOW_END - timedelta(days=days)
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "schema_version": S.SCHEMA_VERSION,
        "synthetic_only": True,
        "seed": seed,
        "customers": customers,
        "days": days,
        "window_start_utc": fmt_ts(window_start),
        "window_end_utc": fmt_ts(WINDOW_END),
        "row_counts": {name: len(df) for name, df in tables.items()},
        "sha256": checksums,
    }
    (output_dir / S.METADATA_FILE).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--customers", type=int, default=500, help="number of synthetic customers")
    parser.add_argument("--days", type=int, default=30, help="length of the event window in days")
    parser.add_argument("--seed", type=int, default=42, help="random seed (same seed = same data)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    try:
        metadata = write_dataset(args.output_dir, args.customers, args.days, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Synthetic dataset written to {args.output_dir}")
    for name, count in metadata["row_counts"].items():
        print(f"  {name:<26} {count:>10,} rows")
    print(f"  window: {metadata['window_start_utc']} -> {metadata['window_end_utc']} (seed={args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
