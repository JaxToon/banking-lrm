"""Small hand-built frames for Phase 2 unit tests."""
import pandas as pd

from bri.data import schema as S

CATALOG = pd.DataFrame({
    "item_id": ["SVC_A", "SVC_B", "SVC_C", "PRD_X", "PRD_Y", "PRD_Z"],
    "item_type": ["service"] * 3 + ["product"] * 3,
    "category": ["payments"] * 3 + ["savings"] * 3,
    "item_name": list("abcxyz"),
    "description": list("abcxyz"),
    "active_status": ["True"] * 6,
})


def events(rows, purpose_override=None) -> pd.DataFrame:
    """rows: (customer, session, 'YYYY-MM-DDTHH:MM:SSZ', event_type, item_id or '')."""
    out = []
    for n, (cust, sess, ts, et, item) in enumerate(rows, start=1):
        surface = S.EVENT_SURFACE[et]
        out.append({
            "event_id": f"SYN_EVT_{n:09d}", "customer_token": cust, "session_id": sess,
            "event_timestamp": ts, "event_type": et, "surface": surface, "item_id": item,
            "purpose": purpose_override or S.SURFACE_PURPOSE[surface], "channel": "mobile_app",
            "event_value": "1",
        })
    df = pd.DataFrame(out, columns=S.columns("events.csv"))
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    return df


def consent(rows) -> pd.DataFrame:
    """rows: (customer, purpose, allowed, captured_at)."""
    df = pd.DataFrame(rows, columns=["customer_token", "purpose", "allowed", "captured_at"])
    df["policy_version"] = "SYN_POLICY_1"
    return df


def allow_all(customers, captured="2024-01-01T00:00:00Z") -> pd.DataFrame:
    return consent([(c, p, True, captured) for c in customers for p in S.PURPOSES])
