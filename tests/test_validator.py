import json
import subprocess
import sys

import pandas as pd
import pytest

from bri.data import schema as S
from bri.data.validate_dataset import validate


def load(path, name):
    return pd.read_csv(path / name, dtype=str, keep_default_na=False)


def save(path, name, df):
    df.to_csv(path / name, index=False, lineterminator="\n")


def mutate(path, name, fn):
    df = fn(load(path, name))
    save(path, name, df)
    # keep metadata row counts in sync so only the intended error fires
    meta_path = path / S.METADATA_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["row_counts"][name] = len(df)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def first_index(df, **where):
    mask = pd.Series(True, index=df.index)
    for col, val in where.items():
        mask &= df[col] == val
    return df.index[mask][0]


def set_value(col, value, **where):
    def fn(df):
        df.loc[first_index(df, **where) if where else df.index[0], col] = value
        return df
    return fn


def test_accepts_generated_dataset(dataset_dir):
    assert validate(dataset_dir) == []


def _swap_terminal_before_start(df):
    i = first_index(df, event_type="service_complete")
    df.loc[i, "event_type"] = "service_view"
    j = df.index[(df.index < i) & (df["event_type"] == "service_start")][-1]
    df.loc[j, "event_type"] = "service_complete"
    df.loc[i, "event_type"] = "service_start"
    return df


def _drop_service_start(df):
    i = first_index(df, event_type="service_start")
    df.loc[i, "event_type"] = "service_view"
    return df


def _swap_timestamps(df):
    df.loc[[0, 1], "event_timestamp"] = df.loc[[1, 0], "event_timestamp"].to_numpy()
    df.loc[0, "event_timestamp"] = "2024-12-31T23:59:59Z"
    return df


def _move_session_to_other_customer(df):
    df.loc[0, "customer_token"] = df["customer_token"].iloc[-1]
    return df


CASES = [
    ("missing column", "events.csv", lambda df: df.drop(columns=["purpose"]), "missing columns"),
    ("unexpected column", "customers.csv", lambda df: df.assign(region="x"), "unexpected columns"),
    ("prohibited column", "customers.csv", lambda df: df.assign(phone_number="x"), "prohibited column"),
    ("prohibited column variant", "events.csv", lambda df: df.assign(transaction_amount="1"), "prohibited column"),
    ("invalid event type", "events.csv", set_value("event_type", "money_moved"), "invalid value(s) in 'event_type'"),
    ("invalid surface enum", "events.csv", set_value("surface", "web"), "invalid value(s) in 'surface'"),
    ("null customer", "events.csv", set_value("customer_token", ""), "null value"),
    ("real-looking id", "customers.csv", set_value("customer_token", "3174012345678901"), "invalid value(s) in 'customer_token'"),
    ("bad boolean", "consent_preferences.csv", set_value("allowed", "yes"), "invalid value(s) in 'allowed'"),
    ("event_value out of range", "events.csv", set_value("event_value", "250000"), "invalid value(s) in 'event_value'"),
    ("unparseable timestamp", "events.csv", set_value("event_timestamp", "yesterday"), "invalid value(s) in 'event_timestamp'"),
    ("naive timestamp", "events.csv", set_value("event_timestamp", "2024-12-20T10:00:00"), "invalid value(s) in 'event_timestamp'"),
    ("non-UTC offset", "customers.csv", set_value("created_at", "2024-01-01T10:00:00+07:00"), "invalid value(s) in 'created_at'"),
    ("duplicate event id", "events.csv", lambda df: df.assign(event_id=df["event_id"].where(df.index != 1, df["event_id"][0])), "duplicate event_id"),
    ("duplicate customer", "customers.csv", lambda df: pd.concat([df, df.head(1)]), "duplicate customer_token"),
    ("unknown customer fk", "events.csv", set_value("customer_token", "SYN_CUST_999999"), "unknown customer_token"),
    ("unknown consent customer", "consent_preferences.csv", set_value("customer_token", "SYN_CUST_999999"), "unknown customer_token"),
    ("unknown item fk", "events.csv", set_value("item_id", "SVC_UNKNOWN", event_type="service_view"), "unknown item_id"),
    ("surface mismatch", "events.csv", set_value("surface", "product", event_type="app_open"), "surface incompatible"),
    ("purpose mismatch", "events.csv", set_value("purpose", "service_personalization", event_type="product_view"), "consent-purpose mismatch"),
    ("item on navigation event", "events.csv", set_value("item_id", "SVC_QR", event_type="home_view"), "wrong item_type"),
    ("product item on service event", "events.csv", set_value("item_id", "PRD_SAVINGS", event_type="service_view"), "wrong item_type"),
    ("missing consent purpose", "consent_preferences.csv", lambda df: df.iloc[1:], "missing a purpose preference"),
    ("out of order", "events.csv", _swap_timestamps, "not ordered within customer"),
    ("event before created_at", "customers.csv", set_value("created_at", "2024-12-31T00:00:00Z"), "before customer created_at"),
    ("event outside window", "events.csv", set_value("event_timestamp", "2030-01-01T00:00:00Z"), "outside metadata window"),
    ("complete without start", "events.csv", _drop_service_start, "without earlier 'service_start'"),
    ("complete before start", "events.csv", _swap_terminal_before_start, "without earlier 'service_start'"),
    ("session shared across customers", "events.csv", _move_session_to_other_customer, "session_id shared"),
]


@pytest.mark.parametrize("label,table,fn,expected", CASES, ids=[c[0] for c in CASES])
def test_rejects_corruption(dataset_copy, label, table, fn, expected):
    mutate(dataset_copy, table, fn)
    errors = validate(dataset_copy)
    assert any(expected in e for e in errors), errors


@pytest.mark.parametrize("name", [*S.TABLES, S.METADATA_FILE])
def test_rejects_missing_file(dataset_copy, name):
    (dataset_copy / name).unlink()
    assert any(f"Missing file: {name}" in e for e in validate(dataset_copy))


def test_rejects_metadata_row_count_mismatch(dataset_copy):
    save(dataset_copy, "events.csv", load(dataset_copy, "events.csv").iloc[:-1])
    assert any("row count for events.csv" in e for e in validate(dataset_copy))


def test_rejects_non_synthetic_metadata(dataset_copy):
    path = dataset_copy / S.METADATA_FILE
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["synthetic_only"] = False
    path.write_text(json.dumps(meta), encoding="utf-8")
    assert any("synthetic_only" in e for e in validate(dataset_copy))


def test_missing_directory(tmp_path):
    assert validate(tmp_path / "nope") == [f"Data directory not found: {tmp_path / 'nope'}"]


def test_cli_exit_code_on_failure(dataset_copy):
    (dataset_copy / "events.csv").unlink()
    result = subprocess.run(
        [sys.executable, "-m", "bri.data.validate_dataset", "--data-dir", str(dataset_copy)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "FAILED" in result.stdout
