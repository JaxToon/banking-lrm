import json
import subprocess
import sys

import pandas as pd
import pytest

from bri.data import schema as S
from bri.data.generate_synthetic import generate_dataset, write_dataset


def read(path, name):
    return pd.read_csv(path / name, dtype=str, keep_default_na=False)


def test_expected_files_and_exact_columns(dataset_dir):
    for name in S.TABLES:
        assert list(read(dataset_dir, name).columns) == S.columns(name)
    meta = json.loads((dataset_dir / S.METADATA_FILE).read_text(encoding="utf-8"))
    assert meta["synthetic_only"] is True
    assert meta["seed"] == 42


def test_same_seed_is_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_dataset(a, 25, 7, seed=123)
    write_dataset(b, 25, 7, seed=123)
    for name in [*S.TABLES, S.METADATA_FILE]:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_different_seed_differs():
    ev_a = generate_dataset(25, 7, seed=1)[3]
    ev_b = generate_dataset(25, 7, seed=2)[3]
    assert not ev_a.equals(ev_b)


def test_customer_streams_are_independent_of_population_size():
    small = generate_dataset(10, 7, seed=5)
    large = generate_dataset(20, 7, seed=5)
    pd.testing.assert_frame_equal(small[0], large[0].head(10))
    ev_small = small[3]
    ev_large = large[3][large[3]["customer_token"].isin(small[0]["customer_token"])]
    pd.testing.assert_frame_equal(ev_small, ev_large.reset_index(drop=True))


def test_all_behaviour_surfaces_present(dataset_dir):
    events = read(dataset_dir, "events.csv")
    assert set(events["surface"]) == set(S.SURFACES)
    assert {"service_complete", "product_detail_view", "app_open"} <= set(events["event_type"])


def test_sessions_start_with_app_open(dataset_dir):
    events = read(dataset_dir, "events.csv")
    first = events.groupby("session_id", sort=False)["event_type"].first()
    assert (first == "app_open").all()


def test_customer_timeline_is_consistent(dataset_dir):
    customers = read(dataset_dir, "customers.csv").set_index("customer_token")
    events = read(dataset_dir, "events.csv")
    consent = read(dataset_dir, "consent_preferences.csv")
    created = pd.to_datetime(customers["created_at"], utc=True)
    meta = json.loads((dataset_dir / S.METADATA_FILE).read_text(encoding="utf-8"))
    window_start = pd.Timestamp(meta["window_start_utc"])
    assert (created < window_start).all()
    assert (pd.to_datetime(consent["captured_at"], utc=True) <= window_start).all()
    ev_ts = pd.to_datetime(events["event_timestamp"], utc=True)
    assert (ev_ts >= window_start).all()


def test_tenure_bucket_matches_created_at(dataset_dir):
    customers = read(dataset_dir, "customers.csv")
    meta = json.loads((dataset_dir / S.METADATA_FILE).read_text(encoding="utf-8"))
    age_days = (pd.Timestamp(meta["window_start_utc"]) - pd.to_datetime(customers["created_at"], utc=True)).dt.days
    limits = {"under_6m": (0, 180), "6_to_24m": (180, 730), "over_24m": (730, 10_000)}
    for bucket, age in zip(customers["tenure_bucket"], age_days):
        lo, hi = limits[bucket]
        assert lo <= age <= hi, (bucket, age)


def test_no_identifier_looks_real(dataset_dir):
    """Every key column must use an obviously synthetic prefix."""
    events = read(dataset_dir, "events.csv")
    assert events["customer_token"].str.startswith("SYN_CUST_").all()
    assert events["event_id"].str.startswith("SYN_EVT_").all()
    assert events["session_id"].str.startswith("SYN_SESS_").all()


def test_no_prohibited_columns_generated():
    from bri.data.validate_dataset import _prohibited

    for df in generate_dataset(5, 3, 9):
        assert not any(_prohibited(c) for c in df.columns)


@pytest.mark.parametrize("kwargs", [{"customers": 0, "days": 7}, {"customers": 5, "days": 0}])
def test_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        generate_dataset(seed=1, **kwargs)


def test_cli_generate_then_validate(tmp_path):
    out = tmp_path / "cli"
    gen = subprocess.run(
        [sys.executable, "-m", "bri.data.generate_synthetic", "--customers", "15", "--days", "5",
         "--seed", "7", "--output-dir", str(out)],
        capture_output=True, text=True,
    )
    assert gen.returncode == 0, gen.stderr
    val = subprocess.run(
        [sys.executable, "-m", "bri.data.validate_dataset", "--data-dir", str(out)],
        capture_output=True, text=True,
    )
    assert val.returncode == 0, val.stdout
    assert "PASSED" in val.stdout


def test_cli_rejects_bad_customers(tmp_path):
    gen = subprocess.run(
        [sys.executable, "-m", "bri.data.generate_synthetic", "--customers", "0",
         "--output-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert gen.returncode == 2
    assert "customers" in gen.stderr
