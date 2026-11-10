"""Generator 0.3.0 behavior profiles, and preservation of generator 0.2.0 output."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from bri.data import schema as S
from bri.data.generate_synthetic import (LEGACY_PROFILE, PRODUCT_IDS, PRODUCT_TRANSITION, PROFILES,
                                         SERVICE_IDS, SERVICE_TRANSITION, BehaviorProfile, generate_dataset,
                                         write_dataset)
from bri.data.validate_dataset import validate
from bri.experiments.integrity import load_manifests, sha256_file

# Source checksums of the locked 0.2.0 benchmark datasets (500 customers, 30 days).
REGISTERED = {
    42: "ca53375dd030f13627ae3b7748ccdf209f32f2007657a3d918d84c3682a34fa8",
    20261109: "9d68d4145a7c251b749ea38464a780126545b02934b07d80f01c145d20a6927b",
}


@pytest.mark.parametrize("seed", sorted(REGISTERED))
def test_generator_020_reproduces_registered_benchmarks_byte_identically(seed, tmp_path):
    write_dataset(tmp_path, customers=500, days=30, seed=seed)
    assert sha256_file(tmp_path / "events.csv") == REGISTERED[seed]
    manifests = load_manifests()
    entries = [e for m in manifests.values() for e in m["datasets"]
               if e["source_files_sha256"]["events.csv"] == REGISTERED[seed]]
    assert len(entries) == 1
    for name, digest in entries[0]["source_files_sha256"].items():
        assert sha256_file(tmp_path / name) == digest, name
    meta = json.loads((tmp_path / S.METADATA_FILE).read_text(encoding="utf-8"))
    assert meta["generator_version"] == "0.2.0" and "behavior_profile" not in meta


def test_legacy_profile_object_equals_default_path():
    a = generate_dataset(30, 7, seed=3)
    b = generate_dataset(30, 7, seed=3, profile=LEGACY_PROFILE)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_profiles_are_byte_reproducible_and_valid(profile, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_dataset(a, 40, 10, seed=7, profile=profile)
    write_dataset(b, 40, 10, seed=7, profile=profile)
    for name in [*S.TABLES, S.METADATA_FILE]:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    assert validate(a) == []
    meta = json.loads((a / S.METADATA_FILE).read_text(encoding="utf-8"))
    assert meta["generator_version"] == "0.3.0"
    assert meta["behavior_profile"] == profile
    assert meta["behavior_params"] == PROFILES[profile].to_dict()
    assert "invented experimental assumptions" in meta["behavior_note"]


def test_profiles_differ_from_each_other_and_from_legacy():
    events = {p: generate_dataset(40, 10, seed=7, profile=p)[3] for p in PROFILES}
    events["legacy"] = generate_dataset(40, 10, seed=7)[3]
    names = list(events)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not events[a].equals(events[b]), (a, b)


def test_customers_and_consent_are_shared_across_profiles_for_one_seed():
    """Only behavior differs between profiles for the same seed (paired design)."""
    ref = generate_dataset(40, 10, seed=11, profile="balanced")
    for p in ("habit_dominant", "transition_dominant"):
        other = generate_dataset(40, 10, seed=11, profile=p)
        pd.testing.assert_frame_equal(ref[0], other[0])
        pd.testing.assert_frame_equal(ref[2], other[2])


def _intents(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """One row per journey intent (the view event that opens a service/product journey)."""
    return events[events["event_type"] == event_type][["customer_token", "session_id", "item_id"]]


def _top_share(intents: pd.DataFrame) -> float:
    """Mean share of each customer's most frequent item (customers with >= 5 intents)."""
    g = intents.groupby("customer_token")["item_id"]
    shares = [c.value_counts().iloc[0] / len(c) for _, c in g if len(c) >= 5]
    return float(np.mean(shares))


def _transition_rate(intents: pd.DataFrame, mapping: dict) -> float:
    hits = total = 0
    for _, s in intents.groupby("session_id", sort=False)["item_id"]:
        items = list(s)
        for prev, nxt in zip(items, items[1:]):
            total += 1
            hits += mapping[prev] == nxt
    return hits / total


@pytest.fixture(scope="module")
def profile_events():
    return {p: generate_dataset(200, 30, seed=2026, profile=p)[3] for p in PROFILES}


def test_concentration_controls_customer_specific_preference(profile_events):
    """Higher Dirichlet concentration = weaker customer-specific item preference."""
    for event_type in ("service_view", "product_view"):
        share = {p: _top_share(_intents(e, event_type)) for p, e in profile_events.items()}
        assert share["habit_dominant"] > share["balanced"] > share["transition_dominant"], (event_type, share)


def test_transition_probability_controls_in_session_transitions(profile_events):
    for event_type, mapping in (("service_view", SERVICE_TRANSITION), ("product_view", PRODUCT_TRANSITION)):
        rate = {p: _transition_rate(_intents(e, event_type), mapping) for p, e in profile_events.items()}
        assert rate["transition_dominant"] > rate["balanced"] > rate["habit_dominant"], (event_type, rate)
        assert rate["transition_dominant"] > 0.6


def test_transitions_never_cross_tasks():
    bad_service = dict(SERVICE_TRANSITION, SVC_QR="PRD_SAVINGS")
    with pytest.raises(ValueError, match="same task"):
        BehaviorProfile("x", 1.0, 1.0, 0.5, 0.5, service_transition_map=bad_service)
    with pytest.raises(ValueError, match="same task"):
        BehaviorProfile("x", 1.0, 1.0, 0.5, 0.5, product_transition_map={"PRD_SAVINGS": "PRD_DEBIT"})
    for p in PROFILES.values():
        assert set(p.service_transition_map) == set(SERVICE_IDS) and set(p.service_transition_map.values()) <= set(SERVICE_IDS)
        assert set(p.product_transition_map) == set(PRODUCT_IDS) and set(p.product_transition_map.values()) <= set(PRODUCT_IDS)


@pytest.mark.parametrize("kwargs", [
    dict(service_concentration=0.0), dict(product_concentration=-1.0),
    dict(service_transition_p=1.5), dict(product_transition_p=-0.1),
])
def test_invalid_profile_parameters_are_rejected(kwargs):
    base = dict(name="x", service_concentration=1.0, product_concentration=1.0,
                service_transition_p=0.5, product_transition_p=0.5)
    with pytest.raises(ValueError):
        BehaviorProfile(**{**base, **kwargs})


def test_unknown_profile_name_is_rejected():
    with pytest.raises(ValueError, match="profile must be one of"):
        generate_dataset(5, 3, seed=1, profile="impulsive")


def test_profile_items_stay_within_their_purpose(profile_events):
    """Service items only appear on service surfaces, product items only on product surfaces."""
    for e in profile_events.values():
        svc = e[e["item_id"].isin(SERVICE_IDS)]
        prd = e[e["item_id"].isin(PRODUCT_IDS)]
        assert set(svc["purpose"]) == {S.SERVICE_PURPOSE}
        assert set(prd["purpose"]) == {S.MARKETING_PURPOSE}


def test_cli_profile_option(tmp_path):
    r = subprocess.run([sys.executable, "-m", "bri.data.generate_synthetic", "--customers", "10", "--days", "5",
                        "--seed", "1", "--profile", "balanced", "--output-dir", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "generator 0.3.0" in r.stdout
    r = subprocess.run([sys.executable, "-m", "bri.data.generate_synthetic", "--profile", "nope",
                        "--output-dir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 2



def test_session_volume_is_comparable_across_profiles(profile_events):
    """Profiles change item choice, not the session process. Counts are not identical
    (Dirichlet sampling consumes a profile-dependent number of random draws, so the
    streams diverge), but volumes must agree closely."""
    for col in ("session_id", "event_id"):
        n = {p: e[col].nunique() for p, e in profile_events.items()}
        assert max(n.values()) / min(n.values()) < 1.05, (col, n)
