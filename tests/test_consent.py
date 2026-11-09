import pandas as pd

from bri.data import schema as S
from bri.prep.consent import EXCLUSION_REASONS, filter_by_consent
from helpers import allow_all, consent, events

SVC, MKT = S.SERVICE_PURPOSE, S.MARKETING_PURPOSE
EV = events([
    ("SYN_CUST_000001", "SYN_SESS_000001_0001", "2024-12-10T08:00:00Z", "app_open", ""),
    ("SYN_CUST_000001", "SYN_SESS_000001_0001", "2024-12-10T08:00:10Z", "service_start", "SVC_A"),
    ("SYN_CUST_000001", "SYN_SESS_000001_0001", "2024-12-10T08:00:20Z", "product_detail_view", "PRD_X"),
])


def test_all_allowed_keeps_everything():
    kept, report = filter_by_consent(EV, allow_all(["SYN_CUST_000001"]))
    assert len(kept) == 3
    assert report["kept_events"] == 3 and sum(report["excluded"].values()) == 0


def test_purpose_maps_to_its_own_consent_setting():
    c = consent([("SYN_CUST_000001", SVC, True, "2024-01-01T00:00:00Z"),
                 ("SYN_CUST_000001", MKT, False, "2024-01-01T00:00:00Z")])
    kept, report = filter_by_consent(EV, c)
    assert set(kept["purpose"]) == {SVC}
    assert report["excluded"]["consent_not_granted"] == 1


def test_missing_consent_record_fails_closed():
    c = consent([("SYN_CUST_000001", SVC, True, "2024-01-01T00:00:00Z")])  # no marketing row
    kept, report = filter_by_consent(EV, c)
    assert (kept["purpose"] == SVC).all()
    assert report["excluded"]["no_consent_record"] == 1


def test_customer_without_any_consent_is_fully_excluded():
    kept, report = filter_by_consent(EV, allow_all(["SYN_CUST_000002"]))
    assert kept.empty
    assert report["excluded"]["no_consent_record"] == 3


def test_unknown_event_purpose_fails_closed():
    ev = events([("SYN_CUST_000001", "SYN_SESS_000001_0001", "2024-12-10T08:00:00Z", "app_open", "")],
                purpose_override="analytics")
    c = pd.concat([allow_all(["SYN_CUST_000001"]),
                   consent([("SYN_CUST_000001", "analytics", True, "2024-01-01T00:00:00Z")])])
    kept, report = filter_by_consent(ev, c)
    assert kept.empty
    assert report["excluded"]["unknown_purpose"] == 1


def test_ambiguous_duplicate_consent_fails_closed():
    c = pd.concat([allow_all(["SYN_CUST_000001"]),
                   consent([("SYN_CUST_000001", SVC, True, "2024-02-01T00:00:00Z")])])
    kept, report = filter_by_consent(EV, c)
    assert (kept["purpose"] == MKT).all()
    assert report["excluded"]["ambiguous_consent"] == 2


def test_non_literal_true_values_do_not_grant():
    for value in ["true", "yes", "1", "", None, "False"]:
        c = consent([("SYN_CUST_000001", SVC, value, "2024-01-01T00:00:00Z"),
                     ("SYN_CUST_000001", MKT, value, "2024-01-01T00:00:00Z")])
        kept, _ = filter_by_consent(EV, c)
        assert kept.empty, value


def test_string_true_from_csv_grants():
    c = consent([("SYN_CUST_000001", p, "True", "2024-01-01T00:00:00Z") for p in S.PURPOSES])
    assert len(filter_by_consent(EV, c)[0]) == 3


def test_events_before_capture_are_excluded():
    c = allow_all(["SYN_CUST_000001"], captured="2024-12-10T08:00:15Z")
    kept, report = filter_by_consent(EV, c)
    assert kept["event_type"].tolist() == ["product_detail_view"]
    assert report["excluded"]["event_before_consent_capture"] == 2


def test_missing_capture_timestamp_is_ambiguous():
    c = allow_all(["SYN_CUST_000001"], captured="not-a-time")
    kept, report = filter_by_consent(EV, c)
    assert kept.empty
    assert report["excluded"]["ambiguous_consent"] == 3


def test_empty_events():
    kept, report = filter_by_consent(EV.iloc[0:0], allow_all(["SYN_CUST_000001"]))
    assert kept.empty and report["input_events"] == 0
    assert set(report["excluded"]) == set(EXCLUSION_REASONS)


def test_generated_dataset_denied_customers_never_leak(dataset_dir):
    from bri.prep.build import load_dataset

    data = load_dataset(dataset_dir)
    kept, report = filter_by_consent(data["events"], data["consent"])
    allowed = data["consent"][data["consent"]["allowed"] == "True"]
    allowed_keys = set(zip(allowed["customer_token"], allowed["purpose"]))
    assert set(zip(kept["customer_token"], kept["purpose"])) <= allowed_keys
    assert report["kept_events"] + sum(report["excluded"].values()) == report["input_events"]
