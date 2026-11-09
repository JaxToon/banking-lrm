import pandas as pd
import pytest

from bri.prep.build import load_dataset
from bri.prep.consent import filter_by_consent
from bri.prep.examples import (
    build_eval_cases, build_interactions, build_train_examples, candidate_sets, sample_negatives,
)
from bri.prep.split import compute_cutoffs
from helpers import CATALOG, allow_all, events

CUT = compute_cutoffs("2024-12-01T00:00:00Z", "2024-12-31T00:00:00Z", val_days=10, test_days=10)
# train < 12-11 <= val < 12-21 <= test
C1, C2 = "SYN_CUST_000001", "SYN_CUST_000002"
ROWS = [
    (C1, "SYN_SESS_000001_0001", "2024-12-02T08:00:00Z", "app_open", ""),
    (C1, "SYN_SESS_000001_0001", "2024-12-02T08:00:10Z", "service_view", "SVC_A"),
    (C1, "SYN_SESS_000001_0001", "2024-12-02T08:00:20Z", "service_start", "SVC_A"),
    (C1, "SYN_SESS_000001_0001", "2024-12-02T08:00:30Z", "product_view", "PRD_X"),
    (C1, "SYN_SESS_000001_0001", "2024-12-02T08:00:40Z", "product_detail_view", "PRD_X"),
    (C1, "SYN_SESS_000001_0001", "2024-12-02T08:00:50Z", "product_calculator_use", "PRD_X"),  # same journey
    (C1, "SYN_SESS_000001_0002", "2024-12-12T09:00:00Z", "service_start", "SVC_B"),        # val
    (C1, "SYN_SESS_000001_0003", "2024-12-22T09:00:00Z", "service_start", "SVC_C"),        # test
    (C2, "SYN_SESS_000002_0001", "2024-12-13T10:00:00Z", "service_start", "SVC_A"),        # val, no history
]


def build(rows=ROWS, consent=None, min_history=1):
    ev = events(rows)
    kept, _ = filter_by_consent(ev, consent if consent is not None else allow_all([C1, C2]))
    inter = build_interactions(kept, CUT)
    cands = candidate_sets(CATALOG)
    return inter, cands, build_train_examples(inter, min_history)


def test_candidate_sets_are_task_specific():
    assert candidate_sets(CATALOG) == {"nbs": ["SVC_A", "SVC_B", "SVC_C"], "nbp": ["PRD_X", "PRD_Y", "PRD_Z"]}


def test_inactive_items_are_not_candidates():
    cat = CATALOG.assign(active_status=["True", "False", "True", "True", "True", "True"])
    assert candidate_sets(cat)["nbs"] == ["SVC_A", "SVC_C"]


def test_nbs_and_nbp_are_separated():
    inter, _, (ex, _) = build()
    nbs = inter[inter["task"] == "nbs"]
    nbp = inter[inter["task"] == "nbp"]
    assert set(nbs["event_type"]) <= {"app_open", "service_view", "service_start"}
    assert set(nbp["event_type"]) <= {"product_view", "product_detail_view", "product_calculator_use"}
    assert set(ex.loc[ex["task"] == "nbs", "target_item_id"]).issubset({"SVC_A", "SVC_B", "SVC_C"})
    assert set(ex.loc[ex["task"] == "nbp", "target_item_id"]) == {"PRD_X"}


def test_nbp_history_never_contains_service_purpose_events():
    """Cross-purpose use is prohibited: NBP sees only marketing-purpose events."""
    inter, _, _ = build()
    assert (inter.loc[inter["task"] == "nbp", "item_id"].str.startswith("PRD_")).all()


def test_train_examples_dedupe_per_journey_and_count_strict_history():
    _, _, (ex, dropped) = build()
    assert ex[["task", "target_item_id", "history_length"]].values.tolist() == [
        ["nbp", "PRD_X", 1],   # product_view precedes detail view; calculator is a duplicate target
        ["nbs", "SVC_A", 2],   # app_open + service_view precede the start
    ]
    assert dropped == 0
    assert ex["example_id"].tolist() == ["SYN_EX_000000001", "SYN_EX_000000002"]


def test_insufficient_history_is_dropped_and_counted():
    _, _, (ex, dropped) = build(min_history=3)
    assert ex.empty and dropped == 2
    _, _, (ex0, dropped0) = build(min_history=0)
    assert len(ex0) == 2 and dropped0 == 0


def test_eval_cases_use_only_pre_cutoff_history():
    inter, cands, _ = build()
    cases, targets, stats = build_eval_cases(inter, CUT, "val", cands, min_history=1)
    # C2 has a val target but no history before the cutoff -> excluded and counted
    assert cases["customer_token"].tolist() == [C1]
    assert cases.iloc[0]["history_length"] == 3  # app_open, service_view, service_start (train only)
    assert targets["item_id"].tolist() == ["SVC_B"]
    assert stats["insufficient_history"] == 1

    test_cases, test_targets, _ = build_eval_cases(inter, CUT, "test", cands, min_history=1)
    assert test_cases.iloc[0]["history_length"] == 4  # train + val events now precede the cutoff
    assert test_targets["item_id"].tolist() == ["SVC_C"]


def test_eval_case_with_min_history_zero_includes_cold_start():
    inter, cands, _ = build()
    cases, _, _ = build_eval_cases(inter, CUT, "val", cands, min_history=0)
    assert sorted(cases["customer_token"]) == [C1, C2]


def test_consent_denied_customer_produces_no_examples_or_cases():
    from helpers import consent
    c = pd.concat([allow_all([C1]), consent([(C2, p, False, "2024-01-01T00:00:00Z")
                                              for p in ("service_personalization", "marketing_personalization")])])
    inter, cands, _ = build(consent=c, min_history=0)
    assert C2 not in set(inter["customer_token"])
    cases, _, _ = build_eval_cases(inter, CUT, "val", cands, min_history=0)
    assert C2 not in set(cases["customer_token"])


def test_empty_inputs():
    inter, cands, (ex, dropped) = build(rows=ROWS[:0])
    assert inter.empty and ex.empty and dropped == 0
    cases, targets, _ = build_eval_cases(inter, CUT, "val", cands)
    assert cases.empty and targets.empty


def test_eval_split_must_be_val_or_test():
    inter, cands, _ = build()
    with pytest.raises(ValueError):
        build_eval_cases(inter, CUT, "train", cands)


def test_future_events_do_not_change_train_or_val_outputs(dataset_dir):
    """Leakage test: removing every event at/after a cutoff must not change what is
    built for periods before that cutoff."""
    data = load_dataset(dataset_dir)
    meta = data["metadata"]
    cut = compute_cutoffs(meta["window_start_utc"], meta["window_end_utc"], 3, 3)
    cands = candidate_sets(data["catalog"])

    def outputs(ev):
        kept, _ = filter_by_consent(ev, data["consent"])
        inter = build_interactions(kept, cut)
        return inter, build_train_examples(inter)[0], build_eval_cases(inter, cut, "val", cands)

    full = outputs(data["events"])
    no_future_val = outputs(data["events"][data["events"]["event_timestamp"] < cut.val_start])
    no_future_test = outputs(data["events"][data["events"]["event_timestamp"] < cut.test_start])
    pd.testing.assert_frame_equal(full[1], no_future_val[1])          # train examples
    pd.testing.assert_frame_equal(full[2][0], no_future_test[2][0])   # val cases (history)
    pd.testing.assert_frame_equal(full[2][1], no_future_test[2][1])   # val targets


def test_train_example_history_strictly_precedes_target(processed_dir):
    inter = pd.read_csv(processed_dir / "interactions.csv", dtype=str, keep_default_na=False)
    ex = pd.read_csv(processed_dir / "train_examples.csv", dtype=str, keep_default_na=False)
    assert len(ex) > 0
    train = inter[inter["split"] == "train"]
    for row in ex.sample(min(25, len(ex)), random_state=0).itertuples():
        mine = train[(train["task"] == row.task) & (train["customer_token"] == row.customer_token)]
        assert int(row.history_length) == int((mine["event_timestamp"] < row.target_timestamp).sum())


def test_negative_sampling_is_deterministic_and_excludes_target():
    _, cands, (ex, _) = build()
    a = sample_negatives(ex, cands, n_negatives=2, seed=7)
    b = sample_negatives(ex, cands, n_negatives=2, seed=7)
    pd.testing.assert_frame_equal(a, b)
    merged = a.merge(ex[["example_id", "task", "target_item_id"]], on="example_id")
    assert (merged["item_id"] != merged["target_item_id"]).all()
    assert all(item in cands[t] for item, t in zip(merged["item_id"], merged["task"]))
    assert a.groupby("example_id").size().tolist() == [2, 2]


def test_negative_sampling_caps_at_pool_size():
    _, cands, (ex, _) = build()
    neg = sample_negatives(ex, cands, n_negatives=10, seed=1)
    assert neg.groupby("example_id").size().tolist() == [2, 2]  # 3 candidates minus target
    with pytest.raises(ValueError):
        sample_negatives(ex, cands, n_negatives=0, seed=1)
