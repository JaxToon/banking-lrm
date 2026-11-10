"""Next-event evaluation: case construction, leakage boundaries, and the shared runner."""
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from bri.eval.next_event import build_next_event_cases, history_counts
from bri.experiments.baselines import load_processed, next_event_data, run_next_event
from bri.models.common import history_matrix
from bri.prep.examples import build_interactions, candidate_sets
from bri.prep.split import compute_cutoffs
from bri.tasks import SYNTHETIC_LABEL
from helpers import CATALOG, events

CUTOFFS = compute_cutoffs("2025-01-01T00:00:00Z", "2025-01-10T00:00:00Z", val_days=3, test_days=3)
CANDS = candidate_sets(CATALOG)
FAST = dict(ks=(1, 3), n_boot=50, seed=42, random_seeds=2)
FIVE = {"popularity", "personal_frequency", "itemknn_self", "itemknn_noself", "mf"}


def interactions(rows):
    return build_interactions(events(rows), CUTOFFS)


def test_one_case_per_first_target_with_strict_history():
    # val window is [Jan 4, Jan 7)
    inter = interactions([
        ("C1", "S1", "2025-01-01T10:00:00Z", "service_start", "SVC_A"),   # train history
        ("C1", "S2", "2025-01-04T10:00:00Z", "service_view", "SVC_B"),    # not a target
        ("C1", "S2", "2025-01-04T10:00:05Z", "service_start", "SVC_B"),   # case 1
        ("C1", "S2", "2025-01-04T10:00:30Z", "service_start", "SVC_B"),   # repeat in session: no case
        ("C1", "S2", "2025-01-04T10:01:00Z", "service_start", "SVC_C"),   # case 2 (within session)
        ("C1", "S3", "2025-01-05T09:00:00Z", "service_start", "SVC_B"),   # case 3 (repeat, new session)
    ])
    cases, targets, stats = build_next_event_cases(inter, CUTOFFS, "val", CANDS)
    assert list(targets["item_id"]) == ["SVC_B", "SVC_C", "SVC_B"]
    assert list(cases["history_length"]) == [1, 2, 3]
    assert list(cases["within_session"]) == [False, True, False]
    assert stats["eligible_target_events"] == 3 and stats["cases"] == 3
    assert list(cases["cutoff_utc"]) == list(pd.to_datetime(
        ["2025-01-04T10:00:05Z", "2025-01-04T10:01:00Z", "2025-01-05T09:00:00Z"], utc=True))


def test_history_never_contains_the_target_or_later_events():
    inter = interactions([
        ("C1", "S1", "2025-01-01T10:00:00Z", "service_start", "SVC_A"),
        ("C1", "S2", "2025-01-04T10:00:00Z", "service_start", "SVC_B"),
        ("C1", "S3", "2025-01-08T10:00:00Z", "service_start", "SVC_C"),   # test window, future
    ])
    cases, _, _ = build_next_event_cases(inter, CUTOFFS, "val", CANDS)
    h = history_counts(inter[inter["task"] == "nbs"], cases, CANDS["nbs"], "target")
    assert h.tolist() == [[1.0, 0.0, 0.0]]  # SVC_A only: neither SVC_B (target) nor SVC_C (future)


def test_timestamp_ties_merge_into_one_case_and_exclude_each_other():
    inter = interactions([
        ("C1", "S1", "2025-01-01T10:00:00Z", "service_start", "SVC_A"),
        ("C1", "S2", "2025-01-04T10:00:00Z", "service_start", "SVC_B"),
        ("C1", "S2", "2025-01-04T10:00:00Z", "service_start", "SVC_C"),
    ])
    cases, targets, stats = build_next_event_cases(inter, CUTOFFS, "val", CANDS)
    assert len(cases) == 1 and stats["tied_timestamp_merges"] == 1
    assert cases.iloc[0]["n_relevant"] == 2 and cases.iloc[0]["history_length"] == 1
    assert sorted(targets["item_id"]) == ["SVC_B", "SVC_C"]
    h = history_counts(inter[inter["task"] == "nbs"], cases, CANDS["nbs"], "target")
    assert h.tolist() == [[1.0, 0.0, 0.0]]


def test_insufficient_history_is_excluded_and_counted():
    inter = interactions([
        ("C1", "S1", "2025-01-04T10:00:00Z", "service_start", "SVC_A"),   # no history -> cold start
        ("C1", "S2", "2025-01-05T10:00:00Z", "service_start", "SVC_B"),   # history 1 -> case
        ("C2", "S3", "2025-01-05T10:00:00Z", "service_start", "SVC_A"),   # single-target session, cold
    ])
    cases, _, stats = build_next_event_cases(inter, CUTOFFS, "val", CANDS, min_history=1)
    assert stats["insufficient_history"] == 2 and stats["cases"] == 1
    assert stats["single_target_sessions"] == 3
    assert cases.iloc[0]["customer_token"] == "C1"
    _, _, stats0 = build_next_event_cases(inter, CUTOFFS, "val", CANDS, min_history=0)
    assert stats0["cases"] == 3


def test_tasks_stay_separate():
    inter = interactions([
        ("C1", "S1", "2025-01-01T10:00:00Z", "service_start", "SVC_A"),
        ("C1", "S1", "2025-01-01T10:01:00Z", "product_detail_view", "PRD_X"),
        ("C1", "S2", "2025-01-04T10:00:00Z", "product_detail_view", "PRD_Y"),
    ])
    cases, targets, _ = build_next_event_cases(inter, CUTOFFS, "val", CANDS)
    assert list(cases["task"]) == ["nbp"] and list(targets["item_id"]) == ["PRD_Y"]
    assert cases.iloc[0]["history_length"] == 1  # only the product target, not the service start
    h = history_counts(inter[inter["task"] == "nbp"], cases, CANDS["nbp"], "target")
    assert h.tolist() == [[1.0, 0.0, 0.0]]


def test_window_bounds_are_half_open():
    inter = interactions([
        ("C1", "S1", "2025-01-01T10:00:00Z", "service_start", "SVC_A"),
        ("C1", "S2", "2025-01-04T00:00:00Z", "service_start", "SVC_B"),   # val start: in val
        ("C1", "S3", "2025-01-07T00:00:00Z", "service_start", "SVC_C"),   # test start: not in val
    ])
    val, _, _ = build_next_event_cases(inter, CUTOFFS, "val", CANDS)
    test, _, _ = build_next_event_cases(inter, CUTOFFS, "test", CANDS)
    assert len(val) == 1 and len(test) == 1
    assert test.iloc[0]["history_length"] == 2


@pytest.fixture(scope="module")
def ne(processed_dir):
    data = load_processed(processed_dir)
    return data, next_event_data(data)


def test_history_counts_matches_reference_on_window_cases(ne):
    data, _ = ne
    for task, items in data["candidates"].items():
        inter = data["interactions"][data["interactions"]["task"] == task]
        cases = data["cases"][data["cases"]["task"] == task]
        for signal in ("target", "item_event"):
            np.testing.assert_array_equal(history_counts(inter, cases, items, signal),
                                          history_matrix(inter, cases, items, signal))


def test_next_event_history_ignores_target_and_future_events(ne):
    """Leakage test: deleting every event at or after a case's cutoff never changes its history."""
    data, (ne_data, _) = ne
    inter = data["interactions"]
    cases = ne_data["cases"]
    rng = np.random.default_rng(0)
    sample = cases.iloc[rng.choice(len(cases), size=min(60, len(cases)), replace=False)]
    for _, case in sample.iterrows():
        task, items = case["task"], data["candidates"][case["task"]]
        one = sample[sample["case_id"] == case["case_id"]]
        mine = inter[(inter["task"] == task) & (inter["customer_token"] == case["customer_token"])]
        past = mine[mine["event_timestamp"] < case["cutoff_utc"]]
        for signal in ("target", "item_event"):
            full = history_counts(mine, one, items, signal)
            np.testing.assert_array_equal(full, history_counts(past, one, items, signal))
        # The target item's first-target event itself is never counted.
        tgt = ne_data["targets"][ne_data["targets"]["case_id"] == case["case_id"]]["item_id"]
        at = mine[(mine["event_timestamp"] == case["cutoff_utc"]) & mine["is_target"]]
        assert set(tgt) <= set(at["item_id"])


def test_next_event_cases_use_only_their_split_window(ne, processed_dir):
    data, (ne_data, stats) = ne
    meta = data["meta"]["cutoffs"]
    for split, lo, hi in (("val", meta["val_start_utc"], meta["test_start_utc"]),
                          ("test", meta["test_start_utc"], meta["window_end_utc"])):
        c = ne_data["cases"][ne_data["cases"]["split"] == split]
        assert (c["cutoff_utc"] >= pd.Timestamp(lo)).all() and (c["cutoff_utc"] < pd.Timestamp(hi)).all()
        assert (c["history_length"] >= 1).all()
        assert stats[split]["cases"] == len(c)
    t = ne_data["targets"].merge(ne_data["cases"][["case_id", "task"]], on="case_id")
    for task, items in data["candidates"].items():
        assert set(t.loc[t["task"] == task, "item_id"]) <= set(items)


@pytest.fixture(scope="module")
def ne_report(processed_dir):
    return run_next_event(processed_dir, **FAST)


def test_next_event_report_shares_the_evaluation_interface(ne_report):
    r = ne_report
    assert r["label"] == SYNTHETIC_LABEL and r["evaluation"] == "next_event"
    assert "customer" in r["protocol"]["bootstrap"] and "one dataset seed" in r["protocol"]["ci_scope"]
    assert "Cold start is not evaluated" in r["population_note"]
    for task in r["tasks"].values():
        assert set(task["models"]) == FIVE
        for m in task["models"].values():
            assert m["selected_params"]["signal"] == "target"
        comps = [c["comparison"] for c in task["key_comparisons"]]
        assert all("personal_frequency" in c for c in comps[:4])


def test_next_event_is_deterministic(processed_dir, ne_report):
    again = run_next_event(processed_dir, **FAST)
    assert json.dumps(again, sort_keys=True) == json.dumps(ne_report, sort_keys=True)


def test_test_events_never_influence_next_event_selection(processed_dir, ne_report, tmp_path):
    altered = tmp_path / "processed"
    shutil.copytree(processed_dir, altered)
    inter = pd.read_csv(altered / "interactions.csv", dtype=str, keep_default_na=False)
    meta = json.loads((altered / "prep_metadata.json").read_text(encoding="utf-8"))
    test_rows = inter["split"] == "test"
    first = {t: v["candidates"][0] for t, v in meta["tasks"].items()}
    has_item = test_rows & (inter["item_id"] != "")
    inter.loc[has_item, "item_id"] = inter.loc[has_item, "task"].map(first)
    inter.to_csv(altered / "interactions.csv", index=False, lineterminator="\n")
    meta["sha256"]["interactions.csv"] = hashlib.sha256((altered / "interactions.csv").read_bytes()).hexdigest()
    (altered / "prep_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    other = run_next_event(altered, **FAST)
    for task, e in ne_report["tasks"].items():
        assert e["n_val_cases"] == other["tasks"][task]["n_val_cases"]
        for name, m in e["models"].items():
            assert m["validation_trials"] == other["tasks"][task]["models"][name]["validation_trials"]
            assert m["selected_params"] == other["tasks"][task]["models"][name]["selected_params"]
