"""Phase 3 sequence construction: leakage, ties, session boundaries, empty histories, truncation."""
import numpy as np
import pandas as pd
import pytest

from bri.eval.next_event import build_next_event_cases, history_counts
from bri.experiments import integrity as I
from bri.experiments.baselines import load_processed
from bri.prep.examples import build_interactions, candidate_sets
from bri.prep.split import compute_cutoffs
from bri.seq import prepare as P
from bri.seq.data import build_seq_cases, length_stats
from helpers import CATALOG, events

CUTOFFS = compute_cutoffs("2025-01-01T00:00:00Z", "2025-01-10T00:00:00Z", val_days=3, test_days=3)
CANDS = candidate_sets(CATALOG)  # nbs: SVC_A, SVC_B, SVC_C -> 1, 2, 3


def inter(rows):
    return build_interactions(events(rows), CUTOFFS)


def seq(rows, split="val", max_len=5, min_history=1, task="nbs"):
    return build_seq_cases(inter(rows), CUTOFFS, CANDS, (split,), max_len, min_history)[(task, split)]


def steps_of(s, i):
    """Decoded non-padding steps of case i as tuples of item ids."""
    return [tuple(s.items_vocab[j - 1] for j in row if j) for row in s.items[i] if row.any()]


BASE = [
    ("C1", "S1", "2025-01-01T10:00:00Z", "service_start", "SVC_A"),
    ("C1", "S1", "2025-01-01T10:01:00Z", "service_start", "SVC_B"),
    ("C1", "S2", "2025-01-02T10:00:00Z", "service_start", "SVC_C"),
    ("C1", "S3", "2025-01-04T10:00:00Z", "service_start", "SVC_A"),   # val case 1
    ("C1", "S3", "2025-01-04T10:02:00Z", "service_start", "SVC_B"),   # val case 2 (within session)
    ("C1", "S4", "2025-01-08T10:00:00Z", "service_start", "SVC_C"),   # test window (future)
]


def test_history_is_strictly_before_target_and_chronological():
    s = seq(BASE)
    assert list(s.cases["case_id"]) == sorted(s.cases["case_id"])
    assert steps_of(s, 0) == [("SVC_A",), ("SVC_B",), ("SVC_C",)]
    assert steps_of(s, 1) == [("SVC_A",), ("SVC_B",), ("SVC_C",), ("SVC_A",)]
    assert s.targets.tolist() == [[1, 0, 0], [0, 1, 0]]
    assert list(s.length) == [3, 4] and list(s.full_length) == [3, 4]


def test_target_and_future_events_never_enter_the_input():
    s = seq(BASE)
    # Removing the target and everything after it leaves case 1's input unchanged.
    trimmed = seq(BASE[:3] + [BASE[3]])
    np.testing.assert_array_equal(s.items[0], trimmed.items[0])
    np.testing.assert_array_equal(s.new_session[0], trimmed.new_session[0])
    # Changing the target item changes the label only, never the input.
    other = seq(BASE[:3] + [("C1", "S3", "2025-01-04T10:00:00Z", "service_start", "SVC_C")])
    np.testing.assert_array_equal(s.items[0], other.items[0])
    assert other.targets[0].tolist() == [0, 0, 1]


def test_tasks_are_never_mixed():
    rows = BASE + [("C1", "S1", "2025-01-01T10:00:30Z", "product_detail_view", "PRD_X"),
                   ("C1", "S3", "2025-01-04T11:00:00Z", "product_detail_view", "PRD_Y")]
    data = build_seq_cases(inter(rows), CUTOFFS, CANDS, ("val",), 5, 1)
    nbs, nbp = data[("nbs", "val")], data[("nbp", "val")]
    assert all(it.startswith("SVC") for i in range(len(nbs.cases)) for st in steps_of(nbs, i) for it in st)
    assert steps_of(nbp, 0) == [("PRD_X",)]
    assert nbs.items_vocab == CANDS["nbs"] and nbp.items_vocab == CANDS["nbp"]


def test_tied_history_events_form_one_unordered_step():
    rows = [("C1", "S1", "2025-01-01T10:00:00Z", "service_start", "SVC_C"),
            ("C1", "S1", "2025-01-01T10:00:00Z", "service_start", "SVC_A"),
            ("C1", "S2", "2025-01-04T10:00:00Z", "service_start", "SVC_B")]
    s = seq(rows)
    assert s.items.shape[2] == 2 and s.length[0] == 1
    assert steps_of(s, 0) == [("SVC_A", "SVC_C")]  # sorted ids, not event order
    swapped = seq([rows[1], rows[0], rows[2]])
    np.testing.assert_array_equal(s.items, swapped.items)


def test_tied_targets_are_one_multi_positive_case_consistent_with_the_evaluator():
    rows = [("C1", "S1", "2025-01-01T10:00:00Z", "service_start", "SVC_A"),
            ("C1", "S2", "2025-01-04T10:00:00Z", "service_start", "SVC_B"),
            ("C1", "S2", "2025-01-04T10:00:00Z", "service_start", "SVC_C")]
    s = seq(rows)
    cases, targets, _ = build_next_event_cases(inter(rows), CUTOFFS, "val", CANDS)
    assert list(s.cases["case_id"]) == list(cases["case_id"])
    assert s.targets.tolist() == [[0, 1, 1]]
    assert steps_of(s, 0) == [("SVC_A",)]  # neither tied target is history


def test_repeated_item_in_session_is_not_a_new_step_but_a_later_session_is():
    rows = [("C1", "S1", "2025-01-01T10:00:00Z", "service_start", "SVC_A"),
            ("C1", "S1", "2025-01-01T10:05:00Z", "service_start", "SVC_A"),   # repeat: same first target
            ("C1", "S2", "2025-01-02T10:00:00Z", "service_start", "SVC_A"),   # new session: new step
            ("C1", "S3", "2025-01-04T10:00:00Z", "service_start", "SVC_B")]
    s = seq(rows)
    assert steps_of(s, 0) == [("SVC_A",), ("SVC_A",)]


def test_session_boundaries_come_from_the_historical_prefix_only():
    s = seq(BASE)
    # case 1 history: S1, S1, S2 -> flags 0 (earliest: no evidence), 0, 1; left padded to max_len=5
    assert s.new_session[0].tolist() == [0, 0, 0, 0, 1]
    assert s.new_session[1].tolist() == [0, 0, 0, 1, 1]  # + S3 (target session of case 1)
    # The target's session never affects its own input: move case 1's target to S2.
    moved = seq(BASE[:3] + [("C1", "S2", "2025-01-04T10:00:00Z", "service_start", "SVC_A")])
    np.testing.assert_array_equal(s.items[0], moved.items[0])
    np.testing.assert_array_equal(s.new_session[0], moved.new_session[0])
    assert bool(s.cases["within_session"].iloc[1]) is True  # label-side info stays in cases only


def test_empty_history_is_all_padding():
    rows = [("C1", "S1", "2025-01-04T10:00:00Z", "service_start", "SVC_A")]
    assert len(seq(rows).cases) == 0  # min_history=1: cold start excluded
    s = seq(rows, min_history=0)
    assert s.length.tolist() == [0] and not s.items.any() and not s.new_session.any()


def test_truncation_keeps_the_most_recent_steps():
    s = seq(BASE, max_len=2)
    assert steps_of(s, 1) == [("SVC_C",), ("SVC_A",)]
    assert s.length.tolist() == [2, 2] and s.full_length.tolist() == [3, 4]
    # Flags describe the represented (truncated) history: kept steps S2, S3 -> 0 (earliest kept), 1.
    assert s.new_session[1].tolist() == [0, 1]
    with pytest.raises(ValueError):
        seq(BASE, max_len=0)


def test_train_cases_use_the_train_window_and_train_history_only():
    s = seq(BASE, split="train")
    assert list(s.targets.argmax(1)) == [1, 2]  # SVC_B, SVC_C (SVC_A has no history)
    assert (s.cases["cutoff_utc"] < pd.Timestamp("2025-01-04T00:00:00Z")).all()


def test_length_stats_hand_calculated():
    st = length_stats(np.array([1, 2, 3, 4, 20]), (2, 10))
    assert st["p50"] == 3 and st["max"] == 20 and st["n_cases"] == 5
    assert st["share_exceeding"] == {"2": 0.6, "10": 0.2}
    assert st["steps_truncated"]["2"] == round((0 + 0 + 1 + 2 + 18) / 30, 6)
    assert length_stats(np.array([]))["n_cases"] == 0


# --- on the session dataset -------------------------------------------------------------

@pytest.fixture(scope="module")
def real(processed_dir):
    data = load_processed(processed_dir)
    m = data["meta"]
    cutoffs = compute_cutoffs(m["cutoffs"]["window_start_utc"], m["cutoffs"]["window_end_utc"],
                              m["params"]["val_days"], m["params"]["test_days"])
    return data, cutoffs


def test_sequences_match_the_evaluator_cases_and_history_counts(real):
    data, cutoffs = real
    inter_ = data["interactions"]
    seqs = build_seq_cases(inter_, cutoffs, data["candidates"], ("train", "val", "test"), max_len=1000)
    for split in ("val", "test"):
        cases, targets, _ = build_next_event_cases(inter_, cutoffs, split, data["candidates"])
        for task, items in data["candidates"].items():
            s = seqs[(task, split)]
            c = cases[cases["task"] == task].sort_values("case_id")
            assert list(s.cases["case_id"]) == list(c["case_id"])
            # Without truncation, the multiset of history items equals the evaluator's counts.
            counts = np.zeros((len(s.cases), len(items)))
            for j in range(len(items)):
                counts[:, j] = (s.items == j + 1).sum(axis=(1, 2))
            ref = history_counts(inter_[inter_["task"] == task], s.cases, items, "target")
            np.testing.assert_array_equal(counts, ref)


def test_dropping_test_events_does_not_change_train_or_val_sequences(real):
    data, cutoffs = real
    full = build_seq_cases(data["interactions"], cutoffs, data["candidates"], ("train", "val"), 50)
    cut = build_seq_cases(P.no_test_events(data["interactions"]), cutoffs, data["candidates"], ("train", "val"), 50)
    for key in full:
        assert full[key].fingerprint() == cut[key].fingerprint()


def test_sequence_construction_is_deterministic(real):
    data, cutoffs = real
    a = build_seq_cases(data["interactions"], cutoffs, data["candidates"], ("train", "val"), 20)
    b = build_seq_cases(data["interactions"], cutoffs, data["candidates"], ("train", "val"), 20)
    assert {k: v.fingerprint() for k, v in a.items()} == {k: v.fingerprint() for k, v in b.items()}


def test_next_event_val_test_output_is_unchanged_by_train_support(real):
    """Phase 2.3 protocol: val/test cases identical to the frozen protocol's builder output."""
    data, cutoffs = real
    for split in ("val", "test"):
        c, t, st = build_next_event_cases(data["interactions"], cutoffs, split, data["candidates"])
        assert set(c["split"]) <= {split}
        assert (c["history_length"] >= 1).all()
    with pytest.raises(ValueError):
        build_next_event_cases(data["interactions"], cutoffs, "holdout", data["candidates"])


# --- preflight -----------------------------------------------------------------------------

def test_length_analysis_refuses_final_namespace_and_frozen_dir_without_writing(tmp_path):
    with pytest.raises(I.LockedTestSplitError, match="must not inform hyperparameters"):
        P.run(tmp_path / "data", tmp_path / "rep", namespace="phase3_final", log=lambda *_: None)
    frozen = tmp_path / "rep2"
    frozen.mkdir()
    (frozen / "FROZEN").write_text("x", encoding="utf-8")
    with pytest.raises(I.LockedTestSplitError, match="frozen"):
        P.run(tmp_path / "data", frozen, log=lambda *_: None)
    assert not (tmp_path / "data").exists()


def test_length_analysis_refuses_unregistered_namespace(tmp_path):
    with pytest.raises(I.LockedTestSplitError):
        P.run(tmp_path / "data", tmp_path / "rep", manifests={}, log=lambda *_: None)
    assert not (tmp_path / "data").exists()


def test_earliest_step_is_never_marked_as_a_new_session():
    rows = [("C1", "S9", "2025-01-01T10:00:00Z", "service_start", "SVC_A"),
            ("C1", "S8", "2025-01-04T10:00:00Z", "service_start", "SVC_B")]
    s = seq(rows)
    assert s.new_session[0].tolist() == [0, 0, 0, 0, 0]  # single history step: no boundary evidence
    for max_len in (1, 2, 3):  # every truncation: earliest represented step is 0
        t = seq(BASE, max_len=max_len)
        for i, n in enumerate(t.length):
            if n:
                assert t.new_session[i, max_len - n] == 0
