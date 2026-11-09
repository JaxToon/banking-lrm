import pandas as pd
import pytest

from bri.prep.split import assign_split, compute_cutoffs

START, END = "2024-12-01T00:00:00Z", "2024-12-31T00:00:00Z"


def ts(*values):
    return pd.Series(pd.to_datetime(list(values), utc=True))


def test_cutoffs_are_last_whole_days():
    c = compute_cutoffs(START, END, val_days=7, test_days=5)
    assert c.to_dict() == {
        "window_start_utc": "2024-12-01T00:00:00Z",
        "val_start_utc": "2024-12-19T00:00:00Z",
        "test_start_utc": "2024-12-26T00:00:00Z",
        "window_end_utc": "2024-12-31T00:00:00Z",
    }


def test_boundaries_are_half_open():
    c = compute_cutoffs(START, END, 7, 5)
    labels = assign_split(ts(
        "2024-12-01T00:00:00Z",   # window start -> train
        "2024-12-18T23:59:59Z",   # last train second
        "2024-12-19T00:00:00Z",   # exactly val cutoff -> val
        "2024-12-25T23:59:59Z",
        "2024-12-26T00:00:00Z",   # exactly test cutoff -> test
        "2024-12-30T23:59:59Z",
    ), c)
    assert labels.tolist() == ["train", "train", "val", "val", "test", "test"]


def test_splits_are_chronologically_ordered():
    c = compute_cutoffs(START, END, 7, 5)
    times = pd.Series(pd.date_range(START, END, freq="37min", inclusive="left"))
    labels = assign_split(times, c)
    assert times[labels == "train"].max() < times[labels == "val"].min()
    assert times[labels == "val"].max() < times[labels == "test"].min()


@pytest.mark.parametrize("value", ["2024-11-30T23:59:59Z", "2024-12-31T00:00:00Z"])
def test_out_of_window_timestamps_rejected(value):
    with pytest.raises(ValueError, match="outside"):
        assign_split(ts(value), compute_cutoffs(START, END, 7, 5))


def test_naive_timestamps_rejected():
    with pytest.raises(ValueError):
        compute_cutoffs("2024-12-01", END, 7, 5)
    with pytest.raises(ValueError):
        assign_split(pd.Series(pd.to_datetime(["2024-12-10"])), compute_cutoffs(START, END, 7, 5))


@pytest.mark.parametrize("val,test", [(0, 5), (5, 0), (-1, 3), (2.5, 3), (True, 3)])
def test_invalid_split_lengths(val, test):
    with pytest.raises(ValueError):
        compute_cutoffs(START, END, val, test)


def test_window_too_short_for_training():
    with pytest.raises(ValueError, match="too short"):
        compute_cutoffs(START, END, 15, 15)  # no training day left


def test_empty_series_is_fine():
    out = assign_split(pd.Series([], dtype="datetime64[ns, UTC]"), compute_cutoffs(START, END, 7, 5))
    assert out.empty
