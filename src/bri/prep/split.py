"""Chronological train/validation/test splitting.

Splits are global calendar cutoffs over the dataset window, using half-open UTC
intervals:

    train = [window_start, val_start)
    val   = [val_start,    test_start)
    test  = [test_start,   window_end)

Every event is assigned by its own timestamp, so an event exactly at a cutoff belongs
to the later split, and a session that straddles a cutoff is split across both sides.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitCutoffs:
    window_start: pd.Timestamp
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    window_end: pd.Timestamp

    def bounds(self, split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        return {
            "train": (self.window_start, self.val_start),
            "val": (self.val_start, self.test_start),
            "test": (self.test_start, self.window_end),
        }[split]

    def to_dict(self) -> dict[str, str]:
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        return {
            "window_start_utc": self.window_start.strftime(fmt),
            "val_start_utc": self.val_start.strftime(fmt),
            "test_start_utc": self.test_start.strftime(fmt),
            "window_end_utc": self.window_end.strftime(fmt),
        }


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware UTC: {ts}")
    return ts.tz_convert("UTC")


def compute_cutoffs(
    window_start, window_end, val_days: int, test_days: int, min_train_days: int = 1
) -> SplitCutoffs:
    """Place val/test as the last whole days of the window; train gets the rest."""
    start, end = _utc(window_start), _utc(window_end)
    for name, value in (("val_days", val_days), ("test_days", test_days), ("min_train_days", min_train_days)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if end <= start:
        raise ValueError("window_end must be after window_start")
    test_start = end - pd.Timedelta(days=int(test_days))
    val_start = test_start - pd.Timedelta(days=int(val_days))
    if val_start - start < pd.Timedelta(days=int(min_train_days)):
        total = (end - start) / pd.Timedelta(days=1)
        raise ValueError(
            f"window of {total:g} days is too short for val_days={val_days}, "
            f"test_days={test_days} and at least {min_train_days} training day(s)"
        )
    return SplitCutoffs(start, val_start, test_start, end)


def assign_split(timestamps: pd.Series, cutoffs: SplitCutoffs) -> pd.Series:
    """Label each timestamp train/val/test. Out-of-window timestamps are an error."""
    if len(timestamps) and timestamps.dt.tz is None:
        raise ValueError("timestamps must be timezone-aware UTC")
    outside = (timestamps < cutoffs.window_start) | (timestamps >= cutoffs.window_end)
    if outside.any():
        raise ValueError(f"{int(outside.sum())} timestamp(s) fall outside the split window")
    labels = np.select(
        [timestamps < cutoffs.val_start, timestamps < cutoffs.test_start],
        ["train", "val"],
        default="test",
    )
    return pd.Series(labels, index=timestamps.index, dtype=object)
