"""Shared building blocks for classical baselines.

All models consume task-scoped, consent-permitted interactions from Phase 2 prep and
produce scores over a task's candidate items. Parameters are fitted on the train split
only. At inference, a case's *history* is that customer's same-task interactions
strictly before the case cutoff, which contains no events from the evaluation window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bri.prep.examples import first_targets

# Which interactions count as a customer-item signal.
#   target     : first target event per (customer, session, item), same as evaluation relevance
#   item_event : every event that references an item (views, starts, outcomes, ...)
SIGNALS = ("target", "item_event")


def _signal_events(df: pd.DataFrame, signal: str, dedupe_key: list[str]) -> pd.DataFrame:
    if signal not in SIGNALS:
        raise ValueError(f"signal must be one of {SIGNALS}")
    if signal == "target":
        t = df[df["is_target"]]
        return t[~t.duplicated([*dedupe_key, "session_id", "item_id"])]
    return df[df["item_id"] != ""]


def user_item_matrix(train: pd.DataFrame, items: list[str], signal: str) -> np.ndarray:
    """(n_customers, n_items) interaction counts from one task's training interactions."""
    if train.empty:
        return np.zeros((0, len(items)))
    ev = _signal_events(train, signal, ["task", "customer_token"])
    ev = ev[ev["item_id"].isin(items)]
    counts = ev.groupby(["customer_token", "item_id"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=items, fill_value=0)
    customers = sorted(train["customer_token"].unique())
    return counts.reindex(customers, fill_value=0).to_numpy(dtype=float)


def history_matrix(
    interactions: pd.DataFrame, cases: pd.DataFrame, items: list[str], signal: str
) -> np.ndarray:
    """(n_cases, n_items) counts of each case's same-task history strictly before its cutoff.

    `interactions.event_timestamp` and `cases.cutoff_utc` must be UTC datetimes.
    """
    out = np.zeros((len(cases), len(items)))
    if cases.empty or interactions.empty:
        return out
    c = cases[["case_id", "task", "customer_token", "cutoff_utc"]]
    hist = interactions.merge(c, on=["task", "customer_token"])
    hist = hist[hist["event_timestamp"] < hist["cutoff_utc"]]
    ev = _signal_events(hist, signal, ["case_id"])
    ev = ev[ev["item_id"].isin(items)]
    counts = ev.groupby(["case_id", "item_id"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=cases["case_id"], columns=items, fill_value=0)
    return counts.to_numpy(dtype=float)


def to_recommendations(
    case_ids, scores: np.ndarray, items: list[str], tiebreak: np.ndarray
) -> pd.DataFrame:
    """Full ranking per case. Ties: higher score, then higher `tiebreak`, then item_id."""
    scores = np.round(np.asarray(scores, dtype=float), 10)  # remove float noise before ranking
    item_rank = np.arange(len(items))  # items are sorted, so index order == item_id order
    rows = []
    for case_id, row in zip(case_ids, scores):
        order = np.lexsort((item_rank, -tiebreak, -row))
        rows += [(case_id, items[j], r) for r, j in enumerate(order, start=1)]
    return pd.DataFrame(rows, columns=["case_id", "item_id", "rank"])
