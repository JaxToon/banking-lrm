"""Next-event evaluation cases (synthetic-data experiments only).

Distinct from the seven-day window protocol: here every eligible target event is its
own prediction case, and a case's history is everything of the same task strictly
before that event.

Definitions, applied to NBS and NBP separately (each task only sees events of its own
consent-permitted purpose):
- eligible target: a *first target* (first target-type event per task, customer,
  session and item, computed over the whole timeline) whose timestamp lies in the split
  window [start, end) and whose item is in the task candidate set. A repeated start of
  the same item later in the same session is not a new case; the same item in a later
  session is (repeat use is a legitimate next event).
- timestamp ties: eligible targets of one customer and task with the same timestamp
  form ONE case whose relevant set is all tied items. Their order is unknown, so none
  of them may appear in the other's history.
- history: the customer's same-task interactions with timestamp strictly before the
  target timestamp (the target, tied events and all later events are excluded). It may
  include earlier events of the same split, e.g. earlier test-window targets, which is
  the point of next-event evaluation; models are still fitted on train only.
- history_length: number of earlier same-task first targets (the signal models use in
  this protocol). Cases with history_length < min_history are excluded and counted as
  cold start (`insufficient_history`).
- within_session: True when the previous first target is in the same session, i.e. a
  case where an in-session transition can be observed. A session with only one eligible
  target yields one case whose history comes from earlier sessions only.

The runner fixes the model signal to `target` for this protocol: with `item_event`, the
history would contain the browse events of the very journey being predicted (e.g. the
service view seconds before the service start). Those are not future events, but they
reveal the intent and would measure journey completion rather than next-intent
prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bri.models.common import _signal_events
from bri.prep.examples import first_targets
from bri.prep.split import SplitCutoffs

NEXT_EVENT_VERSION = "0.1.0"
NE_CASE_COLUMNS = ["case_id", "split", "task", "customer_token", "cutoff_utc", "session_id",
                   "history_length", "n_relevant", "within_session"]
NE_TARGET_COLUMNS = ["case_id", "item_id"]
_TC = ["task", "customer_token"]


def build_next_event_cases(
    interactions: pd.DataFrame,
    cutoffs: SplitCutoffs,
    split: str,
    candidates: dict[str, list[str]],
    min_history: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (cases, targets, stats) for split 'val' or 'test'."""
    if split not in ("val", "test"):
        raise ValueError("evaluation split must be 'val' or 'test'")
    if min_history < 0:
        raise ValueError("min_history must be >= 0")
    start, end = cutoffs.bounds(split)
    stats = {"eligible_target_events": 0, "non_candidate_targets_dropped": 0, "tied_timestamp_merges": 0,
             "insufficient_history": 0, "cases": 0, "customers_with_cases": 0,
             "within_session_cases": 0, "cross_session_cases": 0, "single_target_sessions": 0}
    empty = pd.DataFrame(columns=NE_CASE_COLUMNS), pd.DataFrame(columns=NE_TARGET_COLUMNS), stats
    if interactions.empty:
        return empty

    # Interactions are sorted by task, customer, timestamp, event_id (see build_interactions).
    ft = first_targets(interactions)[[*_TC, "session_id", "event_timestamp", "item_id"]].reset_index(drop=True)
    pos = ft.groupby(_TC, sort=False).cumcount().to_numpy()  # rows of a customer are contiguous
    hist = pos - ft.groupby([*_TC, "event_timestamp"], sort=False).cumcount().to_numpy()
    ft["history_length"] = hist  # strictly earlier first targets only (ties excluded)
    # The latest strictly earlier first target sits hist - 1 rows after the customer's first row.
    prev = np.arange(len(ft)) - pos + hist - 1
    sess = ft["session_id"].to_numpy()
    ft["within_session"] = (hist > 0) & (sess[np.maximum(prev, 0)] == sess)

    win = ft[(ft["event_timestamp"] >= start) & (ft["event_timestamp"] < end)]
    is_cand = np.array([item in candidates[t] for t, item in zip(win["task"], win["item_id"])], dtype=bool)
    stats["non_candidate_targets_dropped"] = int((~is_cand).sum())
    win = win[is_cand]
    stats["eligible_target_events"] = int(len(win))
    if win.empty:
        return empty
    stats["single_target_sessions"] = int((win.groupby([*_TC, "session_id"]).size() == 1).sum())

    keys = [*_TC, "event_timestamp"]
    cases = win.groupby(keys, sort=False).agg(
        session_id=("session_id", "first"), history_length=("history_length", "first"),
        within_session=("within_session", "first"), n_relevant=("item_id", "nunique")).reset_index()
    stats["tied_timestamp_merges"] = int(len(win) - len(cases))
    enough = cases["history_length"] >= min_history
    stats["insufficient_history"] = int((~enough).sum())
    cases = cases[enough].sort_values(keys, kind="mergesort").reset_index(drop=True)
    if cases.empty:
        return empty
    n = cases.groupby(_TC, sort=False).cumcount() + 1
    cases["case_id"] = ("SYN_NE_" + split + "_" + cases["task"] + "_" + cases["customer_token"] + "_"
                        + n.map(lambda i: f"{i:05d}"))
    cases["split"] = split
    cases["cutoff_utc"] = cases["event_timestamp"]
    targets = win[[*keys, "item_id"]].drop_duplicates().merge(cases[[*keys, "case_id"]], on=keys)
    targets = targets[NE_TARGET_COLUMNS].sort_values(NE_TARGET_COLUMNS).reset_index(drop=True)
    stats["cases"] = int(len(cases))
    stats["customers_with_cases"] = int(cases[_TC].drop_duplicates().shape[0])
    stats["within_session_cases"] = int(cases["within_session"].sum())
    stats["cross_session_cases"] = int((~cases["within_session"]).sum())
    return cases[NE_CASE_COLUMNS], targets, stats


def history_counts(interactions: pd.DataFrame, cases: pd.DataFrame, items: list[str], signal: str) -> np.ndarray:
    """(n_cases, n_items) same-task history counts strictly before each case's cutoff.

    Same result as bri.models.common.history_matrix, computed by binary search on each
    customer's sorted signal timestamps so that many cases per customer stay cheap.
    """
    out = np.zeros((len(cases), len(items)))
    if cases.empty or interactions.empty:
        return out
    ev = _signal_events(interactions, signal, _TC)
    ev = ev[ev["item_id"].isin(items)]
    col = {item: j for j, item in enumerate(items)}
    ts_ns = ev["event_timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    ev_groups = {}
    for (task, cust, item), idx in ev.groupby([*_TC, "item_id"], sort=False).indices.items():
        ev_groups.setdefault((task, cust), []).append((col[item], np.sort(ts_ns[idx])))
    cut = cases["cutoff_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    for key, rows in cases.groupby(_TC, sort=False).indices.items():
        for j, ts in ev_groups.get(key, ()):
            out[rows, j] = np.searchsorted(ts, cut[rows], side="left")
    return out
