"""Sequential cases for Phase 3 (synthetic-data experiments only). numpy/pandas only.

Cases are exactly the Phase 2.3 next-event cases (bri.eval.next_event), built for the
train, val and test windows with the same rules, so training and evaluation share one
case protocol. For each case this module encodes the history as a sequence of STEPS:

- A step is one timestamp in the customer's same-task first-target history (the
  `target` signal used by the next-event protocol). First targets with the same
  timestamp form ONE step holding all tied items; they are never given an order.
- History = steps strictly before the case's target timestamp. The target step, tied
  target items and every later event are excluded.
- Only candidate items of the task enter the sequence (as for the classical models'
  history counts). NBS and NBP histories are built separately and never mixed.
- Truncation keeps the most recent `max_len` steps. Sequences are LEFT-padded: the
  last array position is always the most recent step of a non-empty history.
- `new_session[i]` = 1 when step i starts a session different from the previous
  step OF THE REPRESENTED (possibly truncated) HISTORY. The earliest represented step
  is always 0: there is no earlier step in the representation to prove a boundary.
  Flags come from the historical prefix only; the target's session is never read, so
  whether the target is within- or cross-session cannot leak into the input.

Arrays: items (n_cases, max_len, max_tie) int64 with 0 = padding and item index + 1
otherwise; new_session (n_cases, max_len) int8; length (n_cases,) steps kept;
full_length (n_cases,) steps before truncation; targets (n_cases, n_items) multi-hot.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bri.eval.next_event import build_next_event_cases
from bri.prep.examples import first_targets
from bri.prep.split import SplitCutoffs

SEQ_DATA_VERSION = "0.2.0"  # 0.2.0: boundary flags on the represented history, earliest step 0
_TC = ["task", "customer_token"]


@dataclass
class SeqCases:
    task: str
    split: str
    items_vocab: list[str]  # index j <-> item id; encoded as j + 1 (0 = padding)
    max_len: int
    cases: pd.DataFrame  # Phase 2.3 next-event case rows (case_id order = array order)
    items: np.ndarray
    new_session: np.ndarray
    length: np.ndarray
    full_length: np.ndarray
    targets: np.ndarray

    def fingerprint(self) -> str:
        """sha256 over the encoded arrays, case ids, vocabulary and max_len."""
        h = hashlib.sha256()
        h.update(f"{SEQ_DATA_VERSION}|{self.task}|{self.split}|{self.max_len}|{','.join(self.items_vocab)}".encode())
        h.update("\n".join(self.cases["case_id"]).encode())
        for a in (self.items, self.new_session, self.length, self.full_length, self.targets):
            h.update(np.ascontiguousarray(a).tobytes())
            h.update(str(a.shape).encode())
        return h.hexdigest()


def history_steps(interactions: pd.DataFrame, candidates: dict[str, list[str]]) -> dict:
    """(task, customer) -> (step timestamps int64 ns ascending, [item tuples], [session tuples])."""
    ft = first_targets(interactions)
    ft = ft[[item in set(candidates[t]) for t, item in zip(ft["task"], ft["item_id"])]]
    out = {}
    for key, g in ft.groupby(_TC, sort=False):
        ts = g["event_timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        uniq, start = np.unique(ts, return_index=True)  # g is time-sorted; np.unique sorts too
        bounds = list(start) + [len(g)]
        items = g["item_id"].to_numpy()
        sess = g["session_id"].to_numpy()
        out[key] = (uniq, [tuple(sorted(set(items[a:b]))) for a, b in zip(bounds, bounds[1:])],
                    [tuple(sorted(set(sess[a:b]))) for a, b in zip(bounds, bounds[1:])])
    return out


def encode_cases(cases: pd.DataFrame, targets: pd.DataFrame, steps: dict, task: str, split: str,
                 vocab: list[str], max_len: int) -> SeqCases:
    if max_len < 1:
        raise ValueError("max_len must be >= 1")
    cases = cases[(cases["task"] == task) & (cases["split"] == split)].sort_values("case_id").reset_index(drop=True)
    col = {item: j for j, item in enumerate(vocab)}
    n = len(cases)
    cut = cases["cutoff_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64) if n else np.zeros(0, np.int64)
    rows = []
    max_tie = 1
    for i, cust in enumerate(cases["customer_token"]):
        ts, items, sess = steps.get((task, cust), (np.zeros(0, np.int64), [], []))
        k = int(np.searchsorted(ts, cut[i], side="left"))  # steps strictly before the target
        lo = max(0, k - max_len)
        # Boundaries within the represented steps lo..k-1 only; the earliest one is 0.
        flags = [0] + [int(not set(sess[s]) & set(sess[s - 1])) for s in range(lo + 1, k)] if k > lo else []
        rows.append((items[lo:k], flags, k))
        max_tie = max([max_tie] + [len(t) for t in items[lo:k]])
    arr = np.zeros((n, max_len, max_tie), dtype=np.int64)
    new_session = np.zeros((n, max_len), dtype=np.int8)
    length = np.zeros(n, dtype=np.int64)
    full_length = np.zeros(n, dtype=np.int64)
    for i, (kept, flags, k) in enumerate(rows):
        off = max_len - len(kept)
        for p, step in enumerate(kept):
            arr[i, off + p, :len(step)] = [col[it] + 1 for it in step]
        new_session[i, off:] = flags
        length[i], full_length[i] = len(kept), k
    tgt = np.zeros((n, len(vocab)), dtype=np.float32)
    pos = {c: i for i, c in enumerate(cases["case_id"])}
    for c, item in zip(targets["case_id"], targets["item_id"]):
        if c in pos:
            tgt[pos[c], col[item]] = 1.0
    if n and (tgt.sum(axis=1) == 0).any():
        raise ValueError("every case needs at least one candidate target")
    return SeqCases(task, split, list(vocab), max_len, cases, arr, new_session, length, full_length, tgt)


def build_seq_cases(interactions: pd.DataFrame, cutoffs: SplitCutoffs, candidates: dict[str, list[str]],
                    splits: tuple[str, ...], max_len: int, min_history: int = 1) -> dict[tuple[str, str], SeqCases]:
    """{(task, split): SeqCases} for the requested splits only (nothing else is built)."""
    steps = history_steps(interactions, candidates)
    out = {}
    for split in splits:
        cases, targets, _ = build_next_event_cases(interactions, cutoffs, split, candidates, min_history)
        for task, vocab in candidates.items():
            out[(task, split)] = encode_cases(cases, targets, steps, task, split, vocab, max_len)
    return out


def length_stats(full_length: np.ndarray, candidate_lens=(10, 20, 50)) -> dict:
    """Distribution of history lengths (in steps) and truncation rates per candidate max_len."""
    x = np.asarray(full_length, dtype=float)
    if len(x) == 0:
        return {"n_cases": 0}
    q = np.percentile(x, [50, 75, 90, 95])
    total = x.sum()
    return {
        "n_cases": int(len(x)),
        "p50": float(q[0]), "p75": float(q[1]), "p90": float(q[2]), "p95": float(q[3]),
        "max": int(x.max()), "mean": round(float(x.mean()), 3),
        "share_exceeding": {str(L): round(float((x > L).mean()), 6) for L in candidate_lens},
        # share of all history steps dropped by truncation at L
        "steps_truncated": {str(L): round(float(np.clip(x - L, 0, None).sum() / total), 6) if total else 0.0
                            for L in candidate_lens},
    }
