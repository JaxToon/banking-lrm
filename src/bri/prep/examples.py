"""Task-scoped interactions, training examples, evaluation cases, and negatives.

Definitions (see docs/08_PHASE_2_FOUNDATIONS_REPORT.md):
- interaction: a consent-permitted event, assigned to the task bound to its purpose.
- target: the first event of a task's target types per (task, customer, session, item)
  inside the period being labelled.
- training example: one train-split target with its count of strictly earlier
  same-task interactions (history_length) inside the train split.
- evaluation case: one (split, task, customer) whose relevant set is the distinct target
  items in that split's window and whose history is every same-task interaction strictly
  before the window start.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bri.prep.split import SplitCutoffs, assign_split
from bri.tasks import PURPOSE_TO_TASK, TASKS

INTERACTION_COLUMNS = [
    "event_id", "task", "split", "customer_token", "session_id",
    "event_timestamp", "event_type", "item_id", "is_target",
]
EXAMPLE_COLUMNS = [
    "example_id", "task", "customer_token", "target_event_id",
    "target_timestamp", "target_item_id", "history_length",
]
CASE_COLUMNS = ["case_id", "split", "task", "customer_token", "cutoff_utc", "history_length", "n_relevant"]
TARGET_COLUMNS = ["case_id", "item_id"]
_TC = ["task", "customer_token"]


def candidate_sets(catalog: pd.DataFrame) -> dict[str, list[str]]:
    """Candidate set per task = all active catalog items of the task's item_type, sorted."""
    active = catalog[catalog["active_status"].astype(str) == "True"]
    return {
        name: sorted(active.loc[active["item_type"] == t.item_type, "item_id"])
        for name, t in TASKS.items()
    }


def build_interactions(permitted: pd.DataFrame, cutoffs: SplitCutoffs) -> pd.DataFrame:
    """Attach task, split and is_target to consent-permitted events."""
    if permitted.empty:
        return pd.DataFrame(columns=INTERACTION_COLUMNS)
    df = permitted.copy()
    df["task"] = df["purpose"].map(PURPOSE_TO_TASK)
    if df["task"].isna().any():
        raise ValueError("permitted events contain a purpose without a task; filter consent first")
    df["split"] = assign_split(df["event_timestamp"], cutoffs)
    targets = {name: t.target_event_types for name, t in TASKS.items()}
    df["is_target"] = [et in targets[task] for et, task in zip(df["event_type"], df["task"])]
    df = df.sort_values(["task", "customer_token", "event_timestamp", "event_id"], kind="mergesort")
    return df[INTERACTION_COLUMNS].reset_index(drop=True)


def _first_targets(df: pd.DataFrame) -> pd.DataFrame:
    t = df[df["is_target"]]
    return t[~t.duplicated(["task", "customer_token", "session_id", "item_id"])]


def build_train_examples(interactions: pd.DataFrame, min_history: int = 1) -> tuple[pd.DataFrame, int]:
    """Return (examples, n_dropped_for_insufficient_history). Uses the train split only."""
    if min_history < 0:
        raise ValueError("min_history must be >= 0")
    train = interactions[interactions["split"] == "train"]
    if train.empty:
        return pd.DataFrame(columns=EXAMPLE_COLUMNS), 0
    # interactions are sorted by task, customer, timestamp, event_id
    prior = train.groupby(_TC, sort=False).cumcount()
    ties = train.groupby([*_TC, "event_timestamp"], sort=False).cumcount()
    train = train.assign(history_length=prior - ties)  # strictly earlier events only
    targets = _first_targets(train)
    enough = targets["history_length"] >= min_history
    ex = targets[enough]
    out = pd.DataFrame({
        "example_id": [f"SYN_EX_{i + 1:09d}" for i in range(len(ex))],
        "task": ex["task"].to_numpy(),
        "customer_token": ex["customer_token"].to_numpy(),
        "target_event_id": ex["event_id"].to_numpy(),
        "target_timestamp": ex["event_timestamp"].to_numpy(),
        "target_item_id": ex["item_id"].to_numpy(),
        "history_length": ex["history_length"].to_numpy(),
    })
    return out, int((~enough).sum())


def build_eval_cases(
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
    stats = {"customers_with_targets": 0, "insufficient_history": 0, "non_candidate_targets_dropped": 0}
    empty = pd.DataFrame(columns=CASE_COLUMNS), pd.DataFrame(columns=TARGET_COLUMNS), stats
    if interactions.empty:
        return empty

    ts = interactions["event_timestamp"]
    history = interactions[ts < start].groupby(_TC).size().rename("history_length")
    window = interactions[(ts >= start) & (ts < end)]
    rel = _first_targets(window)[[*_TC, "item_id"]].drop_duplicates()
    is_candidate = [item in candidates[task] for task, item in zip(rel["task"], rel["item_id"])]
    stats["non_candidate_targets_dropped"] = int(len(rel) - sum(is_candidate))
    rel = rel[is_candidate]
    if rel.empty:
        return empty

    cases = rel.groupby(_TC).size().rename("n_relevant").reset_index()
    cases = cases.join(history, on=_TC)
    cases["history_length"] = cases["history_length"].fillna(0).astype(int)
    stats["customers_with_targets"] = int(len(cases))
    enough = cases["history_length"] >= min_history
    stats["insufficient_history"] = int((~enough).sum())
    cases = cases[enough].sort_values(_TC).reset_index(drop=True)
    cases["case_id"] = "SYN_CASE_" + split + "_" + cases["task"] + "_" + cases["customer_token"]
    cases["split"] = split
    cases["cutoff_utc"] = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    targets = rel.merge(cases[[*_TC, "case_id"]], on=_TC)[TARGET_COLUMNS]
    targets = targets.sort_values(TARGET_COLUMNS).reset_index(drop=True)
    return cases[CASE_COLUMNS], targets, stats


def sample_negatives(
    examples: pd.DataFrame, candidates: dict[str, list[str]], n_negatives: int, seed: int
) -> pd.DataFrame:
    """Uniformly sample up to n_negatives non-target candidates per training example.

    Negatives are drawn without replacement from the task's candidate set minus the
    example's target item. They are *unobserved*, not confirmed dislikes. Deterministic
    for the same examples (in the same order), candidates, n_negatives, and seed.
    """
    if n_negatives < 1:
        raise ValueError("n_negatives must be >= 1")
    rng = np.random.default_rng(seed)
    ex_ids, items = [], []
    for ex_id, task, target in zip(examples["example_id"], examples["task"], examples["target_item_id"]):
        pool = [c for c in candidates[task] if c != target]
        k = min(n_negatives, len(pool))
        for idx in rng.choice(len(pool), size=k, replace=False):
            ex_ids.append(ex_id)
            items.append(pool[int(idx)])
    return pd.DataFrame({"example_id": ex_ids, "item_id": items})
