"""Phase 3.1 customer-level cross-fitting (synthetic-data experiments only).

One cell = one dataset x task x initialization seed:
- 3 OOF SASRec models (frozen Phase 3 configuration), each trained on the train-split
  cases of the customers outside one fold, with the same validation-only early stopping
  as Phase 3, then scoring the held-out fold's train cases;
- 1 SASRec trained on the full train split, scoring the validation cases;
- the pure evaluation in `bri.seq.hybrid.evaluate_cell`.

The same initialization seed is used for all four fits of a cell. Test cases are never
built here.
"""
from __future__ import annotations

import time

import numpy as np

from bri.eval.next_event import history_counts
from bri.seq import hybrid as H
from bri.seq.data import SeqCases
from bri.seq.sasrec import SASRecConfig
from bri.seq.train import TrainConfig, fit, score

_RT_KEYS = ("train_seconds", "epochs_run", "peak_gpu_memory_mib", "parameters")


def fold_array(tr: SeqCases) -> np.ndarray:
    fmap = H.assign_folds(tr.cases["customer_token"])
    return np.array([fmap[c] for c in tr.cases["customer_token"]], dtype=np.int64)


def pf_counts(interactions, seq: SeqCases) -> np.ndarray:
    """Personal Frequency counts (next-event grid: signal `target`), full prefix before the cutoff."""
    return history_counts(interactions, seq.cases, seq.items_vocab, "target")


def run_cell(interactions, tr: SeqCases, va: SeqCases, tiebreak: np.ndarray, frozen: dict, init_seed: int,
             device: str, log=None) -> dict:
    if "test" in set(interactions["split"]):
        raise ValueError("cross-fitting on tuning data must not see test-window events")
    H.check_candidates(tr.items_vocab, va.items_vocab)
    model_cfg = SASRecConfig(n_items=len(tr.items_vocab), variant="full", **frozen["model"])
    train_cfg = TrainConfig(**{**frozen["train"], "seed": init_seed})
    fold = fold_array(tr)
    oof = np.zeros((len(tr.cases), len(tr.items_vocab)))
    fits = []
    t0 = time.perf_counter()
    for f in range(H.FOLDS):
        res = fit(model_cfg, train_cfg, H.subset_cases(tr, fold != f), va, tiebreak, device)
        oof[fold == f] = score(res.model, H.subset_cases(tr, fold == f), device)
        fits.append({"fit": f"oof_fold{f}", "best_epoch": res.best_epoch, "best_val_ndcg@3": res.best_val,
                     **{k: res.runtime[k] for k in _RT_KEYS}})
    full = fit(model_cfg, train_cfg, tr, va, tiebreak, device)
    val_logits = score(full.model, va, device)
    fits.append({"fit": "full", "best_epoch": full.best_epoch, "best_val_ndcg@3": full.best_val,
                 **{k: full.runtime[k] for k in _RT_KEYS}})
    sasrec_seconds = time.perf_counter() - t0
    cell = H.evaluate_cell(tr, va, oof, val_logits, pf_counts(interactions, tr), pf_counts(interactions, va),
                           tiebreak, fold)
    cell.update({"init_seed": init_seed, "fits": fits, "sasrec_seconds_total": round(sasrec_seconds, 3)})
    if log:
        log(f"      init {init_seed}: status {cell['status']} (min support {cell['min_support']}), "
            f"{sasrec_seconds:.1f}s")
    return cell
