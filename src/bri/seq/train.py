"""Training, scoring, evaluation and checkpoints for Phase 3 sequential models.

- Fit on TRAIN-window sequence cases only; early stopping and model selection use the
  VALIDATION-window cases only. `fit` has no parameter through which test data could
  enter.
- Validation metrics go through the shared evaluator (bri.eval.evaluate via
  to_recommendations + per_case_metrics) with the shared deterministic tie-break
  (train popularity, then item id), exactly as for the classical models.
- Every random choice (initialization, dropout, batch order) derives from one seed.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bri.eval.evaluate import per_case_metrics
from bri.models.baselines import Popularity
from bri.models.common import to_recommendations, user_item_matrix
from bri.seq import runtime as R
from bri.seq.data import SeqCases
from bri.seq.sasrec import (MODEL_VERSION, UNUSED_PARAMS, SASRec, SASRecConfig, count_parameters,
                            multi_positive_loss, shuffle_history)

TRAIN_VERSION = "0.1.0"


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 5  # epochs without strict improvement of the selection metric
    grad_clip: float = 1.0
    seed: int = 0
    select_metric: str = "ndcg"
    select_k: int = 3
    early_stopping: bool = True

    def __post_init__(self) -> None:
        if self.lr <= 0 or self.weight_decay < 0 or self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("invalid training configuration")
        if self.patience < 1 or self.grad_clip <= 0:
            raise ValueError("patience must be >= 1 and grad_clip > 0")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FitResult:
    model: SASRec
    best_epoch: int
    best_val: float
    history: list[dict] = field(default_factory=list)
    runtime: dict = field(default_factory=dict)


def train_popularity(interactions: pd.DataFrame, task: str, items: list[str]) -> np.ndarray:
    """Shared tie-break: distinct train customers per item (target signal), as for classical models."""
    train = interactions[(interactions["task"] == task) & (interactions["split"] == "train")]
    return Popularity().fit(user_item_matrix(train, items, "target")).popularity_


SHUFFLE_SEED_OFFSET = 104_729  # shuffled variant: its own generator, independent of batch order


def _tensors(seq: SeqCases, device: str):
    return (torch.as_tensor(seq.items, device=device), torch.as_tensor(seq.new_session, device=device),
            torch.as_tensor(seq.targets, device=device))


@torch.no_grad()
def score(model: SASRec, seq: SeqCases, device: str, batch_size: int = 1024) -> np.ndarray:
    """(n_cases, n_items) scores, eval mode, chronological input (never shuffled).

    Variant-agnostic: every model receives the same inputs (items and historical
    new_session flags) and ignores what it does not use. Targets are never read.
    """
    model.eval()
    items = torch.as_tensor(seq.items, device=device)
    flags = torch.as_tensor(seq.new_session, device=device)
    out = [model(items[i:i + batch_size], flags[i:i + batch_size]).float().cpu()
           for i in range(0, len(items), batch_size)]
    return torch.cat(out).numpy() if out else np.zeros((0, len(seq.items_vocab)), dtype=np.float32)


def targets_frame(seq: SeqCases) -> pd.DataFrame:
    rows = [(c, seq.items_vocab[j]) for c, r in zip(seq.cases["case_id"], seq.targets) for j in np.flatnonzero(r)]
    return pd.DataFrame(rows, columns=["case_id", "item_id"])


def per_case(seq: SeqCases, scores: np.ndarray, tiebreak: np.ndarray, k: int) -> pd.DataFrame:
    """Shared-evaluator per-case metrics at K for this task's cases."""
    recs = to_recommendations(seq.cases["case_id"], scores, seq.items_vocab, tiebreak)
    return per_case_metrics(seq.cases, targets_frame(seq), recs, {seq.task: seq.items_vocab}, k)


def fit(model_cfg: SASRecConfig, train_cfg: TrainConfig, train: SeqCases, val: SeqCases,
        tiebreak: np.ndarray, device: str = "cpu", log=None) -> FitResult:
    """Train on `train`, select the epoch with the best validation metric (early stopping)."""
    if train.task != val.task or train.split != "train" or val.split != "val":
        raise ValueError("fit needs train-window cases for training and val-window cases for selection")
    if train.max_len != model_cfg.max_len or val.max_len != model_cfg.max_len:
        raise ValueError("sequence max_len must match the model configuration")
    R.seed_everything(train_cfg.seed)
    model = SASRec(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    gen = torch.Generator().manual_seed(train_cfg.seed)  # batch order, on CPU for portability
    # Shuffled variant only: separate CPU generator, so batch order, initialization and
    # dropout draws are identical to Full for the same seed; only the history order differs.
    shuffle_gen = torch.Generator().manual_seed(train_cfg.seed + SHUFFLE_SEED_OFFSET)
    items, flags, targets = _tensors(train, device)
    n = len(items)
    R.reset_peak_memory(device)
    best = (-1.0, 0, None)
    history, bad, t0, eval_s = [], 0, time.perf_counter(), 0.0
    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        order = torch.randperm(n, generator=gen).to(device)
        total = torch.zeros((), device=device)  # accumulated on device: no per-batch host sync
        for i in range(0, n, train_cfg.batch_size):
            idx = order[i:i + train_cfg.batch_size]
            batch_items = items[idx]
            if model_cfg.shuffles_training_history:
                batch_items = shuffle_history(batch_items, shuffle_gen)
            loss = multi_positive_loss(model(batch_items, flags[idx]), targets[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            total += loss.detach() * len(idx)
        total = float(total)
        if not np.isfinite(total):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        te = time.perf_counter()
        pc = per_case(val, score(model, val, device), tiebreak, train_cfg.select_k)
        eval_s += time.perf_counter() - te
        val_metric = round(float(pc[train_cfg.select_metric].mean()), 6)
        history.append({"epoch": epoch, "train_loss": round(total / n, 6),
                        f"val_{train_cfg.select_metric}@{train_cfg.select_k}": val_metric})
        if log:
            log(history[-1])
        if val_metric > best[0]:
            best, bad = (val_metric, epoch, copy.deepcopy(model.state_dict())), 0
        else:
            bad += 1
            if train_cfg.early_stopping and bad >= train_cfg.patience:
                break
    train_seconds = time.perf_counter() - t0
    model.load_state_dict(best[2])
    t1 = time.perf_counter()
    score(model, val, device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    infer = time.perf_counter() - t1
    runtime = {"device": device, "train_seconds": round(train_seconds, 3), "epochs_run": len(history),
               "seconds_per_epoch": round(train_seconds / len(history), 4),
               "val_eval_seconds": round(eval_s, 3),  # shared-evaluator time inside train_seconds
               "optimization_seconds": round(train_seconds - eval_s, 3),
               "inference_seconds_val": round(infer, 4),
               "inference_ms_per_case": round(1000 * infer / max(len(val.cases), 1), 5),
               "peak_gpu_memory_mib": R.peak_memory_mib(device),
               "parameters": count_parameters(model),
               "n_train_cases": n, "n_val_cases": len(val.cases)}
    return FitResult(model, best[1], best[0], history, runtime)


# --- checkpoints ---------------------------------------------------------------------

def state_dict_sha256(model: SASRec) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def vocab_fingerprint(vocab: list[str]) -> str:
    return hashlib.sha256("\n".join(vocab).encode()).hexdigest()


def checkpoint_metadata(result: FitResult, train_cfg: TrainConfig, train: SeqCases, val: SeqCases,
                        device: str) -> dict:
    return {
        "model_version": MODEL_VERSION, "train_version": TRAIN_VERSION,
        "variant": result.model.cfg.variant, "architecture": result.model.cfg.to_dict(),
        "unused_hyperparameters": list(UNUSED_PARAMS[result.model.cfg.variant]),
        "hyperparameters": train_cfg.to_dict(), "training_seed": train_cfg.seed,
        "task": train.task, "item_vocab": train.items_vocab,
        "item_mapping_sha256": vocab_fingerprint(train.items_vocab),
        "dataset_fingerprints": {"train": train.fingerprint(), "val": val.fingerprint()},
        "best_epoch": result.best_epoch, "best_val": result.best_val,
        "torch": torch.__version__, "device": device,
        "state_dict_sha256": state_dict_sha256(result.model),
    }


def save_checkpoint(path: Path, model: SASRec, metadata: dict) -> int:
    """Write {state_dict, metadata JSON}. Returns the file size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metadata_json": json.dumps(metadata, sort_keys=True)}, buf)
    path.write_bytes(buf.getvalue())
    return path.stat().st_size


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[SASRec, dict]:
    blob = torch.load(Path(path), map_location=device, weights_only=True)
    meta = json.loads(blob["metadata_json"])
    model = SASRec(SASRecConfig(**meta["architecture"])).to(device)
    model.load_state_dict(blob["state_dict"])
    if state_dict_sha256(model) != meta["state_dict_sha256"]:
        raise ValueError(f"checkpoint {path} does not match its recorded state_dict sha256")
    model.eval()
    return model, meta
