"""Phase 3.1 hybrid building blocks, exactly as pre-registered (phase3_1-prereg).

- score normalization without fitted parameters (Laplace PF, softmax SASRec);
- prefix-only gate features (groups P, S, H; 9 features);
- deterministic customer-level 3-fold assignment;
- soft gate labels (1 / 0.5 / 0) from NDCG@1 of OOF SASRec vs PF;
- L2 logistic gate with fractional targets, fitted by Newton's method in float64;
- static and gated fusion, support status, oracle router (diagnostic only).

Nothing here reads a target timestamp, a target session, or the evaluator's
within/cross-session label, except `oracle_router`, which is a labelled diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from bri.seq.data import SeqCases

HYBRID_VERSION = "0.1.0"
FOLDS = 3
FOLD_RNG_SEED = 3101
FEATURE_GROUPS = {
    "P": ("pf_top", "pf_entropy", "pf_margin"),
    "S": ("sas_top", "sas_entropy", "sas_margin"),
    "H": ("h_sessions", "h_last_session_steps", "h_last_session_share"),
}
FEATURES = tuple(f for g in ("P", "S", "H") for f in FEATURE_GROUPS[g])
SUBSETS = ("P", "S", "H", "P+S", "P+S+H")
C_GRID = (0.01, 0.1, 1.0, 10.0)
ALPHA_GRID = tuple(round(i / 10, 1) for i in range(11))
NEWTON_MAX_ITER = 100
NEWTON_TOL = 1e-10
VALID_SUPPORT, EXPLORATORY_SUPPORT = 50, 30
NOT_EVALUABLE = "not evaluable under the preregistered protocol"


# --- normalization ---------------------------------------------------------------------

def pf_distribution(counts: np.ndarray) -> np.ndarray:
    """(count(i) + 1) / (sum_j count(j) + |C|) per case."""
    c = np.asarray(counts, dtype=float)
    return (c + 1.0) / (c.sum(axis=1, keepdims=True) + c.shape[1])


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def static_fusion(p_pf: np.ndarray, p_sas: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * p_pf + alpha * p_sas


def gated_fusion(p_pf: np.ndarray, p_sas: np.ndarray, alpha_x: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha_x, dtype=float)[:, None]
    return (1.0 - a) * p_pf + a * p_sas


def oracle_router(p_pf: np.ndarray, p_sas: np.ndarray, within_session_label: np.ndarray) -> np.ndarray:
    """DIAGNOSTIC UPPER BOUND ONLY: reads the evaluator's within/cross-session label (target info)."""
    w = np.asarray(within_session_label, dtype=bool)[:, None]
    return np.where(w, p_sas, p_pf)


# --- features ----------------------------------------------------------------------------

def _top_entropy_margin(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    srt = np.sort(p, axis=1)[:, ::-1]
    top = srt[:, 0]
    margin = srt[:, 0] - srt[:, 1]
    ent = -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=1) / np.log(p.shape[1])
    return top, ent, margin


def history_features(seq: SeqCases) -> dict[str, np.ndarray]:
    """H features from the represented (truncated) history and its history-only boundary flags."""
    L = seq.max_len
    n = len(seq.length)
    sessions = np.zeros(n)
    last = np.zeros(n)
    share = np.zeros(n)
    for i, k in enumerate(seq.length):
        if k == 0:
            continue
        flags = seq.new_session[i, L - k:]  # earliest represented step is always 0
        sessions[i] = 1 + flags.sum()
        starts = np.flatnonzero(flags)
        last[i] = k - (starts[-1] if len(starts) else 0)
        share[i] = last[i] / k
    return {"h_sessions": sessions, "h_last_session_steps": last, "h_last_session_share": share}


def gate_features(p_pf: np.ndarray, p_sas: np.ndarray, seq: SeqCases) -> dict[str, np.ndarray]:
    """All 9 pre-registered features. Inputs are prefix-derived only."""
    pt, pe, pm = _top_entropy_margin(p_pf)
    st, se, sm = _top_entropy_margin(p_sas)
    return {"pf_top": pt, "pf_entropy": pe, "pf_margin": pm, "sas_top": st, "sas_entropy": se, "sas_margin": sm,
            **history_features(seq)}


def feature_matrix(feats: dict[str, np.ndarray], subset: str) -> np.ndarray:
    names = [f for g in subset.split("+") for f in FEATURE_GROUPS[g]]
    return np.column_stack([feats[f] for f in names])


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        sd = x.std(axis=0)
        return cls(x.mean(axis=0), np.where(sd > 0, sd, 1.0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


# --- folds -------------------------------------------------------------------------------

def assign_folds(customers, folds: int = FOLDS, seed: int = FOLD_RNG_SEED) -> dict[str, int]:
    """Sort tokens, permute with default_rng(seed), deal round-robin: customer -> fold."""
    toks = sorted(set(customers))
    order = np.random.default_rng(seed).permutation(len(toks))
    return {toks[j]: pos % folds for pos, j in enumerate(order)}


def subset_cases(seq: SeqCases, mask: np.ndarray) -> SeqCases:
    m = np.asarray(mask, dtype=bool)
    return replace(seq, cases=seq.cases[m].reset_index(drop=True), items=seq.items[m],
                   new_session=seq.new_session[m], length=seq.length[m], full_length=seq.full_length[m],
                   targets=seq.targets[m])


# --- gate labels, support and fitting -------------------------------------------------------

def soft_labels(ndcg1_sas: np.ndarray, ndcg1_pf: np.ndarray) -> np.ndarray:
    a, b = np.asarray(ndcg1_sas, dtype=float), np.asarray(ndcg1_pf, dtype=float)
    return np.where(a > b, 1.0, np.where(a < b, 0.0, 0.5))


def label_counts(y: np.ndarray) -> dict[str, int]:
    return {"y0": int((y == 0.0).sum()), "y05": int((y == 0.5).sum()), "y1": int((y == 1.0).sum())}


def support_status(per_fold_counts: list[dict]) -> tuple[str, int]:
    """Status from min over folds of min(#y=0, #y=1); ties (y=0.5) never count as a class."""
    m = min(min(c["y0"], c["y1"]) for c in per_fold_counts)
    if m >= VALID_SUPPORT:
        return "valid", m
    if m >= EXPLORATORY_SUPPORT:
        return "exploratory", m
    return "rejected", m


@dataclass(frozen=True)
class LogisticGate:
    w: np.ndarray
    b: float
    iterations: int
    converged: bool

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(x @ self.w + self.b)))


def fit_logistic_gate(x: np.ndarray, y: np.ndarray, c: float, max_iter: int = NEWTON_MAX_ITER,
                      tol: float = NEWTON_TOL) -> LogisticGate:
    """minimize sum BCE(y, sigmoid(Xw + b)) + ||w||^2 / (2C); fractional y allowed; bias unregularized.

    Newton's method in float64 from w = 0, b = 0; stops when max |gradient| <= tol.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if c <= 0 or not ((y >= 0) & (y <= 1)).all():
        raise ValueError("C must be > 0 and targets in [0, 1]")
    n, d = x.shape
    xa = np.column_stack([x, np.ones(n)])
    reg = np.full(d + 1, 1.0 / c)
    reg[-1] = 0.0
    theta = np.zeros(d + 1)
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        p = 1.0 / (1.0 + np.exp(-(xa @ theta)))
        grad = xa.T @ (p - y) + reg * theta
        if np.max(np.abs(grad)) <= tol:
            converged = True
            it -= 1
            break
        hess = (xa * (p * (1 - p))[:, None]).T @ xa + np.diag(reg)
        theta = theta - np.linalg.solve(hess, grad)
    else:
        p = 1.0 / (1.0 + np.exp(-(xa @ theta)))
        converged = bool(np.max(np.abs(xa.T @ (p - y) + reg * theta)) <= tol)
    return LogisticGate(theta[:-1], float(theta[-1]), it, converged)


def gate_objective(gate: LogisticGate, x: np.ndarray, y: np.ndarray, c: float) -> float:
    p = np.clip(gate.predict(x), 1e-15, 1 - 1e-15)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum() + (gate.w @ gate.w) / (2 * c))


# --- per-cell evaluation (pure; SASRec logits are computed by bri.seq.crossfit) -------------

def check_candidates(*vocabs) -> list[str]:
    """Identical candidate sets in identical sorted order for every model and split (asserted)."""
    first = list(vocabs[0])
    if first != sorted(first) or len(set(first)) != len(first):
        raise ValueError("candidate set must be unique and sorted by item id")
    for v in vocabs[1:]:
        if list(v) != first:
            raise ValueError("candidate sets differ between models or splits")
    return first


def ndcg1(seq: SeqCases, scores: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Per-case NDCG@1 from the shared evaluator (shared tie-break: train popularity, then item id)."""
    import pandas as pd

    from bri.eval.evaluate import per_case_metrics
    from bri.models.common import to_recommendations
    recs = to_recommendations(seq.cases["case_id"], scores, seq.items_vocab, tiebreak)
    tg = pd.DataFrame([(c, seq.items_vocab[j]) for c, r in zip(seq.cases["case_id"], seq.targets)
                       for j in np.flatnonzero(r)], columns=["case_id", "item_id"])
    pc = per_case_metrics(seq.cases, tg, recs, {seq.task: seq.items_vocab}, 1).set_index("case_id")
    return pc["ndcg"].reindex(seq.cases["case_id"]).to_numpy(dtype=float)


def distribution_summary(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    q = np.percentile(a, [5, 50, 95])
    return {"mean": float(a.mean()), "sd": float(a.std()), "p5": float(q[0]), "p50": float(q[1]), "p95": float(q[2])}


def gate_key(subset: str, c: float) -> str:
    return f"{subset}|C={c:g}"


def evaluate_cell(tr: SeqCases, va: SeqCases, oof_logits: np.ndarray, val_logits: np.ndarray,
                  pf_counts_tr: np.ndarray, pf_counts_va: np.ndarray, tiebreak: np.ndarray,
                  fold: np.ndarray) -> dict:
    """One dataset x task x initialization: labels, support, features, gates, validation NDCG@1.

    Gate inputs are prefix-only: PF counts, SASRec logits and the represented history.
    `within_session` is read only for the train diagnostic counts and the oracle router.
    """
    check_candidates(tr.items_vocab, va.items_vocab)
    if oof_logits.shape != pf_counts_tr.shape or val_logits.shape != pf_counts_va.shape:
        raise ValueError("score matrices must be (cases, candidates) for both models")
    p_pf_tr, p_pf_va = pf_distribution(pf_counts_tr), pf_distribution(pf_counts_va)
    p_sas_tr, p_sas_va = softmax(oof_logits), softmax(val_logits)

    y = soft_labels(ndcg1(tr, p_sas_tr, tiebreak), ndcg1(tr, p_pf_tr, tiebreak))
    within_tr = tr.cases["within_session"].to_numpy(dtype=bool)
    per_fold = [{"fold": f, **label_counts(y[fold == f]), "within_session": int(within_tr[fold == f].sum()),
                 "cross_session": int((~within_tr[fold == f]).sum())} for f in range(FOLDS)]
    status, min_support = support_status(per_fold)

    f_tr, f_va = gate_features(p_pf_tr, p_sas_tr, tr), gate_features(p_pf_va, p_sas_va, va)
    full_tr = feature_matrix(f_tr, "P+S+H")
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(full_tr, rowvar=False) if len(full_tr) > 1 else None

    val = {"pf": ndcg1(va, p_pf_va, tiebreak), "sasrec": ndcg1(va, p_sas_va, tiebreak),
           "oracle_router": ndcg1(va, oracle_router(p_pf_va, p_sas_va, va.cases["within_session"]), tiebreak),
           "static": {a: ndcg1(va, static_fusion(p_pf_va, p_sas_va, a), tiebreak) for a in ALPHA_GRID},
           "gated": {}}
    gates = {}
    for subset in SUBSETS:
        x_tr, x_va = feature_matrix(f_tr, subset), feature_matrix(f_va, subset)
        std = Standardizer.fit(x_tr)  # train-split OOF features only
        for c in C_GRID:
            key = gate_key(subset, c)
            if status == "rejected":  # not fitted, not used; never replaced by static fusion
                val["gated"][key], gates[key] = None, {"status": NOT_EVALUABLE}
                continue
            g = fit_logistic_gate(std.transform(x_tr), y, c)
            a_va = g.predict(std.transform(x_va))
            val["gated"][key] = ndcg1(va, gated_fusion(p_pf_va, p_sas_va, a_va), tiebreak)
            gates[key] = {"w": g.w.tolist(), "b": g.b, "iterations": g.iterations, "converged": g.converged,
                          "alpha_val": distribution_summary(a_va)}
    return {"status": status, "min_support": min_support, "per_fold": per_fold,
            "n_train_cases": len(y), "n_val_cases": len(va.cases),
            "feature_correlation_train": {"features": list(FEATURES),
                                          "matrix": None if corr is None else np.round(corr, 6).tolist()},
            "gates": gates, "val_case_ids": va.cases["case_id"].tolist(), "val_ndcg1": val}


def cell_means(cell: dict) -> dict:
    """Mean validation NDCG@1 per candidate; None stays None (never zero)."""
    v = cell["val_ndcg1"]

    def m(a):
        return None if a is None else float(np.mean(a))
    return {"pf": m(v["pf"]), "sasrec": m(v["sasrec"]), "oracle_router": m(v["oracle_router"]),
            "static": {a: m(x) for a, x in v["static"].items()}, "gated": {k: m(x) for k, x in v["gated"].items()}}


# --- selection on tuning validation (decisions 10, 12, 14) --------------------------------

def select(cells: list[dict], profiles: list[str], tasks: list[str]) -> dict:
    """Pre-registered selection. `cells`: {profile, seed, task, init, status, means (cell_means)}.

    Dataset value = mean over initializations (equal to the mean of per-case initialization
    means, since every initialization scores the same cases); group value = mean over the
    group's tuning datasets; objective = equal-weight mean over groups. Grid ties keep the
    earlier entry. A group with any unavailable value is unavailable, never zero.
    """
    def group(task, profile, get):
        per_seed = {}
        for c in cells:
            if c["task"] == task and c["profile"] == profile:
                per_seed.setdefault(c["seed"], []).append(get(c["means"]))
        if not per_seed or any(v is None for vs in per_seed.values() for v in vs):
            return None
        return float(np.mean([np.mean(vs) for vs in per_seed.values()]))

    def macro(task, get):
        g = {p: group(task, p, get) for p in profiles}
        return (None if any(v is None for v in g.values()) else float(np.mean(list(g.values())))), g

    def argmax(scores: dict):
        best = None
        for k, s in scores.items():
            if s is not None and (best is None or s > scores[best]):
                best = k
        return best

    out = {"per_task": {}}
    for task in tasks:
        st = {a: macro(task, lambda m, a=a: m["static"][a])[0] for a in ALPHA_GRID}
        gt = {gate_key(s, c): macro(task, lambda m, k=gate_key(s, c): m["gated"][k])[0]
              for s in SUBSETS for c in C_GRID}
        a_star = argmax(st)
        evaluable = all(v is not None for v in gt.values())
        g_star = argmax(gt) if evaluable else None
        out["per_task"][task] = {
            "static_grid": {str(a): v for a, v in st.items()}, "alpha": a_star,
            "static_groups": macro(task, lambda m: m["static"][a_star])[1],
            "gated_grid": gt, "gated_evaluable": evaluable, "gate": g_star,
            "gated_groups": macro(task, lambda m: m["gated"][g_star])[1] if evaluable else None,
            "pf_groups": macro(task, lambda m: m["pf"])[1], "sasrec_groups": macro(task, lambda m: m["sasrec"])[1],
            "oracle_router_groups_diagnostic": macro(task, lambda m: m["oracle_router"])[1]}
    statuses = [c["status"] for c in cells]
    all_valid = bool(cells) and all(s == "valid" for s in statuses)
    groups = [(t, p) for t in tasks for p in profiles]
    delta = None
    if all(out["per_task"][t]["gated_evaluable"] for t in tasks):
        delta = float(np.mean([out["per_task"][t]["gated_groups"][p] - out["per_task"][t]["static_groups"][p]
                               for t, p in groups]))
    out.update({
        "n_cells": len(cells), "status_counts": {s: statuses.count(s) for s in ("valid", "exploratory", "rejected")},
        "all_gates_valid": all_valid, "delta_val": delta, "n_groups": len(groups),
        "primary_hybrid": "gated" if all_valid and delta is not None and delta > 0 else "static",
        "note": "Choosing gated on tuning validation is not evidence for RQ3."})
    return out
