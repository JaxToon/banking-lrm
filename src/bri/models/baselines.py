"""Classical recommendation baselines: popularity, item-item CF, implicit-feedback MF.

Each model implements:
    fit(train_matrix)          train_matrix: (n_customers, n_items) counts from the train split
    score(history_matrix)      history_matrix: (n_cases, n_items) counts before each cutoff
and is deterministic for the same inputs and seed. Only numpy is used: the candidate
sets have 5-6 items, so dense matrices are small even for large customer counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Popularity:
    """Score = number of distinct training customers who interacted with the item.

    The same list is given to everyone, which makes it a non-personalized reference.
    """
    signal: str = "target"
    popularity_: np.ndarray = field(default=None, init=False, repr=False)

    def fit(self, train: np.ndarray) -> "Popularity":
        self.popularity_ = (train > 0).sum(axis=0).astype(float)
        return self

    def score(self, history: np.ndarray) -> np.ndarray:
        return np.tile(self.popularity_, (len(history), 1))


@dataclass
class ItemKNN:
    """Item-item collaborative filtering with cosine similarity on binary train usage.

    score(case, j) = sum_i log1p(history_count_i) * sim(i, j).
    include_self=True keeps sim(i, i) = 1, so items the customer already uses rank high
    (repeat use is common for banking services); False recommends only by co-usage.
    """
    signal: str = "target"
    include_self: bool = True
    similarity_: np.ndarray = field(default=None, init=False, repr=False)

    def fit(self, train: np.ndarray) -> "ItemKNN":
        b = (train > 0).astype(float)
        co = b.T @ b
        norms = np.sqrt(np.diag(co))
        denom = np.outer(norms, norms)
        sim = np.divide(co, denom, out=np.zeros_like(co), where=denom > 0)
        if not self.include_self:
            np.fill_diagonal(sim, 0.0)
        self.similarity_ = sim
        return self

    def score(self, history: np.ndarray) -> np.ndarray:
        return np.log1p(history) @ self.similarity_


def _als_solve(fixed: np.ndarray, conf: np.ndarray, pref: np.ndarray, reg: float) -> np.ndarray:
    """Solve every row's least-squares problem of implicit ALS in one batched call.

    For row r: x_r = (F^T C_r F + reg*I)^-1 F^T C_r p_r, with F = fixed factors.
    """
    f = fixed.shape[1]
    a = np.einsum("if,ri,ig->rfg", fixed, conf, fixed) + reg * np.eye(f)
    b = np.einsum("if,ri->rf", fixed, conf * pref)
    return np.linalg.solve(a, b[..., None])[..., 0]


@dataclass
class ImplicitMF:
    """Matrix factorization for implicit feedback, fitted by alternating least squares
    (Hu, Koren & Volinsky, 2008): preference p = 1[count > 0], confidence c = 1 + alpha*count.

    Item factors are learned on train. A case's customer factor is *folded in* from that
    case's history with item factors fixed, so customers need not exist in train and
    pre-cutoff history (including validation events for test cases) is used without
    refitting item parameters.
    """
    signal: str = "target"
    factors: int = 2
    reg: float = 0.1
    alpha: float = 10.0
    iterations: int = 15
    seed: int = 0
    item_factors_: np.ndarray = field(default=None, init=False, repr=False)

    def fit(self, train: np.ndarray) -> "ImplicitMF":
        if self.factors < 1 or self.iterations < 1 or self.reg <= 0 or self.alpha < 0:
            raise ValueError("invalid MF hyperparameters")
        rng = np.random.default_rng(self.seed)
        n_items = train.shape[1]
        conf, pref = 1.0 + self.alpha * train, (train > 0).astype(float)
        items = rng.normal(0, 0.1, size=(n_items, self.factors))
        for _ in range(self.iterations):
            users = _als_solve(items, conf, pref, self.reg)
            items = _als_solve(users, conf.T, pref.T, self.reg)
        self.item_factors_ = items
        return self

    def score(self, history: np.ndarray) -> np.ndarray:
        users = _als_solve(self.item_factors_, 1.0 + self.alpha * history,
                           (history > 0).astype(float), self.reg)
        return users @ self.item_factors_.T


MODEL_FAMILIES = {"popularity": Popularity, "itemknn": ItemKNN, "mf": ImplicitMF}

# Small, fixed hyperparameter grids searched on the validation split (in this order;
# ties keep the earlier configuration).
GRIDS: dict[str, list[dict]] = {
    "popularity": [{"signal": s} for s in ("target", "item_event")],
    "itemknn": [{"signal": s, "include_self": inc}
                for s in ("target", "item_event") for inc in (True, False)],
    "mf": [{"signal": s, "factors": f, "reg": r, "alpha": a}
           for s in ("target", "item_event") for f in (2, 3) for r in (0.1, 1.0) for a in (5.0, 20.0)],
}


def make_model(family: str, params: dict):
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown model family {family!r}")
    return MODEL_FAMILIES[family](**params)
