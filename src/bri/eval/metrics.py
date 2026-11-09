"""Ranking and catalog metrics with binary relevance.

Per-case metrics take a ranked list (best first, no duplicates) and a non-empty set of
relevant items. Lists shorter than K are allowed; missing positions count as
non-relevant. Definitions:

    hits@K        = |top_K ∩ R|
    Precision@K   = hits@K / K
    Recall@K      = hits@K / |R|
    DCG@K         = sum_{i=1..K} rel_i / log2(i + 1)
    NDCG@K        = DCG@K / IDCG@K,  IDCG@K = sum_{i=1..min(|R|,K)} 1 / log2(i + 1)
    AP@K          = (1 / min(|R|, K)) * sum_{i=1..K} Precision@i * rel_i
    MAP@K         = mean of AP@K over cases

Aggregate metrics over all recommendation lists of one task:

    Coverage@K    = |union of top_K lists| / |candidate set|
    Diversity@K   = normalized Shannon entropy of item exposure in top_K lists:
                    H(p) / log(|candidate set|), where p_j = share of top_K slots taken
                    by item j. 0 = every list identical, 1 = exposure perfectly even.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence


def _check(ranked: Sequence[str], relevant: set[str], k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if not relevant:
        raise ValueError("relevant set must be non-empty (cases without positives are excluded upstream)")
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked list contains duplicate items")


def _rels(ranked: Sequence[str], relevant: set[str], k: int) -> list[int]:
    return [1 if item in relevant else 0 for item in ranked[:k]]


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _check(ranked, relevant, k)
    return sum(_rels(ranked, relevant, k)) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _check(ranked, relevant, k)
    return sum(_rels(ranked, relevant, k)) / len(relevant)


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _check(ranked, relevant, k)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(_rels(ranked, relevant, k)))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg


def average_precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _check(ranked, relevant, k)
    hits, total = 0, 0.0
    for i, r in enumerate(_rels(ranked, relevant, k), start=1):
        if r:
            hits += 1
            total += hits / i
    return total / min(len(relevant), k)


def _exposure(lists: Iterable[Sequence[str]], candidates: Sequence[str], k: int) -> Counter:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if not candidates:
        raise ValueError("candidate set must be non-empty")
    allowed = set(candidates)
    counts: Counter = Counter()
    for ranked in lists:
        top = list(ranked[:k])
        if len(set(top)) != len(top):
            raise ValueError("ranked list contains duplicate items")
        unknown = set(top) - allowed
        if unknown:
            raise ValueError(f"items outside the candidate set: {sorted(unknown)}")
        counts.update(top)
    return counts


def catalog_coverage(lists: Iterable[Sequence[str]], candidates: Sequence[str], k: int) -> float:
    return len(_exposure(lists, candidates, k)) / len(set(candidates))


def exposure_diversity(lists: Iterable[Sequence[str]], candidates: Sequence[str], k: int) -> float:
    counts = _exposure(lists, candidates, k)
    n = len(set(candidates))
    total = sum(counts.values())
    if total == 0 or n == 1:
        return 0.0
    entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
    return entropy / math.log(n)
