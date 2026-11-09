"""Metric correctness against hand-calculated values."""
import math

import pytest

from bri.eval import metrics as M

RANKED = ["A", "B", "C", "D"]
REL = {"B", "D", "E"}
L2_3 = 1 / math.log2(3)  # 0.6309...
IDCG3 = 1 + L2_3 + 0.5   # 2.1309...


def test_example_at_k3():
    # top-3 = A,B,C ; one hit (B at rank 2)
    assert M.precision_at_k(RANKED, REL, 3) == pytest.approx(1 / 3)
    assert M.recall_at_k(RANKED, REL, 3) == pytest.approx(1 / 3)
    assert M.ndcg_at_k(RANKED, REL, 3) == pytest.approx(L2_3 / IDCG3)       # 0.29608
    assert M.average_precision_at_k(RANKED, REL, 3) == pytest.approx((1 / 2) / 3)  # 1/6


def test_example_at_k4():
    # hits at ranks 2 and 4
    assert M.precision_at_k(RANKED, REL, 4) == pytest.approx(0.5)
    assert M.recall_at_k(RANKED, REL, 4) == pytest.approx(2 / 3)
    assert M.ndcg_at_k(RANKED, REL, 4) == pytest.approx((L2_3 + 1 / math.log2(5)) / IDCG3)  # 0.49819
    assert M.average_precision_at_k(RANKED, REL, 4) == pytest.approx((1 / 2 + 2 / 4) / 3)  # 1/3


def test_perfect_ranking_scores_one():
    for fn in (M.precision_at_k, M.recall_at_k, M.ndcg_at_k, M.average_precision_at_k):
        assert fn(["B", "D", "E"], REL, 3) == pytest.approx(1.0)


def test_hit_at_rank_one_only():
    assert M.ndcg_at_k(["B", "A"], {"B"}, 2) == pytest.approx(1.0)
    assert M.ndcg_at_k(["A", "B"], {"B"}, 2) == pytest.approx(L2_3)
    assert M.average_precision_at_k(["A", "B"], {"B"}, 2) == pytest.approx(0.5)


def test_short_and_empty_lists():
    assert M.precision_at_k(["B"], {"B"}, 3) == pytest.approx(1 / 3)  # denominator stays K
    assert M.recall_at_k(["B"], {"B"}, 3) == 1.0
    assert M.ndcg_at_k(["B"], {"B"}, 3) == 1.0
    assert M.average_precision_at_k(["B"], {"B"}, 3) == 1.0
    for fn in (M.precision_at_k, M.recall_at_k, M.ndcg_at_k, M.average_precision_at_k):
        assert fn([], {"B"}, 3) == 0.0


@pytest.mark.parametrize("fn", [M.precision_at_k, M.recall_at_k, M.ndcg_at_k, M.average_precision_at_k])
def test_invalid_inputs(fn):
    with pytest.raises(ValueError):
        fn(["A"], set(), 1)          # no relevant items
    with pytest.raises(ValueError):
        fn(["A", "A"], {"A"}, 2)     # duplicate recommendation
    for bad_k in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            fn(["A"], {"A"}, bad_k)


def test_coverage():
    lists, cands = [["A", "B"], ["A", "C"]], ["A", "B", "C", "D"]
    assert M.catalog_coverage(lists, cands, 1) == 0.25
    assert M.catalog_coverage(lists, cands, 2) == 0.75
    assert M.catalog_coverage([], cands, 2) == 0.0


def test_diversity():
    cands = ["A", "B", "C", "D"]
    assert M.exposure_diversity([["A", "B"], ["A", "C"]], cands, 1) == 0.0  # everyone gets A
    # exposure A=2, B=1, C=1 -> p = .5,.25,.25 ; H = 1.5 ln2 ; / ln4 = 0.75
    assert M.exposure_diversity([["A", "B"], ["A", "C"]], cands, 2) == pytest.approx(0.75)
    assert M.exposure_diversity([["A", "B"], ["C", "D"]], cands, 2) == pytest.approx(1.0)
    assert M.exposure_diversity([], cands, 2) == 0.0
    assert M.exposure_diversity([["A"]], ["A"], 1) == 0.0


def test_aggregate_metrics_reject_bad_lists():
    with pytest.raises(ValueError, match="outside the candidate set"):
        M.catalog_coverage([["Z"]], ["A", "B"], 1)
    with pytest.raises(ValueError, match="duplicate"):
        M.exposure_diversity([["A", "A"]], ["A", "B"], 2)
    with pytest.raises(ValueError):
        M.catalog_coverage([["A"]], [], 1)
