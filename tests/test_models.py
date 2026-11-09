import numpy as np
import pandas as pd
import pytest

from bri.eval.evaluate import validate_recommendations
from bri.models.baselines import GRIDS, ImplicitMF, ItemKNN, Popularity, make_model
from bri.models.common import history_matrix, to_recommendations, user_item_matrix
from helpers import events

ITEMS = ["SVC_A", "SVC_B", "SVC_C"]
# 4 customers x 3 items (A, B, C)
TRAIN = np.array([
    [2, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 3],
], dtype=float)


def test_popularity_counts_distinct_customers_and_ignores_history():
    m = Popularity().fit(TRAIN)
    assert m.popularity_.tolist() == [3, 2, 1]
    scores = m.score(np.array([[0, 0, 5], [1, 0, 0]], dtype=float))
    assert (scores == [[3, 2, 1], [3, 2, 1]]).all()


def test_itemknn_cosine_similarity_by_hand():
    m = ItemKNN().fit(TRAIN)
    # binary columns: A=[1,1,1,0], B=[1,1,0,0], C=[0,0,0,1]
    # cos(A,B) = 2 / (sqrt3*sqrt2) ; cos(A,C) = 0 ; cos(B,C) = 0
    expected = 2 / (np.sqrt(3) * np.sqrt(2))
    assert m.similarity_[0, 1] == pytest.approx(expected)
    assert m.similarity_[0, 2] == 0 and m.similarity_[1, 2] == 0
    assert np.allclose(np.diag(m.similarity_), 1.0)
    # history: used B twice -> score = log1p(2) * sim(B, .)
    s = m.score(np.array([[0, 2, 0]], dtype=float))[0]
    assert s == pytest.approx(np.log1p(2) * np.array([expected, 1.0, 0.0]))


def test_itemknn_without_self_scores_by_co_usage_only():
    m = ItemKNN(include_self=False).fit(TRAIN)
    assert np.allclose(np.diag(m.similarity_), 0.0)
    s = m.score(np.array([[0, 1, 0]], dtype=float))[0]
    assert s[1] == 0 and s[0] > 0


def test_itemknn_handles_unused_items():
    m = ItemKNN().fit(np.array([[1, 0, 0]], dtype=float))
    assert np.isfinite(m.similarity_).all()
    assert m.similarity_[1, 1] == 0  # never-used item has zero norm -> zero similarity


def test_mf_is_deterministic_and_learns_block_structure():
    # two groups: (A,B) users and C users
    train = np.array([[3, 2, 0]] * 20 + [[0, 0, 4]] * 20, dtype=float)
    a = ImplicitMF(factors=2, seed=3).fit(train)
    b = ImplicitMF(factors=2, seed=3).fit(train)
    assert np.array_equal(a.item_factors_, b.item_factors_)
    s = a.score(np.array([[1, 0, 0], [0, 0, 1]], dtype=float))
    assert s[0, 1] > s[0, 2]   # an A-user is pointed to B, not C
    assert s[1, 2] > s[1, 1]   # a C-user keeps C on top


def test_mf_scores_cases_with_no_history():
    m = ImplicitMF().fit(TRAIN)
    s = m.score(np.zeros((2, 3)))
    assert s.shape == (2, 3) and np.isfinite(s).all()


@pytest.mark.parametrize("bad", [{"factors": 0}, {"reg": 0.0}, {"iterations": 0}, {"alpha": -1.0}])
def test_mf_rejects_invalid_hyperparameters(bad):
    with pytest.raises(ValueError):
        ImplicitMF(**bad).fit(TRAIN)


def test_models_fit_on_empty_training_data():
    empty = np.zeros((0, 3))
    hist = np.array([[1, 0, 0]], dtype=float)
    for model in (Popularity(), ItemKNN(), ImplicitMF()):
        assert np.isfinite(model.fit(empty).score(hist)).all()


def test_grid_configs_construct():
    for family, grid in GRIDS.items():
        for params in grid:
            make_model(family, params)
    with pytest.raises(ValueError):
        make_model("lightgbm", {})


def test_ranking_tiebreak_is_deterministic():
    scores = np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 2.0 + 1e-13]])
    recs = to_recommendations(["c1", "c2"], scores, ITEMS, tiebreak=np.array([1.0, 5.0, 5.0]))
    assert recs[recs.case_id == "c1"]["item_id"].tolist() == ["SVC_B", "SVC_C", "SVC_A"]
    # 1e-13 is float noise and is rounded away; tie broken by popularity then item_id
    assert recs[recs.case_id == "c2"]["item_id"].tolist() == ["SVC_B", "SVC_C", "SVC_A"]


C1 = "SYN_CUST_000001"
ROWS = [
    (C1, "SYN_SESS_000001_0001", "2024-12-02T08:00:00Z", "service_view", "SVC_A"),
    (C1, "SYN_SESS_000001_0001", "2024-12-02T08:00:10Z", "service_start", "SVC_A"),
    (C1, "SYN_SESS_000001_0002", "2024-12-03T08:00:00Z", "service_start", "SVC_A"),
    (C1, "SYN_SESS_000001_0003", "2024-12-20T08:00:00Z", "service_start", "SVC_B"),  # after cutoff
]


def _inter(rows):
    df = events(rows)
    df["task"] = "nbs"
    df["is_target"] = df["event_type"] == "service_start"
    return df


def _case(cutoff="2024-12-15T00:00:00Z"):
    return pd.DataFrame({"case_id": ["X"], "task": ["nbs"], "customer_token": [C1],
                         "cutoff_utc": [pd.Timestamp(cutoff)]})


def test_history_matrix_uses_only_pre_cutoff_events():
    inter = _inter(ROWS)
    assert history_matrix(inter, _case(), ITEMS, "target").tolist() == [[2, 0, 0]]
    assert history_matrix(inter, _case(), ITEMS, "item_event").tolist() == [[3, 0, 0]]
    # leakage check: adding/removing post-cutoff events changes nothing
    assert np.array_equal(history_matrix(_inter(ROWS[:3]), _case(), ITEMS, "target"),
                          history_matrix(inter, _case(), ITEMS, "target"))


def test_history_matrix_ignores_other_task_and_non_candidates():
    inter = _inter(ROWS[:3])
    other = inter.assign(task="nbp")
    assert history_matrix(other, _case(), ITEMS, "target").sum() == 0
    assert history_matrix(inter, _case(), ["SVC_B"], "target").tolist() == [[0]]


def test_user_item_matrix_rows_are_customers():
    m = user_item_matrix(_inter(ROWS[:3]), ITEMS, "target")
    assert m.tolist() == [[2, 0, 0]]
    with pytest.raises(ValueError):
        user_item_matrix(_inter(ROWS[:3]), ITEMS, "clicks")


def test_model_recommendations_pass_validation():
    cases = pd.DataFrame({"case_id": ["X", "Y"], "task": ["nbs", "nbs"]})
    hist = np.array([[1, 0, 0], [0, 0, 2]], dtype=float)
    cands = {"nbs": ITEMS, "nbp": ["PRD_X"]}
    for model in (Popularity(), ItemKNN(), ImplicitMF()):
        recs = to_recommendations(cases["case_id"], model.fit(TRAIN).score(hist), ITEMS, np.zeros(3))
        assert validate_recommendations(recs, cases, cands) == []
        assert set(recs["item_id"]) <= set(ITEMS)


def test_personal_frequency_ranks_own_history_with_popularity_fallback():
    from bri.models.baselines import PersonalFrequency

    m = PersonalFrequency().fit(TRAIN)
    hist = np.array([[0, 3, 1], [0, 0, 0]], dtype=float)
    assert (m.score(hist) == hist).all()
    pop = Popularity().fit(TRAIN).popularity_  # [3, 2, 1]
    recs = to_recommendations(["c1", "c2"], m.score(hist), ITEMS, tiebreak=pop)
    assert recs[recs.case_id == "c1"]["item_id"].tolist() == ["SVC_B", "SVC_C", "SVC_A"]
    # no item-bearing history -> exactly the popularity order
    assert recs[recs.case_id == "c2"]["item_id"].tolist() == ["SVC_A", "SVC_B", "SVC_C"]


def test_personal_frequency_ignores_other_customers():
    from bri.models.baselines import PersonalFrequency

    hist = np.array([[1, 0, 0]], dtype=float)
    a = PersonalFrequency().fit(TRAIN).score(hist)
    b = PersonalFrequency().fit(np.zeros((0, 3))).score(hist)
    assert np.array_equal(a, b)


def test_suites_are_well_formed():
    from bri.models.baselines import KEY_COMPARISONS, SUITES

    for suite, models in SUITES.items():
        for name, (family, grid) in models.items():
            assert grid, name
            for params in grid:
                make_model(family, params)
        for a, b, _ in KEY_COMPARISONS[suite]:
            assert a in models and b in models
