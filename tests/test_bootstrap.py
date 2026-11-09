import numpy as np
import pandas as pd
import pytest

from bri.eval.bootstrap import bootstrap_ci, paired_bootstrap_diff


def test_constant_values_have_zero_width_interval():
    r = bootstrap_ci([0.5] * 40, n_boot=200, seed=1)
    assert r == {"mean": 0.5, "ci_low": 0.5, "ci_high": 0.5}


def test_ci_contains_mean_and_is_deterministic():
    vals = np.random.default_rng(0).random(300)
    a = bootstrap_ci(vals, n_boot=500, seed=7)
    b = bootstrap_ci(vals, n_boot=500, seed=7)
    assert a == b
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]


def test_ci_matches_normal_approximation_roughly():
    vals = np.random.default_rng(1).integers(0, 2, 2000).astype(float)  # Bernoulli(0.5)
    r = bootstrap_ci(vals, n_boot=2000, seed=3)
    se = vals.std() / np.sqrt(len(vals))
    assert (r["ci_high"] - r["ci_low"]) == pytest.approx(2 * 1.96 * se, rel=0.15)


def test_chunking_does_not_change_results():
    vals = np.arange(50, dtype=float)
    # 450 resamples spans several internal chunks of 200
    assert bootstrap_ci(vals, n_boot=450, seed=5) == bootstrap_ci(vals.copy(), n_boot=450, seed=5)


def test_paired_identical_models_have_zero_difference():
    s = pd.Series(np.random.default_rng(2).random(100), index=[f"c{i}" for i in range(100)])
    r = paired_bootstrap_diff(s, s.copy(), n_boot=300, seed=1)
    assert r["mean_diff"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0
    assert r["share_not_better"] == 1.0


def test_paired_is_tighter_than_unpaired_for_correlated_scores():
    rng = np.random.default_rng(4)
    base = rng.random(400)
    idx = [f"c{i}" for i in range(400)]
    a = pd.Series(np.clip(base + 0.05, 0, 1), index=idx)
    b = pd.Series(base, index=idx)
    paired = paired_bootstrap_diff(a, b, n_boot=1000, seed=0)
    ca, cb = bootstrap_ci(a, 1000, 0), bootstrap_ci(b, 1000, 0)
    assert paired["ci_low"] > 0                                   # consistent improvement detected
    assert (paired["ci_high"] - paired["ci_low"]) < (ca["ci_high"] - ca["ci_low"])
    assert ca["ci_low"] < cb["ci_high"]                           # unpaired CIs overlap


def test_paired_requires_same_cases_in_same_order():
    a = pd.Series([1.0, 0.0], index=["c1", "c2"])
    with pytest.raises(ValueError, match="same cases"):
        paired_bootstrap_diff(a, pd.Series([1.0, 0.0], index=["c2", "c1"]))
    with pytest.raises(ValueError):
        paired_bootstrap_diff(a, pd.Series([1.0], index=["c1"]))


@pytest.mark.parametrize("kwargs", [{"n_boot": 0}, {"confidence": 1.0}, {"confidence": 0.0}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 0.0], **{"n_boot": 10, **kwargs})


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([], n_boot=10)
