"""Phase 3.1 Checkpoint 3: hybrid building blocks, cross-fitting cell, selection, tuning preflight."""
import math

import numpy as np
import pandas as pd
import pytest

from bri.experiments import integrity as I
from bri.seq import hybrid as H
from bri.seq.data import SeqCases


def _seq(new_session, length, max_len=4, n_items=3):
    n = len(length)
    cases = pd.DataFrame({"case_id": [f"c{i}" for i in range(n)], "customer_token": [f"SYN{i}" for i in range(n)],
                          "within_session": [True] * n, "cutoff_utc": pd.Timestamp("2026-01-01", tz="UTC")})
    items = np.zeros((n, max_len, 1), dtype=np.int64)
    for i, k in enumerate(length):
        items[i, max_len - k:, 0] = 1
    return SeqCases("nbs", "train", [f"i{j}" for j in range(n_items)], max_len, cases, items,
                    np.asarray(new_session, dtype=np.int8), np.asarray(length), np.asarray(length),
                    np.eye(n_items, dtype=np.float32)[np.zeros(n, dtype=int)])


# --- pre-registration consistency ------------------------------------------------------------

def test_constants_match_the_pinned_preregistration():
    pre = I.load_manifests()["phase3_1-prereg"]
    assert H.FEATURE_GROUPS == {g: tuple(v) for g, v in pre["features"]["groups"].items()}
    assert list(H.SUBSETS) == pre["features"]["subsets"]
    assert list(H.C_GRID) == pre["models"]["gated_fusion"]["C_grid"]
    assert list(H.ALPHA_GRID) == pre["models"]["static_fusion"]["alpha_grid"]
    opt = pre["gate"]["optimizer"]
    assert (H.NEWTON_MAX_ITER, H.NEWTON_TOL) == (opt["max_iter"], opt["grad_max_norm_tol"])
    sup = pre["gate"]["support"]
    assert (H.VALID_SUPPORT, H.EXPLORATORY_SUPPORT) == (sup["valid_min_per_class_per_fold"],
                                                         sup["exploratory_min_per_class_per_fold"])
    assert (H.FOLDS, H.NOT_EVALUABLE) == (pre["gate"]["cross_fitting"]["folds"], pre["status_labels"]["not_evaluable"])


# --- normalization and fusion ----------------------------------------------------------------

def test_pf_laplace_and_softmax():
    np.testing.assert_allclose(H.pf_distribution([[2, 0, 1]]), [[3 / 6, 1 / 6, 2 / 6]])
    np.testing.assert_allclose(H.pf_distribution([[0, 0, 0]]), [[1 / 3] * 3])
    p = H.softmax([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(p, H.softmax([[101.0, 102.0, 103.0]]))
    np.testing.assert_allclose(p.sum(), 1.0)


def test_fusion_extremes_and_constant_gate_equal_static():
    pf, sas = H.pf_distribution([[3, 1, 0], [0, 0, 2]]), H.softmax([[0.0, 2.0, 1.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(H.static_fusion(pf, sas, 0.0), pf)
    np.testing.assert_allclose(H.static_fusion(pf, sas, 1.0), sas)
    np.testing.assert_allclose(H.gated_fusion(pf, sas, np.full(2, 0.3)), H.static_fusion(pf, sas, 0.3))


def test_candidate_sets_must_be_identical_and_sorted():
    assert H.check_candidates(["a", "b"], ["a", "b"]) == ["a", "b"]
    with pytest.raises(ValueError, match="differ"):
        H.check_candidates(["a", "b"], ["a", "c"])
    with pytest.raises(ValueError, match="sorted"):
        H.check_candidates(["b", "a"])


# --- features -----------------------------------------------------------------------------------

def test_top_entropy_margin():
    top, ent, margin = H._top_entropy_margin(np.array([[0.25] * 4, [0.7, 0.1, 0.1, 0.1]]))
    np.testing.assert_allclose(top, [0.25, 0.7])
    np.testing.assert_allclose(margin, [0.0, 0.6])
    expected = -(0.7 * math.log(0.7) + 3 * 0.1 * math.log(0.1)) / math.log(4)
    np.testing.assert_allclose(ent, [1.0, expected])


def test_history_features_use_the_represented_boundaries_only():
    # case 0: 3 steps, a boundary at the 2nd represented step -> sessions {1}, {2,3}
    # case 1: empty history; case 2: 4 steps, no boundary
    seq = _seq([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [3, 0, 4])
    h = H.history_features(seq)
    np.testing.assert_allclose(h["h_sessions"], [2, 0, 1])
    np.testing.assert_allclose(h["h_last_session_steps"], [2, 0, 4])
    np.testing.assert_allclose(h["h_last_session_share"], [2 / 3, 0, 1])


def test_gate_features_ignore_target_and_label_information():
    """Forbidden inputs: target items, target timestamp, within/cross label, customer identity."""
    seq = _seq([[0, 0, 1, 0], [0, 1, 0, 0]], [3, 3])
    pf, sas = H.pf_distribution([[1, 2, 0], [0, 0, 3]]), H.softmax([[0.1, 0.5, 0.2], [2.0, 0.0, 1.0]])
    base = H.gate_features(pf, sas, seq)
    blind = SeqCases(seq.task, seq.split, seq.items_vocab, seq.max_len,
                     pd.DataFrame({"case_id": ["x", "y"]}), seq.items, seq.new_session, seq.length,
                     seq.full_length, np.zeros_like(seq.targets))
    for k, v in H.gate_features(pf, sas, blind).items():
        np.testing.assert_array_equal(v, base[k])
    assert set(base) == set(H.FEATURES)


# --- folds, labels, support ------------------------------------------------------------------

def test_fold_assignment_is_deterministic_by_customer():
    custs = [f"SYN{i:03d}" for i in range(10)]
    a = H.assign_folds(custs)
    assert a == H.assign_folds(list(reversed(custs)) + custs[:3])  # order and duplicates irrelevant
    assert sorted(np.bincount(list(a.values()))) == [3, 3, 4]
    order = np.random.default_rng(3101).permutation(10)
    assert a == {custs[j]: pos % 3 for pos, j in enumerate(order)}


def test_soft_labels_and_support_status():
    np.testing.assert_array_equal(H.soft_labels([1, 0, 1, 0], [0, 1, 1, 0]), [1, 0, 0.5, 0.5])
    c = lambda y0, y05, y1: {"y0": y0, "y05": y05, "y1": y1}  # noqa: E731
    assert H.support_status([c(60, 999, 50), c(70, 0, 80)]) == ("valid", 50)
    assert H.support_status([c(60, 999, 49), c(70, 0, 80)]) == ("exploratory", 49)
    assert H.support_status([c(29, 999, 500)] * 3) == ("rejected", 29)  # ties never count as a class


# --- logistic gate ---------------------------------------------------------------------------

def test_gate_bias_only_equals_logit_of_mean_target():
    g = H.fit_logistic_gate(np.zeros((4, 0)), np.array([1, 0, 0.5, 1]), c=1.0)
    assert g.converged and math.isclose(g.b, math.log(0.625 / 0.375), abs_tol=1e-12)


def test_gate_hand_computed_two_point_example():
    # x = (-1, 1), y = (0, 1), C = 1: by symmetry b = 0 and w solves 2*sigmoid(w) - 2 + w = 0.
    g = H.fit_logistic_gate(np.array([[-1.0], [1.0]]), np.array([0.0, 1.0]), c=1.0)
    lo, hi = 0.0, 2.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 2 / (1 + math.exp(-mid)) - 2 + mid < 0 else (lo, mid)
    assert g.converged and abs(g.b) < 1e-12 and math.isclose(g.w[0], lo, abs_tol=1e-10)
    x, y = np.array([[-1.0], [1.0]]), np.array([0.0, 1.0])
    for dw in (1e-3, -1e-3):
        worse = H.LogisticGate(g.w + dw, g.b, 0, True)
        assert H.gate_objective(worse, x, y, 1.0) > H.gate_objective(g, x, y, 1.0)


def test_gate_is_invariant_to_case_order_and_accepts_fractional_targets():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    y = np.where(rng.random(300) < 0.2, 0.5, (x[:, 0] + rng.normal(size=300) > 0).astype(float))
    g = H.fit_logistic_gate(x, y, c=0.1)
    perm = rng.permutation(300)
    g2 = H.fit_logistic_gate(x[perm], y[perm], c=0.1)
    assert g.converged and g.iterations <= H.NEWTON_MAX_ITER
    np.testing.assert_allclose(g.w, g2.w, atol=1e-10)
    assert math.isclose(g.b, g2.b, abs_tol=1e-10)
    with pytest.raises(ValueError):
        H.fit_logistic_gate(x, y + 1.0, c=1.0)
    with pytest.raises(ValueError):
        H.fit_logistic_gate(x, y, c=0.0)


# --- per-cell evaluation on real sequence cases (no torch: synthetic logits) -----------------

@pytest.fixture(scope="module")
def nbs_cases(processed_dir):
    from bri.experiments.baselines import load_processed
    from bri.prep.split import compute_cutoffs
    from bri.seq.data import build_seq_cases
    from bri.seq.prepare import no_test_events
    from bri.seq.train import train_popularity
    data = load_processed(processed_dir)
    c, p = data["meta"]["cutoffs"], data["meta"]["params"]
    cut = compute_cutoffs(c["window_start_utc"], c["window_end_utc"], p["val_days"], p["test_days"])
    inter = no_test_events(data["interactions"])
    seqs = build_seq_cases(inter, cut, data["candidates"], ("train", "val"), 10)
    return inter, seqs[("nbs", "train")], seqs[("nbs", "val")], train_popularity(inter, "nbs", data["candidates"]["nbs"])


def test_evaluate_cell_structure_and_extremes(nbs_cases, monkeypatch):
    pytest.importorskip("torch")
    from bri.seq.crossfit import fold_array, pf_counts
    inter, tr, va, tb = nbs_cases
    rng = np.random.default_rng(1)
    oof, val = rng.normal(size=tr.targets.shape), rng.normal(size=va.targets.shape)
    monkeypatch.setattr(H, "VALID_SUPPORT", 1)  # tiny session dataset: force the fitted path
    monkeypatch.setattr(H, "EXPLORATORY_SUPPORT", 1)
    cell = H.evaluate_cell(tr, va, oof, val, pf_counts(inter, tr), pf_counts(inter, va), tb, fold_array(tr))
    v = cell["val_ndcg1"]
    np.testing.assert_array_equal(v["static"][0.0], v["pf"])
    np.testing.assert_array_equal(v["static"][1.0], v["sasrec"])
    assert cell["status"] == "valid" and len(v["gated"]) == len(H.SUBSETS) * len(H.C_GRID)
    assert sum(f["y0"] + f["y05"] + f["y1"] for f in cell["per_fold"]) == len(tr.cases)
    assert all(0 <= g["alpha_val"]["p5"] <= g["alpha_val"]["p95"] <= 1 for g in cell["gates"].values())
    # oracle router: SASRec on within-session cases, PF on cross-session cases
    w = va.cases["within_session"].to_numpy(dtype=bool)
    np.testing.assert_array_equal(v["oracle_router"], np.where(w, v["sasrec"], v["pf"]))


def test_rejected_support_fits_no_gate(nbs_cases, monkeypatch):
    pytest.importorskip("torch")
    from bri.seq.crossfit import fold_array, pf_counts
    inter, tr, va, tb = nbs_cases
    monkeypatch.setattr(H, "VALID_SUPPORT", 10**6)
    monkeypatch.setattr(H, "EXPLORATORY_SUPPORT", 10**6)
    cell = H.evaluate_cell(tr, va, np.zeros(tr.targets.shape), np.zeros(va.targets.shape),
                           pf_counts(inter, tr), pf_counts(inter, va), tb, fold_array(tr))
    assert cell["status"] == "rejected"
    assert all(x is None for x in cell["val_ndcg1"]["gated"].values())
    assert all(g == {"status": H.NOT_EVALUABLE} for g in cell["gates"].values())
    assert H.cell_means(cell)["gated"][H.gate_key("P", 1.0)] is None  # never zero


def test_run_cell_cpu_end_to_end(nbs_cases):
    pytest.importorskip("torch")
    from bri.seq.crossfit import run_cell
    inter, tr, va, tb = nbs_cases
    frozen = {"model": {"blocks": 1, "dim": 16, "dropout": 0.0, "heads": 2, "max_len": 10},
              "train": {"batch_size": 64, "grad_clip": 1.0, "lr": 1e-3, "max_epochs": 2, "patience": 5, "seed": 99,
                        "select_k": 3, "select_metric": "ndcg", "weight_decay": 0.0}}
    cell = run_cell(inter, tr, va, tb, frozen, 2, "cpu")
    assert cell["init_seed"] == 2 and [f["fit"] for f in cell["fits"]] == ["oof_fold0", "oof_fold1", "oof_fold2", "full"]
    assert len(cell["val_ndcg1"]["pf"]) == len(va.cases)
    with pytest.raises(ValueError, match="test-window"):
        run_cell(inter.assign(split="test"), tr, va, tb, frozen, 2, "cpu")


# --- selection ----------------------------------------------------------------------------------

def _cells(static_peak, gated_shift, status="valid", gated_none=False):
    out = []
    for profile in ("p1", "p2"):
        for seed in (1, 2):
            for task in ("nbs", "nbp"):
                for init in (1, 2):
                    st = {a: 0.5 - abs(a - static_peak[task]) / 10 for a in H.ALPHA_GRID}
                    gt = {H.gate_key(s, c): (None if gated_none else st[static_peak[task]] + gated_shift)
                          for s in H.SUBSETS for c in H.C_GRID}
                    out.append({"profile": profile, "seed": seed, "task": task, "init": init, "status": status,
                                "means": {"pf": 0.4, "sasrec": 0.45, "oracle_router": 0.6, "static": st, "gated": gt}})
    return out


def test_selection_picks_alpha_per_task_and_gated_only_when_valid_and_better():
    peak = {"nbs": 0.7, "nbp": 0.2}
    s = H.select(_cells(peak, 0.01), ["p1", "p2"], ["nbs", "nbp"])
    assert s["per_task"]["nbs"]["alpha"] == 0.7 and s["per_task"]["nbp"]["alpha"] == 0.2
    assert s["per_task"]["nbs"]["gate"] == H.gate_key("P", 0.01)  # all tied -> earliest grid entry
    assert math.isclose(s["delta_val"], 0.01) and s["primary_hybrid"] == "gated" and s["n_groups"] == 4
    assert H.select(_cells(peak, 0.0), ["p1", "p2"], ["nbs", "nbp"])["primary_hybrid"] == "static"
    expl = H.select(_cells(peak, 0.05, status="exploratory"), ["p1", "p2"], ["nbs", "nbp"])
    assert expl["primary_hybrid"] == "static" and expl["status_counts"]["exploratory"] == 16


def test_selection_never_replaces_unavailable_gates_with_zero():
    s = H.select(_cells({"nbs": 0.5, "nbp": 0.5}, 0.0, status="rejected", gated_none=True), ["p1", "p2"],
                 ["nbs", "nbp"])
    assert s["delta_val"] is None and s["primary_hybrid"] == "static"
    assert s["per_task"]["nbs"]["gated_evaluable"] is False and s["per_task"]["nbs"]["gate"] is None


# --- tuning preflight -----------------------------------------------------------------------

def test_tune_preflight_accepts_pre_freeze_state_and_refuses_otherwise():
    from bri.experiments import phase3_1 as P31
    all_manifests = I.load_manifests()
    manifests = {k: v for k, v in all_manifests.items() if k != P31.CONFIG_MANIFEST}  # state before Checkpoint 4
    pf = P31.tune_preflight(manifests)
    assert pf["init_seeds"] == [1, 2, 3] and len(pf["entries"]) == 9
    with pytest.raises(P31.Phase31Refused, match="already frozen"):
        P31.tune_preflight(all_manifests)  # the pinned config forbids any further tuning run
    with pytest.raises(P31.Phase31Refused, match="not pinned"):
        P31.tune_preflight({k: v for k, v in manifests.items() if k != P31.TUNING_DATA_MANIFEST})
    bad = {**manifests["phase3-config"], "frozen_config": {**manifests["phase3-config"]["frozen_config"], "device": "x"}}
    with pytest.raises(P31.Phase31Refused, match="differs"):
        P31.tune_preflight({**manifests, "phase3-config": bad})


# --- Checkpoint 4: pinned configuration ----------------------------------------------------

def test_pinned_config_is_the_approved_static_proposal():
    from bri.experiments import phase3_1 as P31
    m = I.load_manifests()
    c = m[P31.CONFIG_MANIFEST]
    f = c["frozen_config"]
    assert I.canonical_hash(f) == c["frozen_config_sha256"]
    assert f["primary_hybrid"] == "static_fusion" and f["alpha"] == {"nbs": 0.6, "nbp": 0.1}
    assert all(a in H.ALPHA_GRID for a in f["alpha"].values())
    assert f["sasrec"]["frozen_config_sha256"] == m["phase3_1-prereg"]["references"]["phase3_frozen_config_sha256"]
    assert f["sasrec"]["initialization_seeds"] == m["phase3_1-prereg"]["seed_policy"]["initialization_seeds"]
    assert c["references"] == {"prereg_sha256": I.MANIFEST_SHA256["phase3_1-prereg"],
                               "seeds_sha256": I.MANIFEST_SHA256["phase3_1-seeds"],
                               "tuning_data_sha256": I.MANIFEST_SHA256["phase3_1-tuning-data"],
                               "phase3_config_manifest_sha256": I.MANIFEST_SHA256["phase3-config"]}
    s = c["selection_record"]
    assert s["all_gates_valid"] is False and s["status_counts"] == {"valid": 18, "exploratory": 3, "rejected": 33}
    assert s["delta_val"] == H.NOT_EVALUABLE and s["alpha_exact_ties"]["nbp"] == [0.1, 0.2]
    assert "NOT written in the phase3_1-prereg" in c["clarifications"][0]["text"]
    assert c["datasets"] == [] and c["tuning_report"]["tuning_code_commit"].startswith("17ce14a")
    r = c["cost_reconciliation"]
    assert math.isclose(r["sasrec_block_minus_fit_training"],
                        r["sasrec_block_seconds_total"] - r["fit_training_seconds_total"], abs_tol=2e-3)
    assert math.isclose(r["wall_minus_fit_training"], r["wall_minus_sasrec_block"] + r["sasrec_block_minus_fit_training"],
                        abs_tol=2e-3)


def test_config_manifest_rebuilds_from_the_local_tuning_report():
    """Skipped where the git-ignored tuning report is absent; otherwise it must match the pin byte for byte."""
    import json
    from bri.experiments import phase3_1 as P31
    path = I.REPO_ROOT / P31.TUNING_DIR / "tuning_report.json"
    if not path.exists():
        pytest.skip("local tuning report not present")
    m = I.load_manifests()
    pinned = m[P31.CONFIG_MANIFEST]
    pre = {k: v for k, v in m.items() if k != P31.CONFIG_MANIFEST}
    rebuilt = P31.config_manifest(json.loads(path.read_text(encoding="utf-8")), I.sha256_file(path),
                                  pinned["tuning_report"]["tuning_code_commit"], pre)
    assert json.loads(json.dumps(rebuilt, sort_keys=True)) == pinned
