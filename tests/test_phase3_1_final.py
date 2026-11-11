"""Phase 3.1 final runner (Checkpoint 6 implementation): small synthetic data only, never final data."""
import copy
import json

import numpy as np
import pytest

from bri.experiments import integrity as I
from bri.experiments import phase3_1_final as F

torch = pytest.importorskip("torch")

TINY = {"model": {"blocks": 1, "dim": 16, "dropout": 0.0, "heads": 2, "max_len": 10},
        "train": {"batch_size": 64, "grad_clip": 1.0, "lr": 1e-3, "max_epochs": 2, "patience": 5, "seed": 99,
                  "select_k": 3, "select_metric": "ndcg", "weight_decay": 0.0}}


@pytest.fixture(scope="module")
def inputs(processed_dir):
    """Same input construction as run_dataset, on the small session dataset (not a final dataset)."""
    from bri.experiments.baselines import load_processed, next_event_data
    from bri.prep.split import compute_cutoffs
    from bri.seq.data import build_seq_cases, encode_cases, history_steps
    data = load_processed(processed_dir)
    ne, _ = next_event_data(data)
    c, p = data["meta"]["cutoffs"], data["meta"]["params"]
    cut = compute_cutoffs(c["window_start_utc"], c["window_end_utc"], p["val_days"], p["test_days"])
    seqs = build_seq_cases(data["interactions"], cut, data["candidates"], ("train", "val", "test"), 10)
    steps = history_steps(data["interactions"], data["candidates"])
    wt = {t: encode_cases(data["cases"][(data["cases"]["split"] == "test") & (data["cases"]["task"] == t)],
                          data["targets"], steps, t, "test", data["candidates"][t], 10) for t in data["candidates"]}
    return data, ne, seqs, wt


_LOAD_MANIFESTS = I.load_manifests  # the real loader, kept before any monkeypatching


def _pre_lock():
    """Manifests as they were before the one-time phase3_1 lock (the state the runner was gated on)."""
    return {k: v for k, v in _LOAD_MANIFESTS().items() if k != F.LOCK_MANIFEST}


def _run(inputs, task="nbs", alpha=0.6, inits=(1, 2), device="cpu"):
    data, ne, seqs, wt = inputs
    return F.evaluate_task(task, ne, data, seqs, wt[task], alpha, TINY, inits, device)


def test_evaluate_task_structure_and_oracle_is_diagnostic_only(inputs):
    out = _run(inputs)
    assert out["pf_selection"]["next_event"]["selected"] == {"signal": "target"}
    assert out["pf_selection"]["window"]["selected"]["signal"] in ("target", "item_event")
    assert [r["init_seed"] for r in out["sasrec_runtime"]] == [1, 2]
    ne, w = out["protocols"]["next_event"], out["protocols"]["window"]
    for p in (ne, w):
        assert set(p["models"]) == set(F.MODELS) and "oracle_router" not in p["models"]
        assert not any("oracle" in c for c in p["comparisons"])
        assert len(p["initialization"]["static_hybrid"]["@1"]["ndcg_per_init"]) == 2
        for c in p["comparisons"].values():
            assert {"mean_diff", "ci_low", "ci_high"} <= set(c["@1"]["ndcg"])
    assert set(ne["diagnostics"]) == {"oracle_router"} and w["diagnostics"] == {}
    s = ne["strata"]["personal_frequency"]["@1"]
    assert s["within_session"]["n"] + s["cross_session"]["n"] == ne["n_test_cases"]


def test_static_hybrid_extremes_equal_pf_and_sasrec(inputs):
    a0, a1 = _run(inputs, alpha=0.0, inits=(1,)), _run(inputs, alpha=1.0, inits=(1,))
    for proto in ("next_event", "window"):
        assert a0["protocols"][proto]["models"]["static_hybrid"] == a0["protocols"][proto]["models"]["personal_frequency"]
        assert a1["protocols"][proto]["models"]["static_hybrid"] == a1["protocols"][proto]["models"]["sasrec_full"]


def test_pf_matches_the_phase23_runners(inputs, processed_dir):
    """PF and its validation-selected signal are exactly those of the Phase 2.3 classical runners."""
    from bri.experiments.baselines import run_baselines, run_next_event
    out = _run(inputs, inits=(1,))
    w_rep = run_baselines(processed_dir, suite="phase2.3", ks=F.KS, select_metric="ndcg", select_k=3)
    n_rep = run_next_event(processed_dir, suite="phase2.3-next-event", ks=F.KS, select_metric="ndcg", select_k=3)
    for proto, rep in (("next_event", n_rep), ("window", w_rep)):
        e = rep["tasks"]["nbs"]["models"]["personal_frequency"]
        assert out["pf_selection"][proto]["selected"] == e["selected_params"]
        for k in F.KS:
            for m in F.RANK:
                assert out["protocols"][proto]["models"]["personal_frequency"][f"@{k}"][m] == \
                    pytest.approx(e["test"][f"@{k}"][m], abs=1e-6)


def test_initialization_mean_is_the_mean_of_per_init_values(inputs):
    p = _run(inputs)["protocols"]["next_event"]
    per = p["initialization"]["sasrec_full"]["@1"]["ndcg_per_init"]
    assert p["models"]["sasrec_full"]["@1"]["ndcg"] == pytest.approx(np.mean(per), abs=2e-6)


# --- final-data protection ---------------------------------------------------------------------

def test_run_dataset_refuses_without_a_gated_plan(monkeypatch, tmp_path):
    import bri.experiments.baselines as B
    monkeypatch.setattr(B, "load_processed", lambda *a, **k: pytest.fail("final data was opened"))
    plan = F.Plan({}, TINY, {"nbs": 0.6, "nbp": 0.1}, (1,), (), "cpu", {})
    with pytest.raises(F.FinalRunRefused, match="passed every gate"):
        F.run_dataset(plan, {"behavior_profile": "balanced", "seed": 20261150}, tmp_path, tmp_path)


def test_only_the_issued_plan_object_passes_and_it_must_match_the_pins(monkeypatch):
    import dataclasses
    monkeypatch.setattr(I, "load_manifests", _pre_lock)  # require_gated_plan re-reads the pins
    fz = F.check_manifests(I.load_manifests())
    good = F.Plan({}, fz["sasrec"], fz["alpha"], fz["init_seeds"], fz["entries"], "cpu", {})
    with pytest.raises(F.FinalRunRefused, match="passed every gate"):  # built by a caller, same fields
        F.require_gated_plan(good)
    monkeypatch.setattr(F, "_ISSUED", [good])  # simulate issuance by gates()
    F.require_gated_plan(good)
    with pytest.raises(F.FinalRunRefused, match="passed every gate"):  # a copy is a different object
        F.require_gated_plan(dataclasses.replace(good))
    other = dataclasses.replace(good, alpha={"nbs": 0.7, "nbp": 0.1})
    monkeypatch.setattr(F, "_ISSUED", [other])
    with pytest.raises(F.FinalRunRefused, match="no longer matches"):  # fields cannot override the pins
        F.require_gated_plan(other)


def test_each_test_split_is_scored_once_per_initialization_and_pf_reads_only_its_selected_signal(inputs, monkeypatch):
    import bri.seq.train as T
    from bri.experiments.baselines import _TaskData
    calls, hist = [], []
    real_score, real_hist = T.score, _TaskData.history
    monkeypatch.setattr(T, "score", lambda model, seq, device, **k: (calls.append(seq.split),
                                                                     real_score(model, seq, device, **k))[1])
    monkeypatch.setattr(_TaskData, "history", lambda self, split, signal: (hist.append((split, signal)),
                                                                            real_hist(self, split, signal))[1])
    out = _run(inputs, inits=(1, 2, 3))
    assert calls.count("test") == 3 * 2  # 3 initializations x (next-event, window), nothing re-scored
    assert set(calls) <= {"val", "test"}  # val = early stopping inside fit
    sel = {p: out["pf_selection"][p]["selected"]["signal"] for p in ("next_event", "window")}
    test_reads = [s for split, s in hist if split == "test"]
    assert sorted(test_reads) == sorted(sel.values())  # one test read per protocol, selected signal only


def test_bootstrap_is_paired_and_customer_clustered_and_intervals_are_over_seeds(inputs, monkeypatch):
    import bri.eval.bootstrap as Bt
    seen = {"cluster": [], "plain": []}
    real_c, real_p = Bt.paired_cluster_bootstrap_diff, Bt.paired_bootstrap_diff

    def cluster(a, b, clusters, *args):
        assert a.index.equals(b.index) and clusters.index.equals(a.index)
        seen["cluster"].append((len(a), clusters.nunique()))
        return real_c(a, b, clusters, *args)

    def plain(a, b, *args):
        assert a.index.equals(b.index)
        seen["plain"].append(len(a))
        return real_p(a, b, *args)
    monkeypatch.setattr(Bt, "paired_cluster_bootstrap_diff", cluster)
    monkeypatch.setattr(Bt, "paired_bootstrap_diff", plain)
    out = _run(inputs, inits=(1, 2))
    n_ne, n_w = (out["protocols"][p]["n_test_cases"] for p in ("next_event", "window"))
    # next-event: customer clusters (fewer customers than cases); window: one case per customer-task
    assert seen["cluster"] and all(n == n_ne and c < n for n, c in seen["cluster"])
    assert seen["plain"] and all(n == n_w for n in seen["plain"])
    # each comparison is one bootstrap on the per-case initialization means, not one per initialization
    assert len(seen["cluster"]) == len(F.COMPARISONS) * len(F.KS) * 2


def test_cli_run_requires_explicit_confirmation(monkeypatch):
    monkeypatch.setattr(F, "gates", lambda *a, **k: pytest.fail("gates must not start without confirmation"))
    assert F.main(["run"]) == 1


def test_check_manifests_accepts_the_pinned_state_and_refuses_tampering():
    with pytest.raises(F.FinalRunRefused, match="already locked"):  # after the lock: never again
        F.check_manifests(I.load_manifests())
    m = _pre_lock()
    fz = F.check_manifests(m)
    assert fz["alpha"] == {"nbs": 0.6, "nbp": 0.1} and fz["init_seeds"] == (1, 2, 3) and len(fz["entries"]) == 30
    with pytest.raises(F.FinalRunRefused, match="already locked"):
        F.check_manifests({**m, F.LOCK_MANIFEST: {}})
    with pytest.raises(F.FinalRunRefused, match="not pinned"):
        F.check_manifests({k: v for k, v in m.items() if k != F.FINAL_DATA_MANIFEST})
    bad = copy.deepcopy(m)
    bad[F.CONFIG_MANIFEST]["frozen_config"]["alpha"]["nbs"] = 0.7  # hash no longer matches
    with pytest.raises(F.FinalRunRefused, match="hash"):
        F.check_manifests(bad)
    bad[F.CONFIG_MANIFEST]["frozen_config"]["alpha"]["nbs"] = 0.65  # off-grid, hash recomputed
    bad[F.CONFIG_MANIFEST]["frozen_config_sha256"] = I.canonical_hash(bad[F.CONFIG_MANIFEST]["frozen_config"])
    with pytest.raises(F.FinalRunRefused, match="grid"):
        F.check_manifests(bad)
    short = copy.deepcopy(m)
    short[F.FINAL_DATA_MANIFEST]["final_datasets"] = short[F.FINAL_DATA_MANIFEST]["final_datasets"][:-1]
    with pytest.raises(F.FinalRunRefused, match="reserved"):
        F.check_manifests(short)


# --- summary and preregistered criteria ----------------------------------------------------------

def _proto(base, diffs):
    ms = {n: {f"@{k}": {m: base for m in F.RANK} for k in F.KS} for n in F.MODELS}
    comps = {f"{a} - {b}": {f"@{k}": {m: {"mean_diff": diffs[(a, b)], "ci_low": diffs[(a, b)] - 0.001,
                                          "ci_high": diffs[(a, b)] + 0.001} for m in ("ndcg", "map")} for k in F.KS}
             for a, b in F.COMPARISONS}
    strata = {n: {f"@{k}": {"within_session": {"n": 3, "ndcg": base}, "cross_session": {"n": 7, "ndcg": base}}
                  for k in F.KS} for n in (*F.MODELS, *F.DIAGNOSTICS)}
    init = {n: {f"@{k}": {"ndcg_per_init": [base] * 3, "ndcg_sd": 0.0} for k in F.KS} for n in ("sasrec_full", "static_hybrid")}
    return {"n_test_cases": 10, "models": ms, "comparisons": comps, "initialization": init, "strata": strata,
            "diagnostics": {"oracle_router": {f"@{k}": {m: base + 0.05 for m in F.RANK} for k in F.KS}}}


def _records(rq1_drop, rq2_gain):
    recs = []
    for profile in ("habit_dominant", "balanced", "transition_dominant"):
        for seed, noise in zip((1, 2, 3), (-0.002, 0.0, 0.002)):
            tasks = {}
            for task in ("nbs", "nbp"):
                d_pf = (rq2_gain if (profile, task) == F.RQ2_GROUP else rq1_drop) + noise
                diffs = {("static_hybrid", "personal_frequency"): d_pf, ("static_hybrid", "sasrec_full"): -0.002 + noise,
                         ("sasrec_full", "personal_frequency"): d_pf}
                p = _proto(0.5, diffs)
                tasks[task] = {"protocols": {"next_event": p, "window": {**p, "diagnostics": {}}},
                               "sasrec_runtime": [{"train_seconds": 2.0, "peak_gpu_memory_mib": 50.0, "parameters": 7,
                                                   "next_event_test_inference_ms_per_case": 0.1,
                                                   "window_test_inference_ms_per_case": 0.2}] * 3}
            recs.append({"dataset_id": f"{profile}/seed{seed}", "profile": profile, "seed": seed, "fingerprints": {},
                         "tasks": tasks, "wall_seconds": 10.0})
    return recs


def test_criteria_follow_the_preregistration():
    interp = I.load_manifests()["phase3_1-prereg"]["evaluation"]["interpretation"]
    profiles = ["habit_dominant", "balanced", "transition_dominant"]
    good = F.summarize({"alpha": {}}, _records(rq1_drop=-0.003, rq2_gain=0.15), profiles, interp)["criteria"]
    assert good["RQ1"]["met"] is True and good["RQ2"]["met"] is True and good["outcome"] == interp["RQ1+RQ2"]
    assert good["RQ3"]["status"] == "not evaluable under the preregistered protocol"
    assert "not evidence that adaptive fusion is worse" in good["RQ3"]["reason"]
    bad = F.summarize({"alpha": {}}, _records(rq1_drop=-0.02, rq2_gain=0.15), profiles, interp)["criteria"]
    assert bad["RQ1"]["met"] is False and bad["outcome"] == interp["RQ2 only"]
    s = F.summarize({"alpha": {"nbs": 0.6, "nbp": 0.1}, "init_seeds": [1, 2, 3]}, _records(-0.003, 0.15), profiles, interp)
    g = s["groups"]["balanced/nbs/next_event"]
    assert "oracle_router" not in g["models"] and set(g["diagnostics"]) == {"oracle_router"}
    assert g["comparisons"]["static_hybrid - personal_frequency"]["@1/ndcg"]["n_seeds"] == 3  # seeds, not inits
    assert s["cost"]["sasrec_fits"] == 9 * 2 * 3
    assert set(s["cost"]["definitions"]) == {"sasrec_train_seconds_total", "dataset_wall_seconds_total",
                                             "peak_gpu_memory_mib_max", "test_inference_ms_per_case_mean"}
    assert "SUM of the per-dataset" in s["cost"]["definitions"]["dataset_wall_seconds_total"]
    assert "NOT single-request or online" in s["cost"]["definitions"]["test_inference_ms_per_case_mean"]
    assert "not online latency" in F.to_markdown(s)
    assert "Phase 3.1 final evaluation" in F.to_markdown(s)


def test_lock_writes_frozen_report_and_manifest(processed_dir, tmp_path):
    import shutil
    root = tmp_path / "data"
    shutil.copytree(processed_dir, root / F.FINAL_NS / "balanced" / "seed7" / "processed")
    out = tmp_path / "out"
    out.mkdir()
    s = {"label": "x", "run_config": {"versions": {"final": F.FINAL_VERSION}}, "run_config_sha256": "abc",
         "datasets": [{"dataset_id": "balanced/seed7", "profile": "balanced", "seed": 7}]}
    (out / "phase3_1_report.json").write_text(json.dumps(s), encoding="utf-8")
    (out / "phase3_1_report.md").write_text("# r\n", encoding="utf-8")
    digest = F.lock(out, root, tmp_path / "frozen", tmp_path / "m.json", manifests={})
    m = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert digest == I.sha256_file(tmp_path / "m.json")
    assert m["datasets"][0]["seed_namespace"] == F.FINAL_NS
    assert set(m["frozen_report_sha256"]) >= {"phase3_1_report.json", "phase3_1_report.md"}
    with pytest.raises(F.FinalRunRefused, match="already locked"):
        F.lock(out, root, tmp_path / "frozen2", tmp_path / "m2.json", manifests={F.LOCK_MANIFEST: {}})


# --- CUDA (no final data) ------------------------------------------------------------------------

def _cuda_ok():
    try:
        from bri.seq import runtime as R
        return R.cuda_usable()[0]
    except Exception:
        return False


@pytest.mark.cuda
@pytest.mark.skipif(not _cuda_ok(), reason="no usable CUDA GPU")
def test_cuda_evaluate_task_is_deterministic(inputs):
    a = _run(inputs, inits=(1,), device="cuda:0")
    b = _run(inputs, inits=(1,), device="cuda:0")
    assert a["protocols"] == {**b["protocols"], **{p: {**b["protocols"][p], "evaluation_seconds":
                                                         a["protocols"][p]["evaluation_seconds"]}
                                                    for p in b["protocols"]}}


@pytest.mark.cuda
@pytest.mark.skipif(not _cuda_ok(), reason="no usable CUDA GPU")
def test_cuda_gates_pass_without_opening_any_dataset(monkeypatch):
    import bri.experiments.baselines as B
    import bri.seq.data as D
    monkeypatch.setattr(B, "load_processed", lambda *a, **k: pytest.fail("gates opened a dataset"))
    monkeypatch.setattr(D, "build_seq_cases", lambda *a, **k: pytest.fail("gates built sequence cases"))
    import dataclasses
    with pytest.raises(F.FinalRunRefused, match="already locked"):  # the locked run cannot be repeated
        F.gates(run_tests=False, log=lambda *_: None)
    monkeypatch.setattr(I, "load_manifests", _pre_lock)
    plan = F.gates(run_tests=False, log=lambda *_: None)
    assert plan.alpha == {"nbs": 0.6, "nbp": 0.1} and len(plan.entries) == 30
    F.require_gated_plan(plan)  # the issued object is accepted ...
    with pytest.raises(F.FinalRunRefused):
        F.require_gated_plan(dataclasses.replace(plan))  # ... a copy is not


# --- the one-time lock (pinned) --------------------------------------------------------------

EXPECTED_FROZEN = {"phase3_1_report.json": "92ce1bcb9c671412aeed7fe7170c9d583b00b7b91a4cfb0b7fb92aa9b2bd6f6a",
                   "phase3_1_report.md": "62e510f66dfe2aab22a3956013b4e58e413f4ad799720c5c474e2c50e44b00ba"}
PRE_AMENDMENT_SHA = "e1d7dbcf03f069c68a5836d6ac638af66336e82ed1f7e3d791406291c7cf4469"


def test_lock_manifest_is_pinned_and_references_the_approved_report():
    m = I.load_manifests()
    lock = m[F.LOCK_MANIFEST]
    assert {k: v for k, v in lock["frozen_report_sha256"].items() if k != "FROZEN"} == EXPECTED_FROZEN
    d = I.REPO_ROOT / lock["frozen_report_dir"]
    assert {n: I.sha256_file(d / n) for n in EXPECTED_FROZEN} == EXPECTED_FROZEN
    rep = json.loads((d / "phase3_1_report.json").read_text(encoding="utf-8"))
    assert rep["run_config_sha256"] == lock["phase3_1_run_config_sha256"] == I.canonical_hash(rep["run_config"])
    assert rep["amendments"][0]["pre_amendment_report_sha256"] == PRE_AMENDMENT_SHA
    assert rep["run_config"]["alpha"] == {"nbs": 0.6, "nbp": 0.1}
    assert rep["run_config"]["config_sha256"] == I.MANIFEST_SHA256["phase3_1-config"]
    assert rep["run_config"]["final_data_sha256"] == I.MANIFEST_SHA256["phase3_1-final-data"]
    c = rep["criteria"]
    assert (c["RQ1"]["met"], c["RQ2"]["met"], c["outcome"]) == (False, True, "trade-off, not overall success")
    assert c["RQ3"]["status"] == "not evaluable under the preregistered protocol"


def test_locked_datasets_are_exactly_the_pinned_final_datasets():
    m = I.load_manifests()
    final = {e["dataset_id"]: e for e in m[F.FINAL_DATA_MANIFEST]["final_datasets"]}
    locked = {e["dataset_id"]: e for e in m[F.LOCK_MANIFEST]["datasets"]}
    assert set(locked) == set(final) and len(locked) == 30
    for k, e in locked.items():
        assert e["seed_namespace"] == F.FINAL_NS
        for field in ("processed_files_sha256", "source_files_sha256", "prep_fingerprint", "seed", "behavior_profile"):
            assert e[field] == final[k][field], (k, field)
