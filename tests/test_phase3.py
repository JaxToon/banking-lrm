"""Phase 3 final-experiment runner: gates, apple-to-apple evaluation, summary, lock (tiny, CPU)."""
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("optuna")

from bri.experiments import integrity as I  # noqa: E402
from bri.experiments import phase3 as P3  # noqa: E402
from bri.seq import runtime as R  # noqa: E402

SEEDS = [7, 8]
SPEC = {"generator_version": "0.3.0", "profiles": ["balanced", "habit_dominant", "transition_dominant"],
        "customers": 40, "days": 10, "split_days": [4, 3, 3], "seeds": SEEDS}


def tmp_manifests(tmp_path, runtime_sha=None):
    from bri.seq.data import SEQ_DATA_VERSION
    from bri.seq.sasrec import MODEL_VERSION
    from bri.seq.train import TRAIN_VERSION
    res = I.build_seed_reservation("seeds", {P3.NAMESPACE: SPEC}, {}, "")
    frozen = {"model": {"dim": 16, "heads": 2, "blocks": 1, "dropout": 0.1, "max_len": 10},
              "train": {"lr": 3e-3, "weight_decay": 1e-5, "batch_size": 64, "max_epochs": 2, "patience": 5,
                        "grad_clip": 1.0, "seed": 1, "select_metric": "ndcg", "select_k": 3},
              "device": "cpu", "precision": "fp32"}
    rt = R.runtime_fingerprint("cpu")
    cfg = {"suite": "phase3-config", "configs": {}, "datasets": [], "frozen_config": frozen,
           "frozen_config_sha256": I.canonical_hash(frozen), "selection": {"trial": 0},
           "versions": {"model": MODEL_VERSION, "train": TRAIN_VERSION},
           "tuning_data": {"seq_data_version": SEQ_DATA_VERSION},
           "runtime": {"runtime_sha256": runtime_sha or rt["runtime_sha256"], "runtime": rt["runtime"]}}
    mdir = tmp_path / "manifests"
    pinned = {"seeds": I.write_manifest(res, mdir / "seeds.json"),
              "phase3-config": I.write_manifest(cfg, mdir / "phase3-config.json")}
    return mdir, pinned


def plan_for(tmp_path, **kw):
    mdir, pinned = tmp_manifests(tmp_path)
    m = I.load_manifests(mdir, pinned)
    return P3.gates(run_tests=False, manifests=m, expected_device="cpu", profiles=["transition_dominant"],
                    log=lambda *_: None, **kw), mdir, pinned


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p3")
    plan, mdir, pinned = plan_for(tmp)
    summary = P3.run(plan, tmp / "data", tmp / "out", log=lambda *_: None)
    return tmp, plan, summary, mdir, pinned


def test_gates_refuse_before_anything_is_written(tmp_path):
    mdir, pinned = tmp_manifests(tmp_path, runtime_sha="0" * 64)
    m = I.load_manifests(mdir, pinned)
    with pytest.raises(P3.GateError, match="runtime fingerprint differs"):
        P3.gates(run_tests=False, manifests=m, expected_device="cpu", log=lambda *_: None)
    with pytest.raises(P3.GateError, match="frozen SASRec configuration missing"):
        P3.gates(run_tests=False, manifests={k: v for k, v in m.items() if k != "phase3-config"},
                 expected_device="cpu", log=lambda *_: None)
    with pytest.raises(I.LockedTestSplitError, match="not reserved"):
        P3.gates(run_tests=False, manifests=m, expected_device="cpu", seeds=[20261115], log=lambda *_: None)
    assert not (tmp_path / "data").exists()


def test_gates_refuse_a_cpu_fallback_for_the_cuda_experiment(tmp_path, monkeypatch):
    import torch
    mdir, pinned = tmp_manifests(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(R.DeviceMismatchError):
        P3.gates(run_tests=False, manifests=I.load_manifests(mdir, pinned), expected_device="cuda:0",
                 log=lambda *_: None)


def test_every_model_is_reported_for_both_protocols(tiny):
    _, plan, s, _, _ = tiny
    assert s["label"].startswith("SYNTHETIC-DATA EXPERIMENT")
    assert len(s["datasets"]) == 2 and all(d["validation"] == "passed" for d in s["datasets"])
    for proto in ("next_event", "window"):
        for task in ("nbs", "nbp"):
            t = s["profiles"]["transition_dominant"][proto]["tasks"][task]
            assert set(t["models"]) == set(P3.MODELS)
            for m in t["models"].values():
                assert set(m["@3"]) == set(P3.METRICS)
            assert set(t["comparisons"]) == {f"{a} - {b}" for a, b in P3.COMPARISONS}
    ne = s["profiles"]["transition_dominant"]["next_event"]["tasks"]["nbs"]
    assert set(ne["strata"]) == set(P3.MODELS) and set(ne["sasrec_runtime"]) == set(P3.SASREC)
    assert "initialization" in s["statistical_design"]["not_covered"]
    P3.to_markdown(s).encode("ascii")


def test_sasrec_and_classical_models_share_cases_and_metrics_are_consistent(tiny):
    tmp, plan, s, _, _ = tiny
    rec = json.loads((tmp / "out" / "runs" / "transition_dominant" / "seed7" / "phase3_dataset_record.json")
                     .read_text(encoding="utf-8"))
    ne = json.loads((tmp / "out" / "runs" / "transition_dominant" / "seed7" / "next_event_report.json")
                    .read_text(encoding="utf-8"))
    for task in ("nbs", "nbp"):
        t = rec["evaluations"]["next_event"]["tasks"][task]
        assert t["n_test_cases"] == ne["tasks"][task]["n_test_cases"]
        assert t["models"]["personal_frequency"] == ne["tasks"][task]["models"]["personal_frequency"]["test"]
        st = t["strata"]["personal_frequency"]["@1"]
        assert st["within_session"]["n"] + st["cross_session"]["n"] == t["n_test_cases"]


def test_run_is_deterministic(tiny, tmp_path):
    _, _, s, _, _ = tiny
    plan, _, _ = plan_for(tmp_path)
    again = P3.run(plan, tmp_path / "data", tmp_path / "out", log=lambda *_: None)
    strip = lambda x: {k: v for k, v in x.items() if k != "profiles"}  # noqa: E731
    assert again["run_config_sha256"] == s["run_config_sha256"] and strip(again) == strip(s)
    for proto in ("next_event", "window"):
        a = again["profiles"]["transition_dominant"][proto]["tasks"]["nbs"]
        b = s["profiles"]["transition_dominant"][proto]["tasks"]["nbs"]
        assert a["models"] == b["models"] and a["comparisons"] == b["comparisons"]


def test_lock_then_only_exact_reproduction_is_allowed(tiny, tmp_path):
    tmp, plan, s, mdir, pinned = tiny
    digest = P3.lock(tmp / "out", tmp / "data", tmp_path / "frozen", mdir / "phase3.json",
                     manifests=I.load_manifests(mdir, pinned))
    m = I.load_manifests(mdir, {**pinned, "phase3": digest})
    I.verify_frozen_reports(m)
    assert len(m["phase3"]["datasets"]) == 2 and all(e["seed_namespace"] == P3.NAMESPACE for e in m["phase3"]["datasets"])
    ok = P3.gates(run_tests=False, manifests=m, expected_device="cpu", profiles=["transition_dominant"],
                  seeds=[7], log=lambda *_: None)
    P3.check_outputs(ok, tmp_path / "subset")  # subset reproduction into its own directory
    with pytest.raises(I.LockedTestSplitError, match="another configuration"):
        P3.check_outputs(ok, tmp / "out")  # would replace the full aggregate report
    other = P3.gates(run_tests=False, manifests=m, expected_device="cpu", profiles=["balanced"], log=lambda *_: None)
    with pytest.raises(I.LockedTestSplitError, match="not all part of the Phase 3 lock"):
        P3.check_outputs(other, tmp_path / "x")
    with pytest.raises(I.LockedTestSplitError, match="frozen"):
        P3.check_outputs(ok, tmp_path / "frozen")
    with pytest.raises(I.LockedTestSplitError, match="already locked"):
        P3.lock(tmp / "out", tmp / "data", tmp_path / "frozen2", mdir / "phase3b.json", manifests=m)


def test_public_summary_is_derived_from_the_pinned_frozen_report(tmp_path):
    import shutil
    m = I.load_manifests()
    s = P3.public_summary(manifests=m)
    assert s["derived_from"]["sha256"] == m["phase3"]["frozen_report_sha256"]["phase3_report.json"]
    assert len(s["groups"]) == 12 and len(s["limitations"]) == 5
    g = s["groups"]["transition_dominant/next_event/nbs"]
    assert set(g["strata_ndcg@1"]) == {"personal_frequency", "sasrec_full"}
    committed = P3.SUMMARY_DIR / "phase3_summary.json"
    if (P3.I.REPO_ROOT / committed).exists():  # the committed summary is exactly what the generator produces
        assert json.loads((P3.I.REPO_ROOT / committed).read_text(encoding="utf-8")) == json.loads(json.dumps(s))
    fake = tmp_path / "phase3_report.json"
    shutil.copyfile(P3.I.REPO_ROOT / "benchmarks" / "phase3" / "phase3_report.json", fake)
    fake.write_text(fake.read_text(encoding="utf-8").replace('"label"', '"label" ', 1), encoding="utf-8")
    with pytest.raises(I.BenchmarkIntegrityError, match="does not match the frozen report hash"):
        P3.public_summary(fake, manifests=m)
