"""Phase 3 Optuna HPO: search space, sampler reproducibility, objective, persistence, freeze."""
import dataclasses
import json

import numpy as np
import pytest

pytest.importorskip("torch")
optuna = pytest.importorskip("optuna")

from bri.experiments import integrity as I  # noqa: E402
from bri.seq import hpo as H  # noqa: E402
from test_sasrec import TIEBREAK, VOCAB, synthetic_cases  # noqa: E402

RUNTIME = {"runtime_sha256": "cpu-test", "runtime": {"device": "cpu"}}


def as_task(seq, task):
    return dataclasses.replace(seq, task=task, cases=seq.cases.assign(task=task))


def fake_data(sizes=None):
    """3 profiles x 1 seed x 2 tasks of synthetic cases; group sizes differ on purpose."""
    sizes = sizes or {"habit_dominant": 40, "balanced": 120, "transition_dominant": 60}
    datasets, cache = [], {}
    for i, (profile, n) in enumerate(sizes.items()):
        datasets.append({"profile": profile, "seed": 1, "candidates": {"nbs": VOCAB, "nbp": VOCAB},
                         "tiebreak": {"nbs": TIEBREAK, "nbp": TIEBREAK}, "fingerprints": {"x": profile}})
        for L in H.SEARCH_SPACE["max_len"]:
            cache[(i, L)] = {(t, s): as_task(synthetic_cases(n if s == "train" else n // 2, s, 10 * i + (s == "val"),
                                                             max_len=L), t)
                             for t in ("nbs", "nbp") for s in ("train", "val")}
    return H.TuningData(datasets, cache)


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setitem(H.TRAIN_FIXED, "max_epochs", 2)


def test_search_space_is_the_approved_one():
    s = H.SEARCH_SPACE
    assert s["dim"] == [16, 32, 64] and s["heads"] == [1, 2, 4] and s["blocks"] == [1, 2, 3]
    assert s["dropout"] == [0.0, 0.3] and s["lr"] == [1e-4, 5e-3] and s["weight_decay"] == [1e-6, 1e-3]
    assert s["batch_size"] == [32, 64, 128] and s["max_len"] == [10, 20, 30]


def test_invalid_architectures_are_rejected():
    base = {"dim": 32, "heads": 2, "max_len": 20}
    H.validate_params(base)
    with pytest.raises(H.InvalidTrialParams, match="divisible"):
        H.validate_params({**base, "dim": 30, "heads": 4})
    with pytest.raises(H.InvalidTrialParams, match="max_len"):
        H.validate_params({**base, "max_len": 50})


def test_sampler_configuration_is_reproducible():
    def params(seed):
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed), direction="maximize")
        study.optimize(lambda t: float(sum(v for v in H.suggest(t).values() if isinstance(v, (int, float)))),
                       n_trials=15)
        return [t.params for t in study.trials]
    assert params(H.SAMPLER_SEED) == params(H.SAMPLER_SEED)
    assert params(H.SAMPLER_SEED) != params(H.SAMPLER_SEED + 1)


def test_objective_is_an_unweighted_mean_of_six_groups_and_records_every_score(fast, tmp_path):
    study = H.run_hpo(fake_data(), 2, tmp_path / "s.db", "cpu", RUNTIME, log=None)
    for t in study.trials:
        g = t.user_attrs["group_scores"]
        assert set(g) == {f"{p}/{k}" for p in ("habit_dominant", "balanced", "transition_dominant")
                          for k in ("nbs", "nbp")}
        assert t.value == pytest.approx(round(float(np.mean(list(g.values()))), 6))
        assert len(t.user_attrs["per_dataset"]) == 6 and t.user_attrs["seconds"] > 0


def test_study_persists_in_sqlite_resumes_and_refuses_protocol_changes(fast, tmp_path, monkeypatch):
    db = tmp_path / "s.db"
    data = fake_data()
    H.run_hpo(data, 2, db, "cpu", RUNTIME, log=None)
    reloaded = H.load_study(db)
    assert len(reloaded.trials) == 2 and reloaded.user_attrs["protocol"]["search_space"] == H.SEARCH_SPACE
    H.run_hpo(data, 3, db, "cpu", RUNTIME, log=None)  # resume to 3 trials in total
    assert len(H.load_study(db).trials) == 3
    with pytest.raises(I.BenchmarkIntegrityError, match="different protocol"):
        H.run_hpo(data, 4, db, "cpu", {"runtime_sha256": "other-device", "runtime": {}}, log=None)
    monkeypatch.setitem(H.TRAIN_FIXED, "patience", 9)
    with pytest.raises(I.BenchmarkIntegrityError, match="different protocol"):
        H.run_hpo(data, 4, db, "cpu", RUNTIME, log=None)


def test_tuning_data_builds_only_train_and_val_without_test_events(processed_dir, monkeypatch, tmp_path):
    import shutil
    from bri.seq import data as D
    root = tmp_path / "data"
    proc = root / H.NAMESPACE / "balanced" / "seed1" / "processed"
    shutil.copytree(processed_dir, proc)
    meta = json.loads((proc / "prep_metadata.json").read_text(encoding="utf-8"))
    meta["source_generation"]["behavior_profile"] = "balanced"
    meta["source"]["seed"] = 1  # relabel the fixture as one tuning dataset (checksums of CSVs unchanged)
    (proc / "prep_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    calls = []
    real = D.build_seq_cases

    def spy(inter, cutoffs, candidates, splits, max_len, min_history=1):
        calls.append((tuple(splits), set(inter["split"])))
        return real(inter, cutoffs, candidates, splits, max_len, min_history)

    monkeypatch.setattr(D, "build_seq_cases", spy)
    data = H.TuningData.load(root, {"profiles": ["balanced"], "seeds": [1]})
    data.seqs(0, 10)
    assert calls and all(splits == ("train", "val") and "test" not in seen for splits, seen in calls)


def test_tuning_data_refuses_tampered_processed_files(processed_dir, tmp_path):
    import shutil
    proc = tmp_path / "data" / H.NAMESPACE / "balanced" / "seed1" / "processed"
    shutil.copytree(processed_dir, proc)
    meta = json.loads((proc / "prep_metadata.json").read_text(encoding="utf-8"))
    meta["source_generation"]["behavior_profile"], meta["source"]["seed"] = "balanced", 1
    (proc / "prep_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with open(proc / "interactions.csv", "a", encoding="utf-8") as f:
        f.write("tampered\n")
    with pytest.raises(I.BenchmarkIntegrityError, match="does not match prep_metadata"):
        H.TuningData.load(tmp_path / "data", {"profiles": ["balanced"], "seeds": [1]})


def test_selection_rule_report_and_immutable_freeze(fast, tmp_path):
    db = tmp_path / "s.db"
    study = H.run_hpo(fake_data(), 3, db, "cpu", RUNTIME, log=None)
    best = H.select_best(study)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert best.value == max(t.value for t in done)
    r = H.report(study)
    assert r["best"]["number"] == best.number and r["counts"]["COMPLETE"] == 3
    assert set(r["best"]["group_rank_among_complete_trials"]) == set(best.user_attrs["group_scores"])
    H.to_markdown(r).encode("ascii")
    path = tmp_path / "phase3-config.json"
    digest = H.freeze(study, db, path)
    m = I.load_manifests(tmp_path, {"phase3-config": digest})["phase3-config"]
    assert m["frozen_config"]["model"]["max_len"] in (10, 20, 30)
    assert m["frozen_config_sha256"] == I.canonical_hash(m["frozen_config"])
    assert m["selection"]["trial"] == best.number and m["frozen_config"]["device"] == "cuda:0"
    with pytest.raises(I.BenchmarkIntegrityError, match="immutable"):
        H.freeze(study, db, path)


def test_hpo_refuses_to_continue_after_freeze_and_without_namespace():
    with pytest.raises(I.LockedTestSplitError, match="already frozen"):
        H.preflight({**I.load_manifests(), "phase3-config": {}})
    with pytest.raises(I.LockedTestSplitError, match="not registered"):
        H.preflight({})


def test_repository_frozen_config_is_pinned_consistent_and_within_the_search_space():
    m = I.load_manifests()["phase3-config"]
    cfg = m["frozen_config"]
    assert m["frozen_config_sha256"] == I.canonical_hash(cfg)
    H.validate_params({**cfg["model"], **cfg["train"]})
    for k in ("dim", "heads", "blocks", "batch_size", "max_len"):
        v = cfg["model"].get(k, cfg["train"].get(k))
        assert v in H.SEARCH_SPACE[k]
    assert cfg["device"] == "cuda:0" and cfg["precision"] == "fp32"
    assert m["study"]["trials_by_state"]["COMPLETE"] == 30
    assert m["tuning_data"]["seq_data_version"] == "0.2.0"
    assert set(m["tuning_data"]["datasets"]) == {f"{p}/seed{s}" for p in ("habit_dominant", "balanced",
                                                  "transition_dominant") for s in (20261120, 20261121, 20261122)}
    assert not any(e["seed"] in range(20261130, 20261140) for e in m["datasets"])
