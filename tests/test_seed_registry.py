"""Phase 3 seed namespaces: reservation, collision protection, and refusal without side effects."""
import json

import pytest

from bri.experiments import integrity as I
from bri.experiments.robustness import RobustnessConfig, run_robustness

TUNING = list(range(20261120, 20261123))
FINAL = list(range(20261130, 20261140))


def test_repository_seed_reservation_is_pinned_and_matches_the_approved_ranges():
    manifests = I.load_manifests()
    res = manifests["phase3-seeds"]["seed_namespaces"]
    assert res["phase3_tuning"]["seeds"] == TUNING
    assert res["phase3_final"]["seeds"] == FINAL
    assert not set(TUNING) & set(FINAL)
    for spec in res.values():
        assert spec["generator_version"] == "0.3.0"
        assert spec["profiles"] == ["balanced", "habit_dominant", "transition_dominant"]


def test_reserved_seeds_collide_with_no_previously_registered_seed():
    manifests = I.load_manifests()
    reserved = set(TUNING) | set(FINAL)
    recorded = manifests["phase3-seeds"]["checked_against"]
    assert set(recorded["manifests"]) == {"benchmark-v1", "phase2.2", "phase2.3"}
    # No collision with any manifest that existed when the seeds were reserved.
    used_then = I.seed_uses({k: manifests[k] for k in recorded["manifests"]})
    assert not reserved & set(used_then)
    assert {42, 20261109, *range(20261110, 20261120)} <= set(used_then)
    assert set(recorded["seeds_in_use"]) == set(used_then)
    # Later manifests may use reserved seeds only from their own namespace.
    owner = I.seed_reservations(manifests)
    for seed, uses in I.seed_uses(manifests).items():
        if seed in owner:
            assert all(ns == owner[seed] for _, ns in uses), seed


def test_namespace_check_accepts_own_seeds_and_refuses_everything_else():
    manifests = I.load_manifests()
    I.check_seed_namespace(TUNING, "phase3_tuning", manifests)
    I.check_seed_namespace(FINAL, "phase3_final", manifests)
    with pytest.raises(I.LockedTestSplitError, match="reserved for 'phase3_final'"):
        I.check_seed_namespace([FINAL[0]], "phase3_tuning", manifests)
    with pytest.raises(I.LockedTestSplitError, match="not reserved"):
        I.check_seed_namespace([20261115], "phase3_final", manifests)  # a locked Phase 2.3 seed
    with pytest.raises(I.LockedTestSplitError, match="not reserved"):
        I.check_seed_namespace([20261125], "phase3_final", manifests)
    with pytest.raises(I.LockedTestSplitError, match="not registered"):
        I.check_seed_namespace(TUNING, "phase4", manifests)


def test_building_a_reservation_refuses_used_or_reserved_seeds():
    manifests = I.load_manifests()
    for seeds, match in (([20261115], "already used"), ([42], "already used"),
                         ([20261121], "already reserved")):
        with pytest.raises(I.BenchmarkIntegrityError, match=match):
            I.build_seed_reservation("x", {"ns": {"seeds": seeds}}, manifests, "")
    with pytest.raises(I.BenchmarkIntegrityError, match="appears in both"):
        I.build_seed_reservation("x", {"a": {"seeds": [1]}, "b": {"seeds": [1]}}, manifests, "")


def _write_pinned(tmp_path, manifests: dict):
    pinned = {}
    for name, m in manifests.items():
        pinned[name] = I.write_manifest(m, tmp_path / f"{name}.json")
    return pinned


def test_loading_refuses_reservations_that_collide_with_locked_datasets(tmp_path):
    bench = {"suite": "b", "configs": {}, "datasets": [{"seed": 7, "dataset_id": "x"}]}
    res = I.build_seed_reservation("r", {"ns": {"seeds": [8]}}, {}, "")
    res["seed_namespaces"]["ns"]["seeds"] = [7]  # forged after the collision check
    pinned = _write_pinned(tmp_path, {"b": bench, "r": res})
    with pytest.raises(I.BenchmarkIntegrityError, match="seed 7 is reserved for namespace 'ns'"):
        I.load_manifests(tmp_path, pinned)


def test_a_dataset_may_use_a_reserved_seed_only_from_its_own_namespace(tmp_path):
    res = I.build_seed_reservation("r", {"ns": {"seeds": [7]}}, {}, "")
    own = {"suite": "f", "configs": {}, "datasets": [{"seed": 7, "seed_namespace": "ns"}]}
    pinned = _write_pinned(tmp_path / "ok", {"r": res, "f": own})
    I.load_manifests(tmp_path / "ok", pinned)
    other = {"suite": "f", "configs": {}, "datasets": [{"seed": 7, "seed_namespace": "other"}]}
    pinned = _write_pinned(tmp_path / "bad", {"r": res, "f": other})
    with pytest.raises(I.BenchmarkIntegrityError, match="used by manifest"):
        I.load_manifests(tmp_path / "bad", pinned)


def test_tampered_seed_reservation_is_refused(tmp_path):
    res = I.build_seed_reservation("r", {"ns": {"seeds": [7]}}, {}, "")
    pinned = _write_pinned(tmp_path, {"r": res})
    path = tmp_path / "r.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["seed_namespaces"]["ns"]["seeds"] = [7, 9]
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(I.BenchmarkIntegrityError, match="was modified"):
        I.load_manifests(tmp_path, pinned)


@pytest.mark.parametrize("seed", [TUNING[0], FINAL[-1]])
def test_phase23_runner_refuses_reserved_seeds_before_writing_anything(seed, tmp_path):
    cfg = RobustnessConfig(profiles=("balanced",), seeds=(seed,), customers=20, days=10, val_days=3, test_days=3)
    with pytest.raises(I.LockedTestSplitError, match="reserved for namespace"):
        run_robustness(cfg, tmp_path / "data", tmp_path / "out", log=lambda *_: None)
    assert not (tmp_path / "data").exists() and not (tmp_path / "out").exists()
