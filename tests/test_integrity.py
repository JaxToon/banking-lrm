"""Benchmark-lock integrity: every tampering scenario must be refused."""
import json
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from bri.experiments import integrity as I
from bri.experiments.baselines import (LOCKED_DATASETS, experiment_config, protocol_config, run_baselines,
                                       run_next_event)
from bri.experiments.integrity import BenchmarkIntegrityError, LockedTestSplitError
from bri.models.baselines import SUITES
from bri.prep.build import build

FAST = dict(ks=(1, 3), n_boot=50, seed=42, random_seeds=2)
SUITE = "phase2.2"


def _config(n_boot=50):
    return experiment_config(SUITE, "window", protocol_config("window", (1, 3), "ndcg", 3, n_boot, 42, 2, 0.95))


@pytest.fixture
def locked(dataset_dir, tmp_path):
    """A private copy of the session dataset, registered as a locked benchmark in a temp manifest dir."""
    data, proc, mdir = tmp_path / "data", tmp_path / "processed", tmp_path / "manifests"
    shutil.copytree(dataset_dir, data)
    meta = build(data, proc, val_days=3, test_days=3, min_history=1)
    entry = I.dataset_entry(data, proc, meta, "fixture", ["window"])
    manifest = I.build_manifest(SUITE, {"window": _config()}, [entry], {"generator": "0.2.0"})
    digest = I.write_manifest(manifest, mdir / "locked-fixture.json")
    pinned = {"locked-fixture": digest}
    return {"data": data, "proc": proc, "mdir": mdir, "pinned": pinned,
            "manifests": I.load_manifests(mdir, pinned)}


def _run(lk, **kw):
    return run_baselines(lk["proc"], **{**FAST, "suite": SUITE, **kw}, manifests=lk["manifests"])


def _rewrite_csv(path, mutate):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    mutate(df)
    df.to_csv(path, index=False, lineterminator="\n")


def _rerecord(proc, name):
    meta = json.loads((proc / "prep_metadata.json").read_text(encoding="utf-8"))
    meta["sha256"][name] = I.sha256_file(proc / name)
    (proc / "prep_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def test_untampered_reproduction_is_allowed(locked):
    report = _run(locked)
    assert report["suite"] == SUITE


@pytest.mark.parametrize("name", I.PROCESSED_FILES)
def test_modified_processed_file_is_refused(locked, name):
    _rewrite_csv(locked["proc"] / name, lambda df: df.drop(df.index[-1], inplace=True))
    with pytest.raises(BenchmarkIntegrityError, match="does not match prep_metadata"):
        _run(locked)


@pytest.mark.parametrize("name", I.PROCESSED_FILES)
def test_modified_processed_file_with_forged_checksum_is_refused(locked, name):
    _rewrite_csv(locked["proc"] / name, lambda df: df.drop(df.index[-1], inplace=True))
    _rerecord(locked["proc"], name)
    with pytest.raises(BenchmarkIntegrityError, match=f"processed file {name}|prep metadata"):
        _run(locked)


def test_changed_consent_input_is_refused(locked):
    def flip(df):
        df.loc[0, "allowed"] = "False" if df.loc[0, "allowed"] == "True" else "True"
    _rewrite_csv(locked["data"] / "consent_preferences.csv", flip)
    build(locked["data"], locked["proc"], val_days=3, test_days=3, min_history=1)
    with pytest.raises(BenchmarkIntegrityError, match="consent_preferences.csv"):
        _run(locked)


def test_changed_catalog_input_is_refused(locked):
    def rename(df):
        df.loc[0, "item_name"] = df.loc[0, "item_name"] + " v2"
    _rewrite_csv(locked["data"] / "catalog.csv", rename)
    build(locked["data"], locked["proc"], val_days=3, test_days=3, min_history=1)
    with pytest.raises(BenchmarkIntegrityError, match="catalog.csv"):
        _run(locked)


def test_edited_events_of_a_locked_dataset_are_refused(locked):
    """Matched by generator identity even though the events checksum changed."""
    def edit(df):
        df.loc[0, "event_value"] = "3" if df.loc[0, "event_value"] != "3" else "2"
    _rewrite_csv(locked["data"] / "events.csv", edit)
    build(locked["data"], locked["proc"], val_days=3, test_days=3, min_history=1)
    with pytest.raises(BenchmarkIntegrityError, match="events.csv of locked dataset|events of locked dataset"):
        _run(locked)


def test_tampered_prep_metadata_is_refused(locked):
    path = locked["proc"] / "prep_metadata.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["params"]["min_history"] = 2
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(BenchmarkIntegrityError, match="prep metadata"):
        _run(locked)


def test_changed_protocol_under_locked_suite_is_refused(locked):
    with pytest.raises(BenchmarkIntegrityError, match="configuration differs"):
        _run(locked, n_boot=60)
    with pytest.raises(BenchmarkIntegrityError, match="configuration differs"):
        _run(locked, select_k=1)


def test_changed_model_grid_under_registered_suite_is_refused(locked, monkeypatch, processed_dir):
    changed = dict(SUITES[SUITE])
    changed["mf"] = ("mf", [{"signal": "target", "factors": 9, "reg": 0.1, "alpha": 5.0}])
    monkeypatch.setitem(SUITES, SUITE, changed)
    with pytest.raises(BenchmarkIntegrityError, match="no longer matches its registered model grids"):
        _run(locked)
    # The suite identifier is protected for any dataset, not only for the locked one.
    with pytest.raises(BenchmarkIntegrityError, match="registered model grids"):
        run_baselines(processed_dir, **FAST, suite=SUITE, manifests=locked["manifests"])


def test_other_suite_and_other_evaluation_are_refused(locked):
    with pytest.raises(LockedTestSplitError, match="locked by suite 'phase2.2'"):
        _run(locked, suite="phase2.3")
    with pytest.raises(LockedTestSplitError, match="locked by suite 'phase2.2'"):
        run_next_event(locked["proc"], **FAST, manifests=locked["manifests"])


def test_modified_manifest_is_refused(locked):
    path = locked["mdir"] / "locked-fixture.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"n_boot": 50', '"n_boot": 60'), encoding="utf-8")
    with pytest.raises(BenchmarkIntegrityError, match="was modified"):
        I.load_manifests(locked["mdir"], locked["pinned"])


def test_unregistered_or_missing_manifest_is_refused(locked):
    (locked["mdir"] / "forged.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BenchmarkIntegrityError, match="unregistered"):
        I.load_manifests(locked["mdir"], locked["pinned"])
    (locked["mdir"] / "forged.json").unlink()
    (locked["mdir"] / "locked-fixture.json").unlink()
    with pytest.raises(BenchmarkIntegrityError, match="missing"):
        I.load_manifests(locked["mdir"], locked["pinned"])


def test_manifests_are_write_once(locked):
    with pytest.raises(BenchmarkIntegrityError, match="immutable"):
        I.write_manifest({}, locked["mdir"] / "locked-fixture.json")


def test_unlocked_dataset_still_needs_consistent_processed_files(processed_dir, tmp_path):
    altered = tmp_path / "p"
    shutil.copytree(processed_dir, altered)
    _rewrite_csv(altered / "eval_cases.csv", lambda df: df.drop(df.index[0], inplace=True))
    with pytest.raises(BenchmarkIntegrityError, match="does not match prep_metadata"):
        run_baselines(altered, **FAST, suite="phase2.3")


def test_canonical_config_hash_is_order_independent():
    a = {"b": [1, {"y": 1, "x": 2}], "a": 1}
    b = {"a": 1, "b": [1, {"x": 2, "y": 1}]}
    assert I.canonical_hash(a) == I.canonical_hash(b)
    assert I.canonical_hash(a) != I.canonical_hash({**a, "a": 2})
    assert I.composite_fingerprint({"x": "1", "y": "2"}) != I.composite_fingerprint({"x": "1", "y": "3"})


# --- the repository's own manifests -------------------------------------------------------

def test_repository_manifests_and_frozen_reports_are_intact():
    manifests = I.load_manifests()
    assert {"benchmark-v1", "phase2.2"} <= set(manifests)
    I.verify_frozen_reports(manifests)
    for m in manifests.values():
        if "seed_namespaces" in m or "frozen_config" in m or "design_document" in m:  # no reports by design
            continue
        assert m["frozen_report_sha256"], "every locked benchmark pins its frozen report files"


def test_legacy_lock_table_agrees_with_manifests():
    manifests = I.load_manifests()
    for events_sha, suite in LOCKED_DATASETS.items():
        entries = [m for m in manifests.values() for e in m["datasets"]
                   if e["source_files_sha256"]["events.csv"] == events_sha]
        assert [m["suite"] for m in entries] == [suite]


def test_registered_suites_still_match_their_manifests():
    from bri.models.baselines import KEY_COMPARISONS
    manifests = I.load_manifests()
    for m in manifests.values():
        for evaluation, c in m["configs"].items():
            suite = c["config"]["suite"]
            if suite == "phase3-sasrec":  # not a classical suite: pinned via the frozen configuration
                frozen = manifests["phase3-config"]["frozen_config_sha256"]
                assert all(g == [{"frozen_config_sha256": frozen}] for g in
                           (v["grid"] for v in c["config"]["models"].values()))
                continue
            I.check_suite_definition(suite, evaluation, SUITES[suite], KEY_COMPARISONS[suite], manifests)


def test_integrity_cli_passes_on_repository():
    r = subprocess.run([sys.executable, "-m", "bri.experiments.integrity"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "manifest ok" in r.stdout


def test_retrospective_manifests_declare_their_provenance():
    manifests = I.load_manifests()
    for suite in ("benchmark-v1", "phase2.2"):
        assert manifests[suite]["provenance"] == "retrospective"
        assert manifests[suite]["notes"].startswith("RETROSPECTIVE PROVENANCE")
    assert "provenance" not in manifests["phase2.3"]  # written at its final evaluation
