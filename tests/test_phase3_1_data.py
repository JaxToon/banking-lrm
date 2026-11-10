"""Phase 3.1 Checkpoint 2: tuning datasets only; final namespace refused; fingerprints pinned."""
import json

import pytest

from bri.experiments import integrity as I
from bri.experiments import phase3_1 as P31

TUNING = [20261140, 20261141, 20261142]


def test_only_the_tuning_namespace_is_authorized(tmp_path):
    manifests = I.load_manifests()
    with pytest.raises(P31.Phase31Refused, match="only 'phase3_1_tuning'"):
        P31.prepare_tuning(manifests, root=tmp_path, namespace=P31.FINAL_NS, log=lambda *_: None)
    with pytest.raises(P31.Phase31Refused, match="not pinned"):
        P31.prepare_tuning({k: v for k, v in manifests.items() if k != "phase3_1-seeds"}, root=tmp_path,
                           log=lambda *_: None)
    assert not any(tmp_path.iterdir())


def test_preflight_uses_the_reserved_tuning_spec():
    spec = P31.preflight(P31.TUNING_NS, I.load_manifests())
    assert spec["seeds"] == TUNING and spec["customers"] == 500 and spec["split_days"] == [16, 7, 7]


def test_tuning_data_manifest_is_pinned_and_consistent():
    manifests = I.load_manifests()
    m = manifests[P31.TUNING_DATA_MANIFEST]
    ds = m["tuning_datasets"]
    assert len(ds) == 9 and m["datasets"] == []  # not a locked benchmark: test splits never evaluated
    assert {(e["behavior_profile"], e["seed"]) for e in ds} == {
        (p, s) for p in ("habit_dominant", "balanced", "transition_dominant") for s in TUNING}
    assert all(e["seed_namespace"] == P31.TUNING_NS and e["generator_version"] == "0.3.0" for e in ds)
    assert m["references"]["prereg_sha256"] == I.MANIFEST_SHA256["phase3_1-prereg"]
    assert m["references"]["seeds_sha256"] == I.MANIFEST_SHA256["phase3_1-seeds"]
    assert not any(e["seed"] in range(20261150, 20261160) for e in ds)


def test_local_tuning_data_matches_the_pinned_fingerprints():
    m = I.load_manifests()[P31.TUNING_DATA_MANIFEST]
    root = I.REPO_ROOT / P31.DATA_ROOT / P31.TUNING_NS
    if not root.exists():
        pytest.skip("Phase 3.1 tuning data not generated locally (git-ignored)")
    for e in m["tuning_datasets"]:
        proc = root / e["behavior_profile"] / f"seed{e['seed']}" / "processed"
        meta = json.loads((proc / "prep_metadata.json").read_text(encoding="utf-8"))
        assert I.processed_fingerprints(proc)["files"] == e["processed_files_sha256"]
        assert meta["source_files_sha256"] == e["source_files_sha256"]
        assert I.prep_fingerprint(meta) == e["prep_fingerprint"]
    final_dirs = [p for p in (I.REPO_ROOT / P31.DATA_ROOT).iterdir() if p.name != P31.TUNING_NS]
    assert final_dirs == []  # no final-namespace data exists
