"""Phase 3.1 pre-registration: built manifest is consistent; tampering is detected."""
import copy

import pytest

from bri.experiments import integrity as I
from bri.experiments import prereg_phase3_1 as PR


@pytest.fixture(scope="module")
def ctx():
    manifests = {k: v for k, v in I.load_manifests().items() if k != PR.NAME}
    return manifests, PR.build(manifests)


def test_built_prereg_is_consistent_with_the_design_document(ctx):
    manifests, m = ctx
    assert PR.verify(m, manifests) == []
    assert m["design_document"]["sha256"] == I.sha256_file(PR.DESIGN_DOC)
    assert sorted(map(int, m["decisions"])) == list(range(1, 16))
    assert m["evaluation"]["criteria"]["margin_ndcg@1"] == 0.01
    assert m["gate"]["support"]["valid_min_per_class_per_fold"] == 50
    assert m["gate"]["support"]["exploratory_min_per_class_per_fold"] == 30
    assert m["seed_policy"]["initialization_seeds"] == [1, 2, 3]
    assert m["datasets"] == [] and "seed_namespaces" not in m  # pre-registration reserves no seeds


def test_prereg_references_the_locked_phase3_artifacts(ctx):
    manifests, m = ctx
    assert m["references"]["phase3_frozen_config_sha256"] == manifests["phase3-config"]["frozen_config_sha256"]
    assert m["references"]["phase3_lock_manifest_sha256"] == I.MANIFEST_SHA256["phase3"]


def test_changed_design_document_is_detected(ctx, tmp_path):
    manifests, m = ctx
    doc = tmp_path / "design.md"
    doc.write_text(PR.DESIGN_DOC.read_text(encoding="utf-8") + "\nlate edit\n", encoding="utf-8")
    assert any("sha256 differs" in p for p in PR.verify(m, manifests, doc))


@pytest.mark.parametrize("mutate,expected", [
    (lambda m: m["decisions"].pop("15"), "decisions 1-15"),
    (lambda m: m["models"]["static_fusion"].__setitem__("alpha_grid", [0.0, 0.5]), "alpha grid"),
    (lambda m: m["gate"]["support"].__setitem__("valid_min_per_class_per_fold", 20), "support thresholds"),
    (lambda m: m["evaluation"]["criteria"].__setitem__("margin_ndcg@1", 0.02), "margin"),
    (lambda m: m["references"].__setitem__("phase3_frozen_config_sha256", "0" * 64), "frozen configuration"),
    (lambda m: m["features"]["subsets"].append("P+X"), "unknown group"),
])
def test_inconsistent_prereg_is_rejected(ctx, mutate, expected):
    manifests, m = ctx
    bad = copy.deepcopy(m)
    mutate(bad)
    assert any(expected in p for p in PR.verify(bad, manifests))


def test_prereg_refuses_already_reserved_namespaces(ctx):
    manifests, m = ctx
    res = I.build_seed_reservation("x", {"phase3_1_final": {"seeds": [99]}}, {}, "")
    assert any("already reserved" in p for p in PR.verify(m, {**manifests, "x": res}))


def test_decision_15_clarification_and_notes_are_locked(ctx):
    _, m = ctx
    rules = m["final_gate_rules"]
    assert rules["status_table"]["valid"]["min_per_class_per_fold"] == 50
    assert rules["status_table"]["rejected"]["below_per_class_per_fold"] == 30
    assert m["status_labels"]["not_evaluable"] == "not evaluable under the preregistered protocol"
    assert any("never dropped from the confirmatory macro-average" in r for r in rules["RQ3"])
    assert any("no fallback from gated to static" in r for r in rules["RQ1_RQ2"])
    assert any("never replace unavailable values with zero" in r for r in m["reporting_rules"])
    assert len(m["documentation_notes"]) == 2
    assert "dataset generation" in m["approval"]["not_authorized_by_this_manifest"]


def test_pinned_prereg_matches_the_builder_when_present():
    manifests = I.load_manifests()
    if PR.NAME not in manifests:
        pytest.skip("pre-registration not pinned yet")
    others = {k: v for k, v in manifests.items() if k != PR.NAME}
    pinned = manifests[PR.NAME]
    assert pinned == PR.build(others)  # the pinned manifest is exactly what the approved inputs produce
    assert PR.verify(pinned, others) == []
    assert I.sha256_file(I.MANIFEST_DIR / f"{PR.NAME}.json") == I.MANIFEST_SHA256[PR.NAME]
    assert not any(ns.startswith("phase3_1") for ns in I.seed_reservations(manifests).values())
