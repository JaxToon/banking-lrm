"""Phase 2.3 multi-seed robustness orchestration (tiny configuration)."""
import json
import math

import pytest

from bri.experiments import integrity as I
from bri.experiments.robustness import (T95, RobustnessConfig, across_seed_stats, lock_manifest, run_robustness,
                                        to_markdown)
from bri.tasks import SYNTHETIC_LABEL

TINY = RobustnessConfig(profiles=("transition_dominant",), seeds=(5, 6), customers=40, days=10,
                        val_days=3, test_days=3, n_boot=20, random_seeds=1)
FIVE = {"popularity", "personal_frequency", "itemknn_self", "itemknn_noself", "mf"}


def test_across_seed_stats_hand_calculated():
    s = across_seed_stats([1.0, 2.0, 3.0])
    half = 4.303 * 1.0 / math.sqrt(3)
    assert s == {"n_seeds": 3, "mean": 2.0, "sd": 1.0, "ci_low": round(2 - half, 6), "ci_high": round(2 + half, 6)}
    assert across_seed_stats([0.5])["sd"] is None
    assert across_seed_stats([])["mean"] is None
    assert T95[9] == 2.262  # 10 seeds


def test_config_validation():
    with pytest.raises(ValueError):
        RobustnessConfig(profiles=("nope",))
    with pytest.raises(ValueError):
        RobustnessConfig(seeds=(1, 1))
    with pytest.raises(ValueError):
        RobustnessConfig(confidence=0.9)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("rob")
    summary = run_robustness(TINY, root / "data", root / "out", manifests={}, log=lambda *_: None)
    return root, summary


def test_summary_structure(tiny_run):
    root, s = tiny_run
    assert s["label"] == SYNTHETIC_LABEL
    assert "not pooled" in s["statistical_design"]["across_seed"]
    assert len(s["datasets"]) == 2 and all(d["validation"] == "passed" for d in s["datasets"])
    prof = s["profiles"]["transition_dominant"]
    assert "invented" not in prof  # params live in behavior_params
    for ev in ("window", "next_event"):
        e = prof["evaluations"][ev]
        assert set(e["cold_start"]) == {"val", "test"}
        for task in ("nbs", "nbp"):
            t = e["tasks"][task]
            assert set(t["models"]) == FIVE
            for comp in t["comparisons"]:
                if comp != "itemknn_self - itemknn_noself":
                    assert comp.endswith("- personal_frequency")
            c = t["comparisons"]["itemknn_self - personal_frequency"]["@1/ndcg"]
            assert c["seeds_ci_above_0"] + c["seeds_ci_below_0"] + c["seeds_ci_includes_0"] == t["n_seeds_with_test_cases"]
    assert set(prof["evaluations"]["next_event"]["session_strata"]["nbs"]) == FIVE
    for seed in (5, 6):
        d = root / "out" / "runs" / "transition_dominant" / f"seed{seed}"
        assert (d / "window_report.json").exists() and (d / "next_event_report.json").exists()
    md = to_markdown(s)
    md.encode("ascii")
    assert "SYNTHETIC-DATA EXPERIMENT" in md and "invented experimental assumptions" in md


def test_multi_seed_run_is_deterministic(tiny_run, tmp_path):
    _, first = tiny_run
    again = run_robustness(TINY, tmp_path / "data", tmp_path / "out", manifests={}, log=lambda *_: None)
    assert json.dumps(again, sort_keys=True) == json.dumps(first, sort_keys=True)


def test_lock_manifest_allows_reproduction_and_refuses_changes(tiny_run, tmp_path):
    root, summary = tiny_run
    path = tmp_path / "m" / "tiny.json"
    digest = lock_manifest(TINY, root / "data", summary, path, None)
    manifests = I.load_manifests(path.parent, {"tiny": digest})
    assert len(manifests["tiny"]["datasets"]) == 2
    again = run_robustness(TINY, root / "data", root / "out2", manifests=manifests, log=lambda *_: None)
    assert json.dumps(again, sort_keys=True) == json.dumps(summary, sort_keys=True)
    changed = RobustnessConfig(**{**TINY.__dict__, "n_boot": 30})
    with pytest.raises(I.BenchmarkIntegrityError, match="configuration differs"):
        run_robustness(changed, root / "data", root / "out3", manifests=manifests, log=lambda *_: None)


def _tree_hashes(root):
    return {p.relative_to(root).as_posix(): I.sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def test_refused_partial_runs_never_touch_locked_data_or_reports(tiny_run, tmp_path):
    """A refused run must fail before generating, building or writing anything."""
    root, summary = tiny_run
    path = tmp_path / "m" / "tiny.json"
    manifests = I.load_manifests(path.parent, {"tiny": lock_manifest(TINY, root / "data", summary, path, None)})
    before = _tree_hashes(root)
    subset = dict(TINY.__dict__, seeds=(5,))
    for kwargs, error in ((dict(customers=30), "different data parameters"),
                          (dict(test_days=2), "different data parameters"),
                          (dict(n_boot=30), "configuration differs")):
        cfg = RobustnessConfig(**{**subset, **kwargs})
        with pytest.raises(I.LockedTestSplitError, match=error):
            run_robustness(cfg, root / "data", tmp_path / "new_out", manifests=manifests, log=lambda *_: None)
    assert _tree_hashes(root) == before
    assert not (tmp_path / "new_out").exists()


def test_partial_reproduction_cannot_replace_an_existing_aggregate_report(tiny_run, tmp_path):
    root, summary = tiny_run
    out = tmp_path / "out"
    out.mkdir()
    (out / "robustness_report.json").write_text(json.dumps(summary), encoding="utf-8")
    partial = RobustnessConfig(**{**TINY.__dict__, "seeds": (5,)})
    with pytest.raises(I.LockedTestSplitError, match="different configuration"):
        run_robustness(partial, root / "data", out, manifests={}, log=lambda *_: None)
    assert json.loads((out / "robustness_report.json").read_text(encoding="utf-8")) == summary
    # A subset reproduction into its own directory is allowed and leaves the full report alone.
    sub = run_robustness(partial, root / "data", tmp_path / "subset", manifests={}, log=lambda *_: None)
    assert sub["config"]["seeds"] == [5]


def test_cli_refuses_frozen_output_dir_without_side_effects(tmp_path):
    import subprocess
    import sys
    out = tmp_path / "frozen"
    out.mkdir()
    (out / "FROZEN").write_text("frozen", encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "bri.experiments.robustness", "--seeds", "1", "--profiles", "balanced",
                        "--customers", "10", "--days", "10", "--val-days", "3", "--test-days", "3",
                        "--data-root", str(tmp_path / "data"), "--output-dir", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "frozen" in r.stderr
    assert not (tmp_path / "data").exists()
    assert sorted(p.name for p in out.iterdir()) == ["FROZEN"]


def test_repository_phase23_lock_rejects_reuse_of_locked_seeds_at_other_sizes(tmp_path):
    manifests = I.load_manifests()
    cfg = RobustnessConfig(profiles=("balanced",), seeds=(20261110,), customers=100)
    with pytest.raises(I.LockedTestSplitError, match="re-expose"):
        run_robustness(cfg, tmp_path / "d", tmp_path / "o", manifests=manifests, log=lambda *_: None)
    assert not (tmp_path / "d").exists()
