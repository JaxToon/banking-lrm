import json
import subprocess
import sys

import pandas as pd
import pytest

from bri.eval.evaluate import InvalidRecommendationsError, evaluate, run, sanity_recommendations
from bri.tasks import SYNTHETIC_LABEL

CANDS = {"nbs": ["SVC_A", "SVC_B", "SVC_C"], "nbp": ["PRD_X", "PRD_Y", "PRD_Z"]}
CASES = pd.DataFrame({
    "case_id": ["S1", "S2", "P1"],
    "task": ["nbs", "nbs", "nbp"],
})
TARGETS = pd.DataFrame({"case_id": ["S1", "S2", "S2", "P1"], "item_id": ["SVC_A", "SVC_B", "SVC_C", "PRD_Z"]})


def recs(rows):
    return pd.DataFrame(rows, columns=["case_id", "item_id", "rank"])


GOOD = recs([
    ("S1", "SVC_A", 1), ("S1", "SVC_B", 2),
    ("S2", "SVC_A", 1), ("S2", "SVC_C", 2),
    ("P1", "PRD_X", 1), ("P1", "PRD_Z", 2),
])


def test_hand_calculated_task_results():
    res = evaluate(CASES, TARGETS, GOOD, CANDS, ks=(1, 2))
    nbs, nbp = res["nbs"], res["nbp"]
    assert nbs["n_cases"] == 2 and nbp["n_cases"] == 1
    # S1: hit@1. S2: miss@1, hit at rank 2 (one of two relevant)
    assert nbs["metrics"]["@1"]["precision"] == pytest.approx(0.5)
    assert nbs["metrics"]["@1"]["recall"] == pytest.approx(0.5)
    assert nbs["metrics"]["@2"]["precision"] == pytest.approx((0.5 + 0.5) / 2)
    assert nbs["metrics"]["@2"]["recall"] == pytest.approx((1 + 0.5) / 2)
    assert nbs["metrics"]["@2"]["map"] == pytest.approx((1 + 0.25) / 2)  # S2: (1/2)/2
    assert nbs["metrics"]["@1"]["coverage"] == pytest.approx(1 / 3)       # only SVC_A at rank 1
    assert nbs["metrics"]["@2"]["coverage"] == pytest.approx(1.0)
    # NBP is scored independently of NBS
    assert nbp["metrics"]["@1"]["precision"] == 0.0
    assert nbp["metrics"]["@2"]["ndcg"] == pytest.approx(0.63093, abs=1e-5)


def test_case_without_recommendations_scores_zero():
    res = evaluate(CASES, TARGETS, GOOD[GOOD["case_id"] != "S2"], CANDS, ks=(1,))
    assert res["nbs"]["cases_without_recommendations"] == 1
    assert res["nbs"]["metrics"]["@1"]["precision"] == pytest.approx(0.5)  # S1=1, S2=0


def test_task_without_cases_is_undefined_not_zero():
    res = evaluate(CASES[CASES["task"] == "nbs"], TARGETS, GOOD[GOOD["case_id"] != "P1"], CANDS, ks=(1,))
    assert res["nbp"]["n_cases"] == 0 and res["nbp"]["metrics"]["@1"] is None


def test_empty_evaluation_set():
    empty = CASES.iloc[0:0]
    res = evaluate(empty, TARGETS.iloc[0:0], GOOD.iloc[0:0], CANDS)
    assert all(r["n_cases"] == 0 for r in res.values())


@pytest.mark.parametrize("bad,message", [
    (recs([("S1", "PRD_X", 1)]), "not in the task candidate set"),       # NBP item in NBS list
    (recs([("S1", "SVC_UNKNOWN", 1)]), "not in the task candidate set"),
    (recs([("ZZ", "SVC_A", 1)]), "unknown case_id"),
    (recs([("S1", "SVC_A", 1), ("S1", "SVC_A", 2)]), "duplicate item"),
    (recs([("S1", "SVC_A", 1), ("S1", "SVC_B", 1)]), "unique and contiguous"),
    (recs([("S1", "SVC_A", 2)]), "unique and contiguous"),
    (recs([("S1", "SVC_A", 0)]), "integer >= 1"),
    (recs([("S1", "SVC_A", "first")]), "integer >= 1"),
    (recs([("S1", "", 1)]), "empty values"),
    (pd.DataFrame({"case_id": ["S1"], "item_id": ["SVC_A"]}), "missing columns"),
])
def test_invalid_recommendations_rejected(bad, message):
    with pytest.raises(InvalidRecommendationsError, match=message):
        evaluate(CASES, TARGETS, bad, CANDS)


def test_inactive_item_rejected_when_absent_from_candidates():
    cands = {**CANDS, "nbs": ["SVC_A", "SVC_C"]}  # SVC_B inactive
    with pytest.raises(InvalidRecommendationsError):
        evaluate(CASES, TARGETS, GOOD, cands)


def test_invalid_k():
    with pytest.raises(ValueError):
        evaluate(CASES, TARGETS, GOOD, CANDS, ks=(0,))


def test_oracle_is_a_ceiling_and_random_is_seeded():
    oracle = evaluate(CASES, TARGETS, sanity_recommendations(CASES, TARGETS, CANDS, "oracle", 0), CANDS, (1, 2))
    assert oracle["nbs"]["metrics"]["@2"]["ndcg"] == 1.0 and oracle["nbs"]["metrics"]["@2"]["map"] == 1.0
    r1 = sanity_recommendations(CASES, TARGETS, CANDS, "random", 3)
    r2 = sanity_recommendations(CASES, TARGETS, CANDS, "random", 3)
    pd.testing.assert_frame_equal(r1, r2)


def test_report_is_labelled_and_deterministic(processed_dir):
    a = run(processed_dir, "val", (1, 3), "random", None, seed=11)
    b = run(processed_dir, "val", (1, 3), "random", None, seed=11)
    assert a == b
    assert a["label"] == SYNTHETIC_LABEL
    assert set(a["results"]) == {"nbs", "nbp"}
    assert a["results"]["nbs"]["n_cases"] > 0 and a["results"]["nbp"]["n_cases"] > 0


def test_build_is_byte_reproducible(dataset_dir, tmp_path):
    from bri.prep.build import build

    build(dataset_dir, tmp_path / "a", 3, 3, 1)
    build(dataset_dir, tmp_path / "b", 3, 3, 1)
    for name in ["interactions.csv", "train_examples.csv", "eval_cases.csv",
                 "eval_targets.csv", "prep_metadata.json"]:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name


def test_build_refuses_invalid_source(dataset_copy, tmp_path):
    from bri.prep.build import InvalidSourceDataError, build

    (dataset_copy / "consent_preferences.csv").unlink()
    with pytest.raises(InvalidSourceDataError):
        build(dataset_copy, tmp_path / "out", 3, 3, 1)


def test_cli_build_and_evaluate(dataset_dir, tmp_path):
    out, report = tmp_path / "processed", tmp_path / "report.json"
    b = subprocess.run([sys.executable, "-m", "bri.prep.build", "--data-dir", str(dataset_dir),
                        "--output-dir", str(out), "--val-days", "3", "--test-days", "3"],
                       capture_output=True, text=True)
    assert b.returncode == 0, b.stderr
    assert "SYNTHETIC-DATA EXPERIMENT" in b.stdout
    e = subprocess.run([sys.executable, "-m", "bri.eval.evaluate", "--processed-dir", str(out),
                        "--split", "test", "--ranker", "oracle", "--output", str(report)],
                       capture_output=True, text=True)
    assert e.returncode == 0, e.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["recommendations"] == "sanity:oracle"
    assert data["results"]["nbp"]["metrics"]["@1"]["precision"] == 1.0


def test_cli_rejects_invalid_recommendations(processed_dir, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("case_id,item_id,rank\nNOT_A_CASE,SVC_QR,1\n", encoding="utf-8")
    e = subprocess.run([sys.executable, "-m", "bri.eval.evaluate", "--processed-dir", str(processed_dir),
                        "--recommendations", str(bad)], capture_output=True, text=True)
    assert e.returncode == 1
    assert "invalid recommendations" in e.stderr


def test_cli_rejects_too_short_window(dataset_dir, tmp_path):
    b = subprocess.run([sys.executable, "-m", "bri.prep.build", "--data-dir", str(dataset_dir),
                        "--output-dir", str(tmp_path), "--val-days", "7", "--test-days", "7"],
                       capture_output=True, text=True)
    assert b.returncode == 2 and "too short" in b.stderr
