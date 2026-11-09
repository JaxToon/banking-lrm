import json
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from bri.experiments.baselines import run_baselines, to_markdown
from bri.tasks import SYNTHETIC_LABEL

FAST = dict(ks=(1, 3), n_boot=100, seed=42, random_seeds=3)


@pytest.fixture(scope="module")
def report(processed_dir):
    return run_baselines(processed_dir, **FAST)


def test_report_is_labelled_and_task_separated(report):
    assert report["label"] == SYNTHETIC_LABEL
    assert set(report["tasks"]) == {"nbs", "nbp"}
    assert "Cold start is not evaluated" in report["population_note"]
    for task in report["tasks"].values():
        assert set(task["models"]) == {"popularity", "itemknn", "mf"}


def test_report_contains_cis_paired_diffs_and_random_floor(report):
    for task in report["tasks"].values():
        if task["n_test_cases"] == 0:
            continue
        for family, m in task["models"].items():
            ci = m["test_bootstrap_ci"]["@3"]["ndcg"]
            assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
            assert ("test_paired_diff_vs_popularity" in m) == (family != "popularity")
        assert len(task["random_floor"]["seeds"]) == 3


def test_experiment_is_deterministic(processed_dir, report):
    again = run_baselines(processed_dir, **FAST)
    assert json.dumps(again, sort_keys=True) == json.dumps(report, sort_keys=True)


def test_test_split_never_influences_model_selection(processed_dir, report, tmp_path):
    """Corrupting the test targets must not change any validation trial or selection."""
    altered = tmp_path / "processed"
    shutil.copytree(processed_dir, altered)
    cases = pd.read_csv(altered / "eval_cases.csv", dtype=str, keep_default_na=False)
    targets = pd.read_csv(altered / "eval_targets.csv", dtype=str, keep_default_na=False)
    meta = json.loads((altered / "prep_metadata.json").read_text(encoding="utf-8"))
    test_ids = set(cases.loc[cases["split"] == "test", "case_id"])
    first_cand = {t: v["candidates"][0] for t, v in meta["tasks"].items()}
    task_of = dict(zip(cases["case_id"], cases["task"]))
    is_test = targets["case_id"].isin(test_ids)
    targets.loc[is_test, "item_id"] = [first_cand[task_of[c]] for c in targets.loc[is_test, "case_id"]]
    targets.drop_duplicates().to_csv(altered / "eval_targets.csv", index=False, lineterminator="\n")

    other = run_baselines(altered, **FAST)
    for task in report["tasks"]:
        for family in report["tasks"][task]["models"]:
            before = report["tasks"][task]["models"][family]
            after = other["tasks"][task]["models"][family]
            assert before["selected_params"] == after["selected_params"]
            assert before["validation_trials"] == after["validation_trials"]


def test_markdown_is_ascii_and_mentions_label(report):
    md = to_markdown(report)
    md.encode("ascii")  # must print on any console (e.g. Windows cp1252)
    assert "SYNTHETIC-DATA EXPERIMENT" in md
    assert "Exposure diversity" in md


def test_invalid_arguments(processed_dir):
    with pytest.raises(ValueError):
        run_baselines(processed_dir, ks=(1,), select_k=3)
    with pytest.raises(ValueError):
        run_baselines(processed_dir, select_metric="auc")
    with pytest.raises(ValueError):
        run_baselines(processed_dir, random_seeds=0)


def test_cli_writes_reports(processed_dir, tmp_path):
    out = tmp_path / "reports"
    r = subprocess.run([sys.executable, "-m", "bri.experiments.baselines", "--processed-dir", str(processed_dir),
                        "--output-dir", str(out), "--n-boot", "50", "--random-seeds", "2"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "baseline_report.json").exists() and (out / "baseline_report.md").exists()


def test_cli_missing_processed_dir(tmp_path):
    r = subprocess.run([sys.executable, "-m", "bri.experiments.baselines", "--processed-dir",
                        str(tmp_path / "none"), "--output-dir", str(tmp_path / "o")],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "bri.prep.build" in r.stderr
