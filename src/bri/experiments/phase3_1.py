"""Phase 3.1 runner (synthetic-data experiments only). Checkpoints 2-3.

`prepare-tuning` generates and validates the 9 `phase3_1_tuning` datasets (3 profiles x
seeds 20261140-20261142); every check runs before anything is written; the final
namespace is refused.

`tune` (Checkpoint 3) runs on the tuning datasets only, with test-window events dropped
before any sequence is built: per dataset x task x initialization seed {1,2,3}, 3 OOF
SASRec fits + 1 full fit (frozen Phase 3 configuration, CUDA, runtime fingerprint equal
to the frozen one), soft labels, support status, gates, static/gated validation NDCG@1,
and the pre-registered selection. It writes a proposal report; it pins nothing and
refuses to run once `phase3_1-config` exists.

Usage:
    python -m bri.experiments.phase3_1 prepare-tuning          # generate + validate
    python -m bri.experiments.phase3_1 pin-tuning-manifest     # immutable fingerprints (once)
    python -m bri.experiments.phase3_1 tune                    # Checkpoint 3 (tuning validation only)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bri.data import schema as S
from bri.data.generate_synthetic import (PRODUCT_TRANSITION, PROFILE_GENERATOR_VERSION, PROFILES,
                                         SERVICE_TRANSITION, write_dataset)
from bri.data.validate_dataset import validate
from bri.experiments import integrity as I
from bri.experiments.prereg_phase3_1 import NAME as PREREG, SEEDS_NAME
from bri.prep.build import PREP_VERSION, build
from bri.tasks import SYNTHETIC_LABEL

TUNING_NS = "phase3_1_tuning"
FINAL_NS = "phase3_1_final"
TUNING_DATA_MANIFEST = "phase3_1-tuning-data"
DATA_ROOT = Path("data/phase3_1")
REPORT = Path("artifacts/phase3_1/tuning_data_report.json")
TUNING_DIR = Path("artifacts/phase3_1/tuning")
CONFIG_MANIFEST = "phase3_1-config"
P3_CONFIG = "phase3-config"


class Phase31Refused(I.LockedTestSplitError):
    pass


def preflight(namespace: str, manifests: dict) -> dict:
    """All checks before writing: manifests, pre-registration, namespace, generator version."""
    if namespace != TUNING_NS:
        raise Phase31Refused(f"Checkpoint 2 authorizes only '{TUNING_NS}' datasets; '{namespace}' is refused")
    for needed in (PREREG, SEEDS_NAME):
        if needed not in manifests:
            raise Phase31Refused(f"manifest '{needed}' is not pinned")
    spec = manifests[SEEDS_NAME]["seed_namespaces"][TUNING_NS]
    I.check_seed_namespace(spec["seeds"], TUNING_NS, manifests)
    if spec["generator_version"] != PROFILE_GENERATOR_VERSION:
        raise Phase31Refused(f"generator {PROFILE_GENERATOR_VERSION} differs from reserved {spec['generator_version']}")
    pre = manifests[PREREG]["data"]
    if (pre["customers"], pre["days"], pre["split_days"]) != (spec["customers"], spec["days"], spec["split_days"]):
        raise Phase31Refused("reservation data parameters differ from the pre-registration")
    return spec


def _hashes(d: Path, names) -> dict:
    return {n: I.sha256_file(d / n) for n in names}


def _profile_checks(events: pd.DataFrame) -> dict:
    """Empirical behavior statistics (data checks only)."""
    out = {}
    for event_type, mapping, key in (("service_view", SERVICE_TRANSITION, "service"),
                                     ("product_view", PRODUCT_TRANSITION, "product")):
        iv = events[events["event_type"] == event_type][["customer_token", "session_id", "item_id"]]
        shares = [c.value_counts().iloc[0] / len(c) for _, c in iv.groupby("customer_token")["item_id"] if len(c) >= 5]
        hits = total = 0
        for _, s in iv.groupby("session_id", sort=False)["item_id"]:
            items = list(s)
            for a, b in zip(items, items[1:]):
                total += 1
                hits += mapping[a] == b
        out[key] = {"mean_top_item_share": round(float(np.mean(shares)), 4),
                    "in_session_transition_rate": round(hits / total, 4) if total else None}
    return out


def prepare_one(spec: dict, profile: str, seed: int, root: Path) -> dict:
    base = root / TUNING_NS / profile / f"seed{seed}"
    src, proc = base / "synthetic", base / "processed"
    gen = write_dataset(src, spec["customers"], spec["days"], seed, profile)
    errors = validate(src)
    if errors:
        raise RuntimeError(f"{profile}/seed{seed} failed validation: {errors[:3]}")
    _, val_d, test_d = spec["split_days"]
    prep = build(src, proc, val_d, test_d, min_history=1)
    if gen["behavior_profile"] != profile or gen["behavior_params"] != PROFILES[profile].to_dict():
        raise RuntimeError(f"{profile}/seed{seed}: generation metadata does not match the profile definition")
    c = prep["cutoffs"]
    days = (pd.Timestamp(c["window_end_utc"]) - pd.Timestamp(c["window_start_utc"])).days
    split = [(pd.Timestamp(c["val_start_utc"]) - pd.Timestamp(c["window_start_utc"])).days,
             (pd.Timestamp(c["test_start_utc"]) - pd.Timestamp(c["val_start_utc"])).days,
             (pd.Timestamp(c["window_end_utc"]) - pd.Timestamp(c["test_start_utc"])).days]
    if days != spec["days"] or split != spec["split_days"]:
        raise RuntimeError(f"{profile}/seed{seed}: split {split} differs from {spec['split_days']}")
    events = pd.read_csv(src / "events.csv", dtype=str, keep_default_na=False)
    sess_owner = events.groupby("session_id")["customer_token"].nunique()
    if (sess_owner != 1).any():
        raise RuntimeError(f"{profile}/seed{seed}: a session spans several customers")
    # Reproducibility: regenerate into a temporary directory and compare byte for byte.
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        write_dataset(t / "s", spec["customers"], spec["days"], seed, profile)
        build(t / "s", t / "p", val_d, test_d, min_history=1)
        same_src = _hashes(t / "s", [*S.TABLES, S.METADATA_FILE]) == _hashes(src, [*S.TABLES, S.METADATA_FILE])
        same_proc = I.processed_fingerprints(t / "p") == I.processed_fingerprints(proc)
    if not (same_src and same_proc):
        raise RuntimeError(f"{profile}/seed{seed}: regeneration is not byte-identical")
    entry = I.dataset_entry(None, proc, prep, f"{profile}/seed{seed}", [])
    entry.pop("evaluations")
    entry["seed_namespace"] = TUNING_NS
    entry["generation_metadata_sha256"] = I.sha256_file(src / S.METADATA_FILE)
    return {"entry": entry, "row_counts": gen["row_counts"],
            "consent_filter": {k: prep["consent_filter"][k] for k in ("input_events", "kept_events")},
            "split_days": split, "sessions_single_customer": True, "regenerated_byte_identical": True,
            "behavior_checks": _profile_checks(events)}


def prepare_tuning(manifests: dict | None = None, root: Path = DATA_ROOT, namespace: str = TUNING_NS,
                   log=print) -> dict:
    manifests = I.load_manifests() if manifests is None else manifests
    spec = preflight(namespace, manifests)
    results = []
    for profile in spec["profiles"]:
        for seed in spec["seeds"]:
            results.append(prepare_one(spec, profile, seed, root))
            log(f"  {profile} seed {seed}: generated, validated, reproduced")
    # Cross-profile behavior ordering per seed (habit > balanced > transition in personal concentration;
    # the reverse for in-session transitions), as defined by the invented profiles.
    by = {(r["entry"]["behavior_profile"], r["entry"]["seed"]): r["behavior_checks"] for r in results}
    ordering = {}
    for seed in spec["seeds"]:
        for key in ("service", "product"):
            share = [by[(p, seed)][key]["mean_top_item_share"] for p in ("habit_dominant", "balanced", "transition_dominant")]
            rate = [by[(p, seed)][key]["in_session_transition_rate"] for p in ("habit_dominant", "balanced", "transition_dominant")]
            ordering[f"seed{seed}/{key}"] = {"concentration_decreasing": share[0] > share[1] > share[2],
                                             "transition_rate_increasing": rate[0] < rate[1] < rate[2]}
    return {"label": SYNTHETIC_LABEL, "namespace": namespace, "n_datasets": len(results),
            "spec": spec, "generator_version": PROFILE_GENERATOR_VERSION, "prep_version": PREP_VERSION,
            "datasets": results, "profile_ordering": ordering,
            "note": "Data preparation only: no model, OOF prediction, gate or evaluation."}


def tuning_data_manifest(report: dict, manifests: dict) -> dict:
    return {
        "manifest_format": I.MANIFEST_FORMAT, "label": SYNTHETIC_LABEL, "suite": TUNING_DATA_MANIFEST,
        "notes": ("Fingerprints of the 9 Phase 3.1 tuning datasets, recorded at Checkpoint 2 before any Phase 3.1 "
                  "model was trained. Their test splits are never evaluated."),
        "references": {"prereg_sha256": I.MANIFEST_SHA256[PREREG], "seeds_sha256": I.MANIFEST_SHA256[SEEDS_NAME]},
        "versions": {"generator": report["generator_version"], "prep": report["prep_version"]},
        "tuning_datasets": [r["entry"] for r in report["datasets"]],
        "configs": {}, "suite_definition_sha256": {}, "frozen_report_sha256": {}, "frozen_report_dir": None,
        "datasets": [],
    }


# --- Checkpoint 3: cross-fitting, gates and selection on TUNING validation only ---------

def tune_preflight(manifests: dict) -> dict:
    """Pinned prereg/seeds/tuning data, frozen Phase 3 config matching the prereg, no Phase 3.1 config yet."""
    from bri.seq.data import SEQ_DATA_VERSION
    from bri.seq.sasrec import MODEL_VERSION
    from bri.seq.train import TRAIN_VERSION
    for needed in (PREREG, SEEDS_NAME, TUNING_DATA_MANIFEST, P3_CONFIG):
        if needed not in manifests:
            raise Phase31Refused(f"manifest '{needed}' is not pinned")
    if CONFIG_MANIFEST in manifests:
        raise Phase31Refused("the Phase 3.1 configuration is already frozen; tuning may not run again")
    pre, cfgm = manifests[PREREG], manifests[P3_CONFIG]
    sha = I.canonical_hash(cfgm["frozen_config"])
    if sha != cfgm["frozen_config_sha256"] or sha != pre["references"]["phase3_frozen_config_sha256"]:
        raise Phase31Refused("frozen SASRec configuration differs from the one referenced by the pre-registration")
    if ({"model": MODEL_VERSION, "train": TRAIN_VERSION} != {k: cfgm["versions"][k] for k in ("model", "train")}
            or cfgm["tuning_data"]["seq_data_version"] != SEQ_DATA_VERSION):
        raise Phase31Refused("SASRec pipeline versions differ from the frozen configuration")
    spec = manifests[SEEDS_NAME]["seed_namespaces"][TUNING_NS]
    I.check_seed_namespace(spec["seeds"], TUNING_NS, manifests)
    return {"spec": spec, "cfgm": cfgm, "init_seeds": pre["seed_policy"]["initialization_seeds"],
            "entries": manifests[TUNING_DATA_MANIFEST]["tuning_datasets"]}


def load_tuning_dataset(entry: dict, root: Path, max_len: int) -> dict:
    """Verified processed files, test-window events dropped, train/val sequence cases only."""
    from bri.prep.split import compute_cutoffs
    from bri.seq.data import build_seq_cases
    from bri.seq.prepare import load_interactions, no_test_events
    from bri.seq.train import train_popularity
    proc = root / TUNING_NS / entry["behavior_profile"] / f"seed{entry['seed']}" / "processed"
    inter, meta, cands = load_interactions(proc)
    if I.verify_processed_dir(proc, meta)["files"] != entry["processed_files_sha256"]:
        raise Phase31Refused(f"{entry['dataset_id']} differs from the pinned tuning-data fingerprints")
    c, p = meta["cutoffs"], meta["params"]
    cut = compute_cutoffs(c["window_start_utc"], c["window_end_utc"], p["val_days"], p["test_days"])
    inter = no_test_events(inter)
    return {"inter": inter, "candidates": cands,
            "tiebreak": {t: train_popularity(inter, t, cands[t]) for t in cands},
            "seqs": build_seq_cases(inter, cut, cands, ("train", "val"), max_len)}


def _jsonable(cell: dict) -> dict:
    v = cell["val_ndcg1"]
    lst = lambda a: None if a is None else [float(x) for x in a]  # noqa: E731
    return {**cell, "val_ndcg1": {"pf": lst(v["pf"]), "sasrec": lst(v["sasrec"]),
                                  "oracle_router": lst(v["oracle_router"]),
                                  "static": {str(a): lst(x) for a, x in v["static"].items()},
                                  "gated": {k: lst(x) for k, x in v["gated"].items()}}}


def tune(root: Path = DATA_ROOT, out_dir: Path = TUNING_DIR, manifests: dict | None = None,
         expected_device: str = "cuda:0", log=print) -> dict:
    from bri.seq import hybrid as H
    from bri.seq import runtime as R
    from bri.seq.crossfit import run_cell
    from bri.tasks import TASKS
    R.configure_determinism()  # before CUDA initialization
    manifests = I.load_manifests() if manifests is None else manifests
    pf = tune_preflight(manifests)
    choice = R.select_device("cuda" if expected_device.startswith("cuda") else "cpu")
    R.require_device(choice, expected_device)
    runtime = R.runtime_fingerprint(choice.device)
    if runtime["runtime_sha256"] != pf["cfgm"]["runtime"]["runtime_sha256"]:
        raise Phase31Refused("runtime fingerprint differs from the one recorded with the frozen SASRec configuration")
    frozen = pf["cfgm"]["frozen_config"]
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)
    summary, t0 = [], time.perf_counter()
    for entry in pf["entries"]:
        profile, seed = entry["behavior_profile"], entry["seed"]
        d = load_tuning_dataset(entry, root, frozen["model"]["max_len"])
        log(f"  {profile} seed {seed}")
        for task in TASKS:
            tr, va = d["seqs"][(task, "train")], d["seqs"][(task, "val")]
            H.check_candidates(d["candidates"][task], tr.items_vocab, va.items_vocab)
            log(f"    {task}: {len(tr.cases)} train / {len(va.cases)} val cases")
            for init in pf["init_seeds"]:
                cell = run_cell(d["inter"], tr, va, d["tiebreak"][task], frozen, init, choice.device, log)
                cell.update({"profile": profile, "seed": seed, "task": task,
                             "train_sequences_sha256": tr.fingerprint(), "val_sequences_sha256": va.fingerprint()})
                name = f"{profile}_seed{seed}_{task}_init{init}.json"
                (out_dir / "cells" / name).write_text(json.dumps(_jsonable(cell)) + "\n", encoding="utf-8",
                                                      newline="\n")
                summary.append({"profile": profile, "seed": seed, "task": task, "init": init,
                                "status": cell["status"], "min_support": cell["min_support"],
                                "per_fold": cell["per_fold"], "means": H.cell_means(cell),
                                "gates_not_converged": sorted(k for k, g in cell["gates"].items()
                                                              if g.get("converged") is False),
                                "alpha_val": {k: g["alpha_val"] for k, g in cell["gates"].items() if "alpha_val" in g},
                                "fits": cell["fits"], "sasrec_seconds_total": cell["sasrec_seconds_total"]})
    selection = H.select(summary, pf["spec"]["profiles"], list(TASKS))
    report = {"label": SYNTHETIC_LABEL, "namespace": TUNING_NS, "hybrid_version": H.HYBRID_VERSION,
              "prereg_sha256": I.MANIFEST_SHA256[PREREG],
              "tuning_data_sha256": I.MANIFEST_SHA256[TUNING_DATA_MANIFEST],
              "frozen_sasrec_config_sha256": pf["cfgm"]["frozen_config_sha256"], "runtime": runtime,
              "init_seeds": pf["init_seeds"], "wall_seconds": round(time.perf_counter() - t0, 1),
              "selection": selection, "cells": summary,
              "note": ("Tuning validation only: no test split was built or evaluated. The selection is a proposal; "
                       "it becomes binding only when frozen in a pinned phase3_1-config manifest after review.")}
    (out_dir / "tuning_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare-tuning", "pin-tuning-manifest", "tune"))
    parser.add_argument("--namespace", default=TUNING_NS)
    args = parser.parse_args(argv)
    try:
        if args.command == "tune":
            rep = tune()
            s = rep["selection"]
            print(f"primary hybrid (proposal): {s['primary_hybrid']}; delta_val {s['delta_val']}; "
                  f"statuses {s['status_counts']}; report {TUNING_DIR / 'tuning_report.json'}")
        elif args.command == "prepare-tuning":
            report = prepare_tuning(namespace=args.namespace)
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
            print(f"{report['n_datasets']} tuning datasets ready; report {REPORT}")
        else:
            manifests = I.load_manifests()
            if TUNING_DATA_MANIFEST in manifests:
                raise Phase31Refused("tuning-data manifest already pinned")
            report = json.loads(REPORT.read_text(encoding="utf-8"))
            for r in report["datasets"]:  # fingerprints must still match the files on disk
                e = r["entry"]
                proc = DATA_ROOT / TUNING_NS / e["behavior_profile"] / f"seed{e['seed']}" / "processed"
                if I.processed_fingerprints(proc)["files"] != e["processed_files_sha256"]:
                    raise Phase31Refused(f"{e['dataset_id']} changed since prepare-tuning")
            digest = I.write_manifest(tuning_data_manifest(report, manifests),
                                      I.MANIFEST_DIR / f"{TUNING_DATA_MANIFEST}.json")
            print(f"tuning-data manifest written; pin sha256 {digest} in bri.experiments.integrity.MANIFEST_SHA256")
    except I.LockedTestSplitError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
