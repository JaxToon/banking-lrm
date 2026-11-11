"""Phase 3.1 pre-registration manifest: build, verify, stage, pin (no seeds, no data).

The manifest records, before any Phase 3.1 seed or dataset exists, the sha256 of the
approved design document (docs/13_PHASE_3_1_HYBRID_DESIGN.md, revision 4) and every
decision, feature, grid, metric, threshold, seed policy and evaluation criterion.

Usage:
    python -m bri.experiments.prereg_phase3_1 stage   # build + verify -> artifacts/phase3_1/
    python -m bri.experiments.prereg_phase3_1 pin     # write the immutable manifest (after approval)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bri.experiments import integrity as I
from bri.tasks import SYNTHETIC_LABEL

NAME = "phase3_1-prereg"
DESIGN_DOC = I.REPO_ROOT / "docs" / "13_PHASE_3_1_HYBRID_DESIGN.md"
STAGING = Path("artifacts/phase3_1/phase3_1-prereg.staged.json")
FEATURES = {
    "P": ["pf_top", "pf_entropy", "pf_margin"],
    "S": ["sas_top", "sas_entropy", "sas_margin"],
    "H": ["h_sessions", "h_last_session_steps", "h_last_session_share"],
}
SUBSETS = ["P", "S", "H", "P+S", "P+S+H"]
DECISIONS = {
    1: "Option A: prefix-history features only; Option B (request-time protocol) deferred to separate research",
    2: "Oracle router (evaluator within/cross label) as a separately reported diagnostic upper bound only; never selected or claimed",
    3: "Gate trained on customer-level 3-fold OOF SASRec scores from the train split; no in-sample SASRec scores",
    4: "Non-inferiority margin 0.01 NDCG@1, fixed before any Phase 3.1 dataset exists",
    5: "Three SASRec initialization seeds {1,2,3}; per-case mean over initializations per dataset seed, then across-seed analysis; initialization variance reported separately",
    6: "7-day window secondary; no window-specific tuning of alpha or gate",
    7: "P_PF Laplace-smoothed counts, P_SASRec softmax; no fitted score normalizers, no temperature scaling, no smoothing tuning; identical deterministic candidate sets",
    8: "Soft-label gate target 1/0.5/0 (SASRec NDCG@1 >, =, < PF); L2 BCE with fractional targets; proxy limitation stated",
    9: "No per-profile parameters and no profile/seed/dataset features; alpha and gate per task",
    10: "Fixed 9 features; subsets P,S,H,P+S,P+S+H; C {0.01,0.1,1,10}; alpha 0.0..1.0 step 0.1; selection on tuning validation by equal-weight macro NDCG@1",
    11: "Gate support: >=50 per class per fold valid, 30-49 exploratory (not RQ3), <30 refused; ties counted separately; no silent changes of folds, regularization, target or features",
    12: "Primary hybrid chosen on tuning validation and frozen; static and gated both reported on final; only the pre-chosen one used for main claims",
    13: "Initializations are not extra dataset seeds; their variance is never merged into bootstrap or across-seed intervals",
    14: "Gated iff all tuning gates valid and Delta_val > 0 (equal-weight macro NDCG@1 over 6 groups after all selections); otherwise static; not evidence for RQ3",
    15: "Final gates: >=50 valid (confirmatory), 30-49 exploratory (descriptive only), <30 rejected (not trained, not used, "
        "'not evaluable under the preregistered protocol'); no fallback to static after final evaluation; details in final_gate_rules",
}
NOT_EVALUABLE = "not evaluable under the preregistered protocol"
# Owner clarification of decision 15 (2026-11-10), locked verbatim in meaning.
FINAL_GATE_RULES = {
    "status_table": {
        "valid": {"min_per_class_per_fold": 50, "treatment": "confirmatory evaluation"},
        "exploratory": {"min_per_class_per_fold": 30, "max_per_class_per_fold": 49,
                        "treatment": "may be reported descriptively; never counted as confirmatory evidence"},
        "rejected": {"below_per_class_per_fold": 30, "treatment": f"gate not trained and not used; reported as '{NOT_EVALUABLE}'"},
    },
    "RQ3": [
        "RQ3 can be declared met only if every final cell needed for the six-group macro-average has a valid gate",
        f"if one or more such cells are exploratory or rejected, RQ3 is '{NOT_EVALUABLE}'",
        "results of valid cells may be reported descriptively, but invalid cells are never dropped from the confirmatory macro-average",
        "a gate is never replaced by static fusion to resolve insufficient support",
    ],
    "RQ1_RQ2": [
        "if validation chose static fusion, gate invalidity does not block evaluating the static primary hybrid",
        "if validation chose gated fusion, exploratory or rejected final cells cannot support confirmatory gated-hybrid claims",
        f"RQ1/RQ2 for affected groups are '{NOT_EVALUABLE}'",
        "no fallback from gated to static for any group after final evaluation",
    ],
}
REPORTING_RULES = [
    "never replace unavailable values with zero",
    "never silently drop a group from a macro-average",
    "report the number of valid cells and the denominator used",
    "distinguish the statuses valid, exploratory and not evaluable",
    "never conclude a model failed on performance when its result is not evaluable because of insufficient support",
]
DOCUMENTATION_NOTES = [
    "Initialization seeds {1,2,3} are model seeds; seed 1 equalling the Phase 3 training seed does not reuse any Phase 3 "
    "dataset. Dataset seeds and initialization seeds are kept separate.",
    "Round-robin fold assignment is deterministic and uses whole customers as the split unit; label-distribution validity is "
    "checked separately through the support thresholds.",
]
APPROVAL = {
    "checkpoint_0": "approved to pin by the project owner on 2026-11-10",
    "not_authorized_by_this_manifest": ["Phase 3.1 seed reservation", "dataset generation", "model training",
                                        "running Phase 3.1 experiments"],
}


def build(manifests: dict | None = None, design_doc: Path = DESIGN_DOC) -> dict:
    manifests = I.load_manifests() if manifests is None else manifests
    cfg = manifests["phase3-config"]
    return {
        "manifest_format": I.MANIFEST_FORMAT, "label": SYNTHETIC_LABEL, "suite": NAME,
        "notes": ("Phase 3.1 pre-registration, written before any Phase 3.1 seed or dataset exists. "
                  "Any change requires a new pre-registration; this manifest is never edited."),
        "design_document": {"path": "docs/13_PHASE_3_1_HYBRID_DESIGN.md", "revision": 4,
                            "sha256": I.sha256_file(design_doc)},
        "references": {
            "phase3_frozen_config_sha256": cfg["frozen_config_sha256"],
            "phase3_config_manifest_sha256": I.MANIFEST_SHA256["phase3-config"],
            "phase3_lock_manifest_sha256": I.MANIFEST_SHA256["phase3"],
        },
        "decisions": {str(k): v for k, v in DECISIONS.items()},
        "research_questions": {
            "RQ1": "non-inferiority of the primary hybrid vs PF in habit/NBS, habit/NBP, balanced/NBS, balanced/NBP",
            "RQ2": "transition/NBS: primary hybrid superior to PF and non-inferior to SASRec Full",
            "RQ3": "gated fusion beats static fusion",
            "RQ4": "cost: training incl. OOF, inference latency, peak GPU memory, parameters",
        },
        "features": {"groups": FEATURES, "subsets": SUBSETS,
                     "entropy_normalization": "divide by log(|C|)",
                     "history_sessions": "delimited by history-only boundary flags; earliest represented step never a boundary",
                     "standardization": "fitted on train-split OOF features of the dataset, applied unchanged to val/test"},
        "forbidden_inputs": ["target timestamp or any difference to it", "target session id or target-derived boundary",
                             "events at or after the recommendation time", "evaluator within/cross-session label",
                             "behavior profile, seed, or dataset/generator identity"],
        "normalization": {"pf": "(count(i)+1)/(sum_j count(j)+|C|)", "sasrec": "softmax over candidates",
                          "temperature_scaling": False, "smoothing_tuning": False,
                          "candidate_sets": "identical across PF/SASRec/static/gated, sorted item ids"},
        "models": {
            "personal_frequency": "Phase 2.3 grid (signal), selected per dataset on validation",
            "sasrec_full": f"Phase 3 frozen configuration {cfg['frozen_config_sha256']}; no SASRec HPO",
            "static_fusion": {"formula": "(1-alpha)*P_PF + alpha*P_SASRec", "alpha_grid": [round(i / 10, 1) for i in range(11)],
                              "per": "task"},
            "gated_fusion": {"formula": "(1-alpha(x))*P_PF + alpha(x)*P_SASRec, alpha(x)=sigmoid(w.x+b)",
                             "C_grid": [0.01, 0.1, 1, 10], "subsets": SUBSETS, "per": "task",
                             "weights": "refitted per dataset on its train split"},
            "oracle_router": "SASRec on evaluator within-session cases, PF on cross-session cases; diagnostic only",
        },
        "gate": {
            "target": {"sasrec_better": 1.0, "tie": 0.5, "pf_better": 0.0, "basis": "NDCG@1 on train-split cases with OOF SASRec scores"},
            "objective": "sum BCE(y, sigmoid(w.x+b)) + ||w||^2/(2C), bias unregularized, fractional targets",
            "optimizer": {"method": "Newton", "dtype": "float64", "max_iter": 100, "grad_max_norm_tol": 1e-10,
                          "init": "w=0, b=0", "library": "numpy (project implementation)"},
            "cross_fitting": {"folds": 3, "unit": "customer", "assignment": "sort tokens, permute with numpy default_rng(3101), round-robin",
                              "oof_sasrec": "frozen configuration, validation-only early stopping"},
            "support": {"count_classes": "y=0 and y=1 only; y=0.5 counted separately",
                        "valid_min_per_class_per_fold": 50, "exploratory_min_per_class_per_fold": 30,
                        "below_exploratory": "refused, not evaluable per protocol",
                        "no_silent_changes": ["folds", "regularization", "target", "features"]},
            "reported": ["counts y=0/0.5/1 per fold, task, initialization, dataset",
                         "within/cross-session train counts per fold (diagnostic)",
                         "alpha(x) distribution on val/test: mean, SD, P5, P50, P95"],
        },
        "selection": {
            "objective": "equal-weight macro-average NDCG@1 over the six (profile, task) groups on tuning validation",
            "primary_hybrid_rule": "gated iff every tuning dataset x task x initialization gate is valid and Delta_val > 0; else static",
            "delta_val": "(1/6) * sum_g (NDCG@1_gated,g - NDCG@1_static,g); group value = mean over its tuning datasets; computed after all selections",
            "frozen_before_final": True, "final_data_used_for_selection": False,
        },
        "seed_policy": {
            "namespaces": {"phase3_1_tuning": {"seeds_per_profile": 3}, "phase3_1_final": {"seeds_per_profile": 10}},
            "reserve_after": "this pre-registration is pinned and verified (later checkpoint)",
            "collision_check": "against every registered seed including all Phase 3 seeds",
            "initialization_seeds": [1, 2, 3],
            "phase3_datasets_reused": False,
        },
        "data": {"generator": "0.3.0", "profiles": ["habit_dominant", "balanced", "transition_dominant"],
                 "customers": 500, "days": 30, "split_days": [16, 7, 7], "case_protocol": "Phase 3 next-event"},
        "evaluation": {
            "primary_metric": "NDCG@1", "ks": [1, 3], "primary_protocol": "next_event", "secondary_protocol": "window_7d",
            "statistics": {"within_seed": "paired customer-level bootstrap on per-case initialization means, n_boot=1000",
                           "across_seed": "Student-t 95% over 10 final dataset seeds", "pooled": False},
            "criteria": {
                "margin_ndcg@1": 0.01,
                "RQ1": "lower bound of (primary hybrid - PF) > -0.01 in each of habit/NBS, habit/NBP, balanced/NBS, balanced/NBP",
                "RQ2": "transition/NBS: (primary hybrid - PF) interval > 0 and lower bound of (primary hybrid - SASRec Full) > -0.01",
                "RQ3": "macro-average over six groups of (gated - static) interval > 0; confirmatory only if every final gate is valid, else 'not satisfiable per protocol'",
                "RQ4": "report cost; no winning requirement",
                "gated_primary_with_invalid_final_gate": "cell not confirmatory; group's RQ1/RQ2 reported not evaluable per protocol; no fallback",
            },
            "interpretation": {
                "RQ1+RQ2": "hybrid achieves the main goal", "RQ2 only": "trade-off, not overall success",
                "RQ1 only": "safe for habit/balanced, sequential value not shown",
                "RQ3 failed or not satisfiable": "static fusion preferred",
                "no fusion beats PF": "no case for a hybrid as main model"},
        },
        "final_gate_rules": FINAL_GATE_RULES,
        "reporting_rules": REPORTING_RULES,
        "status_labels": {"valid": "valid", "exploratory": "exploratory", "not_evaluable": NOT_EVALUABLE},
        "documentation_notes": DOCUMENTATION_NOTES,
        "approval": APPROVAL,
        "phase3_artifacts_modified": False,
        "configs": {}, "suite_definition_sha256": {}, "frozen_report_sha256": {}, "frozen_report_dir": None,
        "datasets": [],
    }


def verify(m: dict, manifests: dict | None = None, design_doc: Path = DESIGN_DOC) -> list[str]:
    """Consistency problems (empty list = consistent)."""
    manifests = I.load_manifests() if manifests is None else manifests
    problems = []
    doc = design_doc.read_text(encoding="utf-8")
    if m["design_document"]["sha256"] != I.sha256_file(design_doc):
        problems.append("design document sha256 differs from the file")
    if "REVISION 4" not in doc or "[to confirm]" in doc:
        problems.append("design document is not the approved revision 4 without open items")
    if sorted(map(int, m["decisions"])) != list(range(1, 16)):
        problems.append("decisions 1-15 incomplete")
    if m["references"]["phase3_frozen_config_sha256"] != manifests["phase3-config"]["frozen_config_sha256"]:
        problems.append("Phase 3 frozen configuration reference mismatch")
    if m["references"]["phase3_lock_manifest_sha256"] != I.MANIFEST_SHA256["phase3"]:
        problems.append("Phase 3 lock reference mismatch")
    feats = [f for g in FEATURES.values() for f in g]
    if len(feats) != 9 or len(set(feats)) != 9:
        problems.append("feature set must be 9 distinct features")
    for f in feats:
        if f"`{f}`" not in doc:
            problems.append(f"feature {f} not defined in the design document")
    for sub in m["features"]["subsets"]:
        if any(part not in FEATURES for part in sub.split("+")):
            problems.append(f"subset {sub} references an unknown group")
    alpha = m["models"]["static_fusion"]["alpha_grid"]
    if alpha[0] != 0.0 or alpha[-1] != 1.0 or len(alpha) != 11:
        problems.append("alpha grid must be 0.0..1.0 step 0.1")
    sup = m["gate"]["support"]
    if not sup["valid_min_per_class_per_fold"] > sup["exploratory_min_per_class_per_fold"] > 0:
        problems.append("support thresholds inconsistent")
    for needle in ("3101", "1e-10", "≥ 50", "30–49", "< 30", "0.01", "{1, 2, 3}", "Δ_val", "1/6"):
        if needle not in doc:
            problems.append(f"design document does not state '{needle}'")
    if m["evaluation"]["criteria"]["margin_ndcg@1"] != 0.01:
        problems.append("margin must be 0.01")
    reserved = set(I.seed_reservations(manifests).values())
    if reserved & set(m["seed_policy"]["namespaces"]):
        problems.append("a Phase 3.1 namespace is already reserved")
    if NAME in manifests:
        problems.append("a Phase 3.1 pre-registration is already pinned")
    return problems


SEEDS_NAME = "phase3_1-seeds"
PREREG_SHA256 = "9834e435dd8ba0a51b2123c57e7b9ae4522e6ec7aa7c8965ee875ab1616840eb"
# Exact dataset-seed values approved by the project owner at Phase 3.1 Checkpoint 1
# (2026-11-10). The pre-registration fixed only the number of seeds per namespace.
APPROVED_SEEDS = {"phase3_1_tuning": list(range(20261140, 20261143)),
                  "phase3_1_final": list(range(20261150, 20261160))}


def build_seed_reservation(manifests: dict) -> dict:
    """Phase 3.1 seed reservation; raises on any collision (checked against `manifests`)."""
    from bri.data.generate_synthetic import PROFILE_GENERATOR_VERSION
    pre = manifests[NAME]
    if I.MANIFEST_SHA256.get(NAME) != PREREG_SHA256:
        raise I.BenchmarkIntegrityError("pinned pre-registration hash differs from the one approved at Checkpoint 1")
    if verify(pre, {k: v for k, v in manifests.items() if k not in (NAME, SEEDS_NAME)}):
        raise I.BenchmarkIntegrityError("pre-registration no longer verifies")
    spec = {"generator_version": PROFILE_GENERATOR_VERSION, "profiles": pre["data"]["profiles"],
            "customers": pre["data"]["customers"], "days": pre["data"]["days"], "split_days": pre["data"]["split_days"]}
    init = set(pre["seed_policy"]["initialization_seeds"])
    namespaces = {}
    for ns, seeds in APPROVED_SEEDS.items():
        if len(seeds) != pre["seed_policy"]["namespaces"][ns]["seeds_per_profile"] or len(set(seeds)) != len(seeds):
            raise I.BenchmarkIntegrityError(f"{ns}: seed count/uniqueness differs from the pre-registration")
        if set(seeds) & init:
            raise I.BenchmarkIntegrityError(f"{ns}: dataset seeds overlap initialization seeds")
        namespaces[ns] = {**spec, "seeds": seeds,
                          "purpose": ("Phase 3.1 tuning: alpha / gate selection on validation splits only; test splits "
                                      "never evaluated." if ns.endswith("tuning") else
                                      "Phase 3.1 final evaluation after the configuration is frozen; each test split used once.")}
    m = I.build_seed_reservation(  # collision check against every registered seed and reservation
        SEEDS_NAME, namespaces, manifests,
        "Phase 3.1 seed reservation, written before any Phase 3.1 dataset exists. Exact seed values were set by "
        "project-owner approval at Phase 3.1 Checkpoint 1 because the pre-registration fixed only the number of "
        "seeds per namespace.")
    m["prereg_reference"] = {"manifest": f"benchmarks/manifests/{NAME}.json", "sha256": PREREG_SHA256}
    m["approval"] = {"checkpoint_1": "exact seed values approved by the project owner on 2026-11-10",
                     "not_authorized_by_this_manifest": ["dataset generation", "model training", "cross-fitting",
                                                         "gate training", "Phase 3.1 evaluation"]}
    m["initialization_seeds_separate"] = sorted(init)
    return m


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("stage", "pin", "reserve"))
    args = parser.parse_args(argv)
    manifests = I.load_manifests()
    if args.command == "reserve":
        if SEEDS_NAME in manifests:
            print("REFUSED: Phase 3.1 seeds are already reserved", file=sys.stderr)
            return 1
        try:
            m = build_seed_reservation(manifests)  # collision check immediately before writing
        except I.BenchmarkIntegrityError as exc:
            print(f"REFUSED (nothing written): {exc}", file=sys.stderr)
            return 1
        digest = I.write_manifest(m, I.MANIFEST_DIR / f"{SEEDS_NAME}.json")
        print(f"seed reservation written; pin sha256 {digest} in bri.experiments.integrity.MANIFEST_SHA256")
        return 0
    m = build(manifests)
    problems = verify(m, manifests)
    if problems:
        print("PRE-REGISTRATION INCONSISTENT:\n- " + "\n- ".join(problems), file=sys.stderr)
        return 1
    if args.command == "stage":
        STAGING.parent.mkdir(parents=True, exist_ok=True)
        STAGING.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(f"verified; staged at {STAGING} (canonical sha256 {I.canonical_hash(m)})")
    else:
        digest = I.write_manifest(m, I.MANIFEST_DIR / f"{NAME}.json")
        print(f"pre-registration written; pin sha256 {digest} in bri.experiments.integrity.MANIFEST_SHA256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
