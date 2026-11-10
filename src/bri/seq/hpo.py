"""Optuna hyperparameter search for SASRec Full on Phase 3 TUNING datasets only.

Protocol (fixed before any trial ran):
- Data: the 9 `phase3_tuning` datasets (3 profiles x seeds 20261120-20261122). Only
  train- and validation-window sequence cases are built; test-window events are dropped
  first, so no tuning test split can influence the objective, early stopping or
  selection.
- Each trial trains SASRec Full on every dataset and task (18 fits) on CUDA `cuda:0`,
  FP32, deterministic, with validation-only early stopping (val NDCG@3, patience 5).
- Group score = mean best validation NDCG@3 over the 3 tuning seeds, for each of the six
  (profile, task) groups. Objective = unweighted mean of the six group scores, so groups
  with more cases do not dominate. Every group and per-dataset score is stored.
- Sampler: TPE with a fixed seed, n_jobs=1, SQLite storage (resumable, auditable). No
  pruner. Trials that raise are recorded as FAIL and do not stop the study.
- Selection rule: the COMPLETE trial with the highest objective; ties -> lowest trial
  number. Per-group stability is reported but never used to override this rule.
- The selected configuration is then frozen in an immutable, pinned manifest; ablation
  variants reuse it unchanged (no per-variant search).

Usage:
    python -m bri.seq.hpo run --n-trials 30
    python -m bri.seq.hpo report
    python -m bri.seq.hpo freeze
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bri.experiments import integrity as I
from bri.tasks import SYNTHETIC_LABEL

NAMESPACE = "phase3_tuning"
STUDY_NAME = "phase3-sasrec-full-hpo-v1"
SAMPLER_SEED = 31
TRAIN_SEED = 1
HPO_VERSION = "0.1.0"
SEARCH_SPACE = {
    "dim": [16, 32, 64],
    "heads": [1, 2, 4],
    "blocks": [1, 2, 3],
    "dropout": [0.0, 0.3],              # uniform
    "lr": [1e-4, 5e-3],                 # log-uniform
    "weight_decay": [1e-6, 1e-3],       # log-uniform
    "batch_size": [32, 64, 128],
    "max_len": [10, 20, 30],            # approved after the train-only length analysis
}
TRAIN_FIXED = {"max_epochs": 40, "patience": 5, "grad_clip": 1.0, "select_metric": "ndcg", "select_k": 3}
DEFAULT_STORAGE = Path("artifacts/phase3/hpo/study.db")
DEFAULT_REPORT = Path("artifacts/phase3/hpo")
CONFIG_MANIFEST = "phase3-config"


class InvalidTrialParams(ValueError):
    pass


def validate_params(p: dict) -> None:
    """Architecture constraints the search space must respect."""
    if p["dim"] % p["heads"]:
        raise InvalidTrialParams(f"dim {p['dim']} is not divisible by heads {p['heads']}")
    if p["max_len"] not in SEARCH_SPACE["max_len"]:
        raise InvalidTrialParams(f"max_len {p['max_len']} outside the approved set")


def suggest(trial) -> dict:
    s = SEARCH_SPACE
    p = {
        "dim": trial.suggest_categorical("dim", s["dim"]),
        "heads": trial.suggest_categorical("heads", s["heads"]),
        "blocks": trial.suggest_categorical("blocks", s["blocks"]),
        "dropout": trial.suggest_float("dropout", *s["dropout"]),
        "lr": trial.suggest_float("lr", *s["lr"], log=True),
        "weight_decay": trial.suggest_float("weight_decay", *s["weight_decay"], log=True),
        "batch_size": trial.suggest_categorical("batch_size", s["batch_size"]),
        "max_len": trial.suggest_categorical("max_len", s["max_len"]),
    }
    validate_params(p)
    return p


def split_params(p: dict) -> tuple[dict, dict]:
    model = {k: p[k] for k in ("dim", "heads", "blocks", "dropout", "max_len")}
    train = {"lr": p["lr"], "weight_decay": p["weight_decay"], "batch_size": p["batch_size"], **TRAIN_FIXED,
             "seed": TRAIN_SEED}
    return model, train


# --- data ----------------------------------------------------------------------------

@dataclass
class TuningData:
    """Per dataset: interactions WITHOUT test-window events, cutoffs, candidates, tie-breaks."""
    datasets: list[dict]  # {profile, seed, inter, cutoffs, candidates, tiebreak, fingerprints}
    _cache: dict = field(default_factory=dict)

    @classmethod
    def load(cls, data_root: Path, spec: dict) -> "TuningData":
        from bri.prep.split import compute_cutoffs
        from bri.seq.prepare import load_interactions, no_test_events
        from bri.seq.train import train_popularity
        out = []
        for profile in spec["profiles"]:
            for seed in spec["seeds"]:
                proc = data_root / NAMESPACE / profile / f"seed{seed}" / "processed"
                inter, meta, cands = load_interactions(proc)
                if meta["source"].get("seed") != seed or meta["source_generation"]["behavior_profile"] != profile:
                    raise I.BenchmarkIntegrityError(f"{proc} does not hold {profile}/seed{seed}")
                fp = I.verify_processed_dir(proc, meta)  # processed files match their prep checksums
                c, p = meta["cutoffs"], meta["params"]
                cut = compute_cutoffs(c["window_start_utc"], c["window_end_utc"], p["val_days"], p["test_days"])
                out.append({"profile": profile, "seed": seed, "inter": no_test_events(inter), "cutoffs": cut,
                            "candidates": cands,
                            "tiebreak": {t: train_popularity(inter, t, cands[t]) for t in cands},
                            "fingerprints": {"source_composite": I.composite_fingerprint(meta["source_files_sha256"]),
                                             "processed_composite": fp["composite"],
                                             "prep": I.prep_fingerprint(meta)}})
        return cls(out)

    def seqs(self, i: int, max_len: int) -> dict:
        """Train/val sequence cases of dataset i (cached). Test is never built."""
        from bri.seq.data import build_seq_cases
        key = (i, max_len)
        if key not in self._cache:
            d = self.datasets[i]
            self._cache[key] = build_seq_cases(d["inter"], d["cutoffs"], d["candidates"], ("train", "val"), max_len)
        return self._cache[key]

    def fingerprints(self) -> dict:
        from bri.seq.data import SEQ_DATA_VERSION
        out = {}
        for i, d in enumerate(self.datasets):
            seq = {str(L): {f"{t}/{s}": v.fingerprint() for (t, s), v in self.seqs(i, L).items()}
                   for L in SEARCH_SPACE["max_len"]}
            out[f"{d['profile']}/seed{d['seed']}"] = {**d["fingerprints"], "sequences": seq}
        return {"seq_data_version": SEQ_DATA_VERSION, "datasets": out}


# --- objective -------------------------------------------------------------------------

def make_objective(data: TuningData, device: str, log=print):
    from bri.seq.sasrec import SASRecConfig
    from bri.seq.train import TrainConfig, fit

    def objective(trial) -> float:
        import optuna
        try:
            params = suggest(trial)
        except InvalidTrialParams as exc:
            trial.set_user_attr("invalid", str(exc))
            raise optuna.TrialPruned(str(exc)) from exc
        model_kw, train_kw = split_params(params)
        t0 = time.perf_counter()
        per_dataset, groups = {}, {}
        for i, d in enumerate(data.datasets):
            seqs = data.seqs(i, model_kw["max_len"])
            for task, items in d["candidates"].items():
                res = fit(SASRecConfig(n_items=len(items), variant="full", **model_kw), TrainConfig(**train_kw),
                          seqs[(task, "train")], seqs[(task, "val")], d["tiebreak"][task], device)
                per_dataset[f"{d['profile']}/seed{d['seed']}/{task}"] = {
                    "val_ndcg@3": res.best_val, "best_epoch": res.best_epoch, "epochs": len(res.history),
                    "train_seconds": res.runtime["train_seconds"]}
                groups.setdefault(f"{d['profile']}/{task}", []).append(res.best_val)
        group_scores = {g: round(float(np.mean(v)), 6) for g, v in sorted(groups.items())}
        value = round(float(np.mean(list(group_scores.values()))), 6)  # equal weight per group
        trial.set_user_attr("group_scores", group_scores)
        trial.set_user_attr("per_dataset", per_dataset)
        trial.set_user_attr("seconds", round(time.perf_counter() - t0, 1))
        if log:
            log(f"trial {trial.number}: objective {value:.4f} in {time.perf_counter() - t0:.0f}s {params}")
        return value

    return objective


def study_attrs(data: TuningData, runtime: dict) -> dict:
    """Everything the study's results depend on; a resumed study must match exactly."""
    from bri.seq.sasrec import MODEL_VERSION
    from bri.seq.train import TRAIN_VERSION
    return {
        "hpo_version": HPO_VERSION, "namespace": NAMESPACE, "search_space": SEARCH_SPACE,
        "train_fixed": TRAIN_FIXED, "train_seed": TRAIN_SEED, "sampler": {"type": "TPESampler", "seed": SAMPLER_SEED},
        "objective": "unweighted mean over (profile, task) groups of the mean best val NDCG@3 over tuning seeds",
        "variant": "full", "model_version": MODEL_VERSION, "train_version": TRAIN_VERSION,
        "data_sha256": I.canonical_hash(data.fingerprints()), "runtime_sha256": runtime["runtime_sha256"],
    }


def run_hpo(data: TuningData, n_trials: int, storage: Path, device: str, runtime: dict,
            study_name: str = STUDY_NAME, sampler_seed: int = SAMPLER_SEED, log=print):
    """Create or resume the study and run trials until n_trials have finished (any state)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage.parent.mkdir(parents=True, exist_ok=True)
    attrs = study_attrs(data, runtime)
    attrs["sampler"]["seed"] = sampler_seed
    study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{storage.as_posix()}",
                                sampler=optuna.samplers.TPESampler(seed=sampler_seed), direction="maximize",
                                load_if_exists=True)
    stored = study.user_attrs.get("protocol")
    if stored is None:
        study.set_user_attr("protocol", attrs)
        study.set_user_attr("data_fingerprints", data.fingerprints())
        study.set_user_attr("runtime", runtime)
    elif stored != json.loads(json.dumps(attrs)):
        raise I.BenchmarkIntegrityError(f"study '{study_name}' exists with a different protocol, data or runtime; "
                                        "use a new study name instead of mixing trials")
    remaining = n_trials - len(study.trials)
    if remaining > 0:
        study.optimize(make_objective(data, device, log), n_trials=remaining, n_jobs=1,
                       catch=(RuntimeError, FloatingPointError))
    return study


# --- report & freeze ----------------------------------------------------------------------

def select_best(study):
    import optuna
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise ValueError("no complete trials")
    return sorted(done, key=lambda t: (-t.value, t.number))[0]


def report(study) -> dict:
    import optuna
    trials = study.trials
    done = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = select_best(study)
    groups = sorted(best.user_attrs["group_scores"])
    rows = []
    for t in trials:
        g = t.user_attrs.get("group_scores", {})
        rows.append({"number": t.number, "state": t.state.name, "value": t.value, "params": t.params,
                     "seconds": t.user_attrs.get("seconds"), "group_scores": g,
                     "min_group": min(g.values()) if g else None,
                     "sd_group": round(float(np.std(list(g.values()))), 6) if g else None})
    group_max = {g: max(t.user_attrs["group_scores"][g] for t in done) for g in groups}
    vals = np.array([t.value for t in done])
    secs = np.array([t.user_attrs["seconds"] for t in done])
    rank = {g: 1 + sum(t.user_attrs["group_scores"][g] > best.user_attrs["group_scores"][g] for t in done)
            for g in groups}
    return {
        "label": SYNTHETIC_LABEL, "study_name": study.study_name, "protocol": study.user_attrs.get("protocol"),
        "counts": {s.name: sum(t.state == s for t in trials) for s in optuna.trial.TrialState},
        "objective_distribution": {"min": float(vals.min()), "p25": float(np.percentile(vals, 25)),
                                   "median": float(np.median(vals)), "p75": float(np.percentile(vals, 75)),
                                   "max": float(vals.max())},
        "seconds_per_trial": {"median": float(np.median(secs)), "min": float(secs.min()), "max": float(secs.max()),
                              "total_hours": round(float(secs.sum()) / 3600, 2)},
        "best": {"number": best.number, "value": best.value, "params": best.params,
                 "group_scores": best.user_attrs["group_scores"],
                 "group_rank_among_complete_trials": rank,
                 "group_gap_to_best_trial_for_that_group": {g: round(group_max[g] - best.user_attrs["group_scores"][g], 6)
                                                            for g in groups}},
        "trials": rows,
    }


def to_markdown(r: dict) -> str:
    b = r["best"]
    groups = sorted(b["group_scores"])
    lines = [f"# Phase 3 SASRec Full HPO ({r['study_name']})", "", f"> **{r['label']}**", "",
             f"- Trials by state: {r['counts']}.",
             f"- Objective (val macro NDCG@3) over complete trials: {r['objective_distribution']}.",
             f"- Seconds per trial: {r['seconds_per_trial']}.",
             f"- Selection rule: highest objective, ties -> lowest trial number (fixed before the search).", "",
             f"## Best trial #{b['number']}: objective {b['value']:.4f}", "", f"Params: `{b['params']}`", "",
             "| Group | best-trial score | rank among complete trials | gap to that group's best trial |",
             "|---|---|---|---|"]
    for g in groups:
        lines.append(f"| {g} | {b['group_scores'][g]:.4f} | {b['group_rank_among_complete_trials'][g]} "
                     f"| {b['group_gap_to_best_trial_for_that_group'][g]:.4f} |")
    lines += ["", "## All trials", "",
              "| # | state | objective | min group | sd groups | s | " + " | ".join(groups) + " | params |",
              "|---|---|---|---|---|---|" + "---|" * len(groups) + "---|"]
    for t in sorted(r["trials"], key=lambda t: -(t["value"] or -1)):
        g = t["group_scores"]
        lines.append(f"| {t['number']} | {t['state']} | {t['value'] if t['value'] is None else round(t['value'], 4)} "
                     f"| {t['min_group']} | {t['sd_group']} | {t['seconds']} | "
                     + " | ".join(f"{g[x]:.4f}" if x in g else "-" for x in groups) + f" | `{t['params']}` |")
    lines.append("")
    return "\n".join(lines)


def freeze(study, storage: Path, path: Path | None = None) -> str:
    """Write the immutable frozen-configuration manifest. Returns its sha256 to pin."""
    import optuna
    from bri.seq.sasrec import MODEL_VERSION
    from bri.seq.train import TRAIN_VERSION
    best = select_best(study)
    params = dict(best.params)
    validate_params(params)
    model_kw, train_kw = split_params(params)
    frozen = {"variant_hyperparameters_from": "full", "model": model_kw, "train": train_kw,
              "applies_to_variants": ["full", "no_position", "shuffled", "meanpool", "session_boundary"],
              "device": "cuda:0", "precision": "fp32"}
    counts = {s.name: sum(t.state == s for t in study.trials) for s in optuna.trial.TrialState}
    manifest = {
        "manifest_format": I.MANIFEST_FORMAT, "label": SYNTHETIC_LABEL, "suite": CONFIG_MANIFEST,
        "notes": ("Frozen SASRec configuration selected by Optuna on phase3_tuning validation splits only, "
                  "before any phase3_final dataset existed. Never changed after final results."),
        "frozen_config": frozen, "frozen_config_sha256": I.canonical_hash(frozen),
        "selection": {"rule": "max objective, ties -> lowest trial number", "trial": best.number,
                      "objective": best.value, "group_scores": best.user_attrs["group_scores"]},
        "study": {"name": study.study_name, "trials_by_state": counts, "protocol": study.user_attrs["protocol"],
                  "storage_file_sha256_at_freeze": I.sha256_file(storage)},
        "tuning_data": study.user_attrs["data_fingerprints"], "runtime": study.user_attrs["runtime"],
        "versions": {"model": MODEL_VERSION, "train": TRAIN_VERSION, "hpo": HPO_VERSION,
                     "optuna": optuna.__version__},
        "configs": {}, "suite_definition_sha256": {}, "frozen_report_sha256": {}, "frozen_report_dir": None,
        "datasets": [],
    }
    return I.write_manifest(manifest, path or (I.MANIFEST_DIR / f"{CONFIG_MANIFEST}.json"))


def load_study(storage: Path, study_name: str = STUDY_NAME):
    import optuna
    return optuna.load_study(study_name=study_name, storage=f"sqlite:///{storage.as_posix()}")


def preflight(manifests: dict) -> dict:
    spec = next((m["seed_namespaces"][NAMESPACE] for m in manifests.values()
                 if NAMESPACE in m.get("seed_namespaces", {})), None)
    if spec is None:
        raise I.LockedTestSplitError(f"seed namespace '{NAMESPACE}' is not registered")
    I.check_seed_namespace(spec["seeds"], NAMESPACE, manifests)
    if CONFIG_MANIFEST in manifests:
        raise I.LockedTestSplitError("the SASRec configuration is already frozen; HPO may not continue")
    return spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "report", "freeze"))
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--data-root", type=Path, default=Path("data/phase3"))
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            from bri.seq import runtime as R
            R.configure_determinism()  # before CUDA initialization
            manifests = I.load_manifests()
            spec = preflight(manifests)
            choice = R.select_device("cuda")
            R.require_device(choice, "cuda:0")
            data = TuningData.load(args.data_root, spec)
            run_hpo(data, args.n_trials, args.storage, choice.device, R.runtime_fingerprint(choice.device))
            args.command = "report"
        study = load_study(args.storage)
        if args.command == "report":
            r = report(study)
            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / "hpo_report.json").write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8",
                                                             newline="\n")
            md = to_markdown(r)
            (args.report_dir / "hpo_report.md").write_text(md, encoding="utf-8", newline="\n")
            print(md)
        elif args.command == "freeze":
            preflight(I.load_manifests())
            digest = freeze(study, args.storage)
            print(f"Frozen configuration written to {I.MANIFEST_DIR / (CONFIG_MANIFEST + '.json')}; "
                  f"pin sha256 {digest} in bri.experiments.integrity.MANIFEST_SHA256")
    except I.LockedTestSplitError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
