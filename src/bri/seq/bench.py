"""CPU vs CUDA runtime benchmark for SASRec Full on a Phase 3 TUNING dataset.

Same data, same fixed (untuned) configurations, same seed on both devices. Each run is
a fresh subprocess (clean CUDA state, comparable process memory) that calls
configure_determinism() first and trains TWICE to verify training determinism. Fixed
epoch counts (no early stopping) make timings comparable. Only train- and val-window
cases are built; test-window events are dropped first.

Usage:
    python -m bri.seq.bench --output-dir artifacts/phase3/bench
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from bri.experiments import integrity as I
from bri.tasks import SYNTHETIC_LABEL

NAMESPACE = "phase3_tuning"
DATASET = ("transition_dominant", 20261120)
# Fixed reference configurations for timing only (NOT tuned, NOT selected).
CONFIGS = {
    "reference": {"model": {"dim": 32, "heads": 2, "blocks": 2, "dropout": 0.2, "max_len": 20},
                  "train": {"lr": 1e-3, "weight_decay": 1e-5, "batch_size": 128}},
    "small_batch": {"model": {"dim": 32, "heads": 2, "blocks": 2, "dropout": 0.2, "max_len": 20},
                    "train": {"lr": 1e-3, "weight_decay": 1e-5, "batch_size": 32}},
    "largest": {"model": {"dim": 64, "heads": 4, "blocks": 3, "dropout": 0.2, "max_len": 30},
                "train": {"lr": 1e-3, "weight_decay": 1e-5, "batch_size": 128}},
}
EPOCHS = 10
SEED = 7


def worker(spec: dict) -> dict:
    from bri.seq import runtime as R
    settings = R.configure_determinism()  # before anything touches CUDA
    choice = R.select_device(spec["device"])
    if spec["device"] == "cuda" and choice.device == "cpu":
        return {"skipped": choice.fallback_reason}
    import numpy as np  # noqa: F401
    from bri.prep.split import compute_cutoffs
    from bri.seq.data import build_seq_cases
    from bri.seq.prepare import load_interactions, no_test_events
    from bri.seq.sasrec import SASRecConfig
    from bri.seq.train import (TrainConfig, checkpoint_metadata, fit, save_checkpoint, state_dict_sha256,
                               train_popularity)
    inter, meta, cands = load_interactions(Path(spec["processed_dir"]))
    c, p = meta["cutoffs"], meta["params"]
    cut = compute_cutoffs(c["window_start_utc"], c["window_end_utc"], p["val_days"], p["test_days"])
    cfg = spec["config"]
    seqs = build_seq_cases(no_test_events(inter), cut, cands, ("train", "val"), cfg["model"]["max_len"])
    task = spec["task"]
    tr, va = seqs[(task, "train")], seqs[(task, "val")]
    tb = train_popularity(inter, task, cands[task])
    mcfg = SASRecConfig(n_items=len(cands[task]), **cfg["model"])
    tcfg = TrainConfig(**cfg["train"], max_epochs=EPOCHS, early_stopping=False, seed=SEED)
    runs = [fit(mcfg, tcfg, tr, va, tb, choice.device) for _ in range(2)]
    with tempfile.TemporaryDirectory() as d:
        size = save_checkpoint(Path(d) / "m.pt", runs[0].model, checkpoint_metadata(runs[0], tcfg, tr, va, choice.device))
    return {
        "device": R.device_metadata(choice.device), "determinism": settings,
        "runtime": runs[0].runtime, "runtime_repeat": runs[1].runtime,
        "deterministic_repeat": {"state_dict_equal": state_dict_sha256(runs[0].model) == state_dict_sha256(runs[1].model),
                                 "history_equal": runs[0].history == runs[1].history},
        "state_dict_sha256": state_dict_sha256(runs[0].model),
        "best_val_ndcg@3": runs[0].best_val, "best_epoch": runs[0].best_epoch,
        "checkpoint_bytes": size, "peak_process_rss_mib": R.peak_rss_mib(),
        "seq_fingerprints": {"train": tr.fingerprint(), "val": va.fingerprint()},
    }


def run(data_root: Path, devices=("cpu", "cuda")) -> dict:
    manifests = I.load_manifests()
    profile, seed = DATASET
    I.check_seed_namespace([seed], NAMESPACE, manifests)
    proc = data_root / NAMESPACE / profile / f"seed{seed}" / "processed"
    if not (proc / "prep_metadata.json").exists():
        raise FileNotFoundError(f"{proc} missing; run `python -m bri.seq.prepare` first")
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    results = []
    for name, cfg in CONFIGS.items():
        for task in ("nbs", "nbp"):
            for device in devices:
                spec = {"processed_dir": str(proc), "config": cfg, "task": task, "device": device}
                r = subprocess.run([sys.executable, "-m", "bri.seq.bench", "--worker", json.dumps(spec)],
                                   capture_output=True, text=True, env=env)
                if r.returncode != 0:
                    raise RuntimeError(f"worker failed ({name}/{task}/{device}):\n{r.stderr[-3000:]}")
                results.append({"config": name, "task": task, "requested": device,
                                **json.loads(r.stdout.strip().splitlines()[-1])})
                print(f"  {name}/{task}/{device}: done", file=sys.stderr)
    return {"label": SYNTHETIC_LABEL, "namespace": NAMESPACE, "dataset": {"profile": profile, "seed": seed},
            "epochs": EPOCHS, "seed": SEED, "configs": CONFIGS, "results": results}


def to_markdown(r: dict) -> str:
    lines = [f"# SASRec Full runtime benchmark ({r['dataset']['profile']} seed {r['dataset']['seed']}, "
             f"{r['epochs']} fixed epochs)", "", f"> **{r['label']}**", "",
             "Timings are from the REPEAT run in the same process (steady state); 'first run' includes "
             "one-off warm-up (CUDA context, kernel loading, allocator).", "",
             "| Config | Task | Device | optim s (steady) | train s (steady) | first run s | s/epoch | infer ms/case "
             "| peak GPU MiB | peak RSS MiB | params | ckpt KiB | repeat identical | val NDCG@3 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in r["results"]:
        if "skipped" in x:
            lines.append(f"| {x['config']} | {x['task']} | {x['requested']} | skipped: {x['skipped']} |")
            continue
        rt, st = x["runtime"], x["runtime_repeat"]
        same = x["deterministic_repeat"]["state_dict_equal"] and x["deterministic_repeat"]["history_equal"]
        lines.append(f"| {x['config']} | {x['task']} | {st['device']} | {st['optimization_seconds']:.2f} "
                     f"| {st['train_seconds']:.2f} | {rt['train_seconds']:.2f} | {st['seconds_per_epoch']:.3f} "
                     f"| {st['inference_ms_per_case']:.4f} | {st['peak_gpu_memory_mib'] or '-'} "
                     f"| {x['peak_process_rss_mib']} | {st['parameters']} | {x['checkpoint_bytes'] / 1024:.0f} "
                     f"| {'yes' if same else 'NO'} | {x['best_val_ndcg@3']:.4f} |")
    lines += ["", "optim s = optimization only; train s also includes per-epoch validation through the shared "
              "evaluator (CPU-side pandas work, identical on both devices). Peak RSS is the whole worker "
              "process (data + model + CUDA runtime), not the model alone.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--data-root", type=Path, default=Path("data/phase3"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase3/bench"))
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"], choices=["cpu", "cuda"])
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(worker(json.loads(args.worker))))
        return 0
    try:
        report = run(args.data_root, tuple(args.devices))
    except I.LockedTestSplitError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "bench_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    md = to_markdown(report)
    (args.output_dir / "bench_report.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
