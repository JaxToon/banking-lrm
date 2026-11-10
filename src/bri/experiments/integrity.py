"""Benchmark integrity: fingerprints, canonical configuration hashes, immutable manifests.

A locked benchmark is described by one manifest in `benchmarks/manifests/<suite>.json`.
Manifests are written once and never edited; the sha256 of every manifest file is
pinned in MANIFEST_SHA256 below, so editing a manifest is detected as tampering.

For each locked dataset a manifest records:
- the sha256 of every source file and a composite fingerprint over all of them,
- the sha256 of every processed file (interactions, train examples, eval cases, targets),
- a fingerprint of the preparation metadata (split cutoffs, candidates, consent counts),
- the canonical hash of each experiment configuration (model grids + protocol),
- generator / preparation / runner version identifiers.

Before an experiment reads any data the runner checks that
1. registered suites still have their registered model grids and key comparisons,
2. processed files on disk match the checksums in their own prep metadata,
3. a dataset that belongs to a locked benchmark (matched by events checksum, or by its
   generator identity) is run only by its own suite, with its registered configuration,
   and with source, processed and prep fingerprints identical to the manifest.

Usage (create a manifest once, after a final evaluation):
    python -m bri.experiments.integrity --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from bri.tasks import SYNTHETIC_LABEL

MANIFEST_FORMAT = 1
SOURCE_FILES = ("customers.csv", "catalog.csv", "consent_preferences.csv", "events.csv")
PROCESSED_FILES = ("interactions.csv", "train_examples.csv", "eval_cases.csv", "eval_targets.csv")
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = REPO_ROOT / "benchmarks" / "manifests"

# sha256 of every immutable manifest file (LF line endings). Register a new manifest
# by adding one line; never change an existing entry.
MANIFEST_SHA256: dict[str, str] = {
    # Retrospective manifests (written in Phase 2.3, after the original runs; see "provenance").
    "benchmark-v1": "4a9896fedc23b98bbff392af9734a2d09900610f9638a1a7dd779777c1b73edb",
    "phase2.2": "9e4747135c84cdf6e3a3f97daa94b1046061cfeebb74c57cae59e70bc912a2d4",
    # Written at the Phase 2.3 final evaluation.
    "phase2.3": "29e2f0c2c534c8f6c1d0c4c18afe54909b92b12c5c0b306a6d56ae3ca72a35f3",
    # Phase 3 seed reservation (phase3_tuning 20261120-20261122, phase3_final 20261130-20261139),
    # pinned before any Phase 3 dataset was generated.
    "phase3-seeds": "013366971304eabe23691af5e5624de1cf81042f79fc31937fcc7633627dd781",
    # Frozen SASRec configuration (Optuna trial 21 on phase3_tuning validation splits),
    # pinned before any phase3_final dataset was generated.
    "phase3-config": "dadfe7555c0d930aa1479a2aabb533dcf4ee748f6efd2471b0049ac1833271db",
    # Phase 3 final evaluation (30 phase3_final datasets), written after the run.
    "phase3": "0f8112a1942f364bd10423e81250ba68b8bc86d7606f801274b33870db25dc49",
}


class LockedTestSplitError(RuntimeError):
    """The requested run would reuse a locked test split for something other than reproduction."""


class BenchmarkIntegrityError(LockedTestSplitError):
    """Data, configuration or a manifest differs from what was registered."""


# --- hashing ---------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_json(obj) -> str:
    """Deterministic JSON: sorted keys, no whitespace, ASCII only."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_hash(obj) -> str:
    return sha256_bytes(canonical_json(obj).encode("ascii"))


def composite_fingerprint(file_hashes: dict[str, str]) -> str:
    """One fingerprint over a set of named files; any file changing changes it."""
    return canonical_hash(dict(sorted(file_hashes.items())))


def source_fingerprints(data_dir: Path) -> dict:
    files = {name: sha256_file(Path(data_dir) / name) for name in SOURCE_FILES}
    return {"files": files, "composite": composite_fingerprint(files)}


def processed_fingerprints(processed_dir: Path) -> dict:
    files = {name: sha256_file(Path(processed_dir) / name) for name in PROCESSED_FILES}
    return {"files": files, "composite": composite_fingerprint(files)}


def prep_fingerprint(prep_meta: dict) -> str:
    """Hash of the preparation metadata, ignoring the label and the prep code version.

    Covers cutoffs, split parameters, candidate sets (from the catalog), consent-filter
    counts, source fingerprints and processed checksums.
    """
    return canonical_hash({k: v for k, v in prep_meta.items() if k not in ("label", "prep_version")})


# --- configuration -----------------------------------------------------------------

def suite_definition(models: dict, key_comparisons: list) -> dict:
    return {
        "models": {name: {"family": family, "grid": grid} for name, (family, grid) in models.items()},
        "key_comparisons": [list(c[:2]) for c in key_comparisons],
    }


def experiment_config(suite: str, evaluation: str, models: dict, key_comparisons: list, protocol: dict) -> dict:
    """Canonical experiment configuration: suite id, model grids, evaluation protocol."""
    return {"suite": suite, "evaluation": evaluation, **suite_definition(models, key_comparisons),
            "protocol": protocol}


# --- manifests ---------------------------------------------------------------------

def load_manifests(manifest_dir: Path | None = None, pinned: dict[str, str] | None = None) -> dict[str, dict]:
    """Read and verify every manifest. Edited, missing or unregistered manifests are errors."""
    manifest_dir = Path(manifest_dir or MANIFEST_DIR)
    pinned = MANIFEST_SHA256 if pinned is None else pinned
    found = {p.stem: p for p in manifest_dir.glob("*.json")} if manifest_dir.exists() else {}
    unregistered = sorted(set(found) - set(pinned))
    if unregistered:
        raise BenchmarkIntegrityError(f"unregistered benchmark manifest(s) {unregistered} in {manifest_dir}")
    manifests = {}
    for suite, expected in pinned.items():
        if suite not in found:
            raise BenchmarkIntegrityError(f"benchmark manifest for suite '{suite}' is missing")
        raw = found[suite].read_bytes()
        if sha256_bytes(raw) != expected:
            raise BenchmarkIntegrityError(f"benchmark manifest for suite '{suite}' was modified "
                                          "(sha256 differs from the pinned value)")
        manifests[suite] = json.loads(raw.decode("utf-8"))
    verify_seed_reservations(manifests)
    return manifests


# --- seed namespaces -----------------------------------------------------------------
# A seed-reservation manifest has "seed_namespaces": {namespace: {"seeds": [...], ...}}.
# It is pinned like every other manifest BEFORE any dataset with those seeds exists.
# A dataset entry may use a reserved seed only if it declares the same "seed_namespace".

def seed_uses(manifests: dict[str, dict]) -> dict[int, list[tuple[str, str | None]]]:
    """seed -> [(manifest name, declared seed_namespace or None)] over all locked datasets."""
    uses: dict[int, list[tuple[str, str | None]]] = {}
    for name, m in manifests.items():
        for e in m.get("datasets", []):
            uses.setdefault(int(e["seed"]), []).append((name, e.get("seed_namespace")))
    return uses


def seed_reservations(manifests: dict[str, dict]) -> dict[int, str]:
    """seed -> namespace over all reservation manifests (raises on any overlap)."""
    owner: dict[int, str] = {}
    for m in manifests.values():
        for ns, spec in m.get("seed_namespaces", {}).items():
            for s in spec["seeds"]:
                if int(s) in owner:
                    raise BenchmarkIntegrityError(f"seed {s} is reserved by both '{owner[int(s)]}' and '{ns}'")
                owner[int(s)] = ns
    return owner


def verify_seed_reservations(manifests: dict[str, dict]) -> None:
    """Reserved seeds must not collide with locked datasets of another namespace."""
    owner = seed_reservations(manifests)
    for seed, uses in seed_uses(manifests).items():
        if seed in owner:
            bad = [name for name, ns in uses if ns != owner[seed]]
            if bad:
                raise BenchmarkIntegrityError(
                    f"seed {seed} is reserved for namespace '{owner[seed]}' but used by manifest(s) {bad}")


def check_seed_namespace(seeds, namespace: str, manifests: dict[str, dict]) -> None:
    """Refuse unless every seed is reserved for `namespace` (run before generating anything)."""
    owner = seed_reservations(manifests)
    if namespace not in set(owner.values()):
        raise LockedTestSplitError(f"seed namespace '{namespace}' is not registered in any pinned manifest")
    for s in seeds:
        ns = owner.get(int(s))
        if ns != namespace:
            where = f"reserved for '{ns}'" if ns else "not reserved"
            raise LockedTestSplitError(f"seed {s} is {where}; namespace '{namespace}' may only use its own "
                                       "reserved seeds")


def refuse_reserved_seeds(seeds, manifests: dict[str, dict]) -> None:
    """For runners outside any namespace: reserved seeds are off-limits."""
    owner = seed_reservations(manifests)
    taken = sorted({int(s) for s in seeds} & set(owner))
    if taken:
        raise LockedTestSplitError(f"seed(s) {taken} are reserved for namespace(s) "
                                   f"{sorted({owner[s] for s in taken})}; choose other seeds")


def build_seed_reservation(name: str, namespaces: dict[str, dict], manifests: dict[str, dict], notes: str) -> dict:
    """A seed-reservation manifest, verified against every seed already in `manifests`."""
    used = seed_uses(manifests)
    existing = seed_reservations(manifests)
    seen: dict[int, str] = {}
    for ns, spec in namespaces.items():
        for s in spec["seeds"]:
            s = int(s)
            if s in used:
                raise BenchmarkIntegrityError(f"seed {s} ({ns}) is already used by {sorted({u for u, _ in used[s]})}")
            if s in existing:
                raise BenchmarkIntegrityError(f"seed {s} ({ns}) is already reserved for '{existing[s]}'")
            if s in seen:
                raise BenchmarkIntegrityError(f"seed {s} appears in both '{seen[s]}' and '{ns}'")
            seen[s] = ns
    return {
        "manifest_format": MANIFEST_FORMAT,
        "label": SYNTHETIC_LABEL,
        "suite": name,
        "notes": notes,
        "seed_namespaces": namespaces,
        "checked_against": {
            "manifests": {k: canonical_hash(v) for k, v in sorted(manifests.items())},
            "seeds_in_use": sorted(used),
        },
        "configs": {},
        "suite_definition_sha256": {},
        "frozen_report_sha256": {},
        "frozen_report_dir": None,
        "datasets": [],
    }


def write_manifest(manifest: dict, path: Path) -> str:
    """Write a new manifest; refuses to overwrite. Returns its sha256 to pin in code."""
    path = Path(path)
    if path.exists():
        raise BenchmarkIntegrityError(f"{path} exists; manifests are immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return sha256_file(path)


def dataset_entry(data_dir: Path | None, processed_dir: Path, prep_meta: dict, dataset_id: str,
                  evaluations: list[str]) -> dict:
    """Manifest entry for one dataset. Source hashes come from data_dir when given, else
    from the prep metadata (which hashes the source files it actually read)."""
    if data_dir is not None:
        src = source_fingerprints(data_dir)
    else:
        files = {k: prep_meta["source_files_sha256"][k] for k in SOURCE_FILES}
        src = {"files": files, "composite": composite_fingerprint(files)}
    gen = prep_meta.get("source_generation", {})
    return {
        "dataset_id": dataset_id,
        "generator_version": prep_meta["source"]["generator_version"],
        "seed": prep_meta["source"]["seed"],
        "customers": gen.get("customers"),
        "days": gen.get("days"),
        "behavior_profile": gen.get("behavior_profile"),
        "source_files_sha256": src["files"],
        "source_composite_sha256": src["composite"],
        "processed_files_sha256": processed_fingerprints(processed_dir)["files"],
        "prep_fingerprint": prep_fingerprint(prep_meta),
        "evaluations": evaluations,
    }


def build_manifest(suite: str, configs: dict[str, dict], datasets: list[dict], versions: dict,
                   frozen_report_dir: str | None = None, notes: str = "", extra: dict | None = None) -> dict:
    frozen = {}
    if frozen_report_dir:
        d = REPO_ROOT / frozen_report_dir
        frozen = {p.name: sha256_file(p) for p in sorted(d.iterdir()) if p.is_file()}
    return {
        "manifest_format": MANIFEST_FORMAT,
        "label": SYNTHETIC_LABEL,
        "suite": suite,
        "notes": notes,
        "versions": versions,
        "suite_definition_sha256": {e: canonical_hash({k: c[k] for k in ("models", "key_comparisons")})
                                    for e, c in configs.items()},
        "configs": {e: {"config_sha256": canonical_hash(c), "config": c} for e, c in configs.items()},
        "frozen_report_dir": frozen_report_dir,
        "frozen_report_sha256": frozen,
        "datasets": datasets,
        **(extra or {}),
    }


# --- verification ------------------------------------------------------------------

def verify_processed_dir(processed_dir: Path, prep_meta: dict) -> dict:
    """Processed files must match the checksums their prep metadata recorded."""
    fp = processed_fingerprints(processed_dir)
    recorded = prep_meta.get("sha256", {})
    for name, digest in fp["files"].items():
        if recorded.get(name) != digest:
            raise BenchmarkIntegrityError(
                f"processed file {name} does not match prep_metadata.json (modified after preparation?); "
                "rebuild with `python -m bri.prep.build`")
    return fp


def check_suite_definition(suite: str, evaluation: str, models: dict, key_comparisons: list,
                           manifests: dict[str, dict]) -> None:
    """A registered suite identifier must keep its registered grids and comparisons."""
    current = canonical_hash(suite_definition(models, key_comparisons))
    for m in manifests.values():
        for e, c in m["configs"].items():
            if c["config"]["suite"] == suite and e == evaluation and current != m["suite_definition_sha256"][e]:
                raise BenchmarkIntegrityError(
                    f"suite '{suite}' ({evaluation}) no longer matches its registered model grids / key "
                    "comparisons; register changed configurations under a new suite identifier")


def find_locked_dataset(prep_meta: dict, manifests: dict[str, dict]) -> tuple[dict, dict] | None:
    """(manifest, dataset entry) if this dataset belongs to a locked benchmark.

    Matched by events checksum, or by generator identity (version, seed, customers,
    days, profile) so that a regenerated-and-edited copy of a locked dataset is caught.
    """
    events = prep_meta.get("source", {}).get("events_sha256")
    gen = prep_meta.get("source_generation")
    identity = None
    if gen is not None:
        identity = (prep_meta["source"].get("generator_version"), prep_meta["source"].get("seed"),
                    gen.get("customers"), gen.get("days"), gen.get("behavior_profile"))
    for m in manifests.values():
        for entry in m["datasets"]:
            if entry["source_files_sha256"]["events.csv"] == events:
                return m, entry
            if identity is not None and identity == (entry["generator_version"], entry["seed"],
                                                     entry["customers"], entry["days"],
                                                     entry["behavior_profile"]):
                return m, entry
    return None


def check_locked_run(prep_meta: dict, processed_fp: dict, suite: str, evaluation: str, config: dict,
                     manifests: dict[str, dict]) -> dict | None:
    """Raise unless this run is an exact reproduction of a locked benchmark (or not locked)."""
    hit = find_locked_dataset(prep_meta, manifests)
    if hit is None:
        return None
    m, entry = hit
    owner = m["suite"]
    registered = {e: c["config"]["suite"] for e, c in m["configs"].items()}  # evaluation -> suite id
    if suite not in registered.values():
        raise LockedTestSplitError(
            f"the test split of this dataset is locked by suite '{owner}'. It may only be re-run with "
            f"--suite {owner} for reproducibility; use a fresh dataset seed for suite '{suite}'.")
    if evaluation not in entry["evaluations"] or registered.get(evaluation) != suite:
        raise LockedTestSplitError(f"suite '{suite}' never ran a '{evaluation}' evaluation on this "
                                   f"locked dataset (benchmark '{owner}')")
    if canonical_hash(config) != m["configs"][evaluation]["config_sha256"]:
        raise BenchmarkIntegrityError(
            f"configuration differs from the one registered for locked suite '{owner}' ({evaluation}); "
            "only the original configuration may be re-run")
    if "source_files_sha256" not in prep_meta:
        raise BenchmarkIntegrityError("prep metadata has no source fingerprints; rebuild with the current "
                                      "`python -m bri.prep.build` before reproducing a locked benchmark")
    if prep_meta["source"].get("events_sha256") != entry["source_files_sha256"]["events.csv"]:
        raise BenchmarkIntegrityError(f"events of locked dataset '{entry['dataset_id']}' were modified")
    for name, digest in entry["source_files_sha256"].items():
        if prep_meta["source_files_sha256"].get(name) != digest:
            raise BenchmarkIntegrityError(
                f"source file {name} of locked dataset '{entry['dataset_id']}' differs from the manifest")
    for name, digest in entry["processed_files_sha256"].items():
        if processed_fp["files"].get(name) != digest:
            raise BenchmarkIntegrityError(
                f"processed file {name} of locked dataset '{entry['dataset_id']}' differs from the manifest")
    if prep_fingerprint(prep_meta) != entry["prep_fingerprint"]:
        raise BenchmarkIntegrityError(
            f"prep metadata of locked dataset '{entry['dataset_id']}' differs from the manifest")
    return entry


def verify_frozen_reports(manifests: dict[str, dict]) -> None:
    for m in manifests.values():
        for name, digest in m.get("frozen_report_sha256", {}).items():
            path = REPO_ROOT / m["frozen_report_dir"] / name
            if not path.exists() or sha256_file(path) != digest:
                raise BenchmarkIntegrityError(f"frozen report {path} was modified or removed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify benchmark manifests and frozen reports.")
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    args = parser.parse_args(argv)
    try:
        manifests = load_manifests(args.manifest_dir)
        verify_frozen_reports(manifests)
    except BenchmarkIntegrityError as exc:
        print(f"INTEGRITY CHECK FAILED: {exc}", file=sys.stderr)
        return 1
    for suite, m in manifests.items():
        if "seed_namespaces" in m:
            print(f"{suite}: seed reservation ok, " + ", ".join(
                f"{ns} {min(v['seeds'])}-{max(v['seeds'])} ({len(v['seeds'])} seeds)"
                for ns, v in m["seed_namespaces"].items()))
            continue
        if "frozen_config" in m:
            print(f"{suite}: frozen configuration ok (trial {m['selection']['trial']}, "
                  f"config sha256 {m['frozen_config_sha256'][:12]}...)")
            continue
        provenance = m.get("provenance", "written at final evaluation")
        print(f"{suite}: manifest ok ({provenance}), {len(m['datasets'])} locked dataset(s), "
              f"{len(m.get('frozen_report_sha256', {}))} frozen report file(s) unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
