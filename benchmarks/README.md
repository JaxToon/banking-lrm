# Benchmark registry

Frozen experiment reports. Files here are never regenerated or overwritten by new
experiments. All results are **synthetic-data experiments**: they are not evidence of
real customer behavior or real-world recommendation effectiveness.

| Benchmark | Dataset | Code | Test split status |
|---|---|---|---|
| `benchmark-v1` | generator 0.2.0, seed 42, 500 customers, 30 days, split 16/7/7; events.csv sha256 `ca53375d…a34fa8` | git tag `benchmark-v1` (suite `benchmark-v1`) | **Locked.** Used once for Phase 2.1 final evaluation. |
| `phase2.2` | generator 0.2.0, seed 20261109, 500 customers, 30 days, split 16/7/7; events.csv sha256 `9d68d414…a6927b` | suite `phase2.2` | **Locked.** Used once for Phase 2.2 final evaluation. |
| `phase2.3` | generator 0.3.0, profiles habit_dominant / balanced / transition_dominant x seeds 20261110-20261119 (30 datasets), 500 customers, 30 days, split 16/7/7 | suites `phase2.3` (window) and `phase2.3-next-event` | **Locked.** Used once for Phase 2.3 final evaluation. |

## Manifests

Every locked benchmark has an immutable manifest in `manifests/` (sha256 pinned in
`bri.experiments.integrity.MANIFEST_SHA256`). It records per dataset the source file
hashes and composite fingerprint, processed file hashes, prep fingerprint and generator
identity; the canonical configuration hash per evaluation; version identifiers; and the
hashes of the frozen report files. `python -m bri.experiments.integrity` verifies all
manifests and frozen reports.

**Provenance.** The `benchmark-v1` and `phase2.2` manifests are *retrospective*
(`"provenance": "retrospective"`): they were written in Phase 2.3, after the original
runs, from the regenerated data and the default configuration. They pin the state as it
exists now and protect it from here on, but they cannot prove what data or configuration
the original runs used. The supporting evidence for that is reproduction: the original
suites reproduce the frozen results from this data (`phase2.2` byte for byte). Their
frozen reports were not touched. The `phase2.3` manifest was written at its final
evaluation, before any re-run.

## Rules for locked test splits

- A locked test split must not be used to select models, tune hyperparameters, or
  declare new winners.
- Re-running the *original* suite with its original configurations is allowed, for
  reproducibility only.
- The experiment runner enforces this before reading data: on a locked dataset
  (identified by its `events.csv` checksum or generator identity) it refuses any other
  suite, any configuration whose canonical hash differs from the manifest, and any
  modified source (including consent and catalog), processed, or prep data. Changing
  the grids of a registered suite id is refused on every dataset.
- Reproducing a subset of a locked benchmark (e.g. one profile and seed of `phase2.3`)
  with the registered configuration is allowed; it is still only a reproduction.
  Guards: the robustness runner checks the locks *before* generating or writing
  anything, so a refused run leaves locked data untouched; a locked seed and profile
  reused with different customers, days or split is refused (customer streams do not
  depend on population size, so it would re-expose locked test customers); it refuses
  a `FROZEN` output directory, and refuses to replace an existing
  `robustness_report.json` produced by a different (e.g. the full) configuration.
  Frozen reports are additionally pinned by hash in the manifest.
- New model-selection experiments need a fresh dataset seed. A new seed from the same
  generator is an independent synthetic sample, not evidence of real-world
  generalization.
- Confidence intervals in these reports capture case-to-case variation within one
  dataset only, not variation across generator seeds, dataset sizes, or populations.

## Reproduction notes

- `benchmark-v1`: re-running `--suite benchmark-v1` with current code reproduces every
  frozen number (selections, validation trials, test metrics, CIs, paired differences,
  random floor). The current JSON adds fields (`family`, per-K pairwise differences,
  `key_comparisons`). For byte-identical output, check out git tag `benchmark-v1`.
- `phase2.2`: re-running `--suite phase2.2` reproduces the frozen JSON byte for byte
  (re-verified in Phase 2.3 after the runner refactor).
- `phase2.3`: `python -m bri.experiments.robustness` with defaults regenerates all 30
  datasets; per-dataset window and next-event reports were spot-checked to reproduce
  byte for byte.
