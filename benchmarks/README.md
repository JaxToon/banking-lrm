# Benchmark registry

Frozen experiment reports. Files here are never regenerated or overwritten by new
experiments. All results are **synthetic-data experiments**: they are not evidence of
real customer behavior or real-world recommendation effectiveness.

| Benchmark | Dataset | Code | Test split status |
|---|---|---|---|
| `benchmark-v1` | generator 0.2.0, seed 42, 500 customers, 30 days, split 16/7/7; events.csv sha256 `ca53375d…a34fa8` | git tag `benchmark-v1` (suite `benchmark-v1`) | **Locked.** Used once for Phase 2.1 final evaluation. |
| `phase2.2` | generator 0.2.0, seed 20261109, 500 customers, 30 days, split 16/7/7; events.csv sha256 `9d68d414…a6927b` | suite `phase2.2` | **Locked.** Used once for Phase 2.2 final evaluation. |

## Rules for locked test splits

- A locked test split must not be used to select models, tune hyperparameters, or
  declare new winners.
- Re-running the *original* suite with its original configurations is allowed, for
  reproducibility only.
- The experiment runner enforces this: on a locked dataset (identified by its
  `events.csv` checksum) it refuses any suite other than the one that originally used
  it.
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
- `phase2.2`: re-running `--suite phase2.2` reproduces the frozen JSON byte for byte.
