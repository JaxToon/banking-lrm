# Banking Recommendation Intelligence

A public portfolio project exploring privacy-aware, hybrid Large Recommendation Model (LRM) + Large Language Model (LLM) personalization for a fictional Indonesian mobile banking environment.

**This repository is an independent educational prototype.** It is not affiliated with, endorsed by, or built using data, systems, policies, or confidential information from any real bank. All customer records, events, products, policies, and outcomes generated here are synthetic.

## Project objective

Build a local-first prototype that learns from synthetic mobile-banking navigation and service-use events, then recommends both:
- **Next Best Service (NBS):** e.g. transfer, bill payment, QR payment, top-up.
- **Next Best Product (NBP):** e.g. savings, time deposit, debit/card information, financing information.

The initial scope is limited to event-level behavioral personalization. It does not perform credit scoring, credit approval, financial suitability assessment, fraud detection, or autonomous financial decisions.

## Phase 0–1 scope

1. Establish a safe public-repository boundary and project conventions.
2. Generate reproducible synthetic customers, product/service catalog, consent preferences, and event sequences.
3. Validate schemas and data quality.
4. Store generated outputs locally under `data/synthetic/` (ignored by Git by default).

No real customer data, account data, transaction amounts, balances, beneficiary data, credentials, or production bank documents are needed or allowed.

## Quick start

Requirements: Python 3.11 (recommended), Git, and VS Code.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m bri.data.generate_synthetic --customers 500 --days 30 --seed 42 --output-dir data/synthetic
python -m bri.data.validate_dataset --data-dir data/synthetic
pytest -q
```

If PowerShell blocks activation, use:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m bri.data.generate_synthetic --customers 500 --days 30 --seed 42 --output-dir data/synthetic
python -m bri.data.validate_dataset --data-dir data/synthetic
pytest -q
```

## Generated dataset

The generator creates, under `--output-dir`:
- `customers.csv`: synthetic profiles (segment, tenure, engagement, language)
- `catalog.csv`: 5 generic services + 6 generic product-information items
- `consent_preferences.csv`: one row per customer per purpose
- `events.csv`: navigation, service-use, and product-discovery events grouped into sessions
- `generation_metadata.json`: seed, window, row counts, SHA-256 checksums

Same arguments produce byte-identical files on Windows and Linux. A 500-customer/30-day
run produces about 44k events in a few seconds; 5,000 customers/90 days produces about
1.3M events in under a minute. Schema and rules: [docs/03_DATA_DICTIONARY.md](docs/03_DATA_DICTIONARY.md).
Design rationale: [docs/07_PHASE_1_DESIGN_DECISIONS.md](docs/07_PHASE_1_DESIGN_DECISIONS.md).

## Phase 2 foundations: splits, consent-aware examples, evaluation

Same commands on Windows PowerShell and macOS/Linux (after generating the dataset above):

```bash
python -m bri.prep.build --data-dir data/synthetic --output-dir data/processed --val-days 7 --test-days 7
python -m bri.eval.evaluate --processed-dir data/processed --split val --ranker random
python -m bri.eval.evaluate --processed-dir data/processed --split val --ranker oracle --output data/processed/report_val_oracle.json
```

- `bri.prep.build` validates the source dataset, keeps only events with explicit,
  unambiguous consent for their purpose, splits by time, and writes training examples
  and evaluation cases for NBS and NBP separately.
- `bri.eval.evaluate` scores a `case_id,item_id,rank` recommendations CSV
  (`--recommendations`), or a built-in sanity ranker, with Precision/Recall/NDCG/MAP@K,
  coverage, and diversity per task.
- `random` and `oracle` are pipeline sanity checks (floor and ceiling), not
  recommendation models. All outputs are labelled as synthetic-data experiments.

Protocol, decisions, and limitations: [docs/08_PHASE_2_FOUNDATIONS_REPORT.md](docs/08_PHASE_2_FOUNDATIONS_REPORT.md).

## Phase 2.1 / 2.2: classical baselines and ablation

Benchmark-v1 (Phase 2.1, dataset seed 42) is frozen in `benchmarks/benchmark-v1/`, and
its test split is locked. It can only be reproduced with its original suite:

```bash
python -m bri.experiments.baselines --suite benchmark-v1 --processed-dir data/processed --output-dir artifacts/benchmark_v1_repro
```

Phase 2.2 (Personal Frequency + ItemKNN ablation) uses a fresh dataset, seed 20261109:

```bash
python -m bri.data.generate_synthetic --customers 500 --days 30 --seed 20261109 --output-dir data/synthetic/seed20261109
python -m bri.data.validate_dataset --data-dir data/synthetic/seed20261109
python -m bri.prep.build --data-dir data/synthetic/seed20261109 --output-dir data/processed/seed20261109 --val-days 7 --test-days 7
python -m bri.experiments.baselines --suite phase2.2 --processed-dir data/processed/seed20261109 --output-dir artifacts/phase2_2_repro
```

The runner fits on train, selects each model's configuration on validation (NDCG@3),
evaluates once on test, and reports paired bootstrap 95% CIs (customer resampling),
pairwise comparisons, and a 5-seed random floor, for NBS and NBP separately. It refuses
to run a different suite on a locked dataset, and refuses to write into a `FROZEN`
benchmark directory. See [benchmarks/README.md](benchmarks/README.md),
[docs/09_PHASE_2_1_BASELINES_REPORT.md](docs/09_PHASE_2_1_BASELINES_REPORT.md), and
[docs/10_PHASE_2_2_PERSONAL_FREQUENCY_REPORT.md](docs/10_PHASE_2_2_PERSONAL_FREQUENCY_REPORT.md).
All results are synthetic-data experiments.

## Phase 2.3: benchmark integrity and multi-seed behavioral robustness

Verify the immutable benchmark manifests and frozen reports:

```bash
python -m bri.experiments.integrity
```

Generator 0.3.0 adds behavior profiles (invented experimental assumptions, not real
banking behavior); omitting `--profile` keeps generator 0.2.0, byte-identical:

```bash
python -m bri.data.generate_synthetic --customers 500 --days 30 --seed 20261110 --profile transition_dominant --output-dir data/phase2_3/transition_dominant/seed20261110/synthetic
```

Run the whole robustness experiment (3 profiles x 10 seeds; generate, validate, split,
window + next-event evaluation of all five classical baselines; about 20 minutes on a
laptop CPU):

```bash
python -m bri.experiments.robustness --data-root data/phase2_3 --output-dir artifacts/phase2_3
```

The Phase 2.3 test splits are locked (`benchmarks/manifests/phase2.3.json`): the
command above with its default configuration is a reproduction; any other configuration
on those seeds is refused. A single dataset's next-event evaluation can be run with
`python -m bri.experiments.baselines --suite phase2.3-next-event --evaluation next_event --processed-dir DIR --output-dir DIR`.
Before reading any data, every experiment checks that processed files match their prep
checksums, that registered suites keep their grids, and that locked datasets are only
reproduced with unmodified data and configuration. Results:
[docs/11_PHASE_2_3_ROBUSTNESS_REPORT.md](docs/11_PHASE_2_3_ROBUSTNESS_REPORT.md).
All results are synthetic-data experiments.

## Phase 3: sequential recommendation (SASRec) and ablations

**Question.** Does sequential modeling add predictive value beyond Personal Frequency
and the classical baselines? SASRec Full and four controlled ablations (NoPosition,
Shuffled, MeanPool, SessionBoundary) are compared with the classical models on 30
fresh synthetic datasets (3 behavior profiles x 10 seeds). The primary protocol is
next-event; the 7-day window is secondary.

**Result (synthetic-data experiment).**
- SASRec beats Personal Frequency clearly in one of six groups, `transition_dominant` /
  NBS: next-event NDCG@1 0.400 vs 0.235, +0.164 [+0.152, +0.176] across seeds,
  positive in 10/10 seeds.
- That gain comes almost entirely from **within-session** targets (0.868 vs 0.213).
  On cross-session targets, and in the habit and balanced regimes, Personal Frequency
  stays as good or better.
- Chronological order and attention matter: Full beats Shuffled and MeanPool.
  Explicit positional embeddings and session-boundary embeddings add nothing
  measurable.
- In the 7-day window protocol, SASRec trails Personal Frequency in every group.

Full report: [docs/12_PHASE_3_SEQUENTIAL_REPORT.md](docs/12_PHASE_3_SEQUENTIAL_REPORT.md).
Release notes: [docs/RELEASE_NOTES_PHASE3.md](docs/RELEASE_NOTES_PHASE3.md). Compact derived
summary: `benchmarks/phase3-summary/phase3_summary.json` (the frozen report stays authoritative).

### Optional dependencies

Phase 3 needs PyTorch 2.11.0 and Optuna 4.9.0 (optional `seq` extra). Nothing before
Phase 3 needs them, and the default test suite skips the Phase 3 tests when torch is
absent. A GPU is never required for tests.

```bash
pip install -r requirements.txt -r requirements-seq-cpu.txt    # CPU build, any machine
pip install -r requirements.txt -r requirements-seq-cuda.txt   # CUDA 12.8 build (driver 12.8+; needed for sm_120)
python -m bri.seq.runtime --device auto                        # device, CPU fallback, determinism, versions
```

Equivalent explicit commands:
`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128` and
`pip install optuna==4.9.0`.

### Commands (same on Windows PowerShell and macOS/Linux)

```bash
python -m bri.experiments.integrity          # verify every manifest, the seed reservation, frozen config and reports
pytest -q                                    # default suite (CUDA tests excluded)
pytest -q -m cuda                            # dedicated CUDA validation run (required before a final run)
python -m bri.seq.prepare                    # phase3_tuning datasets + train-only history-length report
python -m bri.seq.hpo report                 # HPO report from the SQLite study (the search itself is frozen)
python -m bri.experiments.phase3 run --output-dir artifacts/phase3/repro --profiles transition_dominant --seeds 20261134
```

The last command is a subset reproduction of the locked final evaluation. It runs
every gate first: both test runs, manifests, frozen-configuration hash,
`require_device("cuda:0")`, the seed namespace, pipeline versions and the runtime
fingerprint. It must write to a new output directory. Re-running HPO, changing the
frozen configuration, writing into `benchmarks/phase3/`, or replacing the aggregate
report is refused.

Seeds: `phase3_tuning` 20261120-20261122 (validation only) and `phase3_final`
20261130-20261139 (test used once), reserved in `benchmarks/manifests/phase3-seeds.json`
before any Phase 3 dataset existed. Other runners refuse them.

## Repository map

```text
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── docs/
│   ├── 00_PROJECT_CHARTER.md
│   ├── 01_PHASE_0_1_IMPLEMENTATION.md
│   ├── 02_SYSTEM_ARCHITECTURE.md
│   ├── 03_DATA_DICTIONARY.md
│   ├── 04_PRIVACY_COMPLIANCE.md
│   ├── 05_PUBLIC_GITHUB_SAFETY.md
│   ├── 06_CLAUDE_CODE_TASKS.md
│   ├── 07_PHASE_1_DESIGN_DECISIONS.md
│   ├── 08_PHASE_2_FOUNDATIONS_REPORT.md
│   ├── 09_PHASE_2_1_BASELINES_REPORT.md
│   ├── 10_PHASE_2_2_PERSONAL_FREQUENCY_REPORT.md
│   ├── 11_PHASE_2_3_ROBUSTNESS_REPORT.md
│   ├── 12_PHASE_3_SEQUENTIAL_REPORT.md
│   └── RELEASE_NOTES_PHASE3.md
├── benchmarks/              # frozen reports + immutable manifests (benchmark-v1, phase2.2, phase2.3, phase3)
├── data/synthetic/          # generated output (git-ignored)
├── data/processed/          # Phase 2 outputs (git-ignored)
├── data/phase2_3/           # Phase 2.3 datasets (git-ignored)
├── data/phase3/             # Phase 3 tuning + final datasets (git-ignored)
├── src/bri/
│   ├── tasks.py                 # NBS/NBP task definitions
│   ├── data/
│   │   ├── schema.py            # single source of truth for the schema
│   │   ├── generate_synthetic.py
│   │   └── validate_dataset.py
│   ├── prep/
│   │   ├── split.py             # temporal splitting
│   │   ├── consent.py           # fail-closed consent filter
│   │   ├── examples.py          # interactions, train examples, eval cases, negatives
│   │   └── build.py             # CLI
│   ├── eval/
│   │   ├── metrics.py           # ranking + catalog metrics
│   │   ├── evaluate.py          # recommendation validation, reports, CLI
│   │   ├── next_event.py        # next-event cases + strict-history counts
│   │   └── bootstrap.py         # paired (and customer-cluster) bootstrap CIs
│   ├── seq/                     # Phase 3 (optional `seq` extra)
│   │   ├── runtime.py           # device selection, seeding, determinism, runtime fingerprint
│   │   ├── data.py              # next-event sequence cases (train/val/test), tie and boundary rules
│   │   ├── prepare.py           # tuning datasets + train-only history-length report
│   │   ├── sasrec.py            # SASRec Full + NoPosition / Shuffled / MeanPool / SessionBoundary
│   │   ├── train.py             # training, validation-only early stopping, checkpoints
│   │   ├── hpo.py               # Optuna TPE search on tuning seeds, frozen-config manifest
│   │   └── bench.py             # CPU vs CUDA runtime benchmark
│   ├── models/
│   │   ├── common.py            # train/history matrices, deterministic ranking
│   │   └── baselines.py         # popularity, personal frequency, item-item CF, implicit MF, suites
│   └── experiments/
│       ├── baselines.py         # train-fit / val-select / test-evaluate runner (window, next-event)
│       ├── integrity.py         # fingerprints, config hashes, immutable manifests, seed namespaces
│       ├── robustness.py        # Phase 2.3 multi-seed orchestration + statistics
│       └── phase3.py            # Phase 3 gated final experiment, statistics, lock
└── tests/                       # unit + integration tests
```

## Roadmap after Phase 1

- Phase 2: temporal splits, consent-aware examples, and evaluation framework (done); Phase 2.1 classical baselines (done, benchmark-v1); Phase 2.2 personal frequency + ItemKNN ablation (done); Phase 2.3 benchmark integrity + multi-seed behavioral robustness (done).
- Phase 3: sequential recommendation (SASRec + ablations) — done, reviewed and released as v0.3.0. Phase 3.1 (hybrid Personal Frequency + SASRec, pre-registered, fresh seeds) — done: final evaluation locked (`benchmarks/phase3_1/`, manifest `phase3_1`). Synthetic-data outcome: trade-off, not overall success — static fusion keeps SASRec's gain on transition-dominant NBS, non-inferiority to Personal Frequency was not demonstrated for habit/NBP and balanced/NBP, and the adaptive gate was not evaluable under the preregistered protocol.
- Phase 4: multi-behavior LRM-inspired Transformer and scaling experiments.
- Phase 5: LLM product-catalog explanation with strict context minimization.
- Phase 6: deterministic eligibility/consent/frequency policy service.
- Phase 7: API + local dashboard.
- Phase 8: offline evaluation, security testing, model governance, and production-readiness assessment.

## Important terminology

This prototype should be described publicly as an **LRM-inspired hybrid recommendation research prototype**, not as a production-grade or industry-scale Large Recommendation Model. Model size alone does not make a recommender an LRM; scaling behavior, data scale, task breadth, and evaluation must be demonstrated.
