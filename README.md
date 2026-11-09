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
│   └── 10_PHASE_2_2_PERSONAL_FREQUENCY_REPORT.md
├── benchmarks/              # frozen, locked benchmark reports (benchmark-v1, phase2.2)
├── data/synthetic/          # generated output (git-ignored)
├── data/processed/          # Phase 2 outputs (git-ignored)
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
│   │   └── bootstrap.py         # paired bootstrap CIs over cases
│   ├── models/
│   │   ├── common.py            # train/history matrices, deterministic ranking
│   │   └── baselines.py         # popularity, personal frequency, item-item CF, implicit MF, suites
│   └── experiments/
│       └── baselines.py         # train-fit / val-select / test-evaluate runner
└── tests/                       # unit + integration tests
```

## Roadmap after Phase 1

- Phase 2: temporal splits, consent-aware examples, and evaluation framework (done); Phase 2.1 classical baselines (done, benchmark-v1); Phase 2.2 personal frequency + ItemKNN ablation (done, pending review); Phase 2.3 multi-seed robustness (planned).
- Phase 3: sequential recommendation (SASRec-style baseline).
- Phase 4: multi-behavior LRM-inspired Transformer and scaling experiments.
- Phase 5: LLM product-catalog explanation with strict context minimization.
- Phase 6: deterministic eligibility/consent/frequency policy service.
- Phase 7: API + local dashboard.
- Phase 8: offline evaluation, security testing, model governance, and production-readiness assessment.

## Important terminology

This prototype should be described publicly as an **LRM-inspired hybrid recommendation research prototype**, not as a production-grade or industry-scale Large Recommendation Model. Model size alone does not make a recommender an LRM; scaling behavior, data scale, task breadth, and evaluation must be demonstrated.
