# Phase 0–1 Implementation Runbook

## Hardware target

- 16 GB RAM.
- 4+ CPU cores recommended.
- 10 GB free disk space for the initial repository and local experiments.
- GPU is not required for Phase 0–1.
- Python 3.11 recommended.

## Phase 0 — Foundation

### Step 1: Create the repository

Create a public GitHub repository named `banking-recommendation-intelligence`. Use a generic description such as:

> A privacy-aware, synthetic-data research prototype exploring hybrid sequential recommendation and LLM-based explanations for mobile banking.

Do not use any real institution's name, logo, internal vocabulary, product naming, screenshots, or confidential information.

### Step 2: Set up VS Code

Install:
- Python extension by Microsoft.
- Pylance.
- GitLens (optional).
- Claude Code extension or Claude Code CLI (optional).

Open the repository root in VS Code. Open a terminal in the repository root.

### Step 3: Create a virtual environment

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

macOS/Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Select `.venv` as the Python interpreter in VS Code: `Ctrl+Shift+P` → `Python: Select Interpreter`.

### Step 4: Confirm environment

```bash
python --version
python -c "import pandas, numpy; print('environment-ok')"
```

### Step 5: Generate data

```bash
python -m bri.data.generate_synthetic --customers 500 --days 30 --seed 42 --output-dir data/synthetic
```

Scale up after validation:
```bash
python -m bri.data.generate_synthetic --customers 5000 --days 90 --seed 42 --output-dir data/synthetic
```

Start small. A 500-customer, 30-day dataset is enough to verify the pipeline. The generator is designed to remain lightweight; do not generate millions of rows on a 16 GB laptop before profiling.

## Phase 1 — Synthetic data engineering

### Step 1: Inspect generated files

Expected files:
- `customers.csv`
- `catalog.csv`
- `consent_preferences.csv`
- `events.csv`
- `generation_metadata.json`

### Step 2: Validate data

```bash
python -m bri.data.validate_dataset --data-dir data/synthetic
```

### Step 3: Run tests

```bash
pytest -q
```

### Step 4: Inspect event distribution

```bash
python -c "import pandas as pd; d=pd.read_csv('data/synthetic/events.csv'); print(d['event_type'].value_counts()); print(d.groupby('surface').size())"
```

### Step 5: Commit only code and documentation

The generated data directory is ignored by Git. Confirm:
```bash
git status --short
git check-ignore data/synthetic/events.csv
```

## Phase 1 acceptance criteria

- All output CSV files load without parsing errors.
- Every event references an existing synthetic customer.
- Every non-null `item_id` references the catalog.
- Event IDs are unique.
- Event timestamps are valid UTC ISO-8601 timestamps.
- Event types and surfaces are from the documented vocabulary.
- Consent purpose is explicit and compatible with the event purpose.
- No prohibited fields are generated.
- Same seed and same arguments reproduce the same logical dataset.

## Troubleshooting

### `ModuleNotFoundError: No module named 'bri'`
Run `pip install -e .` from the repository root and confirm VS Code uses `.venv`.

### PowerShell activation is blocked
Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

### Parquet output
Phase 1 writes CSV only, so `pyarrow` is intentionally not a dependency. Add it when a later phase needs Parquet.

### Memory pressure
Reduce `--customers` and `--days`. Do not load all future-scale datasets into memory on the 16 GB laptop.
