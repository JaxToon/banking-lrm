# Claude Code Task Pack — Phase 0 and Phase 1

Use these prompts one at a time. Claude Code should read `CLAUDE.md` and the relevant docs before editing.

## Task 1 — Audit the scaffold

Read `CLAUDE.md`, `README.md`, and all files under `docs/`. Check consistency, missing setup steps, unsafe public-repository content, and contradictions. Do not add features. Return a prioritized list of issues.

## Task 2 — Implement synthetic data generator

Implement `src/bri/data/generate_synthetic.py` according to `docs/03_DATA_DICTIONARY.md`. Requirements:
- CLI arguments: `--customers`, `--days`, `--seed`, `--output-dir`.
- Deterministic output for identical arguments.
- Generate customers, catalog, consent preferences, events, and metadata.
- Use only artificial IDs and generic fictional product/service names.
- Include navigation, service-use, and product-discovery events.
- Do not generate transaction amounts, balances, account numbers, beneficiary details, direct identifiers, precise location, or free text.
- Add helpful CLI messages and input validation.

## Task 3 — Implement dataset validator

Implement `src/bri/data/validate_dataset.py`. It must check required files/columns, types, duplicate IDs, foreign keys, event vocabulary, surface compatibility, timestamp parseability, event ordering, explicit purpose, and prohibited columns. Return non-zero exit status on failure.

## Task 4 — Add tests

Add tests for deterministic generation, expected files, schema, foreign-key integrity, valid event vocabulary, timestamp parsing, prohibited columns, and validator failure on corrupted data. Tests must use temporary directories and must not rely on committed generated datasets.

## Task 5 — Verify Phase 0–1

Run:
```bash
pip install -e .
pytest -q
python -m bri.data.generate_synthetic --customers 500 --days 30 --seed 42 --output-dir data/synthetic
python -m bri.data.validate_dataset --data-dir data/synthetic
```
Fix failures. Report exact commands and actual results. Do not claim success without execution.

## Task 6 — Public repository review

Review all tracked files for real institution names, sensitive information, internal terminology, credentials, and unsupported compliance claims. Do not infer or insert a real bank name. Provide a checklist report before the first public push.
