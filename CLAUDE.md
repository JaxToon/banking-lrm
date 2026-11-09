# Instructions for Claude Code

## Project context

You are helping develop **Banking Recommendation Intelligence**, an independent public portfolio project for a fictional Indonesian mobile banking environment. The real-world context is a commercial bank, but **never identify, name, imply, or include any real bank or employer in code, documentation, prompts, examples, commit messages, screenshots, test fixtures, or generated content**.

This repository must remain safe for a public GitHub portfolio. Treat all data as synthetic. Do not request, accept, create examples from, or commit real customer data, internal documents, production schemas, credentials, API keys, internal URLs, screenshots, or proprietary business rules.

## Product scope

- Behavioral scope: both mobile-app navigation behavior and service-use behavior.
- Recommendation scope: both Next Best Service (NBS) and Next Best Product (NBP).
- Local development target: laptop with 16 GB RAM.
- Completed: Phase 0 and Phase 1 (project foundation, synthetic dataset generator, schema validation); Phase 2 foundations (temporal split, consent-aware examples, shared evaluation framework), reviewed and merged.

### Current delivery: Phase 2.1 — Classical Recommendation Baselines

Authorized:
- Popularity-based recommendation for NBS and NBP.
- Item-item collaborative filtering.
- Matrix Factorization.
- Shared model comparison and bootstrap confidence intervals.
- Reproducible offline experiments using the existing Phase 2 foundation.

Not yet authorized:
- LightGBM ranking.
- Sequential recommendation / SASRec.
- Transformer or LRM implementation.
- LLM integration and RAG.
- API, dashboard, and production deployment.

All models must use the existing temporal split, consent-filtered interactions, task-specific candidate sets, and shared evaluation framework.

Fit models using training data, select configurations using validation data, and reserve test data for final evaluation.

Every reported result must be explicitly labelled as a synthetic-data experiment.

Additional Phase 2.1 requirements (from review):
- Model comparisons use paired bootstrap confidence intervals with the customer/case as the resampling unit; the random floor is reported over several seeds.
- `exposure_diversity` measures catalog exposure evenness, not per-customer semantic diversity; keep that name and interpretation.
- Results cover only customers with sufficient history; cold start is an explicit limitation and a separate future experiment.
- Never trade away NBS/NBP purpose separation to improve scores.
- Dependencies: SciPy approved (add only when code needs it). LightGBM and PyTorch not approved.

### Scope notes
- Architectural invariants in `docs/02_SYSTEM_ARCHITECTURE.md` §3 always apply: purpose metadata is not permission; synthetic results are not real-world evidence; recommendation and policy stay separate.
- Later architecture: sequential LRM + LLM explanation layer, deterministic policy/eligibility layer, API, dashboard.
- The prototype must not make credit decisions, credit scoring, financial suitability assessments, fraud decisions, or execute transactions.

## Engineering principles

1. Prefer simple, readable, typed Python over premature abstractions.
2. Keep local execution possible on CPU and 16 GB RAM.
3. Use deterministic synthetic data generation with an explicit seed.
4. Never use real personal data. Synthetic identifiers must be obviously artificial.
5. Keep policy decisions deterministic and separate from ML/LLM inference.
6. Do not send customer-level data to public external APIs.
7. Do not add a dependency unless it is needed and justified.
8. Add tests for schema, referential integrity, temporal ordering, reproducibility, and privacy boundaries.
9. Use UTC timestamps in storage; convert to local time only for presentation.
10. Avoid storing raw free text from users or transaction descriptions.
11. Keep generated data out of Git unless a small, explicitly reviewed fixture is required.
12. Do not claim regulatory compliance. Say "designed with controls informed by" applicable requirements and require qualified review.

## Public repository language

Use generic names only:
- "fictional Indonesian commercial bank"
- "mobile banking application"
- "banking product catalog"
- "synthetic customer"
- "synthetic event"

Do not include any real bank name, logo, color system, internal terminology, employee names, product names, account formats, URLs, or non-public business logic.

## Workflow for every task

1. Inspect existing files before editing.
2. Summarize intended changes briefly.
3. Implement only the requested phase; do not jump ahead to training or deployment.
4. Run tests and relevant CLI commands.
5. Report changed files, commands run, results, and known limitations.
6. Never claim a test passed unless it was actually executed.
7. Ask before adding external services, cloud resources, or dependencies with licensing implications.

## Definition of done for Phase 0–1

- Setup instructions work on Windows PowerShell and Linux/macOS.
- Synthetic generator is deterministic for the same seed.
- Output schema is documented.
- Dataset validator catches missing columns, invalid event types, broken customer/item references, duplicate IDs, invalid timestamps, and consent-purpose inconsistencies.
- Tests pass.
- No real-bank references or real personal data appear in tracked files.

## Definition of done for Phase 2 foundations

- Chronological train/validation/test split with half-open UTC intervals; no feature or history uses events at or after the cutoff it is evaluated against.
- Training examples and evaluation cases use only events whose purpose has explicit, unambiguous consent; missing, unknown, ambiguous, or denied consent excludes the event (fail closed).
- NBS and NBP are built and evaluated separately; each task uses only events of its own purpose.
- Precision@K, Recall@K, NDCG@K, MAP@K, catalog coverage, and diversity are implemented with documented definitions and verified against hand-calculated examples.
- Outputs are deterministic and labelled as synthetic-data experiments.
- Tests pass; documented commands run on Windows PowerShell and Linux/macOS.
