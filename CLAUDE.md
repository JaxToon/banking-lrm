# Instructions for Claude Code

## Project context

You are helping develop **Banking Recommendation Intelligence**, an independent public portfolio project for a fictional Indonesian mobile banking environment. The real-world context is a commercial bank, but **never identify, name, imply, or include any real bank or employer in code, documentation, prompts, examples, commit messages, screenshots, test fixtures, or generated content**.

This repository must remain safe for a public GitHub portfolio. Treat all data as synthetic. Do not request, accept, create examples from, or commit real customer data, internal documents, production schemas, credentials, API keys, internal URLs, screenshots, or proprietary business rules.

## Product scope

- Behavioral scope: both mobile-app navigation behavior and service-use behavior.
- Recommendation scope: both Next Best Service (NBS) and Next Best Product (NBP).
- Local development target: laptop with 16 GB RAM.
- Completed: Phase 0 and Phase 1 (project foundation, synthetic dataset generator, schema validation); Phase 2 foundations (temporal split, consent-aware examples, shared evaluation framework); Phase 2.1 classical baselines (popularity, item-item CF, implicit MF, paired bootstrap), frozen as `benchmark-v1`; Phase 2.2 Personal Frequency & ItemKNN ablation, frozen as `phase2.2`.

### Current delivery: Phase 3 — Sequential Recommendation & SASRec Ablation (complete, pending review)

Approved: torch==2.11.0 (cu128 for GPU, PyPI CPU build otherwise) and optuna==4.9.0 under the optional `seq` extra; SASRec Full + NoPosition, Shuffled, MeanPool, SessionBoundary variants (SessionBoundary reuses Full hyperparameters, uses only historical session boundaries); Optuna TPE on `phase3_tuning` seeds (20261120-20261122) only; final evaluation on `phase3_final` seeds (20261130-20261139) after the configuration is frozen; next-event results split by within-/cross-session targets. Work proceeds checkpoint by checkpoint with review stops; do not commit unless asked. Checkpoint 1 done: dependencies, `bri/seq/runtime.py`, pinned seed reservation `benchmarks/manifests/phase3-seeds.json`, seed-collision preflight. Checkpoint 2 done: `bri/seq/data.py` (sequence cases, train split via the next-event builder), `bri/seq/prepare.py` (tuning datasets + train-only length report); max_seq_len search space approved as {10, 20, 30}. Checkpoint 3 done: `bri/seq/sasrec.py` (SASRec Full), `bri/seq/train.py` (training, early stopping on val via the shared evaluator, checkpoints), `bri/seq/bench.py` (CPU vs CUDA benchmark on one tuning dataset); runtime decision: all Phase 3 SASRec experiments on CUDA `cuda:0` only (FP32, deterministic, TF32 off), CPU only as technical fallback; runs on another device are refused (`require_device`) and the runtime fingerprint is part of the experiment fingerprint. Checkpoint 4 done: variants no_position, shuffled, meanpool, session_boundary (shared construction/initialization; boundary flags from the represented historical prefix only, earliest step 0); methodology approved (results measure component contributions under one controlled configuration, not each architecture's independently tuned best). Checkpoint 5 done: tuning data regenerated with seq data 0.2.0; CUDA tests marked `cuda` (excluded by default, mandatory `pytest -m cuda` run before final experiments); `bri/seq/hpo.py`; 30-trial TPE study (all complete); configuration frozen in pinned `benchmarks/manifests/phase3-config.json` (trial 21) - never change it after final results; Checkpoint 6 done: `bri/experiments/phase3.py`; final experiment on 30 `phase3_final` datasets locked in `benchmarks/manifests/phase3.json` with frozen report `benchmarks/phase3/` - never regenerate or edit; Checkpoint 7: report `docs/12_PHASE_3_SEQUENTIAL_REPORT.md`, README/architecture/data dictionary updated, reproduction and safety audit; pending owner review. Phase 3.1 (hybrid PF + SASRec) is future work only: gate on pre-recommendation features, never the evaluator's within/cross-session label; fresh seeds; Phase 3 final splits are a locked baseline, not a validation set. Not authorized: LRM/HSTU, LightGBM, LLM/RAG, API, dashboard, deployment.

### Next: Phase 3.1 — Hybrid Recommendation (pre-registered, NOT yet authorized to run)

Design: `docs/13_PHASE_3_1_HYBRID_DESIGN.md` (revision 4). Pre-registration pinned in `benchmarks/manifests/phase3_1-prereg.json` (decisions 1-15, features, grids, gate rules, criteria, reporting rules; design-doc sha256). Never edit it; any change needs a new pre-registration. Seeds reserved at Checkpoint 1 in pinned `benchmarks/manifests/phase3_1-seeds.json` (owner-approved exact values: `phase3_1_tuning` 20261140-20261142, `phase3_1_final` 20261150-20261159; initialization seeds {1,2,3} stay separate). Not authorized until a later checkpoint approves it: dataset generation, model training, cross-fitting, gate training, experiments. Key rules: prefix-history features only (time-to-target features reproduce the evaluator's within/cross label exactly and are forbidden); oracle router diagnostic only; customer-level 3-fold OOF gate with soft labels; support >=50 valid / 30-49 exploratory / <30 rejected ("not evaluable under the preregistered protocol"); no fallback from gated to static after final evaluation.

GitHub: pushing completed, committed and verified changes is authorized; uncommitted changes are never pushed; safety rules unchanged.

### Completed delivery: Phase 2.3 — Multi-Seed Behavioral Robustness

Authorized:
- Benchmark integrity: composite source fingerprints, processed-file fingerprints, prep-metadata fingerprint, canonical experiment-config hash, version identifiers, immutable manifests (`benchmarks/manifests/`, sha256 pinned in `bri.experiments.integrity`), adversarial tampering tests.
- Generator 0.3.0 behavior profiles `habit_dominant`, `balanced`, `transition_dominant` (Dirichlet affinity concentration, service/product transition probabilities, task-specific transition maps). Generator 0.2.0 must stay byte-identical. Profile parameters are invented experimental assumptions.
- Multi-seed experiments: 10 seeds per profile (20261110-20261119), 500 customers, 30 days, split 16/7/7; Popularity, Personal Frequency (mandatory reference), ItemKNN self=True, ItemKNN self=False, implicit MF; suites `phase2.3` (window) and `phase2.3-next-event`.
- Next-event evaluation protocol (one case per eligible target event, history strictly before it, signal fixed to `target`, customer-cluster bootstrap), alongside the unchanged window protocol.
- Across-seed reporting (mean, SD, Student-t interval over seeds) next to within-seed paired bootstrap CIs; never pooled.

Not yet authorized:
- LightGBM ranking, PyTorch, SASRec / sequential models, Transformer or LRM, LLM integration and RAG, API, dashboard, production deployment.

Locked test data (enforced by manifests; never overwrite frozen reports or manifests):
- `benchmark-v1` (generator 0.2.0, seed 42): re-run only with suite `benchmark-v1` for reproducibility.
- `phase2.2` (generator 0.2.0, seed 20261109): re-run only with suite `phase2.2`.
- `phase2.3` (generator 0.3.0, 3 profiles x seeds 20261110-20261119): used once for the Phase 2.3 final evaluation; re-run only with suites `phase2.3` / `phase2.3-next-event` and the registered configuration.
- New model-selection work needs fresh seeds that are not in any manifest.

Protocol (unchanged): all models use the existing temporal split, consent-filtered interactions, task-specific candidate sets, and shared evaluation framework. Fit on train, select on validation, evaluate the selected configuration once on test. Every reported result is labelled as a synthetic-data experiment.

Standing requirements:
- Model comparisons use paired bootstrap CIs with the customer/case as resampling unit; the random floor is reported over several seeds.
- Within-seed CIs capture case-to-case variation within one dataset seed only. Across-seed intervals capture variation between synthetic datasets of one profile at a fixed size, not across profiles, sizes, generator versions, or real populations. Do not pool the two levels.
- `exposure_diversity` measures catalog exposure evenness, not per-customer semantic diversity.
- Results cover only customers with sufficient history; cold start is an explicit limitation.
- Never trade away NBS/NBP purpose separation to improve scores.
- Dependencies: SciPy approved (add only when code needs it). PyTorch 2.11.0 and Optuna 4.9.0 approved for Phase 3 (optional `seq` extra). LightGBM not approved.

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
