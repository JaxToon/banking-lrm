# Data Dictionary and Event Taxonomy (schema v1.0)

All data is synthetic. The machine-readable source of truth is
[`src/bri/data/schema.py`](../src/bri/data/schema.py); the generator and validator both
import it, so this document and the code must be changed together.

Conventions:
- **Timestamps** are UTC ISO-8601 strings with an explicit `Z` suffix, e.g.
  `2024-12-20T03:15:42Z`. Timestamps without an explicit UTC designator are rejected.
  Local time (fictional UTC+7) is only used inside the generator's diurnal profile and
  for future presentation layers.
- **Identifiers** are artificial tokens with a `SYN_` (or `SVC_`/`PRD_` for catalog)
  prefix. Real-looking identifiers fail validation.
- **Booleans** are serialized as `True` / `False`. **Nulls** are empty CSV fields.
- **Column sets are an allowlist**: any column not listed here fails validation.

## 1. customers.csv

| Field | Type | Values / format |
|---|---|---|
| customer_token | id | `SYN_CUST_` + 6 digits |
| customer_segment | enum | `everyday`, `digital_active`, `emerging`, `established` |
| tenure_bucket | enum | `under_6m`, `6_to_24m`, `over_24m` (measured at window start) |
| engagement_segment | enum | `low`, `medium`, `high` (drives session frequency) |
| preferred_language | enum | `id`, `en` |
| created_at | timestamp | Synthetic profile creation; consistent with `tenure_bucket`; always before the event window |

No names, phone numbers, emails, addresses, identity numbers, account numbers, balances, birth dates, or locations.

## 2. catalog.csv

| Field | Type | Values / format |
|---|---|---|
| item_id | id | `SVC_*` (services) or `PRD_*` (products) |
| item_type | enum | `service`, `product` |
| category | text | `payments`, `savings`, `cards`, `financing` |
| item_name | text | Fictional generic display name |
| description | text | Generic authored description (not user text) |
| active_status | bool | All items active in schema v1.0 |

Items: 5 services (transfer, bill payment, QR payment, mobile top-up, scheduled payment)
and 6 product-information items (savings, time deposit, debit card, credit card, home
financing, personal financing). The catalog is not an offer, contractual term, or real
product list.

## 3. consent_preferences.csv

| Field | Type | Values / format |
|---|---|---|
| customer_token | id | FK → customers |
| purpose | enum | `service_personalization`, `marketing_personalization` |
| allowed | bool | Synthetic preference state |
| captured_at | timestamp | Between `created_at` and the window start |
| policy_version | id | `SYN_POLICY_` + digits |

Exactly **one row per customer per purpose** (a static snapshot; consent history and
withdrawal events are out of scope for v1.0). A synthetic `allowed=True` is not a
statement about valid consent or legal basis in any real environment.

## 4. events.csv

| Field | Type | Values / format |
|---|---|---|
| event_id | id | `SYN_EVT_` + 9 digits, assigned in stored order |
| customer_token | id | FK → customers |
| session_id | id | `SYN_SESS_<customer#>_<session#>`; belongs to exactly one customer |
| event_timestamp | timestamp | Inside `[window_start_utc, window_end_utc)` from metadata |
| event_type | enum | See vocabulary below |
| surface | enum | `navigation`, `service`, `product`; determined by `event_type` |
| item_id | id, nullable | Empty for navigation; FK → catalog otherwise |
| purpose | enum | Determined by `surface` (see mapping) |
| channel | enum | `mobile_app` |
| event_value | int | Interaction-intensity bucket `1`=brief, `2`=moderate, `3`=extended. **Not a monetary amount.** |

### Event vocabulary and mappings

| Surface | Event types | Purpose | item_id must reference |
|---|---|---|---|
| navigation | `app_open`, `home_view`, `menu_view`, `search_open`, `search_result_view` | `service_personalization` | (empty) |
| service | `service_view`, `service_start`, `service_complete`, `service_cancel` | `service_personalization` | `item_type = service` |
| product | `product_view`, `product_detail_view`, `product_calculator_use`, `product_application_start`, `product_application_complete` | `marketing_personalization` | `item_type = product` |

`product_application_complete` is a synthetic interaction signal only. It is never a
credit decision, approval, or suitability outcome.

### Session and funnel semantics

- A session is a burst of activity. It always begins with `app_open`, `home_view`.
  Consecutive sessions of a customer are at least 30 minutes apart.
- Each session contains 1–3 journeys:
  - service: [`menu_view`] → `service_view` → [`service_start` → `service_complete` | `service_cancel`]
  - product: [`search_open` → `search_result_view`] → `product_view` → [`product_detail_view` → [`product_calculator_use`] → [`product_application_start` → [`product_application_complete`]]]
- Funnel rule (validated): `service_complete` / `service_cancel` require an earlier
  `service_start`, and `product_application_complete` requires an earlier
  `product_application_start`, for the same session and item.
- Stored order: rows are contiguous per customer and non-decreasing in time.

### Events vs consent

Events are emitted regardless of the consent state. The `purpose` field records which
personalization purpose would use the event. **Downstream training/serving must join
`consent_preferences` and exclude events whose purpose is not allowed** (fail closed).
Keeping all events in the raw synthetic log lets later phases test that the policy
layer, not the data generator, enforces consent.

## 5. generation_metadata.json

`generator_version`, `schema_version`, `synthetic_only` (always `true`), `seed`,
`customers`, `days`, `window_start_utc`, `window_end_utc`, `row_counts` per file, and
`sha256` per CSV. It contains no wall-clock timestamp, so the whole output directory is
byte-identical for the same arguments on any OS.

Generator 0.3.0 (`--profile habit_dominant|balanced|transition_dominant`, Phase 2.3)
adds `behavior_profile`, `behavior_params` (the full `BehaviorProfile`) and
`behavior_note`. Without `--profile` the generator is 0.2.0 and its files, including
this metadata, are byte-identical to earlier releases.

### Behavior profiles (generator 0.3.0) — invented experimental assumptions

These parameters create contrasting synthetic regimes for robustness experiments. They
are **not** estimates of real banking customer behavior.

| Parameter | habit_dominant | balanced | transition_dominant | Meaning |
|---|---|---|---|---|
| service_concentration | 0.15 | 0.6 | 5.0 | Symmetric Dirichlet concentration of a customer's service affinity. Higher = weaker customer-specific preference (near-uniform) |
| product_concentration | 0.15 | 0.6 | 5.0 | Same for products (still multiplied by the segment tilt) |
| service_transition_p | 0.10 | 0.40 | 0.85 | P(next service intent in a session follows the service transition map) |
| product_transition_p | 0.05 | 0.30 | 0.75 | P(next product intent in a session follows the product transition map) |

Transition maps are task-specific and never cross tasks:
- NBS: TRANSFER→BILLPAY, BILLPAY→TOPUP, QR→TOPUP, TOPUP→QR, SCHEDULED→TRANSFER (the 0.2.0 map).
- NBP: SAVINGS→DEBIT→CREDIT→PERSONALFIN→HOMEFIN→DEPOSIT→SAVINGS (a fictional cycle).

Generator 0.2.0 equals the profile `legacy_v0_2_0` (0.6, 0.6, 0.4, 0.0). The session
process (sessions per day, 1-3 intents per session, journey funnels, consent) is the
same in every profile; for one seed, customer attributes and consent are identical
across profiles, and only item choice differs.

## 6. Validation rules (`python -m bri.data.validate_dataset`)

1. Required files present (four CSVs + metadata); CSVs parseable.
2. Exact column allowlist; prohibited column-name keyword scan (defence in depth).
3. Non-null, enum, boolean, integer-range, ID-format, and explicit-UTC timestamp checks.
4. Metadata: `synthetic_only=true`, row counts match files.
5. Unique keys: `customer_token`, `item_id`, `event_id`, (`customer_token`, `purpose`).
6. Foreign keys: events/consent → customers; events → catalog; session → single customer.
7. Event semantics: surface, purpose, and item type consistent with `event_type`.
8. Consent coverage: every customer has a preference for every purpose.
9. Temporal: events after `created_at` and inside the metadata window; consent after
   `created_at`; stored order per customer; funnel prerequisites.

Exit code `0` = valid, `1` = invalid (all issues listed), `2` = CLI usage error.

## 7. Phase 2 processed outputs (`python -m bri.prep.build`, git-ignored `data/processed/`)

All files are derived only from consent-permitted events. See
[08_PHASE_2_FOUNDATIONS_REPORT.md](08_PHASE_2_FOUNDATIONS_REPORT.md) for definitions.

### interactions.csv
| Field | Description |
|---|---|
| event_id, customer_token, session_id, event_timestamp, event_type, item_id | Copied from `events.csv` |
| task | `nbs` (purpose `service_personalization`) or `nbp` (purpose `marketing_personalization`) |
| split | `train`, `val`, `test` by event timestamp (half-open UTC intervals) |
| is_target | Event type is a positive for its task (before per-journey dedupe) |

Sorted by task, customer, timestamp, event_id.

### train_examples.csv
| Field | Description |
|---|---|
| example_id | `SYN_EX_` + 9 digits |
| task, customer_token | Task and customer |
| target_event_id, target_timestamp, target_item_id | First target per (task, customer, session, item) in train |
| history_length | Same-task train interactions strictly before the target |

### eval_cases.csv / eval_targets.csv
| Field | Description |
|---|---|
| case_id | `SYN_CASE_<split>_<task>_<customer_token>` |
| split | `val` or `test` |
| task, customer_token | Task and customer |
| cutoff_utc | Split start; history = same-task events strictly before it |
| history_length | Count of those events |
| n_relevant | Distinct relevant candidate items in the split window |

`eval_targets.csv` lists (`case_id`, `item_id`) for every relevant item.

### prep_metadata.json
Synthetic label, source dataset version/seed/checksum, parameters, cutoffs, per-task
purpose/targets/candidates, consent-filter report (kept and excluded counts by
reason), counts, and output SHA-256 hashes. No wall-clock timestamp.
Prep 0.2.0 adds `source_files_sha256` (hash of every source CSV as actually read, not
as claimed by its metadata) and `source_generation` (`customers`, `days`,
`behavior_profile`). Processed CSVs are byte-identical to prep 0.1.0.

### Recommendations input (for `python -m bri.eval.evaluate --recommendations`)
CSV with `case_id,item_id,rank`; ranks unique and contiguous from 1 per case; items
must be in the case's task candidate set.

## 8. Phase 2.1 experiment outputs (`python -m bri.experiments.baselines`, git-ignored `artifacts/`)

`baseline_report.json`: synthetic label, protocol (fit/selection/final split, bootstrap
settings, seed, K values), population note (cold-start exclusions), source checksums,
and per task: candidates, case counts, per-family validation trials and selected
parameters, test metrics, bootstrap CIs, paired differences vs popularity, pairwise
paired differences, and the multi-seed random floor. `baseline_report.md` renders the
same as ASCII tables.

Evaluation reports use the key `exposure_diversity` (renamed from `diversity` in
Phase 2.1) for normalized exposure entropy. It is not a semantic, per-list diversity
measure.

## 9. Phase 2.3 outputs

### Next-event cases (built in memory by `bri.eval.next_event`, never written as CSV)
| Field | Description |
|---|---|
| case_id | `SYN_NE_<split>_<task>_<customer_token>_<nnnnn>` |
| split, task, customer_token | Split window of the target, task, customer |
| cutoff_utc | Target timestamp; history = same-task interactions strictly before it |
| session_id | Session of the target |
| history_length | Earlier same-task first targets (ties excluded) |
| n_relevant | Distinct items targeted at this timestamp (1 unless timestamps tie) |
| within_session | Previous first target is in the same session |

They are derived deterministically from the fingerprinted `interactions.csv`, so the
processed-file fingerprints also pin them (with `next_event_version`).

### Run records and summaries (git-ignored `artifacts/`)
- `<report>.run_manifest.json` (written by `bri.experiments.baselines`): suite,
  evaluation, canonical config and its sha256, generator/prep/runner versions, source
  file hashes and composite fingerprint, processed fingerprints, prep fingerprint.
- `next_event_report.json`: same structure as `baseline_report.json` plus
  `evaluation`, `case_stats` (eligible targets, tie merges, cold-start exclusions,
  within/cross-session counts, single-target sessions) per split.
- `robustness_report.json` (`bri.experiments.robustness`): config and hash, versions,
  statistical design, per profile and evaluation: across-seed mean/SD/t-interval of
  every metric per model, selection counts, random floor, paired differences with
  counts of seeds whose within-seed CI excludes 0, per-seed metrics, cold-start
  totals, next-event session strata; and dataset fingerprints per seed.

### Benchmark manifests (`benchmarks/manifests/<name>.json`, tracked, immutable)
`suite`, `versions`, per evaluation the canonical config and its sha256, suite
definition hashes, frozen report file hashes, and per locked dataset: generator
identity (version, seed, customers, days, profile), source file hashes and composite,
processed file hashes, and prep fingerprint. The sha256 of each manifest file is pinned
in `bri.experiments.integrity.MANIFEST_SHA256`.

## 10. Phase 3 sequential cases (`bri.seq.data`, in memory; never written as CSV)

Cases are the Phase 2.3 next-event cases, also built for the train window. Per task
and split, `SeqCases` holds:

| Array | Shape / type | Meaning |
|---|---|---|
| items | (n_cases, max_len, max_tie) int64 | History steps, left-padded; 0 = padding, j + 1 = item j of the task vocabulary (sorted candidates). A step holds every first-target item sharing one timestamp (ties are never ordered) |
| new_session | (n_cases, max_len) int8 | 1 when a history step starts a session different from the previous history step; from the historical prefix only (the target session is never read) |
| length / full_length | (n_cases,) int64 | Steps kept after truncation / steps before truncation (most recent kept) |
| targets | (n_cases, n_items) float32 | Multi-hot relevant items (several only on timestamp ties) |

`SeqCases.fingerprint()` hashes the arrays, case ids, vocabulary and `max_len`.
`python -m bri.seq.prepare` generates the `phase3_tuning` datasets under
`data/phase3/phase3_tuning/` (git-ignored) and writes `seq_length_report.{json,md}`
from train-window cases (val as a sanity check; test events are dropped before building).

## 11. Phase 3 experiment outputs

Tracked, immutable (sha256 pinned in `bri.experiments.integrity.MANIFEST_SHA256`):
- `benchmarks/manifests/phase3-seeds.json`: `seed_namespaces` (`phase3_tuning`,
  `phase3_final`) with seeds, profiles, size and split, plus `checked_against` (the
  manifests and seeds in use when the reservation was made).
- `benchmarks/manifests/phase3-config.json`: `frozen_config` (model, training,
  device, precision) and `frozen_config_sha256`; `selection` (rule, trial, objective,
  group scores); `study` (name, trial counts, protocol, sha256 of the SQLite file at
  freeze); `tuning_data` (per-dataset source/processed/prep and sequence
  fingerprints); `runtime` (runtime fingerprint and environment); `versions`.
- `benchmarks/manifests/phase3.json`: per final dataset, generator identity, source
  and processed fingerprints, prep fingerprint and `seed_namespace`; `configs` for the
  `phase3` / `phase3-next-event` classical suites and the `phase3-sasrec`
  configuration; `phase3_run_config` and its hash; `runtime`; frozen report hashes.
- `benchmarks/phase3/phase3_report.{json,md}`: aggregate results. Per profile,
  protocol and task: across-seed statistics of every metric for every model, paired
  differences with within-seed CI sign counts, per-seed NDCG, next-event session
  strata with case counts, SASRec runtime, classical selection counts, and cold-start
  totals.

Git-ignored (`artifacts/phase3/`, `data/phase3/`):
- `hpo/study.db` (Optuna SQLite study) and `hpo/hpo_report.{json,md}` (all trials with
  per-group and per-dataset scores).
- `final/runs/<profile>/seed<N>/`: classical window and next-event reports and
  `phase3_dataset_record.json` (per-seed metrics, comparisons, strata, runtime).
- `final/checkpoints/<profile>/seed<N>/<task>_<variant>.pt`: SASRec checkpoints. Each
  holds `state_dict` and metadata: variant, architecture, hyperparameters, unused
  hyperparameters, item-mapping sha256, training seed, train/val sequence
  fingerprints, best epoch, torch version, device, state_dict sha256 (verified on
  load).
- `bench/`, `seq_lengths/`, `ablation_smoke.json`: Checkpoint 2–4 diagnostics.
