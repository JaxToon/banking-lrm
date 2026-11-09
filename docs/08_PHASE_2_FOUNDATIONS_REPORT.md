# Phase 2 Foundations Report: Splitting, Consent-Aware Examples, Evaluation

Scope: temporal splitting, consent-aware training-example generation, and a shared
offline evaluation framework for NBS and NBP. **No recommendation model is
implemented.** Every number produced by this pipeline is a synthetic-data result.

## 1. Pipeline

```text
data/synthetic/ (Phase 1, validated)
  └─ bri.prep.build
       1. validate source dataset (refuse to build if invalid)
       2. consent filter (fail closed)            → prep_metadata.consent_filter
       3. assign task by purpose, split by time   → interactions.csv
       4. train targets + strict history          → train_examples.csv
       5. val/test cases + relevant items         → eval_cases.csv, eval_targets.csv
data/processed/ (git-ignored)
  └─ bri.eval.evaluate  (recommendations CSV or sanity ranker) → JSON report
```

Code: `src/bri/tasks.py`, `src/bri/prep/{split,consent,examples,build}.py`,
`src/bri/eval/{metrics,evaluate}.py`.

## 2. Task definitions

| Task | Purpose (only events with this purpose are used) | Candidate set | Positive (target) events |
|---|---|---|---|
| NBS | `service_personalization` (navigation + service events) | active `service` items | `service_start` |
| NBP | `marketing_personalization` (product events) | active `product` items | `product_detail_view`, `product_calculator_use`, `product_application_start`, `product_application_complete` |

- **Purpose limitation.** A task never reads events of the other purpose. NBP history
  therefore contains no navigation or service events, even when the customer allowed
  both purposes. Allowing cross-purpose features would need an explicit policy
  decision; it is not a modelling convenience.
- **Target dedupe.** Within the period being labelled, only the first target event per
  (task, customer, session, item) counts. A product journey of
  detail view → calculator → application yields one positive, not three.
- **Why `service_start`?** It marks the customer actually beginning the service.
  `service_view` is weaker intent, and complete/cancel are outcomes of a start. Product
  targets exclude the passive `product_view`.

## 3. Temporal splitting

- Global calendar cutoffs over the metadata window: test = last `--test-days`, val =
  the `--val-days` before that, train = the rest (defaults 7/7; a 30-day window gives
  16/7/7).
- Half-open UTC intervals `[start, end)`: an event exactly at a cutoff belongs to the
  later split.
- **Why global cutoffs, not per-user leave-last-out?** Global cutoffs mirror deployment,
  where a model trained up to time *T* serves after *T*. They also rule out leakage
  through other customers' future behavior (e.g. popularity computed with future data),
  which per-user splits permit.
- **Edge cases**
  - A session straddling a cutoff is split by event timestamp. Its post-cutoff part
    never enters training. A funnel may be cut, e.g. a `service_start` before the cutoff
    and its `service_complete` after it; targets use `service_start`, so this does not
    create orphan positives.
  - Out-of-window or timezone-naive timestamps raise errors instead of being silently
    assigned.
  - A window too short to leave at least one training day is rejected (`--val-days 7
    --test-days 7` on a 14-day dataset fails with exit code 2).
- **Leakage guarantees (tested)**
  - Train examples count only strictly earlier train-split events.
  - An eval case's history is every same-task event strictly before the split start.
    For test, that includes validation-period events, which precede the test cutoff
    and are legitimately available at serving time.
  - Deleting all events at or after a cutoff leaves every output for earlier periods
    byte-for-byte unchanged (`test_future_events_do_not_change_train_or_val_outputs`).

## 4. Consent-aware filtering

An event is kept only if its purpose is known, exactly one consent record exists for
(customer, purpose), `allowed` is literally `True`, the record has a valid
`captured_at`, and the event is at or after that time. Exclusions are counted by reason
in `prep_metadata.json`, in this precedence order: `unknown_purpose`,
`no_consent_record`, `ambiguous_consent`, `consent_not_granted`,
`event_before_consent_capture`.

> **The consent table is a snapshot, not historical consent evidence.** It holds one
> current preference per customer and purpose. It cannot show withdrawals, re-grants,
> or the state at the time of each event. Excluding events before `captured_at` is a
> conservative approximation. A real system would need a versioned consent history and
> point-in-time joins, and would also need to apply withdrawals to data already
> collected.

This filter is *data preparation*, not the policy engine. Serving-time policy
(eligibility, suppression, frequency caps, channel rules) remains a separate,
deterministic layer, per the architectural invariants.

## 5. Evaluation protocol

- **Unit:** one evaluation case per (split, task, customer) that has at least one
  relevant item in the split window and at least `--min-history` (default 1) same-task
  interactions before the cutoff. Excluded customers are counted (`insufficient_history`)
  rather than scored as zero.
- **Relevance:** binary. Relevant = the distinct target items in the window that are in
  the candidate set.
- **Candidate set:** every active catalog item of the task's type (5 services, 6
  products). Recommendations outside it, including items of the other task, unknown
  items, and inactive items, are rejected, not ignored.
- **Full ranking, no negative sampling at evaluation.** Sampled-negative metrics are
  biased and can reorder models (Krichene & Rendle, 2020). With a catalog this small,
  every non-relevant candidate is a negative.
- **Repeat consumption allowed:** items already used are not removed from candidates,
  since banking services are habitual.
- **Recommendation format:** `case_id,item_id,rank`. Ranks must be unique and
  contiguous from 1, with no duplicate items. A case with no list scores 0. Rows for
  unknown cases (e.g. a test-split list scored against val) are errors.
- **Aggregation:** macro average over cases, computed per task. NBS and NBP are never
  pooled. A task with no cases reports `null` (undefined), not 0.
- **Metrics:** definitions are in `src/bri/eval/metrics.py` and verified against
  hand-calculated examples.
  - Precision@K = hits/K, keeping K as the denominator for short lists.
  - Recall@K = hits/|R|.
  - NDCG@K uses binary gains with IDCG over min(|R|, K).
  - AP@K is normalized by min(|R|, K).
  - Coverage@K = distinct items in top-K lists / |candidates|.
  - Diversity@K = normalized Shannon entropy of top-K exposure (0 = identical lists,
    1 = perfectly even). Intra-list category diversity was rejected because all services
    share one category.
- **Choosing K:** defaults are 1 and 3. With 5 to 6 candidates, K ≥ |candidates| makes
  Recall and coverage trivial; the report adds a note when this happens.
- **Sanity rankers:** `random` (seeded floor) and `oracle` (reads the answers, ceiling).
  They check the pipeline and are not models. Always report a model next to the random
  floor: with a 5-item catalog, random NBS Precision@1 is about 0.46 on the default
  dataset.
- **Determinism:** fixed seeds, sorted case iteration, rounded metrics, and no
  wall-clock fields. `prep_metadata.json` records source checksums, parameters, and
  output SHA-256 hashes, and the build is byte-reproducible.

## 6. Negative sampling (training only)

`sample_negatives(examples, candidates, n_negatives, seed)` draws up to `n` items
uniformly without replacement from the task candidate set minus the target. Sampled
negatives are *unobserved*, not dislikes. With 5 services, a sampled negative is often
something the customer uses at other times, so false-negative risk is high. Later
phases should compare against full-softmax or all-candidate training, which is cheap
at this catalog size. The function is deterministic but not yet wired into the CLI;
models will choose their own strategy at review.

## 7. Results on the default dataset (500 customers, 30 days, seed 42)

Synthetic-data experiment; sanity rankers only, val split:

| Task | Cases | Ranker | P@1 | R@3 | NDCG@3 | MAP@3 |
|---|---|---|---|---|---|---|
| NBS | 354 | random | 0.458 | 0.570 | 0.544 | 0.458 |
| NBS | 354 | oracle | 1.000 | 0.962 | 1.000 | 1.000 |
| NBP | 151 | random | 0.272 | 0.509 | 0.409 | 0.342 |
| NBP | 151 | oracle | 1.000 | 0.997 | 1.000 | 1.000 |

The consent filter kept 40,876 of 43,741 events; all 2,865 exclusions were
`consent_not_granted`.

## 8. Limitations

- Consent is a static snapshot (see §4).
- The tiny catalog gives high random floors and leaves little room between floor and
  ceiling. Differences between future models need confidence intervals (e.g. bootstrap
  over cases), which are not implemented yet.
- Binary relevance ignores intensity (`event_value`) and funnel depth. Graded relevance
  is a possible extension.
- Window-level relevance (any target in the next 7 days) suits set-based models.
  Next-event sequential evaluation will need an additional protocol in a later phase.
- Customers with no history before the cutoff are excluded (cold start is not
  evaluated).
- All behavior is generated. Rankings learned here reflect the generator's embedded
  rules (affinities, transitions, segment tilts) and are not evidence of real-world
  effectiveness.
