# Phase 2.1 Report: Classical Recommendation Baselines

> **Status: frozen as `benchmark-v1`** (git tag `benchmark-v1`, report in
> `benchmarks/benchmark-v1/`). The seed-42 test split is locked. Phase 2.2 later showed
> that item-item CF's advantage comes mostly from repeat usage: a Personal Frequency
> baseline matches or beats it. See
> [10_PHASE_2_2_PERSONAL_FREQUENCY_REPORT.md](10_PHASE_2_2_PERSONAL_FREQUENCY_REPORT.md).

> **SYNTHETIC-DATA EXPERIMENT.** Every number below comes from data produced by this
> repository's synthetic generator. Models recover patterns the generator deliberately
> embeds (per-customer item affinity, service transitions, segment tilts). The results
> are not evidence of real customer behavior or real-world recommendation
> effectiveness.

## 1. What was built

| Component | Code | Purpose |
|---|---|---|
| Popularity | `src/bri/models/baselines.py` (`Popularity`) | Non-personalized reference: distinct training customers per item |
| Item-item CF | `ItemKNN` | Cosine similarity on binary train usage; score = log1p(history) · similarity |
| Matrix factorization | `ImplicitMF` | Implicit-feedback ALS (Hu, Koren & Volinsky, 2008), numpy only |
| Shared helpers | `src/bri/models/common.py` | Train matrices, pre-cutoff history matrices, deterministic ranking |
| Bootstrap | `src/bri/eval/bootstrap.py` | Percentile CIs and paired differences, resampling cases |
| Experiment runner | `src/bri/experiments/baselines.py` | Train-fit → validation-select → test-evaluate, reports |

No new dependencies. With 5–6 candidates per task, dense numpy matrices are small even
at large customer counts, so SciPy was not needed.

## 2. Protocol

1. **Inputs:** Phase 2 processed data only: consent-filtered, purpose-bound task
   interactions; the temporal split; task candidate sets; the shared evaluation code.
   NBS and NBP are fitted, selected, and evaluated independently. No model reads
   events of the other purpose.
2. **Fit on train.** Model parameters (popularity counts, item similarities, item
   factors) come only from train-split interactions.
3. **Select on validation.** Each family searches a small fixed grid (popularity 2,
   item-item 4, MF 16 configurations) by validation NDCG@3. Ties keep the earlier grid
   entry.
4. **Evaluate once on test.** Only the selected configuration of each family is scored
   on test. A test verifies that corrupting test targets leaves every validation trial
   and selection unchanged.
5. **Inference-time history:** each case uses the customer's same-task interactions
   strictly before the case cutoff. For test cases that includes validation-period
   events, which precede the test cutoff. Model parameters are not refitted. MF folds
   in a customer factor from that history with item factors fixed.
6. **Uncertainty:** 95% percentile bootstrap with 1,000 resamples. The resampling unit
   is the evaluation case (one customer within one task and split). Comparisons use a
   paired bootstrap, applying the same resampled cases to both models. The random floor
   is reported as mean ± sd over 5 seeds.
7. **Determinism:** fixed seeds and grid order, and ranking ties are broken by train
   popularity and then item_id. Repeated runs produce identical JSON (tested).

Signals searched as a hyperparameter: `target` (first target event per session and
item, i.e. the evaluation relevance definition) or `item_event` (every item-bearing
event of the task).

## 3. Results (default dataset: 500 customers, 30 days, seed 42; val/test = 7/7 days)

Commands: see README "Phase 2.1". Test split; brackets are 95% bootstrap CIs.

### NBS (359 test cases, 5 candidates)

| Model | Selected config | NDCG@1 = P@1 | NDCG@3 [CI] | R@3 | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|---|
| random floor (5 seeds) | – | 0.440 ± 0.026 | 0.538 ± 0.011 | 0.587 | 0.451 | 1.000 | 1.000 |
| popularity | signal=target | 0.513 | 0.609 [0.577, 0.643] | 0.664 | 0.527 | 0.600 | 0.683 |
| item-item | signal=target, include_self | **0.710** | **0.765 [0.734, 0.794]** | 0.786 | 0.706 | 1.000 | 0.961 |
| MF | item_event, f=3, reg=0.1, α=20 | 0.613 | 0.701 [0.671, 0.731] | 0.745 | 0.627 | 1.000 | 0.985 |

Paired NDCG@3 differences: item-item − popularity **+0.156 [+0.123, +0.189]**;
MF − popularity **+0.092 [+0.056, +0.130]**; item-item − MF **+0.064 [+0.037, +0.091]**.

### NBP (158 test cases, 6 candidates)

| Model | Selected config | NDCG@1 = P@1 | NDCG@3 [CI] | R@3 | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|---|
| random floor (5 seeds) | – | 0.235 ± 0.027 | 0.388 ± 0.020 | 0.501 | 0.321 | 1.000 | 0.998 |
| popularity | signal=target | 0.323 | 0.460 [0.398, 0.513] | 0.572 | 0.396 | 0.500 | 0.613 |
| item-item | signal=item_event, include_self | **0.614** | **0.723 [0.671, 0.769]** | 0.803 | 0.671 | 1.000 | 0.973 |
| MF | target, f=3, reg=0.1, α=20 | 0.411 | 0.565 [0.507, 0.619] | 0.679 | 0.498 | 1.000 | 0.964 |

Paired NDCG@3 differences: item-item − popularity **+0.264 [+0.197, +0.331]**;
MF − popularity **+0.105 [+0.032, +0.180]**; item-item − MF **+0.158 [+0.099, +0.214]**.
At K = 1, MF − popularity on NBP is +0.089 [+0.000, +0.177]. The interval touches 0,
so that improvement is **not** reliable.

19 test customer-task pairs with targets were excluded as cold start (no prior
same-task history) and are not represented in these numbers.

## 4. Interpretation

- **Item-item CF is the strongest baseline on both tasks**, with paired CIs excluding
  0 against both popularity and MF. With `include_self`, it combines the customer's own
  repeat usage with co-usage. That matches how the generator works: each synthetic
  customer has a stable item affinity and repeats their preferred items.
- **MF beats popularity but not item-item.** With 5–6 items, low-rank structure adds
  little beyond direct item co-usage, and fold-in from short histories is noisy. MF
  selected `factors=3`, the edge of its grid, so a wider grid could change its result
  slightly. That would be a validation-only change.
- **Popularity is the floor that matters, not random.** It beats random on both tasks
  but gives everyone the same list: coverage 0.2 at K = 1 and exposure diversity 0.0.
- **Beware the small catalog.** The random floor is high (NBS P@1 ≈ 0.44), and K = 3
  already covers 50–60% of the catalog. Gaps between models are therefore compressed
  and should be read with their CIs.
- **Exposure diversity measures catalog exposure evenness, not semantic diversity.**
  High values mean recommendations are spread across items over the population. They
  say nothing about whether one customer's list is varied.

## 5. Limitations and requirements carried forward

1. **Synthetic-only evidence.** The ranking of methods reflects the generator's design.
   A different generator (e.g. no per-customer affinity) could reverse it.
2. **Cold start not evaluated.** Results cover customers with at least one prior
   same-task interaction. Cold start needs its own experiment.
3. **Consent is a snapshot** (see report 08). The filters are conservative but are not
   point-in-time consent evidence.
4. **One dataset, one seed.** CIs capture case-sampling variation only, not variation
   across generator seeds or dataset sizes. A multi-seed generator study is a natural
   next step.
5. **Missing simple baseline.** A "personal frequency" ranker (each customer's own
   most-used items) was not in the authorized list. It would show how much of
   item-item's gain is repeat usage versus co-usage, and is worth authorizing before
   any more complex model.
6. **Test reuse.** The test split has now been used once for these selected
   configurations. Further tuning must use validation only. If the protocol changes
   materially, a fresh dataset seed should serve as a new untouched test.
7. **Purpose separation is fixed.** No experiment may combine NBS and NBP purposes to
   improve scores.

## 6. Outputs

`python -m bri.experiments.baselines` writes `artifacts/baselines/baseline_report.json`
(all trials, selected configs, test metrics, CIs, paired and pairwise differences,
random floor, protocol, source checksums) and `baseline_report.md` (ASCII tables).
`artifacts/` is git-ignored; this report quotes the numbers instead.
