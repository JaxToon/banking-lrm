# Phase 2.2 Report: Personal Frequency & ItemKNN Ablation

> **SYNTHETIC-DATA EXPERIMENT.** All numbers come from the repository's synthetic
> generator. Dataset seed 20261109 is an independent sample from the *same* generator
> as benchmark-v1. It is not evidence of real customer behavior or of real-world
> generalization. Confidence intervals capture case-to-case variation within this one
> dataset seed only, not variation across generator seeds, dataset sizes, or
> populations.

## 1. Question

Phase 2.1 found item-item CF (with `include_self=True`) to be the strongest baseline.
Its score mixes two signals: the customer's own repeat usage (self-similarity = 1) and
co-usage between items. This phase separates them:

> Does collaborative filtering add value beyond remembering each customer's habits?

## 2. Design

| Model | Definition | Measures |
|---|---|---|
| Personal Frequency | Rank items by the customer's own same-task, consent-permitted, pre-cutoff counts. No recency weighting, no other customers. | Repeat usage only |
| ItemKNN `self=False` | Cosine co-usage similarity with the diagonal zeroed | Collaborative signal only |
| ItemKNN `self=True` | Same, keeping sim(i, i) = 1 | Repeat usage + co-usage |
| Popularity, MF | As in Phase 2.1 | References |

- **Signals:** `target` and `item_event` (Phase 2.1 definitions), selected per model and
  task on validation. The `include_self` setting is *fixed* per ItemKNN variant, not
  selected.
- **Ties and fallback:** every model breaks ties by train popularity (target signal),
  then item_id. A customer with no item-bearing history therefore gets exactly the
  popularity ranking from Personal Frequency. All evaluated customers have at least one
  same-task interaction, but navigation-only NBS histories exist.
- **Dataset:** generator 0.2.0, seed **20261109**, 500 customers, 30 days, split 16/7/7.
  All five models were re-run on this dataset for a fair comparison.
- **Protocol:** fit on train, select on validation NDCG@3, evaluate once on test.
  Paired percentile bootstrap (1,000 resamples, customer/case unit). Random floor over
  5 seeds.
- **Test-split governance:** seed 42 (benchmark-v1) is locked, and the runner refuses
  any suite other than `benchmark-v1` on it. The seed-20261109 test split was used once
  for this report and is now locked to suite `phase2.2`. Frozen reports:
  `benchmarks/benchmark-v1/`, `benchmarks/phase2.2/`.

## 3. Results (test split, seed 20261109)

### NBS (351 test cases, 5 candidates; random floor NDCG@3 0.555 ± 0.017)

| Model | Selected signal | NDCG@1 = P@1 [CI] | NDCG@3 [CI] | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|
| popularity | item_event | 0.499 [0.447, 0.547] | 0.596 [0.563, 0.627] | 0.513 | 0.600 | 0.683 |
| **personal frequency** | item_event | **0.766 [0.718, 0.806]** | **0.802 [0.774, 0.829]** | 0.748 | 1.000 | 0.986 |
| ItemKNN self=True | item_event | 0.707 [0.658, 0.755] | 0.774 [0.744, 0.803] | 0.715 | 1.000 | 0.953 |
| ItemKNN self=False | target | 0.259 [0.214, 0.305] | 0.355 [0.320, 0.391] | 0.283 | 1.000 | 0.995 |
| MF | item_event (f=3, reg=0.1, α=5) | 0.595 [0.544, 0.641] | 0.706 [0.675, 0.737] | 0.634 | 1.000 | 0.961 |

### NBP (160 test cases, 6 candidates; random floor NDCG@3 0.407 ± 0.038)

| Model | Selected signal | NDCG@1 = P@1 [CI] | NDCG@3 [CI] | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|
| popularity | target | 0.294 [0.225, 0.362] | 0.474 [0.419, 0.531] | 0.403 | 0.500 | 0.613 |
| **personal frequency** | item_event | **0.581 [0.500, 0.662]** | **0.692 [0.640, 0.745]** | 0.633 | 1.000 | 0.952 |
| **ItemKNN self=True** | item_event | **0.575 [0.500, 0.656]** | **0.696 [0.642, 0.750]** | 0.639 | 1.000 | 0.978 |
| ItemKNN self=False | target | 0.181 [0.119, 0.244] | 0.266 [0.212, 0.319] | 0.218 | 1.000 | 0.989 |
| MF | item_event (f=3, reg=1.0, α=5) | 0.412 [0.344, 0.487] | 0.584 [0.534, 0.640] | 0.512 | 1.000 | 0.969 |

### Key comparisons (paired bootstrap, test)

| Comparison | Question | NBS NDCG@3 diff [CI] | NBP NDCG@3 diff [CI] |
|---|---|---|---|
| ItemKNN self=True − Personal Frequency | Does co-usage add value on top of repeat usage? | **−0.028 [−0.047, −0.010]** | +0.004 [−0.026, +0.035] |
| ItemKNN self=False − Personal Frequency | Is the collaborative signal alone enough? | **−0.447 [−0.497, −0.393]** | **−0.426 [−0.516, −0.339]** |
| ItemKNN self=True − ItemKNN self=False | How much does self-similarity contribute? | **+0.419 [+0.369, +0.468]** | **+0.430 [+0.343, +0.521]** |

At K = 1 the picture is the same. NBS self=True − Personal Frequency is −0.060
[−0.097, −0.020], and NBP is −0.006 [−0.037, +0.019], which includes 0. The JSON
report includes Precision, Recall, and MAP for every comparison.

## 4. Answers

1. **Does co-usage add value on top of repeat usage? No, not in this synthetic
   setting.** On NBS, adding co-usage to repeat usage makes rankings slightly but
   reliably *worse* than Personal Frequency. On NBP, the two are statistically
   indistinguishable.
2. **Is the collaborative signal alone enough? No.** ItemKNN without self-similarity
   falls below popularity, and even below the random floor (NBS NDCG@1 0.259 vs random
   0.455).
3. **How much does self-similarity contribute? Almost all of ItemKNN's value.**
   Removing it costs about 0.42–0.43 NDCG@3 on both tasks.

**Why `self=False` is worse than random (diagnostic on validation only):** 82.7% of
NBS and 68.2% of NBP relevant items in the validation window were already in the
customer's own history. Customers had used about 3.2 of 5 services and 2.5 of 6
products on average. Zeroing the diagonal removes an item's main source of score for
the customer who uses it, so the customer's habitual items drop below items they merely
co-occur with. When most relevant items are repeats, that inversion is worse than
random ordering. The diagnostic deliberately used the validation split so the locked
test split was not reused for analysis.

**Why this is unsurprising given the generator:** each synthetic customer draws a fixed
Dirichlet item affinity and repeats preferred items, while cross-item structure is
limited to a small service-transition map and segment tilts. A memory-based baseline
is close to optimal for that process. These results describe the generator, not real
customers.

## 5. Consistency with benchmark-v1

- benchmark-v1 (seed 42) selected `include_self=True` for ItemKNN on both tasks, which
  is consistent with the large self-similarity effect seen here.
- Model ordering among the Phase 2.1 models holds on the new seed: ItemKNN self=True >
  MF > popularity > random on both tasks, with paired CIs excluding 0.
- benchmark-v1 had no Personal Frequency, so "item-item CF is the strongest baseline"
  was true only among the models tested. Phase 2.2 shows its advantage came from
  repeat usage. **Personal Frequency is now the baseline any future model must beat.**
- benchmark-v1 numbers are not re-tuned or re-interpreted. They remain frozen in
  `benchmarks/benchmark-v1/`.

## 6. Limitations

1. **Two seeds, one generator.** Agreement between seed 42 and seed 20261109 is
   encouraging but is not a robustness study. Phase 2.3 should vary generator seeds and
   generator characteristics, e.g. weaker per-customer affinity, stronger cross-item
   transitions, or larger catalogs.
2. **Small catalogs (5–6 items).** These leave little room for co-usage to help and
   compress gaps. With larger catalogs, collaborative signals might matter more. That
   is untested.
3. **Window-level evaluation.** Relevance means "used in the next 7 days", which favors
   habitual items. Next-event (sequential) evaluation, planned for Phase 3, may favor
   transition-aware models differently.
4. **Cold start excluded** (12 test customer-task pairs here). Personal Frequency
   degrades to popularity without history, which is not evaluated.
5. **Consent snapshot and purpose separation** unchanged from Phase 2. NBS and NBP use
   only their own purpose's events.
6. **Selection by validation NDCG@3 only.** Other selection metrics could pick
   different signals, but none was tried on test.

## 7. Recommendation for next phases

- Treat **Personal Frequency as the reference baseline** (alongside popularity) in all
  later comparisons, including SASRec in Phase 3.
- Phase 2.3 (multi-seed robustness) should include generator variants where repeat
  usage is weaker. Otherwise every later model is evaluated in a regime where
  memorization is nearly optimal.
- Do not reuse locked test splits. New selection experiments need fresh dataset seeds.

## 8. Reproduction

```bash
python -m bri.data.generate_synthetic --customers 500 --days 30 --seed 20261109 --output-dir data/synthetic/seed20261109
python -m bri.data.validate_dataset --data-dir data/synthetic/seed20261109
python -m bri.prep.build --data-dir data/synthetic/seed20261109 --output-dir data/processed/seed20261109 --val-days 7 --test-days 7
python -m bri.experiments.baselines --suite phase2.2 --processed-dir data/processed/seed20261109 --output-dir artifacts/phase2_2_repro
```

The last command is a reproduction of the locked run, which the runner allows only
with `--suite phase2.2`. It must not be used to select new models.
