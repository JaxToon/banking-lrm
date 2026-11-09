# Baseline comparison (Phase 2.1)

> **SYNTHETIC-DATA EXPERIMENT: metrics reflect patterns deliberately embedded by the synthetic generator. They are not evidence of real customer behavior or of real-world recommendation effectiveness.**

- Fit: train split; selection: validation ndcg@3; final: test split, selected configurations only.
- Uncertainty: paired percentile bootstrap over cases, n_boot=1000, confidence=0.95. Brackets are 95% CIs unless stated otherwise.
- Results cover only customers with consent-permitted, sufficient same-task history before the cutoff (min_history=1). 19 test customer-task pairs with targets were excluded as cold start. Cold start is not evaluated.

## NBS (test cases: 359, candidates: 5)

### K = 1

| Model | Selected config | P@1 | R@1 | NDCG@1 [CI] | MAP@1 | Coverage@1 | Exposure diversity@1 | Diff NDCG@1 vs popularity [CI] |
|---|---|---|---|---|---|---|---|---|
| random floor (5 seeds, mean+-sd) | - | 0.440+-0.026 | 0.198+-0.012 | 0.440+-0.026 | 0.440+-0.026 | 1.000 | 0.996 | - |
| popularity | signal=target | 0.513 | 0.233 | 0.513 [0.457, 0.563] | 0.513 | 0.200 | 0.000 | (reference) |
| itemknn | signal=target, include_self=True | 0.710 | 0.366 | 0.710 [0.657, 0.755] | 0.710 | 1.000 | 0.935 | +0.198 [+0.142, +0.253] |
| mf | signal=item_event, factors=3, reg=0.1, alpha=20.0 | 0.613 | 0.304 | 0.613 [0.563, 0.666] | 0.613 | 0.800 | 0.717 | +0.100 [+0.045, +0.159] |

### K = 3

| Model | Selected config | P@3 | R@3 | NDCG@3 [CI] | MAP@3 | Coverage@3 | Exposure diversity@3 | Diff NDCG@3 vs popularity [CI] |
|---|---|---|---|---|---|---|---|---|
| random floor (5 seeds, mean+-sd) | - | 0.428+-0.007 | 0.587+-0.018 | 0.538+-0.011 | 0.451+-0.009 | 1.000 | 1.000 | - |
| popularity | signal=target | 0.475 | 0.664 | 0.609 [0.577, 0.643] | 0.527 | 0.600 | 0.683 | (reference) |
| itemknn | signal=target, include_self=True | 0.560 | 0.786 | 0.765 [0.734, 0.794] | 0.706 | 1.000 | 0.961 | +0.156 [+0.123, +0.189] |
| mf | signal=item_event, factors=3, reg=0.1, alpha=20.0 | 0.527 | 0.745 | 0.701 [0.671, 0.731] | 0.627 | 1.000 | 0.985 | +0.092 [+0.056, +0.130] |

### Pairwise paired differences (ndcg@3, test)

| Comparison | Mean diff [CI] | Share of resamples not better |
|---|---|---|
| popularity - itemknn | -0.156 [-0.189, -0.123] | 1.000 |
| popularity - mf | -0.092 [-0.130, -0.056] | 1.000 |
| itemknn - mf | +0.064 [+0.037, +0.091] | 0.000 |

## NBP (test cases: 158, candidates: 6)

### K = 1

| Model | Selected config | P@1 | R@1 | NDCG@1 [CI] | MAP@1 | Coverage@1 | Exposure diversity@1 | Diff NDCG@1 vs popularity [CI] |
|---|---|---|---|---|---|---|---|---|
| random floor (5 seeds, mean+-sd) | - | 0.235+-0.027 | 0.168+-0.022 | 0.235+-0.027 | 0.235+-0.027 | 1.000 | 0.990 | - |
| popularity | signal=target | 0.323 | 0.230 | 0.323 [0.247, 0.392] | 0.323 | 0.167 | 0.000 | (reference) |
| itemknn | signal=item_event, include_self=True | 0.614 | 0.479 | 0.614 [0.538, 0.684] | 0.614 | 1.000 | 0.980 | +0.291 [+0.203, +0.380] |
| mf | signal=target, factors=3, reg=0.1, alpha=20.0 | 0.411 | 0.312 | 0.411 [0.335, 0.487] | 0.411 | 1.000 | 0.975 | +0.089 [+0.000, +0.177] |

### K = 3

| Model | Selected config | P@3 | R@3 | NDCG@3 [CI] | MAP@3 | Coverage@3 | Exposure diversity@3 | Diff NDCG@3 vs popularity [CI] |
|---|---|---|---|---|---|---|---|---|
| random floor (5 seeds, mean+-sd) | - | 0.236+-0.012 | 0.501+-0.026 | 0.388+-0.020 | 0.321+-0.022 | 1.000 | 0.998 | - |
| popularity | signal=target | 0.266 | 0.572 | 0.460 [0.398, 0.513] | 0.396 | 0.500 | 0.613 | (reference) |
| itemknn | signal=item_event, include_self=True | 0.376 | 0.803 | 0.723 [0.671, 0.769] | 0.671 | 1.000 | 0.973 | +0.264 [+0.197, +0.331] |
| mf | signal=target, factors=3, reg=0.1, alpha=20.0 | 0.314 | 0.679 | 0.565 [0.507, 0.619] | 0.498 | 1.000 | 0.964 | +0.105 [+0.032, +0.180] |

### Pairwise paired differences (ndcg@3, test)

| Comparison | Mean diff [CI] | Share of resamples not better |
|---|---|---|
| popularity - itemknn | -0.264 [-0.331, -0.197] | 1.000 |
| popularity - mf | -0.105 [-0.180, -0.032] | 0.997 |
| itemknn - mf | +0.158 [+0.099, +0.214] | 0.000 |

Notes: exposure diversity measures how evenly recommendations spread across the catalog, not semantic diversity within one customer's list. A difference whose CI includes 0 is not a reliable improvement on this synthetic data.
