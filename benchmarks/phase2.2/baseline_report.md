# Baseline comparison (suite phase2.2)

> **SYNTHETIC-DATA EXPERIMENT: metrics reflect patterns deliberately embedded by the synthetic generator. They are not evidence of real customer behavior or of real-world recommendation effectiveness.**

- Dataset: generator 0.2.0, seed 20261109.
- Fit: train split; selection: validation ndcg@3; final: test split, selected configurations only.
- Uncertainty: paired percentile bootstrap over cases, n_boot=1000, confidence=0.95. Brackets are 95% CIs. CIs reflect case-to-case variation within this one dataset seed only.
- Results cover only customers with consent-permitted, sufficient same-task history before the cutoff (min_history=1). 12 test customer-task pairs with targets were excluded as cold start. Cold start is not evaluated.

## NBS (test cases: 351, candidates: 5)

### K = 1

| Model | Selected config | P@1 | R@1 | NDCG@1 [CI] | MAP@1 | Coverage@1 | Exposure diversity@1 | Diff NDCG@1 vs popularity [CI] |
|---|---|---|---|---|---|---|---|---|
| random floor (5 seeds, mean+-sd) | - | 0.455+-0.028 | 0.206+-0.022 | 0.455+-0.028 | 0.455+-0.028 | 1.000 | 0.996 | - |
| popularity | signal=item_event | 0.499 | 0.223 | 0.499 [0.447, 0.547] | 0.499 | 0.200 | 0.000 | (reference) |
| personal_frequency | signal=item_event | 0.766 | 0.409 | 0.766 [0.718, 0.806] | 0.766 | 1.000 | 0.983 | +0.268 [+0.208, +0.328] |
| itemknn_self | signal=item_event, include_self=True | 0.707 | 0.362 | 0.707 [0.658, 0.755] | 0.707 | 1.000 | 0.847 | +0.208 [+0.154, +0.259] |
| itemknn_noself | signal=target, include_self=False | 0.259 | 0.104 | 0.259 [0.214, 0.305] | 0.259 | 1.000 | 0.929 | -0.239 [-0.293, -0.182] |
| mf | signal=item_event, factors=3, reg=0.1, alpha=5.0 | 0.595 | 0.283 | 0.595 [0.544, 0.641] | 0.595 | 0.800 | 0.776 | +0.097 [+0.043, +0.151] |

### K = 3

| Model | Selected config | P@3 | R@3 | NDCG@3 [CI] | MAP@3 | Coverage@3 | Exposure diversity@3 | Diff NDCG@3 vs popularity [CI] |
|---|---|---|---|---|---|---|---|---|
| random floor (5 seeds, mean+-sd) | - | 0.443+-0.007 | 0.593+-0.014 | 0.555+-0.017 | 0.472+-0.020 | 1.000 | 1.000 | - |
| popularity | signal=item_event | 0.472 | 0.628 | 0.596 [0.563, 0.627] | 0.513 | 0.600 | 0.683 | (reference) |
| personal_frequency | signal=item_event | 0.569 | 0.804 | 0.802 [0.774, 0.829] | 0.748 | 1.000 | 0.986 | +0.207 [+0.168, +0.245] |
| itemknn_self | signal=item_event, include_self=True | 0.566 | 0.794 | 0.774 [0.744, 0.803] | 0.715 | 1.000 | 0.953 | +0.178 [+0.143, +0.215] |
| itemknn_noself | signal=target, include_self=False | 0.320 | 0.373 | 0.355 [0.320, 0.391] | 0.283 | 1.000 | 0.995 | -0.241 [-0.275, -0.202] |
| mf | signal=item_event, factors=3, reg=0.1, alpha=5.0 | 0.544 | 0.757 | 0.706 [0.675, 0.737] | 0.634 | 1.000 | 0.961 | +0.111 [+0.072, +0.147] |

### Key comparisons (paired bootstrap, test)

| Comparison | Question | NDCG@1 diff [CI] | NDCG@3 diff [CI] | MAP@1 diff [CI] | MAP@3 diff [CI] |
|---|---|---|---|---|---|
| itemknn_self - personal_frequency | Does co-usage add value on top of repeat usage? | -0.060 [-0.097, -0.020] | -0.028 [-0.047, -0.010] | -0.060 [-0.097, -0.020] | -0.033 [-0.055, -0.013] |
| itemknn_noself - personal_frequency | Is the collaborative signal alone enough? | -0.507 [-0.575, -0.436] | -0.447 [-0.497, -0.393] | -0.507 [-0.575, -0.436] | -0.465 [-0.518, -0.410] |
| itemknn_self - itemknn_noself | How much does self-similarity contribute? | +0.447 [+0.379, +0.510] | +0.419 [+0.369, +0.468] | +0.447 [+0.379, +0.510] | +0.432 [+0.379, +0.483] |

### All pairwise paired differences (ndcg, test)

| Comparison | @1 diff [CI] | @3 diff [CI] |
|---|---|---|
| popularity - personal_frequency | -0.268 [-0.328, -0.208] | -0.207 [-0.245, -0.168] |
| popularity - itemknn_self | -0.208 [-0.259, -0.154] | -0.178 [-0.215, -0.143] |
| popularity - itemknn_noself | +0.239 [+0.182, +0.293] | +0.241 [+0.202, +0.275] |
| popularity - mf | -0.097 [-0.151, -0.043] | -0.111 [-0.147, -0.072] |
| personal_frequency - itemknn_self | +0.060 [+0.020, +0.097] | +0.028 [+0.010, +0.047] |
| personal_frequency - itemknn_noself | +0.507 [+0.436, +0.575] | +0.447 [+0.393, +0.497] |
| personal_frequency - mf | +0.171 [+0.111, +0.225] | +0.096 [+0.067, +0.127] |
| itemknn_self - itemknn_noself | +0.447 [+0.379, +0.510] | +0.419 [+0.369, +0.468] |
| itemknn_self - mf | +0.111 [+0.060, +0.162] | +0.068 [+0.042, +0.096] |
| itemknn_noself - mf | -0.336 [-0.396, -0.274] | -0.351 [-0.399, -0.302] |

## NBP (test cases: 160, candidates: 6)

### K = 1

| Model | Selected config | P@1 | R@1 | NDCG@1 [CI] | MAP@1 | Coverage@1 | Exposure diversity@1 | Diff NDCG@1 vs popularity [CI] |
|---|---|---|---|---|---|---|---|---|
| random floor (5 seeds, mean+-sd) | - | 0.273+-0.033 | 0.182+-0.028 | 0.273+-0.033 | 0.273+-0.033 | 1.000 | 0.991 | - |
| popularity | signal=target | 0.294 | 0.222 | 0.294 [0.225, 0.362] | 0.294 | 0.167 | 0.000 | (reference) |
| personal_frequency | signal=item_event | 0.581 | 0.426 | 0.581 [0.500, 0.662] | 0.581 | 1.000 | 0.973 | +0.287 [+0.187, +0.394] |
| itemknn_self | signal=item_event, include_self=True | 0.575 | 0.429 | 0.575 [0.500, 0.656] | 0.575 | 1.000 | 0.971 | +0.281 [+0.181, +0.388] |
| itemknn_noself | signal=target, include_self=False | 0.181 | 0.124 | 0.181 [0.119, 0.244] | 0.181 | 1.000 | 0.893 | -0.113 [-0.194, -0.031] |
| mf | signal=item_event, factors=3, reg=1.0, alpha=5.0 | 0.412 | 0.299 | 0.412 [0.344, 0.487] | 0.412 | 1.000 | 0.984 | +0.119 [+0.013, +0.225] |

### K = 3

| Model | Selected config | P@3 | R@3 | NDCG@3 [CI] | MAP@3 | Coverage@3 | Exposure diversity@3 | Diff NDCG@3 vs popularity [CI] |
|---|---|---|---|---|---|---|---|---|
| random floor (5 seeds, mean+-sd) | - | 0.250+-0.019 | 0.512+-0.041 | 0.407+-0.038 | 0.341+-0.036 | 1.000 | 0.998 | - |
| popularity | signal=target | 0.285 | 0.598 | 0.474 [0.419, 0.531] | 0.403 | 0.500 | 0.613 | (reference) |
| personal_frequency | signal=item_event | 0.367 | 0.780 | 0.692 [0.640, 0.745] | 0.633 | 1.000 | 0.952 | +0.217 [+0.148, +0.287] |
| itemknn_self | signal=item_event, include_self=True | 0.375 | 0.786 | 0.696 [0.642, 0.750] | 0.639 | 1.000 | 0.978 | +0.222 [+0.143, +0.298] |
| itemknn_noself | signal=target, include_self=False | 0.165 | 0.338 | 0.266 [0.212, 0.319] | 0.218 | 1.000 | 0.989 | -0.209 [-0.285, -0.134] |
| mf | signal=item_event, factors=3, reg=1.0, alpha=5.0 | 0.338 | 0.706 | 0.584 [0.534, 0.640] | 0.512 | 1.000 | 0.969 | +0.110 [+0.031, +0.182] |

### Key comparisons (paired bootstrap, test)

| Comparison | Question | NDCG@1 diff [CI] | NDCG@3 diff [CI] | MAP@1 diff [CI] | MAP@3 diff [CI] |
|---|---|---|---|---|---|
| itemknn_self - personal_frequency | Does co-usage add value on top of repeat usage? | -0.006 [-0.037, +0.019] | +0.004 [-0.026, +0.035] | -0.006 [-0.037, +0.019] | +0.006 [-0.020, +0.032] |
| itemknn_noself - personal_frequency | Is the collaborative signal alone enough? | -0.400 [-0.506, -0.294] | -0.426 [-0.516, -0.339] | -0.400 [-0.506, -0.294] | -0.415 [-0.506, -0.329] |
| itemknn_self - itemknn_noself | How much does self-similarity contribute? | +0.394 [+0.287, +0.500] | +0.430 [+0.343, +0.521] | +0.394 [+0.287, +0.500] | +0.422 [+0.334, +0.511] |

### All pairwise paired differences (ndcg, test)

| Comparison | @1 diff [CI] | @3 diff [CI] |
|---|---|---|
| popularity - personal_frequency | -0.287 [-0.394, -0.187] | -0.217 [-0.287, -0.148] |
| popularity - itemknn_self | -0.281 [-0.388, -0.181] | -0.222 [-0.298, -0.143] |
| popularity - itemknn_noself | +0.113 [+0.031, +0.194] | +0.209 [+0.134, +0.285] |
| popularity - mf | -0.119 [-0.225, -0.013] | -0.110 [-0.182, -0.031] |
| personal_frequency - itemknn_self | +0.006 [-0.019, +0.037] | -0.004 [-0.035, +0.026] |
| personal_frequency - itemknn_noself | +0.400 [+0.294, +0.506] | +0.426 [+0.339, +0.516] |
| personal_frequency - mf | +0.169 [+0.081, +0.256] | +0.107 [+0.056, +0.156] |
| itemknn_self - itemknn_noself | +0.394 [+0.287, +0.500] | +0.430 [+0.343, +0.521] |
| itemknn_self - mf | +0.163 [+0.075, +0.250] | +0.112 [+0.069, +0.153] |
| itemknn_noself - mf | -0.231 [-0.338, -0.131] | -0.319 [-0.404, -0.237] |

Notes: exposure diversity measures how evenly recommendations spread across the catalog, not semantic diversity within one customer's list. A difference whose CI includes 0 is not a reliable difference on this synthetic data.
