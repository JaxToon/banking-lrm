# Phase 2.3 multi-seed behavioral robustness

> **SYNTHETIC-DATA EXPERIMENT: metrics reflect patterns deliberately embedded by the synthetic generator. They are not evidence of real customer behavior or of real-world recommendation effectiveness.**

- Generator 0.3.0 profiles habit_dominant, balanced, transition_dominant; 10 seeds per profile (20261110..20261119); 500 customers, 30 days, split 16/7/7.
- Per dataset: fit on train, select on validation (ndcg@3), evaluate the selected configuration once on test.
- Across-seed cells: mean +- SD [Student-t 95% interval over seeds]. each seed's test mean or paired difference is one observation; mean, sample SD, Student-t 95% interval over seeds; describes variation across independent synthetic datasets of one profile at fixed size; within-seed CIs are not pooled.
- Within-seed CIs: paired percentile bootstrap; resampling unit = customer-task case (window) or customer (next-event, cluster bootstrap); describes variation inside one dataset. 'CI>0 / CI<0' counts seeds whose within-seed CI excludes 0 in that direction.
- Not covered: variation across profiles, population sizes, window lengths, generator versions, or any real population; profiles are not compared with intervals.
- Results cover only customers / target events with sufficient consent-permitted same-task history (min_history=1). Cold start is excluded and counted, not evaluated.
- Behavior parameters are invented experimental assumptions, not real banking behavior.

## Profile `habit_dominant`

Dirichlet concentration service/product = 0.15/0.15; transition probability service/product = 0.1/0.05.

### window evaluation

#### NBS (test cases per seed: 352-385)

| Model | NDCG@1 | NDCG@3 | P@3 | R@3 | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|---|
| random floor | 0.333 +- 0.012 [0.325, 0.342] | 0.491 +- 0.008 [0.485, 0.496] | - | - | 0.410 +- 0.008 [0.404, 0.416] | - | - |
| popularity | 0.362 +- 0.024 [0.345, 0.379] | 0.518 +- 0.016 [0.506, 0.530] | 0.347 | 0.630 | 0.437 | 0.600 | 0.683 |
| personal_frequency | 0.893 +- 0.015 [0.882, 0.904] | 0.912 +- 0.015 [0.902, 0.923] | 0.507 | 0.936 | 0.884 | 1.000 | 0.975 |
| itemknn_self | 0.893 +- 0.015 [0.882, 0.904] | 0.913 +- 0.012 [0.904, 0.921] | 0.507 | 0.937 | 0.884 | 1.000 | 0.990 |
| itemknn_noself | 0.114 +- 0.028 [0.094, 0.134] | 0.155 +- 0.027 [0.135, 0.174] | 0.129 | 0.179 | 0.111 | 1.000 | 0.995 |
| mf | 0.616 +- 0.034 [0.592, 0.640] | 0.783 +- 0.017 [0.771, 0.795] | 0.472 | 0.885 | 0.720 | 1.000 | 0.992 |

| Paired difference (within seed, then across seeds) | NDCG@1 | CI>0 / CI<0 | NDCG@3 | CI>0 / CI<0 |
|---|---|---|---|---|
| itemknn_self - personal_frequency | +0.000 +- 0.005 [-0.003, +0.004] | 0 / 0 | +0.000 +- 0.006 [-0.004, +0.004] | 0 / 2 |
| itemknn_noself - personal_frequency | -0.779 +- 0.039 [-0.807, -0.750] | 0 / 10 | -0.758 +- 0.041 [-0.787, -0.729] | 0 / 10 |
| mf - personal_frequency | -0.277 +- 0.040 [-0.305, -0.248] | 0 / 10 | -0.129 +- 0.014 [-0.139, -0.119] | 0 / 10 |
| popularity - personal_frequency | -0.531 +- 0.030 [-0.552, -0.509] | 0 / 10 | -0.394 +- 0.024 [-0.412, -0.377] | 0 / 10 |
| itemknn_self - itemknn_noself | +0.779 +- 0.039 [+0.751, +0.807] | 10 / 0 | +0.758 +- 0.037 [+0.731, +0.785] | 10 / 0 |

Per-seed test NDCG@1 / NDCG@3:

| Seed | popularity | personal_frequency | itemknn_self | itemknn_noself | mf |
|---|---|---|---|---|---|
| 20261110 | 0.336 / 0.505 | 0.888 / 0.925 | 0.896 / 0.924 | 0.079 / 0.120 | 0.639 / 0.802 |
| 20261111 | 0.344 / 0.521 | 0.917 / 0.931 | 0.919 / 0.925 | 0.092 / 0.117 | 0.558 / 0.774 |
| 20261112 | 0.327 / 0.496 | 0.897 / 0.915 | 0.902 / 0.919 | 0.103 / 0.157 | 0.628 / 0.795 |
| 20261113 | 0.398 / 0.543 | 0.872 / 0.889 | 0.875 / 0.894 | 0.165 / 0.196 | 0.616 / 0.781 |
| 20261114 | 0.396 / 0.537 | 0.919 / 0.923 | 0.916 / 0.924 | 0.101 / 0.139 | 0.632 / 0.787 |
| 20261115 | 0.368 / 0.526 | 0.879 / 0.899 | 0.876 / 0.899 | 0.112 / 0.164 | 0.618 / 0.764 |
| 20261116 | 0.377 / 0.527 | 0.883 / 0.912 | 0.888 / 0.919 | 0.153 / 0.177 | 0.668 / 0.801 |
| 20261117 | 0.348 / 0.498 | 0.893 / 0.915 | 0.888 / 0.917 | 0.101 / 0.149 | 0.616 / 0.783 |
| 20261118 | 0.365 / 0.506 | 0.886 / 0.891 | 0.883 / 0.894 | 0.139 / 0.187 | 0.559 / 0.751 |
| 20261119 | 0.362 / 0.521 | 0.894 / 0.922 | 0.889 / 0.910 | 0.095 / 0.141 | 0.627 / 0.796 |

Selected configurations (seeds): popularity: signal=item_event x1, signal=target x9; personal_frequency: signal=item_event x10; itemknn_self: signal=item_event, include_self=True x10; itemknn_noself: signal=target, include_self=False x10; mf: signal=item_event, factors=3, reg=0.1, alpha=5.0 x2, signal=item_event, factors=3, reg=1.0, alpha=20.0 x1, signal=item_event, factors=3, reg=1.0, alpha=5.0 x1, signal=target, factors=3, reg=0.1, alpha=20.0 x1, signal=target, factors=3, reg=0.1, alpha=5.0 x1, signal=target, factors=3, reg=1.0, alpha=5.0 x4

#### NBP (test cases per seed: 152-178)

| Model | NDCG@1 | NDCG@3 | P@3 | R@3 | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|---|
| random floor | 0.209 +- 0.014 [0.199, 0.219] | 0.375 +- 0.011 [0.367, 0.382] | - | - | 0.315 +- 0.010 [0.307, 0.322] | - | - |
| popularity | 0.245 +- 0.022 [0.230, 0.261] | 0.424 +- 0.020 [0.410, 0.438] | 0.230 | 0.557 | 0.363 | 0.500 | 0.613 |
| personal_frequency | 0.751 +- 0.025 [0.733, 0.768] | 0.837 +- 0.017 [0.825, 0.849] | 0.371 | 0.902 | 0.801 | 1.000 | 0.944 |
| itemknn_self | 0.755 +- 0.028 [0.735, 0.775] | 0.835 +- 0.015 [0.824, 0.846] | 0.370 | 0.898 | 0.800 | 1.000 | 0.978 |
| itemknn_noself | 0.087 +- 0.018 [0.074, 0.099] | 0.141 +- 0.027 [0.122, 0.161] | 0.083 | 0.181 | 0.115 | 1.000 | 0.969 |
| mf | 0.517 +- 0.058 [0.476, 0.559] | 0.710 +- 0.033 [0.687, 0.734] | 0.345 | 0.844 | 0.650 | 1.000 | 0.972 |

| Paired difference (within seed, then across seeds) | NDCG@1 | CI>0 / CI<0 | NDCG@3 | CI>0 / CI<0 |
|---|---|---|---|---|
| itemknn_self - personal_frequency | +0.004 +- 0.007 [-0.001, +0.009] | 0 / 0 | -0.002 +- 0.007 [-0.007, +0.003] | 0 / 0 |
| itemknn_noself - personal_frequency | -0.664 +- 0.032 [-0.687, -0.642] | 0 / 10 | -0.695 +- 0.040 [-0.724, -0.667] | 0 / 10 |
| mf - personal_frequency | -0.234 +- 0.056 [-0.274, -0.193] | 0 / 10 | -0.127 +- 0.022 [-0.142, -0.111] | 0 / 10 |
| popularity - personal_frequency | -0.506 +- 0.036 [-0.532, -0.480] | 0 / 10 | -0.413 +- 0.025 [-0.431, -0.395] | 0 / 10 |
| itemknn_self - itemknn_noself | +0.668 +- 0.035 [+0.643, +0.693] | 10 / 0 | +0.693 +- 0.039 [+0.666, +0.721] | 10 / 0 |

Per-seed test NDCG@1 / NDCG@3:

| Seed | popularity | personal_frequency | itemknn_self | itemknn_noself | mf |
|---|---|---|---|---|---|
| 20261110 | 0.211 / 0.394 | 0.776 / 0.848 | 0.789 / 0.850 | 0.066 / 0.117 | 0.618 / 0.740 |
| 20261111 | 0.247 / 0.438 | 0.715 / 0.839 | 0.722 / 0.833 | 0.070 / 0.131 | 0.576 / 0.698 |
| 20261112 | 0.230 / 0.422 | 0.730 / 0.824 | 0.725 / 0.817 | 0.079 / 0.168 | 0.511 / 0.666 |
| 20261113 | 0.244 / 0.433 | 0.750 / 0.828 | 0.762 / 0.823 | 0.089 / 0.140 | 0.524 / 0.692 |
| 20261114 | 0.265 / 0.419 | 0.753 / 0.837 | 0.753 / 0.840 | 0.093 / 0.147 | 0.506 / 0.730 |
| 20261115 | 0.231 / 0.435 | 0.801 / 0.856 | 0.808 / 0.852 | 0.083 / 0.112 | 0.532 / 0.724 |
| 20261116 | 0.273 / 0.425 | 0.745 / 0.817 | 0.739 / 0.827 | 0.121 / 0.191 | 0.424 / 0.688 |
| 20261117 | 0.263 / 0.450 | 0.737 / 0.821 | 0.743 / 0.813 | 0.111 / 0.167 | 0.439 / 0.666 |
| 20261118 | 0.269 / 0.438 | 0.762 / 0.871 | 0.769 / 0.857 | 0.081 / 0.135 | 0.544 / 0.762 |
| 20261119 | 0.221 / 0.386 | 0.738 / 0.828 | 0.738 / 0.836 | 0.076 / 0.107 | 0.500 / 0.735 |

Selected configurations (seeds): popularity: signal=item_event x1, signal=target x9; personal_frequency: signal=item_event x10; itemknn_self: signal=item_event, include_self=True x10; itemknn_noself: signal=target, include_self=False x10; mf: signal=item_event, factors=3, reg=0.1, alpha=20.0 x4, signal=item_event, factors=3, reg=0.1, alpha=5.0 x1, signal=item_event, factors=3, reg=1.0, alpha=20.0 x1, signal=item_event, factors=3, reg=1.0, alpha=5.0 x1, signal=target, factors=3, reg=0.1, alpha=5.0 x1, signal=target, factors=3, reg=1.0, alpha=20.0 x1, signal=target, factors=3, reg=1.0, alpha=5.0 x1

Cold start / exclusions (test, total over seeds): customers_with_targets = 5400; insufficient_history_excluded = 113

### next-event evaluation

#### NBS (test cases per seed: 1077-1247)

| Model | NDCG@1 | NDCG@3 | P@3 | R@3 | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|---|
| random floor | 0.202 +- 0.005 [0.198, 0.205] | 0.429 +- 0.006 [0.425, 0.433] | - | - | 0.369 +- 0.005 [0.365, 0.373] | - | - |
| popularity | 0.213 +- 0.011 [0.205, 0.221] | 0.447 +- 0.009 [0.441, 0.454] | 0.209 | 0.628 | 0.386 | 0.600 | 0.683 |
| personal_frequency | 0.650 +- 0.022 [0.634, 0.666] | 0.826 +- 0.016 [0.814, 0.837] | 0.315 | 0.945 | 0.784 | 1.000 | 0.977 |
| itemknn_self | 0.650 +- 0.021 [0.636, 0.665] | 0.826 +- 0.014 [0.815, 0.836] | 0.315 | 0.945 | 0.784 | 1.000 | 0.991 |
| itemknn_noself | 0.037 +- 0.008 [0.032, 0.043] | 0.100 +- 0.015 [0.089, 0.111] | 0.051 | 0.152 | 0.082 | 1.000 | 0.996 |
| mf | 0.408 +- 0.019 [0.394, 0.422] | 0.698 +- 0.015 [0.687, 0.709] | 0.299 | 0.897 | 0.629 | 1.000 | 0.992 |

| Paired difference (within seed, then across seeds) | NDCG@1 | CI>0 / CI<0 | NDCG@3 | CI>0 / CI<0 |
|---|---|---|---|---|
| itemknn_self - personal_frequency | +0.000 +- 0.004 [-0.003, +0.003] | 0 / 1 | -0.000 +- 0.003 [-0.002, +0.002] | 0 / 1 |
| itemknn_noself - personal_frequency | -0.613 +- 0.029 [-0.634, -0.592] | 0 / 10 | -0.726 +- 0.030 [-0.748, -0.704] | 0 / 10 |
| mf - personal_frequency | -0.242 +- 0.027 [-0.262, -0.222] | 0 / 10 | -0.127 +- 0.012 [-0.136, -0.119] | 0 / 10 |
| popularity - personal_frequency | -0.437 +- 0.026 [-0.456, -0.418] | 0 / 10 | -0.378 +- 0.019 [-0.392, -0.364] | 0 / 10 |
| itemknn_self - itemknn_noself | +0.613 +- 0.027 [+0.593, +0.633] | 10 / 0 | +0.726 +- 0.029 [+0.705, +0.746] | 10 / 0 |

Per-seed test NDCG@1 / NDCG@3:

| Seed | popularity | personal_frequency | itemknn_self | itemknn_noself | mf |
|---|---|---|---|---|---|
| 20261110 | 0.214 / 0.454 | 0.682 / 0.844 | 0.676 / 0.842 | 0.032 / 0.085 | 0.419 / 0.713 |
| 20261111 | 0.206 / 0.449 | 0.680 / 0.846 | 0.682 / 0.843 | 0.031 / 0.083 | 0.395 / 0.707 |
| 20261112 | 0.192 / 0.432 | 0.634 / 0.817 | 0.636 / 0.819 | 0.038 / 0.114 | 0.407 / 0.695 |
| 20261113 | 0.221 / 0.460 | 0.611 / 0.798 | 0.613 / 0.799 | 0.053 / 0.123 | 0.390 / 0.680 |
| 20261114 | 0.217 / 0.442 | 0.667 / 0.836 | 0.662 / 0.834 | 0.030 / 0.085 | 0.401 / 0.703 |
| 20261115 | 0.225 / 0.450 | 0.639 / 0.818 | 0.644 / 0.822 | 0.035 / 0.101 | 0.400 / 0.696 |
| 20261116 | 0.219 / 0.459 | 0.648 / 0.829 | 0.652 / 0.830 | 0.047 / 0.108 | 0.459 / 0.727 |
| 20261117 | 0.221 / 0.449 | 0.658 / 0.832 | 0.659 / 0.833 | 0.032 / 0.096 | 0.406 / 0.691 |
| 20261118 | 0.220 / 0.444 | 0.631 / 0.805 | 0.633 / 0.807 | 0.046 / 0.117 | 0.403 / 0.677 |
| 20261119 | 0.197 / 0.435 | 0.651 / 0.833 | 0.645 / 0.827 | 0.031 / 0.086 | 0.401 / 0.693 |

Selected configurations (seeds): popularity: signal=target x10; personal_frequency: signal=target x10; itemknn_self: signal=target, include_self=True x10; itemknn_noself: signal=target, include_self=False x10; mf: signal=target, factors=3, reg=0.1, alpha=20.0 x1, signal=target, factors=3, reg=0.1, alpha=5.0 x2, signal=target, factors=3, reg=1.0, alpha=20.0 x2, signal=target, factors=3, reg=1.0, alpha=5.0 x5

#### NBP (test cases per seed: 224-301)

| Model | NDCG@1 | NDCG@3 | P@3 | R@3 | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|---|
| random floor | 0.169 +- 0.013 [0.159, 0.178] | 0.357 +- 0.011 [0.349, 0.365] | - | - | 0.307 +- 0.011 [0.299, 0.315] | - | - |
| popularity | 0.207 +- 0.016 [0.195, 0.218] | 0.400 +- 0.021 [0.385, 0.415] | 0.184 | 0.552 | 0.349 | 0.500 | 0.613 |
| personal_frequency | 0.626 +- 0.033 [0.603, 0.650] | 0.781 +- 0.023 [0.765, 0.798] | 0.295 | 0.886 | 0.745 | 1.000 | 0.930 |
| itemknn_self | 0.627 +- 0.036 [0.601, 0.653] | 0.781 +- 0.023 [0.765, 0.797] | 0.295 | 0.885 | 0.745 | 1.000 | 0.971 |
| itemknn_noself | 0.043 +- 0.010 [0.036, 0.051] | 0.098 +- 0.016 [0.086, 0.109] | 0.047 | 0.140 | 0.083 | 1.000 | 0.967 |
| mf | 0.441 +- 0.069 [0.392, 0.490] | 0.678 +- 0.036 [0.653, 0.704] | 0.282 | 0.845 | 0.620 | 1.000 | 0.981 |

| Paired difference (within seed, then across seeds) | NDCG@1 | CI>0 / CI<0 | NDCG@3 | CI>0 / CI<0 |
|---|---|---|---|---|
| itemknn_self - personal_frequency | +0.001 +- 0.008 [-0.005, +0.007] | 0 / 0 | -0.000 +- 0.007 [-0.005, +0.005] | 0 / 0 |
| itemknn_noself - personal_frequency | -0.583 +- 0.035 [-0.608, -0.558] | 0 / 10 | -0.684 +- 0.036 [-0.709, -0.658] | 0 / 10 |
| mf - personal_frequency | -0.185 +- 0.067 [-0.233, -0.137] | 0 / 10 | -0.103 +- 0.025 [-0.121, -0.086] | 0 / 10 |
| popularity - personal_frequency | -0.420 +- 0.036 [-0.445, -0.394] | 0 / 10 | -0.381 +- 0.023 [-0.397, -0.365] | 0 / 10 |
| itemknn_self - itemknn_noself | +0.584 +- 0.039 [+0.556, +0.611] | 10 / 0 | +0.683 +- 0.035 [+0.658, +0.708] | 10 / 0 |

Per-seed test NDCG@1 / NDCG@3:

| Seed | popularity | personal_frequency | itemknn_self | itemknn_noself | mf |
|---|---|---|---|---|---|
| 20261110 | 0.188 / 0.376 | 0.643 / 0.786 | 0.652 / 0.799 | 0.054 / 0.106 | 0.518 / 0.712 |
| 20261111 | 0.197 / 0.401 | 0.598 / 0.761 | 0.602 / 0.761 | 0.042 / 0.097 | 0.477 / 0.676 |
| 20261112 | 0.234 / 0.408 | 0.605 / 0.772 | 0.605 / 0.768 | 0.041 / 0.094 | 0.502 / 0.678 |
| 20261113 | 0.184 / 0.375 | 0.585 / 0.752 | 0.582 / 0.754 | 0.057 / 0.106 | 0.284 / 0.601 |
| 20261114 | 0.203 / 0.382 | 0.601 / 0.776 | 0.605 / 0.784 | 0.040 / 0.097 | 0.465 / 0.683 |
| 20261115 | 0.209 / 0.443 | 0.698 / 0.822 | 0.711 / 0.825 | 0.047 / 0.083 | 0.472 / 0.708 |
| 20261116 | 0.228 / 0.409 | 0.612 / 0.759 | 0.612 / 0.756 | 0.049 / 0.131 | 0.448 / 0.671 |
| 20261117 | 0.206 / 0.418 | 0.645 / 0.778 | 0.637 / 0.773 | 0.046 / 0.103 | 0.393 / 0.639 |
| 20261118 | 0.216 / 0.404 | 0.636 / 0.805 | 0.644 / 0.794 | 0.020 / 0.079 | 0.460 / 0.716 |
| 20261119 | 0.203 / 0.389 | 0.638 / 0.803 | 0.624 / 0.795 | 0.041 / 0.080 | 0.391 / 0.698 |

Selected configurations (seeds): popularity: signal=target x10; personal_frequency: signal=target x10; itemknn_self: signal=target, include_self=True x10; itemknn_noself: signal=target, include_self=False x10; mf: signal=target, factors=3, reg=0.1, alpha=20.0 x2, signal=target, factors=3, reg=0.1, alpha=5.0 x4, signal=target, factors=3, reg=1.0, alpha=20.0 x1, signal=target, factors=3, reg=1.0, alpha=5.0 x3

Next-event NDCG@1 by session stratum (across-seed mean): within-session = previous target in the same session; cross-session = first target of a session.

| Task | Model | within-session | cross-session |
|---|---|---|---|
| nbs | popularity | 0.272 | 0.202 |
| nbs | personal_frequency | 0.282 | 0.715 |
| nbs | itemknn_self | 0.291 | 0.714 |
| nbs | itemknn_noself | 0.129 | 0.021 |
| nbs | mf | 0.256 | 0.435 |
| nbp | popularity | 0.142 | 0.210 |
| nbp | personal_frequency | 0.263 | 0.645 |
| nbp | itemknn_self | 0.263 | 0.646 |
| nbp | itemknn_noself | 0.069 | 0.042 |
| nbp | mf | 0.176 | 0.455 |

Cold start / exclusions (test, total over seeds): eligible_target_events = 14577; insufficient_history_excluded = 283; tied_timestamp_merges = 0; single_target_sessions = 10861

## Profile `balanced`

Dirichlet concentration service/product = 0.6/0.6; transition probability service/product = 0.4/0.3.

### window evaluation

#### NBS (test cases per seed: 345-380)

| Model | NDCG@1 | NDCG@3 | P@3 | R@3 | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|---|
| random floor | 0.458 +- 0.009 [0.451, 0.465] | 0.556 +- 0.010 [0.550, 0.563] | - | - | 0.471 +- 0.010 [0.463, 0.478] | - | - |
| popularity | 0.522 +- 0.032 [0.499, 0.545] | 0.600 +- 0.017 [0.588, 0.612] | 0.479 | 0.633 | 0.520 | 0.600 | 0.683 |
| personal_frequency | 0.756 +- 0.014 [0.746, 0.766] | 0.788 +- 0.010 [0.781, 0.795] | 0.579 | 0.795 | 0.729 | 1.000 | 0.987 |
| itemknn_self | 0.723 +- 0.024 [0.706, 0.740] | 0.766 +- 0.013 [0.757, 0.775] | 0.569 | 0.782 | 0.703 | 1.000 | 0.968 |
| itemknn_noself | 0.282 +- 0.015 [0.271, 0.293] | 0.369 +- 0.008 [0.363, 0.375] | 0.331 | 0.390 | 0.289 | 1.000 | 0.998 |
| mf | 0.595 +- 0.049 [0.560, 0.631] | 0.708 +- 0.020 [0.694, 0.723] | 0.546 | 0.752 | 0.632 | 1.000 | 0.974 |

| Paired difference (within seed, then across seeds) | NDCG@1 | CI>0 / CI<0 | NDCG@3 | CI>0 / CI<0 |
|---|---|---|---|---|
| itemknn_self - personal_frequency | -0.032 +- 0.027 [-0.052, -0.013] | 0 / 5 | -0.022 +- 0.013 [-0.031, -0.013] | 0 / 5 |
| itemknn_noself - personal_frequency | -0.474 +- 0.018 [-0.487, -0.461] | 0 / 10 | -0.419 +- 0.012 [-0.427, -0.411] | 0 / 10 |
| mf - personal_frequency | -0.161 +- 0.055 [-0.200, -0.121] | 0 / 10 | -0.080 +- 0.023 [-0.096, -0.064] | 0 / 10 |
| popularity - personal_frequency | -0.234 +- 0.036 [-0.259, -0.208] | 0 / 10 | -0.188 +- 0.018 [-0.201, -0.175] | 0 / 10 |
| itemknn_self - itemknn_noself | +0.441 +- 0.033 [+0.418, +0.465] | 10 / 0 | +0.397 +- 0.013 [+0.388, +0.407] | 10 / 0 |

Per-seed test NDCG@1 / NDCG@3:

| Seed | popularity | personal_frequency | itemknn_self | itemknn_noself | mf |
|---|---|---|---|---|---|
| 20261110 | 0.496 / 0.581 | 0.752 / 0.782 | 0.747 / 0.772 | 0.264 / 0.371 | 0.587 / 0.714 |
| 20261111 | 0.561 / 0.628 | 0.749 / 0.798 | 0.729 / 0.766 | 0.298 / 0.366 | 0.622 / 0.723 |
| 20261112 | 0.481 / 0.582 | 0.733 / 0.780 | 0.711 / 0.767 | 0.269 / 0.355 | 0.622 / 0.735 |
| 20261113 | 0.572 / 0.624 | 0.743 / 0.782 | 0.718 / 0.773 | 0.304 / 0.382 | 0.586 / 0.706 |
| 20261114 | 0.536 / 0.604 | 0.756 / 0.795 | 0.761 / 0.789 | 0.267 / 0.366 | 0.603 / 0.713 |
| 20261115 | 0.525 / 0.586 | 0.762 / 0.791 | 0.675 / 0.749 | 0.287 / 0.371 | 0.484 / 0.660 |
| 20261116 | 0.504 / 0.589 | 0.752 / 0.777 | 0.705 / 0.746 | 0.284 / 0.363 | 0.645 / 0.705 |
| 20261117 | 0.478 / 0.595 | 0.775 / 0.799 | 0.739 / 0.765 | 0.286 / 0.365 | 0.541 / 0.697 |
| 20261118 | 0.534 / 0.612 | 0.755 / 0.776 | 0.729 / 0.761 | 0.263 / 0.371 | 0.637 / 0.705 |
| 20261119 | 0.533 / 0.600 | 0.780 / 0.800 | 0.719 / 0.775 | 0.299 / 0.381 | 0.626 / 0.723 |

Selected configurations (seeds): popularity: signal=item_event x4, signal=target x6; personal_frequency: signal=item_event x9, signal=target x1; itemknn_self: signal=item_event, include_self=True x5, signal=target, include_self=True x5; itemknn_noself: signal=target, include_self=False x10; mf: signal=item_event, factors=3, reg=0.1, alpha=20.0 x2, signal=item_event, factors=3, reg=0.1, alpha=5.0 x5, signal=item_event, factors=3, reg=1.0, alpha=20.0 x1, signal=item_event, factors=3, reg=1.0, alpha=5.0 x1, signal=target, factors=3, reg=0.1, alpha=5.0 x1

#### NBP (test cases per seed: 151-176)

| Model | NDCG@1 | NDCG@3 | P@3 | R@3 | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|---|
| random floor | 0.248 +- 0.023 [0.231, 0.264] | 0.396 +- 0.018 [0.384, 0.409] | - | - | 0.331 +- 0.017 [0.320, 0.343] | - | - |
| popularity | 0.304 +- 0.032 [0.281, 0.327] | 0.457 +- 0.022 [0.441, 0.473] | 0.266 | 0.565 | 0.394 | 0.500 | 0.613 |
| personal_frequency | 0.529 +- 0.030 [0.507, 0.550] | 0.656 +- 0.018 [0.644, 0.669] | 0.353 | 0.757 | 0.595 | 1.000 | 0.973 |
| itemknn_self | 0.521 +- 0.025 [0.503, 0.539] | 0.651 +- 0.015 [0.640, 0.662] | 0.349 | 0.750 | 0.590 | 1.000 | 0.985 |
| itemknn_noself | 0.163 +- 0.020 [0.149, 0.177] | 0.251 +- 0.021 [0.236, 0.266] | 0.159 | 0.316 | 0.203 | 1.000 | 0.967 |
| mf | 0.366 +- 0.030 [0.345, 0.388] | 0.545 +- 0.017 [0.533, 0.557] | 0.315 | 0.674 | 0.473 | 1.000 | 0.983 |

| Paired difference (within seed, then across seeds) | NDCG@1 | CI>0 / CI<0 | NDCG@3 | CI>0 / CI<0 |
|---|---|---|---|---|
| itemknn_self - personal_frequency | -0.007 +- 0.012 [-0.016, +0.001] | 0 / 0 | -0.006 +- 0.014 [-0.016, +0.005] | 0 / 2 |
| itemknn_noself - personal_frequency | -0.366 +- 0.046 [-0.399, -0.333] | 0 / 10 | -0.405 +- 0.036 [-0.431, -0.380] | 0 / 10 |
| mf - personal_frequency | -0.162 +- 0.043 [-0.193, -0.131] | 0 / 9 | -0.111 +- 0.020 [-0.126, -0.097] | 0 / 10 |
| popularity - personal_frequency | -0.224 +- 0.037 [-0.251, -0.198] | 0 / 10 | -0.199 +- 0.030 [-0.221, -0.178] | 0 / 10 |
| itemknn_self - itemknn_noself | +0.359 +- 0.040 [+0.330, +0.387] | 10 / 0 | +0.400 +- 0.034 [+0.376, +0.424] | 10 / 0 |

Per-seed test NDCG@1 / NDCG@3:

| Seed | popularity | personal_frequency | itemknn_self | itemknn_noself | mf |
|---|---|---|---|---|---|
| 20261110 | 0.311 / 0.449 | 0.510 / 0.661 | 0.517 / 0.630 | 0.199 / 0.266 | 0.430 / 0.525 |
| 20261111 | 0.294 / 0.474 | 0.525 / 0.646 | 0.512 / 0.646 | 0.181 / 0.266 | 0.394 / 0.560 |
| 20261112 | 0.312 / 0.451 | 0.517 / 0.662 | 0.517 / 0.639 | 0.159 / 0.246 | 0.341 / 0.528 |
| 20261113 | 0.344 / 0.454 | 0.530 / 0.648 | 0.510 / 0.643 | 0.152 / 0.262 | 0.344 / 0.535 |
| 20261114 | 0.249 / 0.433 | 0.556 / 0.675 | 0.562 / 0.676 | 0.148 / 0.237 | 0.361 / 0.554 |
| 20261115 | 0.263 / 0.414 | 0.519 / 0.636 | 0.506 / 0.651 | 0.173 / 0.237 | 0.353 / 0.550 |
| 20261116 | 0.316 / 0.493 | 0.523 / 0.638 | 0.517 / 0.637 | 0.172 / 0.270 | 0.345 / 0.527 |
| 20261117 | 0.282 / 0.473 | 0.474 / 0.650 | 0.481 / 0.653 | 0.167 / 0.268 | 0.372 / 0.566 |
| 20261118 | 0.329 / 0.465 | 0.545 / 0.653 | 0.527 / 0.658 | 0.144 / 0.253 | 0.335 / 0.537 |
| 20261119 | 0.341 / 0.464 | 0.587 / 0.694 | 0.563 / 0.676 | 0.132 / 0.203 | 0.389 / 0.568 |

Selected configurations (seeds): popularity: signal=item_event x4, signal=target x6; personal_frequency: signal=item_event x10; itemknn_self: signal=item_event, include_self=True x10; itemknn_noself: signal=target, include_self=False x10; mf: signal=item_event, factors=3, reg=0.1, alpha=5.0 x1, signal=item_event, factors=3, reg=1.0, alpha=20.0 x3, signal=item_event, factors=3, reg=1.0, alpha=5.0 x2, signal=target, factors=2, reg=1.0, alpha=5.0 x1, signal=target, factors=3, reg=0.1, alpha=20.0 x1, signal=target, factors=3, reg=0.1, alpha=5.0 x1, signal=target, factors=3, reg=1.0, alpha=20.0 x1

Cold start / exclusions (test, total over seeds): customers_with_targets = 5366; insufficient_history_excluded = 123

### next-event evaluation

#### NBS (test cases per seed: 1279-1488)

| Model | NDCG@1 | NDCG@3 | P@3 | R@3 | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|---|
| random floor | 0.198 +- 0.006 [0.194, 0.203] | 0.423 +- 0.005 [0.420, 0.427] | - | - | 0.364 +- 0.005 [0.360, 0.368] | - | - |
| popularity | 0.230 +- 0.017 [0.218, 0.242] | 0.459 +- 0.012 [0.450, 0.468] | 0.211 | 0.634 | 0.399 | 0.600 | 0.683 |
| personal_frequency | 0.395 +- 0.017 [0.382, 0.407] | 0.633 +- 0.013 [0.624, 0.642] | 0.269 | 0.808 | 0.573 | 1.000 | 0.985 |
| itemknn_self | 0.376 +- 0.022 [0.361, 0.392] | 0.620 +- 0.013 [0.611, 0.629] | 0.267 | 0.800 | 0.559 | 1.000 | 0.968 |
| itemknn_noself | 0.099 +- 0.007 [0.094, 0.104] | 0.252 +- 0.011 [0.244, 0.260] | 0.125 | 0.374 | 0.210 | 1.000 | 0.997 |
| mf | 0.282 +- 0.025 [0.264, 0.300] | 0.553 +- 0.017 [0.542, 0.565] | 0.252 | 0.756 | 0.484 | 1.000 | 0.967 |

| Paired difference (within seed, then across seeds) | NDCG@1 | CI>0 / CI<0 | NDCG@3 | CI>0 / CI<0 |
|---|---|---|---|---|
| itemknn_self - personal_frequency | -0.018 +- 0.012 [-0.027, -0.010] | 0 / 5 | -0.012 +- 0.007 [-0.017, -0.008] | 0 / 6 |
| itemknn_noself - personal_frequency | -0.296 +- 0.020 [-0.310, -0.281] | 0 / 10 | -0.381 +- 0.023 [-0.397, -0.365] | 0 / 10 |
| mf - personal_frequency | -0.112 +- 0.028 [-0.133, -0.092] | 0 / 10 | -0.079 +- 0.016 [-0.091, -0.068] | 0 / 10 |
| popularity - personal_frequency | -0.165 +- 0.030 [-0.187, -0.143] | 0 / 10 | -0.174 +- 0.023 [-0.190, -0.157] | 0 / 10 |
| itemknn_self - itemknn_noself | +0.277 +- 0.023 [+0.261, +0.294] | 10 / 0 | +0.369 +- 0.021 [+0.354, +0.384] | 10 / 0 |

Per-seed test NDCG@1 / NDCG@3:

| Seed | popularity | personal_frequency | itemknn_self | itemknn_noself | mf |
|---|---|---|---|---|---|
| 20261110 | 0.241 / 0.465 | 0.400 / 0.632 | 0.399 / 0.628 | 0.102 / 0.268 | 0.284 / 0.531 |
| 20261111 | 0.242 / 0.470 | 0.387 / 0.626 | 0.366 / 0.616 | 0.101 / 0.253 | 0.278 / 0.538 |
| 20261112 | 0.233 / 0.452 | 0.408 / 0.644 | 0.403 / 0.640 | 0.093 / 0.240 | 0.320 / 0.580 |
| 20261113 | 0.254 / 0.480 | 0.366 / 0.610 | 0.354 / 0.606 | 0.106 / 0.269 | 0.252 / 0.542 |
| 20261114 | 0.224 / 0.451 | 0.397 / 0.649 | 0.377 / 0.627 | 0.087 / 0.242 | 0.285 / 0.562 |
| 20261115 | 0.248 / 0.459 | 0.392 / 0.634 | 0.360 / 0.616 | 0.096 / 0.243 | 0.274 / 0.557 |
| 20261116 | 0.222 / 0.461 | 0.415 / 0.642 | 0.386 / 0.626 | 0.099 / 0.244 | 0.312 / 0.579 |
| 20261117 | 0.195 / 0.435 | 0.418 / 0.647 | 0.409 / 0.632 | 0.106 / 0.246 | 0.234 / 0.540 |
| 20261118 | 0.217 / 0.465 | 0.392 / 0.626 | 0.356 / 0.610 | 0.091 / 0.250 | 0.293 / 0.546 |
| 20261119 | 0.224 / 0.452 | 0.371 / 0.618 | 0.352 / 0.601 | 0.107 / 0.263 | 0.290 / 0.560 |

Selected configurations (seeds): popularity: signal=target x10; personal_frequency: signal=target x10; itemknn_self: signal=target, include_self=True x10; itemknn_noself: signal=target, include_self=False x10; mf: signal=target, factors=3, reg=0.1, alpha=20.0 x6, signal=target, factors=3, reg=0.1, alpha=5.0 x1, signal=target, factors=3, reg=1.0, alpha=5.0 x3

#### NBP (test cases per seed: 241-298)

| Model | NDCG@1 | NDCG@3 | P@3 | R@3 | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|---|
| random floor | 0.166 +- 0.014 [0.156, 0.177] | 0.353 +- 0.012 [0.345, 0.361] | - | - | 0.304 +- 0.012 [0.295, 0.313] | - | - |
| popularity | 0.208 +- 0.026 [0.189, 0.227] | 0.405 +- 0.020 [0.391, 0.419] | 0.185 | 0.554 | 0.354 | 0.500 | 0.613 |
| personal_frequency | 0.386 +- 0.031 [0.364, 0.408] | 0.588 +- 0.023 [0.571, 0.605] | 0.246 | 0.737 | 0.537 | 1.000 | 0.961 |
| itemknn_self | 0.386 +- 0.031 [0.363, 0.408] | 0.586 +- 0.021 [0.571, 0.601] | 0.244 | 0.732 | 0.536 | 1.000 | 0.980 |
| itemknn_noself | 0.102 +- 0.021 [0.086, 0.117] | 0.213 +- 0.025 [0.196, 0.231] | 0.100 | 0.301 | 0.184 | 1.000 | 0.971 |
| mf | 0.242 +- 0.033 [0.219, 0.266] | 0.489 +- 0.027 [0.470, 0.508] | 0.224 | 0.672 | 0.426 | 1.000 | 0.976 |

| Paired difference (within seed, then across seeds) | NDCG@1 | CI>0 / CI<0 | NDCG@3 | CI>0 / CI<0 |
|---|---|---|---|---|
| itemknn_self - personal_frequency | -0.000 +- 0.013 [-0.009, +0.009] | 0 / 0 | -0.002 +- 0.010 [-0.009, +0.005] | 0 / 0 |
| itemknn_noself - personal_frequency | -0.284 +- 0.048 [-0.319, -0.250] | 0 / 10 | -0.375 +- 0.046 [-0.408, -0.341] | 0 / 10 |
| mf - personal_frequency | -0.144 +- 0.055 [-0.183, -0.104] | 0 / 9 | -0.099 +- 0.023 [-0.116, -0.082] | 0 / 10 |
| popularity - personal_frequency | -0.178 +- 0.035 [-0.203, -0.153] | 0 / 10 | -0.183 +- 0.033 [-0.207, -0.160] | 0 / 10 |
| itemknn_self - itemknn_noself | +0.284 +- 0.049 [+0.249, +0.319] | 10 / 0 | +0.373 +- 0.043 [+0.342, +0.403] | 10 / 0 |

Per-seed test NDCG@1 / NDCG@3:

| Seed | popularity | personal_frequency | itemknn_self | itemknn_noself | mf |
|---|---|---|---|---|---|
| 20261110 | 0.203 / 0.400 | 0.315 / 0.567 | 0.324 / 0.559 | 0.124 / 0.229 | 0.299 / 0.481 |
| 20261111 | 0.210 / 0.406 | 0.397 / 0.599 | 0.393 / 0.605 | 0.121 / 0.223 | 0.261 / 0.518 |
| 20261112 | 0.235 / 0.422 | 0.403 / 0.605 | 0.430 / 0.617 | 0.091 / 0.199 | 0.268 / 0.520 |
| 20261113 | 0.252 / 0.398 | 0.407 / 0.596 | 0.395 / 0.588 | 0.097 / 0.218 | 0.174 / 0.473 |
| 20261114 | 0.177 / 0.391 | 0.404 / 0.602 | 0.387 / 0.583 | 0.085 / 0.197 | 0.259 / 0.517 |
| 20261115 | 0.176 / 0.367 | 0.380 / 0.568 | 0.367 / 0.576 | 0.114 / 0.210 | 0.233 / 0.467 |
| 20261116 | 0.224 / 0.443 | 0.365 / 0.549 | 0.365 / 0.555 | 0.125 / 0.251 | 0.247 / 0.486 |
| 20261117 | 0.177 / 0.406 | 0.365 / 0.567 | 0.365 / 0.568 | 0.108 / 0.239 | 0.221 / 0.437 |
| 20261118 | 0.225 / 0.415 | 0.412 / 0.608 | 0.415 / 0.601 | 0.090 / 0.205 | 0.232 / 0.505 |
| 20261119 | 0.203 / 0.401 | 0.413 / 0.620 | 0.416 / 0.607 | 0.059 / 0.163 | 0.231 / 0.488 |

Selected configurations (seeds): popularity: signal=target x10; personal_frequency: signal=target x10; itemknn_self: signal=target, include_self=True x10; itemknn_noself: signal=target, include_self=False x10; mf: signal=target, factors=2, reg=0.1, alpha=5.0 x1, signal=target, factors=2, reg=1.0, alpha=20.0 x1, signal=target, factors=3, reg=0.1, alpha=20.0 x2, signal=target, factors=3, reg=0.1, alpha=5.0 x4, signal=target, factors=3, reg=1.0, alpha=20.0 x2

Next-event NDCG@1 by session stratum (across-seed mean): within-session = previous target in the same session; cross-session = first target of a session.

| Task | Model | within-session | cross-session |
|---|---|---|---|
| nbs | popularity | 0.295 | 0.208 |
| nbs | personal_frequency | 0.211 | 0.458 |
| nbs | itemknn_self | 0.236 | 0.424 |
| nbs | itemknn_noself | 0.193 | 0.067 |
| nbs | mf | 0.261 | 0.289 |
| nbp | popularity | 0.148 | 0.213 |
| nbp | personal_frequency | 0.172 | 0.403 |
| nbp | itemknn_self | 0.169 | 0.402 |
| nbp | itemknn_noself | 0.153 | 0.098 |
| nbp | mf | 0.210 | 0.246 |

Cold start / exclusions (test, total over seeds): eligible_target_events = 16222; insufficient_history_excluded = 260; tied_timestamp_merges = 0; single_target_sessions = 9386

## Profile `transition_dominant`

Dirichlet concentration service/product = 5.0/5.0; transition probability service/product = 0.85/0.75.

### window evaluation

#### NBS (test cases per seed: 348-379)

| Model | NDCG@1 | NDCG@3 | P@3 | R@3 | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|---|
| random floor | 0.523 +- 0.019 [0.509, 0.536] | 0.609 +- 0.011 [0.601, 0.617] | - | - | 0.527 +- 0.013 [0.517, 0.536] | - | - |
| popularity | 0.628 +- 0.015 [0.617, 0.639] | 0.675 +- 0.012 [0.666, 0.684] | 0.563 | 0.657 | 0.603 | 0.600 | 0.683 |
| personal_frequency | 0.601 +- 0.015 [0.590, 0.612] | 0.676 +- 0.011 [0.668, 0.684] | 0.566 | 0.667 | 0.603 | 1.000 | 0.965 |
| itemknn_self | 0.621 +- 0.021 [0.605, 0.636] | 0.686 +- 0.017 [0.674, 0.698] | 0.573 | 0.672 | 0.616 | 1.000 | 0.892 |
| itemknn_noself | 0.484 +- 0.026 [0.466, 0.503] | 0.575 +- 0.016 [0.564, 0.587] | 0.501 | 0.571 | 0.493 | 1.000 | 0.997 |
| mf | 0.599 +- 0.028 [0.579, 0.619] | 0.667 +- 0.016 [0.656, 0.678] | 0.561 | 0.655 | 0.594 | 1.000 | 0.905 |

| Paired difference (within seed, then across seeds) | NDCG@1 | CI>0 / CI<0 | NDCG@3 | CI>0 / CI<0 |
|---|---|---|---|---|
| itemknn_self - personal_frequency | +0.020 +- 0.020 [+0.006, +0.034] | 2 / 0 | +0.010 +- 0.009 [+0.003, +0.016] | 1 / 0 |
| itemknn_noself - personal_frequency | -0.117 +- 0.036 [-0.142, -0.091] | 0 / 9 | -0.101 +- 0.024 [-0.118, -0.084] | 0 / 10 |
| mf - personal_frequency | -0.002 +- 0.029 [-0.023, +0.019] | 0 / 0 | -0.009 +- 0.017 [-0.021, +0.003] | 0 / 2 |
| popularity - personal_frequency | +0.028 +- 0.016 [+0.016, +0.039] | 0 / 0 | -0.001 +- 0.013 [-0.010, +0.008] | 0 / 0 |
| itemknn_self - itemknn_noself | +0.136 +- 0.030 [+0.115, +0.158] | 10 / 0 | +0.110 +- 0.026 [+0.092, +0.129] | 10 / 0 |

Per-seed test NDCG@1 / NDCG@3:

| Seed | popularity | personal_frequency | itemknn_self | itemknn_noself | mf |
|---|---|---|---|---|---|
| 20261110 | 0.620 / 0.676 | 0.590 / 0.671 | 0.628 / 0.685 | 0.508 / 0.576 | 0.620 / 0.684 |
| 20261111 | 0.638 / 0.685 | 0.624 / 0.690 | 0.624 / 0.709 | 0.483 / 0.559 | 0.578 / 0.666 |
| 20261112 | 0.598 / 0.648 | 0.574 / 0.657 | 0.576 / 0.647 | 0.496 / 0.574 | 0.568 / 0.644 |
| 20261113 | 0.623 / 0.681 | 0.598 / 0.683 | 0.620 / 0.689 | 0.454 / 0.560 | 0.582 / 0.648 |
| 20261114 | 0.633 / 0.679 | 0.592 / 0.662 | 0.630 / 0.683 | 0.507 / 0.591 | 0.619 / 0.672 |
| 20261115 | 0.625 / 0.672 | 0.614 / 0.682 | 0.606 / 0.693 | 0.460 / 0.563 | 0.577 / 0.651 |
| 20261116 | 0.654 / 0.692 | 0.609 / 0.679 | 0.662 / 0.693 | 0.503 / 0.597 | 0.654 / 0.677 |
| 20261117 | 0.617 / 0.666 | 0.609 / 0.688 | 0.623 / 0.701 | 0.472 / 0.569 | 0.609 / 0.683 |
| 20261118 | 0.642 / 0.684 | 0.586 / 0.671 | 0.617 / 0.679 | 0.517 / 0.603 | 0.572 / 0.660 |
| 20261119 | 0.634 / 0.666 | 0.612 / 0.678 | 0.620 / 0.677 | 0.444 / 0.560 | 0.606 / 0.684 |

Selected configurations (seeds): popularity: signal=item_event x4, signal=target x6; personal_frequency: signal=item_event x5, signal=target x5; itemknn_self: signal=item_event, include_self=True x8, signal=target, include_self=True x2; itemknn_noself: signal=item_event, include_self=False x3, signal=target, include_self=False x7; mf: signal=item_event, factors=2, reg=0.1, alpha=20.0 x3, signal=item_event, factors=2, reg=0.1, alpha=5.0 x1, signal=item_event, factors=2, reg=1.0, alpha=20.0 x1, signal=item_event, factors=3, reg=0.1, alpha=20.0 x1, signal=item_event, factors=3, reg=0.1, alpha=5.0 x1, signal=target, factors=2, reg=0.1, alpha=20.0 x2, signal=target, factors=2, reg=0.1, alpha=5.0 x1

#### NBP (test cases per seed: 142-177)

| Model | NDCG@1 | NDCG@3 | P@3 | R@3 | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|---|
| random floor | 0.262 +- 0.012 [0.253, 0.271] | 0.401 +- 0.008 [0.396, 0.407] | - | - | 0.331 +- 0.007 [0.326, 0.336] | - | - |
| popularity | 0.341 +- 0.042 [0.311, 0.370] | 0.482 +- 0.036 [0.455, 0.508] | 0.301 | 0.579 | 0.413 | 0.500 | 0.613 |
| personal_frequency | 0.373 +- 0.034 [0.348, 0.397] | 0.503 +- 0.023 [0.486, 0.519] | 0.313 | 0.597 | 0.431 | 1.000 | 0.950 |
| itemknn_self | 0.363 +- 0.046 [0.331, 0.396] | 0.492 +- 0.027 [0.473, 0.511] | 0.307 | 0.582 | 0.423 | 1.000 | 0.965 |
| itemknn_noself | 0.244 +- 0.025 [0.226, 0.262] | 0.365 +- 0.024 [0.348, 0.382] | 0.238 | 0.449 | 0.299 | 1.000 | 0.978 |
| mf | 0.324 +- 0.026 [0.305, 0.343] | 0.471 +- 0.032 [0.448, 0.493] | 0.302 | 0.574 | 0.399 | 0.967 | 0.928 |

| Paired difference (within seed, then across seeds) | NDCG@1 | CI>0 / CI<0 | NDCG@3 | CI>0 / CI<0 |
|---|---|---|---|---|
| itemknn_self - personal_frequency | -0.009 +- 0.025 [-0.027, +0.008] | 0 / 1 | -0.011 +- 0.016 [-0.022, +0.000] | 0 / 2 |
| itemknn_noself - personal_frequency | -0.129 +- 0.039 [-0.157, -0.101] | 0 / 7 | -0.138 +- 0.029 [-0.158, -0.117] | 0 / 10 |
| mf - personal_frequency | -0.049 +- 0.037 [-0.076, -0.022] | 0 / 1 | -0.032 +- 0.026 [-0.051, -0.014] | 0 / 1 |
| popularity - personal_frequency | -0.032 +- 0.050 [-0.068, +0.004] | 0 / 1 | -0.021 +- 0.039 [-0.049, +0.007] | 0 / 1 |
| itemknn_self - itemknn_noself | +0.119 +- 0.045 [+0.087, +0.151] | 8 / 0 | +0.127 +- 0.025 [+0.109, +0.145] | 10 / 0 |

Per-seed test NDCG@1 / NDCG@3:

| Seed | popularity | personal_frequency | itemknn_self | itemknn_noself | mf |
|---|---|---|---|---|---|
| 20261110 | 0.295 / 0.430 | 0.376 / 0.488 | 0.369 / 0.469 | 0.221 / 0.340 | 0.275 / 0.405 |
| 20261111 | 0.317 / 0.474 | 0.345 / 0.500 | 0.282 / 0.455 | 0.239 / 0.355 | 0.317 / 0.451 |
| 20261112 | 0.366 / 0.438 | 0.348 / 0.477 | 0.360 / 0.474 | 0.226 / 0.329 | 0.366 / 0.460 |
| 20261113 | 0.419 / 0.539 | 0.419 / 0.536 | 0.425 / 0.543 | 0.287 / 0.394 | 0.341 / 0.505 |
| 20261114 | 0.356 / 0.501 | 0.393 / 0.490 | 0.417 / 0.495 | 0.264 / 0.396 | 0.344 / 0.490 |
| 20261115 | 0.337 / 0.529 | 0.314 / 0.477 | 0.302 / 0.477 | 0.249 / 0.371 | 0.290 / 0.440 |
| 20261116 | 0.327 / 0.478 | 0.345 / 0.487 | 0.352 / 0.478 | 0.255 / 0.367 | 0.333 / 0.495 |
| 20261117 | 0.350 / 0.491 | 0.401 / 0.522 | 0.390 / 0.499 | 0.266 / 0.367 | 0.322 / 0.491 |
| 20261118 | 0.271 / 0.446 | 0.413 / 0.537 | 0.387 / 0.518 | 0.200 / 0.342 | 0.329 / 0.495 |
| 20261119 | 0.368 / 0.489 | 0.374 / 0.515 | 0.351 / 0.511 | 0.234 / 0.390 | 0.322 / 0.473 |

Selected configurations (seeds): popularity: signal=item_event x1, signal=target x9; personal_frequency: signal=item_event x5, signal=target x5; itemknn_self: signal=item_event, include_self=True x5, signal=target, include_self=True x5; itemknn_noself: signal=item_event, include_self=False x2, signal=target, include_self=False x8; mf: signal=item_event, factors=2, reg=0.1, alpha=5.0 x1, signal=item_event, factors=2, reg=1.0, alpha=20.0 x3, signal=item_event, factors=3, reg=0.1, alpha=5.0 x2, signal=target, factors=2, reg=0.1, alpha=20.0 x1, signal=target, factors=3, reg=0.1, alpha=20.0 x1, signal=target, factors=3, reg=0.1, alpha=5.0 x2

Cold start / exclusions (test, total over seeds): customers_with_targets = 5398; insufficient_history_excluded = 109

### next-event evaluation

#### NBS (test cases per seed: 1303-1621)

| Model | NDCG@1 | NDCG@3 | P@3 | R@3 | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|---|
| random floor | 0.201 +- 0.004 [0.198, 0.204] | 0.429 +- 0.003 [0.427, 0.431] | - | - | 0.369 +- 0.003 [0.367, 0.372] | - | - |
| popularity | 0.263 +- 0.008 [0.257, 0.268] | 0.492 +- 0.008 [0.487, 0.498] | 0.223 | 0.669 | 0.432 | 0.600 | 0.683 |
| personal_frequency | 0.230 +- 0.014 [0.219, 0.240] | 0.473 +- 0.010 [0.466, 0.480] | 0.219 | 0.657 | 0.410 | 1.000 | 0.963 |
| itemknn_self | 0.249 +- 0.012 [0.240, 0.257] | 0.494 +- 0.008 [0.489, 0.500] | 0.226 | 0.679 | 0.431 | 1.000 | 0.890 |
| itemknn_noself | 0.191 +- 0.007 [0.186, 0.196] | 0.413 +- 0.010 [0.406, 0.420] | 0.195 | 0.584 | 0.354 | 1.000 | 0.996 |
| mf | 0.241 +- 0.017 [0.229, 0.253] | 0.488 +- 0.008 [0.482, 0.493] | 0.225 | 0.674 | 0.424 | 0.980 | 0.900 |

| Paired difference (within seed, then across seeds) | NDCG@1 | CI>0 / CI<0 | NDCG@3 | CI>0 / CI<0 |
|---|---|---|---|---|
| itemknn_self - personal_frequency | +0.019 +- 0.014 [+0.009, +0.029] | 7 / 0 | +0.021 +- 0.008 [+0.016, +0.027] | 9 / 0 |
| itemknn_noself - personal_frequency | -0.039 +- 0.018 [-0.051, -0.026] | 0 / 6 | -0.060 +- 0.019 [-0.074, -0.047] | 0 / 9 |
| mf - personal_frequency | +0.012 +- 0.022 [-0.004, +0.027] | 4 / 0 | +0.015 +- 0.009 [+0.008, +0.021] | 3 / 0 |
| popularity - personal_frequency | +0.033 +- 0.013 [+0.024, +0.043] | 8 / 0 | +0.019 +- 0.009 [+0.013, +0.026] | 4 / 0 |
| itemknn_self - itemknn_noself | +0.058 +- 0.017 [+0.046, +0.070] | 10 / 0 | +0.082 +- 0.015 [+0.071, +0.092] | 10 / 0 |

Per-seed test NDCG@1 / NDCG@3:

| Seed | popularity | personal_frequency | itemknn_self | itemknn_noself | mf |
|---|---|---|---|---|---|
| 20261110 | 0.252 / 0.484 | 0.217 / 0.462 | 0.234 / 0.479 | 0.200 / 0.413 | 0.229 / 0.480 |
| 20261111 | 0.268 / 0.496 | 0.229 / 0.466 | 0.258 / 0.502 | 0.194 / 0.427 | 0.262 / 0.496 |
| 20261112 | 0.265 / 0.495 | 0.263 / 0.484 | 0.245 / 0.491 | 0.191 / 0.411 | 0.248 / 0.494 |
| 20261113 | 0.262 / 0.493 | 0.220 / 0.476 | 0.240 / 0.496 | 0.189 / 0.411 | 0.232 / 0.485 |
| 20261114 | 0.266 / 0.500 | 0.220 / 0.475 | 0.237 / 0.495 | 0.192 / 0.419 | 0.260 / 0.492 |
| 20261115 | 0.249 / 0.481 | 0.212 / 0.456 | 0.232 / 0.485 | 0.195 / 0.421 | 0.249 / 0.473 |
| 20261116 | 0.276 / 0.508 | 0.229 / 0.476 | 0.262 / 0.498 | 0.188 / 0.406 | 0.214 / 0.479 |
| 20261117 | 0.259 / 0.490 | 0.239 / 0.488 | 0.258 / 0.502 | 0.177 / 0.392 | 0.225 / 0.491 |
| 20261118 | 0.265 / 0.487 | 0.236 / 0.467 | 0.256 / 0.493 | 0.202 / 0.422 | 0.261 / 0.492 |
| 20261119 | 0.266 / 0.488 | 0.231 / 0.478 | 0.264 / 0.502 | 0.184 / 0.405 | 0.233 / 0.493 |

Selected configurations (seeds): popularity: signal=target x10; personal_frequency: signal=target x10; itemknn_self: signal=target, include_self=True x10; itemknn_noself: signal=target, include_self=False x10; mf: signal=target, factors=2, reg=0.1, alpha=20.0 x5, signal=target, factors=2, reg=0.1, alpha=5.0 x1, signal=target, factors=3, reg=0.1, alpha=5.0 x2, signal=target, factors=3, reg=1.0, alpha=5.0 x2

#### NBP (test cases per seed: 243-333)

| Model | NDCG@1 | NDCG@3 | P@3 | R@3 | MAP@3 | Coverage@3 | Exposure div.@3 |
|---|---|---|---|---|---|---|---|
| random floor | 0.171 +- 0.013 [0.161, 0.180] | 0.361 +- 0.013 [0.352, 0.370] | - | - | 0.311 +- 0.012 [0.302, 0.320] | - | - |
| popularity | 0.215 +- 0.026 [0.196, 0.233] | 0.412 +- 0.033 [0.389, 0.435] | 0.186 | 0.559 | 0.361 | 0.500 | 0.613 |
| personal_frequency | 0.241 +- 0.026 [0.222, 0.260] | 0.445 +- 0.017 [0.433, 0.457] | 0.200 | 0.599 | 0.392 | 1.000 | 0.955 |
| itemknn_self | 0.245 +- 0.022 [0.229, 0.261] | 0.438 +- 0.015 [0.427, 0.449] | 0.194 | 0.582 | 0.388 | 1.000 | 0.973 |
| itemknn_noself | 0.152 +- 0.015 [0.141, 0.163] | 0.313 +- 0.017 [0.300, 0.325] | 0.146 | 0.438 | 0.270 | 1.000 | 0.978 |
| mf | 0.211 +- 0.025 [0.193, 0.229] | 0.412 +- 0.028 [0.393, 0.432] | 0.188 | 0.563 | 0.361 | 0.967 | 0.952 |

| Paired difference (within seed, then across seeds) | NDCG@1 | CI>0 / CI<0 | NDCG@3 | CI>0 / CI<0 |
|---|---|---|---|---|
| itemknn_self - personal_frequency | +0.004 +- 0.010 [-0.003, +0.011] | 0 / 0 | -0.007 +- 0.009 [-0.013, -0.000] | 0 / 0 |
| itemknn_noself - personal_frequency | -0.089 +- 0.039 [-0.117, -0.061] | 0 / 7 | -0.132 +- 0.027 [-0.151, -0.113] | 0 / 10 |
| mf - personal_frequency | -0.030 +- 0.028 [-0.049, -0.010] | 0 / 1 | -0.033 +- 0.023 [-0.049, -0.016] | 0 / 3 |
| popularity - personal_frequency | -0.026 +- 0.042 [-0.057, +0.004] | 0 / 1 | -0.033 +- 0.035 [-0.058, -0.008] | 0 / 2 |
| itemknn_self - itemknn_noself | +0.093 +- 0.035 [+0.067, +0.118] | 8 / 0 | +0.125 +- 0.027 [+0.105, +0.145] | 10 / 0 |

Per-seed test NDCG@1 / NDCG@3:

| Seed | popularity | personal_frequency | itemknn_self | itemknn_noself | mf |
|---|---|---|---|---|---|
| 20261110 | 0.202 / 0.398 | 0.226 / 0.421 | 0.214 / 0.422 | 0.173 / 0.311 | 0.185 / 0.366 |
| 20261111 | 0.182 / 0.385 | 0.233 / 0.426 | 0.237 / 0.433 | 0.154 / 0.304 | 0.162 / 0.381 |
| 20261112 | 0.211 / 0.375 | 0.208 / 0.432 | 0.215 / 0.423 | 0.148 / 0.289 | 0.228 / 0.442 |
| 20261113 | 0.250 / 0.443 | 0.224 / 0.458 | 0.237 / 0.440 | 0.154 / 0.321 | 0.212 / 0.401 |
| 20261114 | 0.213 / 0.430 | 0.246 / 0.461 | 0.259 / 0.446 | 0.143 / 0.311 | 0.209 / 0.421 |
| 20261115 | 0.237 / 0.471 | 0.214 / 0.436 | 0.229 / 0.419 | 0.173 / 0.341 | 0.192 / 0.382 |
| 20261116 | 0.217 / 0.409 | 0.238 / 0.434 | 0.249 / 0.436 | 0.159 / 0.328 | 0.242 / 0.431 |
| 20261117 | 0.219 / 0.419 | 0.258 / 0.448 | 0.255 / 0.446 | 0.156 / 0.328 | 0.219 / 0.430 |
| 20261118 | 0.170 / 0.364 | 0.292 / 0.471 | 0.282 / 0.471 | 0.123 / 0.286 | 0.231 / 0.436 |
| 20261119 | 0.247 / 0.426 | 0.269 / 0.461 | 0.269 / 0.445 | 0.138 / 0.309 | 0.233 / 0.435 |

Selected configurations (seeds): popularity: signal=target x10; personal_frequency: signal=target x10; itemknn_self: signal=target, include_self=True x10; itemknn_noself: signal=target, include_self=False x10; mf: signal=target, factors=2, reg=0.1, alpha=20.0 x2, signal=target, factors=2, reg=1.0, alpha=20.0 x1, signal=target, factors=2, reg=1.0, alpha=5.0 x1, signal=target, factors=3, reg=0.1, alpha=20.0 x2, signal=target, factors=3, reg=0.1, alpha=5.0 x2, signal=target, factors=3, reg=1.0, alpha=20.0 x1, signal=target, factors=3, reg=1.0, alpha=5.0 x1

Next-event NDCG@1 by session stratum (across-seed mean): within-session = previous target in the same session; cross-session = first target of a session.

| Task | Model | within-session | cross-session |
|---|---|---|---|
| nbs | popularity | 0.363 | 0.220 |
| nbs | personal_frequency | 0.206 | 0.239 |
| nbs | itemknn_self | 0.292 | 0.230 |
| nbs | itemknn_noself | 0.256 | 0.163 |
| nbs | mf | 0.290 | 0.221 |
| nbp | popularity | 0.147 | 0.222 |
| nbp | personal_frequency | 0.063 | 0.262 |
| nbp | itemknn_self | 0.097 | 0.262 |
| nbp | itemknn_noself | 0.282 | 0.138 |
| nbp | mf | 0.192 | 0.214 |

Cold start / exclusions (test, total over seeds): eligible_target_events = 17274; insufficient_history_excluded = 278; tied_timestamp_merges = 0; single_target_sessions = 8706

Notes: exposure diversity measures how evenly recommendations spread across the catalog, not per-customer semantic diversity. All numbers are synthetic-data experiments and say nothing about real customers or real-world effectiveness.
