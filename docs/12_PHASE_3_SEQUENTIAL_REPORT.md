# Phase 3 — Sequential Recommendation and SASRec Ablation

> **SYNTHETIC-DATA EXPERIMENT.** Every number in this report comes from synthetic
> datasets of a fictional Indonesian commercial bank's mobile banking application,
> produced by invented behavior profiles. None of it is evidence about real customers,
> real banking behavior, or real-world recommendation effectiveness. All runs happened
> in a local research environment (one laptop GPU), not in a production deployment.

Frozen results: [`benchmarks/phase3/phase3_report.md`](../benchmarks/phase3/phase3_report.md)
(full per-profile tables, all metrics, per-seed values) and `phase3_report.json`.
Locks: `benchmarks/manifests/phase3-seeds.json` (seed reservation),
`phase3-config.json` (frozen SASRec configuration), `phase3.json` (final evaluation).

## 1. Research questions

Phase 2.2–2.3 established that Personal Frequency (PF), which ranks items by how
often the customer used them before, is a strong baseline, and that no classical
model captures order. Phase 3 asks:

- **RQ1.** Does a sequential model (SASRec) outperform PF and the other classical
  baselines on next-event prediction?
- **RQ2.** Does any advantage depend on the behavioral regime (habit, balanced,
  transition) and the task (NBS, NBP)?
- **RQ3.** Where does the advantage come from: transitions within an active session,
  or recurring behavior across sessions?
- **RQ4.** Which architectural components matter: chronological order, attention,
  explicit positional embeddings, session-boundary information?

Secondary: does next-event performance carry over to the 7-day window protocol, and
at what computational cost?

## 2. Data and seed design

- Generator 0.3.0 behavior profiles (invented assumptions; see
  [03_DATA_DICTIONARY.md](03_DATA_DICTIONARY.md)): `habit_dominant` (peaked personal
  affinity, rare in-session transitions), `balanced`, and `transition_dominant`
  (near-uniform affinity, 85% / 75% of service / product follow-ups in a session
  follow a fixed map).
- 500 customers, 30 days, chronological split 16/7/7 (train/val/test), fail-closed
  consent filtering, NBS and NBP built and evaluated separately.
- Seeds were reserved in a pinned manifest **before** any Phase 3 dataset existed,
  and checked against every previously registered seed:
  - `phase3_tuning` 20261120–20261122 (9 datasets): hyperparameter search on
    validation splits only. Their test splits were never evaluated.
  - `phase3_final` 20261130–20261139 (30 datasets): final evaluation, after the
    configuration was frozen; every test split used once.
- The same seed list is used for every profile, so for a given seed customer
  attributes and consent are identical across profiles and only behavior differs.

## 3. Sequential cases (shared with the evaluator)

SASRec uses exactly the Phase 2.3 next-event cases, built for the train, val and test
windows with the same rules:

- One case per eligible *first target* event (first target event per task, customer,
  session and item). Its relevant set is the target item, or all items when targets
  tie on a timestamp (none occurred in the 30 final datasets).
- History is the customer's same-task, consent-permitted first targets strictly before
  the target timestamp. Tied history events become one unordered sequence step (mean
  of their embeddings). The target and future events never enter the input, and this
  is tested.
- Sequences are left-padded and truncated to the 20 most recent steps.
- The session-boundary flag of a step is 1 when it starts a new session relative to
  the previous represented history step. The earliest represented step is always 0.
  The target's session is never read.
- The within-/cross-session label (is the previous target in the target's session?)
  is computed by the evaluator for stratified reporting only. No model receives it.

**History-length shift (limitation).** Train-window cases have short histories
(P50 4, P95 14, maximum 30 steps), while validation cases are about twice as long
(P50 9, P95 23, maximum 40). Test histories are longer still, because test comes
later. This is a temporal distribution shift, not leakage, and the protocol was
deliberately not changed. The `max_seq_len` search space {10, 20, 30} was chosen from
train-window statistics only. 50 was excluded because positions beyond 30 are never
trained, so their positional embeddings would inject untrained parameters at
evaluation time.

## 4. Models

**Classical** (unchanged Phase 2.3 grids, suites `phase3` / `phase3-next-event`):
Popularity, Personal Frequency, ItemKNN with and without self-similarity, and
implicit MF. Configurations are selected on validation NDCG@3 per dataset. Random is
the 5-seed random floor.

**SASRec Full** (`bri.seq.sasrec`):
- item embeddings with tied output weights, scored by full softmax over the task's
  5–6 candidates;
- learned positional embeddings;
- pre-LayerNorm Transformer blocks with explicit causal attention and a key-padding
  mask;
- the last valid position as the case representation.

Loss is the multi-positive full softmax `logsumexp(all) − logsumexp(positives)`.
Empty histories get zero scores, so the shared tie-break decides them.

**Ablations** each change exactly one component. All variants share construction and
initialization, so for one seed every shared parameter starts bitwise identical
(tested).

| Variant | Change | What it tests |
|---|---|---|
| NoPosition | no positional embedding table | value of explicit positions |
| Shuffled | history step order randomly permuted in **training** batches only (seeded); validation/test chronological | value of chronological order |
| MeanPool | masked mean of step embeddings; no Transformer, no positions | value of attention |
| SessionBoundary | + 2-entry boundary embedding from historical flags | value of explicit session-change information |

**NoPosition is not order-invariant.** It keeps the causal mask. In stacked causal
layers, position *i* aggregates only positions ≤ *i*, so intermediate representations
depend on the order of the prefix, and the last position's output depends on it too.
A test confirms that permuting the history changes NoPosition's scores. Its result
therefore measures explicit learned positional embeddings on top of causal attention,
not "whether order matters".

All variants use the same frozen hyperparameters (Section 5). The ablations measure
component contributions **under one controlled configuration**. They are not each
architecture's independently tuned best.

## 5. Hyperparameter search (tuning seeds only)

- Optuna TPE (sampler seed 31), SQLite storage, `n_jobs=1`, 30 trials, no pruner.
- Every trial trained SASRec Full on all 9 tuning datasets × 2 tasks on CUDA, with
  validation-only early stopping.
- Objective: the unweighted mean over the six (profile, task) groups of mean
  validation NDCG@3 over the 3 tuning seeds.
- Test-window events were dropped before building sequences.
- Selection rule, fixed before the search: highest objective, ties to the lowest
  trial number.

| | Value |
|---|---|
| Trials | 30 complete, 0 failed, 0 pruned; 1.18 h total, median 136 s per trial |
| Objective distribution | min 0.591, median 0.623, max **0.628** (trial 21) |
| Frozen configuration | dim 64, 2 heads, 3 blocks, dropout 0.043, lr 1.63e-4, weight decay 2.74e-4, batch 64, max_seq_len 20, ≤ 40 epochs, patience 5, grad clip 1.0, training seed 1 |
| Stability | trial 21 ranks 1st (transition/NBS), 2nd (habit/NBS, balanced/NBP), 4th (balanced/NBS) and 5th (habit/NBP) of 30. It ranks **11th on transition/NBP** (gap 0.013 to that group's best trial, whose range over all trials was only 0.41–0.44) |
| Convergence | all top-10 trials share dim 64, 3 blocks, max_seq_len 20; the top 5 differ by ≤ 0.0011 |

The configuration was frozen in a pinned manifest before any final dataset was
generated, and it was not changed afterwards. HPO validation scores are optimistic,
because the same validation split drove early stopping and selection.

## 6. Evaluation and statistics

- **Primary protocol: next-event.** One case per eligible target event, history
  strictly before it. Bootstrap resampling unit: the **customer** (cluster
  bootstrap), because one customer contributes many cases.
- **Secondary protocol: 7-day window.** One case per customer and task. The relevant
  set is the distinct target items in the window, and history is everything before
  the window start. SASRec is applied with the same trained model: its next-item
  scores from history before the window rank the window's item set. That is **not**
  SASRec's training objective, which is the single next item, so window results test
  transfer rather than an optimized model.
- All models are scored on the same cases, candidates and relevance, with the shared
  evaluator and the shared tie-break (train popularity, then item id). Case ids are
  asserted identical across model families.

Two separate uncertainty levels, **never pooled**:

| Level | What is resampled / averaged | What it describes | Where it appears |
|---|---|---|---|
| Within seed | paired percentile bootstrap (1,000): customers (next-event), customer-task cases (window) | variation inside one synthetic dataset | "CI>0 / CI<0" counts: seeds whose within-seed CI excludes 0 |
| Across seeds | each seed's test value or paired difference = 1 observation; mean, sample SD, Student-t 95% interval (df = 9) | variation between independent synthetic datasets of one profile at fixed size | brackets `[lo, hi]` in all tables below |

Not covered: SASRec initialization variation (one training seed, 1, everywhere: the
10 seeds are 10 datasets, **not** 10 independent initializations), other profiles,
sizes, windows or generator versions, and any real population. No interval is computed
across profiles.

## 7. Results

### 7.1 Next-event NDCG@1 (primary)

Across-seed mean [Student-t 95% CI over 10 seeds].

| Model | habit / NBS | habit / NBP | balanced / NBS | balanced / NBP | transition / NBS | transition / NBP |
|---|---|---|---|---|---|---|
| Random floor (mean) | 0.202 | 0.164 | 0.201 | 0.167 | 0.203 | 0.171 |
| Popularity | 0.210 [0.203, 0.217] | 0.189 [0.169, 0.208] | 0.234 [0.223, 0.245] | 0.203 [0.187, 0.219] | 0.264 [0.258, 0.269] | 0.203 [0.180, 0.226] |
| **Personal Frequency** | **0.654** [0.641, 0.667] | 0.639 [0.618, 0.661] | **0.390** [0.380, 0.400] | **0.373** [0.345, 0.401] | 0.235 [0.227, 0.244] | 0.233 [0.214, 0.253] |
| ItemKNN self | 0.653 [0.639, 0.666] | **0.643** [0.622, 0.664] | 0.375 [0.363, 0.387] | 0.372 [0.346, 0.398] | 0.247 [0.236, 0.258] | **0.234** [0.219, 0.249] |
| ItemKNN no-self | 0.041 [0.037, 0.044] | 0.043 [0.034, 0.053] | 0.102 [0.096, 0.109] | 0.092 [0.081, 0.103] | 0.197 [0.188, 0.206] | 0.165 [0.152, 0.177] |
| MF | 0.437 [0.419, 0.454] | 0.439 [0.408, 0.469] | 0.279 [0.264, 0.295] | 0.231 [0.213, 0.249] | 0.244 [0.232, 0.256] | 0.205 [0.183, 0.226] |
| **SASRec Full** | 0.640 [0.630, 0.650] | 0.620 [0.601, 0.639] | 0.373 [0.356, 0.390] | 0.339 [0.313, 0.365] | **0.400** [0.387, 0.412] | 0.218 [0.204, 0.233] |
| NoPosition | 0.642 [0.632, 0.653] | 0.619 [0.600, 0.638] | 0.372 [0.360, 0.384] | 0.338 [0.311, 0.364] | 0.399 [0.387, 0.410] | 0.218 [0.200, 0.236] |
| Shuffled | 0.642 [0.631, 0.654] | 0.619 [0.600, 0.638] | 0.344 [0.335, 0.354] | 0.342 [0.323, 0.361] | 0.300 [0.288, 0.311] | 0.209 [0.198, 0.219] |
| MeanPool | 0.652 [0.640, 0.665] | 0.632 [0.610, 0.655] | 0.385 [0.374, 0.396] | 0.363 [0.334, 0.392] | 0.265 [0.259, 0.271] | 0.231 [0.214, 0.249] |
| SessionBoundary | 0.640 [0.631, 0.650] | 0.618 [0.601, 0.635] | 0.373 [0.360, 0.386] | 0.335 [0.310, 0.359] | 0.399 [0.387, 0.412] | 0.217 [0.199, 0.235] |

Next-event NDCG@3 shows the same pattern. For example, transition/NBS: SASRec Full
0.579 [0.570, 0.587] vs PF 0.479 [0.470, 0.489]; habit/NBS: 0.819 vs 0.826.
Precision, recall, MAP, coverage and exposure diversity for every model and group are
in the frozen report. On transition/NBS, P@3 / R@3 / MAP@3 are SASRec Full 0.238 /
0.715 / 0.532 vs PF 0.222 / 0.665 / 0.416. Coverage@3 is 1.00 for every personalized
model in every group, and 0.98 for SASRec Full and NoPosition on transition/NBP.
Exposure diversity@3 is 0.81–0.99, lowest for MeanPool (0.81) and Shuffled (0.86) on
transition/NBS. Popularity has coverage 0.50–0.60 by construction.

### 7.2 Paired differences, next-event NDCG@1

Format: across-seed mean [t-CI], then within-seed CI>0 / CI<0 seed counts.

| Comparison | habit / NBS | habit / NBP | balanced / NBS | balanced / NBP | transition / NBS | transition / NBP |
|---|---|---|---|---|---|---|
| Full − PF | −0.014 [−0.021, −0.008] 0/4 | −0.019 [−0.032, −0.006] 0/1 | −0.016 [−0.034, +0.001] 0/3 | −0.034 [−0.053, −0.015] 0/3 | **+0.164 [+0.152, +0.176] 10/0** | −0.015 [−0.037, +0.007] 0/1 |
| Full − Shuffled | −0.002 [−0.009, +0.005] 0/1 | +0.001 [−0.009, +0.010] 1/0 | +0.029 [+0.013, +0.045] 5/0 | −0.003 [−0.017, +0.011] 0/2 | **+0.100 [+0.089, +0.110] 10/0** | +0.009 [−0.002, +0.021] 1/0 |
| Full − MeanPool | −0.012 [−0.017, −0.007] 0/3 | −0.012 [−0.028, +0.004] 0/1 | −0.012 [−0.026, +0.002] 0/2 | −0.024 [−0.040, −0.007] 0/1 | **+0.135 [+0.124, +0.146] 10/0** | −0.013 [−0.030, +0.004] 0/0 |
| Full − NoPosition | −0.002 [−0.007, +0.002] 0/1 | +0.001 [−0.001, +0.003] 0/0 | +0.001 [−0.010, +0.013] 0/1 | +0.001 [−0.003, +0.006] 0/0 | +0.001 [−0.002, +0.004] 1/0 | +0.001 [−0.008, +0.009] 0/0 |
| SessionBoundary − Full | +0.000 [−0.001, +0.002] 0/0 | −0.002 [−0.011, +0.007] 0/2 | −0.001 [−0.009, +0.008] 1/0 | −0.004 [−0.010, +0.001] 0/0 | −0.000 [−0.001, +0.001] 0/0 | −0.001 [−0.007, +0.005] 0/0 |
| Full − ItemKNN self | −0.012 [−0.020, −0.005] 0/4 | −0.023 [−0.035, −0.010] 0/2 | −0.002 [−0.013, +0.010] 0/0 | −0.033 [−0.051, −0.015] 0/3 | +0.152 [+0.142, +0.162] 10/0 | −0.016 [−0.032, +0.001] 0/0 |
| Full − MF | +0.203 [+0.185, +0.222] 10/0 | +0.182 [+0.149, +0.214] 10/0 | +0.094 [+0.076, +0.111] 10/0 | +0.108 [+0.088, +0.128] 10/0 | +0.156 [+0.142, +0.169] 10/0 | +0.013 [−0.009, +0.036] 1/0 |
| Full − Popularity | +0.431 [+0.417, +0.444] 10/0 | +0.431 [+0.403, +0.460] 10/0 | +0.139 [+0.122, +0.157] 10/0 | +0.136 [+0.111, +0.162] 9/0 | +0.136 [+0.124, +0.148] 10/0 | +0.015 [−0.017, +0.047] 1/0 |

Full − PF at NDCG@3: −0.007, −0.018, −0.005 [−0.012, +0.002], −0.038, **+0.099
[+0.091, +0.107]**, and −0.011 [−0.032, +0.011], in the same group order.

### 7.3 Within-session vs cross-session targets (next-event NDCG@1)

Within-session means the previous target is in the same session; cross-session means
the target is the first target of its session. The label comes from the evaluator
only.

| Group | within / cross cases per seed | PF within | PF cross | **Full within** | **Full cross** | Shuffled within | MeanPool within | NoPosition within | SessionBoundary within |
|---|---|---|---|---|---|---|---|---|---|
| habit / NBS | 153–184 / 897–982 | 0.301 [0.275, 0.328] | 0.718 [0.702, 0.733] | 0.319 [0.291, 0.348] | 0.698 [0.684, 0.712] | 0.258 | 0.296 | 0.327 | 0.328 |
| habit / NBP | 7–17 / 215–275 | 0.310 [0.223, 0.398] | 0.656 [0.634, 0.678] | 0.206 [0.128, 0.284] | 0.641 [0.622, 0.660] | 0.215 | 0.267 | 0.213 | 0.214 |
| balanced / NBS | 282–334 / 910–999 | 0.203 [0.186, 0.221] | 0.452 [0.443, 0.460] | 0.372 [0.329, 0.414] | 0.374 [0.346, 0.402] | 0.160 | 0.215 | 0.332 | 0.342 |
| balanced / NBP | 15–28 / 219–268 | 0.154 [0.107, 0.201] | 0.392 [0.363, 0.421] | 0.132 [0.090, 0.174] | 0.357 [0.330, 0.385] | 0.128 | 0.166 | 0.139 | 0.132 |
| **transition / NBS** | 361–431 / 913–1000 | 0.213 [0.195, 0.231] | 0.244 [0.237, 0.252] | **0.868 [0.848, 0.889]** | 0.206 [0.200, 0.212] | 0.527 | 0.355 | 0.865 | 0.869 |
| transition / NBP | 17–33 / 222–270 | 0.099 [0.052, 0.147] | 0.248 [0.225, 0.271] | 0.205 [0.092, 0.318] | 0.218 [0.202, 0.235] | 0.143 | 0.138 | 0.160 | 0.170 |

Within-session cases are 14–30% of NBS cases and only 3–12% of NBP cases. NBP strata
rest on 7–33 cases per seed and are very imprecise.

### 7.4 7-day window NDCG@1 (secondary)

| Model | habit / NBS | habit / NBP | balanced / NBS | balanced / NBP | transition / NBS | transition / NBP |
|---|---|---|---|---|---|---|
| Random floor (mean) | 0.325 | 0.223 | 0.446 | 0.248 | 0.522 | 0.261 |
| Popularity | 0.346 [0.331, 0.361] | 0.243 [0.221, 0.266] | 0.509 [0.486, 0.531] | 0.287 [0.260, 0.315] | **0.634** [0.618, 0.650] | 0.334 [0.312, 0.356] |
| Personal Frequency | 0.898 [0.887, 0.910] | 0.750 [0.727, 0.773] | **0.748** [0.729, 0.766] | **0.508** [0.472, 0.544] | 0.597 [0.571, 0.623] | 0.361 [0.339, 0.383] |
| ItemKNN self | **0.899** [0.888, 0.909] | **0.752** [0.730, 0.773] | 0.704 [0.680, 0.728] | 0.505 [0.470, 0.541] | 0.621 [0.594, 0.647] | **0.364** [0.342, 0.386] |
| MF | 0.650 [0.631, 0.668] | 0.506 [0.466, 0.546] | 0.580 [0.552, 0.609] | 0.359 [0.325, 0.394] | 0.602 [0.576, 0.629] | 0.297 [0.272, 0.321] |
| SASRec Full | 0.868 [0.857, 0.878] | 0.696 [0.672, 0.720] | 0.660 [0.630, 0.691] | 0.462 [0.423, 0.502] | 0.576 [0.561, 0.590] | 0.318 [0.301, 0.335] |
| MeanPool | 0.885 [0.869, 0.901] | 0.699 [0.672, 0.727] | 0.727 [0.708, 0.746] | 0.477 [0.445, 0.509] | 0.631 [0.616, 0.647] | 0.353 [0.333, 0.372] |

Full − PF (window, NDCG@1) is negative in all six groups: −0.031, −0.054, −0.087,
−0.046, −0.021 and −0.043, with every across-seed interval below 0. ItemKNN no-self,
NoPosition, Shuffled and SessionBoundary are in the frozen report.

### 7.5 Cost

Local laptop GPU (CUDA, FP32, deterministic). These are indicative wall-clock numbers
on a machine with noticeable timing noise; they are not production estimates.

| Variant | Task | train s | epochs (best) | test ms / case | peak GPU MiB | parameters | checkpoint |
|---|---|---|---|---|---|---|---|
| SASRec Full | NBS | 13.6 | 14.3 (9.3) | 0.009 | 110 | 77,440 | 318 KiB |
| NoPosition | NBS | 11.8 | 13.6 (8.6) | 0.009 | 110 | 76,160 | 313 KiB |
| Shuffled | NBS | 14.6 | 15.3 (10.3) | 0.011 | 110 | 77,440 | 318 KiB |
| MeanPool | NBS | 3.4 | 11.8 (6.8) | 0.002 | 77 | 512 | 5 KiB |
| SessionBoundary | NBS | 12.9 | 14.0 (9.0) | 0.010 | 110 | 77,568 | 319 KiB |
| SASRec Full | NBP | 2.5 | 14.0 (9.0) | 0.025 | 79 | 77,504 | 319 KiB |
| MeanPool | NBP | 0.7 | 12.2 (7.2) | 0.006 | 68 | 576 | 5 KiB |

Per-dataset mean over profiles. Classical training time was not instrumented in
Phase 3. Qualitatively: PF has **no learned parameters** and scores by counting;
ItemKNN stores an item×item similarity matrix (5×5 or 6×6); MF stores a few factors
per item. SASRec Full has about 77k parameters and needs a GPU-friendly training loop
to reach the same NBS accuracy level that PF reaches with zero parameters in the habit
and balanced regimes.

### 7.6 Cold start and exclusions

Excluded cases are counted and kept separate from the evaluated population.

| Profile | Next-event test target events (10 seeds) | Excluded: no earlier same-task target | Tied timestamps merged | Single-target sessions | Window test customer-task pairs with targets | Excluded: no history |
|---|---|---|---|---|---|---|
| habit | 13,979 | 295 | 0 | 10,387 | 5,260 | 135 |
| balanced | 15,507 | 260 | 0 | 9,072 | 5,287 | 104 |
| transition | 16,590 | 284 | 0 | 8,530 | 5,295 | 143 |

Evaluated test cases per seed: next-event NBS 1,062–1,419, NBP 222–301; window NBS
342–375, NBP 144–177. In the window protocol, 1–17 cases per seed have history made
only of browse events and no first target. SASRec gives them zero scores, so the shared
popularity tie-break ranks them.

## 8. Interpretation

**RQ1: partially yes.** SASRec Full beats PF decisively in exactly one of six groups,
transition/NBS: +0.164 NDCG@1 [+0.152, +0.176], with all 10 seeds' within-seed CIs
above 0. It also beats ItemKNN self, MF and Popularity there by 0.136–0.156. In the
other five groups PF has the higher mean. The across-seed interval of Full − PF is
entirely below 0 for habit/NBS, habit/NBP and balanced/NBP. For balanced/NBS it
includes 0 ([−0.034, +0.001]), so SASRec is *not* shown to be worse there. For
transition/NBP it also includes 0 ([−0.037, +0.007]).

**RQ2: the advantage is regime- and task-specific.** It appears only where in-session
transitions are frequent *and* there are enough of them to learn from. Transition/NBP
has strong transitions by construction, but only 17–33 within-session cases per seed
and about 400 training cases per dataset. There, SASRec does not separate from PF,
consistent with transition/NBP being the one group where the frozen configuration
ranked 11th of 30 during HPO.

**RQ3: the gain is concentrated in within-session targets.** On transition/NBS,
SASRec Full reaches 0.868 NDCG@1 on within-session targets vs 0.213 for PF, while on
cross-session targets it is slightly *below* PF (0.206 vs 0.244). In balanced/NBS the
same split appears (within 0.372 vs 0.203, cross 0.374 vs 0.452), and the larger
cross-session share makes the overall difference slightly negative. The supported
statement is:

> SASRec captures short-term behavioral transitions within an active interaction
> session, while Personal Frequency remains a competitive model for recurring behavior
> across sessions.

It is **not** evidence that SASRec better models customers' overall habits.

**RQ4: order and attention matter; explicit positions and boundary flags did not add
measurable value.**
- Chronological order: on transition/NBS, Full beats Shuffled by +0.100 [+0.089,
  +0.110] (10/10 seeds). Within-session NDCG@1 falls from 0.868 to 0.527 when training
  order is destroyed.
- Attention: Full beats MeanPool by +0.135 [+0.124, +0.146] (10/10).
- Explicit positional embeddings: Full − NoPosition is between −0.002 and +0.001 in all
  groups. Within the tested architecture and synthetic benchmark, explicit learned
  positional embeddings provide negligible incremental benefit over causal attention.
  Because NoPosition keeps the causal mask, this does not show that positional
  information is unimportant in general.
- Session boundaries: SessionBoundary − Full is between −0.004 and +0.000. The tested
  boundary embedding adds nothing measurable. That does not show session information
  is useless for other model designs.
- Where there is no transition signal (habit), the ablations barely differ, and the
  simplest neural aggregator (MeanPool) is closest to PF.

**Window protocol.** Next-event gains do not carry over to the 7-day window. Even on
transition/NBS, Popularity (0.634) and PF (0.597) beat SASRec Full (0.576), and
Full − PF is below 0 in every group. This matches the objective mismatch: a model
trained to predict the next item is used to rank a week-long set. It shows a limit of
this transfer, not that SASRec is unsuitable for longer horizons, for which it was not
optimized.

**Cost.** Where SASRec is not better than PF (five of six groups), it adds roughly 77k
parameters, GPU training, and early-stopping infrastructure for no accuracy gain. On
transition/NBS it buys +0.16 NDCG@1 for about 14 s of GPU training per dataset and a
320 KiB model.

## 9. Limitations

1. **Synthetic data only.** Profiles, transition maps and every number here are
   properties of an invented generator. There is no evidence these effects or their
   magnitudes hold for real customers.
2. **One training seed.** SASRec initialization variation was not measured. The 10
   final seeds are 10 datasets, not 10 independent model initializations.
3. **Small NBP training sets** (about 400 train cases per dataset). Early stopping
   often selected epochs 1–3 on NBP during HPO, a sign of fast overfitting.
4. **Temporal history-length shift.** Training sequences are shorter than validation
   and test sequences (Section 3).
5. **Within-session concentration.** The headline gain lives in the within-session
   subset (14–30% of NBS cases). Within-session NBP strata are tiny.
6. **Controlled ablations.** All variants use the Full configuration. They measure
   component contributions under that configuration, not each architecture's tuned
   optimum.
7. **Cold start excluded** (Section 7.6). Cases without prior same-task history are
   counted, not evaluated.
8. **Local environment.** One laptop GPU. Timings are indicative, and cross-device
   bitwise reproducibility is not claimed: CPU and GPU, or different GPUs, produce
   different weights.
9. **HPO optimism.** Validation was used for both early stopping and configuration
   selection. Final numbers come from untouched test splits of fresh seeds.
10. **Exposure diversity** measures catalog exposure evenness, not per-customer
    semantic diversity.

## 10. Reproducibility

- Integrity checks run before any generation, training or writing:
  - pinned manifests;
  - the seed namespace;
  - frozen configuration hash;
  - pipeline versions;
  - the runtime fingerprint (GPU, CUDA/cuDNN/torch build, CPU threads, determinism
    settings).

  A run on a different runtime, or with a changed configuration, is refused.
- Determinism: `torch.use_deterministic_algorithms(True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off, FP32, explicit attention (no fused
  kernels). Repeated training on the same device and thread count is bitwise
  identical, and this is tested on CPU and CUDA.

```bash
pip install -r requirements.txt -r requirements-seq-cuda.txt   # or requirements-seq-cpu.txt
python -m bri.experiments.integrity                             # verify all manifests and frozen reports
python -m bri.seq.runtime --device auto                         # device, determinism, versions
pytest -q && pytest -q -m cuda                                  # default suite + CUDA validation
python -m bri.seq.prepare                                       # tuning datasets + length report
python -m bri.seq.hpo report                                    # HPO report from the SQLite study (search is frozen)
python -m bri.experiments.phase3 run --output-dir artifacts/phase3/repro \
    --profiles transition_dominant --seeds 20261134              # subset reproduction (new directory)
```

Re-running HPO is refused because the configuration is frozen. A final-run
reproduction must use a new output directory. Writing into `benchmarks/phase3/` or
replacing the aggregate report is refused.

**Reproduction audit (Checkpoint 7, same machine and runtime):**

| Check | Result |
|---|---|
| Phase 3 subset reproduction: `transition_dominant` / seed 20261134, full gate sequence, new output directory | Source, processed and prep fingerprints identical. Every metric, paired difference and stratum identical. Classical next-event report byte-identical. **All 10 SASRec checkpoints have identical state_dict sha256** (5 variants × 2 tasks), and the same best epochs and validation scores |
| HPO: frozen trial-21 configuration retrained on 2 tuning datasets | Validation NDCG@3 and best epoch identical to the values stored in the study |
| `python -m bri.seq.prepare` | 9 tuning datasets regenerated; source and processed fingerprints equal those in the frozen-configuration manifest |
| Phase 2.2 / Phase 2.3 / benchmark-v1 | Byte-identical, byte-identical (one next-event dataset), and identical selections, trials, test metrics, CIs and random floor |
| Refusals | Subset run into the aggregate directory, run into `benchmarks/phase3/`, and `hpo run` after the freeze are all refused; the aggregate report is unchanged |

The first final-run attempt was stopped by gate 1 before any dataset was written. In
the full test session, some test modules initialized CUDA before
`configure_determinism()`. The test session now sets `CUBLAS_WORKSPACE_CONFIG` first,
and the log of that refused attempt is kept.

## 11. Future work (not implemented): Phase 3.1 hybrid

The RQ3 split suggests combining PF, strong on recurring cross-session behavior, with
SASRec, strong on within-session transitions, through score fusion or a gate.

Methodological constraints for any such work:
- The gate may only read information available before the recommendation, for
  example: events in the active session so far, time since the last event, steps in
  the current session, item recency and frequency.
- It must **never** read the evaluator's within/cross-session label, which is derived
  from the target.
- It must be developed and evaluated on **new seeds**. The Phase 3 final test splits
  are opened and locked; they are a baseline, not a validation set.
