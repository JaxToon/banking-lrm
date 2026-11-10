# Phase 2.3 — Benchmark Integrity and Multi-Seed Behavioral Robustness

> **SYNTHETIC-DATA EXPERIMENT.** Every number below reflects patterns deliberately
> embedded by the synthetic generator's invented behavior profiles. None of it is
> evidence about real customers, real banking behavior, or real-world recommendation
> effectiveness.

Frozen results: [`benchmarks/phase2.3/robustness_report.md`](../benchmarks/phase2.3/robustness_report.md)
(full per-seed tables) and `robustness_report.json`. Locked datasets and configuration:
`benchmarks/manifests/phase2.3.json`.

## 1. Question

Phase 2.1 and 2.2 found that Personal Frequency (PF) is a strong baseline, and that
ItemKNN's advantage came mostly from self-similarity, meaning repeat use by the same
customer. Both results came from one generator regime (0.2.0) and one seed. Phase 2.3
asks: **how do the five classical baselines rank when the synthetic behavior is
habit-driven, balanced, or transition-driven, and how stable is the ranking across
independent seeds?**

## 2. What was built

### 2.1 Benchmark integrity (`bri.experiments.integrity`)

| Control | What it pins |
|---|---|
| Composite source fingerprint | sha256 of customers, catalog, consent, events + one composite hash |
| Processed fingerprints | interactions, train examples, eval cases, eval targets |
| Prep fingerprint | prep metadata: cutoffs, split parameters, candidates, consent counts, checksums |
| Canonical config hash | suite id, model grids, key comparisons, evaluation protocol (Ks, selection, n_boot, seeds, confidence, next-event version) |
| Versions | generator, prep, runner, next-event protocol |
| Immutable manifests | one per locked benchmark, write-once, file sha256 pinned in code |

Before reading any data, the runner refuses when:
- processed files don't match their prep checksums,
- a registered suite id has different grids or comparisons,
- a locked dataset (matched by events checksum, or by generator identity for an edited
  copy) is run by another suite, with another configuration, or with modified
  source, consent, catalog, processed, or prep data,
- a manifest has been edited, is missing, or isn't registered.

Each scenario has an adversarial test in `tests/test_integrity.py`. The manifests for
`benchmark-v1` and `phase2.2` are **retrospective provenance**: they were written in
Phase 2.3, after the original runs, and are marked `"provenance": "retrospective"`. They
protect the current data, configuration and frozen reports from now on, but they cannot
establish what the original runs used. The supporting evidence is reproduction: the
original suites reproduce the frozen results. Their frozen reports are unchanged, and
their hashes are pinned in the manifests. Integrity metadata goes into a
separate `*.run_manifest.json`, so the old report format stays byte-identical.

Reproducing a *subset* of a locked benchmark (for example one profile and one seed) is
allowed when every dataset's evaluation config matches the registered one. It is still
a reproduction, not a new experiment. The robustness runner checks the locks before it
generates or writes anything. It refuses:
- locked seeds reused with a different population size, window, or split, because
  customer streams don't depend on population size and this would re-expose locked
  test customers;
- `FROZEN` output directories;
- replacing an aggregate report that was produced by a different configuration.

All three cases are tested.

### 2.2 Generator 0.3.0 behavior profiles

Parameters (invented experimental assumptions, see
[03_DATA_DICTIONARY.md](03_DATA_DICTIONARY.md#behavior-profiles-generator-030--invented-experimental-assumptions)):

| Profile | Affinity concentration (svc/prd) | Transition p (svc/prd) |
|---|---|---|
| habit_dominant | 0.15 / 0.15 | 0.10 / 0.05 |
| balanced | 0.6 / 0.6 | 0.40 / 0.30 |
| transition_dominant | 5.0 / 5.0 | 0.85 / 0.75 |

A higher Dirichlet concentration means weaker customer-specific preference. NBS and NBP
each have their own transition map, and transitions never cross tasks. Without
`--profile` the generator is 0.2.0, and both registered 0.2.0 datasets regenerate byte
for byte (tested). For a given seed, customer attributes and consent are the same in
every profile. Session counts differ slightly because Dirichlet sampling uses a
different number of random draws per concentration. An earlier test that required
identical counts was wrong and was replaced by a tolerance test (within 5%).
Byte-level determinism per seed and profile, plus the profile-specific behavior
(concentration and transition orderings), have their own tests.

### 2.3 Next-event protocol (`bri.eval.next_event`)

- One case per eligible *first target* (first target event per task, customer, session,
  and item) in the val or test window, with the item in the candidate set. A repeat of
  the same item within a session is not a new case. The same item in a later session is
  a new case.
- History is the customer's same-task, consent-permitted interactions strictly before
  the target timestamp. The target, tied events, and all later events are excluded.
  Tested directly: deleting all events at or after the cutoff never changes the
  history.
- Timestamp ties merge into one case whose relevant set is all tied items. None
  occurred in the 30 datasets.
- Cold start: cases with fewer than `min_history=1` earlier first targets are excluded
  and counted.
- A session with only one target gives one cross-session case. `within_session` marks
  cases whose previous target was in the same session.
- The signal is fixed to `target`. With `item_event`, the history would include the
  browse events of the journey being predicted (for example the service view seconds
  before the service start). That would measure journey completion, not next-intent
  prediction.
- All five models share the window protocol's fit/select/test core, candidate sets, and
  metric code. The bootstrap resamples **customers** (cluster bootstrap), because one
  customer contributes many cases.

### 2.4 Design of the experiment

- 3 profiles × 10 seeds (20261110–20261119), 500 customers, 30 days, split 16/7/7.
  No seed is shared with a locked benchmark.
- For each dataset: generate, validate, consent-aware split, then window and
  next-event evaluation. Each fits on train, selects on validation NDCG@3, and
  evaluates once on test.
- Models: Popularity, Personal Frequency (mandatory reference), ItemKNN self=True,
  ItemKNN self=False, implicit MF. Grids are the Phase 2.2 grids. Next-event uses only
  the `target`-signal configurations.
- **Two-level statistics.**
  - *Within seed:* paired percentile bootstrap (1,000 resamples), resampling cases in
    the window protocol and customers in next-event. This describes variation inside
    one dataset.
  - *Across seeds:* each seed's test mean, or paired difference, is one observation.
    We report the mean, sample SD, and a Student-t 95% interval over the 10 seeds
    (df = 9). This describes variation between independent synthetic datasets of one
    profile at a fixed size.
  - The two levels are not pooled. No interval is computed across profiles. Profiles
    share seeds, which makes them paired conditions, but they are reported side by
    side, not tested against each other.
- Runtime: about 40 s per dataset, about 22 minutes in total on a laptop CPU.

## 3. Results (test split, across 10 seeds)

Cells are across-seed means. "CI>0 / CI<0" counts the seeds whose within-seed paired CI
excludes 0 in that direction. Brackets are across-seed t-intervals.

### 3.1 NDCG@1, window protocol

| Profile | Task | Random | Popularity | PF | ItemKNN self | ItemKNN no-self | MF |
|---|---|---|---|---|---|---|---|
| habit_dominant | NBS | 0.333 | 0.362 | **0.893** | **0.893** | 0.114 | 0.616 |
| habit_dominant | NBP | 0.209 | 0.245 | **0.751** | **0.755** | 0.087 | 0.517 |
| balanced | NBS | 0.458 | 0.522 | **0.756** | 0.723 | 0.282 | 0.595 |
| balanced | NBP | 0.248 | 0.304 | **0.529** | 0.521 | 0.163 | 0.366 |
| transition_dominant | NBS | 0.523 | **0.628** | 0.601 | 0.621 | 0.484 | 0.599 |
| transition_dominant | NBP | 0.262 | 0.341 | **0.373** | 0.363 | 0.244 | 0.324 |

### 3.2 NDCG@1, next-event protocol

| Profile | Task | Random | Popularity | PF | ItemKNN self | ItemKNN no-self | MF |
|---|---|---|---|---|---|---|---|
| habit_dominant | NBS | 0.202 | 0.213 | **0.650** | **0.650** | 0.037 | 0.408 |
| habit_dominant | NBP | 0.169 | 0.207 | **0.626** | **0.627** | 0.043 | 0.441 |
| balanced | NBS | 0.198 | 0.230 | **0.395** | 0.376 | 0.099 | 0.282 |
| balanced | NBP | 0.166 | 0.208 | **0.386** | **0.386** | 0.102 | 0.242 |
| transition_dominant | NBS | 0.201 | **0.263** | 0.230 | 0.249 | 0.191 | 0.241 |
| transition_dominant | NBP | 0.171 | 0.215 | 0.241 | **0.245** | 0.152 | 0.211 |

### 3.3 Key paired differences (NDCG@1, mean [across-seed 95% t-interval], CI>0 / CI<0)

| Profile | Protocol | Task | ItemKNN self − PF | MF − PF | Popularity − PF | ItemKNN self − no-self |
|---|---|---|---|---|---|---|
| habit | window | NBS | +0.000 [−0.003, +0.004] 0/0 | −0.277 [−0.305, −0.248] 0/10 | −0.531 0/10 | +0.779 10/0 |
| habit | window | NBP | +0.004 [−0.001, +0.009] 0/0 | −0.234 0/10 | −0.506 0/10 | +0.668 10/0 |
| habit | next-event | NBS | +0.000 [−0.003, +0.003] 0/1 | −0.242 0/10 | −0.437 0/10 | +0.613 10/0 |
| habit | next-event | NBP | +0.001 [−0.005, +0.007] 0/0 | −0.185 0/10 | −0.420 0/10 | +0.584 10/0 |
| balanced | window | NBS | −0.032 [−0.052, −0.013] 0/5 | −0.161 0/10 | −0.234 0/10 | +0.441 10/0 |
| balanced | window | NBP | −0.007 [−0.016, +0.001] 0/0 | −0.162 0/9 | −0.224 0/10 | +0.359 10/0 |
| balanced | next-event | NBS | −0.018 [−0.027, −0.010] 0/5 | −0.112 0/10 | −0.165 0/10 | +0.277 10/0 |
| balanced | next-event | NBP | −0.000 [−0.009, +0.009] 0/0 | −0.144 0/9 | −0.178 0/10 | +0.284 10/0 |
| transition | window | NBS | +0.020 [+0.006, +0.034] 2/0 | −0.002 [−0.023, +0.019] 0/0 | +0.028 [+0.016, +0.039] 0/0 | +0.136 10/0 |
| transition | window | NBP | −0.009 [−0.027, +0.008] 0/1 | −0.049 0/1 | −0.032 [−0.068, +0.004] 0/1 | +0.119 8/0 |
| transition | next-event | NBS | +0.019 [+0.009, +0.029] 7/0 | +0.012 [−0.004, +0.027] 4/0 | +0.033 [+0.024, +0.043] 8/0 | +0.058 10/0 |
| transition | next-event | NBP | +0.004 [−0.003, +0.011] 0/0 | −0.030 0/1 | −0.026 [−0.057, +0.004] 0/1 | +0.093 8/0 |

NDCG@3, MAP, precision, and recall for every comparison, plus per-seed values,
coverage, and exposure diversity, are in the frozen report.

### 3.4 Next-event session strata (NDCG@1, across-seed mean: within-session / cross-session)

| Profile | Task | Popularity | PF | ItemKNN self | ItemKNN no-self | MF |
|---|---|---|---|---|---|---|
| habit | NBS | 0.272 / 0.202 | 0.282 / 0.715 | 0.291 / 0.714 | 0.129 / 0.021 | 0.256 / 0.435 |
| balanced | NBS | 0.295 / 0.208 | 0.211 / 0.458 | 0.236 / 0.424 | 0.193 / 0.067 | 0.261 / 0.289 |
| transition | NBS | 0.363 / 0.220 | 0.206 / 0.239 | 0.292 / 0.230 | 0.256 / 0.163 | 0.290 / 0.221 |
| transition | NBP | 0.147 / 0.222 | 0.063 / 0.262 | 0.097 / 0.262 | 0.282 / 0.138 | 0.192 / 0.214 |

### 3.5 Population and cold start

- Window test cases per seed: about 345–385 (NBS) and 142–178 (NBP). Across all 10
  seeds, 109–123 test customer-task pairs per profile were excluded as cold start.
- Next-event test cases per seed: 1,077–1,621 (NBS) and 224–333 (NBP). Across 10 seeds,
  260–283 test target events per profile were excluded as cold start. There were no
  timestamp ties, and 8,706–10,861 test sessions had a single target.
- All 30 datasets passed validation. Fingerprints are in the report and the manifest.

## 4. Findings

1. **PF stays the reference to beat whenever personal habit exists.** In
   habit_dominant and balanced, PF is the best model or tied for best on both tasks,
   both protocols, and all 10 seeds. MF and Popularity are below PF in every seed
   (0–1 exceptions where the within-seed CI includes 0).
2. **ItemKNN self=True is essentially PF.** In the habit regime the difference is
   ≤ 0.004 NDCG@1, and no seed's CI excludes 0. In the balanced regime, ItemKNN self is
   *slightly worse* than PF on NBS: −0.032 window, −0.018 next-event, with across-seed
   intervals excluding 0 and 5 of 10 seeds individually below. Co-usage adds noise
   rather than value when habits drive behavior. This confirms the Phase 2.2 ablation
   across 20 more datasets.
3. **Self-similarity is always the main ingredient of ItemKNN.** self − no-self is
   positive in all 3 profiles × 2 protocols × 2 tasks (8–10 of 10 seeds per cell).
   ItemKNN no-self is at or below the random floor in the habit and balanced regimes.
4. **When personal preference is weak (transition_dominant), PF's advantage
   disappears.** On NBS, Popularity (+0.028 window, +0.033 next-event) and ItemKNN self
   (+0.020, +0.019) are slightly *ahead* of PF, with across-seed intervals excluding 0.
   Even so, every model sits only 0.03–0.11 NDCG@1 above the random floor. On NBP, no
   model is reliably separated from PF.
5. **No classical baseline captures the transition signal.** In transition_dominant,
   85% (NBS) and 75% (NBP) of in-session follow-ups are generated by a deterministic
   map. Yet within-session next-event NDCG@1 is at most 0.36 (NBS, Popularity) and
   0.28 (NBP, ItemKNN no-self), because these models ignore order: a history count
   cannot say what the customer did *last*. This is the measurable gap a sequential
   model would need to close. Phase 2.3 implements no such model.
6. **Uncertainty.** Across-seed SDs of model means are small (typically 0.01–0.03
   NDCG@1). The within-seed CI sign counts show that single-seed conclusions agree with
   the 10-seed conclusion for large effects. For effects of about 0.02, only about half
   of single seeds detect them. That is a concrete reason not to draw small-effect
   conclusions from one seed.

## 5. Limitations

- Everything is synthetic. The profiles are invented to contrast regimes. Their
  parameters, transition maps, and the resulting rankings are not claims about any
  real population.
- Across-seed intervals cover only datasets from the same profile at 500 customers ×
  30 days with a 16/7/7 split and the same generator. They don't cover other sizes,
  windows, profiles, or generator versions.
- Profiles share seeds, so customer attributes and consent are identical across
  profiles. Profile comparisons are descriptive, with no interval across profiles.
- Cold start is excluded (counted, not evaluated).
- The next-event protocol fixes the signal to `target`, so models cannot use browse
  events from earlier journeys in this protocol.
- With 5–6 candidates per task, K = 3 covers half the catalog. NDCG@1 is the more
  discriminating metric.
- `exposure_diversity` measures how evenly recommendations spread across the catalog,
  not per-customer semantic diversity.
- No sequential, neural, or gradient-boosted models were trained (out of scope).

## 6. Reproduction

```bash
python -m bri.experiments.integrity
python -m bri.experiments.robustness --data-root data/phase2_3 --output-dir artifacts/phase2_3
```

The second command regenerates all 30 datasets and reproduces the frozen summary
(spot-checked: per-dataset window and next-event reports reproduce byte for byte). Any
change to configuration, data, or grids on these seeds is refused.
