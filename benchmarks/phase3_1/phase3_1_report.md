# Phase 3.1 final evaluation

> **SYNTHETIC-DATA EXPERIMENT: metrics reflect patterns deliberately embedded by the synthetic generator. They are not evidence of real customer behavior or of real-world recommendation effectiveness.**

- Static hybrid alpha (pinned): {'nbp': 0.1, 'nbs': 0.6}; init seeds [1, 2, 3].
- Brackets: across-seed Student-t 95% intervals (next-event NDCG@1).

## Preregistered criteria

| Criterion | Group | Difference | Met |
|---|---|---|---|
| RQ1 hybrid - PF > -0.01 | habit_dominant/nbs | -0.0020 [-0.0067, +0.0028] | True |
| RQ1 hybrid - PF > -0.01 | habit_dominant/nbp | -0.0069 [-0.0180, +0.0042] | False |
| RQ1 hybrid - PF > -0.01 | balanced/nbs | +0.0102 [+0.0029, +0.0174] | True |
| RQ1 hybrid - PF > -0.01 | balanced/nbp | -0.0005 [-0.0107, +0.0097] | False |
| RQ2 hybrid - PF > 0 | transition_dominant/nbs | +0.1683 [+0.1577, +0.1789] | True |
| RQ2 hybrid - SASRec > -0.01 | transition_dominant/nbs | -0.0020 [-0.0036, -0.0005] | |
| RQ3 | all | not evaluable under the preregistered protocol | - |

Outcome: trade-off, not overall success. RQ3: no gated configuration was selectable on tuning validation (phase3_1-config: gated not selected under the preregistered rule, subset and C unavailable for both tasks); by the owner's Checkpoint 6 decision there is no OOF cross-fitting, gate fitting or gated evaluation on final data. This is not evidence that adaptive fusion is worse than static fusion.

## Models (next-event NDCG@1)

| Group | PF | SASRec Full | Static hybrid | Oracle (diagnostic) |
|---|---|---|---|---|
| habit_dominant/nbs | 0.6429 | 0.6347 | 0.6409 | 0.6448 |
| habit_dominant/nbp | 0.6152 | 0.5912 | 0.6082 | 0.6101 |
| balanced/nbs | 0.3805 | 0.3747 | 0.3907 | 0.4280 |
| balanced/nbp | 0.3798 | 0.3537 | 0.3793 | 0.3807 |
| transition_dominant/nbs | 0.2343 | 0.4046 | 0.4026 | 0.4319 |
| transition_dominant/nbp | 0.2400 | 0.2240 | 0.2326 | 0.2446 |

## Cost (RQ4)

- SASRec fits: 180; training 1384.696 s in total.
- Sum of per-dataset wall times: 1511.135 s (excludes gates).
- Peak PyTorch-allocated GPU memory, max over fits: 110.246 MiB.
- Amortized batch test inference (batch 1024), ms/case: {'next_event': 0.0174, 'window': 0.02907}; not online latency.
- Whole run, from file timestamps (not runner-instrumented): 1686.2 s, gates included.
- Definitions: see `cost.definitions` in the JSON report.
