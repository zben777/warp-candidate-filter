# Transposed-Layout Experiments

These variants changed the candidate input from the mainline B-major contract:

```text
[group][B][element]
```

to a lane-major representation:

```text
[group][lane][B]
```

The benchmark generated that representation directly and did not include a
layout-conversion cost. The variants are therefore retained for historical
analysis but excluded from the default build, benchmark, and main performance
table.

| Archived source | Experiment | RTX 4090 time |
|---|---|---:|
| `v6_lane_major.cu` | Direct `int4` load | 3.020 ms |
| `v7_lane_major_cp_async.cu` | `cp.async` B prefetch | 3.018 ms |
| `v8_lane_major_pipeline.cu` | Two-group `cp.async` pipeline | 3.039 ms |

The current mainline `v6` and `v7` reproduce the direct-vector and `cp.async`
experiments while preserving the original B-major input contract.
