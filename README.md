# Warp Candidate Filter

Standalone CUDA microbenchmarks for filtering fixed-size candidate lists at
warp granularity. The project explores how data layout, search strategy,
instruction-level parallelism, output coalescing, vectorized loads, and
`cp.async` affect the same filtering contract on an NVIDIA RTX 4090.

## Problem

Each independent group contains:

- one reference list `A[32]`;
- four candidate lists `B0[32] ... B3[32]` by default;
- one warp processing the 32 lanes of each list.

For every candidate list, the kernel removes values already present in `A`,
compacts the survivors, and emits their count and checksum. Every executable
initializes a deterministic input, runs the timed kernel, and validates all
outputs on the host.

```text
                 +--> filter B0[32] --> count + checksum
A[32] -- sort ---+--> filter B1[32] --> count + checksum
       once      +--> filter B2[32] --> count + checksum
                 +--> filter B3[32] --> count + checksum
```

The default benchmark processes 4,194,304 groups, or 16,777,216 A-B pairs.

## Kernel Variants

| Version | Main idea |
|---|---|
| `v0` | Register + shuffle brute-force baseline |
| `v0_shared` | Shared-memory broadcast brute-force baseline |
| `v1` | Sort `A` once, then binary-search each candidate |
| `v1.5` | Register-gather compaction experiment |
| `v2` | Two-way binary-search ILP |
| `v3` | Fixed-32 specialized binary search |
| `v4` | Fixed-32 search with two-way ILP |
| `v5` | Block-coalesced output stores |
| `v6` | Four 8-lane subwarps with direct 128-bit candidate loads |
| `v7` | `v6` mapping with 16-byte `cp.async` candidate prefetch |

`v7` is the recommended implementation. It preserves the original input
contract while matching the best measured performance of the layout-changing
experiments.

### Input layouts

Every mainline version consumes the same B-major layout:

```text
[group][B][element]
```

No mainline timing requires or excludes a layout-conversion step. Earlier
lane-major experiments are retained under `archive/transposed-layout/` and are
not part of the default build or benchmark.

## Reference Results

All measurements use an RTX 4090, `NUM_B_LISTS=4`, `-O3 -arch=sm_89`, and
report zero validation errors. Times are means over five benchmark runs.

| Version | Mean time | Logical effective BW | Speedup vs. `v0` |
|---|---:|---:|---:|
| `v0` | 4.208 ms | 669.8 GB/s | 1.00x |
| `v0_shared` | 4.154 ms | 678.5 GB/s | 1.01x |
| `v1` | 3.031 ms | 929.4 GB/s | 1.39x |
| `v1.5` | 3.147 ms | 895.7 GB/s | 1.34x |
| `v2` | 3.030 ms | 930.1 GB/s | 1.39x |
| `v3` | 3.038 ms | 927.6 GB/s | 1.39x |
| `v4` | 3.038 ms | 927.6 GB/s | 1.39x |
| `v5` | 3.025 ms | 931.7 GB/s | 1.39x |
| `v6` | 3.019 ms | 933.5 GB/s | 1.39x |
| `v7` | **3.018 ms** | **934.0 GB/s** | **1.39x** |

The executable's `Effective BW` field is logical traffic divided by kernel
time; it is useful for comparing variants but is not a hardware-counter
measurement. An archived lane-major predecessor with the same search workload
measured approximately 932 GB/s DRAM throughput and 94.9% of peak sustained
throughput in Nsight Compute. Its report remains in the archive and is not
presented as a profile of the mainline `v7`.

The later variants reach the RTX 4090 DRAM bandwidth ceiling, compressing their
timing differences. `v7` improves kernel time by 28.3% over `v0`, or 1.39x.
Relative to direct-load `v6`, asynchronous B prefetch improves time by
approximately 0.05%.

## Requirements

- Linux
- NVIDIA RTX 4090
- CUDA Toolkit with `nvcc`
- GNU Make and Bash

The reference environment uses CUDA 12.2 and targets compute capability 8.9.

## Build

Build every variant:

```bash
make all
```

Build or run one variant:

```bash
make v7
make run VERSION=v7
```

Build products are written to `build/sm_89/`.

## Benchmark

Build all variants and run each executable five times:

```bash
make benchmark RUNS=5
```

Select a GPU with the standard CUDA environment variable:

```bash
CUDA_VISIBLE_DEVICES=1 make benchmark RUNS=5
```

The script prints a Markdown table containing mean kernel time, logical
effective bandwidth, and correctness status.

For stable comparisons, keep application clocks, power limits, GPU temperature,
and background workload consistent. Randomizing version order is useful when
testing GPUs whose memory clocks may thermally throttle.

## Repository Layout

```text
.
|-- src/                 CUDA benchmark variants
|-- profiles/            Nsight Compute reports and SASS snapshots
|-- archive/             Layout-changing experiments excluded from mainline
|-- scripts/             Benchmark automation
|-- artifacts/           Local historical binaries (ignored by Git)
|-- Makefile
`-- README.md
```

Nsight Compute reports can be opened with `ncu-ui` or imported into a newer
compatible Nsight Compute installation. Generated executables are intentionally
not tracked.
