# Profiling Artifacts

This directory contains Nsight Compute reports and selected SASS snapshots
captured on an NVIDIA RTX 4090 during kernel development.

The `*_4b_ncu.ncu-rep` reports use four candidate lists per reference list.
For example, open a report with Nsight Compute UI:

```bash
ncu-ui profiles/v5_4b_ncu.ncu-rep
```

The reports are architecture-specific measurements, not portable performance
predictions. Re-profile on the target GPU before drawing architecture-level
conclusions.

Reports for the retired lane-major variants are stored under
`archive/transposed-layout/profiles/` so they cannot be mistaken for profiles
of the mainline `v6` and `v7`.
