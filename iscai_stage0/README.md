# ISCAI Stage 0 — Dataset, Environment, and Protocol Freeze

Stage 0 establishes the reproducible data and execution foundation for the
remaining PC-FMCW ISCAI pipeline.

## What this stage does

The stage audits the execution environment, the WOMD/WOMD-LiDAR dataset
layout, relevant schemas, and the data-access assumptions before any modeling
stage is allowed to run. It also freezes the conventions and scope boundaries
that later stages inherit.

The final Stage-0 closure defines the canonical paired WOMD/WOMD-LiDAR corpus
used by the project:

- training scenarios: **72,085**
- validation scenarios: **44,097**
- total paired scenarios: **116,182**

Dataset manifests and reconciliation artifacts are used to make the selected
corpus traceable and reproducible.

## Main responsibilities

- audit the software/runtime environment;
- inspect WOMD and WOMD-LiDAR layout and schema;
- verify map and lane-state structures;
- establish deterministic scenario accounting;
- validate paired WOMD/WOMD-LiDAR availability;
- freeze dataset/protocol conventions before downstream processing;
- preserve train/validation separation and prevent selection based on future
  labels or evaluation outcomes.

## Directory structure

- `configs/` — Stage-0 configuration.
- `docs/` — implementation and scope log.
- `reports/` — environment, dataset, and closure reports.
- `scripts/` — environment, schema, dataset, and closure audits.
- `src/` — Stage-0 support code.
- `tests/` — regression tests.
- `STAGE0_RUN.md` — first-run audit instructions.

## Scientific boundary

Stage 0 does not perform forecasting, communication-beam selection, or ADB
control. Its role is to establish the frozen data and reproducibility
foundation used by the later stages.

The final closure supersedes earlier provisional notes in the implementation
log where the dataset layout or Stage-0 status had not yet been resolved.

**Status: COMPLETE / FROZEN.**
