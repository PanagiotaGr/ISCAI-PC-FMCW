# ISCAI Stage 1 — Causal Scene Representation

Stage 1 converts the frozen Stage-0 WOMD/WOMD-LiDAR corpus into the causal
scene representation used by the sensing and prediction pipeline.

## What this stage does

The stage builds an upstream representation of the traffic scene using only
information available up to the current time. It integrates actor state,
causal map context, ego/headlamp-centric geometry, and aligned LiDAR context
needed by later stages.

The frozen artifact semantics are:

`causal_womd_annotation_upstream`

and the Stage-1 representation is explicitly marked as
`sensor_realistic = false`.

Sensor-realistic PC-FMCW-like observations and measurement covariance are
introduced later in Stage 2.

## Main responsibilities

- adapt real WOMD actor tracks into the project representation;
- construct causal road/map context;
- align actor and LiDAR information;
- express the scene in the headlamp/ego-centric coordinate system;
- test one-scenario and small deterministic pilots;
- verify that mutation of future WOMD states does not change causal outputs;
- generate visual and 3-D/elevation sanity checks.

Track IDs are retained for bookkeeping/association and are not used as
numerical model features. `tracks_to_predict` and `objects_of_interest`
remain benchmark/evaluation metadata rather than causal receiver-selection
inputs.

## Directory structure

- `src/iscai_stage1/` — actors, geometry, maps, LiDAR, I/O, contracts, and
  artifact utilities.
- `scripts/` — real-scenario, actor, map, LiDAR, causality, and visual gates.
- `reports/stage1a/` — frozen gate and closure reports.
- `artifacts/stage1a/` — manifests and visualizations.
- `tests/` and `tests_lidar/` — regression tests.

## Scientific boundary

Stage 1 represents real scene dynamics and context, but it does not claim
measured PC-FMCW sensing. The PC-FMCW-compatible observation model begins in
Stage 2.

**Status: COMPLETE / FROZEN.**
