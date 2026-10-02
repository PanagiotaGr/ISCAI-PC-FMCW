# ISCAI Stage 6 — Predictive Class-Aware Adaptive Driving Beam

Stage 6 uses the calibrated future-motion posterior to construct probabilistic
future occupancy and to evaluate predictive, class-aware Adaptive Driving Beam
(ADB) control.

## What this stage does

The stage starts from the frozen reactive ADB baseline and adds bounded
predictive intervention based on future actor occupancy. Vehicles,
pedestrians, and cyclists are treated separately in the illumination policy,
while the evaluation tracks shadow-zone, over-masking, and visibility
constraints.

The repository intentionally preserves the complete scientific lineage:
an earlier V1 protocol produced a negative over-masking result, while the later
versioned V3 method was evaluated with the final common Experiment-5 evaluator.

## Main responsibilities

- bind the Stage-4 future posterior to future actor geometry;
- construct stochastic future boxes and occupancy probabilities;
- implement parameterized predictive masks;
- define class-aware ADB policy behavior;
- compare predictive and reactive ADB under the same evaluator;
- freeze side-effect/visibility acceptance gates;
- preserve failed and superseded development outcomes for provenance;
- seal the final Experiment-5 evaluation and reproducibility artifacts.

## Final frozen result

Under the final Stage6-V3 / Experiment-5 common evaluator:

- vehicle shadow-zone violation: **0.0948125 → 0.0730832**
- over-masking area: **0.1925967 → 0.2016184**
- pedestrian visibility proxy: **0.6188124 → 0.6170516**
- cyclist visibility proxy: **0.7438343 → 0.7399150**

All four frozen final acceptance gates pass.

The historical V1 over-masking result remains preserved as a failure and is
not rewritten by the later V3 result.

## Directory structure

- `configs/` — Stage-6 policy and evaluation contracts.
- `src/iscai_stage6/` — occupancy and ADB support code.
- `scripts/` — geometry, occupancy, policy, evaluator, repair, and final
  execution scripts.
- `reports/` — block reports, historical V1 closure, V2/V3 development
  records, Experiment-5 evidence, and final certificates.
- `artifacts/` — frozen Stage-6 artifacts.
- `tests/` — regression tests.

## Scientific boundary

The ADB quantities are modeled/normalized evaluation quantities rather than
measured photometric hardware outputs or a physiologically calibrated human
response model.

**Implementation status: COMPLETE. Final V3 gate: PASS. Historical V1 gate:
FAIL (preserved provenance).**
