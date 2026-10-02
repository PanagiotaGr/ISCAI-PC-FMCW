# ISCAI Stage 8 — External DeepSense Measured-Beam Validation

Stage 8 provides an external measured-data validation layer for the
communication beam-selection policy using DeepSense mmWave beam-power data.

## What this stage does

The stage defines the DeepSense data/evaluator contract, prepares the
communication-side feature and label interface, trains the frozen encoder,
calibrates output probabilities, replays the beam-selection policy, and runs
the final measured-beam evaluation.

This stage is intentionally separate from the WOMD/PC-FMCW optical evaluation.

## Main responsibilities

- audit the DeepSense data authority and acquisition boundary;
- freeze the canonical corpus/schema/split;
- define the measured-mmWave evaluator contract;
- bind the Stage-5 beam-policy interface;
- preregister and train the LiDAR encoder;
- calibrate beam probabilities;
- replay the frozen policy before formal scoring;
- run the measured DeepSense formal evaluation;
- analyze statistics, failures, and resource trade-offs;
- produce final reporting/reproducibility artifacts.

## Frozen measured-data result

For the primary `adaptive_topk_q095` policy, the final report contains
**1,030 measured samples**:

- empirical coverage: **96.89%**
- mean selected K: **5.02**
- probing overhead: **7.85%**
- mean measured power loss: **0.00864 dB**
- 1 dB outage rate: **0.291%**
- 3 dB outage rate: **0%**
- 6 dB outage rate: **0%**

## Directory structure

- `configs/` — DeepSense acquisition, model, calibration, policy, and formal
  evaluation contracts.
- `data/` — Stage-8 canonical corpus materialization used by the published
  workflow.
- `src/` — DeepSense LiDAR encoder, beam policy, runtime, and measured-mmWave
  evaluator code.
- `scripts/` — acquisition, training, calibration, replay, formal
  evaluation, statistics, and closure scripts.
- `reports/` — all frozen Stage-8 reports.
- `artifacts/` — external-validation artifacts.

## Scientific boundary

DeepSense validates the communication beam-selection policy using measured
**mmWave** beam powers. It does not validate optical headlamp behavior,
PC-FMCW optical hardware, ADB, glare/human response, traffic criticality,
collision probability, or crash-risk reduction.

**Status: COMPLETE / final reporting reproducibility PASS.**
