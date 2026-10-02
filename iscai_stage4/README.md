# ISCAI Stage 4 — Probabilistic Trajectory Forecasting

Stage 4 builds the calibrated probabilistic future-motion representation used
by the downstream communication and illumination stages.

## What this stage does

Using the causal inputs inherited from the earlier stages, Stage 4 trains and
evaluates deterministic and probabilistic trajectory predictors. The main
downstream interface is a calibrated Gaussian GRU that provides a future
trajectory mean together with predictive covariance.

The stage also evaluates a deterministic GRU and a trajectory-level GMM as
comparators/extensions.

## Main responsibilities

- freeze the training/validation split and neural sample contract;
- train the deterministic GRU baseline;
- train the Gaussian probabilistic GRU;
- calibrate predictive uncertainty;
- evaluate the GMM extension;
- run ablation studies;
- perform the frozen formal evaluation;
- verify reproducibility and close the stage.

## Frozen result summary

The formal evaluation contains **120 scenarios**.

Reported formal ADE values are:

- calibrated Gaussian GRU: **1.454 m**
- trajectory-level GMM expected ADE: **1.450 m**
- deterministic GRU: **1.537 m**

Calibration reduces macro ECE from **0.1438** to **0.0806**. The calibrated
Gaussian GRU remains the frozen downstream posterior because it provides the
mean/covariance interface required by later stages.

## Directory structure

- `configs/` — training/evaluation contracts.
- `src/iscai_stage4/` — forecasting and data-interface code.
- `scripts/` — training, calibration, evaluation, audit, repair, and freeze
  scripts.
- `reports/` — block-level reports, formal evaluation, reproducibility, and
  final closure artifacts.
- `artifacts/` — frozen Stage-4 artifacts.
- `tests/` — regression tests.

## Scientific boundary

Future WOMD truth is evaluator/supervision information and is not an online
predictor input. Measurement covariance and predictive covariance remain
distinct quantities.

**Status: COMPLETE / FROZEN.**
