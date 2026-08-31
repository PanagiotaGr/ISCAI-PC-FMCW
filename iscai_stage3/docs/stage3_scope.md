# Stage 3 — Frozen Scope

## Purpose

Classical motion/tracking baselines operating on the frozen Stage-2
PC-FMCW-like observation interface.

## Mandatory baselines

- Constant Velocity (CV)
- Constant Acceleration (CA)
- CTRV
- Kalman filter
- IMM with CV / CA / CTRV modes
- Multidimensional Hough Transform (MHT/Hough)

Here MHT means Multidimensional Hough Transform.
It does NOT mean Multiple Hypothesis Tracking.

## Main information regime

The main fair comparison consumes the same causal Stage-2
algorithm-facing observations.

Inputs may contain:
- range
- radial velocity
- azimuth
- elevation
- measurement covariance R_t
- timestamps

The main mode must support:
- measurement noise
- missed detections
- false alarms / clutter
- gaps

Evaluator truth sidecars are scoring-only and MUST NOT be available
to algorithms.

## Identity / association modes

1. Oracle identity:
   diagnostic upper bound only.

2. Noisy observations with oracle association:
   diagnostic / development mode only.

3. Estimated association:
   main track-based classical mode.

4. Track-before-detect:
   Multidimensional Hough baseline.

No main comparison may give perfect actor IDs to one method while
another method receives only unlabeled detections.

## Causality

Only information available at or before the anchor/current time may
be used by a baseline.

Future WOMD states are evaluator labels only.

## Measurement uncertainty

The complete Stage-2 measurement covariance R_t must remain available
to classical estimators.

Kalman/IMM must use R_t in the measurement update / likelihood.

Measurement uncertainty is not predictive uncertainty.

## Stage boundary

Stage 3 MUST NOT implement:
- neural trajectory predictors
- probabilistic GRU/GMM training
- calibration of learned trajectory distributions
- beam controller
- receiver angular posterior
- adaptive Top-K
- optical BER/effective-rate controller
- predictive ADB
- DeepSense

Those belong to downstream stages.

## Closure criterion

Stage 3 is COMPLETE_FROZEN only if:

1. CV has reproducible metrics.
2. Kalman has reproducible metrics.
3. IMM has reproducible metrics.
4. MHT/Hough consumes the common observation interface.
5. The implementation uses no future information.
6. Evaluator truth does not leak into algorithm inputs.
7. Formal validation scenarios are deterministically selected.
8. Tests, configs, reports and implementation hashes are recorded.

