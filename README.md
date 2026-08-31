# Uncertainty-Aware Predictive ISCAI for Joint Beam Management and Adaptive Driving Beam Control

Uncertainty-aware predictive extension of a phase-coded FMCW (PC-FMCW) automotive integrated sensing, communication, and illumination (ISCAI) framework.

This repository investigates how uncertain sensing observations and uncertain future road-user motion can be propagated into coordinated communication beam-management and adaptive-driving-beam (ADB) decisions. Real multi-agent traffic motion from the Waymo Open Motion Dataset (WOMD) is used to represent scene dynamics, while the sensing layer generates PC-FMCW-like observations conditioned on causal scene geometry. These observations are then used for classical tracking/forecasting, probabilistic trajectory prediction, receiver-aware communication control, predictive ADB, and joint communication–illumination evaluation.

> **Important scope note:** WOMD-LiDAR is not FMCW and does not provide measured point-wise Doppler. The main sensing interface in this repository is therefore based on real WOMD motion with simulated PC-FMCW-like observations and explicit measurement uncertainty. Geometry-derived radial velocity is not claimed as measured FMCW Doppler.

## Research pipeline

```text
WOMD causal traffic scene
        |
        v
PC-FMCW-like observation generation
(range, radial velocity, angles, SNR, measurement covariance,
noise, missed detections, false alarms)
        |
        v
Association / tracking / classical forecasting
(CV, CA, CTRV, EKF, IMM, Multidimensional Hough Transform)
        |
        v
Probabilistic multi-agent trajectory prediction
(calibrated Gaussian GRU; GMM extension evaluated)
        |
        +------------------------------+
        |                              |
        v                              v
Receiver-aware angular posterior       Future actor occupancy
        |                              |
        v                              v
Adaptive communication beam control    Predictive class-aware ADB
        |                              |
        +---------------+--------------+
                        |
                        v
              Joint ISCAI evaluation
```

A central design principle is to keep **measurement uncertainty** and **predictive uncertainty** distinct. The Stage-2 measurement covariance describes uncertainty in the current sensing observations, while Stage 4 learns a separate predictive covariance describing uncertainty in future road-user motion.

## Repository structure

| Directory | Purpose |
| --- | --- |
| `iscai_data_prep/` | Dataset manifests and deterministic data-selection utilities. |
| `iscai_stage0/` | Environment, dataset-layout, schema, coordinate, map, and LiDAR alignment audits. |
| `iscai_stage1/` | Causal WOMD preprocessing, canonical ego/headlamp geometry, map context, receiver geometry, and causal annotation/LiDAR interfaces. |
| `iscai_stage2/` | PC-FMCW-like observations, sensing SNR, measurement covariance, Gaussian corruption, missed detections, and false alarms/clutter. |
| `iscai_stage3/` | Classical tracking and forecasting baselines: CV, CA, CTRV, EKF, IMM, and Multidimensional Hough Transform. |
| `iscai_stage4/` | Deterministic and probabilistic GRU trajectory prediction, calibration, GMM extension, uncertainty analysis, and formal evaluation. |
| `iscai_stage5/` | Receiver-aware angular posterior, adaptive Top-K beam management, and optical pointing/gain/SNR/BER/effective-rate evaluation. |
| `iscai_stage6/` | Predictive class-aware ADB and causal future-occupancy control. |
| `iscai_stage7/` | Joint communication–illumination evaluation using the frozen shared predictive pipeline. |
| `part_a_reference/` | Frozen reference to the previous PC-FMCW ISCAI implementation used for continuity and baseline checks. |
| `audits.zip`, `manifests.zip` | Archived audit and manifest material used by the staged reproducibility workflow. |

## Stage overview

### Stage 0 — Environment and dataset audit

Stage 0 verifies the actual WOMD environment before scientific processing begins. It audits dataset layout, available files, coordinate/schema assumptions, map content, and available LiDAR context. Its role is reproducibility and infrastructure validation rather than model performance.

### Stage 1 — Causal scene representation

Stage 1 converts WOMD annotations into a canonical causal representation suitable for downstream sensing and prediction. Global WOMD coordinates are transformed into ego/headlamp-centered geometry, future information is excluded from algorithm inputs, benchmark metadata is not used as sensing information, and available LiDAR context is treated explicitly as non-FMCW context.

The resulting representation contains causal actor histories, ego/headlamp geometry, static/causal dynamic map information, receiver geometry, and optional causal LiDAR-derived context.

### Stage 2 — PC-FMCW-like sensing observations

Stage 2 converts clean causal geometry into a realistic algorithm-facing observation interface. Ideal geometric quantities are constructed from the WOMD motion and then degraded using an explicit sensing model. The stage supports:

- range, geometry-derived radial velocity, azimuth, and elevation;
- configurable sensing SNR;
- measurement covariance;
- Gaussian measurement corruption;
- SNR-dependent missed detections;
- unlabeled detection frames;
- false alarms / clutter;
- Monte Carlo checks of measurement-covariance consistency.

The frozen Stage-2 semantics explicitly state that WOMD-LiDAR is not measured FMCW and that measurement uncertainty is separate from downstream predictive uncertainty.

### Stage 3 — Classical tracking and forecasting baselines

Stage 3 evaluates classical methods using the common Stage-2 observation interface. The frozen baseline set contains:

- Constant Velocity (CV),
- Constant Acceleration (CA),
- Constant Turn Rate and Velocity (CTRV),
- Extended Kalman filtering,
- IMM with CV / CA / CTRV modes,
- Multidimensional Hough Transform.

Here **MHT means Multidimensional Hough Transform**, not Multiple Hypothesis Tracking.

The main track-based mode uses deterministic gated greedy nearest-neighbor association, while the Hough baseline consumes raw unlabeled detections directly. Measurement covariance from Stage 2 is preserved and used by the estimators rather than replaced by a fixed arbitrary uncertainty model.

A frozen formal validation set contains 120 scenarios, stratified into cyclist, pedestrian-without-cyclist, and vehicle-only groups. The Stage-3 closure records reproducible ADE/FDE and track-reconstruction metrics for all classical baselines.

### Stage 4 — Probabilistic trajectory prediction

Stage 4 introduces the learned predictive component. Training data are deterministically split at scenario level into independent fit, development, and calibration partitions. Neural samples are built from causal Stage-2/Stage-3 information and can include multi-agent context, map context, and the propagated measurement covariance.

The stage first trains a deterministic multi-agent GRU and then a Gaussian probabilistic GRU that predicts a future mean and full 3D symmetric-positive-definite covariance at horizons of 0.1, 0.3, 0.5, and 1.0 s.

The learned covariance is calibrated on a separate calibration partition using per-horizon covariance scaling. A trajectory-level GMM extension is also evaluated, but the frozen downstream posterior is the **calibrated Gaussian GRU**.

On the frozen 120-scenario formal evaluation, the calibrated Gaussian predictor records an ADE of approximately **1.985 m** and improves the formal calibration metrics relative to the uncalibrated posterior. It outperforms the frozen CV, CA, CTRV, Kalman/EKF, and IMM baselines in the recorded Stage-4 comparison; the repository does not claim that it outperforms every classical baseline.

### Stage 5 — Receiver-aware beam management

Stage 5 transforms the predicted future road-user distribution into a receiver-aware angular posterior and then maps that posterior into communication-beam probabilities. Receiver geometry is explicitly distinguished from the actor centroid, with centroid, known-offset, and uncertain-offset cases supported.

The communication controller includes fixed/classical beam baselines and adaptive Top-K policies that select the smallest beam set covering a requested probability mass. Codebooks of 16, 32, and 64 beams are evaluated. The optical link chain connects pointing error to beam gain, received power, SNR, DPSK BER, and effective rate, while also accounting for probing overhead, latency, fallback, and reacquisition behavior.

The frozen Stage-5 acceptance audit passes all mandatory communication-control and reproducibility gates on the formal 120-scenario cohort.

### Stage 6 — Predictive class-aware ADB

Stage 6 extends the previous reactive ADB logic from current deterministic regions toward future probabilistic occupancy while preserving the existing ADB semantics. Full 3D actor boxes are projected into the illumination-control representation, and future occupancy is used to construct predictive control regions.

The frozen Stage-6 method is a causal, budgeted predictive residual ADB controller. The final Stage-6 closure records preservation of pedestrian and cyclist visibility, controlled over-masking, and a reduction in vehicle shadow violations for the accepted Stage-6 configuration. No future ground truth or oracle controller input is allowed.

### Stage 7 — Joint communication and illumination evaluation

Stage 7 evaluates the communication and illumination branches together without retraining or silently modifying the frozen Stage-4, Stage-5, or Stage-6 models. The evaluation code covers common-posterior consistency, beam reliability and probing overhead, ADB glare protection, pedestrian/cyclist visibility, over-masking, latency, uncertainty sweeps, codebook sweeps, failure-case slices, and statistical confidence intervals.

## Scientific boundaries

This repository intentionally avoids several claims that are not supported by the data or implementation:

- WOMD-LiDAR is **not** treated as FMCW sensing.
- Geometry-derived radial velocity is **not** presented as measured FMCW Doppler.
- Dataset identifiers and benchmark-selection metadata are not numerical predictor features.
- Future WOMD states are used only as labels/evaluator truth, not causal model inputs.
- Measurement uncertainty and predictive uncertainty are modeled separately.
- Predictive ADB, trajectory forecasting, and uncertainty-aware beam management are not presented as individually new ideas; the research focus is their integration within a common PC-FMCW-conditioned automotive ISCAI pipeline.

## Reproducibility philosophy

The repository follows a staged freeze-and-audit workflow. Major blocks generate machine-readable reports, implementation hashes, deterministic manifests, leakage checks, regression-test counts, and frozen handoff artifacts. Formal evaluation sets are isolated from model training, calibration, and model selection.

Because the repository contains several staged experimental environments, individual stage directories should be consulted for exact commands, dependencies, reports, and frozen artifacts rather than assuming a single root-level one-command installation.

## Data

The motion component is based on the **Waymo Open Motion Dataset (WOMD)**. Access to the original Waymo data is not redistributed by this repository. Users must obtain the dataset independently and comply with the applicable Waymo dataset terms and licensing conditions.

## Relationship to the previous PC-FMCW ISCAI framework

The repository is an extension of a previous PC-FMCW automotive ISCAI implementation. The previous framework provides the sensing/communication/illumination reference and the Multidimensional Hough-based tracking context. This repository adds a causal uncertainty-aware predictive layer and evaluates how a shared future-motion posterior can drive both communication beam management and predictive ADB.

In compact form:

```text
Previous framework:
PC-FMCW sensing -> current-state tracking -> reactive communication / ADB

This repository:
PC-FMCW-conditioned uncertain sensing
    -> causal tracking
    -> calibrated probabilistic future motion
    -> receiver-aware communication control
    -> predictive ADB
    -> joint evaluation
```

## Status

Stages 0–6 have frozen completion/acceptance artifacts in the repository. Stage 7 contains the joint-evaluation implementation and analysis workflow. Reported numerical values should be interpreted within the scope of the corresponding frozen stage reports and should not be generalized beyond the documented evaluation protocol.
