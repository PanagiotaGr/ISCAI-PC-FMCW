# Uncertainty-Aware Predictive ISCAI for Joint Beam Management and Adaptive Driving Beam Control

A research implementation of a **predictive, uncertainty-aware extension of a phase-coded FMCW (PC-FMCW) automotive Integrated Sensing, Communication, and Illumination (ISCAI) framework**.

The project studies how uncertainty in the current sensing state and uncertainty in future road-user motion can be propagated into **joint communication-beam management** and **predictive Adaptive Driving Beam (ADB)** control. Realistic scene dynamics are taken from the **Waymo Open Motion Dataset (WOMD)** and WOMD-LiDAR context, while the PC-FMCW sensing interface is generated from the real trajectories through a physics-inspired observation model.

The central research idea is simple but important:

> **Part A estimates the present and reacts; Part B estimates uncertainty, predicts the future, and acts proactively.**

The same calibrated future-motion posterior is reused downstream for receiver-aware beam selection and predictive illumination, so communication and ADB decisions are driven by a common probabilistic scene representation rather than by independent deterministic heuristics.

---

## 1. Research motivation

The previous PC-FMCW ISCAI framework couples sensing, communication, and Adaptive Driving Beam illumination, but its tracking and ADB evaluation is primarily based on synthetic motion and simulated scenes. This repository replaces that less realistic scene-level component with a data-driven, causal pipeline based on real traffic trajectories, real actor interactions, ego motion, map context, and available LiDAR history from WOMD.

The goal is **not** to build a generic motion-forecasting project. Motion prediction is used as an intermediate uncertainty representation for two downstream control problems:

1. **adaptive directional beam management**, and
2. **predictive, class-aware ADB control**.

The project therefore focuses on the complete chain

```text
real traffic dynamics
      -> sensing observations + measurement uncertainty
      -> causal tracking / forecasting
      -> calibrated future trajectory distribution
      -> receiver / angular posterior
      -> adaptive beam probing + predictive ADB
      -> joint communication–illumination evaluation
```

---

## 2. Scientific scope and an important limitation

### WOMD-LiDAR is not FMCW

A critical boundary of this repository is that **WOMD-LiDAR does not provide measured FMCW Doppler or point-wise radial velocity**. It provides real scene geometry and calibrated LiDAR context, but it is not a PC-FMCW sensing dataset.

Accordingly, this repository does **not** claim that WOMD-LiDAR supplies real FMCW radial-velocity measurements.

The scientific formulation used here is instead:

> **real scene dynamics + real non-FMCW LiDAR context + simulated PC-FMCW observations**

Range, bearing, and geometry-derived radial velocity are computed from real WOMD actor/ego motion and then passed through a configurable PC-FMCW-like observation model with SNR-dependent noise, covariance, missed detections, and false alarms.

This makes the experimental setup a **real-trajectory, PC-FMCW sensor-in-the-loop evaluation**, not a real-PC-FMCW measurement campaign.

### Measurement uncertainty is not predictive uncertainty

The repository explicitly separates two different uncertainty sources:

- **measurement uncertainty**: uncertainty in the current sensing observations, represented by Stage-2 measurement covariance;
- **predictive uncertainty**: uncertainty in future actor motion, including multiple possible maneuvers, interactions, and long-horizon ambiguity.

The second is learned by the probabilistic predictor and is not treated as a synonym for the first.

---

## 3. End-to-end architecture

```text
WOMD / WOMD-LiDAR causal history
(actor tracks, ego pose, map, optional LiDAR context)
                    |
                    v
       ego/headlamp-centric representation
                    |
                    v
       PC-FMCW-like observation generator
(range, radial velocity, bearing, SNR, covariance,
 noise, missed detections, false alarms)
                    |
                    v
       association / tracking / baselines
(CV, CA, CTRV, EKF, IMM, Multidimensional Hough)
                    |
                    v
       probabilistic multi-agent forecasting
(calibrated Gaussian GRU; GMM extension evaluated)
                    |
          +---------+----------+
          |                    |
          v                    v
 receiver-aware future     future actor occupancy
 angular posterior             posterior
          |                    |
          v                    v
 adaptive Top-K beam       predictive class-aware
 management               Adaptive Driving Beam
          |                    |
          +---------+----------+
                    |
                    v
        joint ISCAI system evaluation
```

The shared posterior is the key interface between perception/prediction and control.

---

## 4. Dataset role

The project uses the **Waymo Open Motion Dataset (WOMD)** for real multi-agent traffic dynamics and map context. WOMD provides actor trajectories for vehicles, pedestrians, and cyclists together with ego motion and road context. WOMD-LiDAR adds synchronized LiDAR history for the observed portion of the forecasting window.

The LiDAR information is useful for, for example:

- actor geometry;
- point-density confidence;
- visibility / occlusion proxies;
- intensity and elongation statistics;
- occupancy or BEV extensions;
- realistic clutter and partial observations;
- optional future sensor-to-track extensions.

It is **not** used as Doppler ground truth.

The repository does not redistribute the original Waymo data. Users must obtain the dataset separately and comply with the applicable Waymo dataset license and attribution requirements.

---

## 5. Coordinate and receiver modeling

All causal actor states are transformed into an ego/headlamp-centered coordinate frame before downstream sensing and control.

The transmitter location is configurable and can represent, for example, the headlamp midpoint or another fixed communication-module extrinsic.

For communication control, the target vehicle centroid is not automatically assumed to be the receiver location. Stage 5 supports:

- **centroid baseline**;
- **known receiver offset**;
- **uncertain receiver offset**.

This distinction matters because beam management should ultimately act on the predicted receiver position, not merely on the predicted vehicle-box center.

---

## 6. Stage-by-stage implementation

### Stage 0 — Environment and dataset audit

Stage 0 establishes a reproducible dataset and software foundation before any modeling begins. It audits:

- dataset release/layout;
- available train/validation/test files;
- schema and coordinate conventions;
- actor and map fields;
- LiDAR availability and decompression;
- scenario counts and split integrity;
- class frequencies and track lengths;
- available causal metadata.

This stage exists to prevent silent assumptions about the actual WOMD version or data layout.

---

### Stage 1 — Causal scene representation

Stage 1 converts WOMD information into the canonical algorithm-facing scene representation.

Main responsibilities include:

- extracting actor histories and ego motion;
- transforming global coordinates to ego/headlamp coordinates;
- enforcing past-only information flow;
- separating labels from inputs;
- extracting map context;
- representing receiver geometry;
- exposing optional causal LiDAR-derived context.

A central anti-leakage rule is that future WOMD states are used only as labels/evaluator truth and are never passed into the online predictor or controller.

---

### Stage 2 — PC-FMCW-like sensing interface

Stage 2 converts the clean causal scene geometry into noisy sensing observations suitable for the tracking and forecasting stages.

Supported quantities include:

- range;
- geometry-derived radial velocity;
- azimuth and elevation from scene/perception geometry;
- configurable sensing SNR;
- SNR-/geometry-dependent measurement covariance;
- Gaussian corruption;
- missed detections;
- unlabeled detection frames;
- false alarms / clutter;
- Monte Carlo covariance-consistency checks.

The angular information is not claimed to arise automatically from the monostatic Range–Doppler waveform. It comes from the scene geometry or an explicitly declared angular-sensing model.

The preferred main experimental mode is the **track-based PC-FMCW observation mode**, in which real WOMD trajectories are converted into realistic noisy PC-FMCW-like measurements rather than given directly to the predictor as perfect states.

---

### Stage 3 — Classical tracking and forecasting baselines

Stage 3 places classical estimators behind the same Stage-2 observation interface so that comparisons are scientifically fair.

The frozen baseline set includes:

- Constant Velocity (CV);
- Constant Acceleration (CA);
- Constant Turn Rate and Velocity (CTRV);
- Kalman / Extended Kalman filtering;
- IMM using CV / CA / CTRV motion models;
- Multidimensional Hough Transform.

> In this repository, **MHT means Multidimensional Hough Transform**, not Multiple Hypothesis Tracking.

The standard track-based branch uses deterministic gated greedy nearest-neighbor association, while the Hough baseline can consume unlabeled detections directly. Measurement covariance is propagated rather than silently replaced by a fixed hand-tuned noise matrix.

A frozen formal validation cohort contains **120 scenarios**, stratified into cyclist, pedestrian-without-cyclist, and vehicle-only scene groups.

---

### Stage 4 — Probabilistic trajectory prediction

Stage 4 introduces the learned uncertainty-aware forecaster.

Training data are split at **scenario level** into independent fit, development, and calibration subsets. Neural samples are constructed only from causal Stage-2/Stage-3 information and may include:

- actor history;
- multi-agent context;
- map context;
- measurement covariance;
- optional LiDAR-derived actor or scene features.

The implemented progression includes:

1. deterministic GRU;
2. Gaussian probabilistic GRU;
3. covariance calibration;
4. trajectory-level GMM extension.

The Gaussian predictor outputs a future mean together with a full positive-definite 3D covariance for the main short horizons:

- 0.1 s;
- 0.3 s;
- 0.5 s;
- 1.0 s.

The covariance is calibrated on a separate calibration partition using per-horizon covariance scaling.

The repository also evaluates the trajectory-level GMM extension for multi-modality, but the frozen downstream posterior used by later stages is the **calibrated Gaussian GRU**.

On the frozen 120-scenario formal evaluation, the calibrated Gaussian predictor records an ADE of approximately **1.985 m** and improves calibration metrics relative to the uncalibrated posterior. In the recorded Stage-4 comparison it outperforms the frozen CV, CA, CTRV, Kalman/EKF, and IMM baselines; the repository does not claim universal dominance over every possible classical or neural baseline.

---

### Stage 5 — Receiver-aware adaptive beam management

Stage 5 propagates the future trajectory posterior into a future **receiver-position / angular posterior**.

The controller then maps that posterior into a directional codebook and chooses an adaptive beam set.

Implemented concepts include:

- receiver-aware geometry rather than centroid-only geometry;
- Monte-Carlo / distribution-based angular propagation;
- 16-, 32-, and 64-beam codebooks;
- beam probability computation;
- fixed Top-1 / Top-3 / Top-5 baselines;
- geometry-nearest-beam baseline;
- previous-beam persistence;
- adaptive Top-K selection;
- fallback / reacquisition logic;
- latency-aware evaluation;
- optical link-budget evaluation.

The adaptive policy chooses the **smallest beam set whose cumulative predicted probability mass reaches a requested target**, with the main frozen formal acceptance using **q = 0.95**.

The communication evaluation does not stop at beam-index accuracy. Pointing error is propagated through the optical chain:

```text
pointing error
    -> beam / optical gain
    -> received power
    -> SNR
    -> DPSK BER
    -> effective communication rate
```

Beam-probing overhead is incorporated into the effective-rate analysis.

The frozen Stage-5 acceptance audit passes the mandatory communication-control, performance-evidence, handoff, and reproducibility gates. Blockage-aware Top-K is treated as an optional extension rather than a mandatory acceptance requirement.

---

### Stage 6 — Predictive class-aware Adaptive Driving Beam control

Stage 6 replaces purely reactive current-region dimming with **future probabilistic occupancy control**.

Predicted actor geometry is projected into the headlamp control representation using full actor boxes rather than only box centroids. The controller preserves the existing ADB semantics while adding a predictive residual component.

The class-aware policy distinguishes the safety objectives of different road users:

#### Vehicles

- stronger glare protection;
- uncertainty-aware angular margins;
- increased protection under larger closing speed / uncertainty;
- windshield/mirror-relevant shadowing semantics.

#### Pedestrians

- avoid full blackout;
- preserve body and road visibility;
- use partial dimming or face-height protection;
- maintain an appropriate minimum illumination floor;
- account for larger lateral uncertainty.

#### Cyclists

- preserve rider/bicycle visibility;
- use larger lateral maneuver margins;
- treat crossing trajectories explicitly;
- balance glare protection against loss of detectability.

The frozen accepted method is a **causal, budgeted predictive residual ADB controller**. The final Stage-6 closure records class-aware-policy compliance together with preserved pedestrian/cyclist visibility, controlled over-masking, and reduced vehicle shadow violations for the accepted configuration. Future ground truth and oracle controller inputs are not allowed.

---

### Stage 7 — Joint communication and illumination evaluation

Stage 7 is the common evaluation layer for the frozen upstream components.

It is designed to test whether the same predictive posterior can support both downstream branches without retraining or silently modifying the Stage-4, Stage-5, or Stage-6 models.

The analysis covers:

- shared-posterior consistency;
- beam reliability and probability coverage;
- probing overhead;
- beam switching and reacquisition behavior;
- ADB glare protection;
- pedestrian/cyclist visibility;
- over-masking;
- latency;
- uncertainty sweeps;
- codebook sweeps;
- failure-case slices;
- confidence intervals.

The current repository contains the Stage-7 joint-evaluation implementation and handoff/analysis workflow. Its minimal handoff artifact explicitly records that Stage-7 formal outcomes have not yet been read or computed in that handoff snapshot, so this README intentionally avoids inventing final Stage-7 numerical conclusions.

---

## 7. Experimental modes

The project distinguishes three evaluation levels so that dataset annotations, realistic observations, and optional raw-sensor processing are not conflated.

### Oracle-track mode

Perfect historical WOMD actor states are provided up to the anchor time. Future states remain labels only. This is an upper-bound experiment for the forecasting/beam/ADB blocks rather than a fully realistic online system.

### Track-based PC-FMCW observation mode — main mode

Real WOMD tracks generate range, bearing, radial velocity, PC-FMCW measurement noise, missed detections, false alarms, and uncertainty. The predictor receives these noisy observations rather than perfect actor states.

### Sensor-to-track mode — optional realism extension

A future extension can place a LiDAR detector/tracker in front of forecasting, for example:

```text
raw LiDAR
  -> 3D detection
  -> association / tracking
  -> probabilistic forecasting
  -> beam + ADB control
```

This is intentionally optional and is not required for the first complete trajectory-to-control pipeline.

---

## 8. Evaluation metrics

### Trajectory and uncertainty metrics

The evaluation supports metrics such as:

- ADE / FDE;
- minADE / minFDE where applicable;
- miss rate;
- angular MAE / RMSE;
- radial and velocity error;
- heading error;
- negative log-likelihood;
- Brier score;
- calibration error;
- empirical confidence-region coverage.

Results can be sliced by actor class, horizon, distance, maneuver type, visibility/occlusion, point density, and measurement-uncertainty level.

### Beam-management metrics

Beam evaluation includes:

- Top-1 / Top-K hit rate;
- average selected K;
- probability coverage;
- probing overhead and overhead reduction;
- beam-gain loss;
- received-power loss;
- SNR loss;
- DPSK BER;
- effective-rate loss;
- outage probability;
- beam-switching rate;
- reacquisition latency;
- reliability–overhead trade-offs.

### ADB metrics

Because WOMD does not provide true ADB commands, ADB evaluation uses a **constructed oracle reference**, not measured ground truth.

Metrics include:

- predictive mask IoU against the constructed reference;
- vehicle shadow-zone violations;
- glare-risk exposure;
- over-masking;
- road-illumination retention;
- pedestrian visibility proxy;
- cyclist visibility proxy;
- false dimming;
- temporal smoothness / flicker;
- actuation latency.

---

## 9. Repository structure

| Directory | Purpose |
| --- | --- |
| `iscai_data_prep/` | Dataset manifests and deterministic data-selection utilities. |
| `iscai_stage0/` | Environment, dataset-layout, schema, coordinate, map, and LiDAR alignment audits. |
| `iscai_stage1/` | Causal WOMD preprocessing, ego/headlamp geometry, map context, receiver geometry, and causal annotation/LiDAR interfaces. |
| `iscai_stage2/` | PC-FMCW-like observations, sensing SNR, measurement covariance, corruption, missed detections, and false alarms/clutter. |
| `iscai_stage3/` | Classical tracking and forecasting baselines: CV, CA, CTRV, EKF, IMM, and Multidimensional Hough Transform. |
| `iscai_stage4/` | Deterministic/probabilistic GRU forecasting, calibration, GMM extension, uncertainty analysis, and formal evaluation. |
| `iscai_stage5/` | Receiver-aware angular posterior, adaptive Top-K beam control, and optical pointing/gain/SNR/BER/effective-rate evaluation. |
| `iscai_stage6/` | Predictive class-aware ADB and causal future-occupancy control. |
| `iscai_stage7/` | Joint communication–illumination evaluation using the frozen shared predictive pipeline. |
| `part_a_reference/` | Frozen reference to the previous PC-FMCW ISCAI implementation. |
| `audits.zip`, `manifests.zip` | Archived audit and manifest material used by the staged reproducibility workflow. |

---

## 10. Reproducibility and leakage prevention

The repository follows a staged **freeze-and-audit** workflow rather than a single monolithic training script.

Major blocks produce, where applicable:

- deterministic manifests;
- machine-readable reports;
- leakage checks;
- implementation hashes;
- regression-test counts;
- frozen upstream/downstream handoff artifacts;
- fixed evaluation cohorts;
- calibration partitions separate from model fitting;
- scenario-level split discipline.

Important leakage rules include:

- no future states as model inputs;
- no future validity masks as causal features;
- no future LiDAR;
- no train/validation leakage through overlapping windows;
- no validation/test normalization statistics during training;
- no benchmark-selection metadata as numerical sensing features.

Because each stage was developed and frozen independently, **individual stage directories are the authoritative source for exact commands, dependencies, configuration files, frozen reports, and audit outputs**. A single root-level one-command installation is intentionally not assumed.

---

## 11. Relationship to Part A

The repository is an extension of the previous PC-FMCW automotive ISCAI implementation.

### Previous framework

```text
PC-FMCW / DPSK sensing
      -> coherent processing
      -> Range–Doppler / CFAR
      -> current-state tracking
      -> reactive communication / ADB
```

### This repository

```text
PC-FMCW-conditioned uncertain sensing
      -> causal tracking
      -> calibrated probabilistic future motion
      -> receiver-aware angular posterior
      -> adaptive communication beam control
      -> predictive class-aware ADB
      -> joint evaluation
```

The intent is therefore continuity rather than replacement: Part A contributes the sensing/communication/illumination reference and legacy Hough context; Part B adds realistic traffic dynamics, explicit uncertainty propagation, forecasting, and proactive control.

---

## 12. Scientific claims the repository intentionally does not make

The project deliberately avoids over-claiming.

It does **not** claim that:

- WOMD-LiDAR is FMCW;
- geometry-derived radial velocity is measured FMCW Doppler;
- the monostatic PC-FMCW waveform alone estimates full 3D bearing;
- WOMD contains real optical communication beam labels;
- WOMD contains measured ADB ground-truth commands;
- the constructed ADB oracle is measured ground truth;
- predictive ADB by itself is novel;
- trajectory forecasting by itself is novel;
- uncertainty-aware beam management by itself is novel;
- using a Transformer alone constitutes the research contribution.

The research contribution is the **integration and calibration of these components in a common PC-FMCW-conditioned automotive ISCAI pipeline**, with a shared future-motion posterior driving both communication and illumination decisions.

---

## 13. Main research contribution

The strongest contribution of the project is the combination of:

1. real multi-agent vehicle/VRU trajectories;
2. PC-FMCW physics-grounded measurement uncertainty;
3. explicit separation of measurement and predictive uncertainty;
4. calibrated probabilistic future motion;
5. receiver-aware future angular uncertainty;
6. adaptive resource-constrained Top-K beam probing;
7. predictive class-aware ADB;
8. a shared posterior for communication and illumination;
9. mapping beam-pointing uncertainty to optical SNR, DPSK BER, and effective rate;
10. a staged, leakage-aware reproducibility workflow.

A compact description of the research direction is:

> **We develop a real-traffic, PC-FMCW-conditioned vehicular ISCAI framework in which calibrated multi-agent trajectory uncertainty is transformed into a shared future receiver/angular occupancy representation that jointly controls adaptive directional beam probing and predictive class-aware ADB illumination.**

---

## 14. Status

- **Stages 0–6:** implemented with frozen completion / acceptance artifacts in the repository.
- **Stage 7:** joint-evaluation implementation and handoff/analysis workflow are present; final formal Stage-7 outcome values are not asserted here unless backed by the corresponding frozen report.
- **Advanced extensions:** blockage prediction, LiDAR BEV encoders, sensor-to-track integration, graph/Transformer variants, conformal beam sets, ray tracing, and external beam-data validation remain optional research directions.

Reported numerical values in this README should be interpreted only within the scope of the corresponding frozen stage reports and documented evaluation protocol.

---

## 15. Suggested citation / project description

For a report, thesis, or project summary, the repository can be described as:

> A staged uncertainty-aware automotive ISCAI framework that uses real WOMD traffic dynamics to generate causal PC-FMCW-like sensing observations, predicts calibrated future multi-agent motion, and converts the resulting shared posterior into receiver-aware adaptive communication beams and predictive class-aware Adaptive Driving Beam control.
