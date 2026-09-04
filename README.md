# Uncertainty-Aware Predictive ISCAI for Joint Beam Management and Adaptive Driving Beam Control

A research implementation of a **predictive, uncertainty-aware extension of a phase-coded FMCW (PC-FMCW) automotive Integrated Sensing, Communication, and Illumination (ISCAI) framework**.

The project studies how uncertainty in the current sensing state and uncertainty in future road-user motion can be propagated into **communication beam management** and **predictive Adaptive Driving Beam (ADB)** control. Real traffic dynamics are taken from the **Waymo Open Motion Dataset (WOMD)** and WOMD-LiDAR context, while the PC-FMCW sensing interface is generated from those real trajectories through a physics-grounded observation and uncertainty model.

> **Part A estimates the present and reacts; Part B estimates uncertainty, predicts the future, and acts proactively.**

The central idea is to use one calibrated future-motion posterior as a common interface between sensing/prediction and downstream control: the same probabilistic representation is transformed into a receiver-aware angular posterior for communication and into future actor occupancy for illumination.

---

## 1. Research objective

The previous PC-FMCW ISCAI framework combines sensing, communication and Adaptive Driving Beam illumination, but its tracking and ADB evaluation is primarily based on synthetic trajectories and simulated scenes. This repository replaces that less realistic scene-level block with a causal data-driven pipeline based on real multi-agent traffic motion.

The goal is **not** to build a generic motion-forecasting benchmark. Forecasting is an intermediate uncertainty representation for two control problems:

1. **adaptive directional beam management**, and
2. **predictive, class-aware ADB control**.

The complete research chain is

```text
real WOMD traffic dynamics
        -> PC-FMCW-like sensing observations
        -> measurement uncertainty
        -> causal tracking / forecasting
        -> calibrated future trajectory posterior
        -> receiver / angular posterior + future occupancy
        -> adaptive Top-K beam control + predictive ADB
        -> system-level evaluation
```

---

## 2. Scientific scope and boundaries

### WOMD-LiDAR is not FMCW

A critical boundary of this repository is that **WOMD-LiDAR does not provide measured FMCW Doppler or point-wise radial velocity**. It provides real scene geometry and synchronized LiDAR context, but it is not a PC-FMCW sensing dataset.

The scientific formulation used here is therefore

> **real traffic dynamics + real non-FMCW LiDAR context + simulated PC-FMCW-like observations**

Range, bearing and geometry-derived radial velocity are computed from real WOMD actor/ego motion and passed through a configurable PC-FMCW-like observation model with SNR-dependent uncertainty, corruption, missed detections and false alarms.

This is a **real-trajectory, PC-FMCW sensor-in-the-loop evaluation**, not a real-PC-FMCW measurement campaign.

### Measurement uncertainty is different from predictive uncertainty

Two uncertainty sources are modeled explicitly and separately:

- **measurement uncertainty** describes uncertainty in the current sensing observations and is represented by the Stage-2 measurement covariance;
- **predictive uncertainty** describes uncertainty in future motion, including maneuver ambiguity, interaction effects and multi-modal behavior.

The predictor is conditioned on the first and learns the second.

### Causality and leakage prevention

Future WOMD states are used only as labels/evaluator truth. The causal pipeline does not use future actor states, future validity masks, future LiDAR or benchmark-selection metadata as numerical predictor inputs.

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
(range, radial velocity, angles, SNR, covariance,
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
 receiver-aware future      future actor occupancy
 angular posterior          distribution
          |                    |
          v                    v
 adaptive Top-K beam       predictive class-aware
 management               Adaptive Driving Beam
          |                    |
          +---------+----------+
                    |
                    v
          system-level evaluation
```

---

# 4. Formal results

The values below are taken from the repository's frozen scientific closure/evaluation artifacts. They should be interpreted **within the documented evaluation protocol and frozen cohorts**, not as general claims beyond this experimental setup.

## 4.1 Stage 4 — Probabilistic trajectory forecasting

The frozen Stage-4 formal evaluation contains **120 scenarios**. The primary downstream posterior is the calibrated Gaussian GRU; a deterministic GRU and a trajectory-level GMM are also evaluated.

### Overall formal forecasting results

| Model | Formal ADE |
| --- | ---: |
| Oracle future trajectory | **0.000 m** |
| Trajectory-level GMM, expected ADE | **1.450 m** |
| Calibrated Gaussian GRU | **1.454 m** |
| Deterministic GRU | **1.537 m** |

The GMM gives the lowest non-oracle expected ADE in this frozen formal report, but the **calibrated Gaussian GRU remains the frozen posterior used by downstream stages** because it provides the required mean/covariance interface and calibrated uncertainty representation.

### Calibration

| Metric | Before calibration | After calibration |
| --- | ---: | ---: |
| Macro ECE | 0.1438 | **0.0806** |

Calibration reduces macro ECE by approximately **44% relative to the uncalibrated posterior**. This is important because Stage 5 selects beam sets by integrating predicted probability mass; a poorly calibrated posterior would make a nominal 95% beam set scientifically difficult to interpret.

### Exact common-support comparison against the frozen CV comparator

On the final exact common-support evaluation:

| Horizon | CV ADE | Gaussian ADE | Common samples |
| --- | ---: | ---: | ---: |
| 0.1 s | 1.650 m | **0.614 m** | 331 |
| 0.3 s | 3.863 m | **0.938 m** | 323 |
| 0.5 s | 6.216 m | **1.307 m** | 320 |
| 1.0 s | 11.950 m | **2.335 m** | 312 |
| **Aggregate** | **5.841 m** | **1.285 m** | 332 actors / 1286 events |

On this exact common support, the Gaussian predictor reduces aggregate ADE by about **78% relative to CV**. The gap increases with prediction horizon, which is consistent with the intended use of learned motion context rather than simple constant-velocity propagation.

The Stage-4 closure also confirms the causal experimental contract: no future truth is used as model input, annotated velocity is not the primary input, track IDs are not used numerically, and measurement covariance remains distinct from predictive covariance.

---

## 4.2 Stage 5 — Receiver-aware adaptive Top-K beam management

Stage 5 converts the trajectory posterior into a receiver-aware angular posterior and then chooses the **smallest beam set whose cumulative predicted probability reaches a requested target**. The primary formal acceptance uses **q = 0.95** and the uncertain-receiver-geometry setting.

### Probability coverage at q = 0.95

| Codebook | 0.1 s | 0.3 s | 0.5 s | 1.0 s |
| --- | ---: | ---: | ---: | ---: |
| 16 beams | 100.0% | 100.0% | 100.0% | **98.78%** |
| 32 beams | 100.0% | 100.0% | 100.0% | **98.78%** |
| 64 beams | 100.0% | 100.0% | **98.78%** | **98.78%** |

All frozen coverage tests pass the pre-specified statistical acceptance rule for the requested 95% mass.

### Adaptive probing cost

| Codebook | Mean adaptive probes | Exhaustive probes | Approx. probe reduction |
| --- | ---: | ---: | ---: |
| 16 beams | **2.54** | 16 | **84.1%** |
| 32 beams | **4.17** | 32 | **87.0%** |
| 64 beams | **7.39** | 64 | **88.5%** |

The formal acceptance artifact therefore supports the central Stage-5 claim: **high empirical coverage can be maintained while probing only a small fraction of the full codebook**.

For 16 beams, adaptive probing also passes against the valid fixed Top-K baselines with fixed Top-3 as the best valid comparator; for 32 beams, fixed Top-5 is the valid comparator. At 64 beams, none of the tested fixed Top-1/3/5 alternatives satisfy the required coverage criterion, while the adaptive policy still passes the formal coverage and exhaustive-overhead conditions.

The Stage-5 evaluation additionally propagates pointing decisions through the optical communication chain:

```text
beam pointing
    -> optical / beam gain
    -> received power
    -> SNR
    -> DPSK BER
    -> effective rate
```

This makes the beam controller a communication-system evaluation rather than only a beam-index classification experiment.

**Stage-5 formal status: PASS.**

---

## 4.3 Stage 6 — Predictive class-aware ADB

Stage 6 tests whether the shared probabilistic future occupancy can improve ADB behavior while preserving vehicle/VRU safety constraints.

The implementation successfully satisfies the functional requirements for:

- future 3D box projection;
- probabilistic occupancy masks;
- predictive covariance use;
- vehicle/pedestrian/cyclist class-aware policies;
- continuity with the Part-A reactive illumination model;
- temporal smoothing and actuation handling.

The frozen scientific outcome, however, is deliberately preserved as a **negative result** rather than being post-hoc retuned into a pass.

### Frozen scientific outcome

| Criterion | Outcome |
| --- | --- |
| Vehicle shadow-zone violation | **PASS — strict improvement** |
| Pedestrian visibility | **PASS — non-inferior** |
| Cyclist visibility | **PASS — non-inferior** |
| Over-masking non-inferiority | **FAIL** |

The over-masking result is:

| Quantity | Value |
| --- | ---: |
| Reactive over-masking area | **0.1886** |
| Predictive over-masking area | **0.2652** |
| Predictive − reactive | **+0.0766** |
| Pre-outcome allowed delta | **+0.0200** |
| Excess beyond allowed delta | **+0.0566** |

Thus, the predictive controller reduces vehicle shadow-zone violations and preserves pedestrian/cyclist visibility, but does so with **too much additional masking under the frozen non-inferiority criterion**.

This is scientifically important: the repository does **not** change the threshold after observing the result, does not reinterpret the failure as a pass, and does not use post-outcome retuning. The Stage-6 closure explicitly records the negative result and requires a **new scientific protocol/method version** for a future pass.

**Stage-6 implementation status: COMPLETE.**  
**Stage-6 scientific completion gate: FAIL.**

---

## 4.4 Stage 7 — Joint communication–illumination evaluation

The intended Stage-7 goal is to evaluate communication and illumination jointly using the frozen shared posterior. However, the authoritative Stage-6 closure blocks progression of the current scientific protocol because the Stage-6 over-masking criterion failed.

Therefore, this README does **not** present final Stage-7 system-level outcome numbers for the current protocol.

The repository may contain Stage-7 implementation, handoff, pre-formal and evaluator-development artifacts, but these must not be interpreted as a valid final joint scientific result that supersedes the frozen Stage-6 gate.

---

## 5. Dataset and scene representation

The project uses the **Waymo Open Motion Dataset (WOMD)** for real multi-agent traffic dynamics and map context. Actor classes include vehicles, pedestrians and cyclists. WOMD-LiDAR provides synchronized LiDAR history for the observed part of the forecasting window.

LiDAR context can support:

- actor geometry;
- point-density confidence;
- visibility / occlusion proxies;
- intensity and elongation statistics;
- occupancy / BEV extensions;
- clutter and partial-observation modeling;
- future sensor-to-track extensions.

It is **not** used as FMCW Doppler ground truth.

The repository does not redistribute the original Waymo data. Users must obtain the data independently and comply with the applicable dataset license and attribution requirements.

---

## 6. Coordinate and receiver modeling

All causal actor states are transformed into an ego/headlamp-centered coordinate frame before sensing and control.

The transmitter extrinsic is configurable. For communication, the target vehicle centroid is not automatically treated as the receiver location. Stage 5 supports:

- **centroid baseline**;
- **known receiver offset**;
- **uncertain receiver offset**.

This distinction is important because the communication controller acts on a **receiver posterior**, not merely a vehicle-center posterior.

---

## 7. Stage-by-stage implementation

### Stage 0 — Environment and dataset audit

Audits dataset layout, schema, coordinate conventions, map content, LiDAR availability, splits and reproducibility assumptions before modeling begins.

### Stage 1 — Causal scene representation

Builds actor histories, ego/headlamp transforms, map context, receiver geometry and optional causal LiDAR context while enforcing strict past-only information flow.

### Stage 2 — PC-FMCW-like sensing observations

Generates range, geometry-derived radial velocity, angular information, SNR, measurement covariance, Gaussian corruption, missed detections, unlabeled detections and false alarms/clutter.

### Stage 3 — Classical baselines

Implements CV, CA, CTRV, EKF/Kalman, IMM and Multidimensional Hough Transform behind a common observation interface.

> In this repository, **MHT means Multidimensional Hough Transform**, not Multiple Hypothesis Tracking.

### Stage 4 — Probabilistic trajectory prediction

Implements deterministic GRU, Gaussian probabilistic GRU, covariance calibration and a trajectory-level GMM extension at the main horizons 0.1, 0.3, 0.5 and 1.0 s.

### Stage 5 — Adaptive beam management

Transforms the receiver posterior into beam probabilities over 16-, 32- and 64-beam codebooks and evaluates fixed and adaptive Top-K policies, probing overhead, link quality, BER and effective rate.

### Stage 6 — Predictive class-aware ADB

Projects future actor occupancy into the illumination controller and evaluates vehicle glare protection, pedestrian/cyclist visibility and masking cost. The current frozen protocol produces a scientifically preserved negative result due to over-masking.

### Stage 7 — Joint evaluation

Contains the implementation path for common-posterior communication/illumination analysis, but the current protocol cannot claim final Stage-7 closure because Stage 6 did not satisfy its frozen completion gate.

---

## 8. Experimental modes

The project distinguishes three levels of realism.

### Oracle-track mode

Perfect historical WOMD states are supplied up to the anchor time. Future states remain evaluator truth only. This provides an upper-bound experiment rather than a realistic online sensing system.

### Track-based PC-FMCW observation mode — primary mode

Real WOMD motion generates PC-FMCW-like range/radial-velocity/angular observations, noise, covariance, missed detections and false alarms. The predictor receives these degraded observations rather than perfect current states.

### Sensor-to-track mode — optional extension

A future realism extension can use

```text
raw LiDAR
  -> 3D detector
  -> data association / tracking
  -> probabilistic forecasting
  -> beam + ADB control
```

This is not required for the frozen core trajectory-to-control pipeline.

---

## 9. Evaluation metrics

### Trajectory and uncertainty

- ADE / FDE;
- minADE / minFDE where applicable;
- miss rate;
- angular MAE / RMSE;
- radial, velocity and heading errors;
- negative log-likelihood;
- Brier score;
- calibration error;
- empirical confidence-region coverage.

### Beam management

- Top-1 / Top-K hit rate;
- empirical probability coverage;
- average selected K / probe count;
- probing overhead and overhead reduction;
- beam-gain and received-power loss;
- SNR loss;
- DPSK BER;
- effective-rate loss;
- outage probability;
- beam-switching rate;
- reacquisition latency.

### Adaptive Driving Beam

Because WOMD does not contain measured ADB command ground truth, the ADB evaluation uses constructed references and controller-specific metrics rather than claiming real ADB labels.

Metrics include:

- vehicle shadow-zone violations;
- glare-risk exposure;
- over-masking;
- road illumination retention;
- pedestrian visibility;
- cyclist visibility;
- false dimming;
- temporal smoothness / flicker;
- actuation latency.

---

## 10. Repository structure

| Directory | Purpose |
| --- | --- |
| `iscai_data_prep/` | Dataset manifests and deterministic data-selection utilities. |
| `iscai_stage0/` | Environment, dataset-layout, schema, coordinate, map and LiDAR audits. |
| `iscai_stage1/` | Causal WOMD preprocessing, ego/headlamp geometry, map and receiver representation. |
| `iscai_stage2/` | PC-FMCW-like observations, sensing SNR, covariance, corruption, misses and clutter. |
| `iscai_stage3/` | Classical tracking and forecasting baselines. |
| `iscai_stage4/` | Deterministic/probabilistic GRU forecasting, GMM, calibration and formal evaluation. |
| `iscai_stage5/` | Receiver-aware angular posterior, adaptive Top-K beam control and optical link evaluation. |
| `iscai_stage6/` | Predictive class-aware ADB and frozen scientific closure. |
| `iscai_stage7/` | Joint communication–illumination implementation and evaluation-development artifacts. |
| `part_a_reference/` | Frozen reference to the previous PC-FMCW ISCAI implementation. |
| `audits.zip`, `manifests.zip` | Archived reproducibility material. |

---

## 11. Reproducibility philosophy

The repository follows a staged **freeze-and-audit** workflow. Major stages generate machine-readable reports, deterministic manifests, hashes, leakage checks, regression-test evidence and frozen handoff artifacts.

Important principles are:

- scenario-level split discipline;
- separate fit/development/calibration roles;
- no post-hoc use of formal outcomes for tuning;
- frozen acceptance criteria before outcome interpretation;
- preservation of negative results;
- explicit separation between implementation completion and scientific acceptance.

Individual stage directories and their frozen reports are the authoritative source for exact commands, dependencies, configurations and numerical results.

---

## 12. Relationship to Part A

### Previous framework

```text
PC-FMCW / DPSK sensing
      -> coherent processing
      -> Range-Doppler / CFAR
      -> current-state tracking
      -> reactive communication / ADB
```

### Part B

```text
PC-FMCW-conditioned uncertain sensing
      -> causal tracking
      -> calibrated probabilistic future motion
      -> receiver-aware angular posterior
      -> adaptive communication beam control
      -> predictive class-aware ADB
      -> scientific evaluation
```

Part A provides the sensing/communication/illumination reference and the Hough-based legacy tracking context. Part B adds real traffic dynamics, explicit uncertainty propagation, probabilistic forecasting and proactive downstream control.

---

## 13. Scientific claims intentionally not made

This repository does **not** claim that:

- WOMD-LiDAR is FMCW;
- geometry-derived radial velocity is measured FMCW Doppler;
- the monostatic PC-FMCW waveform alone estimates full 3D bearing;
- WOMD contains real optical communication beam labels;
- WOMD contains measured ADB ground-truth commands;
- the constructed ADB reference is measured ground truth;
- trajectory forecasting alone is the novelty;
- predictive ADB alone is the novelty;
- adaptive Top-K beam management alone is the novelty;
- a Transformer architecture by itself constitutes the contribution;
- the current Stage-6 protocol has passed its scientific completion criterion;
- final Stage-7 joint conclusions exist for the currently frozen failed Stage-6 protocol.

---

## 14. Main research contribution

The project combines:

1. real multi-agent vehicle/VRU motion from WOMD;
2. PC-FMCW physics-grounded measurement uncertainty;
3. explicit separation of measurement and predictive uncertainty;
4. calibrated probabilistic future motion;
5. receiver-aware angular uncertainty;
6. adaptive probability-mass-constrained beam probing;
7. predictive class-aware ADB;
8. a shared posterior for communication and illumination;
9. mapping beam-pointing uncertainty to received power, SNR, DPSK BER and effective rate;
10. a reproducible workflow that preserves both positive and negative scientific results.

A compact project description is:

> **We develop a real-traffic, PC-FMCW-conditioned vehicular ISCAI framework in which calibrated multi-agent trajectory uncertainty is transformed into future receiver/angular and occupancy representations for adaptive directional beam probing and predictive class-aware illumination.**

---

## 15. Current scientific status

| Stage | Status |
| --- | --- |
| Stages 0–3 | Implemented / frozen upstream pipeline |
| Stage 4 | **COMPLETE — formal probabilistic forecasting evaluation frozen** |
| Stage 5 | **PASS — formal adaptive beam-management acceptance** |
| Stage 6 | **IMPLEMENTATION COMPLETE, SCIENTIFIC GATE FAIL — over-masking non-inferiority** |
| Stage 7 | **Not scientifically closed for the current protocol because Stage 6 blocks progression** |

The most important current conclusion is therefore not simply that every downstream component succeeds. Rather, the repository demonstrates a strong probabilistic forecasting and adaptive beam-management result, while the first frozen predictive-ADB protocol exposes a real **safety/utility trade-off between glare reduction and excess masking** that must be addressed in a new Stage-6 method version.

That negative result is part of the scientific contribution: it identifies where uncertainty-aware proactive illumination improves one objective while violating another pre-specified constraint, without changing the evaluation rules after observing the outcome.
