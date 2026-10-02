# A Unified Probabilistic Framework for PC-FMCW Beam Selection, Geometric Criticality, and Adaptive Driving Beam Control

A research implementation of a **predictive, uncertainty-aware extension of a phase-coded FMCW (PC-FMCW) automotive Integrated Sensing, Communication, and Illumination (ISCAI) framework**.

The project studies how uncertainty in the current sensing state and uncertainty in future road-user motion can be propagated into **communication beam management** and **predictive Adaptive Driving Beam (ADB)** control. Real traffic dynamics are taken from the **Waymo Open Motion Dataset (WOMD)** and WOMD-LiDAR context, while the PC-FMCW sensing interface is generated from those real trajectories through a physics-grounded observation and uncertainty model.

> **Part A estimates the present and reacts; Part B estimates uncertainty, predicts the future, and acts proactively.**

The central idea is to use one calibrated future-motion posterior as a common interface between sensing/prediction and downstream control: the same probabilistic representation is transformed into a receiver-aware angular posterior for communication, future actor occupancy for illumination, and — in the Stage-9 paper extension — same-sample future traffic criticality for decision-layer resource allocation.

<img width="803" height="825" alt="image" src="https://github.com/user-attachments/assets/266c6416-d980-4e02-88c3-ac1e5e41e6fd" />


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
        -> joint system-level evaluation
        -> external measured-beam validation
        -> Stage-9 shared-posterior decision layer
           (criticality-constrained communication
            + secondary recovery-aware ADB)
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

### Oracle results are evaluation bounds, not deployable predictors

Whenever an **oracle future trajectory** is reported, it uses evaluator-only future ground truth as the reference trajectory. Therefore its trajectory displacement error is identically zero by construction. It is included only as a **non-deployable ideal evaluation bound** and must not be interpreted as a trained forecasting model or an online system result.

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
       frozen joint Stage-7 evaluation
                    |
                    v
      measured DeepSense beam validation
```

---

# 4. Frozen and formal results

The numerical values below are taken from the repository's **final frozen closure/evaluation artifacts** rather than intermediate development logs. Older repair logs and pre-freeze runs remain in the repository for auditability, but they are not treated as the reporting authority when a later frozen closure exists.

Results should be interpreted **within the documented protocol and frozen cohorts**. They are not universal claims beyond this experimental setup.

## 4.1 Stage 3 — Classical tracking and forecasting baselines

<img width="717" height="369" alt="image" src="https://github.com/user-attachments/assets/85c34592-d5ed-44f6-bc45-028ec15db502" />

Stage 3 evaluates classical baselines behind the same frozen Stage-2 observation interface. The final validation cohort contains **120 scenarios**, stratified as 40 cyclist, 40 pedestrian-without-cyclist and 40 vehicle-only scenarios. The final closure records **109/109 regression tests passed** and exact reproducibility of the frozen formal run.

| Method | ADE | FDE | Reconstruction recall |
| --- | ---: | ---: | ---: |
| Constant Acceleration | 30.181 m | 81.641 m | 0.628 |
| Constant Velocity | 6.751 m | 13.783 m | 0.728 |
| CTRV | 4.752 m | 8.146 m | 0.671 |
| IMM (CV/CA/CTRV) | 3.646 m | 7.228 m | 0.584 |
| Kalman / EKF | 3.537 m | 6.756 m | 0.612 |
| **Multidimensional Hough** | **1.615 m** | **2.537 m** | **0.466** |

The Hough baseline has the lowest matched-track displacement error among these Stage-3 methods, but this result must be interpreted together with reconstruction coverage: its recall is lower than several classical track-based baselines. Therefore the repository does **not** claim that Hough universally dominates the other methods based on ADE/FDE alone.

> In this repository, **MHT means Multidimensional Hough Transform**, not Multiple Hypothesis Tracking.

---

## 4.2 Stage 4 — Probabilistic trajectory forecasting

The frozen Stage-4 formal evaluation contains **120 scenarios**. The primary downstream posterior is the calibrated Gaussian GRU; a deterministic GRU and a trajectory-level GMM are also evaluated.

### Overall formal forecasting results

| Model | Formal ADE |
| --- | ---: |
| Oracle future-GT reference — non-deployable evaluation bound | **0.000 m** |
| Trajectory-level GMM, expected ADE | **1.450 m** |
| Calibrated Gaussian GRU | **1.454 m** |
| Deterministic GRU | **1.537 m** |

The oracle value is zero because its trajectory equals evaluator ground truth by definition. It is not an online predictor.

The GMM gives the lowest non-oracle expected ADE in this frozen formal report, but the **calibrated Gaussian GRU remains the frozen posterior used by downstream stages** because it provides the required mean/covariance interface and calibrated uncertainty representation.

### Calibration

| Metric | Before calibration | After calibration |
| --- | ---: | ---: |
| Macro ECE | 0.1438 | **0.0806** |

Calibration reduces macro ECE by approximately **44% relative to the uncalibrated posterior**. This matters directly to Stage 5 because adaptive beam sets are chosen from predicted probability mass; nominal 95% sets are only meaningful when posterior probabilities are reasonably calibrated.

### Exact common-support comparison against the frozen CV comparator

| Horizon | CV ADE | Gaussian ADE | Common samples |
| --- | ---: | ---: | ---: |
| 0.1 s | 1.650 m | **0.614 m** | 331 |
| 0.3 s | 3.863 m | **0.938 m** | 323 |
| 0.5 s | 6.216 m | **1.307 m** | 320 |
| 1.0 s | 11.950 m | **2.335 m** | 312 |
| **Aggregate** | **5.841 m** | **1.285 m** | 332 actors / 1286 events |

On this exact common support, the Gaussian predictor reduces aggregate ADE by about **78% relative to CV**.

The Stage-4 closure also confirms the causal experimental contract: no future truth is used as model input, annotated velocity is not the primary predictor input, track IDs are not used numerically, and measurement covariance remains distinct from predictive covariance.

**Stage-4 status: COMPLETE / FROZEN.**

---

## 4.3 Stage 5 — Receiver-aware adaptive Top-K beam management

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

The formal acceptance artifact therefore supports the central Stage-5 result: **high empirical coverage is maintained while probing only a small fraction of the full codebook**.

For 16 beams, adaptive probing also passes against the valid fixed Top-K alternatives with fixed Top-3 as the best valid comparator; for 32 beams, fixed Top-5 is the valid comparator. At 64 beams, none of the tested fixed Top-1/3/5 alternatives satisfies the frozen coverage criterion, while the adaptive policy still passes the formal coverage and exhaustive-overhead conditions.

The communication evaluation propagates beam pointing through

```text
beam pointing
    -> optical / beam gain
    -> received power
    -> SNR
    -> DPSK BER
    -> effective rate
```

so the Stage-5 controller is evaluated as a communication-system policy rather than only as beam-index classification.

**Stage-5 formal status: PASS.**

---

## 4.4 Stage 6 — Predictive class-aware ADB

<img width="973" height="292" alt="image" src="https://github.com/user-attachments/assets/d2eac3dc-9515-4162-8191-5471ab28a9bf" />

Stage 6 tests whether probabilistic future occupancy can improve ADB behavior while preserving vehicle/VRU safety constraints.

The repository preserves the complete scientific lineage rather than treating every Stage-6 artifact as interchangeable. An earlier V1 protocol produced a frozen negative over-masking result. A later, explicitly versioned **Stage6-V3 conservative-budgeted-residual** method was then frozen and evaluated with the final common Experiment-5 evaluator. The V1 result remains provenance and is not deleted or relabeled.

### Final Stage6-V3 / Experiment-5 reporting result

| Quantity | Reactive ADB | Final class-aware predictive ADB | Difference / outcome |
| --- | ---: | ---: | ---: |
| Vehicle shadow-zone violation | **0.0948125** | **0.0730832** | **strict improvement — PASS** |
| Over-masking area | **0.1925967** | **0.2016184** | **+0.0090217 ≤ +0.0200 — PASS** |
| Pedestrian visibility proxy | **0.6188124** | **0.6170516** | **−0.0017608 ≥ −0.05 — PASS** |
| Cyclist visibility proxy | **0.7438343** | **0.7399150** | **−0.0039193 ≥ −0.05 — PASS** |

The final Experiment-5 artifact uses the **same final evaluator on both sides** of the reactive-versus-predictive comparison. Under that evaluator, all four frozen Stage-6 acceptance gates pass.

<img width="970" height="1010" alt="image" src="https://github.com/user-attachments/assets/c0df3207-b5b3-4ce8-a2a5-9e6446972b4e" />

### Historical V1 negative result — preserved provenance

The earlier V1 scientific closure recorded reactive over-masking **0.1886246**, predictive over-masking **0.2652175**, and a difference of **+0.0765929**, which exceeded the frozen **+0.0200** non-inferiority bound. That V1 result remains an immutable historical negative result.

The final V3 closure therefore does **not** rewrite the V1 outcome. It records a new method/version and a new frozen completion status while preserving the V1 failure as scientific provenance.

**Stage-6 implementation status: COMPLETE.**  
**Final Stage6-V3 / Experiment-5 gate: PASS.**  
**Historical V1 gate: FAIL — preserved, not erased.**

---

## 4.5 Stage 7 — Frozen joint communication–illumination evaluation

The repository contains a **final frozen Stage-7 evaluation** that is scientifically separate from the Stage-6 lineage. Stage 7 does not erase or reinterpret the historical Stage-6 V1 over-masking failure, and it also does not substitute for the final Stage6-V3 / Experiment-5 reporting result.

The final Stage-7 formal evaluator records:

| Item | Frozen result |
| --- | --- |
| Formal scenarios | **120** |
| Five-system joint rows | **600** |
| Formal evaluation status | **PASS_FROZEN** |
| Future GT used as controller input | **No** |
| Controllers rerun after future-GT access | **No** |
| Post-outcome tuning | **No** |
| Historical Stage-6 outcomes reused | **No** |

Future ground truth is accessed by the evaluator to score already-frozen controller outputs, not as an online controller feature.

The subsequent Stage-7 statistical block operates on the same **600 joint rows**, performs **10,000 bootstrap resamples**, and records status **PASS** without new model forward passes or post-outcome tuning.

The frozen Stage-7 sweep block also records:

- codebook sweep: **PASS**;
- coverage sweep: **PASS**;
- uncertainty sweep: **PASS**;
- exact reproduction check: **PASS**;
- 120 scenario sweep records;
- no new model forward pass;
- no post-outcome tuning.

### Important Stage-7 latency limitation

The Stage-7 latency/resource closure explicitly states that **full joint end-to-end latency is not evaluable from the exact frozen authorities**, because complete measurements for data loading, preprocessing, predictor inference, beam selection and ADB generation are not all available in the frozen authority set.

Therefore this README does **not** fabricate a total end-to-end latency number.

**Stage-7 formal joint-evaluation status: PASS_FROZEN, with the stated latency limitation.**

---

## 4.6 Stage 8 — External measured-beam validation with DeepSense

Stage 8 provides a separate external validation layer using **measured mmWave beam-power data**. It validates the **adaptive beam-selection policy**, not the optical headlamp hardware, and it does not pool optical and mmWave measurements as if they were the same physical modality.

For the primary policy `adaptive_topk_q095`, the final reproducibility report records **n = 1030** measured samples:

| Metric | Point estimate | 95% bootstrap interval |
| --- | ---: | ---: |
| Empirical coverage | **96.89%** | 95.73% – 97.96% |
| Mean selected K | **5.02** | 4.93 – 5.12 |
| Probing overhead | **7.85%** | 7.70% – 8.00% |
| Mean measured power loss | **0.00864 dB** | 0.00467 – 0.01357 dB |
| 1 dB outage rate | **0.291%** | 0 – 0.680% |
| 3 dB outage rate | **0%** | 0 – 0% |
| 6 dB outage rate | **0%** | 0 – 0% |
| Normalized SE gap at 10 dB | **0.00259** | 0.00140 – 0.00407 |

The measured result is consistent with the central adaptive-Top-K principle: a requested 95% policy attains approximately **96.9% empirical coverage** while selecting about **5 beams on average**, with small measured power loss and very low measured outage under this DeepSense evaluation.

The Stage-8 closure also explicitly records:

- optical and mmWave evaluations remain separate;
- DeepSense validates the policy rather than the optical headlamp;
- no post-outcome tuning;
- formal outcomes were not modified during final reporting.

**Stage-8 core status: COMPLETE / final reporting reproducibility PASS.**

---

## 4.7 Stage 9 — Criticality-constrained communication and secondary recovery-aware ADB

Stage 9 is the paper-oriented extension of the frozen Stages 0–8 chain. It does **not** introduce another trajectory predictor. Instead, it reuses the same frozen sensing-informed Monte Carlo future samples as a shared probabilistic interface for:

1. receiver / beam uncertainty;
2. future traffic criticality; and
3. a secondary recovery-aware ADB profile-sensitivity analysis.

The primary paper scope is **C1+C2 communication**. The recovery-aware ADB component **C3** is secondary and does not enlarge the primary communication claim.

The current manuscript authority is the certified **Stage 9.16-v2R1** paper-ready package. A dedicated `iscai_stage9/` directory has not yet been committed to this repository, so the Stage-9 section below reports the manuscript-side frozen evidence and claim boundaries rather than claiming that the full Stage-9 reproducibility package is already contained in the GitHub tree.

### Frozen FORMAL cohort and hypothesis status

The frozen FORMAL cohort contains **26,209 scenarios**. A primary receiver was selected pre-outcome in **25,239** scenarios; **970** had no eligible receiver. Among **24,917 valid critical-event labels**, **zero ground-truth critical events were observed**. Zero observed events are treated as an evidence limitation, not as proof of zero risk.

| Hypothesis | Final disposition |
| --- | --- |
| H1 — trajectory posterior coverage | **Confirmatory test not available**; later RAW-vs-CAL comparison is non-confirmatory only |
| H2 — critical-event reliability | **Not evaluable** — zero observed ground-truth critical events |
| H3 — critical reliability / resource | **Partial** — resource component evaluable; critical-reliability component not evaluable |
| H4 — matched-overhead critical-event comparison | **Not evaluable** — zero observed ground-truth critical events |
| H5 — C3 profile sensitivity | **Secondary** — FAST no clear difference; NOMINAL/SLOW reductions supported |
| H6 — frozen Stage-6 core side-effect gates | **PASS with scope** |

### C2 communication resource component

C2 selects a minimum-size beam set subject to nominal **0.95** coverage and, when raw critical Monte Carlo support exists, a conditional **0.99** critical-sample coverage constraint.

Against fixed `q=0.99`, the evaluable H3 resource component gives the following paired mean-`K` differences:

| Codebook | C2 − fixed q=0.99 mean K | 95% CI |
| --- | ---: | ---: |
| 16 | **−0.602 beams** | −0.619 to −0.585 |
| 32 | **−1.303 beams** | −1.335 to −1.270 |
| 64 | **−2.747 beams** | −2.809 to −2.685 |

This supports **lower communication resource use** for the frozen C2 resource component. It does **not** establish critical-event reliability because the FORMAL set contained no observed ground-truth critical events.

The later RAW-versus-CAL trajectory-coverage comparison is explicitly non-confirmatory: macro absolute requested-versus-empirical coverage error changed by **−0.002321** (95% CI **−0.002621 to −0.002034**). The absolute errors remained large, so this does not establish good absolute trajectory-posterior calibration.

### Secondary C3 recovery-aware ADB

C3 is reported as profile sensitivity rather than a universal human-safety claim.

- **FAST:** no clear difference relative to either frozen baseline;
- **NOMINAL:** lower modeled late-response violation rate versus Stage6-V3 and reactive ADB;
- **SLOW:** lower modeled late-response violation rate versus Stage6-V3 and reactive ADB.

The frozen H6 core side-effect gates are also satisfied relative to reactive ADB. The observed C3-minus-reactive point differences are:

| Core side-effect metric | Difference | Frozen gate |
| --- | ---: | --- |
| Vehicle shadow-zone violation | **−0.003185** | must be < 0 — PASS |
| Over-masking area | **+0.008175** | must be ≤ +0.02 — PASS |
| Pedestrian visibility | **−0.001284** | must be ≥ −0.05 — PASS |
| Cyclist visibility | **−0.001505** | must be ≥ −0.05 — PASS |

### Stage-9 claim boundary

The Stage-9 evidence supports the **shared posterior-to-decision interface**, the **H3 communication resource component**, and **bounded profile-sensitive C3 effects**. It does **not** establish:

- critical-event reliability on ground-truth critical events;
- real-time or end-to-end latency performance;
- optical/headlamp hardware safety validation;
- universal human-safety benefit;
- population-level crash reduction.

**Stage-9 manuscript status: PAPER-READY FROZEN EVIDENCE / bounded claims.**


---

## 5. Dataset and scene representation

The project uses the **Waymo Open Motion Dataset (WOMD)** for real multi-agent traffic dynamics and map context. Actor classes include vehicles, pedestrians and cyclists. WOMD-LiDAR provides synchronized LiDAR history for the observed part of the forecasting window.

LiDAR context can support actor geometry, point-density confidence, visibility/occlusion proxies, intensity and elongation statistics, occupancy/BEV extensions, clutter and partial-observation modeling, and future sensor-to-track extensions.

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

Implements CV, CA, CTRV, EKF/Kalman, IMM and Multidimensional Hough behind a common observation interface, with displacement error interpreted together with reconstruction support.

### Stage 4 — Probabilistic trajectory prediction

Implements deterministic GRU, Gaussian probabilistic GRU, covariance calibration and a trajectory-level GMM extension at the main horizons 0.1, 0.3, 0.5 and 1.0 s.

### Stage 5 — Adaptive beam management

Transforms the receiver posterior into beam probabilities over 16-, 32- and 64-beam codebooks and evaluates fixed and adaptive Top-K policies, probing overhead, link quality, BER and effective rate.

### Stage 6 — Predictive class-aware ADB

Projects future actor occupancy into the illumination controller and evaluates vehicle shadow-zone protection, pedestrian/cyclist visibility and masking cost. Historical V1 preserves a negative over-masking result, while the later frozen Stage6-V3 / final Experiment-5 common-evaluator result passes the four core acceptance gates.

### Stage 7 — Joint evaluation

Runs frozen multi-system communication/illumination evaluation, evaluator-only future-truth scoring, bootstrap statistics, codebook/coverage/uncertainty sweeps and reproducibility checks. The final frozen Stage-7 evaluator records `PASS_FROZEN`; complete joint end-to-end latency remains explicitly not evaluable from the frozen authorities.

### Stage 8 — External measured-beam validation

Evaluates the adaptive Top-K beam policy on measured DeepSense mmWave beam-power data while maintaining strict separation from the optical headlamp/link model.

### Stage 9 — Shared-posterior criticality-constrained decision layer

Reuses the frozen future-motion posterior for same-sample receiver/beam uncertainty and traffic-criticality reasoning. The primary C1+C2 scope studies minimum-resource criticality-constrained beam selection; the secondary C3 scope studies recovery-aware ADB profile sensitivity under frozen claim boundaries.

---

## 8. Experimental modes

The project distinguishes three levels of realism for the WOMD-centered pipeline.

### Oracle-track mode

Perfect **historical** WOMD states are supplied only up to the anchor time. Future states remain evaluator truth. This is an upper-bound historical-state experiment rather than a realistic sensor-to-track implementation.

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

Because WOMD does not contain measured ADB command ground truth, the ADB evaluation uses constructed evaluator references and controller-specific metrics rather than claiming real ADB labels.

Metrics include vehicle shadow-zone violations, glare-risk exposure, over-masking, road illumination retention, pedestrian visibility, cyclist visibility, false dimming, temporal smoothness/flicker and actuation latency.

### External measured-beam validation

DeepSense evaluation includes empirical coverage, selected beam count, probing overhead, measured power loss, outage thresholds and spectral-efficiency-gap metrics with bootstrap uncertainty intervals.

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
| `iscai_stage6/` | Predictive class-aware ADB, preserved historical V1 negative result, and final frozen V3 / Experiment-5 pass artifacts. |
| `iscai_stage7/` | Frozen joint communication–illumination evaluation, statistics and sweeps. |
| `iscai_stage8/` | External DeepSense measured-mmWave beam-policy validation and final reporting. |
| `part_a_reference/` | Frozen reference to the previous PC-FMCW ISCAI implementation. |
| `audits.zip`, `manifests.zip` | Archived reproducibility material. |

---

## 11. Reproducibility philosophy

The repository follows a staged **freeze-and-audit** workflow. Major stages generate machine-readable reports, deterministic manifests, hashes, leakage checks, regression-test evidence and frozen handoff artifacts.

Important principles are:

- scenario-level split discipline;
- separate fit/development/calibration roles;
- no post-hoc use of formal outcomes for model/controller tuning;
- frozen acceptance criteria before outcome interpretation;
- preservation of negative results;
- explicit separation between implementation completion and scientific acceptance;
- evaluator-only access to future ground truth after causal outputs are frozen;
- separation of optical-model evaluation from measured-mmWave external validation;
- reporting from final frozen authorities rather than intermediate repair logs.

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
      -> frozen joint evaluation
      -> measured beam-policy validation
```

Part A provides the sensing/communication/illumination reference and Hough-based legacy tracking context. Part B adds real traffic dynamics, explicit uncertainty propagation, probabilistic forecasting, proactive downstream control and external measured-beam validation.

---

## 13. Scientific claims intentionally not made

This repository does **not** claim that:

- WOMD-LiDAR is FMCW;
- geometry-derived radial velocity is measured FMCW Doppler;
- the monostatic PC-FMCW waveform alone estimates full 3D bearing;
- WOMD contains real optical communication beam labels;
- WOMD contains measured ADB ground-truth commands;
- the constructed ADB reference is measured ground truth;
- the oracle future-GT reference is a deployable predictor;
- trajectory forecasting alone is the novelty;
- predictive ADB alone is the novelty;
- adaptive Top-K beam management alone is the novelty;
- Hough universally dominates other baselines based only on matched-track ADE;
- the historical Stage-6 V1 over-masking failure was erased or relabeled as a pass;
- the final Stage6-V3 / Experiment-5 pass makes the historical V1 negative result disappear;
- Stage 7 substitutes for either Stage-6 authority;
- full joint end-to-end Stage-7 latency has been measured when the frozen authorities explicitly mark it not evaluable;
- DeepSense validates the optical headlamp or optical PC-FMCW link;
- mmWave and optical measurements are physically interchangeable;
- Stage 9 establishes critical-event reliability despite zero observed ground-truth critical events;
- the Stage-9 C3 profile analysis proves universal human-safety benefit or population-level crash reduction;
- Stage 9 establishes real-time/end-to-end latency performance.

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
10. frozen joint system evaluation without future-GT leakage into controller decisions;
11. external validation of the adaptive beam-selection policy on measured DeepSense beam powers;
12. a reproducible workflow that preserves both positive and negative scientific outcomes;
13. a Stage-9 shared posterior-to-decision extension linking same-sample beam uncertainty and traffic criticality;
14. minimum-resource criticality-constrained beam selection with a frozen resource-use result;
15. a secondary recovery-aware ADB profile-sensitivity extension with frozen side-effect gates.

A compact project description is:

> **We develop a real-traffic, PC-FMCW-conditioned vehicular ISCAI framework in which calibrated multi-agent trajectory uncertainty is transformed into future receiver/angular and occupancy representations for adaptive directional beam probing and predictive class-aware illumination, followed by frozen joint evaluation and external measured-beam validation.**

---

## 15. Current scientific status

| Stage | Status |
| --- | --- |
| Stages 0–3 | **IMPLEMENTED / FROZEN upstream pipeline and classical baselines** |
| Stage 4 | **COMPLETE — frozen probabilistic forecasting evaluation** |
| Stage 5 | **PASS — formal adaptive beam-management acceptance** |
| Stage 6 | **COMPLETE — historical V1 over-masking FAIL preserved; final frozen V3 / Experiment-5 common-evaluator gate PASS** |
| Stage 7 | **PASS_FROZEN — final joint evaluator and frozen statistical/sweep analysis complete; full joint latency not evaluable from frozen authorities** |
| Stage 8 | **COMPLETE / PASS — external measured DeepSense beam-policy validation and final reporting reproducibility** |
| Stage 9 | **PAPER-READY FROZEN EVIDENCE — H3 resource component supported; H2/H4 and H3 critical reliability not evaluable; C3 secondary/profile-sensitive; dedicated Stage-9 repo directory not yet committed** |

The overall scientific picture is intentionally mixed rather than artificially all-positive:

- **trajectory forecasting is strong and calibrated**;
- **adaptive Top-K beam management achieves high coverage with large probing reduction**;
- **Stage 6 preserves the historical V1 over-masking failure while the later frozen V3 / Experiment-5 common-evaluator version passes all four core gates**;
- **the frozen Stage-7 evaluator completes a joint communication/illumination evaluation without using future GT as a controller input**;
- **Stage 8 externally validates the beam-selection policy on measured mmWave beam powers**;
- **Stage 9 supports lower beam-resource use for the frozen C2 resource component and bounded profile-sensitive C3 effects, while rare-event reliability remains unresolved because no ground-truth critical event was observed in the valid FORMAL labels**.

The repository therefore preserves both the historical Stage-6 negative result and the later versioned Stage-6 pass rather than collapsing them into a single retrospective outcome.


<!-- BEGIN STAGE9 PAPER UPDATE -->

## Stage 9 — final paper implementation

The repository now includes the curated Stage-9 paper implementation under
[`iscai_stage9/`](iscai_stage9/).

Stage 9 uses the frozen sensing-informed future-motion posterior as a common
uncertainty interface for communication beam management and traffic-criticality
reasoning. The primary contribution is the C1+C2 chain: shared uncertainty
representation plus minimum-cardinality beam probing under nominal and stricter
critical-conditional reliability constraints.

The recovery-aware predictive ADB component (C3) is included as a secondary
FORMAL human-profile-sensitivity extension rather than as the original
confirmatory winner.

The published Stage-9 directory contains:

- byte-identical authoritative execution snapshots,
- current and historical-lineage configuration contracts,
- compact paper-ready aggregate results,
- claim/evidence and limitations metadata,
- final authority/seal artifacts,
- runtime dependency provenance, and
- SHA256 verification metadata.

Raw WOMD/WOMD-LiDAR, raw DeepSense, and large per-scene/future-ground-truth
evaluation caches are not redistributed.

For the exact scientific authority and reporting boundaries, see:

- [`iscai_stage9/README.md`](iscai_stage9/README.md)
- [`iscai_stage9/AUTHORITY.md`](iscai_stage9/AUTHORITY.md)
- [`iscai_stage9/REPRODUCIBILITY.md`](iscai_stage9/REPRODUCIBILITY.md)

The v2R1 Stage-9 package is paper-ready but its sealed state remains
**awaiting independent certification**.

<!-- END STAGE9 PAPER UPDATE -->
