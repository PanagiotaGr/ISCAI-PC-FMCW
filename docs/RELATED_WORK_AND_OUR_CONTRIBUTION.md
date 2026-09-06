# Related Work and Our Contribution

## Purpose

This document maps the papers and datasets that are most relevant to **ISCAI-PC-FMCW Part B**, explains what those works already establish, and states the narrower contribution that this repository can defend.

The goal is not to claim that PC-FMCW, ISAC/ISCAI, motion forecasting, beam prediction, or adaptive driving-beam control are new by themselves. The paper should instead position Part B as a **predictive, uncertainty-aware extension of PC-FMCW ISCAI in which one calibrated future-motion posterior is reused by both communication beam management and predictive illumination control**.

---

## 1. Immediate predecessor: PC-FMCW laser-headlamp ISCAI

### Paper

**S. Liu, T. Sun, X. Shu, J. Song, and Y. Dong, “Phase-Coded FMCW Laser Headlamp for Integrated Sensing, Communication, and Illumination,” IEEE Photonics Technology Letters, 2025/2026 issue record, DOI: `10.1109/LPT.2025.3649597`.**

### What they did

This work introduces a PC-FMCW laser-headlamp architecture that combines:

- communication through phase coding;
- FMCW sensing/ranging;
- adaptive driving-beam illumination;
- multidimensional-Hough-based track-before-detect processing.

The reported system demonstrates the feasibility of combining gigabit-class communication, centimeter-scale ranging, target tracking, and ADB functionality in one optical ISCAI architecture.

### What is already established by this paper

We must **not** claim novelty for:

- the general PC-FMCW ISCAI concept;
- embedding communication data into FMCW through phase coding;
- joint sensing + communication + illumination;
- multidimensional Hough processing as a concept;
- reactive ADB operation as part of the integrated headlamp system.

### How Part B differs

Part B asks a different question:

> Given an uncertain current sensing state and uncertain future road-user motion, can the ISCAI system use a calibrated future-motion posterior to act **proactively** rather than only reactively?

The main extension is therefore from

```text
present sensing state -> reactive communication / illumination action
```

to

```text
uncertain sensing history
        -> probabilistic future-motion prediction
        -> calibrated future posterior
        -> receiver-aware angular posterior + future occupancy
        -> adaptive beam management + predictive ADB
```

---

## 2. Waymo Open Motion Dataset

### Paper / dataset

**S. Ettinger et al., “Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset,” 2021.**

### What they did

The Waymo Open Motion Dataset (WOMD) provides large-scale real traffic trajectories, 3D maps, object tracks, and interactive scenarios involving vehicles, pedestrians, and cyclists. It is designed for motion forecasting and interaction modeling rather than for PC-FMCW sensing.

### What this establishes

Realistic motion forecasting datasets and strong trajectory-prediction baselines are already well established.

We must not claim that:

- using WOMD for trajectory forecasting is new;
- probabilistic motion prediction is new;
- multi-agent traffic forecasting is new.

### How Part B uses WOMD differently

Part B uses WOMD as **real traffic-dynamics ground truth**, then creates a causal PC-FMCW-like sensing interface from those trajectories. WOMD-LiDAR is not treated as measured FMCW data.

The scientific role of WOMD in Part B is:

```text
real road-user dynamics
        -> PC-FMCW-like noisy observations
        -> causal tracking / forecasting
        -> downstream ISCAI control
```

Thus, the contribution is not a new forecasting benchmark. Forecasting is an intermediate probabilistic state representation used for communication and illumination decisions.

---

## 3. State-of-the-art trajectory forecasting

### Representative paper

**J. Gu, Q. Sun, and H. Zhao, “DenseTNT: Waymo Open Dataset Motion Prediction Challenge 1st Place Solution,” 2021.**

### What they did

DenseTNT and related modern forecasting systems use learned map-aware and multimodal prediction architectures to generate high-quality future trajectories on WOMD.

### What this establishes

High-accuracy learned motion forecasting is already a mature research area. Part B therefore should not position the Gaussian GRU/GMM predictor itself as the main algorithmic novelty.

### How Part B differs

The predictor in Part B is selected for its **calibrated posterior interface**, not solely for minimum ADE/FDE.

The important object is not only

```text
future mean trajectory
```

but

```text
future mean + covariance / probability mass
```

because downstream beam selection and illumination act directly on uncertainty.

This supports a stronger systems question:

> Does a calibrated motion posterior provide useful decision information for communication and illumination control beyond point prediction alone?

---

## 4. DeepSense 6G and sensing-aided beam prediction

### Paper / dataset

**A. Alkhateeb et al., “DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and Communication Dataset,” 2022.**

DeepSense contains real co-existing sensing and communication measurements. Relevant vehicular scenarios include GPS, camera/radar sensing, and measured beam-power vectors over mmWave codebooks.

### What this establishes

Sensing-aided beam prediction and beam-selection evaluation on measured wireless data are already established research topics.

We must not claim that:

- beam prediction from mobility/sensing information is new;
- adaptive beam selection is new;
- measured vehicular beam-power datasets are new.

### How Part B differs

DeepSense is used in Part B as an **external measured-beam validation layer** for the adaptive Top-K policy.

The repository explicitly does not claim that mmWave DeepSense measurements validate the optical headlamp physical layer. Instead, they validate the more general policy principle:

```text
predicted probability over future receiver direction
        -> smallest beam subset achieving target mass q
        -> measured coverage / power-loss evaluation
```

This separation is a strength because it avoids conflating optical and mmWave propagation while still adding measured-data evidence for the beam-selection policy.

---

## 5. Predictive / position-aided / sensing-aided beam management

A broad literature already exists on vehicular beam prediction, beam tracking, position-aided beamforming, and sensing-assisted beam management in mmWave/THz and ISAC systems.

Representative themes include:

- predicting future beam directions from vehicle motion;
- using radar, camera, GPS, LiDAR, or channel maps to reduce beam-training overhead;
- predictive beamforming for high-mobility links;
- uncertainty-aware or learning-based beam tracking.

### What this means for our paper

We should **not** claim that predictive beam management is new.

The defensible distinction is the specific Part-B construction:

1. the input is a calibrated **future actor/receiver trajectory posterior** derived from PC-FMCW-like sensing history;
2. receiver geometry is treated explicitly rather than equating the vehicle centroid with the communication receiver;
3. posterior probability mass is transformed into an angular distribution over a codebook;
4. the controller chooses the **minimum Top-K set satisfying a requested probability target**;
5. the same upstream motion posterior is simultaneously reused by the illumination branch.

This joint use of one uncertainty representation is more specific than generic beam prediction.

---

## 6. Adaptive Driving Beam and predictive illumination

ADB and glare-free high-beam control are established automotive technologies and research topics. Existing work includes object-aware illumination, pedestrian/vehicle masking, perception-assisted headlight control, and increasingly predictive or map/perception-aware lighting strategies.

### What is already known

We must not claim novelty for:

- Adaptive Driving Beam itself;
- object masking for vehicle glare protection;
- class-aware illumination in general;
- using perception to control automotive headlights.

### How Part B differs

Part B uses **future probabilistic actor occupancy** rather than only current detections to construct predictive illumination masks.

The key scientific question is not simply whether prediction can reduce glare. It is whether predictive occupancy creates a useful trade-off among:

- vehicle shadow-zone violations;
- pedestrian visibility;
- cyclist visibility;
- over-masking / lost illuminated area.

The frozen Stage-6 result is especially important because it shows that prediction is not automatically better: the predictive controller improves vehicle shadow-zone behavior and preserves VRU visibility, but violates the predeclared over-masking non-inferiority bound.

This should be presented as a genuine **safety–utility trade-off**, not hidden or post-hoc repaired.

---

## 7. Uncertainty calibration as a control interface

Probability calibration is well established in machine learning and probabilistic forecasting. Therefore, calibration itself is not a standalone novelty.

Part B uses calibration for a downstream control reason:

> A nominal 95% angular beam set is only meaningful if posterior probability mass is sufficiently calibrated.

The paper can therefore motivate calibration as part of a **decision-valid uncertainty interface** rather than as a generic predictive-model improvement.

This is important because the same posterior must support two different downstream decisions:

```text
trajectory posterior
       |                         |
       v                         v
angular probability          future occupancy
       |                         |
       v                         v
beam-set selection           predictive ADB
```

---

## 8. Closest combined research gap

The individual components are established:

- PC-FMCW ISCAI;
- motion forecasting on real traffic datasets;
- probabilistic trajectory prediction;
- sensing-aided beam prediction;
- adaptive beam selection;
- ADB illumination;
- predictive/perception-aware vehicle lighting;
- uncertainty calibration.

The more defensible gap is at their **cross-layer combination**.

A conservative positioning statement is:

> **Prior work has established PC-FMCW-based integrated sensing, communication and illumination, probabilistic motion forecasting, sensing-aided beam management, and perception-aware adaptive lighting. Comparatively less attention has been given to using one calibrated future-motion posterior, derived from a PC-FMCW sensing interface, as a common uncertainty representation that simultaneously drives receiver-aware communication beam selection and predictive class-aware ADB control under real traffic dynamics.**

Do not use an absolute “first” claim unless a final systematic literature review supports it.

---

# Our contribution

## Main paper idea

**Uncertainty-Aware Predictive ISCAI for Joint Beam Management and Adaptive Driving Beam Control**

### Main research question

> **Can a calibrated probabilistic prediction of future road-user motion improve communication beam management and anticipatory illumination control in a PC-FMCW ISCAI system, and what communication–illumination trade-offs emerge when the same future posterior is used by both tasks?**

---

## Proposed contribution C1 — Predictive PC-FMCW ISCAI architecture

We extend a reactive PC-FMCW ISCAI pipeline into a causal predictive system:

```text
real WOMD history
    -> PC-FMCW-like sensing observations
    -> tracking / forecasting
    -> calibrated future-motion posterior
    -> communication + illumination decisions
```

The contribution is the predictive decision layer and its cross-task use, not the basic PC-FMCW waveform.

---

## Proposed contribution C2 — Common probabilistic future-state interface

The same calibrated posterior is transformed into:

- a receiver-aware angular posterior for beam management;
- future actor occupancy for ADB control.

This creates a common probabilistic state representation across two traditionally separate downstream subsystems.

---

## Proposed contribution C3 — Receiver-aware adaptive Top-K beam management

Instead of predicting only one best beam, the communication controller chooses the smallest set of beams whose cumulative posterior probability reaches a requested target `q`.

This explicitly trades probing overhead against coverage reliability.

The frozen Stage-5/Stage-8 evidence already supports this principle strongly:

- high empirical coverage at `q = 0.95`;
- large reduction in probing relative to exhaustive search;
- external measured-beam validation on DeepSense;
- small measured power loss and low measured outage in the evaluated cohort.

---

## Proposed contribution C4 — Predictive class-aware ADB with preserved negative result

Future actor occupancy is propagated into illumination control for vehicles, pedestrians, and cyclists.

The Stage-6 result is scientifically useful because it reveals that prediction improves some safety-related objectives while increasing over-masking beyond the frozen non-inferiority allowance.

The paper should explicitly report this as evidence that predictive illumination has a **non-trivial safety–utility trade-off**.

---

## Proposed contribution C5 — Real-trajectory and external-measurement validation layers

The evaluation combines two distinct realism layers:

1. **WOMD:** real multi-agent traffic dynamics, used to drive the causal sensing/prediction/control chain;
2. **DeepSense:** measured mmWave beam-power data, used to externally validate the adaptive beam-selection policy.

These datasets are used for different purposes and are never presented as measured optical PC-FMCW validation.

---

# What the paper should claim

A strong but conservative contribution statement is:

> **We extend reactive PC-FMCW integrated sensing, communication and illumination toward uncertainty-aware predictive operation by using a calibrated future-motion posterior as a common control interface for receiver-aware beam management and class-aware adaptive driving-beam control. The framework is evaluated on real traffic dynamics, preserves causal separation from future ground truth, quantifies the communication–illumination trade-off, and externally validates the adaptive beam-selection principle using measured beam-power data.**

---

# What the paper should NOT claim

Do not claim:

- “we introduce PC-FMCW ISCAI”;
- “we are the first to combine sensing, communication and illumination”;
- “we introduce motion forecasting for autonomous driving”;
- “we introduce predictive beam management”;
- “we introduce Adaptive Driving Beam”;
- “we introduce probability calibration”;
- “DeepSense validates our optical PC-FMCW hardware”;
- “WOMD provides measured FMCW Doppler”;
- “the predictive ADB universally improves the reactive controller.”

The last point is especially important because the frozen Stage-6 over-masking criterion fails.

---

# Recommended related-work structure in the manuscript

A compact manuscript Related Work section can be organized into four subsections:

1. **PC-FMCW / ISCAI systems**  
   Establish the upstream architecture and immediate predecessor.

2. **Probabilistic motion forecasting for autonomous driving**  
   Establish that forecasting is mature and explain why calibrated posterior quality matters here.

3. **Sensing-aided predictive beam management**  
   Position adaptive Top-K against beam-prediction/beam-tracking literature.

4. **Adaptive and predictive automotive illumination**  
   Position the ADB branch and motivate the communication–illumination trade-off.

The final paragraph of Related Work should then state the cross-layer gap rather than repeating the novelty of any individual component.

---

# Priority bibliography shortlist

## PC-FMCW / ISCAI

1. S. Liu, T. Sun, X. Shu, J. Song, and Y. Dong, “Phase-Coded FMCW Laser Headlamp for Integrated Sensing, Communication, and Illumination,” IEEE Photonics Technology Letters, DOI: `10.1109/LPT.2025.3649597`.

## Motion forecasting / WOMD

2. S. Ettinger et al., “Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset,” 2021.
3. J. Gu, Q. Sun, and H. Zhao, “DenseTNT: Waymo Open Dataset Motion Prediction Challenge 1st Place Solution,” 2021.

## Measured sensing + communication / beam validation

4. A. Alkhateeb et al., “DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and Communication Dataset,” 2022.

## Additional literature to complete before submission

The final bibliography should add focused papers from IEEE Xplore / Google Scholar / Scopus on:

- trajectory- or position-aided vehicular beam prediction;
- uncertainty-aware beam tracking;
- sensing-assisted beam management in ISAC;
- probabilistic or risk-aware beam selection;
- predictive Adaptive Driving Beam / glare-free high beam;
- perception-aware automotive illumination;
- calibrated probabilistic trajectory forecasting used for downstream control.

These should be checked again immediately before submission, especially for 2025–2026 publications.

---

# One-sentence paper story

> **Part B asks whether a PC-FMCW ISCAI vehicle can use calibrated uncertainty about where road users will be next—not only where they are now—to proactively choose communication beams and illumination actions.**

# Bottom line

The paper is not about inventing any one component in isolation. Its strongest identity is the **cross-layer reuse of one calibrated future-motion posterior for two physically different control tasks, evaluated causally on real traffic dynamics and supported by external measured-beam evidence**.
