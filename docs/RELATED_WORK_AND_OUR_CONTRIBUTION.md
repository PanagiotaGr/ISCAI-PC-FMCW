# Related Work and Our Contribution

This note keeps only the literature needed to position **ISCAI-PC-FMCW Part B** and the contribution we can defend.

## 1. Liu et al. — PC-FMCW ISCAI laser headlamp

**S. Liu, T. Sun, X. Shu, J. Song, and Y. Dong, “Phase-Coded FMCW Laser Headlamp for Integrated Sensing, Communication, and Illumination,” IEEE Photonics Technology Letters, DOI: `10.1109/LPT.2025.3649597`.**

### What they did

They introduced a PC-FMCW laser-headlamp ISCAI system combining:

- sensing/ranging;
- phase-coded communication;
- Adaptive Driving Beam (ADB) illumination;
- multidimensional-Hough-based target processing/tracking.

### What we do differently

We do **not** claim PC-FMCW ISCAI itself as new. We extend this type of system from mainly present-state/reactive operation to **predictive, uncertainty-aware operation**.

---

## 2. Cheng, Wu, and Zhao — calibrated trajectory uncertainty for beam management

**Q. Cheng, W. Wu, and Z. Zhao, “Uncertainty-Calibrated UAV Trajectory Prediction for Beam Management in UAV-Assisted ISAC Scenarios,” Drones, vol. 10, no. 6, 434, 2026. DOI: `10.3390/drones10060434`.**

### What they did

They predict future UAV trajectories, calibrate trajectory uncertainty, and use that uncertainty for adaptive communication beam management.

Their chain is approximately:

```text
trajectory prediction
    -> uncertainty calibration
    -> beam management
```

### What this means for us

We must **not** claim that uncertainty-calibrated trajectory prediction for predictive beam management is new by itself.

### What we do differently

Our predicted/calibrated future-motion posterior is used as a **common interface for two different downstream functions**:

```text
calibrated future-motion posterior
            |
      +-----+-----+
      |           |
      v           v
receiver-aware   future actor
angular posterior occupancy
      |           |
      v           v
adaptive Top-K   predictive
beam management  class-aware ADB
```

The communication branch is therefore only one part of the contribution. The main distinction is the reuse of the same probabilistic future state across **communication and illumination inside automotive PC-FMCW ISCAI**.

---

## 3. Waymo Open Motion Dataset (WOMD)

**S. Ettinger et al., “Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset,” 2021.**

### What they did

WOMD provides real multi-agent road trajectories and map/context information for motion forecasting.

### How we use it

We use WOMD for **real traffic dynamics**, not as measured FMCW data:

```text
real WOMD motion
    -> simulated PC-FMCW-like observations
    -> tracking / probabilistic prediction
    -> ISCAI control
```

Therefore our contribution is not a new motion-forecasting dataset or a claim of real PC-FMCW measurements.

---

## 4. DeepSense 6G

**A. Alkhateeb et al., “DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and Communication Dataset,” 2022.**

### What they did

DeepSense provides real multimodal sensing/communication measurements, including measured beam-power information useful for beam prediction and selection studies.

### How we use it

We use measured DeepSense data as an **external validation of the adaptive beam-selection policy**.

It does not validate the optical PC-FMCW headlamp hardware. Optical and mmWave evaluations remain separate.

---

# What is our contribution?

The individual ingredients already exist in the literature:

- PC-FMCW ISCAI;
- trajectory prediction;
- uncertainty calibration;
- predictive beam management;
- Adaptive Driving Beam control.

Our contribution is their specific predictive cross-function integration.

## Core contribution

> **We extend PC-FMCW automotive ISCAI from reactive operation toward uncertainty-aware predictive operation, using one calibrated future-motion posterior as a common probabilistic interface for both receiver-aware communication beam management and predictive class-aware Adaptive Driving Beam control.**

In simple form:

```text
PC-FMCW-like sensing
        -> probabilistic future-motion prediction
        -> calibrated future-motion posterior
                     |
              +------+------+
              |             |
              v             v
       communication     illumination
       beam control      ADB control
```

The important point is therefore **not** “we use uncertainty for beam management.” Cheng et al. already do a closely related prediction-calibration-beam-management chain.

The stronger distinction is:

> **one future-motion posterior -> two physically different ISCAI control functions**

The posterior becomes:

- a receiver/angular probability distribution for adaptive Top-K beam selection; and
- a future actor-occupancy distribution for predictive class-aware illumination.

The experiments then evaluate both the communication benefit and the illumination safety/utility trade-off.

## Conservative novelty statement

> **Prior work has separately established PC-FMCW integrated sensing/communication/illumination and uncertainty-calibrated predictive beam management. Comparatively less attention has been given to using a common calibrated future-motion posterior to drive both receiver-aware communication control and predictive class-aware illumination within an automotive PC-FMCW ISCAI system.**

We should use wording such as **“comparatively underexplored”** or **“to the best of our literature review”**, rather than an absolute “first ever” claim.

## What we should NOT claim

- We invented PC-FMCW ISCAI.
- We invented trajectory prediction.
- We invented uncertainty calibration.
- We invented predictive beam management.
- We invented ADB.
- WOMD is measured FMCW data.
- DeepSense validates the optical headlamp hardware.

## One-sentence paper story

> **Part B uses calibrated uncertainty about where road users will be next as a shared control representation so that communication and illumination can act proactively rather than only react to the present state.**
