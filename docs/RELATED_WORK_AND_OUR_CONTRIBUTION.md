# Related Work and Our Contribution

This note summarizes the main papers we can cite, **what the other authors actually did**, and **what Part B does differently**.

## 1. Liu et al. — PC-FMCW ISCAI laser headlamp

**S. Liu, T. Sun, X. Shu, J. Song, and Y. Dong, “Phase-Coded FMCW Laser Headlamp for Integrated Sensing, Communication, and Illumination,” IEEE Photonics Technology Letters, DOI: `10.1109/LPT.2025.3649597`.**

### What they did

They proposed a **PC-FMCW laser-headlamp ISCAI system** that combines three functions in the same architecture:

- FMCW sensing/ranging;
- phase-coded high-rate communication;
- Adaptive Driving Beam (ADB) illumination.

They also used a **Multidimensional Hough Transform** track-before-detect method for target detection/tracking. Their work establishes the feasibility of PC-FMCW for integrated sensing, communication, and illumination.

### What we do differently

We do not introduce PC-FMCW ISCAI itself. We extend this architecture toward **predictive and uncertainty-aware operation**: instead of using only the current estimated target state, we estimate a probability distribution over future road-user motion and use it for future control decisions.

---

## 2. Cheng, Wu, and Zhao — calibrated trajectory uncertainty for beam management

**Q. Cheng, W. Wu, and Z. Zhao, “Uncertainty-Calibrated UAV Trajectory Prediction for Beam Management in UAV-Assisted ISAC Scenarios,” Drones, vol. 10, no. 6, 434, 2026. DOI: `10.3390/drones10060434`.**

### What they did

They studied a **UAV-assisted ISAC** scenario. Their method:

1. predicts the future UAV trajectory;
2. estimates uncertainty around that prediction;
3. calibrates the uncertainty using conformal calibration;
4. uses the calibrated spatial risk for adaptive beam management.

Their main idea is therefore:

```text
trajectory prediction
    -> calibrated uncertainty
    -> communication beam management
```

They show that calibrated predictive uncertainty can improve beam coverage and reduce outage/misalignment in high-risk UAV scenarios.

### What we do differently

This means we cannot claim that `prediction + calibrated uncertainty -> beam management` is new.

Our distinction is that the **same calibrated future-motion posterior is reused for two different physical functions** inside an automotive PC-FMCW ISCAI system:

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

So our contribution is broader than the communication branch alone.

---

## 3. Ettinger et al. — Waymo Open Motion Dataset (WOMD)

**S. Ettinger et al., “Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset,” ICCV, 2021.**

### What they did

They introduced a large-scale autonomous-driving motion dataset containing real traffic scenes, trajectories of vehicles/pedestrians/cyclists, interactions, and map information. The dataset was designed to support **motion forecasting**, including interactive multi-agent prediction.

### What we do differently

We do not introduce a new forecasting dataset and do not treat WOMD as FMCW measurements.

We use the real traffic motion as the physical scenario behind our experiment:

```text
real WOMD trajectories
    -> simulated PC-FMCW-like observations
    -> tracking / probabilistic forecasting
    -> communication + illumination control
```

Thus WOMD gives us realistic road-user dynamics, while the PC-FMCW observation layer remains simulated/model-based.

---

## 4. Alkhateeb et al. — DeepSense 6G

**A. Alkhateeb et al., “DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and Communication Dataset,” IEEE Communications Magazine, 2023, DOI: `10.1109/MCOM.006.2200730`.**

### What they did

They developed a **real-world multimodal sensing and communication dataset** for research on sensing-aided wireless systems. DeepSense scenarios include modalities such as GPS, cameras, radar/LiDAR, together with measured mmWave beam-power information.

The dataset supports tasks such as position-, radar-, LiDAR-, and sensing-aided beam prediction and future beam selection.

### What we do differently

We use DeepSense only as an **external measured-data test of our adaptive beam-selection principle**.

We do not claim that DeepSense validates the optical PC-FMCW headlamp. The optical ISCAI experiment and measured mmWave beam experiment remain physically separate.

---

# What did the others do, and what did we do?

| Work | What they did | What we add |
|---|---|---|
| **Liu et al.** | PC-FMCW sensing + communication + ADB + Hough tracking | Future-motion prediction and uncertainty-aware proactive control |
| **Cheng et al.** | Calibrated trajectory uncertainty -> beam management | Same future posterior drives **beam management + ADB** |
| **WOMD / Ettinger et al.** | Real multi-agent traffic trajectories for motion forecasting | Use real traffic dynamics behind a PC-FMCW-like sensing-to-control pipeline |
| **DeepSense / Alkhateeb et al.** | Real multimodal sensing + measured mmWave beam data | External measured-data validation of our adaptive beam-selection policy |

# Our contribution

The individual components are not new by themselves. Our main contribution is the **cross-function predictive integration**:

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

> **We extend PC-FMCW automotive ISCAI from reactive operation toward uncertainty-aware predictive operation, using one calibrated future-motion posterior as a common probabilistic interface for both receiver-aware communication beam management and predictive class-aware Adaptive Driving Beam control.**

In the simplest form:

> **The others have already used prediction and uncertainty for beam management. We use one future probabilistic representation to proactively control both communication and illumination inside automotive PC-FMCW ISCAI.**

This is the contribution we should emphasize. We should describe it as **comparatively underexplored / to the best of our literature review**, rather than claim “first ever.”
