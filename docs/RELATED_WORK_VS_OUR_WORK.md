# Related Work vs. Our Work

## Purpose

This document records what closely related work has already done, what the current repository does, and where the defensible research gap / contribution appears to be.

> **Important:** the novelty claim should not be "predictive beamforming is new" or "adaptive Top-K beam selection is new". Both ideas have prior art. The stronger and safer positioning is the **end-to-end propagation of calibrated motion uncertainty from PC-FMCW-like sensing into receiver-aware angular/beam probabilities, followed by adaptive minimum-cardinality probing under a prescribed probability-mass target**, together with systematic evaluation and measured-data validation.

---

## 1. Research question

A suitable research question for the paper is:

> **Can calibrated motion-prediction uncertainty derived from PC-FMCW-like sensing be propagated into a receiver-aware angular posterior and exploited to adapt the number of probed mmWave beams while maintaining a prescribed beam-coverage target?**

The proposed pipeline can be summarized as

```text
PC-FMCW-like sensing
        |
        v
Probabilistic trajectory prediction
        |
        v
Calibrated spatial uncertainty p(x_{t+tau})
        |
        v
Receiver-aware angular posterior p(theta_{t+tau})
        |
        v
Beam-domain probabilities P(B_i)
        |
        v
Adaptive minimum-cardinality Top-K probing
```

The beam set is selected according to

\[
K^* = \min K \quad \text{s.t.} \quad
\sum_{i=1}^{K} P(B_{(i)}) \ge q,
\]

where the beam probabilities are sorted in descending order and `q` is a prescribed probability-mass target.

---

## 2. What prior work has already done

### 2.1 Situational-awareness-based beam prediction (2018)

**Va et al., "MmWave Beam Prediction with Situational Awareness: A Machine Learning Approach," IEEE SPAWC, 2018.**

DOI: `10.1109/SPAWC.2018.8445969`

What they do:
- use vehicular situational information, including receiver/surrounding-vehicle locations;
- learn beam information / received power from past observations;
- aim to reduce beam-training overhead in high-mobility mmWave links.

Why it matters to us:
- beam prediction from mobility/context is not new;
- therefore our novelty cannot simply be "we use position/trajectory information to predict beams."

Difference from our work:
- our main object is an explicit **probabilistic future state distribution**;
- its uncertainty is propagated through receiver geometry into an angular/beam posterior;
- adaptive probing is directly driven by this propagated uncertainty.

---

### 2.2 Deep-learning-assisted calibrated/adaptive beam training (2021)

**"Deep Learning Assisted Calibrated Beam Training for Millimeter-Wave Communication Systems," IEEE Transactions on Communications, 2021.**

DOI: `10.1109/TCOMM.2021.3098683`

What they do:
- CNN-based prediction from wide-beam measurements;
- LSTM-based tracking for mobile users;
- additional narrow-beam training based on predicted probabilities;
- adaptive partial wide-beam training to reduce overhead.

Why it matters to us:
- probability-assisted and adaptive partial beam training already exists;
- LSTM/mobility-aware beam calibration also already exists.

Difference from our work:
- our beam probabilities are not produced directly from wide-beam received-signal patterns;
- they are obtained by propagating a **calibrated probabilistic motion forecast** through receiver geometry into the beam domain;
- this gives the uncertainty a physically interpretable chain: spatial uncertainty -> angular uncertainty -> beam uncertainty.

---

### 2.3 DL-based beam-management literature

**"Deep Learning for mmWave Beam-Management: State-of-the-Art, Opportunities and Challenges," IEEE Wireless Communications, 2023.**

DOI: `10.1109/MWC.018.2100713`

and

**"A Survey of Beam Management for mmWave and THz Communications Towards 6G," IEEE Communications Surveys & Tutorials, 2024.**

DOI: `10.1109/COMST.2024.3361991`

These surveys establish that:
- beam acquisition/tracking overhead is a central mmWave/THz problem;
- mobility and blockage make narrow-beam management difficult;
- AI and ISAC are already major directions for reducing beam-management overhead.

Implication for our paper:
- we should position the work inside **uncertainty-aware sensing-assisted beam management**, rather than claiming a new general beam-management paradigm.

---

### 2.4 Sensing-enabled predictive beamforming for V2I (2023/2024)

**"Sensing-Enabled Predictive Beamforming Design for RIS-Assisted V2I Systems: A Deep Learning Approach," IEEE Transactions on Wireless Communications, 2024.**

DOI: `10.1109/TWC.2023.3327362`

What they do:
- exploit ISAC echo signals in high-mobility V2I;
- predict time-varying CSI and optimize BS/RIS beamforming;
- also propose an end-to-end DL beamforming architecture;
- optimize communication performance / achievable rate.

Why it matters:
- sensing-assisted predictive beamforming is clearly established prior art.

Difference from our work:
- our focus is not direct CSI/beamformer prediction;
- we explicitly model future motion uncertainty and propagate it into a beam candidate distribution;
- the downstream decision is an adaptive beam-probing set rather than an end-to-end beamforming-vector regressor.

---

### 2.5 End-to-end ISAC predictive beamforming (2024)

**"Integrated Sensing and Communications for End-to-End Predictive Beamforming Design in Vehicle-to-Infrastructure Networks," IEEE Journal of Selected Topics in Signal Processing, 2024.**

DOI: `10.1109/JSTSP.2024.3474254`

What they do:
- identify error propagation as a weakness of two-stage state-estimation -> beamforming pipelines;
- directly predict beamformers from reflected sensing signals using DNNs;
- propose one-sided and two-sided predictive beamforming;
- evaluate achievable sum-rate.

Relationship to us:
- this is conceptually important because our approach deliberately keeps the intermediate probabilistic state representation rather than bypassing it;
- our argument must therefore be that the intermediate posterior is useful because it provides **interpretable, calibratable uncertainty** that can be consumed by downstream decision policies.

Potential paper discussion:
- end-to-end prediction may optimize beamforming directly;
- our modular approach exposes uncertainty and supports risk/coverage-aware beam-set sizing.

---

### 2.6 SCAN-BEST: candidate beam sets with formal reliability (2025)

**Deng, Shi, Li, Simeone, "SCAN-BEST: Efficient Sub-6GHz-Aided Near-field Beam Selection with Formal Reliability Guarantee," 2025.**

arXiv: `2503.13801`

What they do:
- use sub-6 GHz channel estimates as side information;
- use a CNN to predict probabilities that near-field mmWave beams are optimal;
- use **Conformal Risk Control (CRC)** to construct candidate beam sets;
- provide a formal user-defined target-coverage guarantee under their assumptions.

This is the closest work to our adaptive candidate-set idea.

Consequences for our claims:
- we must **not** claim that probability-based adaptive beam candidate sets are new;
- we must **not** claim a formal finite-sample coverage guarantee unless we actually implement and validate such a method;
- our current `q` should be described as a **prescribed probability-mass / design coverage target**, with empirical coverage evaluated separately.

Key difference:
- SCAN-BEST maps sub-6G channel observations directly to beam probabilities;
- our pipeline maps PC-FMCW-like sensing -> probabilistic future trajectory -> calibrated spatial uncertainty -> receiver-aware angular posterior -> beam probabilities;
- therefore our main scientific object is **uncertainty propagation across sensing, mobility, geometry, and communication domains**.

Possible future extension:
- adding conformal calibration / CRC on top of our beam posterior could convert the current empirical coverage story into a stronger formal reliability story.

---

### 2.7 Predictive error-ellipse adaptive beamforming (2026)

**"Extended Target Adaptive Beamforming for ISAC: A Perspective of Predictive Error Ellipse," IEEE Transactions on Wireless Communications, 2026.**

DOI: `10.1109/TWC.2026.3652714`

What they do:
- use predictive error ellipses for extended-target ISAC;
- exploit predicted scatterer and communication-receiver positions;
- adapt beam width / beamforming based on geometric uncertainty;
- report achievable-rate gains over conventional beam sweeping.

Why this is especially relevant:
- it confirms that explicit geometric predictive uncertainty is now being used in ISAC beam design;
- consequently, our novelty must be narrower than "using prediction uncertainty for beamforming."

Difference to emphasize:
- our pipeline creates a calibrated **beam probability distribution** from probabilistic trajectory forecasts;
- the decision variable is the **minimum candidate-set cardinality** needed to capture target posterior mass;
- we additionally evaluate the approach through frozen sweeps/statistics and measured mmWave beam-power validation.

---

### 2.8 Predictive beam management with trajectory learning (2025)

**"DeepBeam: A Multi-Agent Deep Reinforcement Learning Framework for Predictive mmWave Beam Management in Dynamic V2X Networks," IEEE Transactions on Vehicular Technology, 2025.**

DOI: `10.1109/TVT.2025.3574081`

What they do:
- trajectory prediction for predictive mmWave beam management;
- multi-agent reinforcement learning;
- federated learning and coordination mechanisms;
- evaluate beam-alignment accuracy, throughput and latency.

Implication:
- trajectory prediction + predictive beam management is not itself sufficient as a novelty claim.

Our distinction:
- calibrated probabilistic trajectory forecasting and explicit uncertainty propagation, rather than a deterministic trajectory-prediction / policy-learning pipeline.

---

## 3. What our repository does

### 3.1 Probabilistic mobility prediction

The repository evaluates classical and learned motion predictors, including constant-velocity / acceleration models, filtering/model-based approaches, deterministic GRU and probabilistic predictors.

The strongest current forecasting result reported in the repository gives approximately:

- Constant Velocity aggregate ADE: **5.841 m**
- Gaussian predictor aggregate ADE: **1.285 m**

corresponding to about a **78% ADE reduction** on the common evaluation support.

The uncertainty calibration stage reduces macro ECE approximately from:

- **0.1438 -> 0.0806**

so uncertainty quality is evaluated separately from point-prediction accuracy.

### 3.2 Uncertainty propagation to the receiver/beam domain

Instead of discarding the predictive covariance, the repository carries the probabilistic state into the receiver geometry and constructs an angular posterior.

Conceptually:

\[
p(\mathbf{x}_{t+\tau}|\mathcal D_t)
\rightarrow p(\theta_{t+\tau}|\mathcal D_t)
\rightarrow \{P(B_i|\mathcal D_t)\}_{i=1}^{N}.
\]

This is the main methodological component that should be foregrounded in the paper.

### 3.3 Adaptive minimum-cardinality beam probing

For sorted beam probabilities

\[
P(B_{(1)}) \ge P(B_{(2)}) \ge \cdots,
\]

we select

\[
K^*=\min\left\{K:\sum_{i=1}^{K}P(B_{(i)})\ge q\right\}.
\]

At `q = 0.95`, current repository results report mean probe counts of approximately:

| Codebook size | Mean adaptive K | Exhaustive K | Probe reduction |
|---:|---:|---:|---:|
| 16 | 2.54 | 16 | ~84.1% |
| 32 | 4.17 | 32 | ~87.0% |
| 64 | 7.39 | 64 | ~88.5% |

This is the clearest system-level performance result: uncertainty is converted into a reduction in communication probing cost.

### 3.4 Systematic evaluation rather than a single diagram

The repository contains frozen evaluation protocols, codebook/coverage/uncertainty sweeps, bootstrap statistics, failure slices and reproducibility artifacts.

The Stage-7 frozen evaluation includes:
- **120 scenarios** in the sweep;
- codebook, coverage and uncertainty sweeps;
- **10,000 bootstrap resamples** in the statistical evaluation;
- explicit controls against post-outcome tuning / new model forwards in the frozen evaluation.

### 3.5 External measured-data validation

The main mobility/sensing pipeline uses real mobility data together with PC-FMCW-like simulated sensing observations. It must **not** be described as real PC-FMCW radar measurements.

The communication-side result is complemented with external measured mmWave beam-power validation using DeepSense data.

Current reported external-validation values include approximately:

- `n = 1030` samples;
- target `q = 0.95`;
- empirical coverage: **96.89%**;
- mean candidate-set size: **K = 5.02**;
- probing overhead: **7.85%**;
- mean measured power loss: **0.00864 dB**;
- 3-dB outage: **0%**.

This measured-data validation should be presented as a major strength, while clearly separating it from the simulated PC-FMCW-like sensing stage.

### 3.6 Negative result / limitation: predictive ADB

The predictive ADB branch should not be sold as a universally successful contribution.

It improves some safety-oriented criteria but increases over-masking relative to the reactive baseline and fails the frozen non-inferiority criterion. This is useful as a scientific negative result:

> predictive uncertainty can benefit beam management while conservative future-occupancy propagation can introduce an illumination-efficiency penalty.

For a focused beam-management paper, ADB should therefore be secondary / limitation material rather than the main contribution.

---

## 4. Direct comparison

| Dimension | Prior literature | Our current work |
|---|---|---|
| Mobility/context-assisted beam prediction | Already established | Yes |
| ISAC sensing-assisted predictive beamforming | Already established | Yes, but not the novelty by itself |
| DL/LSTM beam prediction | Already established | Used only as part of the forecasting comparison/pipeline |
| Adaptive/partial beam training | Already established | Yes |
| Probability-based candidate beam sets | Already established | Yes |
| Formal target-coverage guarantee | Exists (e.g. SCAN-BEST/CRC) | **No formal guarantee currently; empirical target/coverage** |
| Explicit probabilistic future trajectory | Present in parts of related literature | **Core representation in our pipeline** |
| Calibration of motion uncertainty | Less common in beam-management pipeline | **Explicitly evaluated** |
| Spatial -> angular -> beam uncertainty propagation | Related geometric uncertainty work exists | **Central end-to-end mechanism** |
| Adaptive K driven by propagated posterior mass | Related candidate-set approaches exist | **Central decision rule** |
| Classical + neural forecasting baselines | Varies | **Extensive** |
| Frozen sweeps / bootstrap evaluation | Varies | **Strong repository feature** |
| Measured mmWave beam-power validation | Not universal | **Included through DeepSense external validation** |
| Negative downstream result retained | Rarely emphasized | **ADB over-masking failure retained** |

---

## 5. What is NOT a defensible novelty claim

Do **not** write any of the following without much stronger evidence:

1. "We are the first to use AI for beam management."
2. "We are the first to predict mmWave beams from mobility."
3. "We are the first to use sensing for predictive beamforming."
4. "We are the first to use adaptive Top-K beam training."
5. "We are the first to use uncertainty in ISAC beamforming."
6. "Our q=0.95 policy guarantees 95% coverage."
7. "The WOMD-based experiment uses real PC-FMCW radar measurements."

These claims are either contradicted by existing literature or stronger than the current evidence supports.

---

## 6. Defensible contribution statement

A safer contribution statement is:

> **We develop and evaluate an uncertainty-propagated predictive beam-management framework in which calibrated probabilistic motion forecasts derived from PC-FMCW-like sensing are transformed through receiver geometry into angular and beam-domain probability distributions. These distributions drive a minimum-cardinality adaptive probing policy that selects the smallest beam subset capturing a prescribed posterior probability mass.**

The evaluation contribution can be stated separately:

> **The framework is evaluated against classical and learned mobility predictors using calibration analysis, frozen sensitivity sweeps, bootstrap statistics, and external measured mmWave beam-power validation.**

Notice that these statements describe exactly what is done without claiming unsupported global priority.

---

## 7. Candidate paper title

### Preferred

**Calibrated Uncertainty Propagation for Adaptive Predictive Beam Management in PC-FMCW ISAC**

### Alternative

**Uncertainty-Propagated Predictive Beam Management for PC-FMCW-Based Integrated Sensing and Communication**

---

## 8. Recommended paper contributions

Keep the paper to three main contributions.

### C1 — Calibrated uncertainty propagation

A framework that maps probabilistic motion forecasts into receiver-aware angular and beam-domain posterior distributions instead of collapsing the prediction to a single future position.

### C2 — Adaptive posterior-mass beam probing

A minimum-cardinality beam-set policy that dynamically changes `K` according to propagated uncertainty and a prescribed probability-mass target `q`.

### C3 — Rigorous evaluation

Evaluation using multiple forecasting baselines, uncertainty calibration, codebook/coverage/uncertainty sweeps, bootstrap statistics and measured mmWave beam-power external validation.

---

## 9. The most important remaining experiment: ablation

The final report should isolate where the gain comes from. A recommended table is:

| Predictor | Calibrated uncertainty | Beam policy | Empirical coverage | Mean K | Overhead | Power loss |
|---|---:|---|---:|---:|---:|---:|
| Constant Velocity | No | Fixed / deterministic | TBD | TBD | TBD | TBD |
| Deterministic GRU | No | Fixed / deterministic | TBD | TBD | TBD | TBD |
| Gaussian GRU | No | Adaptive posterior-mass | TBD | TBD | TBD | TBD |
| Gaussian GRU | **Yes** | **Adaptive posterior-mass** | **TBD** | **TBD** | **TBD** | **TBD** |

This is needed to answer:

> Is the system-level improvement caused only by better trajectory prediction, or does calibrated uncertainty + adaptive posterior-mass probing provide an additional benefit?

If the existing frozen artifacts contain all required quantities, generate the table from them without retraining. Otherwise this should be the priority additional experiment.

---

## 10. Possible stronger extension

A particularly strong future extension would be to add **conformal calibration / conformal risk control** to the beam-set construction.

Current scheme:

\[
\sum_{i=1}^{K^*}P(B_{(i)})\ge q
\]

with empirical coverage validation.

Potential extension:

```text
calibrated motion posterior
        -> beam scores/probabilities
        -> conformal / risk-control calibration
        -> beam candidate set with a statistically justified reliability target
```

This would directly address the strongest difference between our current approach and SCAN-BEST's formal reliability framework.

It is an extension, not something the current repository should claim to already provide.

---

## 11. Bottom line

The repository contains enough technical depth for a paper-style Part B, but the scientific story should be narrow and precise.

The main message should **not** be:

> "We used AI and PC-FMCW to predict beams."

It should be:

> **"We preserve, calibrate, and propagate predictive motion uncertainty from sensing to the beam domain, and use that uncertainty to adapt beam-training effort to the predicted difficulty of the future alignment problem."**

That is the clearest distinction between the complete pipeline in this repository and much of the existing deterministic/direct beam-prediction literature.

---

## References / starting bibliography

1. Va et al., "MmWave Beam Prediction with Situational Awareness: A Machine Learning Approach," IEEE SPAWC, 2018. DOI: `10.1109/SPAWC.2018.8445969`.
2. "Deep Learning Assisted Calibrated Beam Training for Millimeter-Wave Communication Systems," IEEE Transactions on Communications, 2021. DOI: `10.1109/TCOMM.2021.3098683`.
3. "Deep Learning for mmWave Beam-Management: State-of-the-Art, Opportunities and Challenges," IEEE Wireless Communications, 2023. DOI: `10.1109/MWC.018.2100713`.
4. "Sensing-Enabled Predictive Beamforming Design for RIS-Assisted V2I Systems: A Deep Learning Approach," IEEE Transactions on Wireless Communications, 2024. DOI: `10.1109/TWC.2023.3327362`.
5. "A Survey of Beam Management for mmWave and THz Communications Towards 6G," IEEE Communications Surveys & Tutorials, 2024. DOI: `10.1109/COMST.2024.3361991`.
6. "Integrated Sensing and Communications for End-to-End Predictive Beamforming Design in Vehicle-to-Infrastructure Networks," IEEE JSTSP, 2024. DOI: `10.1109/JSTSP.2024.3474254`.
7. Deng, Shi, Li, Simeone, "SCAN-BEST: Efficient Sub-6GHz-Aided Near-field Beam Selection with Formal Reliability Guarantee," 2025, arXiv:2503.13801.
8. "DeepBeam: A Multi-Agent Deep Reinforcement Learning Framework for Predictive mmWave Beam Management in Dynamic V2X Networks," IEEE Transactions on Vehicular Technology, 2025. DOI: `10.1109/TVT.2025.3574081`.
9. "Extended Target Adaptive Beamforming for ISAC: A Perspective of Predictive Error Ellipse," IEEE Transactions on Wireless Communications, 2026. DOI: `10.1109/TWC.2026.3652714`.

### Literature-review note

This file is a working novelty map, not yet a systematic review. Before final submission, bibliography metadata should be exported/verified from IEEE Xplore/Crossref and the novelty language should be re-checked against any additional papers found through backward/forward citation search.
