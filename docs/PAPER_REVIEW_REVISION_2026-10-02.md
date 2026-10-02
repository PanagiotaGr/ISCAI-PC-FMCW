# Paper review revision — 2026-10-02

This note records the manuscript-facing corrections made after a technical/IEEE-style review of **Shared Calibrated Predictive Uncertainty for PC-FMCW ISCAI: Beam Selection, Geometric Criticality, and Adaptive Driving Beam Control**.

The rule for this revision is fail-closed: manuscript wording is limited to the current Stage-9 authority and paper-ready aggregates. No new post-FORMAL experiment, endpoint, profile, threshold, or paired confidence interval is introduced to rescue an unsupported claim.

## 1. Monte Carlo semantics

The manuscript now distinguishes the three frozen sampling roles instead of presenting them as if they were one inconsistent sample count:

- **2048 samples:** earlier Stage-5 uncertainty-only communication implementation;
- **4096 samples:** current Stage-9 consequence-aware C2 communication/criticality decision; the exact same sample is mapped to beam and criticality;
- **8192 samples:** Stage-6 ADB occupancy approximation, drawn from the same calibrated posterior but not claimed to be the identical C2 sample set.

## 2. C2 criticality geometry

The Stage-9 primary event uses:

- horizon: **1.0 s**;
- actor representation: sampled actor **center** in the ground plane;
- ego representation: causally propagated closed oriented rectangle;
- point-to-rectangle distance: exact Euclidean 2-D distance, zero inside the rectangle;
- primary clearance: **1.0 m**;
- ego yaw-rate estimate: OLS over the previous **1.0 s** causal heading history;
- CTRV propagation, with CV reduction when `|yaw_rate| < 0.001 rad/s`;
- Monte Carlo count: **4096**.

This remains a geometric critical-proximity surrogate, not a collision probability, actor-box overlap event, or TTC.

## 3. Executed C2 optimization

For beam counts `n_b`, critical beam counts `c_b`, and `C=sum_b c_b`, the executed C2 decision minimizes selected beam count subject to:

- missed nominal samples <= **204** (equivalent to >=95% of 4096);
- when `C>0`, missed critical samples <= `floor(C/100)` (equivalent to >=99% of predicted critical samples).

For `1 <= C <= 99`, the integer form therefore requires all predicted critical samples to be covered.

The final implementation is an **exact dynamic program over excluded beams**, not a generic MILP call. Tie-break order is:

1. minimum cardinality;
2. minimum nominal missed count;
3. minimum raw critical missed count;
4. lexicographically smallest sorted beam-index set.

A fail-closed all-beam fallback exists, but the final paper-ready communication table reports **zero fallback records**. FORMAL solver/system timing was not materialized, so no real-time, latency, or speedup claim is authorized.

## 4. q=0.95 / proposed / q=0.99 ablation

The frozen point estimates on the common (N=24,917) support are:

| B | Policy | Mean K | Coverage |
| ---: | --- | ---: | ---: |
| 16 | fixed q=0.95 | 3.158 | 98.050% |
| 16 | proposed C2 | 3.379 | 98.118% |
| 16 | fixed q=0.99 | 3.981 | 99.037% |
| 32 | fixed q=0.95 | 5.435 | 97.777% |
| 32 | proposed C2 | 5.788 | 97.869% |
| 32 | fixed q=0.99 | 7.091 | 98.924% |
| 64 | fixed q=0.95 | 9.950 | 97.612% |
| 64 | proposed C2 | 10.482 | 97.668% |
| 64 | fixed q=0.99 | 13.228 | 98.832% |

The proposed C2 policy uses approximately **0.221 / 0.353 / 0.532** more beams than fixed q=0.95 for 16/32/64 beams, respectively. This is a descriptive ablation. No new paired interval is introduced for proposed-vs-q=0.95.

The inferentially supported H3 resource comparison remains proposed C2 vs fixed q=0.99:

| B | Delta mean K | 95% paired bootstrap CI |
| ---: | ---: | --- |
| 16 | -0.602 | [-0.619, -0.585] |
| 32 | -1.303 | [-1.335, -1.270] |
| 64 | -2.747 | [-2.809, -2.685] |

## 5. Critical-event support boundary

Among **24,917** valid FORMAL critical labels, **0** ground-truth critical-proximity events are observed. The 95% Wilson occurrence-frequency interval is approximately **[0, 1.54e-4]**.

Therefore:

- critical-conditioned realized reliability is **not estimable**;
- zero observed events are not reported as zero risk;
- no critical-event reliability advantage over q=0.95 is claimed;
- the prediction-side critical-mass constraint remains a method definition, not an empirically validated collision-safety guarantee.

## 6. Calibration reproducibility

The frozen covariance variance multipliers are:

| Horizon (s) | alpha_h |
| ---: | ---: |
| 0.1 | 1.2347064500 |
| 0.3 | 1.3451250295 |
| 0.5 | 1.3548220577 |
| 1.0 | 1.2829700217 |

Calibration changes predictive covariance only; the predictive mean and measurement covariance remain unchanged. The frozen primary fit objective is macro absolute coverage error over nominal 50/80/90/95/99% regions.

## 7. Recovery-aware ADB reporting

The final Stage-9 recovery-aware policy is kept distinct from the older Stage-6 predictive-ADB baseline.

The selected frozen recovery policy is `V5_RHO2_DIRECT_RAW_PRIORITY` with:

- dimming rate: **2 s^-1**;
- brightening/recovery rate: **1 s^-1**.

FAST, NOMINAL, and SLOW are pre-specified virtual sensitivity profiles. The current sealed package does **not** contain an aggregate reaction-margin distribution; none is reconstructed after FORMAL. Manuscript reporting is limited to the sealed pairwise (P(C_late)) comparisons and H6 side-effect metrics.

### H5 profile sensitivity

| Profile | Baseline | Delta late-response rate | 95% CI |
| --- | --- | ---: | --- |
| FAST | Stage-6 predictive | 0 | [0, 0] |
| FAST | Reactive | 0 | [0, 0] |
| NOMINAL | Stage-6 predictive | -1.11e-3 | [-1.76e-3, -0.57e-3] |
| NOMINAL | Reactive | -3.29e-3 | [-4.90e-3, -1.92e-3] |
| SLOW | Stage-6 predictive | -2.05e-3 | [-3.28e-3, -1.02e-3] |
| SLOW | Reactive | -5.07e-3 | [-7.38e-3, -2.99e-3] |

### H6 recovery-aware ADB minus reactive ADB

| Metric | Delta | 95% CI |
| --- | ---: | --- |
| Vehicle shadow-zone violation | -0.003185 | [-0.003467, -0.002919] |
| Over-masking area | +0.008175 | [0.008132, 0.008217] |
| Pedestrian visibility | -0.001284 | [-0.002012, -0.000609] |
| Cyclist visibility | -0.001505 | [-0.003455, 0.000161] |

These are the final Stage-9 paired deltas and must not be confused with the older absolute Stage-6 V3-vs-reactive table.

## 8. DeepSense boundary

DeepSense is described as **supporting measured-mmWave evidence from a prior evaluation**, not as a new independent Stage-9 validation. It does not validate optical PC-FMCW, geometric criticality, ADB, or human-response claims.

## 9. Prior technical report and repository citation

The prior technical report remains a separate work:

- *Predictive PC-FMCW ISCAI*, Technical Report, September 2026;
- DOI: **10.5281/zenodo.23089733**;
- canonical repository: **https://github.com/PanagiotaGr/ISCAI-PC-FMCW**.

The manuscript explicitly distinguishes the earlier 120-scenario report evaluation from the later Stage-9 CAL/FORMAL protocol. The Stage-9 split itself freezes a 121-scenario prior-inspection exclusion set; exact membership is repository-manifest authority rather than something inferred from the technical-report sample count.

## 10. Manuscript wording / layout changes

The revised manuscript:

- renames **System Implementation** to **System Model and Shared Predictive Framework**;
- adds IEEE-style **Index Terms**;
- replaces the ambiguous “V3 = final ADB” wording with explicit Stage-6-baseline versus Stage-9-recovery terminology;
- labels both ADB comparators explicitly;
- corrects the GitHub URL;
- cites the technical report through its DOI;
- removes duplicate `\end{document}`;
- uses hidden hyperlinks in the PDF;
- keeps the current paper as a two-column preprint. Exact `IEEEtran` conversion should be performed only after a target IEEE venue (conference vs journal) is selected, because the class, page size, author block, and page limit are venue-specific.

## 11. Files that remain authoritative

For exact paper claims and current reporting boundaries, prefer:

- `iscai_stage9/AUTHORITY.md`;
- `iscai_stage9/results/paper_ready/stage9_block916_v2R1_full_communication_table.csv`;
- `iscai_stage9/results/paper_ready/stage9_block916_v2R1_table_H3_resource.csv`;
- `iscai_stage9/results/paper_ready/stage9_block916_v2R1_table_H5_profile_sensitivity.csv`;
- `iscai_stage9/results/paper_ready/stage9_block916_v2R1_table_H6_side_effects.csv`;
- `iscai_stage9/results/paper_ready/stage9_block916_v2R1_methods_mathematical_problem.json`;
- `iscai_stage9/results/paper_ready/stage9_block916_v2R1_dataset_provenance.json`;
- `iscai_stage9/results/paper_ready/stage9_block916_v2R1_adb_reporting_boundary.json`.

No statement in this note should be interpreted as independent certification of the v2R1 package; the package remains paper-ready and awaiting independent certification.
