# Stage 3 Implementation Log

Append-only log.

## Block 3.0A — Clean bootstrap

- New canonical Stage 3 created from frozen Stage 2.
- Legacy implementation remains isolated under iscai_stage3_panagiota.
- No code imported from legacy Stage 3.
- No Stage 4+ functionality allowed in Stage 3.
- Main fair input regime: frozen Stage-2 causal algorithm-facing observations.
- Evaluator truth sidecars are scoring-only.
- MHT is Multidimensional Hough Transform.

## Block 3.0B-F — Common Stage2-to-Stage3 contract

Status: PASS

- Frozen Stage2 algorithm-facing observation schema resolved directly.
- Stage3 does not duplicate or reinterpret Stage2 detections.
- Canonical algorithm input is Stage2 UnlabeledDetectionFrame.
- Complete measurement covariance R_t remains available.
- Truth/identity fields are absent from algorithm-facing observations.
- Evaluator truth is isolated under iscai_stage3.evaluation.
- Four association regimes are explicit:
  oracle identity diagnostic,
  oracle association diagnostic,
  estimated association,
  track-before-detect.
- Main regime is estimated association.
- MHT is frozen as Multidimensional Hough Transform.
- IMM modes are frozen as CV / CA / CTRV.
- No imports from iscai_stage3_panagiota.
- No Stage4+ imports.
- Frozen Stage2 closure and validation manifest hashes revalidated.
- Block 3.0 formal contract gate passed.

## Block 3.1 — Common estimated association

Status: PASS

- Main association implemented as deterministic gated global nearest-neighbor.
- Association operates directly in frozen Stage2 measurement space:
  [range, radial velocity, azimuth, elevation].
- Complete Stage2 measurement covariance R_t contributes to gating.
- Detection keys are carried only for provenance and never used for matching.
- No WOMD/Stage2 truth actor IDs are used.
- No actor class is used for association.
- No future information is used.
- False alarms create independent short tracks unless observations support continuation.
- Missed detections/gaps are handled by a bounded causal track lifecycle.
- Azimuth differences use circular wrapping.
- Stage3 track IDs are generated internally and are not truth IDs.
- Association parameters were frozen before formal validation metrics.
- Track-before-detect MHT/Hough remains separate and will consume raw unlabeled observations.
- No Cartesian inter-frame differencing is performed here because measurements belong to dynamic sensor/headlamp frames.
- Ego/headlamp-frame state reconstruction is deferred to Block 3.2A.

## Block 3.2 — H0 causal state reconstruction + CV

Status: PASS

- Stage2 spherical measurements are converted from dynamic headlamp Ht into Cartesian position.
- Full Stage2 4x4 measurement covariance R_t is propagated through the spherical-to-Cartesian Jacobian.
- Position covariance is rotated from Ht into canonical anchor headlamp frame H0.
- Frozen Stage1 RigidTransform APIs are reused; no coordinate system is reimplemented.
- Dynamic headlamp/ego motion is compensated through Ht -> W -> H0.
- All finite differences are performed only after observations are expressed in the same H0 frame.
- Realistic CV velocity is estimated strictly from past noisy positions.
- WOMD annotated velocity is not used.
- Future information is not used.
- Causal 6x6 [position, velocity] covariance is derived from measurement uncertainty.
- Constant Velocity prediction is implemented as p(t+tau) = p(t) + tau*v(t).
- Canonical short horizons are 0.1, 0.3, 0.5 and 1.0 seconds.
- No claim is made of annotated ground truth below the WOMD 100 ms sampling interval.
- A moving-headlamp synthetic gate proves that naive dynamic-frame differencing is avoided.
- Formal real-scenario trajectory metrics remain reserved for the common Stage3 evaluation block.

## Block 3.3 — Constant Acceleration + CTRV

Status: PASS

- Constant Acceleration is implemented from the same causal Stage2-derived H0 tracklets as CV.
- CA requires three past/current observations.
- Acceleration is estimated from consecutive interval velocities divided by their midpoint-time separation.
- Current CA velocity is reconstructed causally from the latest interval velocity plus half-step acceleration correction.
- The formulation is valid for unequal observation intervals and is exact for ideal constant-acceleration motion.
- CA does not use WOMD annotated velocity, annotated acceleration or future information.
- The full CA 9x9 [position, velocity, acceleration] covariance is propagated analytically from the three H0 position covariances.
- CA prediction uses p(t+tau) = p + v*tau + 0.5*a*tau^2.
- CTRV is implemented as a planar constant-turn-rate/constant-speed model in H0 with constant vertical velocity.
- CTRV heading and turn rate are estimated only from causal H0 position differences.
- Angular differences are circularly wrapped, preventing the legacy +/-pi heading discontinuity bug.
- Latest interval heading is compensated from the interval midpoint to the current anchor time.
- A chord-to-arc speed correction is used in the identifiable turning regime.
- A low-speed fallback prevents undefined/noisy heading estimates for nearly stationary tracks.
- The CTRV propagation has an explicit continuous near-zero-turn straight-line branch.
- CTRV does not use annotated WOMD heading, annotated WOMD velocity or future information.
- CA and CTRV use the same canonical 0.1 / 0.3 / 0.5 / 1.0 s prediction horizons.
- No formal validation metrics were used to tune Block 3.3 parameters.
- Formal real-scenario CA/CTRV metrics remain reserved for the common Stage3 evaluation block.

## Block 3.4 — Covariance-aware Kalman / EKF

Status: PASS

- The Kalman baseline is a genuine nonlinear EKF and not a CV alias.
- State is [px, py, pz, vx, vy, vz] in canonical H0.
- Initialization uses only the first two causal noisy Stage2-derived H0 positions.
- WOMD annotated velocity is not used.
- Process dynamics use a CV state transition with continuous white-acceleration process covariance.
- Process-noise spectral density is a declared Stage3 baseline parameter, not a measured sensor fact.
- The genuine nonlinear measurement vector is [range, radial velocity, azimuth, elevation].
- Measurement predictions are produced in the dynamic headlamp frame Ht.
- Actor state remains in H0 and is transformed through the frozen Stage1 rigid-transform API.
- Headlamp translational velocity is reconstructed causally from strict-adjacent Stage1 poses.
- Predicted radial velocity uses actor velocity relative to the moving headlamp projected onto line of sight.
- The nonlinear measurement Jacobian is implemented analytically and verified against numerical finite differences.
- Azimuth innovations are circularly wrapped.
- The complete Stage2 4x4 R_t enters every EKF measurement update.
- R_t is not replaced by a fixed covariance.
- Posterior covariance uses Joseph form.
- Synthetic moving-sensor closed-form tests verify position, velocity and radial-velocity consistency.
- Kalman trajectory prediction uses the actual filtered posterior, not a separate CV placeholder.
- No evaluator truth, annotated velocity or future information enters the Kalman algorithm.
- No formal validation metrics were used to tune Kalman parameters.
- Formal real-scenario Kalman metrics remain reserved for the common Stage3 evaluation block.

## Block 3.5 — Full IMM with CV / CA / CTRV

Status: PASS

- IMM is implemented as a genuine interacting multiple-model filter.
- Required modes are CV, CA and CTRV.
- Each mode maintains its own state mean and covariance.
- A common 10D H0 interaction state [p, v, a, turn-rate] enables mathematically explicit cross-model mixing.
- The source-to-destination Markov transition matrix is explicit and frozen before formal validation.
- Predicted mode probabilities are computed from the previous posterior probabilities and the transition matrix.
- Source-to-destination conditional mixing probabilities are explicitly normalized.
- Mixed state means are computed for every destination mode.
- Mixed covariances include both within-mode covariance and between-mode mean spread.
- CV, CA and CTRV use genuinely different process dynamics.
- CTRV prediction is nonlinear and uses a numerical process Jacobian only for covariance propagation.
- Every mode uses the same nonlinear Stage2 measurement model [range, radial velocity, azimuth, elevation].
- Every mode uses the complete Stage2 4x4 R_t.
- Azimuth innovations are circularly wrapped.
- Each mode computes a multivariate Gaussian measurement likelihood from its innovation and innovation covariance.
- Posterior mode probabilities are normalized in the log domain.
- A combined IMM posterior mean and covariance are computed after every measurement.
- Combined covariance includes both mode covariance and between-mode state disagreement.
- Open-loop trajectory prediction continues IMM interaction and Markov mode-probability propagation every 100 ms without using future measurements.
- Canonical horizons remain 0.1, 0.3, 0.5 and 1.0 seconds.
- No evaluator truth, WOMD annotated velocity, WOMD annotated heading or future information enters the IMM algorithm.
- IMM is not a deterministic trajectory selector and is not the legacy lightweight mode-probability implementation.
- No formal validation metrics were used to tune IMM parameters.
- Formal real-scenario IMM metrics remain reserved for the common Stage3 evaluation block.

## Block 3.6 — Multidimensional Hough Transform

Status: PASS

- The frozen Part-A repository was audited before implementation.
- Frozen Part-A commit 44d62e3478e3818d1757b00971890f844cb032f7 was revalidated.
- Part A contains a genuine geometric Hough pipeline with rho/theta voting, accumulator peaks and final AND-logic trajectory validation.
- Part A does not provide a measured-FMCW Range-Doppler Hough implementation; Stage 3 does not claim otherwise.
- Stage 3 preserves the Part-A voting -> accumulator -> peaks -> final-validation architecture.
- The Stage-3 time-aware Hough parameter space is [x_anchor, y_anchor, vx, vy].
- x_anchor/y_anchor are defined at the last causal observation timestamp.
- The Hough branch consumes raw unlabeled Stage2 detection frames directly.
- The Stage3 estimated-association frontend is not used by Hough.
- Each raw Stage2 detection is transformed causally from Ht into canonical H0 using frozen Stage1 transforms.
- Stage2 position covariance contributes to covariance-aware local accumulator voting.
- Stage2 radial-velocity measurement and variance contribute an additional candidate-velocity consistency term.
- The accumulator is sparse and four-dimensional.
- A single frame can contribute at most one effective vote per accumulator cell, preventing same-frame clutter density from masquerading as temporal support.
- Local peak suppression is explicit and deterministic.
- Final Hough tracks require accumulator support AND temporal support AND temporal span AND bounded miss gaps AND spatial/radial consistency.
- Validated trajectories are refined using covariance-weighted linear position-vs-time fitting.
- Missing detections are tolerated within the frozen causal gap policy.
- Synthetic crossing-target tests demonstrate that distinct velocities remain separable.
- Synthetic clutter and false-alarm tests demonstrate that isolated detections do not become accepted trajectories.
- Detection keys are not used for temporal matching.
- Actor identity and actor class are not algorithm inputs.
- Evaluator truth is not used.
- Future information is not used.
- The Hough output supports the canonical Stage3 prediction horizons 0.1 / 0.3 / 0.5 / 1.0 s through the prediction adapter.
- Formal real-scenario Hough metrics remain reserved for the common Stage3 evaluation block.

## Block 3.7 — Common Stage3 trajectory evaluator

Status: PASS

- A single deterministic trajectory-evaluation contract is shared by CV, CA, CTRV, Kalman/EKF, IMM and Multidimensional Hough outputs.
- WOMD future trajectory truth is represented exclusively as an evaluator-side sidecar.
- Evaluator truth identity is never supplied to Stage3 algorithms.
- Prediction-to-truth assignment is global one-to-one Hungarian assignment implemented without SciPy.
- Assignment cost uses only current/anchor H0 position.
- Future trajectory truth is not read during assignment.
- A future-truth mutation invariance test proves that future trajectories cannot change prediction-to-truth matching.
- Anchor assignment uses a frozen 5 m gate selected before formal real validation.
- Primary trajectory ADE/FDE are defined in planar H0 XY space.
- Supplementary 3D ADE/FDE are also reported.
- Per-horizon planar and 3D MAE/RMSE are supported at 0.1, 0.3, 0.5 and 1.0 s.
- Reconstruction precision, recall and F1 are explicitly reported.
- Unmatched truth targets and false prediction tracks are reported separately.
- Endpoint miss rate uses a frozen 2 m planar displacement threshold.
- Runtime is a supported reporting field but is not part of trajectory-error scoring.
- NLL, Brier score, calibration error and confidence-region coverage remain deferred to Stage4.
- No sub-100-ms WOMD ground-truth claim is made.
- Formal real WOMD validation metrics remain reserved for Block 3.8.

## Block 3.8B — Frozen real-validation access and WOMD truth sidecar

Status: PASS

- The frozen validation manifest contains exactly 44,097 exact-matching validation scenarios.
- Validation ordering is deterministic from SHA256(scenario_id) and independent of predictor performance or future motion.
- Canonical paired motion shards are compact paired-from-* shards.
- Original WOMD record_offset values are provenance only.
- Canonical offsets are reconstructed deterministically from cumulative selected TFRecord byte lengths within each original source-shard group.
- All 150 canonical compact motion shard sizes exactly match the frozen manifest reconstruction.
- Multiple real random-access probes across beginning, middle and end positions of different compact shards reproduce the expected scenario IDs.
- Runtime motion access remains inside the canonical paired corpus and has no dependency on deleted/original /waymo WOMD motion shards.
- No second motion dataset is materialized.
- Paired LiDAR presence and frozen byte size are verified.
- WOMD temporal structure remains 91 timestamps, current_time_index=10, 10 causal-history samples and 80 future samples at nominal 10 Hz.
- WOMD future trajectory truth is evaluator-only.
- Eligibility is fixed from an explicit upstream current/anchor eligible set plus current-state validity.
- Future states are never used for eligibility or prediction-to-truth assignment.
- Future validity controls only availability of individual future scoring points.
- No interpolation of WOMD ground truth is performed.
- Canonical evaluation horizons remain 0.1, 0.3, 0.5 and 1.0 s.
- The deterministic 12-scenario list remains a runtime/eligibility pilot candidate set only.
- Formal validation N and formal scenario IDs remain deliberately unfrozen.
- Stage3 algorithm packages contain no dependency on evaluator WOMD truth.
- Block 3.8B targeted regression: 11/11 PASS.
- Full Stage3 regression after Block 3.8B: 106/106 PASS.

## Block 3.8C — Real one-scenario end-to-end integration

Status: PASS

- A real canonical WOMD validation scenario was propagated end-to-end through the clean Stage1 -> Stage2 -> Stage3 pipeline.
- Scenario: b85e1bd6cc8e74c0.
- Stage1 causal adaptation and Stage2 ideal/clean/degraded observation generation were used directly; no legacy Stage3 or raw-WOMD predictor bypass was introduced.
- Stage3 received the canonical truth-free AlgorithmObservationSequence produced by the existing Stage2 bridge.
- The algorithm-facing observation sequence remained immutable before prediction, after all predictors and after evaluation.
- The integration profile retained Gaussian PC-FMCW-like measurement corruption and measurement covariance R_t.
- P_D=1 and false alarms=0 were used only for this integration proof so wiring defects were not confounded with robustness to missing/cluttered detections.
- This single-scenario run is not a formal performance experiment.
- CV, CA, CTRV, Kalman EKF and IMM consumed the same estimated-association frontend derived from the same degraded Stage2 sequence.
- Multidimensional Hough consumed the same raw unlabeled Stage2 frames directly and did not use estimated association, identities, truth or future information.
- Real predictions were produced by all six classical methods:
  CV=10, CA=9, CTRV=9, Kalman_EKF=10, IMM_CV_CA_CTRV=9, Multidimensional_Hough=8.
- Evaluator-only WOMD future truth was constructed only after all algorithm predictions had completed.
- Canonical evaluation horizons remained 0.1, 0.3, 0.5 and 1.0 s.
- Formal validation remains unfrozen; formal N was not selected in Block 3.8C.
- Full Stage3 regression after Block 3.8C: 106/106 PASS.

## Block 3.8D — Full-degraded deterministic runtime/eligibility pilot

Status: PASS

- The deterministic pre-formal pilot contained 12 validation scenarios selected solely by ascending SHA256(scenario_id).
- The pilot manifest SHA256 is 6f0c383eac6df2e5eb49d033c40d2c3e4094d09423996864cafe918fd4e4c772.
- Scenario selection used neither future WOMD trajectory information nor predictor performance.
- The full frozen Stage2 degraded profile was used: Gaussian PC-FMCW-like measurement corruption, SNR-dependent missed detections, Poisson false alarms and measurement covariance R_t.
- All 12/12 scenarios completed successfully.
- Full Stage3 regression after the pilot remained 106/106 PASS.
- All six classical methods produced at least one prediction in all 12 scenarios.
- CV and Kalman EKF produced 458 predictions each.
- CA, CTRV and IMM produced 353 predictions each.
- Multidimensional Hough produced 110 predictions directly from raw unlabeled detections.
- Median total scenario runtime was 5125.357 ms; observed p95/max was 20723.781 ms.
- Hough dominated computational cost, with observed p95 model runtime 19559.811 ms.
- The 12-scene SHA-ranked pilot contained 390 current-valid vehicles and 52 current-valid pedestrians, but no current-valid cyclist.
- Therefore a simple globally SHA-ranked formal subset would not guarantee the mandatory vehicle/pedestrian/cyclist coverage.
- No ADE/FDE or other trajectory-performance result from this pilot is a formal scientific result.
- Formal validation N remained unfrozen throughout Block 3.8D.

## Block 3.8E — Frozen formal real-validation manifest

Status: PASS

- The complete frozen WOMD validation manifest of 44,097 scenarios was scanned using current/anchor metadata only.
- Future states, tracks_to_predict, degraded observations, association outcomes and predictor performance were not used for formal scenario selection.
- Causal stratum availability was:
  cyclist=7147,
  pedestrian_no_cyclist=20027,
  vehicle_only=16923.
- The formal validation subset is frozen at N=120.
- The subset contains exactly 40 cyclist-containing scenarios, 40 pedestrian-containing/no-cyclist scenarios and 40 vehicle-only scenarios.
- Selection within each mutually-exclusive stratum is deterministic by ascending SHA256(scenario_id).
- The frozen formal manifest SHA256 is 2208e7287ddf6439fda4597c435a9cba1d1b9d0e4c4547bc5dd92e56e8124e46.
- Formal anchor coverage contains 57 current-valid cyclists, 361 current-valid pedestrians and 3755 current-valid vehicles.
- Formal selection is neither future-based, performance-based nor degraded-outcome-based.
- The formal manifest and N are now immutable for Stage3 evaluation.
- Full Stage3 regression after Block 3.8E: 106/106 PASS.

## Block 3.8F — Formal real WOMD validation

Status: PASS

- Formal evaluation used the frozen 120-scenario validation manifest with SHA256 2208e7287ddf6439fda4597c435a9cba1d1b9d0e4c4547bc5dd92e56e8124e46.
- The full frozen Stage2 degraded observation profile was used for every scenario.
- CV, CA, CTRV, Kalman EKF and IMM shared the same estimated-association frontend.
- Multidimensional Hough consumed the same raw unlabeled Stage2 observation frames directly.
- Algorithm-visible Stage2 observations remained truth-free and immutable.
- WOMD tracks_to_predict was used only as primary evaluator metadata after all predictions had completed.
- WOMD future states were evaluator-only and were never used for prediction or anchor assignment.
- Primary aggregation uses micro ADE over available matched actor-horizon errors.
- FDE uses matched tracks with an evaluator-defined final endpoint; matched tracks without available FDE are reported explicitly and are not assigned artificial zero/error values.
- Reconstruction precision/recall/F1 uses pooled TP/FP/FN over the frozen 120 scenarios.
- Endpoint miss rates use pooled eligible truth denominators at 0.1, 0.3, 0.5 and 1.0 s.
- Formal primary results:
  CA: ADE=42.494382 m, FDE=115.390678 m.
  CTRV: ADE=6.453930 m, FDE=10.415681 m.
  CV: ADE=9.538706 m, FDE=19.598319 m.
  IMM_CV_CA_CTRV: ADE=11.232591 m, FDE=20.139908 m.
  Kalman_EKF: ADE=12.373488 m, FDE=19.791891 m.
  Multidimensional_Hough: ADE=1.614700 m, FDE=2.536681 m.
- Hough achieved lower matched-track displacement error but substantially lower reconstruction recall; therefore displacement error and coverage must be interpreted jointly.
- No model parameters, association thresholds, Hough settings or formal scenarios were changed after seeing formal results.
- The complete 120-scenario prediction/evaluation checkpoint was preserved after an aggregate-only FDE-availability wrapper defect and was recovered without rerunning predictors.
- Formal sanity gate: PASS.
- Full Stage3 regression after formal evaluation: 106/106 PASS.
- Deterministic formal run SHA256: 5fbb4642f87ab8e62cf1e1b3ef4214d92df5b5a1423bd9fb72b9733ced02e433.

## Block 3.8G — Formal reproducibility

Status: PASS

- The frozen 120-scenario formal evaluation was executed a second time from the same canonical WOMD data and frozen Stage2 degraded profile.
- The formal manifest remained unchanged with SHA256 2208e7287ddf6439fda4597c435a9cba1d1b9d0e4c4547bc5dd92e56e8124e46.
- The reference deterministic run SHA256 was 5fbb4642f87ab8e62cf1e1b3ef4214d92df5b5a1423bd9fb72b9733ced02e433.
- The repeat deterministic run SHA256 was exactly 5fbb4642f87ab8e62cf1e1b3ef4214d92df5b5a1423bd9fb72b9733ced02e433.
- All 120 per-scenario deterministic hashes matched exactly.
- Primary formal metrics matched exactly.
- Supplementary formal metrics matched exactly.
- Formal scenario order matched exactly.
- Runtime equality was intentionally not required because wall-clock runtime is not a deterministic scientific output.
- No model, threshold, association setting, Hough setting, scenario selection or evaluation denominator was changed between runs.
- Full Stage3 regression after reproducibility verification: 106/106 PASS.

## Stage 3 — Final closure

Status: COMPLETE_FROZEN

- All Stage3 classical-baseline implementation blocks are closed.
- Formal real-WOMD validation uses the frozen N=120 manifest.
- Formal manifest SHA256: 2208e7287ddf6439fda4597c435a9cba1d1b9d0e4c4547bc5dd92e56e8124e46.
- Exact deterministic formal run SHA256: 5fbb4642f87ab8e62cf1e1b3ef4214d92df5b5a1423bd9fb72b9733ced02e433.
- Independent repeat reproduced all 120 scenario hashes and all primary and supplementary metrics exactly.
- Full final regression: 106/106 PASS.
- Algorithm truth dependency: NONE.
- Future WOMD truth is evaluator-only and is not used for assignment.
- Legacy runtime dependencies: NONE.
- Downstream Stage4+ dependencies: NONE.
- Multidimensional Hough is preserved as the Hough baseline and consumes raw unlabeled Stage2 observations directly.
- Stage3 does not claim probabilistic NLL/Brier/calibration, shared posterior, beam/ADB, optical-chain or DeepSense completion; these remain downstream.
- Final implementation files: 87.
- Final implementation SHA256: 12b7a236cf1ea8e8a02b1c74da44d7db79cdaef363542a2be564b697036e6bbf.
