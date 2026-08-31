# Stage 4 Implementation Log

## Block 4.0 — Clean Stage4 bootstrap and contract freeze

Status: PASS / FROZEN

- Stage2 and Stage3 are immutable upstream dependencies.
- Stage3 implementation SHA256: 12b7a236cf1ea8e8a02b1c74da44d7db79cdaef363542a2be564b697036e6bbf.
- Frozen formal validation N=120; manifest SHA256: 2208e7287ddf6439fda4597c435a9cba1d1b9d0e4c4547bc5dd92e56e8124e46.
- Frozen Stage3 formal run SHA256: 5fbb4642f87ab8e62cf1e1b3ef4214d92df5b5a1423bd9fb72b9733ced02e433.
- Main Stage4 observation mode is the frozen full-degraded Stage2 stream with the frozen Stage3 estimated-association frontend.
- Measurement covariance R_t is a mandatory predictor input and is distinct from predictive uncertainty.
- Future states/validity/track duration/LiDAR are not model inputs; future trajectory is supervision/evaluation only.
- tracks_to_predict and objects_of_interest are not model inputs.
- Perfect track IDs are not numeric neural features.
- Deterministic GRU, Gaussian probabilistic GRU and trajectory calibration are mandatory.
- GMM will be implemented/evaluated and becomes closure-critical if the unimodal Gaussian is inadequate.
- Lightweight multi-agent context is mandatory; map context will be implemented and ablated.
- Transformer/raw-LiDAR BEV are not closure requirements.
- Receiver/angular posterior, beams, optical link, ADB and DeepSense are explicitly downstream of Stage4.
- PDF acceptance requires the probabilistic predictor to beat at least one classical baseline and confidence regions to have measured calibration.
- The stricter project gate additionally requires calibrated probabilistic planar ADE to beat the frozen Stage3 CV ADE.
- Calibration levels are 50/80/90/95/99%.
- Normalization/calibration may not use formal validation data.
- Block4.0 regression: 9/9 PASS.
- Block4.0 implementation files: 5.
- Block4.0 implementation SHA256: 22f27101f5c6e57a1bc66a1299c05be911ea6af26a18a0fe36807cfa4e944a4f.

## Block 4.1 — Frozen training partitions

Status: PASS / FROZEN

- Canonical paired WOMD training corpus contains 72,085 scenarios.
- Partitioning is scenario-level and deterministic by ascending SHA256(scenario_id).
- Frozen counts: fit=57,668, development=7,208, calibration=7,209.
- Internal partition overlap: 0.
- Training/official-validation scenario overlap: 0.
- Training/frozen-formal-N120 scenario overlap: 0.
- Future trajectories, future validity, performance outcomes, tracks_to_predict and objects_of_interest do not influence partition membership.
- Three real-scenario future-mutation probes leave the allowed causal index metadata unchanged.
- Current-time vehicle, pedestrian and cyclist support exists in fit, development and calibration.
- Actor class is audited but does not alter scenario partitioning; class balancing will occur later inside the fit sample sampler.
- Normalization statistics are restricted to fit only.
- Early stopping/model development is restricted to development only.
- Calibration fitting is restricted to calibration only.
- Official/frozen validation is evaluation-only.
- No future labels have yet been materialized by Stage4.
- Fit manifest SHA256: 284a61c877d937deb07137345beaabf3165690138785f9cc41f81655066d5276.
- Development manifest SHA256: e4689698bddd80e58add7267f791aea90ef4bef0309ca18d0d7a9e7e94fe7e5c.
- Calibration manifest SHA256: e003dd5c4d5a253729700b12c3754c47ffb99dadfa035b46627f01ec91eed4be.
- Full Stage4 regression: 20/20 PASS.
- Implementation files: 10.
- Implementation SHA256: 2021ef79a5f875239e132e404805f4c512e45926d0fdac9d2d6c7a5b51eab7eb.

## Block 4.2 — Real causal neural sample contract

Status: PASS / FROZEN

- Real Stage4 samples consume the frozen full-degraded Stage2 stream through the frozen Stage3 estimated association and Cartesian geometry.
- Each target history contains exactly 11 causal WOMD frames.
- The frozen per-step numeric feature dimension is 14.
- Position is noisy H0 position produced from the degraded observation stream.
- Velocity is strict-backward and derived only from causal estimated positions; annotated WOMD velocity is not a realistic input.
- Stage2 measurement covariance R_t is propagated to H0 by the frozen Stage3 geometry layer and is a mandatory numeric model input.
- Measurement covariance remains distinct from predictive uncertainty, which has not yet been created.
- Missing observations remain explicit through observed/velocity-valid masks.
- Each target receives up to 8 nearest causal associated neighbour histories.
- A lightweight static WOMD map-context summary is included as causal model context; learned map encoding remains a later ablation.
- Perfect WOMD track ID, actor class, truth track index and sample ID are metadata only and are not numeric model inputs.
- tracks_to_predict and objects_of_interest are not model inputs.
- Supervision pairing is one-to-one and uses historical/current information only with a 5-m gate.
- Future trajectory is accessed only after causal model-input construction and only as supervision.
- Future validity is a label mask only and does not control sample inclusion.
- Strict future-mutation testing leaves Stage2 algorithm input, Stage4 causal scene input and numeric model payload unchanged while changing labels.
- Real deterministic pilot: first 4 frozen-manifest scenarios from each fit/development/calibration partition.
- No normalization, class sampler, GRU, predictive covariance, NLL or calibration is claimed by Block4.2.
- Full Stage4 regression: 34/34 PASS.
- Implementation files: 15.
- Implementation SHA256: 4f6ef90bd3455bf55f100561d66ef93549d8472112b85fce45930ec169f89fd0.

## Block 4.3 — Deterministic multi-agent GRU

Status: PASS / FROZEN

- A deterministic lightweight multi-agent GRU baseline was trained on real frozen Stage4 causal samples.
- Frozen model-development subset: 4096 fit scenes and 512 development scenes, selected only by pre-frozen SHA order.
- Per-scene supervised-sample cap is deterministic by causal prediction ID and does not use future-label quality.
- Cache generation is resumable; valid completed scene shards are reused and corrupt generated shards are archived before regeneration.
- Input uses the frozen Stage2 degraded observations through Stage3 estimated association.
- Stage2 measurement covariance remains a mandatory numeric predictor input.
- Main deterministic architecture contains target-history GRU, neighbour-history GRU aggregation and lightweight map encoder.
- Actor class is used only for the fit sampler and is not a neural model feature.
- Primary vehicle, pedestrian and cyclist samples receive equal total fit-sampling mass.
- All normalization statistics are fit-only; missing rows and padded neighbours are excluded.
- Early stopping uses development planar ADE only.
- Calibration and formal validation are not used by Block4.3.
- Future trajectory is supervision only; future validity is a loss mask only.
- Deterministic checkpoint inference repeats exactly from the same frozen checkpoint.
- Best development planar ADE: 1.605454724 m.
- Best epoch: 18.
- Checkpoint SHA256: 5456a76b84d558e9983a59b9f1d3060ba245e0d36d60883519654f809996dbc5.
- State-dict SHA256: d2ffbc03c7cb2826fef2175c95f48725707ec6d791eaeffbd6f59bc507b8595a.
- Prediction repeat SHA256: 9f080ddb9cc55766e607b65d0bdef7ef0f924082b81e1afc2cce3c1799d02e67.
- No predictive covariance/NLL/calibration/GMM or formal superiority is claimed yet.
- Full Stage4 regression: 44/44 PASS.
- Implementation files: 31.
- Implementation SHA256: 7f42e0cf534ed3ffb8d69e7230f5e04eecde8639428911017af771c7b2993def.

## Block 4.4 — Gaussian probabilistic GRU

Status: PASS / FROZEN

- The frozen Block4.3 fit/development caches and fit-only normalization are reused without regeneration.
- The probabilistic predictor uses the same causal target, measurement-covariance, multi-agent and map inputs as the deterministic GRU.
- Stage2 R_t remains measurement uncertainty input; predictive uncertainty is a separate learned Gaussian output.
- The learned distribution is a single full 3-D Gaussian at 0.1/0.3/0.5/1.0 s.
- Predictive covariance is parameterized by a six-parameter Cholesky factor per horizon with strictly positive diagonal.
- The Gaussian backbone and mean head initialize exactly from the frozen deterministic Block4.3 checkpoint.
- Training uses full Gaussian NLL and the frozen class-balanced fit sampler.
- Model selection uses development metric Gaussian NLL only, with development ADE as tiebreaker.
- Raw 50/80/90/95/99 empirical coverage is measured but is not used for training/model selection.
- No calibration is applied in Block4.4; calibration is reserved for Block4.5.
- Formal N=120 validation remains untouched.
- Initial development Gaussian NLL: 4.984338414.
- Best development Gaussian NLL: 3.153939398.
- Best development planar ADE: 1.547782888 m.
- Best epoch: 12.
- Checkpoint SHA256: 49ff64d145eaa633f295c16f660df380c35383e7e3b61279a5aad7cd700d619f.
- State-dict SHA256: 1d66cf082d0d9be41319f7dcbe18fc259910de7f45de006f9043129837b86be3.
- Probabilistic inference SHA256: a0e96b4d9ba051bb3ceb0e1270d592fc13cfa0b3b4ddeb3dc16ed7a29a82e66a.
- Exact checkpoint inference repeat: PASS.
- Full Stage4 regression: 60/60 PASS.
- Implementation files: 40.
- Implementation SHA256: b9ea474443d3a54235f09cd4e2b616374b37456f2b9e080f5334f76f4dd7c432.

## Block 4.5 — Trajectory uncertainty calibration

Status: PASS / FROZEN

- The full frozen Block4.1 calibration partition of 7,209 scenarios is used.
- Fit/development/formal validation data are not used to fit the calibrator.
- Calibration inference reuses the frozen Block4.4 Gaussian checkpoint and Block4.3 fit-only normalization.
- Per-scenario calibration inference shards are resumable.
- Calibration changes predictive covariance only; predictive mean and Stage2 measurement covariance R_t are unchanged.
- Four independent scalar variance scales are fitted, one per 0.1/0.3/0.5/1.0-s horizon.
- Primary calibration objective is macro absolute coverage error at 50/80/90/95/99% confidence.
- Coverage-event Brier semantics are frozen as (inside_indicator - nominal_p)^2.
- Calibration-set macro ECE does not exceed raw macro ECE.
- Calibrated predictive covariance remains SPD.
- Exact calibration fitting is reproducible.
- Frozen Gaussian inference repeats exactly on four calibration scenes.
- Formal validation remains untouched; calibration generalization is deferred to Block4.8.
- Raw calibration macro ECE: 0.071127217.
- Calibrated macro ECE: 0.049012135.
- Raw coverage-event Brier: 0.166878901.
- Calibrated coverage-event Brier: 0.136294135.
- Raw calibration metric NLL: 3.093513818.
- Calibrated metric NLL: 2.939034469.
- Variance scales: [1.2347064500315355, 1.3451250295202921, 1.354822057728461, 1.2829700217319266].
- Calibrator content SHA256: 616ad562c699f29103157b27d79ecc77570634be9b660f69e94bf7e9606cfdf7.
- Calibrator file SHA256: 508ff2e3fbcfafe8e001155340c25baaf3772fe2561a8022a9ed1cf780e66087.
- Full Stage4 regression: 76/76 PASS.
- Implementation files: 49.
- Implementation SHA256: 482fa179b958f05526657d9970346b3ca699c649b77783f63d4c8de2fd7be7cf.

## Block 4.6 — Trajectory-level GMM ablation

Status: PASS / FROZEN

- The mandatory calibrated Gaussian from Block4.5 remains frozen and is not automatically replaced.
- A K=3 trajectory-level GMM is implemented and trained using the same frozen causal input/cache and fit-only normalization.
- One mixture weight applies to each complete future trajectory mode; independent horizon mode switching is not allowed.
- Every component has a full 3-D SPD Cholesky covariance at every prediction horizon.
- Initialization copies the frozen Block4.4 Gaussian encoder/fusion/covariance heads and uses only a small deterministic symmetry-break offset between component means.
- No maneuver labels, perfect track IDs, future states, tracks_to_predict or objects_of_interest are model inputs.
- Training objective is joint trajectory GMM NLL.
- Development model selection uses joint GMM NLL; mixture-mean ADE is only a tiebreaker.
- minADE, 1.0-s minFDE, mixture entropy, effective mode count and mode separation are measured.
- Mode diversity is diagnostic and not a post-hoc closure threshold.
- The GMM is not calibrated or selected for downstream use in Block4.6. If later selected, calibration is required first.
- Formal validation remains untouched until Block4.8.
- Initial joint GMM NLL: 11.975549276.
- Best joint GMM NLL: 9.193963769.
- Mixture-mean development ADE: 1.548554899 m.
- MAP-component development ADE: 1.549374944 m.
- minADE: 1.359680885 m.
- 1.0-s minFDE: 2.040403398 m.
- Effective mode count: 2.802933022.
- Mean 1.0-s pairwise mode separation: 0.943001508 m.
- Best epoch: 1.
- Checkpoint SHA256: 5aebeaf40d522d58345d424ae2557d6f87e2c02bdc26c23635cc6e3a28cbe3ee.
- State-dict SHA256: 3bd803104a5de694b9c6073fa651230e16a08899db14894e0f1fd1e2f11c1a97.
- Prediction SHA256: 0b35669827e92e027349b385fa9249c6b10833b01fc46033262ef0a4f0830d88.
- Exact inference repeat: PASS.
- Full Stage4 regression: 92/92 PASS.
- Implementation files: 58.
- Implementation SHA256: bf7ba0ef8c83d0b6a8d5ef2acd62419035f4367b17d4d6844f7a99c288f1c961.

## Block 4.7 — Measurement/context uncertainty ablations

Status: PASS / FROZEN

- The exact six Cartesian H0 measurement-covariance features R_t are frozen at indices 6–11.
- Measurement uncertainty is separate from learned predictive uncertainty.
- Measurement uncertainty scalar for analysis is the latest causal target sqrt(trace(R_t)/3), in m.
- Measurement-to-predictive uncertainty correlations and equal-count uncertainty bins are measured on development only.
- Physical R_t sensitivity evaluates positive covariance variance scales 0.5/1/2 over the complete observed target and neighbour histories.
- no-R_t, actor-only and no-map Gaussian predictors are retrained using the same fit cache, fit-only normalization, class-balanced sampler, seed, optimizer and development model-selection rule as the full Gaussian.
- no-R_t uses zero standardized covariance features, equivalent to fit-mean imputation and removal of sample-specific R_t information.
- actor-only removes all neighbour information; no-map removes all scene-varying map information.
- Ablation performance direction is reported, not post-hoc gated.
- Formal N=120 validation remains untouched until Block4.8.
- Full Gaussian ADE: 1.547782888 m.
- no-R_t ADE: 1.552895580 m.
- actor-only ADE: 1.648196881 m.
- no-map ADE: 1.615293676 m.
- Full Stage4 regression: 108/108 PASS.
- Implementation files: 65.
- Implementation SHA256: 4b32edff0c08f23de4ec5109b46b64a8eccc5357da0deb4fc516b72fe3cad006.

## Block 4.8 — Frozen formal N=120 evaluation

Status: PASS / FROZEN EVALUATION

- The immutable Stage3 formal N=120 manifest is reused exactly.
- The same frozen full-degraded Stage2 observation interface and Stage3 estimated-GNN association feed all Stage4 neural models.
- No formal scenario contributes to training, normalization, model selection or calibration fitting.
- All causal neural predictions are generated before tracks_to_predict is accessed for evaluator filtering.
- Formal ADE follows Stage3's available matched actor-horizon micro-mean semantics; target availability/recall is reported separately.
- Deterministic GRU, raw Gaussian, calibrated Gaussian and GMM are evaluated on the same neural target population.
- Raw and calibrated Gaussian NLL, 50/80/90/95/99 coverage, coverage ECE and coverage-event Brier are measured without recalibration.
- GMM remains an uncalibrated multimodal ablation and is not selected downstream here.
- Classical comparisons reuse the exact frozen Stage3 formal report; classical algorithms are not rerun.
- Calibrated Gaussian formal ADE: 1.984895283 m.
- Classical baselines beaten: ['CA', 'CTRV', 'CV', 'IMM_CV_CA_CTRV', 'Kalman_EKF'].
- PDF performance gate: True.
- Internal CV gate: True.
- Raw formal macro ECE: 0.183620465.
- Calibrated formal macro ECE: 0.111655270.
- Formal ECE non-worsening: True.
- Exact repeated formal inference: 4/4 PASS.
- Full Stage4 regression: 114/114 PASS.
- Implementation files: 74.
- Implementation SHA256: f21abaed100d1337ad95e6005298727ab5e103d10d3a173b7d2e561d886dc88d.

## Block 4.9 — Reproducibility and artifact freeze

Status: PASS / FROZEN

- Independent Block4.9 Part1 reconstructed all 120 formal shards and reproduced the exact Block4.8 formal content SHA.
- Four formal scenes are selected by the deterministic rule: first four non-empty scenes in frozen manifest order.
- Each probe executes in a separate fresh Python process and reloads normalization plus deterministic, Gaussian and GMM checkpoints directly from frozen artifacts.
- Each process rebuilds the real full-degraded Stage2 / Stage3-GNN causal input path for its scenario.
- The combined pre-evaluator deterministic/Gaussian/GMM prediction SHA matches the frozen Block4.8 per-scene SHA exactly for all four probes.
- tracks_to_predict is not accessed by the fresh-process reproducibility probes.
- No training, model selection, normalization refit or calibration refit occurs in Block4.9.
- Probe-plan SHA256: ac16063f47979361391c013f418744c40c9f6b6d4e1caa94dff87b68b7609813.
- Stage4 reproducibility manifest SHA256: 135a3c60e5a2c4419125fa0daa6d2d55452f923255a04698f0bc1aaca5f474b8.
- Final implementation SHA256: e1c779c05155dd86794f405a8567c994f0acac617015f717944e5ff6868491cd.
- Full Stage4 regression: 124/124 PASS.

## Stage 4 Final Closure

Status: COMPLETE / FROZEN

- Blocks 4.0 through 4.10 completed with all mandatory Stage4 acceptance criteria PASS.
- Mandatory acceptance matrix: 14/14 PASS.
- Frozen formal evaluation: 120 scenarios, 494/547 matched targets.
- Calibrated Gaussian formal ADE: 1.984895283 m.
- Formal macro coverage ECE improved from 0.183620465 to 0.111655270.
- Calibrated Gaussian is the frozen default Stage5 trajectory posterior.
- GMM is preserved as a multimodal diagnostic/ablation and is not selected downstream post hoc.
- Measurement covariance R_t remains a model input and is distinct from learned predictive covariance.
- Fresh-process exact reproducibility: 4/4 PASS.
- Final Stage4 regression: 130/130 PASS.
- Final Stage4 implementation SHA256: 64526bcd8f64772a04183583b63facace2db45a7311ac96412cbd19114ac9f91.
- Canonical closure SHA256: 570da4feb918b1025b5e85cc919360d922b471c468c85b3844f13fb7774e7c2f.
- Stage4→Stage5 handoff SHA256: 491bce010d35c2a394f879ecf35ed26ef1de92f7fff1465dcbf46b072a87c6fd.
- Final freeze-manifest SHA256: 88e3f290e1f7c1d037684adc132ea8ae78af057b3e3ab564fb29516687eb25e7.
- Stage5 has not been started by this closure block.
