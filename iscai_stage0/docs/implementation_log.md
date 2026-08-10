# Implementation Log

## Scope commitments — approved before Stage 0

1. The Part-A numerical optical/waveform assumptions remain the reproducibility baseline for the mandatory core. A physically consistent blue-visible redesign is optional and must not block Part B.
2. The primary Part-B observation mode is geometry-derived range/radial-velocity/angle with SNR/CRLB-conditioned measurement covariance. Full WOMD -> PC-FMCW waveform -> RDM -> CFAR is an optional strong extension. Existing waveform Doppler ambiguity must block invalid high-speed waveform experiments, not the main pipeline.
3. The mandatory probabilistic baseline is a calibrated Gaussian predictor. GMM/multi-modal forecasting is a strong ablation/extension and becomes mandatory only if results show the unimodal Gaussian is inadequate.
4. Part-A issues are split into:
   - blocking conventions to freeze before use: one-way vs round-trip delay, oracle-GDF naming, waveform parameter manifest, Doppler validity limits;
   - documented limitations/side quests: true GDF redesign, blue-visible optical redesign, new high-speed waveform/chirp schedule, CFAR extensions.
   Part B will not become a full reimplementation of Part A.
5. `tracks_to_predict` is for benchmark target evaluation, not causal online communication-receiver selection. Receiver selection will be defined separately.
6. Final scope retains: measurement covariance as predictor input; receiver-aware posterior; CV/Kalman/IMM/MHT baselines; deterministic and probabilistic forecasting; fixed/adaptive Top-K; optical pointing -> gain -> SNR -> DPSK BER -> effective rate; reactive/predictive class-aware ADB; shared-posterior joint evaluation; latency/failure-case analysis; DeepSense external validation.
7. Stage 1 and all training are blocked until Stage 0 completion criteria are completed and approved.

## Stage 0 status

- Status: STARTED — environment/dataset audit skeleton only.
- Dataset root expected by configuration: `/waymo`.
- Confirmed dataset fields: none yet from the local dataset copy.
- Versions: pending execution inside the dataset container.
- Known issue: the current assistant execution environment does not expose `/waymo`; dataset audit must be run inside the user's dataset container.
- Next exact step: run the environment and dataset-layout audits inside that container and inspect the generated JSON reports before adding schema/alignment code.


## Storage / data strategy — binding constraint

- The server has a hard total storage limit of 1 TB SSD.
- The complete WOMD-LiDAR dataset MUST NOT be downloaded or retained locally
  in addition to the existing WOMD Scenario dataset.
- WOMD v1.3.0 Scenario remains the canonical locally stored motion source.
- WOMD-LiDAR must be retrieved selectively by exact scenario_id.
- Stage 0 downloads only the small validation subset required for LiDAR
  schema, timestamp, calibration, coordinate-alignment, visualization,
  and point-density verification.
- Larger later experiments must use a bounded-cache / streaming strategy:
  deterministic scenario lists -> bounded raw LiDAR batch -> compact derived
  features/statistics -> optional raw-cache eviction.
- Every downloaded LiDAR item must be recorded in a manifest containing at
  least dataset version, split, scenario_id, source object/path and file size;
  checksum/etag must also be stored when available.
- No pipeline component may assume that the complete WOMD-LiDAR corpus exists
  locally.
- Selective retrieval must preserve train/validation/test separation and must
  never select samples using future labels or evaluation outcomes.

### WOMD-LiDAR release/layout confirmation

- Canonical motion source: WOMD Scenario v1.3.0.
- Canonical LiDAR sidecars: WOMD-LiDAR from the same Motion v1.3.0 release.
- Official bucket confirmed:
  gs://waymo_open_dataset_motion_v_1_3_0/
- Confirmed v1.3.0 layout:
  uncompressed/lidar_and_camera/{training,validation,testing}/
- LiDAR matching uses the official {scenario_id}.tfrecord convention.
- Full WOMD-LiDAR corpus will not be stored locally.
- Stage 0 uses a deterministic bounded validation subset.
- TensorFlow/AVX is not required for the Stage-0 audit:
  compressed range images may be decoded using the documented
  DeltaEncodedData/zlib format and NumPy-equivalent geometry.
- The TF-free implementation will later be regression-tested against
  official womd_lidar_utils on an AVX-capable environment.
## Stage 0 closure / Stage 1A start — superseding status (2026-08-09T18:43:59+03:00)

- Stage 0 status is COMPLETE/FROZEN.
- Canonical paired corpus:
  `/home/agni/waymo/data/paired_womd_lidar_v1_3_0`
- Frozen exact paired counts:
  - training: 72,085
  - validation: 44,097
  - total: 116,182
- The historical entries above stating `Stage 0: STARTED`,
  `Confirmed dataset fields: none yet`, and the earlier selective-LiDAR
  local-storage assumption are superseded by the frozen Stage-0 closure,
  manifests, reconciliation artifacts, and canonical paired corpus.
- The approved scope commitments remain binding; only obsolete status/data-layout
  statements are superseded.
- Stage 1A artifact semantics:
  `causal_womd_annotation_upstream`.
- Stage 1A `sensor_realistic = false`; Stage-2 PC-FMCW-like observations and
  measurement covariance will later provide the sensor-realistic predictor inputs.
- Track IDs remain bookkeeping/oracle-association metadata and are not numeric
  model features.
- `tracks_to_predict` and `objects_of_interest` remain benchmark/evaluation
  metadata and are not causal receiver-selection or model inputs.
- Stage 1A Blocks 1-2 regression suite: 49 tests PASS.
- Existing frozen Stage-0 TFRecord/Scenario reader will be reused rather than
  reimplemented.
- Next implementation gate: one-real-scenario causal Stage-1A smoke extraction.
  No corpus scan, package installation, LiDAR extraction, or scaling is approved
  by this entry.

## Stage 1A closure — COMPLETE/FROZEN

- Status: COMPLETE/FROZEN.
- Artifact semantics: `causal_womd_annotation_upstream`; `sensor_realistic = false`.
- Final deterministic tiny pilot: first 3 records of the frozen validation selection manifest.
- Final regression: 52 core + 8 LiDAR tests PASS.
- Real gates PASS: actor adapter, causal maps, causal actor-LiDAR MVP, strict future-mutation hash invariance, headlamp-centric visualization, 3-D/elevation visualization.
- Free space at closure: 1071.071 GiB; 250 GiB hard reserve PASS.
- Stage 1A does not claim sensor-realistic PC-FMCW measurements; those begin in official Stage 2.
- Next stage: PC-FMCW-like observations + measurement covariance.
