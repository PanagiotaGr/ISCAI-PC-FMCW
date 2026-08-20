# ISCAI Stage 2

Stage 2 owns the PC-FMCW-like observation and measurement-uncertainty
pipeline.

## Upstream dependency

Stage 1 is frozen and remains the source of:

- causal WOMD actor histories,
- ego/headlamp frames,
- receiver/actor geometry,
- Stage-1 contracts.

Stage 2 imports those through:

- iscai_stage1.actors
- iscai_stage1.geometry
- iscai_stage1.contracts

Stage 2 owns:

- iscai_stage2.observations
- iscai_stage2.pc_fmcw

Stage 2 must not modify Stage-1 causal semantics.

## Observation semantics

WOMD/WOMD-LiDAR is not measured FMCW.

The Stage-2 main mode is:

real causal traffic
-> geometry-derived ideal observables
-> synthetic PC-FMCW-like measurement model
-> SNR/CRLB covariance
-> noise / missed detections / false alarms

Measurement covariance is distinct from predictive uncertainty.

The optional full waveform/RDM/CFAR path is not required for the
core Stage-2 pipeline.
