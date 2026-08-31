# ISCAI Stage 3

Canonical clean Stage-3 implementation built on the frozen Stage-2
PC-FMCW-like observation interface.

The legacy implementation is isolated in:

    /home/agni/waymo/iscai_stage3_panagiota

and is not an implementation dependency.

Stage 3 contains only classical tracking / motion baselines:

- CV
- CA
- CTRV
- Kalman
- IMM with CV / CA / CTRV modes
- Multidimensional Hough Transform

The main fair comparison consumes the same Stage-2 unlabeled noisy
observation stream.

Evaluator truth is scoring-only.

Neural trajectory prediction, beam control and ADB belong to later
stages.
