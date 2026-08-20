# ISCAI Stage 7 — Joint Evaluation

Stage 7 performs joint evaluation of the shared probabilistic trajectory posterior
for both communication beam management and predictive ADB control.

Main inputs:
- Stage 4 final probabilistic trajectory model / posterior
- Stage 5 real beam-management outputs
- Stage 6 real predictive ADB outputs

Main evaluation goals:
- common posterior consistency
- communication reliability
- beam probing overhead
- ADB glare protection
- pedestrian/cyclist visibility
- over-masking
- latency
- uncertainty sweeps
- latency sweeps
- codebook sweeps
- statistical confidence intervals

Stage 7 must not retrain or silently modify Stage 4, Stage 5, or Stage 6 models.
It evaluates their joint behavior and trade-offs.
