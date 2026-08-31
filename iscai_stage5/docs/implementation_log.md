
## Block 5.0 — Stage5 bootstrap and contract freeze

Status: PASS / STRUCTURAL CONTRACT FROZEN

- Frozen Stage4 calibrated Gaussian is the mandatory Stage5 trajectory posterior.
- Stage4 training, normalization fitting, recalibration and post-hoc GMM selection remain forbidden.
- Primary receiver policy: nearest causal vehicle ahead; connectivity is a constructed experimental assumption, not a WOMD label.
- Receiver geometry cases: centroid baseline, known offset and uncertain offset.
- Receiver offset uncertainty must be propagated jointly with trajectory uncertainty.
- Stage5 horizons: 0.1/0.3/0.5/1.0 s.
- Codebook sizes: 16/32/64.
- Adaptive Top-K requested masses: 90/95/97.5/99%; nominal default 95%.
- Formal N=120 is evaluation-only and cannot be used to tune receiver policy, codebook, thresholds or fallback parameters.
- Numerical controller parameters explicitly listed in the development-freeze contract must be frozen before Block5.8.
- Optical chain is fixed as pointing error → gain → received power → SNR → DPSK BER → effective rate. Part-A freezes the underlying link quantities/raw data rate; Stage5 derives the overhead-aware effective-rate mapping and must freeze it before formal evaluation.
- Part-A notebook source and relevant evidence cells are hashed and frozen for later Block5.6 numerical cross-check.
- Predictive ADB is Stage6 scope; full joint beam+ADB evaluation is later scope; DeepSense remains Stage8 scope.
- No training, inference or formal evaluation was performed in Block5.0.
- Stage5 contract SHA256: 04fd32e4bc6e160f2627f465a7820c74e19ca2e1a1f47f23ed1884f842cb7db4.
- Part-A source contract SHA256: 80a06892e1721e28e2599c20acc29c9bea44fd74e256b8af7eadd02416600fca.
- Block5.0 implementation SHA256: f92a4112ef9c23b5db4c24260e18b2e1da2749c50ac125f623101e45d688ab02.

## Block 5.1 — Communication receiver and geometry

Status: PASS / FROZEN

- Primary communication receiver policy: nearest causal vehicle ahead in H0.
- Connectivity is a constructed hypothetical experimental assumption because WOMD does not provide a measured communication-connectivity label.
- Receiver selection consumes only current causal associated actor estimates and current semantic metadata.
- tracks_to_predict, future truth, perfect track IDs and oracle connectivity are not receiver-selector inputs.
- Receiver geometry modes remain centroid, known offset and uncertain offset.
- Receiver-placement covariance remains distinct from Stage2 measurement R_t and Stage4 predictive covariance.
- No additional finite receiver-range cutoff is used in the primary policy; distance is reported/sliced rather than tuned as an eligibility threshold.
- No formal N=120 data, Stage4 inference, training or recalibration was used in Block5.1.
- Frozen receiver-policy SHA256: b5b6fc2b0a3c4575ed9567aff15f36f3430f4887862c245f045fae7cf9e349c0.
- Block5.1 implementation SHA256: 604c0ea15353830ed6dd74b40ff7e42313bf9f86c816aa1e76f0057ab6bc8a53.

## Block 5.2 — Receiver-aware angular posterior

Status: PASS / FROZEN

- Primary receiver/angular posterior uses deterministic Monte Carlo.
- Frozen Stage4 calibrated Gaussian means remain unchanged; covariance is scaled with the frozen Block4.5 alpha_h values.
- No unsupported cross-horizon covariance is invented; trajectory samples follow the product of the frozen per-horizon Gaussian marginals.
- Receiver heading is constructed from samplewise predicted trajectory tangents with a 0.25 m/s low-speed carry-forward fallback; future GT heading is never used.
- Uncertain receiver placement uses one body-frame offset draw per MC sample shared across all horizons.
- Azimuth statistics use circular means/wrapped residuals.
- MC sample count is frozen from the non-formal Block4.4 development posterior using a preregistered convergence rule.
- Formal N=120 was not accessed and Stage4 model inference was not rerun.
- MC sample count: 2048.
- Angular policy SHA256: 846f6bf3d3419a2ea99fabc6293514726388bf60a31bb2a889aaf18b487fd82e.
- MC convergence SHA256: 2a66b9128b06250b90b61ed0cd3e0a089b86946bd34c0eac4483de150306fc5e.
- Block5.2 implementation SHA256: e6a6c3996a1c09b493ebfe61516f7a4619994dd1aed34c7ada7e17f3711f9878.
