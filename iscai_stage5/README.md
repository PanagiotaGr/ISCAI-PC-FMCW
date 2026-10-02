# ISCAI Stage 5 — Receiver-Aware Adaptive Beam Management

Stage 5 transforms the calibrated Stage-4 trajectory posterior into a
receiver-aware angular posterior and uses it for adaptive communication-beam
selection.

## What this stage does

For each communication receiver, the predicted future-position distribution is
mapped into angular uncertainty and then into probabilities over a beam
codebook. The adaptive policy selects the smallest Top-K beam set whose
cumulative predicted probability reaches the requested mass target.

The primary formal setting uses **q = 0.95** and evaluates codebooks with
**16, 32, and 64 beams**.

## Main responsibilities

- freeze the receiver-selection and geometry contracts;
- transform future-position uncertainty into receiver angular uncertainty;
- compute beam-codebook probability mass;
- implement fixed Top-K and adaptive Top-K comparators;
- map pointing/beam decisions through the communication link model;
- evaluate probing cost, coverage, and communication consequences;
- perform the formal frozen evaluation and reproducibility checks.

The communication chain includes

beam pointing → gain → received power → SNR → DPSK BER → effective rate.

## Frozen result summary

At q = 0.95 the adaptive policy maintains the frozen coverage requirement while
using substantially fewer probes than exhaustive search.

Mean adaptive probes are approximately:

- **2.54 / 16**
- **4.17 / 32**
- **7.39 / 64**

corresponding to roughly **84.1%**, **87.0%**, and **88.5%** probe reduction
relative to exhaustive probing.

## Directory structure

- `configs/` — Stage-5 contracts and frozen configuration.
- `src/iscai_stage5/` — receiver, angular-posterior, codebook, policy, and
  communication support code.
- `scripts/` — audits, materialization, formal evaluation, reconciliation,
  reproducibility, and freeze scripts.
- `reports/` — receiver/angle/codebook/policy/link/formal-evaluation reports.
- `artifacts/` — frozen Stage-5 artifacts.
- `tests/` — regression tests.

## Scientific boundary

This stage evaluates the communication policy and modeled optical link
consequences. It does not by itself validate real optical hardware.

**Formal status: PASS.**
