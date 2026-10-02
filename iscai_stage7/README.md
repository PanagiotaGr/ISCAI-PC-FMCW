# ISCAI Stage 7 — Frozen Joint Communication–Illumination Evaluation

Stage 7 performs the frozen joint system-level evaluation of the communication
and illumination branches produced by the earlier stages.

## What this stage does

The stage locks non-oracle controller outputs before future ground truth is
used for scoring. It then evaluates the frozen systems jointly, computes
paired statistics, runs frozen sensitivity sweeps, and records the available
latency/resource evidence.

The formal evaluator contains **120 scenarios** and **600 joint rows** across
five evaluated systems.

## Main responsibilities

- verify shared-posterior identity across the joint pipeline;
- freeze the pre-outcome experimental protocol;
- lock causal/non-oracle controller outputs;
- run the final formal joint evaluator;
- compute paired statistical summaries with **10,000 bootstrap resamples**;
- run codebook, coverage, and uncertainty sweeps;
- verify exact frozen reproduction;
- audit latency/resource evidence without inventing unavailable measurements.

## Frozen evaluation boundary

Future ground truth is used only to score already-frozen controller outputs.
It is not used as an online controller feature, and controllers are not rerun
after future-ground-truth access.

The final evaluation status is **PASS_FROZEN**.

A complete joint end-to-end latency value is not reported because the frozen
authority set does not contain all required timing components for data
loading, preprocessing, predictor inference, beam selection, and ADB
generation.

## Directory structure

- `configs/` — Stage-7 experimental/freeze contracts.
- `scripts/` — lock, evaluator, statistics, sweep, and latency/resource
  closure scripts.
- `reports/` — shared-posterior, protocol, formal-evaluation, statistics,
  sweep, and latency/resource reports.
- `artifacts/` — frozen joint-evaluation artifacts.

**Status: PASS_FROZEN, with the documented end-to-end latency limitation.**
