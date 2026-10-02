# Stage 9.16-v2R1 — Paper-Ready Package Repair

## Repair scope

This package is an append-only reporting/packaging repair of the scientifically compatible Stage-9.16-v2 package.
It performs no outcome scan, RNG, bootstrap, new statistic, model run, policy run, solver run, threshold change,
baseline change, or hypothesis redefinition.

## Executed C2 mathematical formulation

The C2 formulation actually executed in the frozen 9.13A -> R4 -> FORMAL chain is:

- binary decision: `z_b in {0,1}`
- objective: `minimize K = sum_b z_b`
- nominal constraint: `sum_b p_b z_b >= 0.95`
- critical constraint when `Pcrit_raw > 0`: `sum_b u_b_raw z_b >= 0.99 * Pcrit_raw`
- exact critical integer form: `missed critical samples <= floor(C/100)`
- `Pcrit_cal` is descriptive/provenance only and is **not** a C2 solver input.

The server's joint-absolute `P(Ccrit ∩ Bmiss | Dt) <= 0.01` / `<=40 of 4096` branch is preserved as
historical non-primary lineage and must not be presented as the executed FORMAL C2 formulation.

## Full communication table

A dedicated 30-row table is packaged for all 10 frozen policies across codebooks 16/32/64.
It is copied exactly from the current sealed Stage-9.13 aggregate summary and includes coverage, joint critical
miss, mean K, beam-gain loss, received-power loss, probing overhead, outage, effective rate, and the other
already-materialized aggregate communication quantities.

No communication statistic is recomputed here.

## H1

Confirmatory status: **NOT TESTED** (`NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM`).

Later non-confirmatory numerical evidence:
- RAW macro absolute coverage error: 0.7024050191723012
- CALIBRATED macro absolute coverage error: 0.700083970619727
- CAL-RAW delta: -0.0023210485525742472
- paired bootstrap 95% CI: [-0.0026211442844785993, -0.002033862704679424]

The mandatory exact 22-scene technical-recovery exclusion remains secondary and cannot replace the full cohort.

## Critical events / H2-H4

Observed FORMAL ground-truth critical events: **0 / 24,917 valid critical labels**.
Zero observed events are not interpreted as zero risk.

- H2: not evaluable.
- H3: resource component only; full reliability claim is not established.
- H4: not evaluable.

The communication table's zero aggregate joint-critical-miss entries do not rescue these critical-conditioned claims.

## C3 / H5-H6

H5 remains a secondary profile sensitivity: FAST 0/2 clear reductions, NOMINAL 2/2, SLOW 2/2.
There is no global 6/6 H5 success claim.

H6 retains the frozen Stage-6 core-gate interpretation.

No current sealed aggregate reaction-margin distribution was materialized. No post-FORMAL reaction-margin
distribution is constructed in this repair; only the already-sealed H5 `P(C_late)` comparisons are reported.

## Event calibration

Current Block-9.4 authority is 15-bin equal-width ECE. This package includes the exact raw and calibrated
15-bin reliability tables from the sealed CAL diagnostics, in JSON and CSV form.

## Version-control / provenance rule

Authorities are selected by explicit binding into the current executed chain, not by the most recent-looking
filename or version number. Historical, failed, superseded, and intermediate Stage-9 scripts are not bulk-added
to the canonical source manifest.

## Manuscript authority

Stage-9.16-v2R1 is a repaired candidate package. It should supersede 9.16-v2 for manuscript writing only after
an independent read-only certification audit passes.
