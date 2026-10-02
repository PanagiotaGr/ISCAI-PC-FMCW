# Stage 9 Authority and Claim Boundary

This file defines which Stage-9 branch is authoritative for paper reporting.

Authority is determined by explicit binding into the executed scientific
chain, **not** by filename recency or by choosing the largest version number.

## Final lineage

The reporting lineage is:

Stages 0–8 frozen foundation

→ Stage-9 C1+C2 primary scope

→ C3 CAL evaluation with no original confirmatory winner

→ pre-FORMAL scope narrowing

→ C3 retained as secondary FORMAL profile sensitivity using
  `V5_RHO2_DIRECT_RAW_PRIORITY`

→ Stage 9.13 FORMAL evaluation

→ Stage 9.14 communication-only DeepSense boundary

→ Stage 9.15 statistical / robustness / failure / latency disposition

→ Stage 9.16-v2R1 paper-ready packaging repair

The current v2R1 seal still records that independent certification is required.

## Primary and secondary scope

Primary:

- C1: shared calibrated/frozen future-motion posterior used as the common
  uncertainty interface.
- C2: minimum-cardinality beam probing under nominal and stricter
  critical-conditional reliability requirements.

Secondary:

- C3: recovery-aware predictive ADB profile sensitivity.
- C3 must not be described as the original confirmatory winner.
- FAST, NOMINAL, and SLOW profiles remain part of the reporting set.

## Executed C2 mathematics

For binary beam-selection variables `z_b`, the executed objective is

    minimize K = sum_b z_b.

Nominal condition:

    sum_b p_b z_b >= 0.95.

With M = 4096 this is equivalent to

    missed nominal Monte-Carlo samples <= 204.

Critical condition, when `Pcrit_raw > 0`:

    sum_b u_b_raw z_b >= 0.99 * Pcrit_raw.

Equivalently:

    missed critical samples <= floor(C / 100),

where `C` is the raw critical Monte-Carlo sample count.

`Pcrit_cal` may exist for descriptive/event-calibration provenance but is not
the primary C2 solver input.

### Superseded C2 branch

The historical Block-9.7 joint-absolute condition

    P(Ccrit intersect Bmiss | D_t) <= 0.01

is retained only for lineage.

Files under `configs/lineage/` must not be described as the executed primary
FORMAL C2 policy.

## H1–H6 reporting status

### H1

Confirmatory H1 status:

`NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM`

Later materialized RAW-vs-CAL results can be reported only as
**non-confirmatory numerical evidence** of reduced macro absolute coverage
error.

### H2

`NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS`

The FORMAL evaluator observed zero ground-truth critical events among valid
critical labels. This must not be interpreted as zero underlying risk.

### H3

`PARTIALLY_EVALUABLE_RESOURCE_COMPONENT_ONLY`

The resource component is evaluable: the C2 policy can be compared with the
fixed q=.99 policy in terms of probing resources / beam count.

The critical-reliability component is not evaluable because the FORMAL cohort
contains zero observed critical events.

### H4

`NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS`

No claim of reduced realized critical miss/outage is authorized.

### H5

C3 is a secondary, human-profile-sensitive result.

- FAST: no clear reduction in either frozen comparison.
- NOMINAL: reductions supported in both comparisons.
- SLOW: reductions supported in both comparisons.

Thus the six frozen pairwise comparisons yield four supported reductions and
two no-clear-difference results. This is not a global H5 confirmation.

### H6

The frozen Stage-6 core side-effect gates pass with the Stage-9 bootstrap
context. This is a bounded side-effect result and not a universal illumination
or safety-performance claim.

## Technical recovery boundary

The primary full cohort remains authoritative.

Twenty-two scenes were involved in documented physical future-GT rereads
during technical recovery. Their exact exclusion is retained as a mandatory
secondary sensitivity and must not replace the primary full-cohort result.

## Latency boundary

The final Stage-9 evidence does not authorize claims of:

- real-time execution,
- low latency,
- end-to-end latency improvement, or
- computational speedup.

The required FORMAL timing carriers were not available for those claims.

## DeepSense boundary

DeepSense supports the communication-policy evidence only.

It does not validate:

- optical PC-FMCW/headlamp behavior,
- glare,
- human response or recovery,
- C3 safety effects,
- traffic-criticality,
- collision probability, or
- real-world crash reduction.

## Final package state

The Stage-9 v2R1 package is a packaging/completeness repair that records:

- no scientific-results change,
- no new bootstrap,
- no new statistics,
- no new RNG,
- no new model/policy/solver execution,
- no retuning,
- no threshold change, and
- no new future-GT read.

Its state is still **awaiting independent certification**.
