# Stage 9 — Safety-Consequence-Aware Joint Beam Management and Predictive ADB

This directory is the curated paper-facing publication projection of the final
Stage-9 implementation.

It is intentionally **not** a mirror of every Stage-9 development, failed,
superseded, or diagnostic artifact retained on the execution server.

## Scientific scope

Stage 9 reuses the frozen sensing-informed future-motion posterior from the
preceding pipeline as a shared uncertainty carrier for two primary tasks:

1. beam uncertainty / communication policy reasoning; and
2. future traffic-criticality reasoning.

The primary Stage-9 scientific scope is **C1 + C2**.

The C3 recovery-aware predictive ADB extension is retained as a **secondary
FORMAL profile-sensitivity analysis**, using the frozen
`V5_RHO2_DIRECT_RAW_PRIORITY` policy. It is not the original confirmatory
winner and must not be reported as a globally successful H5 result.

## Executed C2 formulation

The executed primary FORMAL C2 policy minimizes selected beam count

    K = sum_b z_b

subject to the nominal reliability condition

    sum_b p_b z_b >= 0.95

and, when raw critical probability is positive, the critical conditional
coverage condition

    sum_b u_b_raw z_b >= 0.99 * Pcrit_raw.

For M = 4096 Monte-Carlo samples, the corresponding exact integer rules are

    nominal missed samples <= 204

and

    critical missed samples <= floor(C / 100),

where `C` is the number of raw critical Monte-Carlo samples.

The older joint-absolute formulation

    P(Ccrit intersect Bmiss | D_t) <= 0.01

is preserved only under `configs/lineage/`. It is **not** the C2 constraint
executed in the primary FORMAL chain.

## Directory structure

- `src/iscai_stage9/`
  Runtime support source required by the authoritative Stage-9 execution
  snapshots.

- `scripts/authoritative/`
  Byte-identical source snapshots bound into the final authority chain.
  These preserve the original server execution paths and are provided for
  provenance and exact code-to-result traceability.

- `configs/current/`
  Current Stage-9 contracts and scope/configuration authorities.

- `configs/lineage/`
  Historical configurations retained only to document superseded semantics.

- `results/paper_ready/`
  Compact aggregate tables, paper-ready summaries, claim-evidence mapping,
  mathematical specification, limitations, and provenance.

- `reproducibility/authority/`
  Seals, registries, reports, and exact authority artifacts referenced by the
  final package.

- `reproducibility/preformal/`
  Pre-FORMAL claim/scope freezes documenting the transition from the original
  C3 confirmatory plan to the final C1+C2-primary / C3-secondary scope.

- `reproducibility/final/`
  Final Stage-9 reproducibility seal.

## Dependencies on earlier stages

The final Stage-9 snapshots reuse code from previously published stages,
including Stage 4 and Stage 6.

Four Stage-4 Python source files that existed on the execution server but were
missing from the earlier public repository are added in this publication
update:

- `iscai_stage4.data`
- `iscai_stage4.data.neural_inputs`
- `iscai_stage4.data.real_pipeline`
- `iscai_stage4.data.supervision`

They are copied byte-for-byte from the execution tree. No previously published
Stage-0-to-8 file is replaced or modified for this dependency closure.

See:

- `reproducibility/RUNTIME_DEPENDENCIES.json`
- `reproducibility/EXTERNAL_STAGE4_RUNTIME_DEPENDENCIES.json`

## Data boundary

Raw WOMD / WOMD-LiDAR and raw DeepSense datasets are not redistributed here.

The published repository contains code, frozen contracts, compact aggregate
results, authority metadata, and reproducibility hashes. Large per-scene
evaluation caches and future-ground-truth materializations are intentionally
excluded.

Future ground truth was evaluator-only at the FORMAL outcome stage; it was not
used as a controller input or for post-FORMAL policy selection.

## DeepSense boundary

The inherited DeepSense evidence is communication-only measured-mmWave
validation. It does not validate optical headlamp behavior, glare, human
reaction/recovery, traffic-criticality, collision, or crash-reduction claims.

## Reporting boundary

The final v2R1 package is a paper-ready append-only packaging repair. It reports
that scientific results were not changed by the repair.

Its sealed state remains:

`FROZEN_COMPLETE_STAGE9_916_V2R1_PACKAGING_REPAIR_AWAITING_INDEPENDENT_CERTIFICATION`

Accordingly, this repository must not state that independent certification of
v2R1 has already completed unless a later explicit certification artifact is
added.

See `AUTHORITY.md` and `REPRODUCIBILITY.md` for the exact authority and
reproduction boundaries.
