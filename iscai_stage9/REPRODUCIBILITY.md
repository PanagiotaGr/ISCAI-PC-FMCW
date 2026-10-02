# Stage 9 Reproducibility Notes

## Purpose

The public Stage-9 tree is a curated projection of the executed and sealed
scientific chain.

It serves two distinct reproducibility goals:

1. exact provenance of the code and artifacts used in the final analysis; and
2. public inspection of the core implementation and aggregate results.

These goals should not be confused with redistributing all private/raw data or
every intermediate execution artifact.

## Exact authoritative snapshots

Files under:

    scripts/authoritative/

are copied byte-for-byte from the execution server.

Do not edit those files and still describe them as the exact executed source.
Their SHA256 values are recorded in the final manifests and in:

    reproducibility/SHA256SUMS.txt

The snapshots intentionally preserve original execution-root assumptions such
as `/home/agni/waymo`.

Those absolute paths are provenance from the original environment, not secrets
and not a statement that every script is directly portable to an arbitrary
checkout.

## Python source layout

For repository-local inspection/imports, Stage-9 source is under:

    iscai_stage9/src/iscai_stage9/

The main direct Stage-9 runtime support module required by the executed C1/C2
source is:

    iscai_stage9.criticality_geometry_v1

Earlier-stage Python packages are under their corresponding `src/` trees.

A typical local import setup is therefore:

    export PYTHONPATH="$PWD/iscai_stage9/src:$PWD/iscai_stage4/src:$PWD/iscai_stage6/src:${PYTHONPATH:-}"

This does not by itself reproduce the full server run; raw datasets, private
execution-time caches, and some large sealed per-scene artifacts are not
redistributed.

## Stage-4 dependency closure

The authoritative Stage-9 scripts require:

    iscai_stage4.data.neural_inputs
    iscai_stage4.data.real_pipeline
    iscai_stage4.data.supervision

These modules existed on the execution server but were absent from the prior
public repository.

The paper-code publication adds the exact server copies together with the
package `__init__.py`.

Their provenance and hashes are recorded in:

    reproducibility/EXTERNAL_STAGE4_RUNTIME_DEPENDENCIES.json

No pre-existing Stage-0-to-8 file is overwritten.

## Verifying the curated Stage-9 tree

From the repository root:

    cd iscai_stage9
    sha256sum -c reproducibility/SHA256SUMS.txt

This checks the publication projection hashes.

The sealed artifacts inside `results/paper_ready/` and
`reproducibility/authority/` may contain original absolute server paths.
Changing those paths would invalidate the corresponding original artifact
hashes, so they are intentionally preserved.

## Data requirements

Raw datasets are not included.

The scientific pipeline uses earlier repository stages together with external
datasets such as WOMD / WOMD-LiDAR. DeepSense evidence is inherited only for
measured-mmWave communication-policy support.

Users must obtain external datasets under their original licenses and configure
their local paths independently.

## FORMAL/future-GT boundary

Causal decisions were frozen before future-ground-truth outcome evaluation.

Future GT was evaluator-only and was not supplied to the controller or used for
post-FORMAL policy/threshold selection.

The repository does not redistribute the full future-GT evaluation cache.

## Statistics

The final Stage-9 statistics chain includes frozen paired scenario-level
bootstrap analyses and the mandatory documented technical-recovery exclusion
sensitivity.

Do not silently rerun or redefine hypothesis membership, failure slices,
calibration perturbations, or unavailable latency estimands and then present
those results as part of the sealed analysis.

## Independent certification

The final v2R1 reproducibility seal explicitly records that independent
certification remains required.

Therefore the present public package should be described as the final
paper-ready Stage-9 implementation package **awaiting independent
certification**, unless a subsequent certification artifact is published.
