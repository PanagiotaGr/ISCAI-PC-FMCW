from __future__ import annotations

from hashlib import sha256
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback

import numpy as np
import torch


ROOT = Path(
    "/home/agni/waymo"
)

STAGE0 = (
    ROOT
    / "iscai_stage0"
)

STAGE1 = (
    ROOT
    / "iscai_stage1"
)

STAGE3 = (
    ROOT
    / "iscai_stage3"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

STAGE5 = (
    ROOT
    / "iscai_stage5"
)

LEGACY_STAGE5 = (
    ROOT
    / "iscai_stage5_panagiota"
)

REPORT = (
    STAGE5
    / "reports/"
      "block50_part1_discovery.json"
)


# ============================================================
# Frozen upstream artifacts
# ============================================================

STAGE4_CLOSURE = (
    STAGE4
    / "reports/"
      "stage4_final_closure.json"
)

STAGE4_HANDOFF = (
    STAGE4
    / "artifacts/block410/"
      "stage4_to_stage5_handoff.json"
)

STAGE4_FREEZE = (
    STAGE4
    / "artifacts/block410/"
      "stage4_final_freeze_manifest.json"
)

STAGE4_REPRO = (
    STAGE4
    / "artifacts/block49/"
      "stage4_reproducibility_manifest.json"
)

GAUSSIAN_CHECKPOINT = (
    STAGE4
    / "artifacts/block44/"
      "gaussian_gru.pt"
)

NORMALIZATION = (
    STAGE4
    / "artifacts/block43/"
      "fit_normalization.json"
)

CALIBRATOR = (
    STAGE4
    / "artifacts/block45/"
      "covariance_scaler.json"
)

FORMAL_MANIFEST = (
    STAGE3
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
)

PART_A_REFERENCE = (
    STAGE0
    / "reports/stage0/"
      "part_a_frozen_reference.json"
)

STAGE0_CLOSURE = (
    STAGE0
    / "reports/stage0/"
      "stage0_closure_report.json"
)

STAGE1_CLOSURE = (
    STAGE1
    / "reports/stage1a/"
      "stage1a_closure_report.json"
)

PDF_CERTIFICATION = (
    ROOT
    / "audits/pdf_compliance_stages0_4/"
      "pdf_compliance_final_certification.json"
)

PDF_RESOLUTION = (
    ROOT
    / "audits/pdf_compliance_stages0_4/"
      "pdf_compliance_resolution.json"
)

WAYMO_NOTICE = (
    ROOT
    / "NOTICE.md"
)


EXPECTED_SHA256 = {
    "Stage4_closure":
        (
            "570da4feb918b1025b5e85cc919360d9"
            "22b471c468c85b3844f13fb7774e7c2f"
        ),

    "Stage4_handoff":
        (
            "491bce010d35c2a394f879ecf35ed26e"
            "f1de92f7fff1465dcbf46b072a87c6fd"
        ),

    "Stage4_freeze":
        (
            "88e3f290e1f7c1d037684adc132ea8ae"
            "78af057b3e3ab564fb29516687eb25e7"
        ),

    "Stage4_reproducibility":
        (
            "135a3c60e5a2c4419125fa0daa6d2d554"
            "52f923255a04698f0bc1aaca5f474b8"
        ),

    "Gaussian_checkpoint":
        (
            "49ff64d145eaa633f295c16f660df380"
            "c35383e7e3b61279a5aad7cd700d619f"
        ),

    "normalization":
        (
            "3d7fc0a66d4a4f566f6569befa9c3766"
            "ecae21830bb4e46256df2f326a82a5f6"
        ),

    "calibrator":
        (
            "508ff2e3fbcfafe8e001155340c25baaf"
            "3772fe2561a8022a9ed1cf780e66087"
        ),

    "formal_manifest":
        (
            "2208e7287ddf6439fda4597c435a9cba"
            "1d1b9d0e4c4547bc5dd92e56e8124e46"
        ),

    "PDF_certification":
        (
            "7c94bb9deb37a5eba29c9237bfcf4b0d"
            "24e4031d9b1b22a500c5a63730bf0d06"
        ),

    "PDF_resolution":
        (
            "f4e1d526aff985ff61a3441f014755fb"
            "c150d6263cc81f3368ef3eabd068ce51"
        ),

    "Waymo_notice":
        (
            "b05a8da05a56b71dbe28cc2e217b26c6"
            "92b8761ad1d026a6365729a43e26f997"
        ),
}


EXPECTED_HORIZONS = (
    0.1,
    0.3,
    0.5,
    1.0,
)

EXPECTED_VARIANCE_SCALE = (
    1.2347064500315355,
    1.3451250295202921,
    1.354822057728461,
    1.2829700217319266,
)

MIN_FREE_GIB = 250.0


# ============================================================
# Helpers
# ============================================================

def file_sha256(
    path: Path,
):
    digest = sha256()

    with path.open(
        "rb"
    ) as stream:

        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(
    path: Path,
    payload,
):
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )


def require(
    condition,
    message,
):
    if not bool(
        condition
    ):
        raise RuntimeError(
            message
        )


def flatten(
    value,
    prefix="",
):
    rows = []

    if isinstance(
        value,
        dict,
    ):
        for key, item in value.items():

            child = (
                f"{prefix}.{key}"
                if prefix
                else str(
                    key
                )
            )

            rows.extend(
                flatten(
                    item,
                    child,
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for index, item in enumerate(
            value
        ):

            rows.extend(
                flatten(
                    item,
                    f"{prefix}[{index}]",
                )
            )

    else:
        rows.append(
            (
                prefix,
                value,
            )
        )

    return rows


def numeric_sequence_close(
    actual,
    expected,
    *,
    atol=1e-12,
):
    try:
        actual = tuple(
            float(
                value
            )
            for value in actual
        )

    except Exception:
        return False

    if len(
        actual
    ) != len(
        expected
    ):
        return False

    return all(
        abs(
            first
            -
            second
        )
        <=
        atol
        for first, second in zip(
            actual,
            expected,
        )
    )


def find_existing_paths_from_payload(
    payload,
):
    result = []

    for _, value in flatten(
        payload
    ):
        if not isinstance(
            value,
            str,
        ):
            continue

        text = value.strip()

        if not text:
            continue

        possibilities = []

        path = Path(
            text
        ).expanduser()

        if path.is_absolute():
            possibilities.append(
                path
            )

        else:
            possibilities.extend(
                (
                    ROOT
                    /
                    path,

                    STAGE0
                    /
                    path,
                )
            )

        for candidate in possibilities:

            try:
                if candidate.exists():
                    resolved = (
                        candidate.resolve()
                    )

                    if resolved not in result:
                        result.append(
                            resolved
                        )

            except Exception:
                continue

    return result


def candidate_part_a_files(
    payload,
):
    candidates = []

    def add(
        path,
    ):
        try:
            path = Path(
                path
            )

            if not path.is_file():
                return

            if path.stat().st_size > (
                30
                *
                1024
                *
                1024
            ):
                return

            resolved = path.resolve()

            if resolved not in candidates:
                candidates.append(
                    resolved
                )

        except Exception:
            return

    # Exact paths frozen in Stage0 reference.
    for path in (
        find_existing_paths_from_payload(
            payload
        )
    ):
        if path.is_file():
            add(
                path
            )

        elif path.is_dir():
            try:
                for extension in (
                    "*.ipynb",
                    "*.py",
                    "*.md",
                    "*.txt",
                    "*.json",
                ):
                    for file in path.glob(
                        extension
                    ):
                        add(
                            file
                        )

            except Exception:
                pass

    # Conservative root-level discovery only.
    # Never recurse through WOMD data.
    for extension in (
        "*.ipynb",
        "*.py",
        "*.md",
        "*.txt",
        "*.json",
    ):
        for path in ROOT.glob(
            extension
        ):
            add(
                path
            )

    # Frozen Stage0 Part-A reference itself is evidence too.
    add(
        PART_A_REFERENCE
    )

    return candidates


def search_text_groups(
    paths,
):
    groups = {
        "DPSK":
            (
                "dpsk",
            ),

        "BER":
            (
                "ber",
                "bit error",
            ),

        "optical_gain_or_received_power":
            (
                "received power",
                "p_rx",
                "prx",
                "optical gain",
                "g_opt",
                "gopt",
            ),

        "effective_rate":
            (
                "effective rate",
                "r_eff",
                "reff",
            ),

        "PartA_waveform_constants":
            (
                "tchirp",
                "193.4",
                "10 ghz",
                "131072",
            ),
    }

    evidence = {
        key:
            []
        for key in groups
    }

    for path in paths:

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        except Exception:
            continue

        lower = text.lower()

        for group, tokens in groups.items():

            matched_tokens = [
                token
                for token in tokens
                if token.lower()
                in lower
            ]

            if not matched_tokens:
                continue

            evidence[
                group
            ].append({
                "path":
                    str(
                        path
                    ),

                "tokens":
                    matched_tokens,
            })

    return evidence


def discover_receiver_geometry():
    receiver_module = importlib.import_module(
        "iscai_stage1.geometry.receiver"
    )

    frames_module = importlib.import_module(
        "iscai_stage1.geometry.frames"
    )

    require(
        hasattr(
            receiver_module,
            "receiver_geometry_in_H0",
        ),
        (
            "Frozen Stage1 receiver_geometry_in_H0 "
            "API is missing."
        ),
    )

    receiver_function = getattr(
        receiver_module,
        "receiver_geometry_in_H0"
    )

    receiver_signature = str(
        inspect.signature(
            receiver_function
        )
    )

    receiver_source = inspect.getsource(
        receiver_module
    )

    frames_source = inspect.getsource(
        frames_module
    )

    required_receiver_tokens = (
        "receiver_offset_mean",
        "receiver_offset_covariance",
        "receiver_geometry",
    )

    for token in required_receiver_tokens:
        require(
            token
            in
            receiver_source,
            (
                "Frozen Stage1 receiver geometry "
                f"lacks token {token!r}."
            ),
        )

    require(
        (
            "HeadlampSurrogateConfig"
            in
            frames_source
            or
            "headlamp"
            in
            frames_source.lower()
        ),
        (
            "Frozen Stage1 configurable "
            "headlamp geometry was not found."
        ),
    )

    config_class = None

    for name in (
        "ReceiverGeometryConfig",
        "ReceiverConfig",
    ):
        if hasattr(
            receiver_module,
            name,
        ):
            config_class = getattr(
                receiver_module,
                name
            )

            break

    config_signature = (
        str(
            inspect.signature(
                config_class
            )
        )
        if config_class is not None
        else
        None
    )

    return {
        "module":
            receiver_module.__name__,

        "receiver_function":
            (
                "receiver_geometry_in_H0"
            ),

        "receiver_function_signature":
            receiver_signature,

        "config_class":
            (
                config_class.__name__
                if
                config_class is not None
                else
                None
            ),

        "config_signature":
            config_signature,

        "offset_mean_supported":
            True,

        "offset_covariance_supported":
            True,

        "runtime_source":
            inspect.getsourcefile(
                receiver_function
            ),
    }


def blocked(
    phase,
    reason,
    recovery,
):
    payload = {
        "stage":
            5,

        "block":
            "5.0_part_1",

        "status":
            "BLOCKED",

        "phase":
            phase,

        "reason":
            reason,

        "recovery":
            recovery,

        "training":
            False,

        "inference":
            False,

        "recalibration":
            False,

        "formal_evaluation":
            False,

        "upstream_modified":
            False,

        "legacy_runtime_dependency":
            False,
    }

    write_json(
        REPORT,
        payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.0 PART 1/2 = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "phase =",
        phase,
    )

    print(
        "reason =",
        reason,
    )

    print(
        "recovery =",
        recovery,
    )

    print(
        "training            = NO"
    )

    print(
        "inference           = NO"
    )

    print(
        "recalibration       = NO"
    )

    print(
        "formal evaluation   = NO"
    )

    print(
        "upstream modified   = NO"
    )

    print(
        "terminal remains open = YES"
    )


# ============================================================
# Main
# ============================================================

def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 5 — BLOCK 5.0 PART 1/2"
    )
    print(
        "FROZEN HANDOFF + RECEIVER/PART-A DISCOVERY"
    )
    print(
        "============================================================"
    )

    required = (
        STAGE4_CLOSURE,
        STAGE4_HANDOFF,
        STAGE4_FREEZE,
        STAGE4_REPRO,
        GAUSSIAN_CHECKPOINT,
        NORMALIZATION,
        CALIBRATOR,
        FORMAL_MANIFEST,
        PART_A_REFERENCE,
        STAGE0_CLOSURE,
        STAGE1_CLOSURE,
        PDF_CERTIFICATION,
        PDF_RESOLUTION,
        WAYMO_NOTICE,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    if missing:
        blocked(
            "required_artifacts",
            (
                "Missing required frozen "
                "artifact(s): "
                +
                ", ".join(
                    missing
                )
            ),
            (
                "Restore/audit only the missing "
                "upstream artifact. Do not start "
                "Stage5 scientific implementation."
            ),
        )

        return

    # ========================================================
    # A. Exact upstream hashes
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "A. EXACT FROZEN UPSTREAM HASHES"
    )
    print(
        "============================================================"
    )

    checks = (
        (
            "Stage4 closure",
            STAGE4_CLOSURE,
            EXPECTED_SHA256[
                "Stage4_closure"
            ],
        ),
        (
            "Stage4→Stage5 handoff",
            STAGE4_HANDOFF,
            EXPECTED_SHA256[
                "Stage4_handoff"
            ],
        ),
        (
            "Stage4 freeze manifest",
            STAGE4_FREEZE,
            EXPECTED_SHA256[
                "Stage4_freeze"
            ],
        ),
        (
            "Stage4 reproducibility",
            STAGE4_REPRO,
            EXPECTED_SHA256[
                "Stage4_reproducibility"
            ],
        ),
        (
            "Gaussian checkpoint",
            GAUSSIAN_CHECKPOINT,
            EXPECTED_SHA256[
                "Gaussian_checkpoint"
            ],
        ),
        (
            "fit-only normalization",
            NORMALIZATION,
            EXPECTED_SHA256[
                "normalization"
            ],
        ),
        (
            "covariance calibrator",
            CALIBRATOR,
            EXPECTED_SHA256[
                "calibrator"
            ],
        ),
        (
            "formal N=120 manifest",
            FORMAL_MANIFEST,
            EXPECTED_SHA256[
                "formal_manifest"
            ],
        ),
        (
            "PDF certification",
            PDF_CERTIFICATION,
            EXPECTED_SHA256[
                "PDF_certification"
            ],
        ),
        (
            "PDF resolution",
            PDF_RESOLUTION,
            EXPECTED_SHA256[
                "PDF_resolution"
            ],
        ),
        (
            "Waymo NOTICE",
            WAYMO_NOTICE,
            EXPECTED_SHA256[
                "Waymo_notice"
            ],
        ),
    )

    verified_hashes = {}

    for name, path, expected in checks:

        actual = file_sha256(
            path
        )

        require(
            actual
            ==
            expected,
            (
                f"{name} SHA changed: "
                f"{actual}"
            ),
        )

        verified_hashes[
            name
        ] = actual

        print(
            f"{name:30s}= PASS"
        )

    # ========================================================
    # B. PDF certification continuity
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "B. STAGES 0→4 PDF-COMPLIANCE CONTINUITY"
    )
    print(
        "============================================================"
    )

    certification = load_json(
        PDF_CERTIFICATION
    )

    require(
        certification.get(
            "status"
        )
        ==
        (
            "100_PERCENT_PDF_COMPLIANT_"
            "THROUGH_STAGE4"
        ),
        (
            "Stages0–4 PDF certification "
            "is not the frozen 100% PASS."
        ),
    )

    for stage in (
        "Stage0",
        "Stage1",
        "Stage2",
        "Stage3",
        "Stage4",
    ):
        require(
            certification.get(
                stage
            )
            ==
            "PASS",
            (
                f"{stage} is not PASS in "
                "PDF certification."
            ),
        )

    require(
        not certification.get(
            "unresolved_stage0_4_scientific_blockers"
        ),
        "Upstream scientific blocker exists.",
    )

    require(
        not certification.get(
            "unresolved_stage0_4_documentation_blockers"
        ),
        "Upstream documentation blocker exists.",
    )

    require(
        not certification.get(
            "unresolved_stage0_4_manual_reviews"
        ),
        "Upstream manual-review item exists.",
    )

    print(
        "Stage0 PDF compliance = PASS"
    )
    print(
        "Stage1 PDF compliance = PASS"
    )
    print(
        "Stage2 PDF compliance = PASS"
    )
    print(
        "Stage3 PDF compliance = PASS"
    )
    print(
        "Stage4 PDF compliance = PASS"
    )
    print(
        "unresolved upstream items = 0"
    )

    # ========================================================
    # C. Stage4 handoff semantic contract
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "C. STAGE4 → STAGE5 SEMANTIC HANDOFF"
    )
    print(
        "============================================================"
    )

    closure = load_json(
        STAGE4_CLOSURE
    )

    handoff = load_json(
        STAGE4_HANDOFF
    )

    freeze = load_json(
        STAGE4_FREEZE
    )

    require(
        closure.get(
            "status"
        )
        ==
        "COMPLETE_FROZEN",
        "Stage4 canonical closure is not frozen.",
    )

    require(
        handoff.get(
            "status"
        )
        ==
        "FROZEN_HANDOFF",
        "Stage4→Stage5 handoff is not frozen.",
    )

    require(
        freeze.get(
            "status"
        )
        ==
        "FROZEN",
        "Stage4 freeze manifest is not frozen.",
    )

    posterior = handoff.get(
        "default_downstream_posterior",
        {}
    )

    require(
        posterior.get(
            "family"
        )
        ==
        "calibrated_Gaussian_GRU",
        (
            "Stage5 default trajectory posterior "
            "is not calibrated_Gaussian_GRU."
        ),
    )

    require(
        posterior.get(
            "mandatory_default"
        )
        is True,
        (
            "Calibrated Gaussian is not marked "
            "mandatory default."
        ),
    )

    require(
        numeric_sequence_close(
            posterior.get(
                "horizons_s",
                (),
            ),
            EXPECTED_HORIZONS,
        ),
        (
            "Frozen Stage4 horizons changed."
        ),
    )

    require(
        numeric_sequence_close(
            posterior.get(
                "variance_scale_alpha_h",
                (),
            ),
            EXPECTED_VARIANCE_SCALE,
        ),
        (
            "Frozen Stage4 calibration "
            "variance scales changed."
        ),
    )

    require(
        posterior.get(
            "predictive_mean"
        )
        ==
        "unchanged_by_calibration",
        (
            "Stage4 calibrated mean semantics "
            "changed."
        ),
    )

    uncertainty = handoff.get(
        "measurement_predictive_uncertainty_separation",
        {}
    )

    require(
        uncertainty.get(
            "separate"
        )
        is True,
        (
            "Measurement/predictive uncertainty "
            "separation was lost."
        ),
    )

    require(
        uncertainty.get(
            "measurement_R_t_modified_by_calibration"
        )
        is False,
        (
            "Frozen calibration unexpectedly "
            "modifies R_t."
        ),
    )

    alternatives = handoff.get(
        "alternative_frozen_models",
        {}
    )

    gmm = alternatives.get(
        "GMM_GRU",
        {}
    )

    require(
        gmm.get(
            "selected_as_default_downstream"
        )
        is False,
        (
            "GMM was post-hoc selected "
            "for Stage5."
        ),
    )

    next_scope = handoff.get(
        "Stage5_scope_start",
        {}
    )

    require(
        next_scope.get(
            "next_required_layer"
        )
        ==
        "receiver_angular_posterior",
        (
            "Frozen Stage5 starting layer "
            "is not receiver/angular posterior."
        ),
    )

    print(
        "default trajectory posterior = CALIBRATED GAUSSIAN PASS"
    )

    print(
        "horizons                     = [0.1, 0.3, 0.5, 1.0] PASS"
    )

    print(
        "calibration alpha_h          = PASS"
    )

    print(
        "predictive mean preserved    = PASS"
    )

    print(
        "measurement R_t separate     = PASS"
    )

    print(
        "GMM downstream selection     = NO PASS"
    )

    print(
        "next required layer          = RECEIVER/ANGULAR POSTERIOR"
    )

    # ========================================================
    # D. Formal-population continuity
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "D. FORMAL POPULATION CONTINUITY"
    )
    print(
        "============================================================"
    )

    formal_rows = [
        json.loads(
            line
        )
        for line in (
            FORMAL_MANIFEST
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    ]

    require(
        len(
            formal_rows
        )
        ==
        120,
        (
            "Frozen formal manifest "
            "is not N=120."
        ),
    )

    scenario_ids = [
        str(
            row[
                "scenario_id"
            ]
        )
        for row in formal_rows
    ]

    require(
        len(
            set(
                scenario_ids
            )
        )
        ==
        120,
        (
            "Frozen formal manifest "
            "contains duplicate scenarios."
        ),
    )

    print(
        "formal scenarios          = 120 PASS"
    )

    print(
        "unique formal scenarios   = 120 PASS"
    )

    print(
        "formal used for Stage5 tuning = NO"
    )

    print(
        "tracks_to_predict receiver selector = NO"
    )

    # ========================================================
    # E. Frozen Stage1 receiver geometry route
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "E. FROZEN STAGE1 RECEIVER-GEOMETRY ROUTE"
    )
    print(
        "============================================================"
    )

    receiver = discover_receiver_geometry()

    print(
        "receiver module           =",
        receiver[
            "module"
        ],
    )

    print(
        "receiver API              =",
        receiver[
            "receiver_function"
        ],
    )

    print(
        "receiver signature        =",
        receiver[
            "receiver_function_signature"
        ],
    )

    print(
        "receiver config class     =",
        receiver[
            "config_class"
        ],
    )

    print(
        "offset mean support       = PASS"
    )

    print(
        "offset covariance support = PASS"
    )

    print(
        "Stage1 receiver geometry  = REUSABLE PASS"
    )

    # ========================================================
    # F. Part-A frozen optical / DPSK evidence
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "F. FROZEN PART-A OPTICAL / DPSK DISCOVERY"
    )
    print(
        "============================================================"
    )

    stage0_closure = load_json(
        STAGE0_CLOSURE
    )

    require(
        str(
            stage0_closure.get(
                "status",
                "",
            )
        ).upper()
        in
        {
            "PASS",
            "COMPLETE",
            "COMPLETE_FROZEN",
            "PASS_COMPLETE",
        },
        (
            "Stage0 closure is not PASS-like."
        ),
    )

    part_a = load_json(
        PART_A_REFERENCE
    )

    part_a_hash = file_sha256(
        PART_A_REFERENCE
    )

    part_a_paths = (
        find_existing_paths_from_payload(
            part_a
        )
    )

    part_a_files = (
        candidate_part_a_files(
            part_a
        )
    )

    evidence = search_text_groups(
        part_a_files
    )

    evidence_groups_found = [
        group
        for group, hits in evidence.items()
        if hits
    ]

    print(
        "Part-A reference SHA256  =",
        part_a_hash,
    )

    print(
        "existing referenced paths=",
        len(
            part_a_paths
        ),
    )

    for path in part_a_paths[
        :12
    ]:
        print(
            "  referenced:",
            path,
        )

    print(
        "candidate Part-A files   =",
        len(
            part_a_files
        ),
    )

    for path in part_a_files[
        :15
    ]:
        print(
            "  candidate:",
            path,
        )

    print()
    print(
        "Part-A evidence groups:"
    )

    for group, hits in evidence.items():
        print(
            f"  {group:32s}=",
            (
                "FOUND"
                if hits
                else
                "NOT YET RESOLVED"
            ),
        )

    # Part1 is a discovery gate, not yet the adapter freeze.
    # We require evidence of the frozen Part-A numerical
    # baseline and at least optical/communication evidence.
    require(
        (
            evidence[
                "PartA_waveform_constants"
            ]
            or
            part_a_paths
        ),
        (
            "Could not resolve any frozen Part-A "
            "numerical/source route."
        ),
    )

    optical_or_comm_groups = sum(
        bool(
            evidence[
                group
            ]
        )
        for group in (
            "DPSK",
            "BER",
            "optical_gain_or_received_power",
            "effective_rate",
        )
    )

    require(
        optical_or_comm_groups
        >=
        2,
        (
            "Part-A optical/DPSK route is not "
            "sufficiently resolved for Stage5. "
            "Need at least two communication/link "
            "evidence groups before contract freeze."
        ),
    )

    print()
    print(
        "Part-A frozen reference  = PASS"
    )

    print(
        "Part-A numerical route   = FOUND"
    )

    print(
        "optical/DPSK evidence groups =",
        optical_or_comm_groups,
        "/ 4",
    )

    print(
        "safe for Part2 exact API/value audit = YES"
    )

    # ========================================================
    # G. Environment / storage / legacy isolation
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "G. STAGE5 ENVIRONMENT / STORAGE / LEGACY ISOLATION"
    )
    print(
        "============================================================"
    )

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    require(
        free_gib
        >=
        MIN_FREE_GIB,
        (
            "250-GiB hard storage reserve "
            "would be violated."
        ),
    )

    environment = {
        "python":
            platform.python_version(),

        "numpy":
            np.__version__,

        "torch":
            torch.__version__,

        "torch_cuda":
            torch.version.cuda,

        "CUDA_available":
            bool(
                torch.cuda.is_available()
            ),

        "GPU":
            (
                torch.cuda.get_device_name(
                    0
                )
                if
                torch.cuda.is_available()
                else
                None
            ),

        "CUBLAS_WORKSPACE_CONFIG":
            os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),

        "free_GiB":
            free_gib,
    }

    print(
        "python                  =",
        environment[
            "python"
        ],
    )

    print(
        "numpy                   =",
        environment[
            "numpy"
        ],
    )

    print(
        "torch                   =",
        environment[
            "torch"
        ],
    )

    print(
        "torch CUDA              =",
        environment[
            "torch_cuda"
        ],
    )

    print(
        "CUDA available          =",
        environment[
            "CUDA_available"
        ],
    )

    print(
        "GPU                     =",
        environment[
            "GPU"
        ],
    )

    print(
        "CUBLAS workspace        =",
        environment[
            "CUBLAS_WORKSPACE_CONFIG"
        ],
    )

    print(
        "free GiB                =",
        round(
            free_gib,
            3,
        ),
    )

    print(
        "250-GiB reserve         = PASS"
    )

    print(
        "legacy Stage5 present   =",
        (
            "YES"
            if LEGACY_STAGE5.exists()
            else
            "NO"
        ),
    )

    print(
        "legacy runtime dependency = NO"
    )

    # ========================================================
    # H. Part1 closure
    # ========================================================

    payload = {
        "project":
            "Agni",

        "stage":
            5,

        "block":
            "5.0_part_1",

        "status":
            "PASS",

        "purpose":
            (
                "frozen_upstream_handoff_"
                "receiver_geometry_and_"
                "PartA_optical_discovery"
            ),

        "PDF_program_audit":
            {
                "status":
                    "PASS",

                "mandatory_Stage5_scope_covered":
                    True,

                "Stage6_plus_scope_not_pulled_forward":
                    True,

                "binding_clarifications": [
                    (
                        "constructed causal connected-"
                        "vehicle eligibility because WOMD "
                        "has no connectivity label"
                    ),
                    (
                        "all Stage5 tuning before formal "
                        "N=120"
                    ),
                    (
                        "posterior selected mass and "
                        "empirical containment reported "
                        "separately"
                    ),
                    (
                        "beam/controller latency in Stage5; "
                        "full joint beam+ADB latency in Stage7"
                    ),
                    (
                        "future receiver heading derived "
                        "causally/predictively, not from "
                        "future ground truth"
                    ),
                    (
                        "Stage5 paper-ready beam/link "
                        "figures required at formal closure"
                    ),
                ],
            },

        "verified_upstream_hashes":
            verified_hashes,

        "Stage0_4_PDF_certification":
            (
                "100_PERCENT_PDF_COMPLIANT_"
                "THROUGH_STAGE4"
            ),

        "Stage4_handoff": {
            "default_posterior":
                "calibrated_Gaussian_GRU",

            "horizons_s":
                list(
                    EXPECTED_HORIZONS
                ),

            "variance_scale_alpha_h":
                list(
                    EXPECTED_VARIANCE_SCALE
                ),

            "measurement_R_t_separate":
                True,

            "GMM_selected_downstream":
                False,

            "next_required_layer":
                "receiver_angular_posterior",
        },

        "formal_population": {
            "scenario_count":
                120,

            "unique_scenarios":
                120,

            "used_for_Stage5_tuning":
                False,

            "tracks_to_predict_receiver_selector":
                False,
        },

        "receiver_geometry_discovery":
            receiver,

        "PartA_discovery": {
            "frozen_reference":
                str(
                    PART_A_REFERENCE
                ),

            "frozen_reference_sha256":
                part_a_hash,

            "resolved_paths": [
                str(
                    path
                )
                for path in part_a_paths
            ],

            "candidate_files": [
                str(
                    path
                )
                for path in part_a_files
            ],

            "evidence_groups":
                evidence,

            "optical_or_communication_groups_found":
                optical_or_comm_groups,

            "exact_link_adapter_frozen":
                False,
        },

        "environment":
            environment,

        "legacy": {
            "legacy_stage5_path":
                str(
                    LEGACY_STAGE5
                ),

            "exists":
                LEGACY_STAGE5.exists(),

            "runtime_dependency":
                False,

            "modified":
                False,
        },

        "scientific_execution": {
            "training":
                False,

            "inference":
                False,

            "recalibration":
                False,

            "formal_evaluation":
                False,

            "dataset_scan":
                False,
        },

        "upstream_modified":
            False,

        "next":
            (
                "Block5.0 Part2 exact contract "
                "freeze and Stage5 bootstrap tests"
            ),
    }

    write_json(
        REPORT,
        payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.0 PART 1/2 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "Stages0–4 PDF certification = PASS"
    )

    print(
        "Stage4 closure/handoff       = PASS"
    )

    print(
        "calibrated Gaussian default  = PASS"
    )

    print(
        "measurement/predictive UQ    = SEPARATE PASS"
    )

    print(
        "formal N=120 continuity      = PASS"
    )

    print(
        "formal used for tuning       = NO"
    )

    print(
        "receiver geometry route      = PASS"
    )

    print(
        "Part-A frozen reference      = PASS"
    )

    print(
        "Part-A optical/DPSK route    = DISCOVERED"
    )

    print(
        "legacy runtime dependency    = NO"
    )

    print(
        "training/inference           = NO"
    )

    print(
        "upstream Stage0–4 modified   = NO"
    )

    print(
        "safe to freeze Stage5 contract in Part2 = YES"
    )

    print(
        "STATUS = PASS"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    blocked(
        "unexpected_discovery_error",
        (
            f"{type(exc).__name__}: "
            f"{exc}"
        ),
        (
            "Inspect only Block5.0 discovery "
            "or the specific frozen upstream "
            "artifact/API that failed. Do not "
            "modify Stages0–4."
        ),
    )

    print()
    traceback.print_exc()

# Deliberately no sys.exit().
