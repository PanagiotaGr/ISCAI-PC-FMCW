from __future__ import annotations

import ast
from dataclasses import (
    asdict,
    is_dataclass,
)
from hashlib import sha256
import importlib
import inspect
import json
import math
from pathlib import Path
import re
import traceback

from iscai_stage5.contracts import (
    CALIBRATION_VARIANCE_SCALE_ALPHA_H,
    CODEBOOK_SIZES,
    DEFAULT_PROBABILITY_MASS_TARGET,
    DEVELOPMENT_FIELDS_TO_FREEZE_BEFORE_FORMAL,
    EFFECTIVE_RATE_FORMULA_STATUS,
    EFFECTIVE_RATE_PROVENANCE,
    FORMAL_POPULATION_N,
    HORIZONS_S,
    OPTICAL_LINK_CHAIN,
    PART_A_SOURCE_EFFECTIVE_RATE_REQUIRED,
    PRIMARY_RECEIVER_POLICY,
    PROBABILITY_MASS_TARGETS,
    RECEIVER_GEOMETRY_MODES,
    STAGE4_DEFAULT_POSTERIOR,
    STAGE5_ACCEPTANCE_REQUIREMENTS,
    STAGE5_OUT_OF_SCOPE,
    contract_dict,
)


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

PART1 = (
    STAGE5
    / "reports/"
      "block50_part1_discovery.json"
)

PART_A_REFERENCE = (
    STAGE0
    / "reports/stage0/"
      "part_a_frozen_reference.json"
)

PART_A_NOTEBOOK = (
    ROOT
    / "part_a_reference/"
      "ISCAI_pc_fmcw/"
      "notebooks/"
      "ISCAI_PC_FMCW.ipynb"
)

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

FORMAL_MANIFEST = (
    STAGE3
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
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

PDF_CERTIFICATION = (
    ROOT
    / "audits/pdf_compliance_stages0_4/"
      "pdf_compliance_final_certification.json"
)

WAYMO_NOTICE = (
    ROOT
    / "NOTICE.md"
)

CONTRACT_CONFIG = (
    STAGE5
    / "configs/"
      "stage5_contract.json"
)

PART_A_CONTRACT = (
    STAGE5
    / "artifacts/block50/"
      "part_a_source_contract.json"
)

REPORT = (
    STAGE5
    / "reports/"
      "block50_contract_freeze.json"
)

IMPLEMENTATION_LOG = (
    STAGE5
    / "docs/"
      "implementation_log.md"
)


EXPECTED_SHA = {
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

    "Stage4_repro":
        (
            "135a3c60e5a2c4419125fa0daa6d2d554"
            "52f923255a04698f0bc1aaca5f474b8"
        ),

    "Gaussian":
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

    "formal":
        (
            "2208e7287ddf6439fda4597c435a9cba"
            "1d1b9d0e4c4547bc5dd92e56e8124e46"
        ),

    "PDF":
        (
            "7c94bb9deb37a5eba29c9237bfcf4b0d"
            "24e4031d9b1b22a500c5a63730bf0d06"
        ),

    "NOTICE":
        (
            "b05a8da05a56b71dbe28cc2e217b26c6"
            "92b8761ad1d026a6365729a43e26f997"
        ),

    "PartA_reference":
        (
            "fcdfc0a7b9c14fa9447ceb8563a6f20a"
            "c277706223388dc9aa8a1edf416c5e2b"
        ),
}


PART_A_EXPECTED_CONSTANTS = {
    "carrier_frequency_Hz": {
        "value":
            193.4e12,

        "name_hints":
            (
                "fc",
                "f_c",
                "carrier",
                "frequency",
                "f0",
            ),
    },

    "waveform_bandwidth_Hz": {
        "value":
            10.0e9,

        "name_hints":
            (
                "b",
                "bw",
                "bandwidth",
            ),
    },

    "chirp_duration_s": {
        "value":
            10.0e-6,

        "name_hints":
            (
                "tchirp",
                "t_chirp",
                "chirp",
                "tc",
                "t",
            ),
    },

    "data_rate_bps": {
        "value":
            1.0e9,

        "name_hints":
            (
                "data_rate",
                "datarate",
                "bitrate",
                "bit_rate",
                "rb",
                "r_b",
            ),
    },

    "sample_period_s": {
        "value":
            1.0e-9,

        "name_hints":
            (
                "ts",
                "t_s",
                "sample_period",
                "sampling_period",
            ),
    },

    "N_fast": {
        "value":
            131072.0,

        "name_hints":
            (
                "nfast",
                "n_fast",
                "nfft",
                "n_fft",
                "n",
            ),
    },

    "M_chirps": {
        "value":
            64.0,

        "name_hints":
            (
                "mchirps",
                "m_chirps",
                "num_chirps",
                "nchirps",
                "m",
            ),
    },
}


PART_A_EVIDENCE_GROUPS = {
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
}


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


def bytes_sha256(
    value: bytes,
):
    return sha256(
        value
    ).hexdigest()


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
    temporary = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    temporary.write_text(
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

    temporary.replace(
        path
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


def close_number(
    first,
    second,
):
    first = float(
        first
    )

    second = float(
        second
    )

    scale = max(
        1.0,
        abs(
            second
        ),
    )

    return (
        abs(
            first
            -
            second
        )
        <=
        1e-12
        *
        scale
    )


def notebook_cell_source(
    cell,
):
    source = cell.get(
        "source",
        "",
    )

    if isinstance(
        source,
        list,
    ):
        return "".join(
            source
        )

    return str(
        source
    )


def safe_numeric_eval(
    node,
    env,
):
    if isinstance(
        node,
        ast.Constant,
    ):
        if isinstance(
            node.value,
            (
                int,
                float,
            ),
        ):
            return float(
                node.value
            )

        raise ValueError

    if isinstance(
        node,
        ast.Name,
    ):
        if node.id in env:
            return float(
                env[
                    node.id
                ]
            )

        raise ValueError

    if isinstance(
        node,
        ast.UnaryOp,
    ):
        value = safe_numeric_eval(
            node.operand,
            env,
        )

        if isinstance(
            node.op,
            ast.UAdd,
        ):
            return value

        if isinstance(
            node.op,
            ast.USub,
        ):
            return -value

        raise ValueError

    if isinstance(
        node,
        ast.BinOp,
    ):
        left = safe_numeric_eval(
            node.left,
            env,
        )

        right = safe_numeric_eval(
            node.right,
            env,
        )

        if isinstance(
            node.op,
            ast.Add,
        ):
            return left + right

        if isinstance(
            node.op,
            ast.Sub,
        ):
            return left - right

        if isinstance(
            node.op,
            ast.Mult,
        ):
            return left * right

        if isinstance(
            node.op,
            ast.Div,
        ):
            return left / right

        if isinstance(
            node.op,
            ast.Pow,
        ):
            return left ** right

        raise ValueError

    raise ValueError


def extract_numeric_assignments(
    notebook,
):
    assignments = []

    env = {}

    parse_failures = []

    for index, cell in enumerate(
        notebook.get(
            "cells",
            []
        )
    ):
        if cell.get(
            "cell_type"
        ) != "code":
            continue

        source = notebook_cell_source(
            cell
        )

        try:
            tree = ast.parse(
                source
            )

        except Exception as exc:
            parse_failures.append({
                "cell_index":
                    index,

                "exception":
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
            })

            continue

        for node in tree.body:

            if isinstance(
                node,
                ast.Assign,
            ):
                targets = [
                    target.id
                    for target in node.targets
                    if isinstance(
                        target,
                        ast.Name,
                    )
                ]

                if not targets:
                    continue

                try:
                    value = safe_numeric_eval(
                        node.value,
                        env,
                    )

                except Exception:
                    continue

                for name in targets:
                    env[
                        name
                    ] = value

                    assignments.append({
                        "name":
                            name,

                        "value":
                            value,

                        "cell_index":
                            index,

                        "line":
                            int(
                                getattr(
                                    node,
                                    "lineno",
                                    0,
                                )
                            ),

                        "source":
                            (
                                ast.get_source_segment(
                                    source,
                                    node,
                                )
                                or
                                ""
                            )[
                                :400
                            ],
                    })

            elif isinstance(
                node,
                ast.AnnAssign,
            ):
                if not isinstance(
                    node.target,
                    ast.Name,
                ):
                    continue

                if node.value is None:
                    continue

                try:
                    value = safe_numeric_eval(
                        node.value,
                        env,
                    )

                except Exception:
                    continue

                name = node.target.id

                env[
                    name
                ] = value

                assignments.append({
                    "name":
                        name,

                    "value":
                        value,

                    "cell_index":
                        index,

                    "line":
                        int(
                            getattr(
                                node,
                                "lineno",
                                0,
                            )
                        ),

                    "source":
                        (
                            ast.get_source_segment(
                                source,
                                node,
                            )
                            or
                            ""
                        )[
                            :400
                        ],
                })

    return (
        assignments,
        parse_failures,
    )


def normalized_name(
    name,
):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(
            name
        ).lower(),
    )


def assignment_score(
    name,
    hints,
):
    name_normalized = normalized_name(
        name
    )

    best = 0

    for hint in hints:
        hint_normalized = normalized_name(
            hint
        )

        if not hint_normalized:
            continue

        if (
            name_normalized
            ==
            hint_normalized
        ):
            best = max(
                best,
                4,
            )

        elif (
            name_normalized.startswith(
                hint_normalized
            )
            or
            name_normalized.endswith(
                hint_normalized
            )
        ):
            best = max(
                best,
                3,
            )

        elif (
            hint_normalized
            in
            name_normalized
        ):
            best = max(
                best,
                2,
            )

    return best


def resolve_expected_constant(
    assignments,
    *,
    expected,
    hints,
):
    candidates = [
        item
        for item in assignments
        if close_number(
            item[
                "value"
            ],
            expected,
        )
    ]

    require(
        candidates,
        (
            "Could not find Part-A assignment "
            f"with expected value {expected!r}."
        ),
    )

    ranked = sorted(
        (
            (
                assignment_score(
                    item[
                        "name"
                    ],
                    hints,
                ),
                item,
            )
            for item in candidates
        ),
        key=lambda pair: (
            pair[
                0
            ],
            -int(
                pair[
                    1
                ][
                    "cell_index"
                ]
            ),
        ),
        reverse=True,
    )

    best_score = ranked[
        0
    ][
        0
    ]

    best_items = [
        item
        for score, item in ranked
        if score
        ==
        best_score
    ]

    if best_score <= 0:
        require(
            len(
                candidates
            )
            ==
            1,
            (
                "Part-A expected numeric value "
                f"{expected!r} is ambiguous and "
                "no variable-name hint resolves it. "
                f"Candidates="
                f"{[(c['name'], c['value']) for c in candidates]}"
            ),
        )

        return candidates[
            0
        ]

    return best_items[
        0
    ]


def extract_evidence_cells(
    notebook,
):
    result = {
        group:
            []
        for group in PART_A_EVIDENCE_GROUPS
    }

    for index, cell in enumerate(
        notebook.get(
            "cells",
            []
        )
    ):
        source = notebook_cell_source(
            cell
        )

        lower = source.lower()

        for group, tokens in (
            PART_A_EVIDENCE_GROUPS.items()
        ):
            matched = [
                token
                for token in tokens
                if token.lower()
                in lower
            ]

            if not matched:
                continue

            result[
                group
            ].append({
                "cell_index":
                    index,

                "cell_type":
                    cell.get(
                        "cell_type"
                    ),

                "source_sha256":
                    bytes_sha256(
                        source.encode(
                            "utf-8"
                        )
                    ),

                "matched_tokens":
                    matched,

                "source_preview":
                    " ".join(
                        source.split()
                    )[
                        :500
                    ],
            })

    return result


def receiver_geometry_evidence():
    module = importlib.import_module(
        "iscai_stage1.geometry.receiver"
    )

    config_class = getattr(
        module,
        "ReceiverGeometryConfig",
        None,
    )

    function = getattr(
        module,
        "receiver_geometry_in_H0",
        None,
    )

    require(
        config_class is not None,
        (
            "ReceiverGeometryConfig "
            "is no longer available."
        ),
    )

    require(
        function is not None,
        (
            "receiver_geometry_in_H0 "
            "is no longer available."
        ),
    )

    config = config_class()

    if is_dataclass(
        config
    ):
        defaults = asdict(
            config
        )

    else:
        defaults = {
            "repr":
                repr(
                    config
                )
        }

    source = inspect.getsource(
        module
    )

    require(
        "receiver_offset_mean"
        in source,
        (
            "Stage1 receiver offset mean "
            "route disappeared."
        ),
    )

    require(
        "receiver_offset_covariance"
        in source,
        (
            "Stage1 receiver covariance "
            "route disappeared."
        ),
    )

    return {
        "module":
            module.__name__,

        "source_file":
            inspect.getsourcefile(
                function
            ),

        "function_signature":
            str(
                inspect.signature(
                    function
                )
            ),

        "config_class":
            config_class.__name__,

        "config_signature":
            str(
                inspect.signature(
                    config_class
                )
            ),

        "default_config":
            defaults,
    }


def implementation_fingerprint():
    roots = (
        STAGE5
        / "src",

        STAGE5
        / "tests",

        STAGE5
        / "configs",

        STAGE5
        / "scripts",
    )

    files = []

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob(
            "*"
        ):
            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if path.suffix in (
                ".pyc",
                ".pyo",
            ):
                continue

            files.append(
                path
            )

    files.sort(
        key=lambda path:
            str(
                path.relative_to(
                    STAGE5
                )
            )
    )

    digest = sha256()

    for path in files:
        relative = str(
            path.relative_to(
                STAGE5
            )
        )

        digest.update(
            relative.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            path.read_bytes()
        )

        digest.update(
            b"\0"
        )

    return (
        len(
            files
        ),
        digest.hexdigest(),
    )


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 5 — BLOCK 5.0 PART 2/2"
    )
    print(
        "EXACT CONTRACT FREEZE"
    )
    print(
        "============================================================"
    )

    required = (
        PART1,
        PART_A_REFERENCE,
        PART_A_NOTEBOOK,
        STAGE4_CLOSURE,
        STAGE4_HANDOFF,
        STAGE4_FREEZE,
        STAGE4_REPRO,
        FORMAL_MANIFEST,
        GAUSSIAN_CHECKPOINT,
        NORMALIZATION,
        CALIBRATOR,
        PDF_CERTIFICATION,
        WAYMO_NOTICE,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing required frozen file(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Part1 must be PASS
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "A. PART1 CONTINUITY"
    )
    print(
        "============================================================"
    )

    part1 = load_json(
        PART1
    )

    require(
        part1.get(
            "status"
        )
        ==
        "PASS",
        "Block5.0 Part1 is not PASS.",
    )

    require(
        part1.get(
            "upstream_modified"
        )
        is False,
        (
            "Part1 claims upstream "
            "modification."
        ),
    )

    scientific = part1.get(
        "scientific_execution",
        {}
    )

    for key in (
        "training",
        "inference",
        "recalibration",
        "formal_evaluation",
        "dataset_scan",
    ):
        require(
            scientific.get(
                key
            )
            is False,
            (
                f"Part1 unexpected execution: "
                f"{key}"
            ),
        )

    print(
        "Block5.0 Part1             = PASS"
    )

    print(
        "training/inference          = NO"
    )

    print(
        "upstream modified           = NO"
    )

    # ========================================================
    # B. Recheck frozen upstream hashes
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "B. FROZEN UPSTREAM RECHECK"
    )
    print(
        "============================================================"
    )

    frozen_checks = (
        (
            "Stage4 closure",
            STAGE4_CLOSURE,
            EXPECTED_SHA[
                "Stage4_closure"
            ],
        ),
        (
            "Stage4 handoff",
            STAGE4_HANDOFF,
            EXPECTED_SHA[
                "Stage4_handoff"
            ],
        ),
        (
            "Stage4 freeze",
            STAGE4_FREEZE,
            EXPECTED_SHA[
                "Stage4_freeze"
            ],
        ),
        (
            "Stage4 reproducibility",
            STAGE4_REPRO,
            EXPECTED_SHA[
                "Stage4_repro"
            ],
        ),
        (
            "Gaussian checkpoint",
            GAUSSIAN_CHECKPOINT,
            EXPECTED_SHA[
                "Gaussian"
            ],
        ),
        (
            "normalization",
            NORMALIZATION,
            EXPECTED_SHA[
                "normalization"
            ],
        ),
        (
            "calibrator",
            CALIBRATOR,
            EXPECTED_SHA[
                "calibrator"
            ],
        ),
        (
            "formal manifest",
            FORMAL_MANIFEST,
            EXPECTED_SHA[
                "formal"
            ],
        ),
        (
            "PDF certification",
            PDF_CERTIFICATION,
            EXPECTED_SHA[
                "PDF"
            ],
        ),
        (
            "Waymo NOTICE",
            WAYMO_NOTICE,
            EXPECTED_SHA[
                "NOTICE"
            ],
        ),
        (
            "Part-A frozen reference",
            PART_A_REFERENCE,
            EXPECTED_SHA[
                "PartA_reference"
            ],
        ),
    )

    upstream_hashes = {}

    for name, path, expected in (
        frozen_checks
    ):
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

        upstream_hashes[
            name
        ] = actual

        print(
            f"{name:29s}= PASS"
        )

    # ========================================================
    # C. Exact Part-A notebook source freeze
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "C. EXACT PART-A NOTEBOOK SOURCE FREEZE"
    )
    print(
        "============================================================"
    )

    notebook_sha = file_sha256(
        PART_A_NOTEBOOK
    )

    notebook = load_json(
        PART_A_NOTEBOOK
    )

    require(
        isinstance(
            notebook.get(
                "cells"
            ),
            list,
        ),
        "Part-A notebook has no cell list.",
    )

    print(
        "Part-A notebook          =",
        PART_A_NOTEBOOK,
    )

    print(
        "notebook SHA256          =",
        notebook_sha,
    )

    print(
        "notebook cells           =",
        len(
            notebook[
                "cells"
            ]
        ),
    )

    assignments, parse_failures = (
        extract_numeric_assignments(
            notebook
        )
    )

    print(
        "numeric assignments      =",
        len(
            assignments
        ),
    )

    print(
        "non-Python parse cells   =",
        len(
            parse_failures
        ),
    )

    resolved_constants = {}

    for name, specification in (
        PART_A_EXPECTED_CONSTANTS.items()
    ):
        resolved = (
            resolve_expected_constant(
                assignments,
                expected=(
                    specification[
                        "value"
                    ]
                ),
                hints=(
                    specification[
                        "name_hints"
                    ]
                ),
            )
        )

        resolved_constants[
            name
        ] = {
            "expected_value":
                float(
                    specification[
                        "value"
                    ]
                ),

            "resolved_variable":
                resolved[
                    "name"
                ],

            "resolved_value":
                float(
                    resolved[
                        "value"
                ]),

            "cell_index":
                int(
                    resolved[
                        "cell_index"
                    ]
                ),

            "line":
                int(
                    resolved[
                        "line"
                    ]
                ),

            "source":
                resolved[
                    "source"
                ],
        }

        print(
            f"{name:26s}= PASS |",
            resolved[
                "name"
            ],
            "=",
            resolved[
                "value"
            ],
        )

    # ========================================================
    # D. Part-A communication evidence-cell freeze
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "D. PART-A OPTICAL / DPSK EVIDENCE-CELL FREEZE"
    )
    print(
        "============================================================"
    )

    evidence_cells = (
        extract_evidence_cells(
            notebook
        )
    )

    #
    # Frozen Part-A source-level requirements.
    #
    # effective_rate is deliberately NOT required
    # here.  It is a Stage5-derived system metric:
    # frozen Part-A link quantities + raw data rate
    # + Stage5 beam-probing overhead.
    #
    for group in (
        "DPSK",
        "BER",
        "optical_gain_or_received_power",
    ):
        hits = evidence_cells[
            group
        ]

        require(
            hits,
            (
                "Part-A notebook no longer "
                f"contains {group} evidence."
            ),
        )

        print(
            f"{group:32s}= PASS | cells =",
            [
                item[
                    "cell_index"
                ]
                for item in hits[
                    :10
                ]
            ],
        )

    effective_rate_hits = (
        evidence_cells[
            "effective_rate"
        ]
    )

    require(
        PART_A_SOURCE_EFFECTIVE_RATE_REQUIRED
        is False,
        (
            "Contract unexpectedly requires "
            "Part-A source-level effective rate."
        ),
    )

    require(
        EFFECTIVE_RATE_FORMULA_STATUS
        ==
        "MUST_FREEZE_IN_BLOCK5.6_BEFORE_FORMAL",
        (
            "Effective-rate freeze timing "
            "contract changed."
        ),
    )

    print(
        f"{'effective_rate':32s}= "
        "STAGE5 DERIVED / NOT REQUIRED IN PART-A SOURCE"
    )

    print(
        "effective-rate Part-A source hits =",
        len(
            effective_rate_hits
        ),
    )

    print(
        "effective-rate provenance =",
        EFFECTIVE_RATE_PROVENANCE,
    )

    print(
        "effective-rate formula    =",
        EFFECTIVE_RATE_FORMULA_STATUS,
    )

    part_a_contract = {
        "status":
            "FROZEN_SOURCE_EVIDENCE",

        "source_notebook":
            str(
                PART_A_NOTEBOOK
            ),

        "source_notebook_sha256":
            notebook_sha,

        "frozen_Stage0_reference":
            str(
                PART_A_REFERENCE
            ),

        "frozen_Stage0_reference_sha256":
            EXPECTED_SHA[
                "PartA_reference"
            ],

        "core_numeric_constants":
            resolved_constants,

        "communication_evidence_cells":
            evidence_cells,

        "PartA_source_effective_rate_required":
            False,

        "effective_rate_provenance":
            EFFECTIVE_RATE_PROVENANCE,

        "effective_rate_formula_status":
            EFFECTIVE_RATE_FORMULA_STATUS,

        "effective_rate_source_hits":
            len(
                effective_rate_hits
            ),

        "Stage5_link_adapter_policy":
            (
                "reuse_or_numerically_reproduce_"
                "frozen_PartA_link_assumptions"
            ),

        "new_arbitrary_optical_model_allowed":
            False,

        "exact_numerical_link_adapter":
            "TO_BE_IMPLEMENTED_AND_CROSSCHECKED_IN_BLOCK5.6",

        "PartA_source_may_be_modified":
            False,
    }

    write_json(
        PART_A_CONTRACT,
        part_a_contract,
    )

    part_a_contract_sha = (
        file_sha256(
            PART_A_CONTRACT
        )
    )

    print()
    print(
        "Part-A source contract SHA256 =",
        part_a_contract_sha,
    )

    # ========================================================
    # E. Receiver geometry exact source contract
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "E. STAGE1 RECEIVER-GEOMETRY CONTRACT"
    )
    print(
        "============================================================"
    )

    receiver = (
        receiver_geometry_evidence()
    )

    print(
        "module                    =",
        receiver[
            "module"
        ],
    )

    print(
        "source                    =",
        receiver[
            "source_file"
        ],
    )

    print(
        "function signature        =",
        receiver[
            "function_signature"
        ],
    )

    print(
        "config class              =",
        receiver[
            "config_class"
        ],
    )

    print(
        "receiver offset mean      = SUPPORTED PASS"
    )

    print(
        "receiver covariance       = SUPPORTED PASS"
    )

    print(
        "Stage5 conceptual modes   =",
        list(
            RECEIVER_GEOMETRY_MODES
        ),
    )

    # ========================================================
    # F. Freeze canonical Stage5 contract JSON
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "F. CANONICAL STAGE5 STRUCTURAL CONTRACT"
    )
    print(
        "============================================================"
    )

    contract = contract_dict()

    contract[
        "frozen_upstream"
    ] = {
        "hashes":
            upstream_hashes,

        "PartA_source_contract":
            str(
                PART_A_CONTRACT
            ),

        "PartA_source_contract_sha256":
            part_a_contract_sha,
    }

    contract[
        "receiver_geometry_upstream"
    ] = receiver

    write_json(
        CONTRACT_CONFIG,
        contract,
    )

    contract_sha = file_sha256(
        CONTRACT_CONFIG
    )

    print(
        "default posterior         =",
        STAGE4_DEFAULT_POSTERIOR,
    )

    print(
        "primary receiver policy   =",
        PRIMARY_RECEIVER_POLICY,
    )

    print(
        "receiver modes            =",
        list(
            RECEIVER_GEOMETRY_MODES
        ),
    )

    print(
        "horizons s                =",
        list(
            HORIZONS_S
        ),
    )

    print(
        "codebook sizes            =",
        list(
            CODEBOOK_SIZES
        ),
    )

    print(
        "probability masses        =",
        list(
            PROBABILITY_MASS_TARGETS
        ),
    )

    print(
        "nominal mass target       =",
        DEFAULT_PROBABILITY_MASS_TARGET,
    )

    print(
        "optical chain             =",
        list(
            OPTICAL_LINK_CHAIN
        ),
    )

    print(
        "formal N                  =",
        FORMAL_POPULATION_N,
    )

    print(
        "formal tuning             = NO"
    )

    print(
        "acceptance requirements   =",
        list(
            STAGE5_ACCEPTANCE_REQUIREMENTS
        ),
    )

    print(
        "development fields pending=",
        len(
            DEVELOPMENT_FIELDS_TO_FREEZE_BEFORE_FORMAL
        ),
    )

    print(
        "Stage6+ out-of-scope      =",
        list(
            STAGE5_OUT_OF_SCOPE
        ),
    )

    print(
        "contract SHA256           =",
        contract_sha,
    )

    # ========================================================
    # G. Contract readback / anti-leakage checks
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "G. CONTRACT READBACK / ANTI-LEAKAGE"
    )
    print(
        "============================================================"
    )

    readback = load_json(
        CONTRACT_CONFIG
    )

    require(
        readback.get(
            "status"
        )
        ==
        "STRUCTURAL_CONTRACT_FROZEN",
        "Stage5 contract readback failed.",
    )

    require(
        readback[
            "trajectory_posterior"
        ][
            "family"
        ]
        ==
        "calibrated_Gaussian_GRU",
        "Wrong Stage5 trajectory posterior.",
    )

    require(
        readback[
            "receiver_selection"
        ][
            "tracks_to_predict_selector"
        ]
        is False,
        (
            "tracks_to_predict became "
            "a receiver selector."
        ),
    )

    require(
        readback[
            "receiver_selection"
        ][
            "future_truth_selector"
        ]
        is False,
        (
            "Future truth became "
            "a receiver selector."
        ),
    )

    require(
        readback[
            "receiver_geometry"
        ][
            "future_GT_heading_controller"
        ]
        is False,
        (
            "Future GT heading entered "
            "controller contract."
        ),
    )

    require(
        readback[
            "formal"
        ][
            "tuning_allowed"
        ]
        is False,
        "Formal tuning became allowed.",
    )

    require(
        readback[
            "formal"
        ][
            "receiver_policy_selection_allowed"
        ]
        is False,
        (
            "Formal receiver-policy selection "
            "became allowed."
        ),
    )

    require(
        readback[
            "formal"
        ][
            "codebook_selection_allowed"
        ]
        is False,
        (
            "Formal codebook selection "
            "became allowed."
        ),
    )

    require(
        readback[
            "formal"
        ][
            "threshold_selection_allowed"
        ]
        is False,
        (
            "Formal threshold selection "
            "became allowed."
        ),
    )

    require(
        readback[
            "optical_link"
        ][
            "new_arbitrary_model_allowed"
        ]
        is False,
        (
            "Arbitrary new optical model "
            "became allowed."
        ),
    )

    require(
        readback[
            "development_freeze"
        ][
            "must_complete_before_formal"
        ]
        is True,
        (
            "Development freeze-before-formal "
            "contract disappeared."
        ),
    )

    print(
        "tracks_to_predict selector = NO PASS"
    )

    print(
        "future truth selector      = NO PASS"
    )

    print(
        "future GT heading          = NO PASS"
    )

    print(
        "formal tuning              = NO PASS"
    )

    print(
        "formal policy selection    = NO PASS"
    )

    print(
        "formal threshold selection = NO PASS"
    )

    print(
        "arbitrary optical model    = NO PASS"
    )

    print(
        "dev freeze before formal   = YES PASS"
    )

    # ========================================================
    # H. Implementation fingerprint + report
    # ========================================================

    (
        implementation_files,
        implementation_sha,
    ) = implementation_fingerprint()

    report = {
        "project":
            "Agni",

        "stage":
            5,

        "block":
            "5.0",

        "status":
            "PASS",

        "Part1":
            {
                "status":
                    "PASS",

                "report":
                    str(
                        PART1
                    ),

                "report_sha256":
                    file_sha256(
                        PART1
                    ),
            },

        "frozen_upstream_hashes":
            upstream_hashes,

        "PartA": {
            "notebook":
                str(
                    PART_A_NOTEBOOK
                ),

            "notebook_sha256":
                notebook_sha,

            "source_contract":
                str(
                    PART_A_CONTRACT
                ),

            "source_contract_sha256":
                part_a_contract_sha,

            "resolved_core_constants":
                resolved_constants,

            "DPSK":
                "PASS",

            "BER":
                "PASS",

            "optical_gain_or_received_power":
                "PASS",

            "effective_rate":
                "STAGE5_DERIVED_REQUIRED_METRIC",

            "effective_rate_provenance":
                EFFECTIVE_RATE_PROVENANCE,

            "effective_rate_formula_status":
                EFFECTIVE_RATE_FORMULA_STATUS,

            "PartA_source_effective_rate_required":
                False,

            "exact_link_adapter_implemented":
                False,

            "exact_link_adapter_block":
                "5.6",
        },

        "receiver_geometry":
            receiver,

        "Stage5_contract": {
            "path":
                str(
                    CONTRACT_CONFIG
                ),

            "sha256":
                contract_sha,

            "status":
                "STRUCTURAL_CONTRACT_FROZEN",
        },

        "pre_registered": {
            "default_trajectory_posterior":
                STAGE4_DEFAULT_POSTERIOR,

            "receiver_policy":
                PRIMARY_RECEIVER_POLICY,

            "receiver_geometry_modes":
                list(
                    RECEIVER_GEOMETRY_MODES
                ),

            "horizons_s":
                list(
                    HORIZONS_S
                ),

            "codebook_sizes":
                list(
                    CODEBOOK_SIZES
                ),

            "probability_mass_targets":
                list(
                    PROBABILITY_MASS_TARGETS
                ),

            "nominal_probability_mass":
                DEFAULT_PROBABILITY_MASS_TARGET,

            "optical_link_chain":
                list(
                    OPTICAL_LINK_CHAIN
                ),

            "acceptance_requirements":
                list(
                    STAGE5_ACCEPTANCE_REQUIREMENTS
                ),
        },

        "development_only_pending_freeze": {
            "fields":
                list(
                    DEVELOPMENT_FIELDS_TO_FREEZE_BEFORE_FORMAL
                ),

            "formal_data_allowed":
                False,

            "must_be_frozen_before_Block5.8":
                True,
        },

        "scope_boundary": {
            "Stage5_out_of_scope":
                list(
                    STAGE5_OUT_OF_SCOPE
                ),

            "predictive_ADB_in_Stage5":
                False,

            "DeepSense_in_Stage5":
                False,

            "full_joint_beam_ADB_analysis_in_Stage5":
                False,
        },

        "leakage_contract": {
            "formal_tuning":
                False,

            "formal_receiver_policy_selection":
                False,

            "formal_codebook_selection":
                False,

            "formal_threshold_selection":
                False,

            "tracks_to_predict_receiver_selector":
                False,

            "future_truth_receiver_selector":
                False,

            "future_GT_heading_controller":
                False,

            "Stage4_retraining":
                False,

            "Stage4_recalibration":
                False,

            "Stage4_normalization_refit":
                False,
        },

        "scientific_execution": {
            "training":
                False,

            "trajectory_inference":
                False,

            "receiver_inference":
                False,

            "beam_selection":
                False,

            "formal_evaluation":
                False,

            "recalibration":
                False,

            "dataset_scan":
                False,
        },

        "implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "upstream_modified":
            False,

        "legacy_runtime_dependency":
            False,

        "next":
            (
                "Block5.1 communication receiver "
                "selection and receiver geometry"
            ),
    }

    write_json(
        REPORT,
        report,
    )

    # ========================================================
    # I. Implementation log
    # ========================================================

    marker = (
        "## Block 5.0 — "
        "Stage5 bootstrap and contract freeze"
    )

    existing = (
        IMPLEMENTATION_LOG.read_text(
            encoding="utf-8"
        )
        if IMPLEMENTATION_LOG.is_file()
        else ""
    )

    if marker not in existing:
        with IMPLEMENTATION_LOG.open(
            "a",
            encoding="utf-8",
        ) as stream:

            stream.write(
                "\n"
                +
                marker
                +
                "\n\n"
                "Status: PASS / STRUCTURAL CONTRACT FROZEN\n\n"
                "- Frozen Stage4 calibrated Gaussian "
                  "is the mandatory Stage5 trajectory posterior.\n"
                "- Stage4 training, normalization fitting, "
                  "recalibration and post-hoc GMM selection "
                  "remain forbidden.\n"
                "- Primary receiver policy: nearest causal "
                  "vehicle ahead; connectivity is a constructed "
                  "experimental assumption, not a WOMD label.\n"
                "- Receiver geometry cases: centroid baseline, "
                  "known offset and uncertain offset.\n"
                "- Receiver offset uncertainty must be propagated "
                  "jointly with trajectory uncertainty.\n"
                "- Stage5 horizons: 0.1/0.3/0.5/1.0 s.\n"
                "- Codebook sizes: 16/32/64.\n"
                "- Adaptive Top-K requested masses: "
                  "90/95/97.5/99%; nominal default 95%.\n"
                "- Formal N=120 is evaluation-only and cannot "
                  "be used to tune receiver policy, codebook, "
                  "thresholds or fallback parameters.\n"
                "- Numerical controller parameters explicitly "
                  "listed in the development-freeze contract "
                  "must be frozen before Block5.8.\n"
                "- Optical chain is fixed as pointing error → "
                  "gain → received power → SNR → DPSK BER → "
                  "effective rate. Part-A freezes the underlying "
                  "link quantities/raw data rate; Stage5 derives "
                  "the overhead-aware effective-rate mapping and "
                  "must freeze it before formal evaluation.\n"
                "- Part-A notebook source and relevant evidence "
                  "cells are hashed and frozen for later "
                  "Block5.6 numerical cross-check.\n"
                "- Predictive ADB is Stage6 scope; full joint "
                  "beam+ADB evaluation is later scope; "
                  "DeepSense remains Stage8 scope.\n"
                "- No training, inference or formal evaluation "
                  "was performed in Block5.0.\n"
                f"- Stage5 contract SHA256: {contract_sha}.\n"
                f"- Part-A source contract SHA256: "
                  f"{part_a_contract_sha}.\n"
                f"- Block5.0 implementation SHA256: "
                  f"{implementation_sha}.\n"
            )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.0 PART 2/2 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "Part1 continuity             = PASS"
    )

    print(
        "frozen upstream hashes       = PASS"
    )

    print(
        "Part-A notebook source       = FROZEN"
    )

    print(
        "Part-A core constants        = PASS"
    )

    print(
        "Part-A DPSK evidence         = PASS"
    )

    print(
        "Part-A BER evidence          = PASS"
    )

    print(
        "Part-A optical power/gain    = PASS"
    )

    print(
        "effective-rate provenance    = STAGE5 DERIVED PASS"
    )

    print(
        "effective-rate formula       = FREEZE IN BLOCK5.6 BEFORE FORMAL"
    )

    print(
        "receiver geometry route      = PASS"
    )

    print(
        "Stage5 structural contract   = FROZEN"
    )

    print(
        "primary receiver policy      = NEAREST CAUSAL VEHICLE AHEAD"
    )

    print(
        "receiver modes               = CENTROID / KNOWN / UNCERTAIN"
    )

    print(
        "codebooks                    = 16 / 32 / 64"
    )

    print(
        "Top-K mass targets           = 90 / 95 / 97.5 / 99 %"
    )

    print(
        "nominal mass target          = 95 %"
    )

    print(
        "formal N=120                 = EVALUATION ONLY"
    )

    print(
        "formal tuning                = NO"
    )

    print(
        "tracks_to_predict selector   = NO"
    )

    print(
        "future truth selector        = NO"
    )

    print(
        "Stage4 retraining            = NO"
    )

    print(
        "Stage4 recalibration         = NO"
    )

    print(
        "Stage6+ scope pulled forward = NO"
    )

    print(
        "training/inference           = NO"
    )

    print(
        "upstream modified            = NO"
    )

    print(
        "contract SHA256              =",
        contract_sha,
    )

    print(
        "Part-A contract SHA256       =",
        part_a_contract_sha,
    )

    print(
        "implementation files         =",
        implementation_files,
    )

    print(
        "implementation SHA256        =",
        implementation_sha,
    )

    print(
        "STATUS = PASS"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "next = BLOCK 5.1"
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.0 PART 2/2 = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "exception =",
        type(
            exc
        ).__name__,
        str(
            exc
        ),
    )

    print()
    traceback.print_exc()

    print(
        "training/inference         = NO"
    )

    print(
        "formal evaluation          = NO"
    )

    print(
        "Stage4 retraining          = NO"
    )

    print(
        "Stage4 recalibration       = NO"
    )

    print(
        "upstream modified          = NO"
    )

    print(
        "terminal remains open      = YES"
    )

# Deliberately no sys.exit().
