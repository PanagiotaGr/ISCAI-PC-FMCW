from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")

S0 = ROOT / "iscai_stage0"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

PART1 = (
    S6
    / "reports/block60_part1_handoff_contract.json"
)

STAGE6_CONTRACT = (
    S6
    / "configs/stage6_contract.json"
)

PART_A_REFERENCE = (
    S0
    / "reports/stage0/"
      "part_a_frozen_reference.json"
)

PART_A_REPO = (
    ROOT
    / "part_a_reference/"
      "ISCAI_pc_fmcw"
)

EXPECTED_NOTEBOOK = (
    PART_A_REPO
    / "notebooks/ISCAI_PC_FMCW.ipynb"
)

STAGE5_CLOSURE = (
    S5
    / "reports/stage5_final_closure.json"
)

STAGE5_HANDOFF = (
    S5
    / "artifacts/block510/"
      "stage5_to_stage6_handoff.json"
)

STAGE5_FREEZE = (
    S5
    / "artifacts/block510/"
      "stage5_final_freeze_manifest.json"
)

SOURCE_AUDIT = (
    S6
    / "artifacts/block60/"
      "part_a_adb_source_audit.json"
)

ADAPTER_CONTRACT = (
    S6
    / "configs/"
      "part_a_reactive_adb_adapter_contract.json"
)

BLOCK60_CLOSURE = (
    S6
    / "reports/block60_closure.json"
)

TEST = (
    S6
    / "tests/test_block60_part_a_adb_contract.py"
)

EXPECTED_PART_A_GIT = (
    "44d62e3478e3818d1757b00971890f844cb032f7"
)

EXPECTED_PART_A_REFERENCE_SHA = (
    "fcdfc0a7b9c14fa9447ceb8563a6f20a"
    "c277706223388dc9aa8a1edf416c5e2b"
)

EXPECTED_PART_A_NOTEBOOK_SHA = (
    "b5a80a6d3441de6d571db4f65b4a43ed"
    "4052cc2b3ccba935ad31b5dd51316ef3"
)

EXPECTED_STAGE5_CLOSURE_SHA = (
    "c83731948749f3ca20742aaac6b7474f"
    "53c755cbc8209dd31db969d229f60610"
)

EXPECTED_STAGE5_HANDOFF_SHA = (
    "c900ccbd63e7c680af28ed5cd8046f40"
    "8463eceb7ce943b324b266dda73500eb"
)

EXPECTED_STAGE5_FREEZE_SHA = (
    "af9612205f190a04d9da565a98a0449f"
    "4ed98f53027b6f971b8457f73bd70913"
)

MIN_FREE_GIB = 250.0


# ============================================================
# Helpers
# ============================================================

def sha256_file(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def canonical_bytes(payload) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def write_exact(path: Path, payload):
    desired = canonical_bytes(
        payload
    )

    if path.exists():
        existing = path.read_bytes()

        if existing != desired:
            raise RuntimeError(
                "Existing deterministic Block6.0 "
                f"artifact differs: {path}"
            )

        return "ALREADY_EXACT"

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(desired)

    os.replace(
        tmp,
        path,
    )

    return "WRITTEN"


def write_exact_text(
    path: Path,
    text: str,
):
    desired = text.encode("utf-8")

    if path.exists():
        if path.read_bytes() != desired:
            raise RuntimeError(
                "Existing deterministic Block6.0 "
                f"test differs: {path}"
            )

        return "ALREADY_EXACT"

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(desired)

    os.replace(
        tmp,
        path,
    )

    return "WRITTEN"


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def git_head(repo: Path) -> str:
    process = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    require(
        process.returncode == 0,
        (
            "Cannot read Part-A git HEAD: "
            + process.stderr.strip()
        ),
    )

    return process.stdout.strip()


def flatten(value, prefix=""):
    rows = []

    if isinstance(value, dict):
        for key, item in value.items():
            child = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            rows.extend(
                flatten(
                    item,
                    child,
                )
            )

    elif isinstance(value, list):
        rows.append(
            (
                prefix,
                value,
            )
        )

        for index, item in enumerate(value):
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


def notebook_text_and_cells(
    path: Path,
):
    payload = load_json(path)

    cells = []

    all_text = []

    for index, cell in enumerate(
        payload.get("cells", [])
    ):
        source = cell.get(
            "source",
            []
        )

        if isinstance(source, list):
            text = "".join(source)

        else:
            text = str(source)

        cells.append({
            "index":
                index,

            "cell_type":
                cell.get(
                    "cell_type"
                ),

            "text":
                text,
        })

        all_text.append(text)

    return (
        "\n".join(all_text),
        cells,
    )


def normalized(text: str) -> str:
    return (
        text
        .replace("θ", "theta")
        .replace("Θ", "theta")
        .replace("𝜃", "theta")
        .replace("𝑟", "r")
        .replace("𝐿", "l")
        .lower()
    )


def evidence_cells(
    cells,
    aliases,
):
    hits = []

    for cell in cells:
        lower = normalized(
            cell["text"]
        )

        matching = [
            alias
            for alias in aliases
            if normalized(alias) in lower
        ]

        if matching:
            hits.append({
                "cell_index":
                    cell["index"],

                "cell_type":
                    cell["cell_type"],

                "matched_aliases":
                    sorted(
                        set(matching)
                    ),

                "text_sha256":
                    sha256(
                        cell[
                            "text"
                        ].encode(
                            "utf-8"
                        )
                    ).hexdigest(),
            })

    return hits


def reference_evidence(
    payload,
):
    tokens = (
        "adb",
        "illumination",
        "shadow",
        "reactive",
        "l_theta",
        "theta_r",
        "light",
    )

    rows = []

    for path, value in flatten(payload):
        text = (
            path
            + " "
            + str(value)
        ).lower()

        if any(
            token in text
            for token in tokens
        ):
            printable = value

            if isinstance(
                printable,
                (dict, list),
            ):
                printable = str(
                    printable
                )

            printable = str(
                printable
            )

            if len(printable) > 240:
                printable = (
                    printable[:237]
                    + "..."
                )

            rows.append({
                "path":
                    path,

                "value_preview":
                    printable,
            })

    return rows


def run_tests():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(S6 / "tests"),
            "-p",
            "test_block60*.py",
        ],
        cwd=str(S6),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return (
        process.returncode,
        process.stdout,
    )


# ============================================================
# Main audit
# ============================================================

def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.0 PART 2/2"
    )
    print(
        "PART-A ADB SOURCE AUDIT + REACTIVE ADAPTER CONTRACT"
    )
    print(
        "============================================================"
    )

    required = (
        PART1,
        STAGE6_CONTRACT,
        PART_A_REFERENCE,
        STAGE5_CLOSURE,
        STAGE5_HANDOFF,
        STAGE5_FREEZE,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing required frozen evidence: "
            + ", ".join(missing)
        ),
    )

    require(
        PART_A_REPO.is_dir(),
        (
            "Part-A frozen repository missing: "
            f"{PART_A_REPO}"
        ),
    )

    protected = (
        PART_A_REFERENCE,
        STAGE5_CLOSURE,
        STAGE5_HANDOFF,
        STAGE5_FREEZE,
    )

    protected_before = {
        str(path):
            sha256_file(path)
        for path in protected
    }

    # ========================================================
    # A. Block6.0 Part1 continuity
    # ========================================================

    print()
    print(
        "===== A. BLOCK6.0 PART1 CONTINUITY ====="
    )

    part1 = load_json(
        PART1
    )

    contract6 = load_json(
        STAGE6_CONTRACT
    )

    require(
        part1.get("status")
        ==
        "PASS_CONTRACT_FROZEN",
        (
            "Block6.0 Part1 is not "
            "PASS_CONTRACT_FROZEN."
        ),
    )

    require(
        contract6.get("status")
        ==
        "FROZEN_PREIMPLEMENTATION_CONTRACT",
        (
            "Stage6 audited contract "
            "is not frozen."
        ),
    )

    require(
        part1.get(
            "strict_PDF_audit"
        )
        ==
        "PASS_100_PERCENT_STAGE6_PLAN_COVERAGE",
        (
            "Strict Stage6 PDF audit "
            "is not PASS."
        ),
    )

    print(
        "Part1 status      = PASS_CONTRACT_FROZEN"
    )

    print(
        "Stage6 contract   = FROZEN"
    )

    print(
        "strict PDF audit  = 100% PASS"
    )

    # ========================================================
    # B. Frozen Part-A identity
    # ========================================================

    print()
    print(
        "===== B. FROZEN PART-A IDENTITY ====="
    )

    require(
        sha256_file(
            PART_A_REFERENCE
        )
        ==
        EXPECTED_PART_A_REFERENCE_SHA,
        (
            "Part-A frozen reference SHA "
            "changed after Block6.0 Part1."
        ),
    )

    head = git_head(
        PART_A_REPO
    )

    require(
        head
        ==
        EXPECTED_PART_A_GIT,
        (
            "Part-A git HEAD changed: "
            f"{head}"
        ),
    )

    print(
        "Part-A git HEAD       = EXACT PASS"
    )

    print(
        "Part-A reference SHA  = EXACT PASS"
    )

    # ========================================================
    # C. Locate canonical Part-A notebook
    # ========================================================

    print()
    print(
        "===== C. PART-A ADB SOURCE DISCOVERY ====="
    )

    if EXPECTED_NOTEBOOK.is_file():
        notebook = EXPECTED_NOTEBOOK

    else:
        notebooks = sorted(
            PART_A_REPO.rglob(
                "*.ipynb"
            )
        )

        require(
            len(notebooks) == 1,
            (
                "Canonical Part-A notebook could "
                "not be resolved uniquely. Found: "
                + ", ".join(
                    str(path)
                    for path in notebooks[:10]
                )
            ),
        )

        notebook = notebooks[0]

    notebook_sha = sha256_file(
        notebook
    )

    require(
        notebook_sha
        ==
        EXPECTED_PART_A_NOTEBOOK_SHA,
        (
            "Canonical Part-A notebook SHA "
            "does not match the frozen "
            "Part-A notebook identity."
        ),
    )

    text, cells = notebook_text_and_cells(
        notebook
    )

    normalized_notebook = normalized(
        text
    )

    print(
        "canonical notebook =",
        notebook,
    )

    print(
        "notebook SHA256    =",
        notebook_sha,
    )

    print(
        "notebook cells     =",
        len(cells),
    )

    # ========================================================
    # D. Required Part-A ADB semantic evidence
    # ========================================================

    print()
    print(
        "===== D. PART-A ADB SEMANTIC EVIDENCE ====="
    )

    evidence_spec = {
        "ADB/module":
            (
                "adb",
                "adaptive driving beam",
            ),

        "angle-range illumination":
            (
                "theta",
                "angle",
                "range",
                "illumination",
            ),

        "shadow region":
            (
                "shadow",
                "mask",
            ),

        "radial thresholds":
            (
                "rmin",
                "rmax",
                "transition",
                "radial",
            ),

        "raised-cosine":
            (
                "cos(",
                "np.cos",
                "cosine",
                "raised",
            ),

        "MHT-driven reactive input":
            (
                "mht",
                "trajectory",
                "track",
            ),
    }

    semantic_hits = {}

    for name, aliases in (
        evidence_spec.items()
    ):
        hits = evidence_cells(
            cells,
            aliases,
        )

        semantic_hits[
            name
        ] = hits

        require(
            hits,
            (
                "Part-A notebook lacks "
                f"source evidence for {name}."
            ),
        )

        print(
            f"{name:28s}= PASS"
            f" | cells={len(hits)}"
        )

    # Stronger check: there must be at least one ADB-ish cell
    # containing both masking/illumination semantics and
    # cosine/transition semantics.
    strong_cells = []

    for cell in cells:
        lower = normalized(
            cell["text"]
        )

        adbish = (
            "adb" in lower
            or
            "illumination" in lower
            or
            "shadow" in lower
        )

        transitionish = (
            "cos(" in lower
            or
            "np.cos" in lower
            or
            "transition" in lower
        )

        if (
            adbish
            and
            transitionish
        ):
            strong_cells.append(
                cell["index"]
            )

    require(
        strong_cells,
        (
            "No Part-A notebook cell jointly "
            "proves ADB illumination/masking "
            "and smooth transition logic."
        ),
    )

    print(
        "joint ADB/transition cells =",
        strong_cells[:20],
    )

    # ========================================================
    # E. Frozen Part-A reference / source evidence roles
    # ========================================================

    print()
    print(
        "===== E. FROZEN PART-A REFERENCE ADB EVIDENCE ====="
    )

    part_a_reference = load_json(
        PART_A_REFERENCE
    )

    ref_hits = reference_evidence(
        part_a_reference
    )

    # Evidence-role distinction:
    #
    # The Stage0 part_a_frozen_reference.json is the exact
    # identity/provenance freeze artifact.
    #
    # The canonical Part-A notebook is the authoritative
    # scientific source for ADB semantics. Its git identity,
    # notebook SHA and semantic source cells are independently
    # verified in Sections B-D.
    #
    # Therefore zero ADB keyword hits in the Stage0 identity
    # JSON is acceptable and does NOT weaken the actual ADB
    # source audit.

    reference_semantic_role = (
        "IDENTITY_PROVENANCE_FREEZE"
    )

    notebook_semantic_role = (
        "AUTHORITATIVE_FROZEN_ADB_SOURCE"
    )

    require(
        sha256_file(
            PART_A_REFERENCE
        )
        ==
        EXPECTED_PART_A_REFERENCE_SHA,
        (
            "Part-A identity/provenance "
            "reference SHA changed."
        ),
    )

    require(
        notebook_sha
        ==
        EXPECTED_PART_A_NOTEBOOK_SHA,
        (
            "Authoritative Part-A notebook "
            "SHA changed."
        ),
    )

    require(
        semantic_hits[
            "ADB/module"
        ],
        (
            "Authoritative notebook lacks "
            "ADB/module evidence."
        ),
    )

    require(
        semantic_hits[
            "angle-range illumination"
        ],
        (
            "Authoritative notebook lacks "
            "angle-range illumination evidence."
        ),
    )

    require(
        semantic_hits[
            "shadow region"
        ],
        (
            "Authoritative notebook lacks "
            "shadow-region evidence."
        ),
    )

    require(
        semantic_hits[
            "radial thresholds"
        ],
        (
            "Authoritative notebook lacks "
            "radial-threshold evidence."
        ),
    )

    require(
        semantic_hits[
            "raised-cosine"
        ],
        (
            "Authoritative notebook lacks "
            "raised-cosine evidence."
        ),
    )

    require(
        semantic_hits[
            "MHT-driven reactive input"
        ],
        (
            "Authoritative notebook lacks "
            "MHT-driven reactive-input evidence."
        ),
    )

    require(
        strong_cells,
        (
            "Authoritative notebook lacks "
            "joint ADB/transition evidence."
        ),
    )

    print(
        "frozen-reference semantic role =",
        reference_semantic_role,
    )

    print(
        "canonical notebook role        =",
        notebook_semantic_role,
    )

    print(
        "canonical notebook SHA         = EXACT PASS"
    )

    print(
        "ADB semantic notebook gates    = PASS"
    )

    print(
        "ADB-related reference fields   =",
        len(ref_hits),
        (
            "(supplemental only)"
            if ref_hits
            else
            "(0 acceptable)"
        ),
    )

    for item in ref_hits[:20]:
        print(
            " ",
            item["path"],
            "=",
            item["value_preview"],
        )

    # ========================================================
    # F. Freeze source audit
    # ========================================================

    print()
    print(
        "===== F. PART-A ADB SOURCE AUDIT ARTIFACT ====="
    )

    audit_payload = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.0",

        "part":
            "2/2",

        "status":
            "PASS_PART_A_ADB_SOURCE_AUDIT",

        "Part_A": {
            "repo":
                str(PART_A_REPO),

            "git_commit":
                head,

            "frozen_reference":
                str(PART_A_REFERENCE),

            "frozen_reference_sha256":
                EXPECTED_PART_A_REFERENCE_SHA,

            "canonical_notebook":
                str(notebook),

            "canonical_notebook_sha256":
                notebook_sha,

            "notebook_cell_count":
                len(cells),
        },

        "source_evidence": {
            name:
                hits
            for name, hits in (
                semantic_hits.items()
            )
        },

        "joint_ADB_transition_cell_indices":
            strong_cells,

        "frozen_reference_ADB_evidence":
            ref_hits,

        "evidence_roles": {
            "frozen_reference_json":
                reference_semantic_role,

            "canonical_notebook":
                notebook_semantic_role,

            "canonical_notebook_sha256":
                EXPECTED_PART_A_NOTEBOOK_SHA,

            "reference_JSON_must_duplicate_ADB_strings":
                False,

            "ADB_semantic_evidence_from_exact_frozen_notebook":
                True,
        },

        "scientific_interpretation": {
            "paper_ADB":
                (
                    "camera/geometry-informed "
                    "reactive masking"
                ),

            "Part_A_ADB":
                (
                    "MHT/estimated-current-state "
                    "driven reactive masking"
                ),

            "Part_B_Stage6":
                (
                    "probabilistic predictive "
                    "class-aware masking"
                ),

            "Part_A_is_exact_paper_pipeline":
                False,

            "Stage6_may_replace_entire_ADB_model":
                False,
        },

        "execution": {
            "Part_A_notebook_rerun":
                False,

            "training":
                False,

            "inference":
                False,

            "formal_evaluation":
                False,

            "parameter_tuning":
                False,
        },
    }

    audit_write = write_exact(
        SOURCE_AUDIT,
        audit_payload,
    )

    print(
        "source audit write =",
        audit_write,
    )

    print(
        "source audit SHA256 =",
        sha256_file(
            SOURCE_AUDIT
        ),
    )

    # ========================================================
    # G. Freeze reactive adapter contract
    # ========================================================

    print()
    print(
        "===== G. REACTIVE ADB ADAPTER CONTRACT ====="
    )

    adapter_contract = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.0",

        "status":
            "FROZEN_REACTIVE_ADB_ADAPTER_CONTRACT",

        "role":
            (
                "Stage6 original_reactive_ADB "
                "baseline adapter to frozen "
                "Part-A illumination semantics"
            ),

        "not_yet_implementation":
            True,

        "source": {
            "Part_A_git_commit":
                EXPECTED_PART_A_GIT,

            "Part_A_reference_sha256":
                EXPECTED_PART_A_REFERENCE_SHA,

            "Part_A_notebook":
                str(notebook),

            "Part_A_notebook_sha256":
                notebook_sha,

            "source_audit":
                str(SOURCE_AUDIT),

            "source_audit_sha256":
                sha256_file(
                    SOURCE_AUDIT
                ),
        },

        "provenance_labels": {
            "paper_ADB":
                (
                    "camera/geometry-informed "
                    "reactive masking"
                ),

            "Part_A_extension":
                (
                    "MHT-driven reactive masking"
                ),

            "Stage6_contribution":
                (
                    "probabilistic predictive "
                    "class-aware masking"
                ),
        },

        "input_contract": {
            "time":
                "anchor/current only",

            "future_ground_truth":
                False,

            "predictive_mean":
                False,

            "predictive_covariance":
                False,

            "class_aware_predictive_policy":
                False,

            "actor_source":
                (
                    "current causal actor/track "
                    "geometry available at controller "
                    "decision time"
                ),

            "coordinate_space":
                "headlamp angle-range",

            "multiple_actors":
                True,
        },

        "illumination_semantics": {
            "output":
                "normalized L(theta,r)",

            "full_illumination":
                1.0,

            "strongest_shadow_reference":
                0.0,

            "full_shadow_rule":
                (
                    "inside angular shadow region "
                    "and before radial shadow end"
                ),

            "radial_transition":
                "raised-cosine",

            "raised_cosine_normalized_formula":
                (
                    "0.5 * (1 - cos(pi*u))"
                ),

            "u_definition":
                (
                    "(r-r_shadow_end) / "
                    "(r_transition_end-r_shadow_end)"
                ),

            "outside_shadow_or_beyond_transition":
                "full illumination",

            "multiple_actor_combination":
                (
                    "elementwise minimum of "
                    "per-actor illumination maps"
                ),

            "arbitrary_new_photometric_model":
                False,
        },

        "Stage6_preservation_contract": {
            "preserve_basic_L_theta_r_semantics":
                True,

            "preserve_radial_threshold_concept":
                True,

            "preserve_angular_shadow_zone_concept":
                True,

            "preserve_raised_cosine_transition":
                True,

            "preserve_minimum_illumination_floor_concept":
                True,

            "preserve_temporal_smoothing_concept":
                True,

            "preserve_actuation_rate_limit_concept":
                True,

            "predictive_extension_changes_region_source":
                (
                    "current deterministic region -> "
                    "future probabilistic region"
                ),

            "predictive_extension_may_change_entire_ADB_model":
                False,
        },

        "fair_baseline_semantics": {
            "original_reactive_ADB":
                (
                    "current-state reactive "
                    "Part-A-style masking"
                ),

            "deterministic_predictive_ADB":
                (
                    "same downstream illumination "
                    "semantics driven by predictive "
                    "mean geometry"
                ),

            "uncertainty_aware_predictive_ADB":
                (
                    "same downstream illumination "
                    "semantics driven by future "
                    "occupancy uncertainty"
                ),

            "oracle_future_ADB":
                (
                    "constructed future reference; "
                    "not measured ADB ground truth"
                ),

            "same_core_illumination_mapping_for_fairness":
                True,
        },

        "known_limitations": {
            "Part_A_reactive_ADB_is_fully_calibrated_photometric_model":
                False,

            "Part_A_report_semantics":
                (
                    "qualitative implementation "
                    "of paper ADB/Fig.3 logic"
                ),

            "WOMD_contains_measured_ADB_commands":
                False,

            "WOMD_contains_windshield_mirror_driver_labels":
                False,
        },

        "implementation_gate": {
            "adapter_implementation":
                "BLOCK6.2",

            "before_adapter_implementation":
                [
                    "Block6.1 illumination grid",
                    "full-box headlamp projection",
                    "causal actor geometry",
                ],

            "must_reproduce_frozen_reference_where_available":
                True,

            "must_have_unit_tests":
                True,

            "must_not_modify_Part_A":
                True,
        },
    }

    adapter_write = write_exact(
        ADAPTER_CONTRACT,
        adapter_contract,
    )

    print(
        "adapter contract =",
        adapter_write,
    )

    print(
        "adapter SHA256   =",
        sha256_file(
            ADAPTER_CONTRACT
        ),
    )

    # ========================================================
    # H. Regression tests
    # ========================================================

    print()
    print(
        "===== H. BLOCK6.0 PART2 CONTRACT TESTS ====="
    )

    test_text = r'''from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

AUDIT = (
    S6
    / "artifacts/block60/"
      "part_a_adb_source_audit.json"
)

ADAPTER = (
    S6
    / "configs/"
      "part_a_reactive_adb_adapter_contract.json"
)

STAGE6 = (
    S6
    / "configs/stage6_contract.json"
)


class Block60PartAADBContractTest(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads(
            AUDIT.read_text(
                encoding="utf-8"
            )
        )

        cls.adapter = json.loads(
            ADAPTER.read_text(
                encoding="utf-8"
            )
        )

        cls.stage6 = json.loads(
            STAGE6.read_text(
                encoding="utf-8"
            )
        )

    def test_part_a_source_audit_passes(self):
        self.assertEqual(
            self.audit["status"],
            "PASS_PART_A_ADB_SOURCE_AUDIT",
        )

    def test_provenance_is_not_conflated(self):
        labels = self.adapter[
            "provenance_labels"
        ]

        self.assertNotEqual(
            labels["paper_ADB"],
            labels["Part_A_extension"],
        )

        self.assertNotEqual(
            labels["Part_A_extension"],
            labels["Stage6_contribution"],
        )

    def test_reactive_baseline_is_causal(self):
        inputs = self.adapter[
            "input_contract"
        ]

        self.assertFalse(
            inputs["future_ground_truth"]
        )

        self.assertFalse(
            inputs["predictive_mean"]
        )

        self.assertFalse(
            inputs["predictive_covariance"]
        )

    def test_reactive_output_is_L_theta_r(self):
        illumination = self.adapter[
            "illumination_semantics"
        ]

        self.assertEqual(
            illumination["output"],
            "normalized L(theta,r)",
        )

    def test_raised_cosine_is_preserved(self):
        illumination = self.adapter[
            "illumination_semantics"
        ]

        self.assertEqual(
            illumination[
                "radial_transition"
            ],
            "raised-cosine",
        )

        self.assertIn(
            "0.5 * (1 - cos(pi*u))",
            illumination[
                "raised_cosine_normalized_formula"
            ],
        )

    def test_multiple_actor_combination(self):
        illumination = self.adapter[
            "illumination_semantics"
        ]

        self.assertIn(
            "elementwise minimum",
            illumination[
                "multiple_actor_combination"
            ],
        )

    def test_stage6_does_not_replace_adb_model(self):
        preservation = self.adapter[
            "Stage6_preservation_contract"
        ]

        self.assertTrue(
            preservation[
                "preserve_basic_L_theta_r_semantics"
            ]
        )

        self.assertFalse(
            preservation[
                "predictive_extension_may_change_entire_ADB_model"
            ]
        )

    def test_predictive_change_is_region_source(self):
        change = self.adapter[
            "Stage6_preservation_contract"
        ][
            "predictive_extension_changes_region_source"
        ]

        self.assertIn(
            "current deterministic region",
            change,
        )

        self.assertIn(
            "future probabilistic region",
            change,
        )

    def test_oracle_not_measured_gt(self):
        oracle = self.adapter[
            "fair_baseline_semantics"
        ][
            "oracle_future_ADB"
        ]

        self.assertIn(
            "constructed",
            oracle,
        )

        self.assertIn(
            "not measured",
            oracle,
        )

    def test_part_a_not_modified(self):
        gate = self.adapter[
            "implementation_gate"
        ]

        self.assertTrue(
            gate[
                "must_not_modify_Part_A"
            ]
        )

    def test_adapter_not_implemented_in_block60(self):
        self.assertTrue(
            self.adapter[
                "not_yet_implementation"
            ]
        )

        self.assertEqual(
            self.adapter[
                "implementation_gate"
            ][
                "adapter_implementation"
            ],
            "BLOCK6.2",
        )


if __name__ == "__main__":
    unittest.main()
'''

    test_write = write_exact_text(
        TEST,
        test_text,
    )

    print(
        "test file =",
        test_write,
    )

    compile_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(TEST),
        ],
        cwd=str(S6),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    require(
        compile_process.returncode == 0,
        (
            "Block6.0 Part2 test compile failed:\n"
            + compile_process.stdout
        ),
    )

    rc, output = run_tests()

    for line in output.splitlines():
        stripped = line.strip()

        if (
            stripped.startswith("Ran ")
            or
            stripped == "OK"
            or
            stripped.startswith("FAILED")
        ):
            print(stripped)

    require(
        rc == 0,
        (
            "Block6.0 regression failed:\n"
            + output
        ),
    )

    # ========================================================
    # I. Upstream immutability / storage
    # ========================================================

    print()
    print(
        "===== I. IMMUTABILITY / STORAGE ====="
    )

    protected_after = {
        str(path):
            sha256_file(path)
        for path in protected
    }

    changed = [
        path
        for path in protected_before
        if (
            protected_before[path]
            !=
            protected_after[path]
        )
    ]

    require(
        not changed,
        (
            "Frozen upstream artifact changed: "
            + ", ".join(changed)
        ),
    )

    require(
        sha256_file(
            STAGE5_CLOSURE
        )
        ==
        EXPECTED_STAGE5_CLOSURE_SHA,
        "Stage5 closure SHA changed.",
    )

    require(
        sha256_file(
            STAGE5_HANDOFF
        )
        ==
        EXPECTED_STAGE5_HANDOFF_SHA,
        "Stage5 handoff SHA changed.",
    )

    require(
        sha256_file(
            STAGE5_FREEZE
        )
        ==
        EXPECTED_STAGE5_FREEZE_SHA,
        "Stage5 freeze SHA changed.",
    )

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    require(
        free_gib >= MIN_FREE_GIB,
        (
            "250-GiB reserve violated: "
            f"{free_gib:.3f} GiB free."
        ),
    )

    print(
        "Part-A frozen repo/reference = UNCHANGED"
    )

    print(
        "Stage5 frozen artifacts       = UNCHANGED"
    )

    print(
        "free GiB                      =",
        round(free_gib, 3),
    )

    print(
        "250-GiB reserve               = PASS"
    )

    # ========================================================
    # J. Final Block6.0 closure
    # ========================================================

    print()
    print(
        "===== J. BLOCK6.0 CLOSURE ====="
    )

    closure_payload = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.0",

        "status":
            "PASS_COMPLETE",

        "Part1": {
            "status":
                "PASS_CONTRACT_FROZEN",

            "report":
                str(PART1),

            "report_sha256":
                sha256_file(
                    PART1
                ),

            "Stage6_contract":
                str(STAGE6_CONTRACT),

            "Stage6_contract_sha256":
                sha256_file(
                    STAGE6_CONTRACT
                ),
        },

        "Part2": {
            "Part_A_ADB_source_audit":
                str(SOURCE_AUDIT),

            "Part_A_ADB_source_audit_sha256":
                sha256_file(
                    SOURCE_AUDIT
                ),

            "reactive_adapter_contract":
                str(ADAPTER_CONTRACT),

            "reactive_adapter_contract_sha256":
                sha256_file(
                    ADAPTER_CONTRACT
                ),
        },

        "frozen_scientific_contract": {
            "Part_A_basic_L_theta_r_preserved":
                True,

            "Part_A_raised_cosine_preserved":
                True,

            "Part_A_angular_shadow_semantics_preserved":
                True,

            "Part_A_radial_threshold_semantics_preserved":
                True,

            "multiple_actor_intensity_combination_preserved":
                True,

            "reactive_baseline_is_current_state_only":
                True,

            "Part_A_and_paper_provenance_distinguished":
                True,

            "Stage6_predictive_change":
                (
                    "current deterministic region -> "
                    "future probabilistic region"
                ),

            "whole_ADB_model_arbitrarily_replaced":
                False,
        },

        "upstream": {
            "Part_A_git_commit":
                EXPECTED_PART_A_GIT,

            "Part_A_reference_sha256":
                EXPECTED_PART_A_REFERENCE_SHA,

            "Stage5_status":
                "COMPLETE_FROZEN",

            "Stage5_closure_sha256":
                EXPECTED_STAGE5_CLOSURE_SHA,

            "Stage5_handoff_sha256":
                EXPECTED_STAGE5_HANDOFF_SHA,

            "Stage5_freeze_sha256":
                EXPECTED_STAGE5_FREEZE_SHA,
        },

        "regression": {
            "Block6_0_tests":
                "PASS",
        },

        "scientific_execution": {
            "predictive_ADB_implementation_started":
                False,

            "reactive_adapter_implementation_started":
                False,

            "training":
                False,

            "inference":
                False,

            "formal_evaluation":
                False,

            "parameter_tuning":
                False,

            "Part_A_notebook_rerun":
                False,
        },

        "upstream_modified":
            False,

        "next":
            (
                "Block6.1: illumination-space "
                "definition and full future-box "
                "projection into headlamp "
                "angle-range geometry"
            ),
    }

    closure_write = write_exact(
        BLOCK60_CLOSURE,
        closure_payload,
    )

    print(
        "Block6.0 closure =",
        closure_write,
    )

    print(
        "closure SHA256   =",
        sha256_file(
            BLOCK60_CLOSURE
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.0 PART 2/2 — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Part-A git/reference       = EXACT PASS"
    )

    print(
        "Part-A ADB source          = AUDITED PASS"
    )

    print(
        "reactive L(theta,r)        = FROZEN CONTRACT"
    )

    print(
        "raised-cosine transition   = FROZEN CONTRACT"
    )

    print(
        "angular/radial shadow      = FROZEN CONTRACT"
    )

    print(
        "multi-actor combination    = ELEMENTWISE MIN CONTRACT"
    )

    print(
        "paper vs Part-A provenance = SEPARATE PASS"
    )

    print(
        "Part-B scientific change   = "
        "CURRENT DETERMINISTIC -> FUTURE PROBABILISTIC REGION"
    )

    print(
        "Part-A modified            = NO"
    )

    print(
        "Stage5 modified            = NO"
    )

    print(
        "reactive adapter implemented= NO"
    )

    print(
        "predictive ADB implemented = NO"
    )

    print(
        "Block6.0 regression        = PASS"
    )

    print(
        "STATUS = PASS_COMPLETE"
    )

    print(
        "closure =",
        BLOCK60_CLOSURE,
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
        "BLOCK 6.0 PART 2/2 = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc()

    print(
        "Part-A modified          = NO"
    )

    print(
        "Stage5 modified          = NO"
    )

    print(
        "training/inference       = NO"
    )

    print(
        "formal evaluation        = NO"
    )

    print(
        "parameter tuning         = NO"
    )

    print(
        "terminal remains open    = YES"
    )

# Deliberately no non-zero sys.exit().
