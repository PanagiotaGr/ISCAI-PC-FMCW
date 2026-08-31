from __future__ import annotations

from pathlib import Path
import py_compile
import subprocess
import sys
import traceback


ROOT = Path(
    "/home/agni/waymo"
)

STAGE5 = (
    ROOT
    / "iscai_stage5"
)

CONTRACTS = (
    STAGE5
    / "src/iscai_stage5/contracts.py"
)

TESTS = (
    STAGE5
    / "tests/test_block50_contracts.py"
)

RUNNER = (
    STAGE5
    / "scripts/run_block50_contract_freeze.py"
)

SAFE = (
    STAGE5
    / "scripts/run_block50_part2_safe.py"
)

STALE = (
    STAGE5
    / "artifacts/block50/"
      "part_a_source_contract.json",

    STAGE5
    / "configs/"
      "stage5_contract.json",

    STAGE5
    / "reports/"
      "block50_contract_freeze.json",
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


def replace_once(
    text,
    old,
    new,
    *,
    label,
):
    count = text.count(
        old
    )

    require(
        count == 1,
        (
            f"{label}: expected exactly "
            f"one source match; found {count}."
        ),
    )

    return text.replace(
        old,
        new,
        1,
    )


def patch_contracts():
    text = CONTRACTS.read_text(
        encoding="utf-8"
    )

    old = '''PART_A_NUMERICAL_CROSSCHECK_REQUIRED = True

BEAM_PROBING_OVERHEAD_MUST_AFFECT_EFFECTIVE_RATE = True
'''

    new = '''PART_A_NUMERICAL_CROSSCHECK_REQUIRED = True

#
# The frozen Part-A notebook contains the physical
# communication/link basis through received power,
# SNR/DPSK BER and the raw data-rate constant.
#
# It does NOT contain a source-level Stage5
# overhead-aware effective-rate implementation.
#
# Therefore Stage5 must derive effective rate from
# the frozen Part-A link quantities plus explicit
# Stage5 beam-probing overhead.  The exact formula
# must be frozen on non-formal development evidence
# before the formal N=120 evaluation.
#
PART_A_SOURCE_EFFECTIVE_RATE_REQUIRED = False

EFFECTIVE_RATE_PROVENANCE = (
    "Stage5_derived_from_frozen_PartA_link_"
    "quantities_raw_data_rate_and_beam_probing_overhead"
)

EFFECTIVE_RATE_FORMULA_STATUS = (
    "MUST_FREEZE_IN_BLOCK5.6_BEFORE_FORMAL"
)

BEAM_PROBING_OVERHEAD_MUST_AFFECT_EFFECTIVE_RATE = True
'''

    text = replace_once(
        text,
        old,
        new,
        label="contracts Part-A provenance",
    )

    old = '''            "PartA_numerical_crosscheck_required":
                PART_A_NUMERICAL_CROSSCHECK_REQUIRED,

            "probing_overhead_affects_effective_rate":
                BEAM_PROBING_OVERHEAD_MUST_AFFECT_EFFECTIVE_RATE,
'''

    new = '''            "PartA_numerical_crosscheck_required":
                PART_A_NUMERICAL_CROSSCHECK_REQUIRED,

            "PartA_source_effective_rate_required":
                PART_A_SOURCE_EFFECTIVE_RATE_REQUIRED,

            "effective_rate_provenance":
                EFFECTIVE_RATE_PROVENANCE,

            "effective_rate_formula_status":
                EFFECTIVE_RATE_FORMULA_STATUS,

            "probing_overhead_affects_effective_rate":
                BEAM_PROBING_OVERHEAD_MUST_AFFECT_EFFECTIVE_RATE,
'''

    text = replace_once(
        text,
        old,
        new,
        label="contract_dict effective-rate fields",
    )

    CONTRACTS.write_text(
        text,
        encoding="utf-8",
    )


def patch_tests():
    text = TESTS.read_text(
        encoding="utf-8"
    )

    old = '''    DEFAULT_PROBABILITY_MASS_TARGET,
    FORMAL_POPULATION_N,
'''

    new = '''    DEFAULT_PROBABILITY_MASS_TARGET,
    EFFECTIVE_RATE_FORMULA_STATUS,
    EFFECTIVE_RATE_PROVENANCE,
    FORMAL_POPULATION_N,
'''

    text = replace_once(
        text,
        old,
        new,
        label="test imports A",
    )

    old = '''    OPTICAL_LINK_CHAIN,
    PRIMARY_RECEIVER_POLICY,
'''

    new = '''    OPTICAL_LINK_CHAIN,
    PART_A_SOURCE_EFFECTIVE_RATE_REQUIRED,
    PRIMARY_RECEIVER_POLICY,
'''

    text = replace_once(
        text,
        old,
        new,
        label="test imports B",
    )

    old = '''        self.assertFalse(
            NEW_ARBITRARY_OPTICAL_MODEL_ALLOWED
        )

    def test_10_formal_is_evaluation_only(self):
'''

    new = '''        self.assertFalse(
            NEW_ARBITRARY_OPTICAL_MODEL_ALLOWED
        )

        self.assertFalse(
            PART_A_SOURCE_EFFECTIVE_RATE_REQUIRED
        )

        self.assertIn(
            "Stage5_derived",
            EFFECTIVE_RATE_PROVENANCE,
        )

        self.assertEqual(
            EFFECTIVE_RATE_FORMULA_STATUS,
            "MUST_FREEZE_IN_BLOCK5.6_BEFORE_FORMAL",
        )

    def test_10_formal_is_evaluation_only(self):
'''

    text = replace_once(
        text,
        old,
        new,
        label="effective-rate semantic test",
    )

    TESTS.write_text(
        text,
        encoding="utf-8",
    )


def patch_runner():
    text = RUNNER.read_text(
        encoding="utf-8"
    )

    old = '''    CALIBRATION_VARIANCE_SCALE_ALPHA_H,
    CODEBOOK_SIZES,
    DEFAULT_PROBABILITY_MASS_TARGET,
    DEVELOPMENT_FIELDS_TO_FREEZE_BEFORE_FORMAL,
'''

    new = '''    CALIBRATION_VARIANCE_SCALE_ALPHA_H,
    CODEBOOK_SIZES,
    DEFAULT_PROBABILITY_MASS_TARGET,
    DEVELOPMENT_FIELDS_TO_FREEZE_BEFORE_FORMAL,
    EFFECTIVE_RATE_FORMULA_STATUS,
    EFFECTIVE_RATE_PROVENANCE,
'''

    text = replace_once(
        text,
        old,
        new,
        label="runner imports A",
    )

    old = '''    OPTICAL_LINK_CHAIN,
    PRIMARY_RECEIVER_POLICY,
'''

    new = '''    OPTICAL_LINK_CHAIN,
    PART_A_SOURCE_EFFECTIVE_RATE_REQUIRED,
    PRIMARY_RECEIVER_POLICY,
'''

    text = replace_once(
        text,
        old,
        new,
        label="runner imports B",
    )

    # --------------------------------------------------------
    # Replace the incorrect requirement that all four groups,
    # including effective_rate, must exist in Part-A source.
    # --------------------------------------------------------

    section_marker = (
        '        "D. PART-A OPTICAL / DPSK '
        'EVIDENCE-CELL FREEZE"'
    )

    section_index = text.find(
        section_marker
    )

    require(
        section_index >= 0,
        (
            "Could not locate Part-A evidence "
            "section in runner."
        ),
    )

    loop_start = text.find(
        '    for group in (\n'
        '        "DPSK",',
        section_index,
    )

    require(
        loop_start >= 0,
        (
            "Could not locate mandatory "
            "Part-A evidence loop."
        ),
    )

    contract_start = text.find(
        "    part_a_contract = {",
        loop_start,
    )

    require(
        contract_start > loop_start,
        (
            "Could not locate Part-A "
            "contract dictionary."
        ),
    )

    replacement = '''    #
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

'''

    text = (
        text[
            :loop_start
        ]
        +
        replacement
        +
        text[
            contract_start:
        ]
    )

    old = '''        "communication_evidence_cells":
            evidence_cells,

        "Stage5_link_adapter_policy":
'''

    new = '''        "communication_evidence_cells":
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
'''

    text = replace_once(
        text,
        old,
        new,
        label="Part-A contract provenance",
    )

    old = '''            "effective_rate":
                "PASS",

            "exact_link_adapter_implemented":
'''

    new = '''            "effective_rate":
                "STAGE5_DERIVED_REQUIRED_METRIC",

            "effective_rate_provenance":
                EFFECTIVE_RATE_PROVENANCE,

            "effective_rate_formula_status":
                EFFECTIVE_RATE_FORMULA_STATUS,

            "PartA_source_effective_rate_required":
                False,

            "exact_link_adapter_implemented":
'''

    text = replace_once(
        text,
        old,
        new,
        label="report effective-rate semantics",
    )

    old = '''    print(
        "Part-A effective rate        = PASS"
    )
'''

    new = '''    print(
        "effective-rate provenance    = STAGE5 DERIVED PASS"
    )

    print(
        "effective-rate formula       = FREEZE IN BLOCK5.6 BEFORE FORMAL"
    )
'''

    text = replace_once(
        text,
        old,
        new,
        label="final gate effective-rate print",
    )

    old = '''                "- Optical chain is fixed as pointing error → "
                  "gain → received power → SNR → DPSK BER → "
                  "effective rate, using/reproducing frozen "
                  "Part-A assumptions.\\n"
'''

    new = '''                "- Optical chain is fixed as pointing error → "
                  "gain → received power → SNR → DPSK BER → "
                  "effective rate. Part-A freezes the underlying "
                  "link quantities/raw data rate; Stage5 derives "
                  "the overhead-aware effective-rate mapping and "
                  "must freeze it before formal evaluation.\\n"
'''

    text = replace_once(
        text,
        old,
        new,
        label="implementation-log wording",
    )

    RUNNER.write_text(
        text,
        encoding="utf-8",
    )


def remove_stale_partial_outputs():
    for path in STALE:

        if path.exists():
            path.unlink()

            print(
                "removed stale partial output:",
                path,
            )


def compile_modified():
    for path in (
        CONTRACTS,
        TESTS,
        RUNNER,
        SAFE,
    ):
        py_compile.compile(
            str(
                path
            ),
            doraise=True,
        )

        print(
            path.name,
            "= COMPILE PASS",
        )


def run_contract_tests():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                STAGE5
                / "tests"
            ),
            "-p",
            "test_block50_contracts.py",
        ],
        cwd=str(
            STAGE5
        ),
        text=True,
        capture_output=True,
    )

    output = (
        process.stdout
        +
        "\n"
        +
        process.stderr
    )

    print(
        output
    )

    require(
        process.returncode == 0,
        (
            "Block5.0 contract tests failed "
            "after semantic repair."
        ),
    )

    require(
        "Ran 12 tests"
        in
        output,
        (
            "Expected the same 12 Block5.0 "
            "tests after repair."
        ),
    )


def run_safe_controller():
    print()
    print(
        "============================================================"
    )

    print(
        "RERUNNING BLOCK 5.0 PART 2/2 SAFE CONTROLLER"
    )

    print(
        "============================================================"
    )

    process = subprocess.Popen(
        [
            sys.executable,
            str(
                SAFE
            ),
        ],
        cwd=str(
            STAGE5
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    if process.stdout is not None:

        for line in process.stdout:
            print(
                line,
                end="",
                flush=True,
            )

    raw_code = process.wait()

    print()
    print(
        "safe-controller raw code =",
        raw_code,
    )


def main():
    print(
        "============================================================"
    )

    print(
        "BLOCK 5.0 PART 2/2 — EFFECTIVE-RATE PROVENANCE REPAIR"
    )

    print(
        "============================================================"
    )

    patch_contracts()

    print(
        "contracts.py semantic repair = PASS"
    )

    patch_tests()

    print(
        "contract test semantic repair = PASS"
    )

    patch_runner()

    print(
        "contract-freeze runner repair = PASS"
    )

    remove_stale_partial_outputs()

    print()
    print(
        "===== CONTROLLED COMPILE ====="
    )

    compile_modified()

    print()
    print(
        "===== NARROW CONTRACT REGRESSION ====="
    )

    run_contract_tests()

    print(
        "Block5.0 tests = 12 / 12 PASS"
    )

    print()
    print(
        "semantic conclusion:"
    )

    print(
        "Part-A DPSK/BER/power      = FROZEN SOURCE"
    )

    print(
        "Part-A raw data rate       = FROZEN SOURCE"
    )

    print(
        "Stage5 effective rate      = REQUIRED DERIVED METRIC"
    )

    print(
        "effective-rate formula     = FREEZE IN BLOCK5.6 BEFORE FORMAL"
    )

    print(
        "scientific upstream change = NO"
    )

    print(
        "training/inference         = NO"
    )

    run_safe_controller()


try:
    main()

except BaseException as exc:

    print()
    print(
        "============================================================"
    )

    print(
        "BLOCK 5.0 EFFECTIVE-RATE REPAIR = BLOCKED"
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
        "Stage0-4 modified       = NO"
    )

    print(
        "training/inference      = NO"
    )

    print(
        "terminal remains open   = YES"
    )

# Deliberately no sys.exit().
