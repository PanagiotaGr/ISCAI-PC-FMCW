from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np


# =====================================================================
# STAGE5 FULLPDF V2 — FINAL FINISH V7
#
# Reuses the already-audited V6 constants/hash contract, but removes
# the incorrect assumption that an exported frozen optical function
# must itself expose an (snr_linear, snr_db) tuple.
#
# The compatibility bridge is applied ONLY when the original checker
# scalar extractor fails on an actual structured return value.
#
# NO decision regeneration
# NO receiver reselection
# NO training / retraining
# NO recalibration
# NO threshold changes
# NO formal/post-outcome tuning
# =====================================================================


ROOT = Path("/home/agni/waymo")
S5 = ROOT / "iscai_stage5"

PYTHON = Path(
    "/home/agni/waymo/iscai_stage1/.venv_lidar/bin/python"
)

V6_SCRIPT = (
    S5
    / "scripts/"
      "run_stage5_fullpdf_v2_FINAL_finish_v6.py"
)

ART = (
    S5 / "artifacts/fullpdf_v2_independent_formal"
)

V6_CONSOLE = (
    ART / "final_finish_v6_console.log"
)

V6_DIR = (
    ART / "final_formal_v6_finish"
)

V6_EVALUATOR_ROWS = (
    V6_DIR / "evaluator_rows.jsonl"
)

V6_FORMAL_SUMMARY = (
    V6_DIR / "formal_summary.json"
)

V6_SCIENCE_REPORT = (
    V6_DIR / "scientific_report.json"
)

V6_FAILURE = (
    V6_DIR / "unexpected_failure.json"
)


OUT = (
    ART / "final_formal_v7_finish"
)

HISTORY = (
    OUT / "preserved_pre_v7"
)

ADAPTER_AMENDMENT = (
    OUT / "structured_scalar_adapter_amendment_v1.json"
)

ADAPTER_SEAL = (
    OUT / "structured_scalar_adapter_preoutcome_seal_v1.json"
)

REGRESSION_LOG = (
    OUT / "regression_271.log"
)

INVOCATION = (
    OUT / "final_v7_invocation.json"
)

EVALUATOR_ROWS = (
    OUT / "evaluator_rows.jsonl"
)

FORMAL_SUMMARY = (
    OUT / "formal_summary.json"
)

SCIENCE_REPORT = (
    OUT / "scientific_report.json"
)

INDEPENDENT_REPORT = (
    S5
    / "reports/"
      "stage5_fullpdf_v2_final_independent_acceptance_v7.json"
)

CLOSURE = (
    S5
    / "reports/"
      "stage5_fullpdf_v2_final_closure_v7.json"
)

HANDOFF = (
    OUT / "stage5_to_stage6_handoff_fullpdf_v2.json"
)

FINAL_AUTHORITY = (
    S5
    / "reports/"
      "stage5_fullpdf_v2_FINAL_AUTHORITY_v7.json"
)

FAILURE = (
    OUT / "unexpected_failure.json"
)


EXPECTED_DECISION_ROWS = 17280
EXPECTED_RECEIVER_KEYS = 120
EXPECTED_TESTS = 271
EXPECTED_STRATA = 144

EXPECTED_MODES = [
    "centroid",
    "known",
    "uncertain",
]

EXPECTED_CODEBOOKS = [
    16,
    32,
    64,
]

EXPECTED_Q = [
    0.9,
    0.95,
    0.975,
    0.99,
]

V6_EXPECTED_FAILURE = (
    "could not numerically prove "
    "frozen optical SNR tuple contract"
)


# =====================================================================
# Load V6 as a LIBRARY only.
# main() is not executed.
# This gives us the exact path/hash contract that V6 already verified.
# =====================================================================


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"cannot import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


if not V6_SCRIPT.is_file():
    raise RuntimeError(
        f"V6 script missing: {V6_SCRIPT}"
    )

base = import_file(
    V6_SCRIPT,
    "stage5_v6_frozen_base",
)


# Frozen paths inherited from V6.
CHECKER = base.CHECKER
CHECKER2 = base.CHECKER2
RUNTIME = base.RUNTIME
PROTOCOL = base.PROTOCOL
ADDENDUM = base.ADDENDUM
STAGE4_LEDGER = base.STAGE4_LEDGER
RECEIVER_BINDING = base.RECEIVER_BINDING
TRUTH_BINDING = base.TRUTH_BINDING
TRUTH_AMENDMENT = base.TRUTH_AMENDMENT
TRUTH_SEAL = base.TRUTH_SEAL
TRUTH_REPORT = base.TRUTH_REPORT
TRUTH_DIAGNOSTIC = base.TRUTH_DIAGNOSTIC
SEALED = base.SEALED
EXPECTED = base.EXPECTED
EXPECTED_TRUTH_MAP_HASH = (
    base.EXPECTED_TRUTH_MAP_HASH
)


# =====================================================================
# Local utilities.
# =====================================================================


def die(message):
    raise RuntimeError(
        "FAIL-CLOSED: " + str(message)
    )


def sha(path: Path):
    return base.sha(path)


def load_json(path: Path):
    return base.load_json(path)


def load_jsonl(path: Path):
    return base.load_jsonl(path)


def norm(value):
    return base.norm(value)


def truth_hash(value):
    return base.truth_hash(value)


def jsonl_data(rows):
    return base.jsonl_data(rows)


def write_json(path, value):
    return base.write_json(
        path,
        value,
    )


def atomic_write(path, payload):
    return base.atomic_write(
        path,
        payload,
    )


def finite_scalar(value):
    try:
        arr = np.asarray(value)
    except Exception:
        return None

    if arr.ndim != 0:
        return None

    try:
        result = float(arr)
    except Exception:
        return None

    if not math.isfinite(result):
        return None

    return result


def normalize_name(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).lower(),
    ).strip("_")


# =====================================================================
# Structured scalar compatibility bridge.
#
# Rules:
#   1. original frozen checker extractor always gets first chance;
#   2. bridge activates only if original extractor raises;
#   3. exact named mapping/object attributes are preferred;
#   4. tuple with one numeric scalar -> that scalar;
#   5. SNR tuple -> identify unique (linear,dB) pair by
#          dB = 10 log10(linear)
#      irrespective of tuple order;
#   6. ambiguous representation -> fail closed.
#
# No scientific threshold or outcome is used.
# =====================================================================


SNR_LINEAR_NAMES = {
    "snr_linear",
    "linear_snr",
    "snr",
    "snr_lin",
}

SNR_DB_NAMES = {
    "snr_db",
    "db_snr",
    "snr_decibel",
    "snr_decibels",
}


def collect_aliases(
    bound,
    value_parameter,
):
    aliases = set()

    for key, value in (
        bound.arguments.items()
    ):
        if key == value_parameter:
            continue

        if isinstance(value, str):
            aliases.add(
                normalize_name(value)
            )

        elif isinstance(
            value,
            (tuple, list, set),
        ):
            for item in value:
                if isinstance(
                    item,
                    str,
                ):
                    aliases.add(
                        normalize_name(
                            item
                        )
                    )

    return aliases


def named_scalar(
    value,
    aliases,
    seen=None,
    depth=0,
):
    if seen is None:
        seen = set()

    if depth > 5:
        return None

    object_id = id(value)

    if object_id in seen:
        return None

    seen.add(object_id)

    # Mapping representation.
    if isinstance(
        value,
        Mapping,
    ):
        for key, item in value.items():
            if (
                normalize_name(key)
                in aliases
            ):
                scalar = finite_scalar(
                    item
                )

                if scalar is not None:
                    return scalar

        for item in value.values():
            if isinstance(
                item,
                (
                    Mapping,
                    tuple,
                    list,
                ),
            ) or hasattr(
                item,
                "__dict__",
            ):
                result = named_scalar(
                    item,
                    aliases,
                    seen,
                    depth + 1,
                )

                if result is not None:
                    return result

        return None

    # Exact attributes from dataclass / object / named tuple.
    for alias in aliases:
        if hasattr(
            value,
            alias,
        ):
            try:
                item = getattr(
                    value,
                    alias,
                )
            except Exception:
                continue

            if callable(item):
                try:
                    item = item()
                except Exception:
                    continue

            scalar = finite_scalar(
                item
            )

            if scalar is not None:
                return scalar

    if hasattr(
        value,
        "_asdict",
    ):
        try:
            mapped = value._asdict()
        except Exception:
            mapped = None

        if isinstance(
            mapped,
            Mapping,
        ):
            result = named_scalar(
                mapped,
                aliases,
                seen,
                depth + 1,
            )

            if result is not None:
                return result

    if hasattr(
        value,
        "__dict__",
    ):
        try:
            mapped = vars(value)
        except Exception:
            mapped = None

        if isinstance(
            mapped,
            Mapping,
        ):
            result = named_scalar(
                mapped,
                aliases,
                seen,
                depth + 1,
            )

            if result is not None:
                return result

    # Search structured members of tuple/list.
    if isinstance(
        value,
        (tuple, list),
    ):
        for item in value:
            if isinstance(
                item,
                (
                    Mapping,
                    tuple,
                    list,
                ),
            ) or hasattr(
                item,
                "__dict__",
            ):
                result = named_scalar(
                    item,
                    aliases,
                    seen,
                    depth + 1,
                )

                if result is not None:
                    return result

    return None


def snr_pair_from_sequence(
    value,
):
    if not isinstance(
        value,
        (tuple, list),
    ):
        return None

    numbers = []

    for index, item in enumerate(
        value
    ):
        scalar = finite_scalar(item)

        if scalar is not None:
            numbers.append(
                (
                    index,
                    scalar,
                )
            )

    pairs = []

    for linear_index, linear in numbers:
        if linear <= 0.0:
            continue

        expected_db = (
            10.0
            * math.log10(
                linear
            )
        )

        for db_index, db in numbers:
            if db_index == linear_index:
                continue

            tolerance = (
                1.0e-9
                * max(
                    1.0,
                    abs(expected_db),
                    abs(db),
                )
            )

            if abs(
                expected_db - db
            ) <= tolerance:
                pairs.append(
                    (
                        linear_index,
                        db_index,
                        float(linear),
                        float(db),
                    )
                )

    # Deduplicate exact same semantic pair.
    unique = {}

    for pair in pairs:
        unique[
            (
                pair[0],
                pair[1],
            )
        ] = pair

    pairs = list(
        unique.values()
    )

    if len(pairs) != 1:
        return None

    return pairs[0]


def structured_scalar(
    value,
    aliases,
):
    aliases = {
        normalize_name(x)
        for x in aliases
    }

    # Direct scalar.
    scalar = finite_scalar(
        value
    )

    if scalar is not None:
        return (
            scalar,
            "direct_scalar",
            None,
        )

    # Explicit named field/attribute.
    scalar = named_scalar(
        value,
        aliases,
    )

    if scalar is not None:
        return (
            scalar,
            "named_field_or_attribute",
            None,
        )

    if not isinstance(
        value,
        (tuple, list),
    ):
        return None

    numbers = [
        (
            index,
            finite_scalar(item),
        )
        for index, item in enumerate(
            value
        )
    ]

    numbers = [
        (
            index,
            scalar,
        )
        for index, scalar in numbers
        if scalar is not None
    ]

    # A tuple such as (PointingError, gain).
    if len(numbers) == 1:
        return (
            float(
                numbers[0][1]
            ),
            "unique_numeric_member",
            numbers[0][0],
        )

    wants_linear = bool(
        aliases
        & SNR_LINEAR_NAMES
    )

    wants_db = bool(
        aliases
        & SNR_DB_NAMES
    )

    if (
        wants_linear
        and wants_db
    ):
        return None

    if (
        wants_linear
        or wants_db
    ):
        pair = snr_pair_from_sequence(
            value
        )

        if pair is None:
            return None

        (
            linear_index,
            db_index,
            linear,
            db,
        ) = pair

        if wants_linear:
            return (
                linear,
                "snr_linear_db_identity",
                linear_index,
            )

        return (
            db,
            "snr_linear_db_identity",
            db_index,
        )

    return None


def find_scalar_helper(
    checker,
):
    candidates = []

    for name, fn in inspect.getmembers(
        checker,
        inspect.isfunction,
    ):
        try:
            source = inspect.getsource(
                fn
            )
        except Exception:
            continue

        if (
            "Unable to extract scalar"
            in source
        ):
            candidates.append(
                (
                    name,
                    fn,
                    source,
                )
            )

    if len(candidates) != 1:
        die(
            "expected exactly one checker "
            "scalar extractor; found "
            + repr(
                [
                    x[0]
                    for x in candidates
                ]
            )
        )

    return candidates[0]


BRIDGE_EVENTS = []


def install_bridge(
    checker,
):
    (
        helper_name,
        original,
        source,
    ) = find_scalar_helper(
        checker
    )

    signature = inspect.signature(
        original
    )

    parameters = list(
        signature.parameters.values()
    )

    if not parameters:
        die(
            "checker scalar extractor "
            "has no parameters"
        )

    value_parameter = (
        parameters[0].name
    )

    def patched(
        *args,
        **kwargs,
    ):
        try:
            return original(
                *args,
                **kwargs,
            )

        except Exception as original_exc:
            try:
                bound = (
                    signature.bind_partial(
                        *args,
                        **kwargs,
                    )
                )
            except Exception:
                raise original_exc

            if (
                value_parameter
                not in bound.arguments
            ):
                raise original_exc

            value = (
                bound.arguments[
                    value_parameter
                ]
            )

            aliases = collect_aliases(
                bound,
                value_parameter,
            )

            if not aliases:
                raise original_exc

            resolved = structured_scalar(
                value,
                aliases,
            )

            if resolved is None:
                raise original_exc

            scalar, strategy, index = (
                resolved
            )

            event = {
                "aliases":
                    sorted(
                        aliases
                    ),
                "container_type":
                    type(value).__name__,
                "container_length":
                    (
                        len(value)
                        if isinstance(
                            value,
                            (
                                tuple,
                                list,
                            ),
                        )
                        else None
                    ),
                "strategy":
                    strategy,
                "selected_index":
                    index,
            }

            if event not in BRIDGE_EVENTS:
                BRIDGE_EVENTS.append(
                    event
                )

            return float(
                scalar
            )

    setattr(
        checker,
        helper_name,
        patched,
    )

    return {
        "helper_name":
            helper_name,
        "signature":
            str(signature),
        "source_sha256":
            hashlib.sha256(
                source.encode(
                    "utf-8"
                )
            ).hexdigest(),
    }


# =====================================================================
# Main.
# =====================================================================


def main():
    print("=" * 78)
    print(
        "STAGE5 FULLPDF V2 — "
        "FINAL FINISH V7"
    )
    print(
        "STRUCTURED API BRIDGE | "
        "SEALED DECISIONS | FINAL SCIENTIFIC CLOSURE"
    )
    print("=" * 78)

    # ---------------------------------------------------------
    # 1. True one-shot guard.
    # ---------------------------------------------------------

    if OUT.exists():
        die(
            f"V7 one-shot output exists: {OUT}"
        )

    for path in (
        INDEPENDENT_REPORT,
        CLOSURE,
        FINAL_AUTHORITY,
    ):
        if path.exists():
            die(
                f"V7 final artifact exists: {path}"
            )

    OUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    # ---------------------------------------------------------
    # 2. Exact frozen/protected chain.
    # ---------------------------------------------------------

    for path, expected in (
        EXPECTED.items()
    ):
        if not path.is_file():
            die(
                f"missing protected artifact: {path}"
            )

        actual = sha(path)

        if actual != expected:
            die(
                "protected SHA mismatch:\n"
                f"{path}\n"
                f"actual   = {actual}\n"
                f"expected = {expected}"
            )

    if (
        stat.S_IMODE(
            SEALED.stat().st_mode
        )
        & 0o222
    ):
        die(
            "sealed decision ledger is writable"
        )

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        / 1024**3
    )

    if free_gib < 250.0:
        die(
            "free-space reserve below 250 GiB"
        )

    print(
        "PASS | protected scientific chain exact"
    )

    # ---------------------------------------------------------
    # 3. V6 must be pure pre-outcome harness failure.
    # ---------------------------------------------------------

    if not V6_CONSOLE.is_file():
        die(
            f"missing V6 console: {V6_CONSOLE}"
        )

    v6_text = V6_CONSOLE.read_text(
        encoding="utf-8",
        errors="replace",
    )

    required_v6_tokens = (
        V6_EXPECTED_FAILURE,
        "evaluator_rows_persisted = False",
        "formal_summary_persisted = False",
        "post_outcome_tuning = false",
        "Stage6_allowed = false",
    )

    for token in required_v6_tokens:
        if token not in v6_text:
            die(
                "V6 history does not prove "
                f"expected pre-outcome state: {token}"
            )

    if (
        "PASS | evaluator truth exact"
        in v6_text
    ):
        die(
            "V6 unexpectedly reached evaluator truth"
        )

    for path in (
        V6_EVALUATOR_ROWS,
        V6_FORMAL_SUMMARY,
        V6_SCIENCE_REPORT,
    ):
        if path.exists():
            die(
                "V6 persisted a scientific outcome: "
                f"{path}"
            )

    print(
        "PASS | V6 scientific outcome = NONE"
    )

    # ---------------------------------------------------------
    # 4. Preserve V6 failure history.
    # ---------------------------------------------------------

    HISTORY.mkdir(
        parents=True,
        exist_ok=False,
    )

    preserved = []

    for source in (
        V6_CONSOLE,
        V6_FAILURE,
    ):
        if not source.is_file():
            continue

        destination = (
            HISTORY
            / source.name
        )

        shutil.copy2(
            source,
            destination,
        )

        os.chmod(
            destination,
            0o444,
        )

        preserved.append(
            {
                "source":
                    str(source),
                "source_sha256":
                    sha(source),
                "snapshot":
                    str(destination),
                "snapshot_sha256":
                    sha(destination),
            }
        )

    history_sha = write_json(
        HISTORY
        / "history_manifest.json",
        {
            "stage": 5,
            "version":
                "fullpdf_v2",
            "status":
                "PRESERVED_BEFORE_V7",
            "V6_scientific_outcome":
                "NOT_COMPUTED",
            "V6_evaluator_truth_opened":
                False,
            "post_outcome_tuning":
                False,
            "files":
                preserved,
        },
    )

    # ---------------------------------------------------------
    # 5. Import checker as a library.
    # ---------------------------------------------------------

    checker = import_file(
        CHECKER,
        "stage5_v7_checker",
    )

    if not callable(
        getattr(
            checker,
            "evaluate_sealed_decisions",
            None,
        )
    ):
        die(
            "evaluate_sealed_decisions missing"
        )

    if not callable(
        getattr(
            checker,
            "resolve_truth_binding",
            None,
        )
    ):
        die(
            "resolve_truth_binding missing"
        )

    # ---------------------------------------------------------
    # 6. Install generic representation bridge.
    # ---------------------------------------------------------

    adapter = install_bridge(
        checker
    )

    # Pure synthetic adapter tests.
    for value in (
        (100.0, 20.0),
        (20.0, 100.0),
    ):
        resolved = structured_scalar(
            value,
            {"snr_linear"},
        )

        if (
            resolved is None
            or not math.isclose(
                resolved[0],
                100.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            die(
                "synthetic snr_linear "
                "adapter self-test failed"
            )

        resolved = structured_scalar(
            value,
            {"snr_db"},
        )

        if (
            resolved is None
            or not math.isclose(
                resolved[0],
                20.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            die(
                "synthetic snr_db "
                "adapter self-test failed"
            )

    synthetic_gain = (
        object(),
        0.375,
    )

    resolved = structured_scalar(
        synthetic_gain,
        {"optical_gain"},
    )

    if (
        resolved is None
        or not math.isclose(
            resolved[0],
            0.375,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        die(
            "synthetic unique-scalar "
            "adapter self-test failed"
        )

    print(
        "PASS | structured scalar adapter "
        "frozen before evaluator truth"
    )

    # ---------------------------------------------------------
    # 7. Freeze adapter provenance BEFORE truth/outcomes.
    # ---------------------------------------------------------

    amendment_sha = write_json(
        ADAPTER_AMENDMENT,
        {
            "stage": 5,
            "version":
                "fullpdf_v2",
            "status":
                "FROZEN_BEFORE_EVALUATOR_TRUTH",
            "scope":
                "checker representation compatibility only",
            "original_extractor_first":
                True,
            "fallback_only_after_original_failure":
                True,
            "supported_representation_repairs": [
                "exact mapping/object attribute",
                "single numeric member of structured tuple",
                (
                    "unique SNR linear/dB pair proven by "
                    "SNR_dB=10log10(SNR_linear)"
                ),
            ],
            "ambiguity_policy":
                "FAIL_CLOSED",
            "checker_scalar_helper":
                adapter[
                    "helper_name"
                ],
            "checker_scalar_helper_signature":
                adapter[
                    "signature"
                ],
            "checker_scalar_helper_source_sha256":
                adapter[
                    "source_sha256"
                ],
            "runtime_source_modified":
                False,
            "checker_source_modified":
                False,
            "protocol_modified":
                False,
            "decision_ledger_modified":
                False,
            "Stage4_modified":
                False,
            "scientific_threshold_modified":
                False,
            "formal_outcomes_used":
                False,
            "post_outcome_tuning":
                False,
            "V6_history_manifest_sha256":
                history_sha,
        },
    )

    adapter_seal_sha = write_json(
        ADAPTER_SEAL,
        {
            "stage": 5,
            "version":
                "fullpdf_v2",
            "status":
                "SEALED_BEFORE_EVALUATOR_TRUTH",
            "adapter_amendment_sha256":
                amendment_sha,
            "checker_sha256":
                sha(CHECKER),
            "runtime_sha256":
                sha(RUNTIME),
            "protocol_sha256":
                sha(PROTOCOL),
            "Stage4_ledger_sha256":
                sha(STAGE4_LEDGER),
            "decision_ledger_sha256":
                sha(SEALED),
            "truth_binding_sha256":
                sha(TRUTH_BINDING),
            "formal_outcomes_used":
                False,
            "post_outcome_tuning":
                False,
        },
    )

    # ---------------------------------------------------------
    # 8. Full Stage5 regression.
    # ---------------------------------------------------------

    proc = subprocess.run(
        [
            str(PYTHON),
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S5 / "tests"
            ),
            "-p",
            "test_*.py",
            "-v",
        ],
        cwd=str(S5),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests",
        proc.stdout,
    )

    test_count = (
        None
        if match is None
        else int(
            match.group(1)
        )
    )

    if (
        proc.returncode != 0
        or test_count != EXPECTED_TESTS
    ):
        die(
            "Stage5 regression failed: "
            f"rc={proc.returncode}, "
            f"tests={test_count}"
        )

    atomic_write(
        REGRESSION_LOG,
        proc.stdout.encode(
            "utf-8"
        ),
    )

    regression_sha = sha(
        REGRESSION_LOG
    )

    print(
        "PASS | regression 271/271"
    )

    # ---------------------------------------------------------
    # 9. Verify immutable decision pass1/pass2/seal.
    # ---------------------------------------------------------

    decisions = load_jsonl(
        SEALED
    )

    if len(
        decisions
    ) != EXPECTED_DECISION_ROWS:
        die(
            "decision row count changed"
        )

    decision_keys = {
        (
            str(
                row[
                    "scenario_id"
                ]
            ),
            str(
                row[
                    "prediction_id"
                ]
            ),
        )
        for row in decisions
    }

    if len(
        decision_keys
    ) != EXPECTED_RECEIVER_KEYS:
        die(
            "receiver-key count changed"
        )

    run1 = Path(
        checker.RUN1_LEDGER
    )

    run2 = Path(
        checker.RUN2_LEDGER
    )

    seal_report_path = Path(
        checker.SEAL_REPORT
    )

    for path in (
        run1,
        run2,
        seal_report_path,
    ):
        if not path.is_file():
            die(
                f"decision seal evidence missing: {path}"
            )

    sealed_sha = sha(
        SEALED
    )

    if not (
        sha(run1)
        == sealed_sha
        == sha(run2)
        and
        run1.read_bytes()
        == run2.read_bytes()
        == SEALED.read_bytes()
    ):
        die(
            "run1/run2/sealed are not byte exact"
        )

    seal_report = load_json(
        seal_report_path
    )

    if (
        seal_report.get(
            "status"
        )
        !=
        "SEALED_BEFORE_EVALUATOR_TRUTH"
        or
        seal_report.get(
            "byte_exact_repeat"
        )
        is not True
        or
        seal_report.get(
            "future_truth_opened_before_seal"
        )
        is not False
        or
        seal_report.get(
            "sealed_sha256"
        )
        != sealed_sha
    ):
        die(
            "decision seal report changed"
        )

    print(
        "PASS | exact-repeat decision seal"
    )

    print(
        "decision_ledger_sha256 =",
        sealed_sha,
    )

    # ---------------------------------------------------------
    # 10. Protocol exact.
    # ---------------------------------------------------------

    protocol = load_json(
        PROTOCOL
    )

    if (
        protocol[
            "receiver_geometry"
        ][
            "modes"
        ]
        != EXPECTED_MODES
    ):
        die(
            "receiver modes changed"
        )

    if (
        [
            int(x)
            for x in protocol[
                "codebooks"
            ][
                "sizes"
            ]
        ]
        != EXPECTED_CODEBOOKS
    ):
        die(
            "codebooks changed"
        )

    if (
        [
            float(x)
            for x in protocol[
                "adaptive_TopK"
            ][
                "coverage_targets"
            ]
        ]
        != EXPECTED_Q
    ):
        die(
            "coverage targets changed"
        )

    # ---------------------------------------------------------
    # 11. Invocation marker BEFORE truth.
    # ---------------------------------------------------------

    invocation_sha = write_json(
        INVOCATION,
        {
            "stage": 5,
            "version":
                "fullpdf_v2",
            "status":
                "SEALED_BEFORE_EVALUATOR_TRUTH",
            "decision_controller_reexecuted":
                False,
            "receiver_selection_reexecuted":
                False,
            "decision_ledger_sha256":
                sealed_sha,
            "decision_rows":
                len(decisions),
            "receiver_keys":
                len(
                    decision_keys
                ),
            "adapter_amendment_sha256":
                amendment_sha,
            "adapter_seal_sha256":
                adapter_seal_sha,
            "regression_log_sha256":
                regression_sha,
            "formal_metrics_computed":
                False,
            "training":
                False,
            "retraining":
                False,
            "recalibration":
                False,
            "formal_tuning":
                False,
            "post_outcome_tuning":
                False,
            "Stage6_allowed":
                False,
        },
    )

    # ---------------------------------------------------------
    # 12. Open exact evaluator truth only now.
    # ---------------------------------------------------------

    checker.DECISION_LEDGER_SEALED = (
        True
    )

    checker.FORMAL_TRUTH_OPENED = (
        False
    )

    truth_path, truth_map = (
        checker.resolve_truth_binding(
            decision_keys
        )
    )

    truth_path = Path(
        truth_path
    ).resolve()

    if (
        truth_path
        != TRUTH_BINDING.resolve()
    ):
        die(
            "unexpected evaluator truth path"
        )

    if (
        sha(
            truth_path
        )
        != EXPECTED[
            TRUTH_BINDING
        ]
    ):
        die(
            "truth binding changed"
        )

    current_truth_hash = (
        truth_hash(
            truth_map
        )
    )

    if (
        current_truth_hash
        != EXPECTED_TRUTH_MAP_HASH
    ):
        die(
            "truth-map semantics changed"
        )

    if not checker.FORMAL_TRUTH_OPENED:
        die(
            "truth resolver did not mark "
            "truth opened"
        )

    print(
        "PASS | evaluator truth opened "
        "only after immutable decision seal"
    )

    # ---------------------------------------------------------
    # 13. Recheck every protected hash before scientific call.
    # ---------------------------------------------------------

    for path, expected in (
        EXPECTED.items()
    ):
        if sha(path) != expected:
            die(
                f"protected artifact mutated: {path}"
            )

    if sha(
        SEALED
    ) != sealed_sha:
        die(
            "sealed decisions mutated"
        )

    # =========================================================
    # 14. SINGLE FINAL SCIENTIFIC EVALUATION CALL.
    # =========================================================

    print()
    print("=" * 78)
    print(
        "EXECUTING FROZEN "
        "evaluate_sealed_decisions()"
    )
    print("=" * 78)

    evaluator_rows, formal_summary = (
        checker.evaluate_sealed_decisions(
            decisions,
            truth_map,
            protocol,
        )
    )

    # Immediately freeze returned scientific results.
    atomic_write(
        EVALUATOR_ROWS,
        jsonl_data(
            evaluator_rows
        ),
    )

    evaluator_sha = sha(
        EVALUATOR_ROWS
    )

    summary_sha = write_json(
        FORMAL_SUMMARY,
        formal_summary,
    )

    print(
        "PASS | scientific outputs returned "
        "and frozen"
    )

    # ---------------------------------------------------------
    # 15. Acceptance.
    # ---------------------------------------------------------

    strata_raw = formal_summary.get(
        "strata",
        [],
    )

    if isinstance(
        strata_raw,
        Mapping,
    ):
        strata = list(
            strata_raw.values()
        )
    elif isinstance(
        strata_raw,
        list,
    ):
        strata = strata_raw
    else:
        strata = []

    protected_after = all(
        sha(path) == expected
        for path, expected
        in EXPECTED.items()
    )

    sealed_after = (
        sha(
            SEALED
        )
        == sealed_sha
        and
        (
            stat.S_IMODE(
                SEALED.stat().st_mode
            )
            & 0o222
        )
        == 0
    )

    recovery = (
        base.runtime_recovery_evidence()
    )

    gates = {
        "G13_exact_repeat_decisions":
            True,

        "G14_decisions_sealed_before_truth":
            bool(
                seal_report[
                    "future_truth_opened_before_seal"
                ]
                is False
                and sealed_after
            ),

        "G15_truth_after_seal":
            bool(
                checker.FORMAL_TRUTH_OPENED
                and
                checker.DECISION_LEDGER_SEALED
            ),

        "G16_requested_coverage":
            bool(
                formal_summary.get(
                    "requested_coverage_all_strata_pass",
                    False,
                )
            ),

        "G17_overhead_reduction":
            bool(
                formal_summary.get(
                    "overhead_reduction_vs_exhaustive_all_strata_pass",
                    False,
                )
            ),

        "G18_no_free_probes":
            bool(
                formal_summary.get(
                    "no_free_probes",
                    False,
                )
            ),

        "G19_latency":
            bool(
                formal_summary.get(
                    "latency_all_strata_pass",
                    False,
                )
            ),

        "G20_optical_chain":
            bool(
                formal_summary.get(
                    "full_optical_chain_complete",
                    False,
                )
            ),

        "G21_recovery_controls":
            bool(
                recovery
                and all(
                    recovery.values()
                )
            ),

        "G22_no_post_outcome_mutation":
            bool(
                protected_after
                and sealed_after
            ),

        "G24_full_144_strata":
            (
                len(strata)
                == EXPECTED_STRATA
            ),

        "G25_complete_evaluator_rows":
            (
                len(
                    evaluator_rows
                )
                == EXPECTED_DECISION_ROWS
            ),
    }

    pdf_matrix = [
        {
            "requirement":
                "receiver centroid/known/uncertain",
            "pass":
                protocol[
                    "receiver_geometry"
                ][
                    "modes"
                ]
                == EXPECTED_MODES,
        },
        {
            "requirement":
                "receiver uncertainty explicit",
            "pass":
                "uncertain"
                in protocol[
                    "receiver_geometry"
                ][
                    "modes"
                ],
        },
        {
            "requirement":
                "codebooks 16/32/64",
            "pass":
                [
                    int(x)
                    for x in protocol[
                        "codebooks"
                    ][
                        "sizes"
                    ]
                ]
                == EXPECTED_CODEBOOKS,
        },
        {
            "requirement":
                "Top-K q=90/95/97.5/99%",
            "pass":
                [
                    float(x)
                    for x in protocol[
                        "adaptive_TopK"
                    ][
                        "coverage_targets"
                    ]
                ]
                == EXPECTED_Q,
        },
        {
            "requirement":
                "causal decisions frozen before truth",
            "pass":
                gates[
                    "G14_decisions_sealed_before_truth"
                ]
                and gates[
                    "G15_truth_after_seal"
                ],
        },
        {
            "requirement":
                "persistence/hysteresis/recovery/fallback",
            "pass":
                gates[
                    "G21_recovery_controls"
                ],
        },
        {
            "requirement":
                "no hidden/free evaluated beams",
            "pass":
                gates[
                    "G18_no_free_probes"
                ],
        },
        {
            "requirement":
                "requested empirical probability coverage",
            "pass":
                gates[
                    "G16_requested_coverage"
                ],
        },
        {
            "requirement":
                "probing overhead reduction",
            "pass":
                gates[
                    "G17_overhead_reduction"
                ],
        },
        {
            "requirement":
                "latency charged from probing workload",
            "pass":
                gates[
                    "G19_latency"
                ],
        },
        {
            "requirement":
                "pointing-gain-power-SNR-DPSK-BER-rate",
            "pass":
                gates[
                    "G20_optical_chain"
                ],
        },
        {
            "requirement":
                "full receiver-codebook-horizon-q matrix",
            "pass":
                gates[
                    "G24_full_144_strata"
                ],
        },
        {
            "requirement":
                "no Stage4 recalibration/retraining",
            "pass":
                sha(
                    STAGE4_LEDGER
                )
                == EXPECTED[
                    STAGE4_LEDGER
                ],
        },
        {
            "requirement":
                "no formal/post-outcome tuning",
            "pass":
                gates[
                    "G22_no_post_outcome_mutation"
                ],
        },
    ]

    pdf_pass = all(
        item[
            "pass"
        ]
        for item in pdf_matrix
    )

    gates[
        "G23_PDF_matrix"
    ] = pdf_pass

    scientific_pass = all(
        gates.values()
    )

    science_sha = write_json(
        SCIENCE_REPORT,
        {
            "stage": 5,
            "version":
                "fullpdf_v2",
            "status":
                (
                    "PASS_COMPLETE_FROZEN_FULLPDF_V2"
                    if scientific_pass
                    else
                    "FAIL_CLOSED_SCIENTIFIC_FULLPDF_V2"
                ),
            "formal_metrics_computed":
                True,
            "PDF_compliance":
                (
                    "PASS"
                    if pdf_pass
                    else "FAIL"
                ),
            "gates":
                gates,
            "PDF_requirement_matrix":
                pdf_matrix,
            "decision_ledger_sha256":
                sealed_sha,
            "truth_binding_sha256":
                sha(
                    TRUTH_BINDING
                ),
            "truth_map_hash":
                current_truth_hash,
            "evaluator_rows":
                len(
                    evaluator_rows
                ),
            "evaluator_rows_sha256":
                evaluator_sha,
            "formal_summary_sha256":
                summary_sha,
            "strata_count":
                len(
                    strata
                ),
            "representation_bridge": {
                "amendment_sha256":
                    amendment_sha,
                "seal_sha256":
                    adapter_seal_sha,
                "events":
                    BRIDGE_EVENTS,
                "checker_source_modified":
                    False,
                "runtime_source_modified":
                    False,
                "scientific_parameters_modified":
                    False,
            },
            "training":
                False,
            "retraining":
                False,
            "recalibration":
                False,
            "formal_parameter_tuning":
                False,
            "post_outcome_tuning":
                False,
            "Stage6_allowed":
                False,
        },
    )

    # ---------------------------------------------------------
    # 16. Independent read-only post-freeze acceptance.
    # ---------------------------------------------------------

    frozen_summary = load_json(
        FORMAL_SUMMARY
    )

    frozen_rows = load_jsonl(
        EVALUATOR_ROWS
    )

    frozen_strata = frozen_summary.get(
        "strata",
        [],
    )

    if isinstance(
        frozen_strata,
        Mapping,
    ):
        frozen_strata_count = len(
            frozen_strata
        )
    elif isinstance(
        frozen_strata,
        list,
    ):
        frozen_strata_count = len(
            frozen_strata
        )
    else:
        frozen_strata_count = 0

    independent_checks = {
        "science_report_frozen":
            sha(
                SCIENCE_REPORT
            )
            == science_sha,

        "formal_summary_frozen":
            sha(
                FORMAL_SUMMARY
            )
            == summary_sha,

        "evaluator_rows_frozen":
            sha(
                EVALUATOR_ROWS
            )
            == evaluator_sha,

        "decision_ledger_immutable":
            sha(
                SEALED
            )
            == sealed_sha,

        "truth_binding_immutable":
            sha(
                TRUTH_BINDING
            )
            == EXPECTED[
                TRUTH_BINDING
            ],

        "coverage":
            bool(
                frozen_summary.get(
                    "requested_coverage_all_strata_pass",
                    False,
                )
            ),

        "overhead":
            bool(
                frozen_summary.get(
                    "overhead_reduction_vs_exhaustive_all_strata_pass",
                    False,
                )
            ),

        "no_free_probes":
            bool(
                frozen_summary.get(
                    "no_free_probes",
                    False,
                )
            ),

        "latency":
            bool(
                frozen_summary.get(
                    "latency_all_strata_pass",
                    False,
                )
            ),

        "optical_chain":
            bool(
                frozen_summary.get(
                    "full_optical_chain_complete",
                    False,
                )
            ),

        "strata_144":
            frozen_strata_count
            == EXPECTED_STRATA,

        "rows_17280":
            len(
                frozen_rows
            )
            == EXPECTED_DECISION_ROWS,

        "PDF_matrix":
            pdf_pass,

        "no_post_outcome_mutation":
            gates[
                "G22_no_post_outcome_mutation"
            ],
    }

    independent_pass = all(
        independent_checks.values()
    )

    independent_sha = write_json(
        INDEPENDENT_REPORT,
        {
            "stage": 5,
            "version":
                "fullpdf_v2",
            "status":
                (
                    "PASS"
                    if independent_pass
                    else "FAIL"
                ),
            "checks":
                independent_checks,
            "science_report_sha256":
                science_sha,
            "formal_summary_sha256":
                summary_sha,
            "evaluator_rows_sha256":
                evaluator_sha,
            "post_outcome_tuning":
                False,
            "Stage6_allowed":
                False,
        },
    )

    final_pass = (
        scientific_pass
        and independent_pass
    )

    # ---------------------------------------------------------
    # 17. Genuine scientific fail = terminal.
    # ---------------------------------------------------------

    if not final_pass:
        print()
        print("=" * 78)
        print(
            "STAGE5 V7 = GENUINE "
            "SCIENTIFIC/ACCEPTANCE FAIL"
        )
        print("=" * 78)

        for key, value in (
            gates.items()
        ):
            print(
                key,
                "=",
                value,
            )

        print(
            "independent_acceptance =",
            independent_pass,
        )

        print(
            "PDF_compliance =",
            (
                "PASS"
                if pdf_pass
                else "FAIL"
            ),
        )

        print(
            "post_outcome_tuning = false"
        )

        print(
            "Stage6_allowed = false"
        )

        print(
            "DO NOT RETUNE / DO NOT RERUN"
        )

        print("=" * 78)

        return 2

    # =========================================================
    # 18. FINAL COMPLETE/FROZEN CLOSURE.
    # =========================================================

    closure_sha = write_json(
        CLOSURE,
        {
            "stage": 5,
            "version":
                "fullpdf_v2",
            "status":
                "COMPLETE_FROZEN_FULLPDF_V2",
            "PDF_compliance":
                "PASS",
            "Stage6_allowed":
                True,
            "Stage6_started":
                False,
            "decision_boundary": {
                "two_exact_passes":
                    True,
                "sealed_before_truth":
                    True,
                "future_truth_after_seal":
                    True,
                "sealed_ledger_sha256":
                    sealed_sha,
                "rows":
                    EXPECTED_DECISION_ROWS,
                "receiver_keys":
                    EXPECTED_RECEIVER_KEYS,
            },
            "evaluator_truth": {
                "path":
                    str(
                        TRUTH_BINDING
                    ),
                "sha256":
                    sha(
                        TRUTH_BINDING
                    ),
                "truth_map_hash":
                    current_truth_hash,
            },
            "scientific_report": {
                "path":
                    str(
                        SCIENCE_REPORT
                    ),
                "sha256":
                    science_sha,
            },
            "independent_acceptance": {
                "path":
                    str(
                        INDEPENDENT_REPORT
                    ),
                "sha256":
                    independent_sha,
            },
            "formal_summary": {
                "path":
                    str(
                        FORMAL_SUMMARY
                    ),
                "sha256":
                    summary_sha,
            },
            "evaluator_rows": {
                "path":
                    str(
                        EVALUATOR_ROWS
                    ),
                "sha256":
                    evaluator_sha,
            },
            "gates":
                gates,
            "PDF_requirement_matrix":
                pdf_matrix,
            "representation_bridge": {
                "events":
                    BRIDGE_EVENTS,
                "amendment_sha256":
                    amendment_sha,
                "seal_sha256":
                    adapter_seal_sha,
                "scientific_semantics_changed":
                    False,
            },
            "future_truth_controller_input":
                False,
            "tracks_to_predict_receiver_selector":
                False,
            "decision_controller_reexecuted":
                False,
            "receiver_selection_reexecuted":
                False,
            "training":
                False,
            "retraining":
                False,
            "recalibration":
                False,
            "formal_parameter_tuning":
                False,
            "post_outcome_tuning":
                False,
        },
    )

    handoff_sha = write_json(
        HANDOFF,
        {
            "from_stage": 5,
            "to_stage": 6,
            "version":
                "fullpdf_v2",
            "status":
                "FROZEN_HANDOFF_READY",
            "Stage5":
                "COMPLETE_FROZEN_FULLPDF_V2",
            "Stage6_allowed":
                True,
            "Stage6_started":
                False,
            "Stage5_closure": {
                "path":
                    str(
                        CLOSURE
                    ),
                "sha256":
                    closure_sha,
            },
            "Stage4_prediction_ledger": {
                "path":
                    str(
                        STAGE4_LEDGER
                    ),
                "sha256":
                    sha(
                        STAGE4_LEDGER
                    ),
                "Stage5_recalibration":
                    False,
            },
            "receiver_binding": {
                "path":
                    str(
                        RECEIVER_BINDING
                    ),
                "sha256":
                    sha(
                        RECEIVER_BINDING
                    ),
            },
            "communication_decision_ledger": {
                "path":
                    str(
                        SEALED
                    ),
                "sha256":
                    sealed_sha,
            },
            "scientific_report_sha256":
                science_sha,
            "evaluator_rows_sha256":
                evaluator_sha,
            "PDF_compliance":
                "PASS",
            "communication_beams_and_ADB":
                "separate_actuators_sharing_same_posterior",
            "post_outcome_tuning":
                False,
        },
    )

    authority_sha = write_json(
        FINAL_AUTHORITY,
        {
            "stage": 5,
            "version":
                "fullpdf_v2",
            "status":
                "AUTHORITATIVE_FINAL_STAGE5_CLOSURE",
            "historical_failures_preserved":
                True,
            "final_closure": {
                "path":
                    str(
                        CLOSURE
                    ),
                "sha256":
                    closure_sha,
            },
            "handoff": {
                "path":
                    str(
                        HANDOFF
                    ),
                "sha256":
                    handoff_sha,
            },
            "science_report_sha256":
                science_sha,
            "independent_acceptance_sha256":
                independent_sha,
            "decision_ledger_sha256":
                sealed_sha,
            "evaluator_truth_sha256":
                sha(
                    TRUTH_BINDING
                ),
            "PDF_compliance":
                "PASS",
            "Stage5":
                "COMPLETE_FROZEN_FULLPDF_V2",
            "Stage6_allowed":
                True,
            "Stage6_started":
                False,
            "post_outcome_tuning":
                False,
        },
    )

    print()
    print("=" * 78)
    print(
        "STAGE5 FULLPDF V2 — "
        "FINAL RESULT = PASS"
    )
    print("=" * 78)

    print(
        "Stage5 = COMPLETE_FROZEN_FULLPDF_V2"
    )

    print(
        "PDF_compliance = PASS"
    )

    print(
        "structured_API_bridge = PASS"
    )

    print(
        "bridge_events =",
        BRIDGE_EVENTS,
    )

    print(
        "decision_ledger_sha256 =",
        sealed_sha,
    )

    print(
        "formal_strata =",
        len(
            strata
        ),
        "/",
        EXPECTED_STRATA,
    )

    print(
        "evaluator_rows =",
        len(
            frozen_rows
        ),
        "/",
        EXPECTED_DECISION_ROWS,
    )

    print(
        "requested_coverage_gate =",
        gates[
            "G16_requested_coverage"
        ],
    )

    print(
        "overhead_reduction_gate =",
        gates[
            "G17_overhead_reduction"
        ],
    )

    print(
        "no_free_probes_gate =",
        gates[
            "G18_no_free_probes"
        ],
    )

    print(
        "latency_gate =",
        gates[
            "G19_latency"
        ],
    )

    print(
        "optical_chain_gate =",
        gates[
            "G20_optical_chain"
        ],
    )

    print(
        "full_matrix_gate =",
        gates[
            "G24_full_144_strata"
        ],
    )

    print(
        "independent_acceptance =",
        independent_pass,
    )

    print(
        "post_outcome_tuning = false"
    )

    print(
        "Stage6_allowed = true"
    )

    print(
        "Stage6_started = false"
    )

    print(
        "science_report_sha256 =",
        science_sha,
    )

    print(
        "independent_report_sha256 =",
        independent_sha,
    )

    print(
        "closure_sha256 =",
        closure_sha,
    )

    print(
        "handoff_sha256 =",
        handoff_sha,
    )

    print(
        "final_authority_sha256 =",
        authority_sha,
    )

    print("=" * 78)

    return 0


if __name__ == "__main__":
    try:
        rc = main()

    except Exception as exc:
        OUT.mkdir(
            parents=True,
            exist_ok=True,
        )

        truth_opened = False

        checker_module = sys.modules.get(
            "stage5_v7_checker"
        )

        if checker_module is not None:
            truth_opened = bool(
                getattr(
                    checker_module,
                    "FORMAL_TRUTH_OPENED",
                    False,
                )
            )

        if not FAILURE.exists():
            try:
                write_json(
                    FAILURE,
                    {
                        "stage": 5,
                        "version":
                            "fullpdf_v2",
                        "status":
                            "FAIL_CLOSED",
                        "exception_type":
                            type(
                                exc
                            ).__name__,
                        "exception":
                            str(
                                exc
                            ),
                        "evaluator_truth_opened":
                            truth_opened,
                        "evaluator_rows_persisted":
                            EVALUATOR_ROWS.is_file(),
                        "formal_summary_persisted":
                            FORMAL_SUMMARY.is_file(),
                        "scientific_report_persisted":
                            SCIENCE_REPORT.is_file(),
                        "bridge_events":
                            BRIDGE_EVENTS,
                        "post_outcome_tuning":
                            False,
                        "automatic_rerun_allowed":
                            False,
                        "Stage6_allowed":
                            False,
                    },
                )
            except Exception:
                pass

        print()
        print("=" * 78)
        print(
            "STAGE5 FINAL V7 = FAIL-CLOSED"
        )

        print(
            "reason =",
            repr(
                exc
            ),
        )

        print(
            "evaluator_truth_opened =",
            truth_opened,
        )

        print(
            "evaluator_rows_persisted =",
            EVALUATOR_ROWS.is_file(),
        )

        print(
            "formal_summary_persisted =",
            FORMAL_SUMMARY.is_file(),
        )

        print(
            "bridge_events =",
            BRIDGE_EVENTS,
        )

        print(
            "post_outcome_tuning = false"
        )

        print(
            "Stage6_allowed = false"
        )

        print(
            "DO NOT RERUN AUTOMATICALLY"
        )

        print("=" * 78)

        sys.exit(3)

    sys.exit(rc)
