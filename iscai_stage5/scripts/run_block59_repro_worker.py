from __future__ import annotations

import argparse
import ast
import builtins
import dataclasses
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path("/home/agni/waymo")
S5 = ROOT / "iscai_stage5"

FORMAL_REPORT = (
    S5
    / "reports/block58_formal_evaluation.json"
)

FROZEN_CACHE = (
    S5
    / "artifacts/block58/formal_clean_cache"
)

TARGET_RANKS = (
    1,
    60,
    120,
)

TRACE_FUNCTIONS = {
    "deterministic_receiver_angular_posterior":
        "receiver_posterior",

    "beam_probability_mass_from_samples":
        "beam_probability",

    "complete_partition_beam_probability_mass_from_samples":
        "beam_probability",

    "adaptive_temporal_step":
        "topk_selection",

    "physical_adaptive_indices":
        "topk_selection",

    "adaptive_probe_indices":
        "topk_selection",

    "evaluate_optical_link":
        "optical_metrics",
}


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def _strict_json_value(
    value: Any,
) -> Any:
    """
    Convert an arbitrary deterministic result structure into
    strict RFC-compatible JSON material without losing the
    distinction between finite values, NaN, +inf and -inf.

    This is applied identically to frozen and freshly reconstructed
    Stage-5 semantic results before hashing.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if math.isnan(value):
            return {
                "__nonfinite_float__":
                    "nan",
            }

        if value == math.inf:
            return {
                "__nonfinite_float__":
                    "+inf",
            }

        if value == -math.inf:
            return {
                "__nonfinite_float__":
                    "-inf",
            }

        return value

    if isinstance(value, str):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key):
                _strict_json_value(child)
            for key, child
            in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _strict_json_value(child)
            for child in value
        ]

    if isinstance(value, set):
        normalized = [
            _strict_json_value(child)
            for child in value
        ]

        return sorted(
            normalized,
            key=lambda child:
                json.dumps(
                    child,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
        )

    # NumPy scalars/arrays may exist in a fresh in-memory route
    # even though the frozen JSON representation contains only
    # ordinary Python values.
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _strict_json_value(
                value.item()
            )

        if isinstance(value, np.ndarray):
            return _strict_json_value(
                value.tolist()
            )

    except ImportError:
        pass

    # Defensive Torch support. This does not alter model execution;
    # it only normalizes a value if a tensor reaches the hash boundary.
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return _strict_json_value(
                value
                .detach()
                .cpu()
                .tolist()
            )

    except ImportError:
        pass

    if dataclasses.is_dataclass(value):
        return _strict_json_value(
            dataclasses.asdict(value)
        )

    if hasattr(value, "_asdict"):
        try:
            return _strict_json_value(
                value._asdict()
            )
        except Exception:
            pass

    raise TypeError(
        "Unsupported value at strict JSON hash boundary: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def canonical_bytes(
    value: Any,
) -> bytes:
    strict_value = _strict_json_value(
        value
    )

    return (
        json.dumps(
            strict_value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha(
    value: Any,
) -> str:
    return hashlib.sha256(
        canonical_bytes(
            value
        )
    ).hexdigest()


def atomic_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name
        + f".tmp.{os.getpid()}"
    )

    data = canonical_bytes(
        payload
    )

    try:
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        os.replace(
            tmp,
            path,
        )

    finally:
        if tmp.exists():
            tmp.unlink()


def load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def finite_float_value(
    value: float,
) -> Any:
    x = float(value)

    if math.isfinite(x):
        return x

    return {
        "__nonfinite_float__":
            repr(x),
    }


def trace_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        return finite_float_value(
            value
        )

    if isinstance(
        value,
        str,
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return {
            "__path__":
                str(value),
        }

    if isinstance(
        value,
        bytes,
    ):
        return {
            "__bytes__": {
                "length":
                    len(value),

                "sha256":
                    hashlib.sha256(
                        value
                    ).hexdigest(),
            }
        }

    # numpy support without requiring it at module import time
    try:
        import numpy as np

        if isinstance(
            value,
            np.generic,
        ):
            return trace_value(
                value.item()
            )

        if isinstance(
            value,
            np.ndarray,
        ):
            array = np.ascontiguousarray(
                value
            )

            raw = array.tobytes(
                order="C"
            )

            payload = {
                "__ndarray__": {
                    "dtype":
                        str(
                            array.dtype
                        ),

                    "shape":
                        list(
                            array.shape
                        ),

                    "sha256":
                        hashlib.sha256(
                            raw
                        ).hexdigest(),
                }
            }

            if array.size <= 64:
                payload[
                    "__ndarray__"
                ][
                    "values"
                ] = trace_value(
                    array.tolist()
                )

            return payload

    except ImportError:
        pass

    # torch support
    try:
        import torch

        if isinstance(
            value,
            torch.Tensor,
        ):
            tensor = (
                value
                .detach()
                .cpu()
                .contiguous()
            )

            array = tensor.numpy()

            raw = array.tobytes(
                order="C"
            )

            payload = {
                "__tensor__": {
                    "dtype":
                        str(
                            tensor.dtype
                        ),

                    "shape":
                        list(
                            tensor.shape
                        ),

                    "sha256":
                        hashlib.sha256(
                            raw
                        ).hexdigest(),
                }
            }

            if tensor.numel() <= 64:
                payload[
                    "__tensor__"
                ][
                    "values"
                ] = trace_value(
                    tensor.tolist()
                )

            return payload

    except ImportError:
        pass

    if dataclasses.is_dataclass(
        value
    ):
        return {
            "__dataclass__":
                (
                    value.__class__.__module__
                    + "."
                    + value.__class__.__qualname__
                ),

            "fields": {
                field.name:
                    trace_value(
                        getattr(
                            value,
                            field.name,
                        )
                    )
                for field
                in dataclasses.fields(
                    value
                )
            },
        }

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key):
                trace_value(
                    child
                )
            for key, child
            in sorted(
                value.items(),
                key=lambda item:
                    str(
                        item[0]
                    ),
            )
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            trace_value(
                child
            )
            for child
            in value
        ]

    if isinstance(
        value,
        set,
    ):
        converted = [
            trace_value(
                child
            )
            for child
            in value
        ]

        return sorted(
            converted,
            key=lambda child:
                json.dumps(
                    child,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
        )

    if hasattr(
        value,
        "_asdict",
    ):
        try:
            return trace_value(
                value._asdict()
            )
        except Exception:
            pass

    if hasattr(
        value,
        "__dict__",
    ):
        try:
            state = {
                str(key):
                    child
                for key, child
                in vars(
                    value
                ).items()
                if not str(
                    key
                ).startswith("_")
            }

            return {
                "__object__":
                    (
                        value.__class__.__module__
                        + "."
                        + value.__class__.__qualname__
                    ),

                "state":
                    trace_value(
                        state
                    ),
            }

        except Exception:
            pass

    # Do NOT serialize repr(), since repr may contain memory addresses.
    return {
        "__opaque_type__":
            (
                value.__class__.__module__
                + "."
                + value.__class__.__qualname__
            )
    }


def strip_nondeterministic_runtime(
    value: Any,
) -> Any:
    if isinstance(
        value,
        dict,
    ):
        result = {}

        for key, child in value.items():
            name = str(
                key
            )

            low = name.lower()

            if (
                low
                ==
                "gpu_peak_memory_bytes"
            ):
                continue

            if low.endswith(
                "_runtime_s"
            ):
                continue

            if low.endswith(
                "_elapsed_s"
            ):
                continue

            if (
                "wall_clock"
                in low
            ):
                continue

            result[
                name
            ] = (
                strip_nondeterministic_runtime(
                    child
                )
            )

        return result

    if isinstance(
        value,
        list,
    ):
        return [
            strip_nondeterministic_runtime(
                child
            )
            for child
            in value
        ]

    # Canonical JSON hashing must remain standards-compliant even
    # when physically meaningful downstream metrics contain
    # +/-inf or NaN (for example logarithmic loss quantities).
    #
    # Frozen and fresh results pass through this exact same
    # normalization, so no numerical discrepancy is hidden.
    if isinstance(
        value,
        float,
    ):
        return finite_float_value(
            value
        )

    # Defensive support for numpy scalar values that may appear
    # in a fresh in-memory result before JSON materialization.
    try:
        import numpy as np

        if isinstance(
            value,
            np.generic,
        ):
            scalar = value.item()

            if isinstance(
                scalar,
                float,
            ):
                return finite_float_value(
                    scalar
                )

            return scalar

    except ImportError:
        pass

    return value


def count_embedded_hashes(
    value: Any,
) -> int:
    count = 0

    if isinstance(
        value,
        dict,
    ):
        for key, child in value.items():
            if (
                str(key)
                ==
                "sha256"
            ):
                count += 1

            count += (
                count_embedded_hashes(
                    child
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for child in value:
            count += (
                count_embedded_hashes(
                    child
                )
            )

    return count


def find_canonical_runner(
    formal_report: dict[str, Any],
) -> tuple[Path, str]:
    expected = (
        formal_report[
            "provenance"
        ][
            "runner_sha256"
        ]
    )

    matches = []

    for path in sorted(
        (
            S5
            /
            "scripts"
        ).glob(
            "*.py"
        )
    ):
        if path.name.startswith(
            "run_block59_"
        ):
            continue

        try:
            digest = (
                sha256_file(
                    path
                )
            )
        except OSError:
            continue

        if digest == expected:
            matches.append(
                path
            )

    require(
        len(matches) == 1,
        (
            "Could not uniquely resolve canonical successful "
            f"Block5.8 runner for SHA {expected}. "
            f"Matches={matches}"
        ),
    )

    return (
        matches[0],
        expected,
    )


def import_runner(
    path: Path,
):
    module_name = (
        "_iscai_stage5_block58_"
        + hashlib.sha256(
            str(
                path
            ).encode(
                "utf-8"
            )
        ).hexdigest()[
            :16
        ]
    )

    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            path,
        )
    )

    require(
        spec is not None
        and
        spec.loader is not None,
        "Could not construct canonical runner import spec.",
    )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def target_names(
    target: ast.AST,
) -> set[str]:
    result: set[str] = set()

    if isinstance(
        target,
        ast.Name,
    ):
        result.add(
            target.id
        )

    elif isinstance(
        target,
        (ast.Tuple, ast.List),
    ):
        for child in target.elts:
            result.update(
                target_names(
                    child
                )
            )

    return result


def assignment_names(
    statement: ast.stmt,
) -> set[str]:
    result: set[str] = set()

    if isinstance(
        statement,
        ast.Assign,
    ):
        for target in statement.targets:
            result.update(
                target_names(
                    target
                )
            )

    elif isinstance(
        statement,
        ast.AnnAssign,
    ):
        result.update(
            target_names(
                statement.target
            )
        )

    return result


def loaded_names(
    node: ast.AST,
) -> set[str]:
    return {
        child.id
        for child
        in ast.walk(
            node
        )
        if isinstance(
            child,
            ast.Name,
        )
        and isinstance(
            child.ctx,
            ast.Load,
        )
    }


def call_name(
    node: ast.Call,
) -> str | None:
    if isinstance(
        node.func,
        ast.Name,
    ):
        return node.func.id

    return None


def find_main_call_and_loop(
    source: str,
) -> tuple[
    ast.FunctionDef,
    ast.Call,
    ast.For,
    ast.Module,
]:
    tree = ast.parse(
        source
    )

    main_node = None

    for node in tree.body:
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and
            node.name == "main"
        ):
            main_node = node
            break

    require(
        main_node is not None,
        "Canonical runner main() not found.",
    )

    parent: dict[
        ast.AST,
        ast.AST,
    ] = {}

    for node in ast.walk(
        main_node
    ):
        for child in ast.iter_child_nodes(
            node
        ):
            parent[
                child
            ] = node

    candidates = []

    for node in ast.walk(
        main_node
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        name = call_name(
            node
        )

        if name in {
            "process_scene",
            "process_scenario",
        }:
            candidates.append(
                node
            )

    require(
        len(candidates) == 1,
        (
            "Expected exactly one canonical process_scene/"
            f"process_scenario call; found {len(candidates)}."
        ),
    )

    call = candidates[0]

    current: ast.AST | None = call
    loop = None

    while current is not None:
        current = parent.get(
            current
        )

        if isinstance(
            current,
            ast.For,
        ):
            loop = current
            break

    require(
        loop is not None,
        "Canonical process call is not inside a for-loop.",
    )

    return (
        main_node,
        call,
        loop,
        tree,
    )


def compile_statement(
    statement: ast.stmt,
    filename: str,
):
    module = ast.Module(
        body=[
            statement
        ],
        type_ignores=[],
    )

    ast.fix_missing_locations(
        module
    )

    return compile(
        module,
        filename,
        "exec",
    )


def build_setup_namespace(
    module,
    source: str,
    filename: str,
    main_node: ast.FunctionDef,
    process_call: ast.Call,
    loop: ast.For,
) -> dict[str, Any]:
    # Canonical main() setup is reconstructed only from assignments
    # that feed the process call / loop iterable.
    assignments: dict[
        str,
        ast.stmt,
    ] = {}

    for statement in main_node.body:
        if (
            getattr(
                statement,
                "lineno",
                0,
            )
            >=
            loop.lineno
        ):
            break

        names = (
            assignment_names(
                statement
            )
        )

        for name in names:
            assignments[
                name
            ] = statement

    required: set[str] = set()

    for arg in process_call.args:
        required.update(
            loaded_names(
                arg
            )
        )

    for keyword in process_call.keywords:
        required.update(
            loaded_names(
                keyword.value
            )
        )

    required.update(
        loaded_names(
            loop.iter
        )
    )

    loop_bound = target_names(
        loop.target
    )

    required -= loop_bound

    selected: dict[
        int,
        ast.stmt,
    ] = {}

    visiting: set[str] = set()

    def include_name(
        name: str,
    ) -> None:
        if (
            name
            in module.__dict__
        ):
            return

        if hasattr(
            builtins,
            name,
        ):
            return

        if name in visiting:
            return

        statement = (
            assignments.get(
                name
            )
        )

        if statement is None:
            return

        visiting.add(
            name
        )

        selected[
            id(
                statement
            )
        ] = statement

        for dependency in loaded_names(
            statement
        ):
            if dependency == name:
                continue

            if dependency in assignments:
                include_name(
                    dependency
                )

        visiting.remove(
            name
        )

    for name in required:
        include_name(
            name
        )

    namespace = dict(
        module.__dict__
    )

    for statement in sorted(
        selected.values(),
        key=lambda item:
            item.lineno,
    ):
        exec(
            compile_statement(
                statement,
                filename,
            ),
            namespace,
            namespace,
        )

    unresolved = [
        name
        for name in required
        if (
            name not in namespace
            and
            name not in loop_bound
            and
            not hasattr(
                builtins,
                name,
            )
        )
    ]

    require(
        not unresolved,
        (
            "Could not reconstruct canonical main setup "
            f"variables: {unresolved}"
        ),
    )

    return namespace


def bind_target(
    target: ast.AST,
    value: Any,
    namespace: dict[str, Any],
) -> None:
    if isinstance(
        target,
        ast.Name,
    ):
        namespace[
            target.id
        ] = value
        return

    if isinstance(
        target,
        (ast.Tuple, ast.List),
    ):
        values = list(
            value
        )

        require(
            len(values)
            ==
            len(
                target.elts
            ),
            "Could not bind canonical loop target.",
        )

        for child, child_value in zip(
            target.elts,
            values,
        ):
            bind_target(
                child,
                child_value,
                namespace,
            )

        return

    raise RuntimeError(
        "Unsupported canonical loop target."
    )


def install_trace_wrappers(
    module,
    capture: dict[str, Any],
) -> list[str]:
    wrapped = []

    for name, category in (
        TRACE_FUNCTIONS.items()
    ):
        original = getattr(
            module,
            name,
            None,
        )

        if not callable(
            original
        ):
            continue

        @functools.wraps(
            original
        )
        def wrapper(
            *args,
            __original=original,
            __name=name,
            __category=category,
            **kwargs,
        ):
            input_payload = (
                trace_value(
                    {
                        "args":
                            args,

                        "kwargs":
                            kwargs,
                    }
                )
            )

            result = __original(
                *args,
                **kwargs,
            )

            output_payload = (
                trace_value(
                    result
                )
            )

            event = {
                "function":
                    __name,

                "category":
                    __category,

                "input":
                    input_payload,

                "output":
                    output_payload,

                "event_sha256":
                    canonical_sha(
                        {
                            "function":
                                __name,

                            "category":
                                __category,

                            "input":
                                input_payload,

                            "output":
                                output_payload,
                        }
                    ),
            }

            capture[
                "events"
            ].append(
                event
            )

            return result

        setattr(
            module,
            name,
            wrapper,
        )

        wrapped.append(
            name
        )

    return wrapped


def redirect_canonical_cache(
    module,
    workdir: Path,
) -> dict[str, str]:
    workdir.mkdir(
        parents=True,
        exist_ok=False,
    )

    redirects = {}

    for name, value in list(
        vars(
            module
        ).items()
    ):
        if not isinstance(
            value,
            Path,
        ):
            continue

        upper = name.upper()

        if "CACHE" not in upper:
            continue

        text = str(
            value
        )

        if (
            "artifacts/block58"
            not in text
        ):
            continue

        if (
            value.suffix
        ):
            replacement = (
                workdir
                /
                value.name
            )

        else:
            replacement = (
                workdir
                /
                value.name
            )

            replacement.mkdir(
                parents=True,
                exist_ok=True,
            )

        setattr(
            module,
            name,
            replacement,
        )

        redirects[
            name
        ] = str(
            replacement
        )

    require(
        any(
            "CACHE_DIR"
            in name.upper()
            for name
            in redirects
        ),
        (
            "Canonical runner cache directory could "
            "not be redirected."
        ),
    )

    return redirects


def evaluate_expression(
    expression: ast.AST,
    namespace: dict[str, Any],
    module_globals: dict[str, Any],
    filename: str,
) -> Any:
    node = ast.Expression(
        body=expression
    )

    ast.fix_missing_locations(
        node
    )

    return eval(
        compile(
            node,
            filename,
            "eval",
        ),
        module_globals,
        namespace,
    )


def call_process(
    module,
    process_call: ast.Call,
    namespace: dict[str, Any],
    filename: str,
):
    function = evaluate_expression(
        process_call.func,
        namespace,
        module.__dict__,
        filename,
    )

    args = [
        evaluate_expression(
            arg,
            namespace,
            module.__dict__,
            filename,
        )
        for arg in process_call.args
    ]

    kwargs = {}

    for keyword in process_call.keywords:
        require(
            keyword.arg is not None,
            "Canonical **kwargs expansion unsupported.",
        )

        kwargs[
            keyword.arg
        ] = (
            evaluate_expression(
                keyword.value,
                namespace,
                module.__dict__,
                filename,
            )
        )

    returned = function(
        *args,
        **kwargs,
    )

    reused = None

    if (
        isinstance(
            returned,
            tuple,
        )
        and
        len(
            returned
        )
        ==
        2
        and
        isinstance(
            returned[
                1
            ],
            bool,
        )
    ):
        result = returned[
            0
        ]

        reused = returned[
            1
        ]

    else:
        result = returned

    require(
        isinstance(
            result,
            dict,
        ),
        (
            "Canonical process call did not return "
            "a dictionary scenario result."
        ),
    )

    if reused is True:
        raise RuntimeError(
            "Fresh-process worker unexpectedly reused a cache."
        )

    return (
        result,
        reused,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--label",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--workdir",
        required=True,
    )

    args = parser.parse_args()

    output = Path(
        args.output
    )

    workdir = Path(
        args.workdir
    )

    require(
        not workdir.exists(),
        (
            "Fresh worker workdir already exists; "
            "refusing possible reuse."
        ),
    )

    formal_report = load_json(
        FORMAL_REPORT
    )

    require(
        formal_report.get(
            "status"
        )
        ==
        "PASS",
        "Frozen Block5.8 formal report is not PASS.",
    )

    runner_path, runner_sha = (
        find_canonical_runner(
            formal_report
        )
    )

    module = import_runner(
        runner_path
    )

    source = (
        runner_path.read_text(
            encoding="utf-8"
        )
    )

    (
        main_node,
        process_call,
        loop,
        _tree,
    ) = (
        find_main_call_and_loop(
            source
        )
    )

    # Read-only canonical preformal verification when exposed.
    verify = getattr(
        module,
        "verify_prefreeze",
        None,
    )

    if callable(
        verify
    ):
        verify()

    namespace = (
        build_setup_namespace(
            module,
            source,
            str(
                runner_path
            ),
            main_node,
            process_call,
            loop,
        )
    )

    loop_values = list(
        evaluate_expression(
            loop.iter,
            namespace,
            module.__dict__,
            str(
                runner_path
            ),
        )
    )

    require(
        len(
            loop_values
        )
        ==
        120,
        (
            "Canonical formal loop did not resolve "
            "to exactly 120 scenarios."
        ),
    )

    frozen_files = sorted(
        FROZEN_CACHE.glob(
            "*.json"
        )
    )

    require(
        len(
            frozen_files
        )
        ==
        120,
        "Frozen Block5.8 current cache count != 120.",
    )

    redirects = (
        redirect_canonical_cache(
            module,
            workdir,
        )
    )

    capture = {
        "events":
            [],
    }

    wrapped = (
        install_trace_wrappers(
            module,
            capture,
        )
    )

    require(
        "deterministic_receiver_angular_posterior"
        in wrapped,
        "Receiver posterior function was not instrumented.",
    )

    require(
        any(
            name
            in wrapped
            for name
            in (
                "beam_probability_mass_from_samples",
                "complete_partition_beam_probability_mass_from_samples",
            )
        ),
        "Beam probability function was not instrumented.",
    )

    require(
        "adaptive_temporal_step"
        in wrapped,
        "Adaptive Top-K function was not instrumented.",
    )

    require(
        "evaluate_optical_link"
        in wrapped,
        "Optical evaluator was not instrumented.",
    )

    rows_by_rank = {}

    for item in loop_values:
        temp_namespace = dict(
            namespace
        )

        bind_target(
            loop.target,
            item,
            temp_namespace,
        )

        rank = temp_namespace.get(
            "rank"
        )

        if rank is None:
            # Resolve first integer loop-bound value if the
            # canonical variable has another name.
            for name in target_names(
                loop.target
            ):
                candidate = (
                    temp_namespace.get(
                        name
                    )
                )

                if isinstance(
                    candidate,
                    int,
                ):
                    rank = candidate
                    break

        require(
            isinstance(
                rank,
                int,
            ),
            "Could not resolve formal rank from canonical loop.",
        )

        rows_by_rank[
            rank
        ] = item

    require(
        set(
            TARGET_RANKS
        ).issubset(
            rows_by_rank
        ),
        "Predetermined ranks 1/60/120 not in canonical loop.",
    )

    scene_outputs = []

    total_categories = {
        "receiver_posterior":
            0,

        "beam_probability":
            0,

        "topk_selection":
            0,

        "optical_metrics":
            0,
    }

    receiver_posterior_array_hashes = 0

    for rank in TARGET_RANKS:
        scene_namespace = dict(
            namespace
        )

        bind_target(
            loop.target,
            rows_by_rank[
                rank
            ],
            scene_namespace,
        )

        capture[
            "events"
        ] = []

        result, reused = (
            call_process(
                module,
                process_call,
                scene_namespace,
                str(
                    runner_path
                ),
            )
        )

        events = (
            capture[
                "events"
            ]
        )

        category_counts = {
            key:
                0
            for key
            in total_categories
        }

        for event in events:
            category = event[
                "category"
            ]

            if category in category_counts:
                category_counts[
                    category
                ] += 1

                total_categories[
                    category
                ] += 1

            if (
                category
                ==
                "receiver_posterior"
            ):
                receiver_posterior_array_hashes += (
                    count_embedded_hashes(
                        event[
                            "output"
                        ]
                    )
                )

        frozen_path = (
            frozen_files[
                rank
                -
                1
            ]
        )

        frozen = load_json(
            frozen_path
        )

        require(
            int(
                frozen.get(
                    "rank"
                )
            )
            ==
            rank,
            (
                "Frozen cache rank mismatch at "
                f"position {rank}."
            ),
        )

        # ----------------------------------------------------
        # Reproducibility comparison must be performed at the
        # SAME persistence boundary used by frozen Block5.8.
        #
        # The canonical runner intentionally keeps non-finite
        # scientific values (e.g. snr_db=-inf) unchanged in
        # memory, while json_safe() serializes them as null.
        #
        # Therefore:
        #   frozen persisted JSON
        #       vs
        #   freshly persisted JSON
        #
        # is the canonical exact-repeat comparison.
        # Comparing raw in-memory result against persisted JSON
        # would incorrectly report -inf != null.
        # ----------------------------------------------------

        fresh_cache_dir = getattr(
            module,
            "CACHE_DIR",
            None,
        )

        require(
            isinstance(
                fresh_cache_dir,
                Path,
            ),
            (
                "Canonical runner CACHE_DIR was not resolved "
                "as a redirected Path."
            ),
        )

        fresh_persisted_path = (
            fresh_cache_dir
            /
            frozen_path.name
        )

        require(
            fresh_persisted_path.is_file(),
            (
                "Fresh canonical process did not persist its "
                "scenario cache: "
                f"{fresh_persisted_path}"
            ),
        )

        fresh_persisted = load_json(
            fresh_persisted_path
        )

        # Additional persistence-contract check:
        # canonical json_safe(raw_result) must reproduce the
        # just-written fresh JSON structure exactly.
        canonical_json_safe = getattr(
            module,
            "json_safe",
            None,
        )

        require(
            callable(
                canonical_json_safe
            ),
            (
                "Canonical successful runner does not expose "
                "its persistence json_safe() function."
            ),
        )

        expected_persisted_from_raw = (
            canonical_json_safe(
                result
            )
        )

        require(
            canonical_sha(
                expected_persisted_from_raw
            )
            ==
            canonical_sha(
                fresh_persisted
            ),
            (
                "Fresh persisted cache does not match "
                "canonical json_safe(raw result)."
            ),
        )

        frozen_semantic = (
            strip_nondeterministic_runtime(
                frozen
            )
        )

        fresh_semantic = (
            strip_nondeterministic_runtime(
                fresh_persisted
            )
        )

        frozen_sha = (
            canonical_sha(
                frozen_semantic
            )
        )

        fresh_sha = (
            canonical_sha(
                fresh_semantic
            )
        )

        exact = (
            frozen_sha
            ==
            fresh_sha
        )

        require(
            exact,
            (
                "Fresh deterministic semantic result differs "
                f"from frozen Block5.8 cache for rank {rank}. "
                f"frozen={frozen_sha} fresh={fresh_sha}"
            ),
        )

        trace_sha = (
            canonical_sha(
                events
            )
        )

        scene_outputs.append(
            {
                "rank":
                    rank,

                "scenario_id":
                    result.get(
                        "scenario_id"
                    ),

                "selected_receiver":
                    result.get(
                        "selected_receiver"
                    ),

                "selected_prediction_available":
                    result.get(
                        "selected_prediction_available"
                    ),

                "fresh_cache_reused":
                    reused,

                "frozen_cache_file":
                    str(
                        frozen_path
                    ),

                "frozen_cache_file_sha256":
                    sha256_file(
                        frozen_path
                    ),

                "frozen_semantic_sha256":
                    frozen_sha,

                "fresh_semantic_sha256":
                    fresh_sha,

                "exact_match_to_frozen":
                    exact,

                "trace_category_counts":
                    category_counts,

                "trace_sha256":
                    trace_sha,

                "trace":
                    events,
            }
        )

    require(
        total_categories[
            "receiver_posterior"
        ]
        >
        0,
        (
            "No receiver-posterior execution was captured "
            "in predetermined fresh scenes."
        ),
    )

    require(
        total_categories[
            "beam_probability"
        ]
        >
        0,
        (
            "No beam-probability execution was captured "
            "in predetermined fresh scenes."
        ),
    )

    require(
        total_categories[
            "topk_selection"
        ]
        >
        0,
        (
            "No adaptive Top-K execution was captured "
            "in predetermined fresh scenes."
        ),
    )

    require(
        total_categories[
            "optical_metrics"
        ]
        >
        0,
        (
            "No optical-link execution was captured "
            "in predetermined fresh scenes."
        ),
    )

    require(
        receiver_posterior_array_hashes
        >
        0,
        (
            "Receiver-posterior trace did not contain "
            "numerical sample/tensor hashes."
        ),
    )

    semantic_signature = (
        canonical_sha(
            [
                {
                    "rank":
                        scene[
                            "rank"
                        ],

                    "fresh_semantic_sha256":
                        scene[
                            "fresh_semantic_sha256"
                        ],
                }
                for scene
                in scene_outputs
            ]
        )
    )

    trace_signature = (
        canonical_sha(
            [
                {
                    "rank":
                        scene[
                            "rank"
                        ],

                    "trace_sha256":
                        scene[
                            "trace_sha256"
                        ],
                }
                for scene
                in scene_outputs
            ]
        )
    )

    payload = {
        "stage":
            5,

        "block":
            "5.9",

        "phase":
            "fresh_process_worker",

        "label":
            args.label,

        "status":
            "PASS",

        "canonical_runner": {
            "path":
                str(
                    runner_path
                ),

            "sha256":
                runner_sha,

            "process_function":
                call_name(
                    process_call
                ),
        },

        "cache_redirection": {
            "frozen_cache_used_for_execution":
                False,

            "fresh_cache_root":
                str(
                    workdir
                ),

            "redirected_globals":
                redirects,
        },

        "predetermined_ranks":
            list(
                TARGET_RANKS
            ),

        "trace_functions_wrapped":
            wrapped,

        "trace_category_counts":
            total_categories,

        "receiver_posterior_embedded_array_hash_count":
            receiver_posterior_array_hashes,

        "all_fresh_semantics_equal_frozen":
            all(
                scene[
                    "exact_match_to_frozen"
                ]
                for scene
                in scene_outputs
            ),

        "semantic_signature_sha256":
            semantic_signature,

        "trace_signature_sha256":
            trace_signature,

        "scenes":
            scene_outputs,
    }

    atomic_json(
        output,
        payload,
    )

    print(
        f"fresh process {args.label} = PASS"
    )
    print(
        "canonical runner         =",
        runner_path,
    )
    print(
        "runner SHA256            =",
        runner_sha,
    )
    print(
        "ranks                    = 1 / 60 / 120"
    )
    print(
        "frozen semantic match    = EXACT"
    )
    print(
        "receiver posterior trace =",
        total_categories[
            "receiver_posterior"
        ],
    )
    print(
        "beam probability trace   =",
        total_categories[
            "beam_probability"
        ],
    )
    print(
        "Top-K trace              =",
        total_categories[
            "topk_selection"
        ],
    )
    print(
        "optical trace            =",
        total_categories[
            "optical_metrics"
        ],
    )
    print(
        "semantic signature       =",
        semantic_signature,
    )
    print(
        "trace signature          =",
        trace_signature,
    )


if __name__ == "__main__":
    main()
