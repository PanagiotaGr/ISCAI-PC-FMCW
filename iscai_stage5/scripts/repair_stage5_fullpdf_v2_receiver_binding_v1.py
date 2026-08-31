from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/agni/waymo")
S3 = ROOT / "iscai_stage3"
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"

HISTORICAL = (
    S5 / "scripts/run_block58_formal_clean.py"
)
FORMAL_CHECKER = (
    S5
    / "scripts/run_stage5_fullpdf_v2_independent_formal_checker.py"
)
CHECKER2 = (
    S5 / "scripts/check_stage5_fullpdf_v2.py"
)
RUNTIME = (
    S5 / "src/iscai_stage5/fullpdf_v2.py"
)
RECEIVER_SELECTION = (
    S5 / "src/iscai_stage5/receiver_selection.py"
)
PROTOCOL = (
    S5 / "configs/stage5_fullpdf_v2_protocol.json"
)
ADDENDUM = (
    S5
    / "configs/stage5_fullpdf_v2_preformal_addendum.json"
)
STAGE4_LEDGER = (
    S4
    / "artifacts/fullpdf_v2/formal_prediction_ledger_run1.jsonl"
)
FORMAL_MANIFEST = (
    S3
    / "artifacts/block38e/formal_validation_120.jsonl"
)
VALIDATION_MANIFEST = (
    ROOT
    / "iscai_data_prep/manifests/selected_validation.jsonl"
)

BIND_DIR = (
    S5 / "artifacts/fullpdf_v2_repair"
)
BINDING = (
    BIND_DIR
    / "primary_receiver_binding_fullpdf_v2.jsonl"
)
REGRESSION_LOG = (
    BIND_DIR
    / "receiver_binding_regression_v1.log"
)

REPORT = (
    S5
    / "reports/stage5_fullpdf_v2_receiver_binding_materialization_v1.json"
)
AMENDMENT = (
    S5
    / "configs/stage5_fullpdf_v2_receiver_binding_amendment_v1.json"
)

FORMAL_ART = (
    S5 / "artifacts/fullpdf_v2_independent_formal"
)
HISTORY_ROOT = (
    FORMAL_ART
    / "receiver_binding_failed_attempt_history"
)
SEAL = (
    FORMAL_ART
    / "stage5_fullpdf_v2_receiver_binding_superseding_pretruth_seal_v1.json"
)

EXPECTED = {
    HISTORICAL:
        "b1276eefd51444e0e648794d8bbb61ef829cc49bd1a09ba68ff79d9b61c8a9a5",
    RUNTIME:
        "e57708e9332bb40e84f18270e483d91165e26c149b80e82c8f10b4df44376249",
    PROTOCOL:
        "91dec3cc3d3a7b72385f012d1d2bfdb8981a497e1344f585d38f3e05f2e4fa5a",
    ADDENDUM:
        "570fbada05c8e8ed71775bbb6eb668f4016dd65c98b09b4f158abeb379eca6aa",
    STAGE4_LEDGER:
        "2dd3c396d5c365677c41b3a0df6344a7c28d423daaa368dbe466bd4b17b4c2a8",
    VALIDATION_MANIFEST:
        "dc10609ef18a2ba881657eb3da3a3df7a81bdcc8345ecbc2227102ab16b8833c",
}

OLD_SEAL = (
    "44b1ba5bd702f8081f4e7e8c07a9139ccaa9e588ec938851b652d05c82693803"
)


def fail(msg):
    raise SystemExit(
        "FAIL-CLOSED: " + str(msg)
    )


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def sha_bytes(data):
    return hashlib.sha256(
        data
    ).hexdigest()


def load_jsonl(path):
    rows = []
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for lineno, line in enumerate(
            f,
            1,
        ):
            if not line.strip():
                continue

            obj = json.loads(line)

            if not isinstance(obj, dict):
                fail(
                    f"{path}:{lineno} "
                    "is not a JSON object"
                )

            rows.append(obj)

    return rows


def canonical_json(obj):
    return (
        json.dumps(
            obj,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_jsonl(rows):
    return "".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def atomic_write(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    if tmp.exists():
        tmp.unlink()

    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def write_once(path, data):
    digest = sha_bytes(data)

    if path.exists():
        if (
            sha(path) != digest
            or path.read_bytes() != data
        ):
            fail(
                "write-once artifact "
                f"already differs: {path}"
            )

        return digest

    atomic_write(path, data)
    os.chmod(path, 0o444)

    return digest


def import_path(name, path):
    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        fail(
            f"cannot import {path}"
        )

    module = (
        importlib.util
        .module_from_spec(spec)
    )

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def has_main_guard(path):
    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    for node in tree.body:
        if not isinstance(
            node,
            ast.If,
        ):
            continue

        text = ast.unparse(
            node.test
        )

        if (
            "__name__" in text
            and "__main__" in text
        ):
            return True

    return False


def assignment_map(fn):
    """
    Resolve assignments inside historical main(), including
    tuple/list unpacking such as:

        clean_config, degraded_config = helper()

    This is source-resolution only. It does not execute main().
    """
    import copy

    found = {}

    def bind_target(
        target,
        value,
        lineno,
    ):
        if isinstance(target, ast.Name):
            found.setdefault(
                target.id,
                [],
            ).append(
                (
                    lineno,
                    value,
                )
            )
            return

        if isinstance(
            target,
            (ast.Tuple, ast.List),
        ):
            for index, element in enumerate(
                target.elts
            ):
                if isinstance(
                    element,
                    ast.Starred,
                ):
                    fail(
                        "starred historical "
                        "assignment is not "
                        "supported fail-closed"
                    )

                subscript = ast.Subscript(
                    value=copy.deepcopy(value),
                    slice=ast.Constant(
                        value=index
                    ),
                    ctx=ast.Load(),
                )

                ast.copy_location(
                    subscript,
                    value,
                )
                ast.fix_missing_locations(
                    subscript
                )

                bind_target(
                    element,
                    subscript,
                    lineno,
                )

            return

        # Attribute/subscript assignment is deliberately
        # not interpreted as a local scientific binding.
        return

    for node in ast.walk(fn):
        if isinstance(
            node,
            ast.Assign,
        ):
            for target in node.targets:
                bind_target(
                    target,
                    node.value,
                    node.lineno,
                )

        elif isinstance(
            node,
            ast.AnnAssign,
        ):
            if node.value is not None:
                bind_target(
                    node.target,
                    node.value,
                    node.lineno,
                )

    return {
        name: sorted(
            values,
            key=lambda x: x[0],
        )[0][1]
        for name, values
        in found.items()
    }



def resolve_main_value(
    source,
    tree,
    module,
    name,
):
    main_fn = next(
        (
            n
            for n in tree.body
            if isinstance(
                n,
                ast.FunctionDef,
            )
            and n.name == "main"
        ),
        None,
    )

    if main_fn is None:
        fail(
            "historical source "
            "has no main()"
        )

    amap = assignment_map(
        main_fn
    )

    env = dict(
        module.__dict__
    )

    resolving = set()

    forbidden = (
        "attach_supervision",
        "future_center",
        "future_heading",
        "oracle",
        "evaluator",
        "tracks_to_predict",
        "process_scene",
    )

    def resolve(var):
        if var in env:
            return env[var]

        if var in resolving:
            fail(
                "cyclic assignment "
                f"while resolving {var}"
            )

        expr = amap.get(var)

        if expr is None:
            fail(
                "cannot resolve exact "
                f"historical variable {var}"
            )

        expr_text = (
            ast.get_source_segment(
                source,
                expr,
            )
            or ast.unparse(expr)
        )

        low = expr_text.lower()

        if any(
            token in low
            for token in forbidden
        ):
            fail(
                "unsafe expression "
                f"while resolving {var}: "
                f"{expr_text}"
            )

        resolving.add(var)

        dependencies = {
            n.id
            for n in ast.walk(expr)
            if isinstance(
                n,
                ast.Name,
            )
            and isinstance(
                n.ctx,
                ast.Load,
            )
        }

        for dep in dependencies:
            if (
                dep not in env
                and dep in amap
            ):
                env[dep] = resolve(dep)

        try:
            value = eval(
                compile(
                    ast.Expression(expr),
                    str(HISTORICAL),
                    "eval",
                ),
                env,
            )
        except Exception as exc:
            fail(
                "could not evaluate "
                f"{var}={expr_text!r}: "
                f"{exc}"
            )

        resolving.remove(var)
        env[var] = value

        return value

    return resolve(name)


def resolve_vehicle_semantic(
    source,
    tree,
    module,
):
    try:
        return resolve_main_value(
            source,
            tree,
            module,
            "vehicle_semantic",
        )
    except SystemExit:
        pass

    main_fn = next(
        n
        for n in tree.body
        if isinstance(
            n,
            ast.FunctionDef,
        )
        and n.name == "main"
    )

    for node in ast.walk(main_fn):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if not (
            isinstance(
                node.func,
                ast.Name,
            )
            and node.func.id
            == "process_scene"
        ):
            continue

        for kw in node.keywords:
            if (
                kw.arg
                != "vehicle_semantic"
            ):
                continue

            text = (
                ast.get_source_segment(
                    source,
                    kw.value,
                )
                or ast.unparse(
                    kw.value
                )
            )

            if any(
                x in text.lower()
                for x in (
                    "future",
                    "truth",
                    "evaluator",
                    "oracle",
                    "tracks_to_predict",
                )
            ):
                fail(
                    "unsafe "
                    "vehicle_semantic "
                    "expression"
                )

            try:
                return eval(
                    compile(
                        ast.Expression(
                            kw.value
                        ),
                        str(HISTORICAL),
                        "eval",
                    ),
                    dict(
                        module.__dict__
                    ),
                )
            except Exception:
                pass

    fail(
        "cannot resolve exact "
        "historical vehicle_semantic"
    )


def snapshot_failed_attempt():
    HISTORY_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    dest = (
        HISTORY_ROOT
        / (
            "before_receiver_binding_"
            + stamp
        )
    )

    if dest.exists():
        fail(
            "history destination "
            f"exists: {dest}"
        )

    dest.mkdir()

    candidates = [
        (
            S5
            / "reports/stage5_fullpdf_v2_independent_formal_checker.json"
        ),
        (
            S5
            / "reports/stage5_fullpdf_v2_checker2_gate.json"
        ),
        (
            FORMAL_ART
            / "checker2_frozen_checker.log"
        ),
    ]

    if FORMAL_ART.is_dir():
        for p in sorted(
            FORMAL_ART.iterdir()
        ):
            if (
                p.is_file()
                and p.suffix.lower()
                in {
                    ".json",
                    ".jsonl",
                    ".log",
                    ".txt",
                }
                and p.stat().st_size
                <= 100 * 1024 * 1024
            ):
                candidates.append(p)

    unique = []
    seen = set()

    for p in candidates:
        if not p.is_file():
            continue

        resolved = p.resolve()

        if resolved in seen:
            continue

        seen.add(resolved)
        unique.append(p)

    entries = []

    for index, src in enumerate(
        unique
    ):
        dst = (
            dest
            / (
                f"{index:03d}_"
                + src.name
            )
        )

        shutil.copy2(
            src,
            dst,
        )

        entries.append(
            {
                "source":
                    str(src),
                "source_sha256":
                    sha(src),
                "snapshot":
                    str(dst),
                "snapshot_sha256":
                    sha(dst),
            }
        )

    manifest = {
        "stage": 5,
        "version": "fullpdf_v2",
        "status":
            "PRESERVED_FAILED_PREFORMAL_BINDING_ATTEMPT",
        "previous_checker2_pretruth_seal_sha256":
            OLD_SEAL,
        "scientific_interpretation":
            "previous checker stopped "
            "before causal decision pass 1; "
            "no formal scientific outcome "
            "is inferred by this repair",
        "files": entries,
    }

    manifest_path = (
        dest
        / "history_manifest.json"
    )

    atomic_write(
        manifest_path,
        canonical_json(
            manifest
        ),
    )

    os.chmod(
        manifest_path,
        0o444,
    )

    return (
        manifest_path,
        sha(manifest_path),
    )


def compact_offset(
    formal,
    validation,
):
    """
    Read the already-existing compact/motion offset from either
    a JSON mapping or the canonical typed Stage3 manifest row.

    No offset is inferred, searched, or recomputed here.
    """

    def exact_value(
        obj,
        key,
    ):
        if isinstance(
            obj,
            dict,
        ):
            if key in obj:
                return obj[key]
            return None

        if hasattr(
            obj,
            key,
        ):
            return getattr(
                obj,
                key,
            )

        return None

    for obj in (
        formal,
        validation,
    ):
        for key in (
            "compact_record_offset",
            "motion_record_offset",
        ):
            value = exact_value(
                obj,
                key,
            )

            if value is not None:
                return int(
                    value
                )

    fail(
        "no compact/motion "
        "record offset for "
        + str(
            formal.get(
                "scenario_id"
            )
        )
    )



def main():
    print(
        "=" * 79
    )
    print(
        "STAGE5 FULLPDF V2 — "
        "PREFORMAL RECEIVER-BINDING "
        "MATERIALIZATION V1"
    )
    print(
        "NO FORMAL METRICS / "
        "NO FUTURE STATE / "
        "NO TUNING"
    )
    print(
        "=" * 79
    )

    generated = (
        BINDING,
        AMENDMENT,
        SEAL,
        REPORT,
    )

    if any(
        p.exists()
        for p in generated
    ):
        if all(
            p.is_file()
            for p in generated
        ):
            print(
                "PASS | repair already "
                "materialized; no rewrite"
            )

            for p in generated:
                print(
                    sha(p),
                    p,
                )

            print(
                "Stage6_allowed = false"
            )
            return

        fail(
            "partial prior V1 repair "
            "exists; do not overwrite"
        )

    required = (
        HISTORICAL,
        FORMAL_CHECKER,
        CHECKER2,
        RUNTIME,
        RECEIVER_SELECTION,
        PROTOCOL,
        ADDENDUM,
        STAGE4_LEDGER,
        FORMAL_MANIFEST,
        VALIDATION_MANIFEST,
    )

    for path in required:
        if not path.is_file():
            fail(
                "missing required path: "
                f"{path}"
            )

    for path, expected in (
        EXPECTED.items()
    ):
        actual = sha(path)

        if actual != expected:
            fail(
                "frozen SHA mismatch: "
                f"{path}\n"
                f"actual   = {actual}\n"
                f"expected = {expected}"
            )

        print(
            "PASS | frozen SHA |",
            path.name,
        )

    if not has_main_guard(
        HISTORICAL
    ):
        fail(
            "historical materializer "
            "has no __main__ guard"
        )

    protected_paths = (
        HISTORICAL,
        FORMAL_CHECKER,
        CHECKER2,
        RUNTIME,
        RECEIVER_SELECTION,
        PROTOCOL,
        ADDENDUM,
        STAGE4_LEDGER,
        FORMAL_MANIFEST,
        VALIDATION_MANIFEST,
    )

    protected_before = {
        str(p): sha(p)
        for p in protected_paths
    }

    (
        history_manifest,
        history_manifest_sha,
    ) = snapshot_failed_attempt()

    print(
        "PASS | previous failed "
        "attempt preserved"
    )
    print(
        "history =",
        history_manifest,
    )

    for src in (
        S5 / "src",
        S4 / "src",
        S3 / "src",
        ROOT / "iscai_stage2/src",
        ROOT / "iscai_stage1/src",
        ROOT / "iscai_stage0/src",
    ):
        if src.is_dir():
            sys.path.insert(
                0,
                str(src),
            )

    source = (
        HISTORICAL.read_text(
            encoding="utf-8"
        )
    )
    tree = ast.parse(
        source
    )

    old = import_path(
        "_stage5_frozen_block58_"
        "causal_materializer",
        HISTORICAL,
    )

    clean_config = (
        resolve_main_value(
            source,
            tree,
            old,
            "clean_config",
        )
    )

    degraded_config = (
        resolve_main_value(
            source,
            tree,
            old,
            "degraded_config",
        )
    )

    vehicle_semantic = (
        resolve_vehicle_semantic(
            source,
            tree,
            old,
        )
    )

    print(
        "PASS | exact historical "
        "causal configs resolved"
    )
    print(
        "vehicle_semantic =",
        vehicle_semantic,
    )

    ledger_rows = load_jsonl(
        STAGE4_LEDGER
    )

    if len(ledger_rows) != 3952:
        fail(
            "Stage4 ledger row count "
            f"changed: {len(ledger_rows)}"
        )

    ledger_keys = set()
    ledger_scenes = set()

    for row in ledger_rows:
        sid = str(
            row["scenario_id"]
        )
        pid = str(
            row["prediction_id"]
        )
        key = (
            sid,
            pid,
        )

        if key in ledger_keys:
            fail(
                "duplicate Stage4 key: "
                f"{key}"
            )

        ledger_keys.add(key)
        ledger_scenes.add(sid)

    if len(
        ledger_scenes
    ) != 120:
        fail(
            "Stage4 scene count "
            f"changed: "
            f"{len(ledger_scenes)}"
        )

    formal_by_sid = {}

    for row in load_jsonl(
        FORMAL_MANIFEST
    ):
        sid = str(
            row["scenario_id"]
        )

        if sid in formal_by_sid:
            fail(
                "duplicate formal "
                f"scene: {sid}"
            )

        formal_by_sid[
            sid
        ] = row

    if (
        set(formal_by_sid)
        != ledger_scenes
    ):
        fail(
            "Stage3 formal scene set "
            "!= frozen Stage4 ledger "
            "scene set"
        )

    # Canonical Stage3 typed manifest reader.
    # Do NOT pass raw JSON dicts to read_motion_scenario().
    from iscai_stage3.validation.womd_access import (
        read_validation_manifest as _read_validation_manifest,
    )

    _typed_validation_rows = list(
        _read_validation_manifest(
            VALIDATION_MANIFEST
        )
    )

    if not _typed_validation_rows:
        fail(
            "canonical Stage3 validation "
            "manifest reader returned zero rows"
        )

    validation_by_id = {}

    for _row in _typed_validation_rows:
        if not hasattr(
            _row,
            "scenario_id",
        ):
            fail(
                "canonical validation row "
                "has no scenario_id attribute"
            )

        _sid = str(
            _row.scenario_id
        )

        if _sid in validation_by_id:
            fail(
                "duplicate canonical validation "
                f"manifest scene: {_sid}"
            )

        validation_by_id[
            _sid
        ] = _row

    print(
        "PASS | canonical Stage3 "
        "typed validation manifest loaded"
    )
    print(
        "typed validation rows =",
        len(validation_by_id),
    )


    missing = (
        ledger_scenes
        - set(
            validation_by_id
        )
    )

    if missing:
        fail(
            "formal scenes missing "
            "from validation manifest: "
            + repr(
                sorted(missing)[:5]
            )
        )

    for name in (
        "process_scene",
        "causal_actor_bridge",
        "current_heading",
        "read_motion_scenario",
        "ReceiverCandidate",
        "ReceiverSelectionConfig",
        "select_primary_receiver",
        "attach_supervision",
        "atomic_json",
        "CACHE_DIR",
    ):
        if not hasattr(
            old,
            name,
        ):
            fail(
                "historical source "
                f"missing {name}"
            )

    original_attach = (
        old.attach_supervision
    )
    original_atomic = (
        old.atomic_json
    )
    original_cache = (
        old.CACHE_DIR
    )
    original_bridge = (
        old.causal_actor_bridge
    )
    original_read = (
        old.read_motion_scenario
    )
    original_selector = (
        old.select_primary_receiver
    )

    binding_rows = []
    no_receiver = []
    no_stage4_prediction = []

    with tempfile.TemporaryDirectory(
        prefix=(
            "stage5_receiver_"
            "binding_"
        )
    ) as tmp:

        old.CACHE_DIR = Path(tmp)

        # Critical safety repair:
        # DO NOT attach supervision.
        # process_scene therefore stops
        # immediately after causal selection.
        old.attach_supervision = (
            lambda *a, **k: []
        )

        # Do not write historical caches.
        old.atomic_json = (
            lambda *a, **k: None
        )

        for rank, sid in enumerate(
            sorted(
                ledger_scenes
            )
        ):
            formal = dict(
                formal_by_sid[
                    sid
                ]
            )

            validation = (
                validation_by_id[
                    sid
                ]
            )

            if (
                "compact_record_offset"
                not in formal
            ):
                formal[
                    "compact_record_offset"
                ] = compact_offset(
                    formal,
                    validation,
                )

            capture = {}

            def read_capture(
                *args,
                **kwargs,
            ):
                scenario = (
                    original_read(
                        *args,
                        **kwargs,
                    )
                )

                capture[
                    "scenario"
                ] = scenario

                return scenario

            def bridge_capture(
                *args,
                **kwargs,
            ):
                result = (
                    original_bridge(
                        *args,
                        **kwargs,
                    )
                )

                capture[
                    "bridge"
                ] = result

                return result

            old.read_motion_scenario = (
                read_capture
            )
            old.causal_actor_bridge = (
                bridge_capture
            )

            base, _ = (
                old.process_scene(
                    rank,
                    formal,
                    validation_by_id,
                    clean_config=(
                        clean_config
                    ),
                    degraded_config=(
                        degraded_config
                    ),
                    runtime=None,
                    receiver_parameters=None,
                    geometry_modes=None,
                    vehicle_semantic=(
                        vehicle_semantic
                    ),
                    codebooks=None,
                    tbeam=0.0,
                    tframe=1.0,
                    runner_sha=(
                        EXPECTED[
                            HISTORICAL
                        ]
                    ),
                )
            )

            if (
                "scenario"
                not in capture
                or "bridge"
                not in capture
            ):
                fail(
                    "causal capture "
                    f"missing: {sid}"
                )

            scenario = (
                capture[
                    "scenario"
                ]
            )

            (
                histories,
                actors,
                matches,
            ) = capture[
                "bridge"
            ]

            candidate_rows = []

            for hi, history in enumerate(
                histories
            ):
                if hi not in matches:
                    continue

                ai, distance = (
                    matches[hi]
                )

                actor, _ = (
                    actors[ai]
                )

                track_index = int(
                    actor.track_index
                )

                object_type = int(
                    scenario
                    .tracks[
                        track_index
                    ]
                    .object_type
                )

                semantic = (
                    vehicle_semantic
                    if object_type == 1
                    else "other"
                )

                candidate_rows.append(
                    {
                        "history":
                            history,
                        "track_index":
                            track_index,
                        "semantic":
                            semantic,
                        "bridge_distance_m":
                            float(
                                distance
                            ),
                    }
                )

            candidates = [
                old.ReceiverCandidate(
                    semantic_class=(
                        row[
                            "semantic"
                        ]
                    ),
                    position_h0_m=tuple(
                        float(x)
                        for x in (
                            row[
                                "history"
                            ]
                            .latest_position_H0_m
                        )
                    ),
                    current_available=True,
                    association_valid=True,
                )
                for row
                in candidate_rows
            ]

            selection = (
                original_selector(
                    candidates,
                    old.ReceiverSelectionConfig(),
                )
            )

            idx = (
                selection
                .selected_candidate_index
            )

            if idx is None:
                if bool(
                    base.get(
                        "selected_receiver"
                    )
                ):
                    fail(
                        "base/selector "
                        f"mismatch: {sid}"
                    )

                no_receiver.append(
                    sid
                )
                continue

            selected = (
                candidate_rows[
                    int(idx)
                ]
            )

            pid = str(
                selected[
                    "history"
                ].prediction_id
            )

            if (
                not bool(
                    base.get(
                        "selected_receiver"
                    )
                )
                or str(
                    base.get(
                        "receiver_prediction_id"
                    )
                )
                != pid
            ):
                fail(
                    "historical exact "
                    "selector replay "
                    f"mismatch: {sid}"
                )

            key = (
                sid,
                pid,
            )

            if key not in ledger_keys:
                no_stage4_prediction.append(
                    {
                        "scenario_id":
                            sid,
                        "prediction_id":
                            pid,
                    }
                )
                continue

            current_index = int(
                scenario.current_time_index
            )

            if current_index != 10:
                fail(
                    "current_time_index "
                    f"changed: {sid}"
                )

            track_index = int(
                selected[
                    "track_index"
                ]
            )

            state = (
                scenario
                .tracks[
                    track_index
                ]
                .states[
                    current_index
                ]
            )

            if not bool(
                state.valid
            ):
                fail(
                    "selected current "
                    f"state invalid: {sid}"
                )

            heading = float(
                old.current_heading(
                    selected[
                        "history"
                    ]
                )
            )

            length = float(
                state.length
            )
            width = float(
                state.width
            )
            height = float(
                state.height
            )

            if not (
                math.isfinite(
                    heading
                )
                and math.isfinite(
                    length
                )
                and math.isfinite(
                    width
                )
                and math.isfinite(
                    height
                )
                and length > 0.0
                and width > 0.0
                and height > 0.0
            ):
                fail(
                    "invalid current "
                    "receiver geometry: "
                    f"{sid}"
                )

            # Deliberately minimal schema:
            # no truth/future/evaluator fields.
            binding_rows.append(
                {
                    "actor_class":
                        str(
                            selected[
                                "semantic"
                            ]
                        ),
                    "heading_h0_rad":
                        heading,
                    "height_m":
                        height,
                    "length_m":
                        length,
                    "prediction_id":
                        pid,
                    "scenario_id":
                        sid,
                    "selected_receiver":
                        True,
                    "selected_receiver_prediction_id":
                        pid,
                    "width_m":
                        width,
                }
            )

    old.attach_supervision = (
        original_attach
    )
    old.atomic_json = (
        original_atomic
    )
    old.CACHE_DIR = (
        original_cache
    )
    old.causal_actor_bridge = (
        original_bridge
    )
    old.read_motion_scenario = (
        original_read
    )

    if not binding_rows:
        fail(
            "zero selected receivers "
            "bind to Stage4 ledger"
        )

    binding_rows.sort(
        key=lambda r: (
            r["scenario_id"],
            r["prediction_id"],
        )
    )

    represented = [
        r["scenario_id"]
        for r in binding_rows
    ]

    if (
        len(represented)
        != len(
            set(represented)
        )
    ):
        fail(
            "more than one selected "
            "receiver per represented "
            "scene"
        )

    for row in binding_rows:
        key = (
            row["scenario_id"],
            row["prediction_id"],
        )

        if key not in ledger_keys:
            fail(
                "binding escaped "
                "Stage4 ledger"
            )

    binding_sha = write_once(
        BINDING,
        canonical_jsonl(
            binding_rows
        ),
    )

    print(
        "PASS | primary receiver "
        "binding materialized"
    )
    print(
        "represented scenes =",
        len(binding_rows),
    )
    print(
        "no receiver scenes =",
        len(no_receiver),
    )
    print(
        "selected receiver "
        "without Stage4 prediction =",
        len(
            no_stage4_prediction
        ),
    )
    print(
        "binding_sha256 =",
        binding_sha,
    )

    # Independent proof using the
    # UNMODIFIED frozen checker.
    if not has_main_guard(
        FORMAL_CHECKER
    ):
        fail(
            "formal checker lacks "
            "__main__ guard"
        )

    checker = import_path(
        "_stage5_v2_binding_probe",
        FORMAL_CHECKER,
    )

    (
        chosen,
        parsed,
    ) = checker.resolve_causal_binding(
        ledger_keys
    )

    if (
        chosen.resolve()
        != BINDING.resolve()
    ):
        fail(
            "unchanged checker "
            "resolved another binding: "
            f"{chosen}"
        )

    if len(parsed) != len(
        binding_rows
    ):
        fail(
            "checker parsed count "
            f"{len(parsed)} != "
            f"{len(binding_rows)}"
        )

    print(
        "PASS | unchanged formal "
        "checker resolves binding "
        "uniquely"
    )

    env = dict(
        os.environ
    )

    pyroots = [
        str(S5 / "src"),
        str(S4 / "src"),
        str(S3 / "src"),
        str(
            ROOT
            / "iscai_stage2/src"
        ),
        str(
            ROOT
            / "iscai_stage1/src"
        ),
        str(
            ROOT
            / "iscai_stage0/src"
        ),
    ]

    if env.get(
        "PYTHONPATH"
    ):
        pyroots.append(
            env[
                "PYTHONPATH"
            ]
        )

    env[
        "PYTHONPATH"
    ] = os.pathsep.join(
        pyroots
    )

    proc = subprocess.run(
        [
            sys.executable,
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
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if REGRESSION_LOG.exists():
        fail(
            "regression V1 log "
            "already exists; "
            "will not overwrite"
        )

    atomic_write(
        REGRESSION_LOG,
        proc.stdout.encode(
            "utf-8"
        ),
    )

    os.chmod(
        REGRESSION_LOG,
        0o444,
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests",
        proc.stdout,
    )

    count = (
        None
        if match is None
        else int(
            match.group(1)
        )
    )

    if (
        proc.returncode != 0
        or count != 271
    ):
        fail(
            "Stage5 regression "
            f"failed: rc="
            f"{proc.returncode}, "
            f"tests={count}"
        )

    print(
        "PASS | full Stage5 "
        "regression 271/271"
    )

    for (
        path_text,
        before,
    ) in protected_before.items():
        path = Path(
            path_text
        )

        after = sha(path)

        if after != before:
            fail(
                "protected frozen "
                f"artifact mutated: {path}"
            )

    print(
        "PASS | protected frozen "
        "artifacts unchanged"
    )

    amendment = {
        "stage": 5,
        "version": "fullpdf_v2",
        "status":
            "FROZEN_PREFORMAL_RECEIVER_BINDING_MATERIALIZED_V1",
        "repair_scope":
            "materialize already-frozen "
            "primary causal receiver "
            "selection as explicit "
            "checker-readable artifact",
        "scientific_policy_change":
            False,
        "receiver_policy":
            "nearest_causal_vehicle_ahead",
        "selector":
            "iscai_stage5.receiver_selection.select_primary_receiver",
        "selector_config":
            "ReceiverSelectionConfig()",
        "candidate_semantics":
            "exact historical "
            "causal_actor_bridge -> "
            "ReceiverCandidate -> "
            "select_primary_receiver replay",
        "attach_supervision_called":
            False,
        "formal_metrics_read":
            False,
        "formal_parameter_tuning":
            False,
        "retraining":
            False,
        "recalibration":
            False,
        "binding": {
            "path":
                str(BINDING),
            "sha256":
                binding_sha,
            "represented_scenes":
                len(
                    binding_rows
                ),
            "selected_receiver_prediction_unavailable":
                len(
                    no_stage4_prediction
                ),
            "no_receiver_scenes":
                len(
                    no_receiver
                ),
        },
        "historical_materializer": {
            "path":
                str(
                    HISTORICAL
                ),
            "sha256":
                sha(
                    HISTORICAL
                ),
        },
        "failed_attempt_history": {
            "manifest":
                str(
                    history_manifest
                ),
            "manifest_sha256":
                history_manifest_sha,
            "previous_checker2_pretruth_seal_sha256":
                OLD_SEAL,
        },
        "unchanged_frozen_checker": {
            "path":
                str(
                    FORMAL_CHECKER
                ),
            "sha256":
                sha(
                    FORMAL_CHECKER
                ),
            "resolves_binding_uniquely":
                True,
        },
        "regression": {
            "tests":
                271,
            "status":
                "PASS",
            "log":
                str(
                    REGRESSION_LOG
                ),
            "log_sha256":
                sha(
                    REGRESSION_LOG
                ),
        },
        "formal_rerun_executed":
            False,
        "Stage6_allowed":
            False,
    }

    amendment_sha = write_once(
        AMENDMENT,
        canonical_json(
            amendment
        ),
    )

    seal_payload = {
        "stage": 5,
        "version": "fullpdf_v2",
        "status":
            "SUPERSEDING_PREFORMAL_SEAL_AFTER_RECEIVER_BINDING_V1",
        "supersedes_only":
            OLD_SEAL,
        "does_not_erase_prior_attempt":
            True,
        "formal_rerun_not_started":
            True,
        "binding_sha256":
            binding_sha,
        "amendment_sha256":
            amendment_sha,
        "regression_log_sha256":
            sha(
                REGRESSION_LOG
            ),
        "history_manifest_sha256":
            history_manifest_sha,
        "frozen_inputs": {
            "runtime":
                sha(RUNTIME),
            "protocol":
                sha(PROTOCOL),
            "preformal_addendum":
                sha(ADDENDUM),
            "Stage4_ledger":
                sha(
                    STAGE4_LEDGER
                ),
            "formal_checker":
                sha(
                    FORMAL_CHECKER
                ),
            "checker2":
                sha(CHECKER2),
            "receiver_selection":
                sha(
                    RECEIVER_SELECTION
                ),
            "historical_materializer":
                sha(
                    HISTORICAL
                ),
        },
        "scientific_boundary": {
            "formal_metrics_read":
                False,
            "formal_parameter_tuning":
                False,
            "retraining":
                False,
            "recalibration":
                False,
            "future_state_access":
                False,
            "tracks_to_predict_selector":
                False,
        },
    }

    seal_sha = write_once(
        SEAL,
        canonical_json(
            seal_payload
        ),
    )

    report = dict(
        amendment
    )

    report[
        "amendment"
    ] = {
        "path":
            str(AMENDMENT),
        "sha256":
            amendment_sha,
    }

    report[
        "superseding_pretruth_seal"
    ] = {
        "path":
            str(SEAL),
        "sha256":
            seal_sha,
    }

    report[
        "status"
    ] = (
        "PASS_READY_FOR_"
        "SUPERSEDING_FORMAL_RERUN"
    )

    report_sha = write_once(
        REPORT,
        canonical_json(
            report
        ),
    )

    print()
    print(
        "=" * 79
    )
    print(
        "RECEIVER-BINDING "
        "REPAIR = PASS"
    )
    print(
        "Stage5 = "
        "READY_FOR_SUPERSEDING_FORMAL_RERUN"
    )
    print(
        "Stage6_allowed = false"
    )
    print(
        "formal scientific outcome = "
        "NOT COMPUTED BY THIS REPAIR"
    )
    print(
        "binding_sha256 =",
        binding_sha,
    )
    print(
        "amendment_sha256 =",
        amendment_sha,
    )
    print(
        "superseding_pretruth_seal_sha256 =",
        seal_sha,
    )
    print(
        "report_sha256 =",
        report_sha,
    )
    print(
        "=" * 79
    )


if __name__ == "__main__":
    main()
