from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from collections.abc import Mapping

import numpy as np


ROOT = Path("/home/agni/waymo")
S3 = ROOT / "iscai_stage3"
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"

HISTORICAL = (
    S5 / "scripts/run_block58_formal_clean.py"
)

CHECKER = (
    S5
    / "scripts/run_stage5_fullpdf_v2_independent_formal_checker.py"
)

CHECKER2 = (
    S5 / "scripts/check_stage5_fullpdf_v2.py"
)

RUNTIME = (
    S5 / "src/iscai_stage5/fullpdf_v2.py"
)

PROTOCOL = (
    S5 / "configs/stage5_fullpdf_v2_protocol.json"
)

ADDENDUM = (
    S5 / "configs/stage5_fullpdf_v2_preformal_addendum.json"
)

STAGE4_LEDGER = (
    S4 / "artifacts/fullpdf_v2/formal_prediction_ledger_run1.jsonl"
)

VALIDATION_MANIFEST = (
    ROOT
    / "iscai_data_prep/manifests/selected_validation.jsonl"
)

FORMAL_MANIFEST = (
    S3 / "artifacts/block38e/formal_validation_120.jsonl"
)

FORMAL_MANIFEST_REPORT = (
    S3 / "reports/block38e_formal_manifest_gate.json"
)

BINDING = (
    S5
    / "artifacts/fullpdf_v2_repair/"
      "primary_receiver_binding_fullpdf_v2.jsonl"
)

ART = (
    S5 / "artifacts/fullpdf_v2_independent_formal"
)

SEALED = (
    ART / "decision_ledger_sealed.jsonl"
)

V3_LOG = (
    ART
    / "checker2_superseding_receiver_binding_checkerfix_"
      "pretruthfix_v3.log"
)

V3_POSTRUN = (
    ART / "superseding_formal_v3_postrun_audit.json"
)

TRUTH_DIAGNOSTIC = (
    ART
    / "truth_candidate_postseal_diagnostic_v1/"
      "truth_candidate_postseal_diagnostic_v1.json"
)

FINAL_TRUTH = (
    S5
    / "artifacts/fullpdf_v2_repair/"
      "evaluator_truth_binding_fullpdf_v2.jsonl"
)

OUTDIR = (
    ART / "evaluator_truth_binding_materialization_v1"
)

PROBE_REPORT = (
    OUTDIR / "schema_probe_report.json"
)

AMENDMENT = (
    S5
    / "configs/"
      "stage5_fullpdf_v2_evaluator_truth_binding_amendment_v1.json"
)

SEAL = (
    OUTDIR
    / "stage5_fullpdf_v2_evaluator_truth_binding_seal_v1.json"
)

REPORT = (
    S5
    / "reports/"
      "stage5_fullpdf_v2_evaluator_truth_binding_materialization_v1.json"
)

EXPECTED = {
    HISTORICAL:
        "b1276eefd51444e0e648794d8bbb61ef829cc49bd1a09ba68ff79d9b61c8a9a5",

    CHECKER:
        "342344902e6b00cb841036dcdc0e54f1e1ba396344fa602864521890f5eb5da5",

    CHECKER2:
        "5e77d678af188d3f0b34b84ab53ddc78fc1b2921c6bea573dd9722c439c63599",

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

    BINDING:
        "5a5516ee050d2fd1c0a571193007650e75f23836b2734ee3df1950cc04f2e1c8",

    SEALED:
        "751703975b3fc837cc1cf660cf1321db5768323d6cf2fbef524d05a3dbb48af4",

    V3_LOG:
        "56f3c3c6cbade9546b1792b7a79735f02e95f96a1f956ae9a6b5d927f38dbbc1",

    V3_POSTRUN:
        "8ae664005203cf80175f0da0afcd01431b6dac6e83c1c1026e8c66878366ab67",

    TRUTH_DIAGNOSTIC:
        "e94dbb37fc115f28dbea07703ebc9a9de8a3068d7859fa22d5d58943590e2ebf",
}


def fail(msg):
    raise RuntimeError(
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


def canonical(obj):
    return (
        json.dumps(
            obj,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows):
    return (
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        )
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
        os.fsync(
            f.fileno()
        )

    os.replace(
        tmp,
        path,
    )


def write_once_json(path, obj):
    if path.exists():
        fail(
            f"write-once artifact exists: {path}"
        )

    atomic_write(
        path,
        canonical(obj),
    )

    os.chmod(
        path,
        0o444,
    )

    return sha(path)


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        obj = json.load(f)

    return obj


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
            text = line.strip()

            if not text:
                continue

            obj = json.loads(text)

            if not isinstance(
                obj,
                dict,
            ):
                fail(
                    f"non-object JSONL row: "
                    f"{path}:{lineno}"
                )

            rows.append(obj)

    return rows


def import_module(path, name):
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

    mod = (
        importlib.util
        .module_from_spec(spec)
    )

    sys.modules[name] = mod

    spec.loader.exec_module(
        mod
    )

    return mod


def field(obj, *names):
    for name in names:
        if isinstance(
            obj,
            Mapping,
        ):
            if name in obj:
                return obj[name]

        if hasattr(
            obj,
            name,
        ):
            return getattr(
                obj,
                name,
            )

    return None


def target_names(node):
    if isinstance(
        node,
        ast.Name,
    ):
        return {
            node.id
        }

    if isinstance(
        node,
        (
            ast.Tuple,
            ast.List,
        ),
    ):
        out = set()

        for item in node.elts:
            out.update(
                target_names(item)
            )

        return out

    return set()


def bind_target(
    node,
    value,
    output,
):
    if isinstance(
        node,
        ast.Name,
    ):
        output[
            node.id
        ] = value

        return

    if isinstance(
        node,
        (
            ast.Tuple,
            ast.List,
        ),
    ):
        values = list(
            value
        )

        if len(values) != len(
            node.elts
        ):
            fail(
                "tuple assignment arity "
                "changed in historical runner"
            )

        for child, child_value in zip(
            node.elts,
            values,
        ):
            bind_target(
                child,
                child_value,
                output,
            )

        return

    fail(
        "unsupported config-assignment target"
    )


def resolve_historical_configs(
    mod,
):
    source = HISTORICAL.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    main = None

    for node in tree.body:
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name == "main"
        ):
            main = node
            break

    if main is None:
        fail(
            "historical main() missing"
        )

    hits = []

    for node in ast.walk(
        main
    ):
        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        for target in node.targets:
            names = target_names(
                target
            )

            if {
                "clean_config",
                "degraded_config",
            }.issubset(
                names
            ):
                hits.append(
                    (
                        node,
                        target,
                    )
                )

    if len(hits) != 1:
        fail(
            "expected exactly one historical "
            "clean/degraded tuple assignment; "
            f"found {len(hits)}"
        )

    node, target = hits[0]

    expression = ast.Expression(
        body=node.value
    )

    ast.fix_missing_locations(
        expression
    )

    value = eval(
        compile(
            expression,
            str(HISTORICAL),
            "eval",
        ),
        mod.__dict__,
    )

    bound = {}

    bind_target(
        target,
        value,
        bound,
    )

    if (
        "clean_config" not in bound
        or "degraded_config"
        not in bound
    ):
        fail(
            "historical config tuple did "
            "not resolve required values"
        )

    return (
        bound["clean_config"],
        bound["degraded_config"],
        node.lineno,
    )


def read_typed_validation(
    mod,
):
    fn = getattr(
        mod,
        "read_validation_manifest",
        None,
    )

    if not callable(fn):
        fail(
            "historical module lacks "
            "read_validation_manifest"
        )

    errors = []

    for args in (
        (),
        (
            VALIDATION_MANIFEST,
        ),
    ):
        try:
            rows = list(
                fn(*args)
            )

            if rows:
                break

        except TypeError as exc:
            errors.append(
                repr(exc)
            )

    else:
        fail(
            "could not load canonical typed "
            "validation manifest: "
            + " | ".join(errors)
        )

    if len(rows) != 44097:
        fail(
            "typed validation row count changed: "
            f"{len(rows)}"
        )

    result = {}

    for row in rows:
        sid = field(
            row,
            "scenario_id",
            "scenario",
        )

        if sid is None:
            fail(
                "typed validation row lacks "
                "scenario_id"
            )

        key = str(sid)

        if key in result:
            fail(
                f"duplicate validation scenario: {key}"
            )

        result[key] = row

    return result


def normalizable(obj):
    if isinstance(
        obj,
        np.ndarray,
    ):
        return normalizable(
            obj.tolist()
        )

    if isinstance(
        obj,
        np.generic,
    ):
        return obj.item()

    if isinstance(
        obj,
        Mapping,
    ):
        return {
            repr(key):
                normalizable(value)
            for key, value in sorted(
                obj.items(),
                key=lambda kv:
                    repr(kv[0]),
            )
        }

    if isinstance(
        obj,
        (
            list,
            tuple,
        ),
    ):
        return [
            normalizable(x)
            for x in obj
        ]

    if isinstance(
        obj,
        (
            str,
            int,
            float,
            bool,
        ),
    ) or obj is None:
        return obj

    if hasattr(
        obj,
        "__dict__",
    ):
        return normalizable(
            vars(obj)
        )

    return repr(obj)


def object_hash(obj):
    payload = json.dumps(
        normalizable(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def identity_row(record):
    pid = record[
        "prediction_id"
    ]

    return {
        "scenario_id":
            record[
                "scenario_id"
            ],
        "prediction_id":
            pid,
        "receiver_prediction_id":
            pid,
        "selected_receiver_prediction_id":
            pid,
        "valid_future":
            record[
                "valid_future"
            ],
    }


MODE_STYLES = [
    {
        "centroid": "centroid",
        "known": "known",
        "uncertain": "uncertain",
    },
    {
        "centroid": "centroid_baseline",
        "known": "known_receiver_offset",
        "uncertain": "uncertain_receiver_offset",
    },
    {
        "centroid": "centroid_truth",
        "known": "known_truth",
        "uncertain": "uncertain_truth",
    },
]


def mode_map(
    record,
    aliases,
    *,
    horizon_wrapper=None,
    explicit_invalid=False,
):
    output = {}

    horizons = [
        str(float(x))
        for x in record[
            "horizons"
        ]
    ]

    for canonical_mode in (
        "centroid",
        "known",
        "uncertain",
    ):
        values = {}

        source = record[
            "truth"
        ][
            canonical_mode
        ]

        for h in horizons:
            if h in source:
                values[h] = source[h]

            elif explicit_invalid:
                values[h] = {
                    "valid":
                        False,
                    "horizon_s":
                        float(h),
                }

        if horizon_wrapper is None:
            payload = values
        else:
            payload = {
                horizon_wrapper:
                    values
            }

        output[
            aliases[
                canonical_mode
            ]
        ] = payload

    return output


def nested_layout(
    canonical_rows,
    wrapper,
    aliases,
    horizon_wrapper,
    explicit_invalid,
):
    rows = []

    for record in canonical_rows:
        row = identity_row(
            record
        )

        payload = mode_map(
            record,
            aliases,
            horizon_wrapper=(
                horizon_wrapper
            ),
            explicit_invalid=(
                explicit_invalid
            ),
        )

        if wrapper is None:
            row.update(
                payload
            )
        else:
            row[
                wrapper
            ] = payload

        rows.append(
            row
        )

    return rows


def list_layout(
    canonical_rows,
    wrapper,
    mode_field,
):
    rows = []

    for record in canonical_rows:
        row = identity_row(
            record
        )

        points = []

        for mode in (
            "centroid",
            "known",
            "uncertain",
        ):
            for h, point in sorted(
                record[
                    "truth"
                ][mode].items(),
                key=lambda kv:
                    float(kv[0]),
            ):
                points.append(
                    {
                        mode_field:
                            mode,
                        "horizon_s":
                            float(h),
                        **point,
                    }
                )

        row[
            wrapper
        ] = points

        rows.append(
            row
        )

    return rows


def main():
    print("=" * 76)
    print(
        "STAGE5 EVALUATOR-TRUTH "
        "MATERIALIZATION V1"
    )
    print(
        "POST-SEAL ONLY | NO METRICS | "
        "NO CONTROLLER REEXECUTION"
    )
    print("=" * 76)

    # --------------------------------------------------------
    # A. Immutable V3/protected chain.
    # --------------------------------------------------------

    for path, expected in (
        EXPECTED.items()
    ):
        if not path.is_file():
            fail(
                f"missing protected artifact: {path}"
            )

        actual = sha(
            path
        )

        if actual != expected:
            fail(
                "protected SHA mismatch:\n"
                f"{path}\n"
                f"actual   = {actual}\n"
                f"expected = {expected}"
            )

    mode = stat.S_IMODE(
        SEALED.stat().st_mode
    )

    if mode & 0o222:
        fail(
            "V3 sealed decision ledger "
            "is writable"
        )

    if (
        FINAL_TRUTH.exists()
        or AMENDMENT.exists()
        or SEAL.exists()
        or REPORT.exists()
    ):
        fail(
            "truth-binding materialization "
            "V1 already exists"
        )

    if (
        shutil.disk_usage(
            ROOT
        ).free
        < 250 * 1024**3
    ):
        fail(
            "free-space reserve <250 GiB"
        )

    diagnostic = load_json(
        TRUTH_DIAGNOSTIC
    )

    if (
        diagnostic.get(
            "status"
        )
        != "NO_EXISTING_VALID_BINDING"
    ):
        fail(
            "post-seal diagnostic no longer "
            "states NO_EXISTING_VALID_BINDING"
        )

    # --------------------------------------------------------
    # B. V3 decision identity.
    # --------------------------------------------------------

    sealed_rows = load_jsonl(
        SEALED
    )

    if len(
        sealed_rows
    ) != 17280:
        fail(
            "V3 sealed ledger row count changed"
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
        for row in sealed_rows
    }

    if len(
        decision_keys
    ) != 120:
        fail(
            "V3 unique decision-key count "
            "is not 120"
        )

    # --------------------------------------------------------
    # C. Frozen formal population + selected receiver binding.
    # --------------------------------------------------------

    formal_rows = load_jsonl(
        FORMAL_MANIFEST
    )

    if len(
        formal_rows
    ) != 120:
        fail(
            "Stage3 formal manifest count "
            "is not 120"
        )

    formal_by_sid = {}

    for row in formal_rows:
        sid = str(
            row[
                "scenario_id"
            ]
        )

        if sid in formal_by_sid:
            fail(
                f"duplicate formal scenario {sid}"
            )

        formal_by_sid[
            sid
        ] = row

    decision_sids = {
        sid
        for sid, _
        in decision_keys
    }

    if (
        set(
            formal_by_sid
        )
        != decision_sids
    ):
        fail(
            "Stage3 formal population differs "
            "from V3 decision scenes"
        )

    binding_rows = load_jsonl(
        BINDING
    )

    binding_keys = set()

    selected_pid_by_sid = {}

    for row in binding_rows:
        if (
            row.get(
                "selected_receiver"
            )
            is not True
        ):
            continue

        sid = str(
            row[
                "scenario_id"
            ]
        )

        pid = str(
            row.get(
                "selected_receiver_prediction_id",
                row[
                    "prediction_id"
                ],
            )
        )

        if sid in selected_pid_by_sid:
            fail(
                f"duplicate receiver binding scene: {sid}"
            )

        selected_pid_by_sid[
            sid
        ] = pid

        binding_keys.add(
            (
                sid,
                pid,
            )
        )

    if binding_keys != decision_keys:
        fail(
            "receiver binding keys differ "
            "from sealed V3 decision keys"
        )

    # --------------------------------------------------------
    # D. Import historical implementation and repaired checker
    #    as libraries. Neither main() is called.
    # --------------------------------------------------------

    historical = import_module(
        HISTORICAL,
        "stage5_historical_truth_source_v1",
    )

    checker = import_module(
        CHECKER,
        "stage5_formal_truth_resolver_v1",
    )

    for name in (
        "build_real_causal_inputs",
        "causal_actor_bridge",
        "read_motion_scenario",
        "transform_h0",
        "future_center",
        "future_heading",
        "rotate_offset",
        "spherical",
        "stable_seed",
        "receiver_offset",
    ):
        if not callable(
            getattr(
                historical,
                name,
                None,
            )
        ):
            fail(
                f"historical helper missing: {name}"
            )

    for name in (
        "candidate_v2_paths",
        "resolve_truth_binding",
    ):
        if not callable(
            getattr(
                checker,
                name,
                None,
            )
        ):
            fail(
                f"checker helper missing: {name}"
            )

    clean_config, degraded_config, config_line = (
        resolve_historical_configs(
            historical
        )
    )

    validation_by_sid = (
        read_typed_validation(
            historical
        )
    )

    receiver_parameters = (
        historical.receiver_offset()
    )

    if len(
        receiver_parameters
    ) != 3:
        fail(
            "historical receiver_offset() "
            "return arity changed"
        )

    mean_offset = np.asarray(
        receiver_parameters[0],
        dtype=float,
    )

    cov_offset = np.asarray(
        receiver_parameters[1],
        dtype=float,
    )

    offset_source = str(
        receiver_parameters[2]
    )

    if (
        mean_offset.shape != (3,)
        or cov_offset.shape != (3, 3)
        or not np.all(
            np.isfinite(
                mean_offset
            )
        )
        or not np.all(
            np.isfinite(
                cov_offset
            )
        )
    ):
        fail(
            "receiver offset parameters invalid"
        )

    geometries = list(
        historical.GEOMETRIES
    )

    horizons = [
        float(x)
        for x in historical.HORIZONS
    ]

    offsets = [
        int(x)
        for x in historical.OFFSETS
    ]

    if geometries != [
        "centroid",
        "known",
        "uncertain",
    ]:
        fail(
            f"historical geometries changed: "
            f"{geometries}"
        )

    if horizons != [
        0.1,
        0.3,
        0.5,
        1.0,
    ]:
        fail(
            f"historical horizons changed: "
            f"{horizons}"
        )

    if len(offsets) != 4:
        fail(
            "historical future offsets changed"
        )

    # --------------------------------------------------------
    # E. POST-SEAL evaluator truth construction.
    #
    #    Frozen selected PID is used. Receiver selection is
    #    NOT executed again.
    # --------------------------------------------------------

    canonical_rows = []

    valid_counts = {
        str(float(h)):
            0
        for h in horizons
    }

    for sid, pid in sorted(
        decision_keys
    ):
        formal = formal_by_sid[
            sid
        ]

        validation_row = (
            validation_by_sid.get(
                sid
            )
        )

        if validation_row is None:
            fail(
                f"validation scene absent: {sid}"
            )

        compact_offset = field(
            formal,
            "compact_record_offset",
        )

        if compact_offset is None:
            compact_offset = field(
                validation_row,
                "compact_record_offset",
                "record_offset",
            )

        if compact_offset is None:
            fail(
                f"compact record offset "
                f"unresolved: {sid}"
            )

        paired_root = getattr(
            historical,
            "PAIRED_ROOT",
            ROOT
            / "data/paired_womd_lidar_v1_3_0",
        )

        scenario = (
            historical.read_motion_scenario(
                validation_row,
                paired_root=paired_root,
                compact_record_offset=int(
                    compact_offset
                ),
            )
        )

        current_index = int(
            scenario.current_time_index
        )

        if current_index != 10:
            fail(
                f"{sid}: current_time_index "
                f"changed to {current_index}"
            )

        built = (
            historical.build_real_causal_inputs(
                scenario,
                clean_config=clean_config,
                degraded_config=degraded_config,
            )
        )

        histories, actors, matches = (
            historical.causal_actor_bridge(
                built,
                scenario,
            )
        )

        matching_tracks = []

        for hi, history in enumerate(
            histories
        ):
            if hi not in matches:
                continue

            if str(
                history.prediction_id
            ) != pid:
                continue

            ai, _distance = matches[
                hi
            ]

            actor, _ = actors[
                ai
            ]

            matching_tracks.append(
                int(
                    actor.track_index
                )
            )

        matching_tracks = sorted(
            set(
                matching_tracks
            )
        )

        if len(
            matching_tracks
        ) != 1:
            fail(
                f"{sid}/{pid}: expected one "
                f"causally associated track; "
                f"got {matching_tracks}"
            )

        track_index = (
            matching_tracks[0]
        )

        track = scenario.tracks[
            track_index
        ]

        if int(
            track.object_type
        ) != 1:
            fail(
                f"{sid}/{pid}: selected "
                "receiver is not TYPE_VEHICLE"
            )

        # ----------------------------------------------------
        # EVALUATOR-ONLY FUTURE ACCESS STARTS HERE.
        # V3 decision ledger is already sealed.
        # ----------------------------------------------------

        T = historical.transform_h0(
            built,
            current_index,
        )

        rng = np.random.default_rng(
            historical.stable_seed(
                sid,
                "constructed_receiver_truth",
            )
        )

        uncertain_truth_offset = (
            rng.multivariate_normal(
                mean_offset,
                cov_offset,
                check_valid="raise",
            )
        )

        truth = {
            geometry: {}
            for geometry in geometries
        }

        valid_future = {}

        for horizon, offset in zip(
            horizons,
            offsets,
        ):
            hkey = str(
                float(
                    horizon
                )
            )

            state_index = (
                current_index
                + int(
                    offset
                )
            )

            valid = (
                state_index
                < len(
                    track.states
                )
                and bool(
                    track.states[
                        state_index
                    ].valid
                )
            )

            valid_future[
                hkey
            ] = bool(
                valid
            )

            if not valid:
                continue

            valid_counts[
                hkey
            ] += 1

            state = track.states[
                state_index
            ]

            actor_center = np.asarray(
                historical.future_center(
                    state,
                    T,
                ),
                dtype=float,
            )

            actor_heading = float(
                historical.future_heading(
                    state,
                    T,
                )
            )

            truth_offsets = {
                "centroid":
                    np.zeros(
                        3,
                        dtype=float,
                    ),
                "known":
                    mean_offset,
                "uncertain":
                    uncertain_truth_offset,
            }

            for geometry in geometries:
                rotated = np.asarray(
                    historical.rotate_offset(
                        truth_offsets[
                            geometry
                        ],
                        actor_heading,
                    ),
                    dtype=float,
                )

                position = (
                    actor_center
                    + rotated
                )

                (
                    radius,
                    azimuth,
                    elevation,
                ) = historical.spherical(
                    position
                )

                values = [
                    float(x)
                    for x in position
                ]

                truth[
                    geometry
                ][
                    hkey
                ] = {
                    "valid":
                        True,
                    "horizon_s":
                        float(
                            horizon
                        ),
                    "position_H0_m":
                        values,
                    "position_h0_m":
                        values,
                    "receiver_position_H0_m":
                        values,
                    "range_m":
                        float(
                            radius
                        ),
                    "distance_m":
                        float(
                            radius
                        ),
                    "azimuth_rad":
                        float(
                            azimuth
                        ),
                    "elevation_rad":
                        float(
                            elevation
                        ),
                }

        canonical_rows.append(
            {
                "scenario_id":
                    sid,
                "prediction_id":
                    pid,
                "track_index_evaluator_only":
                    track_index,
                "horizons":
                    horizons,
                "valid_future":
                    valid_future,
                "truth":
                    truth,
            }
        )

    if len(
        canonical_rows
    ) != 120:
        fail(
            "truth materialization did "
            "not produce 120 receiver rows"
        )

    # --------------------------------------------------------
    # F. Confirm no truth candidate existed before commit.
    # --------------------------------------------------------

    original_candidate_fn = (
        checker.candidate_v2_paths
    )

    try:
        before_candidates = list(
            original_candidate_fn(
                truth_phase=True
            )
        )
    except TypeError:
        before_candidates = list(
            original_candidate_fn(
                True
            )
        )

    before_candidates = [
        Path(x)
        for x in before_candidates
        if Path(x).is_file()
    ]

    if before_candidates:
        fail(
            "truth candidates appeared before "
            "materialization:\n"
            + "\n".join(
                str(x)
                for x
                in before_candidates
            )
        )

    # --------------------------------------------------------
    # G. Parser-only schema compatibility search.
    #
    # No metrics are called. Same truth values in every trial.
    # Fixed ordering; first parser-valid layout wins.
    # --------------------------------------------------------

    checker_source = (
        CHECKER.read_text(
            encoding="utf-8"
        )
    )

    checker_tree = ast.parse(
        checker_source
    )

    dynamic_wrappers = []

    for node in ast.walk(
        checker_tree
    ):
        if (
            isinstance(
                node,
                ast.Constant,
            )
            and isinstance(
                node.value,
                str,
            )
        ):
            value = node.value

            if (
                re.fullmatch(
                    r"[A-Za-z][A-Za-z0-9_]{1,60}",
                    value,
                )
                and (
                    "truth"
                    in value.lower()
                    or "evaluator"
                    in value.lower()
                )
            ):
                dynamic_wrappers.append(
                    value
                )

    wrappers = [
        None,
        "truth",
        "future_truth",
        "evaluator_truth",
        "receiver_truth",
        "realized_truth",
        "ground_truth",
        "truth_by_mode",
        "receiver_future_truth",
        "formal_truth",
    ]

    for value in sorted(
        set(
            dynamic_wrappers
        )
    ):
        if value not in wrappers:
            wrappers.append(
                value
            )

    layout_specs = []

    # Exact deterministic priority.
    for explicit_invalid in (
        False,
        True,
    ):
        for aliases in MODE_STYLES:
            for horizon_wrapper in (
                None,
                "horizons",
                "future",
            ):
                for wrapper in wrappers:
                    layout_specs.append(
                        {
                            "kind":
                                "nested",
                            "wrapper":
                                wrapper,
                            "aliases":
                                aliases,
                            "horizon_wrapper":
                                horizon_wrapper,
                            "explicit_invalid":
                                explicit_invalid,
                        }
                    )

    for wrapper in (
        "truth_points",
        "future_truth",
        "evaluator_truth",
        "truth",
    ):
        for mode_field in (
            "receiver_geometry_mode",
            "geometry",
            "mode",
        ):
            layout_specs.append(
                {
                    "kind":
                        "list",
                    "wrapper":
                        wrapper,
                    "mode_field":
                        mode_field,
                }
            )

    probe_results = []

    chosen_spec = None
    chosen_rows = None
    chosen_map_hash = None

    with tempfile.TemporaryDirectory(
        prefix="stage5_truth_schema_probe_"
    ) as temp_name:
        temp_dir = Path(
            temp_name
        )

        for index, spec in enumerate(
            layout_specs
        ):
            if spec[
                "kind"
            ] == "nested":
                rows = nested_layout(
                    canonical_rows,
                    spec[
                        "wrapper"
                    ],
                    spec[
                        "aliases"
                    ],
                    spec[
                        "horizon_wrapper"
                    ],
                    spec[
                        "explicit_invalid"
                    ],
                )
            else:
                rows = list_layout(
                    canonical_rows,
                    spec[
                        "wrapper"
                    ],
                    spec[
                        "mode_field"
                    ],
                )

            candidate = (
                temp_dir
                / (
                    f"{index:03d}_"
                    "evaluator_truth_binding.jsonl"
                )
            )

            candidate.write_bytes(
                jsonl_bytes(
                    rows
                )
            )

            def only_candidate(
                *args,
                _candidate=candidate,
                **kwargs,
            ):
                return [
                    _candidate
                ]

            checker.candidate_v2_paths = (
                only_candidate
            )

            if hasattr(
                checker,
                "DECISION_LEDGER_SEALED",
            ):
                checker.DECISION_LEDGER_SEALED = (
                    True
                )

            if hasattr(
                checker,
                "FORMAL_TRUTH_OPENED",
            ):
                checker.FORMAL_TRUTH_OPENED = (
                    False
                )

            try:
                (
                    resolved_path,
                    truth_map,
                ) = (
                    checker
                    .resolve_truth_binding(
                        decision_keys
                    )
                )

                map_hash = object_hash(
                    truth_map
                )

                probe_results.append(
                    {
                        "index":
                            index,
                        "status":
                            "PASS",
                        "spec":
                            spec,
                        "truth_map_hash":
                            map_hash,
                    }
                )

                chosen_spec = spec
                chosen_rows = rows
                chosen_map_hash = (
                    map_hash
                )

                break

            except BaseException as exc:
                probe_results.append(
                    {
                        "index":
                            index,
                        "status":
                            "FAIL",
                        "spec":
                            spec,
                        "exception":
                            type(
                                exc
                            ).__name__,
                        "message":
                            str(
                                exc
                            )[:300],
                    }
                )

    checker.candidate_v2_paths = (
        original_candidate_fn
    )

    if (
        chosen_spec is None
        or chosen_rows is None
    ):
        OUTDIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_once_json(
            PROBE_REPORT,
            {
                "stage": 5,
                "status":
                    "FAIL_NO_RESOLVER_COMPATIBLE_SCHEMA",
                "probe_count":
                    len(
                        probe_results
                    ),
                "probe_results":
                    probe_results,
                "formal_metrics_computed":
                    False,
                "Stage6_allowed":
                    False,
            },
        )

        fail(
            "unchanged resolver rejected "
            "all deterministic interface layouts"
        )

    # --------------------------------------------------------
    # H. Commit exactly ONE evaluator-truth binding.
    # --------------------------------------------------------

    final_data = jsonl_bytes(
        chosen_rows
    )

    atomic_write(
        FINAL_TRUTH,
        final_data,
    )

    os.chmod(
        FINAL_TRUTH,
        0o444,
    )

    final_truth_sha = sha(
        FINAL_TRUTH
    )

    # --------------------------------------------------------
    # I. Validate with NORMAL, unchanged candidate discovery.
    # --------------------------------------------------------

    if hasattr(
        checker,
        "DECISION_LEDGER_SEALED",
    ):
        checker.DECISION_LEDGER_SEALED = (
            True
        )

    if hasattr(
        checker,
        "FORMAL_TRUTH_OPENED",
    ):
        checker.FORMAL_TRUTH_OPENED = (
            False
        )

    (
        resolved_path,
        resolved_truth_map,
    ) = checker.resolve_truth_binding(
        decision_keys
    )

    resolved_path = Path(
        resolved_path
    ).resolve()

    if (
        resolved_path
        != FINAL_TRUTH.resolve()
    ):
        fail(
            "normal resolver selected "
            f"unexpected truth file: "
            f"{resolved_path}"
        )

    normal_map_hash = object_hash(
        resolved_truth_map
    )

    if (
        normal_map_hash
        != chosen_map_hash
    ):
        fail(
            "normal resolver truth map differs "
            "from schema-probe truth map"
        )

    # --------------------------------------------------------
    # J. Protected artifacts must remain unchanged.
    # --------------------------------------------------------

    for path, expected in (
        EXPECTED.items()
    ):
        actual = sha(
            path
        )

        if actual != expected:
            fail(
                f"protected artifact changed: {path}"
            )

    if (
        stat.S_IMODE(
            SEALED.stat().st_mode
        )
        & 0o222
    ):
        fail(
            "V3 sealed decision ledger became writable"
        )

    # --------------------------------------------------------
    # K. Freeze provenance.
    # --------------------------------------------------------

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    probe_sha = write_once_json(
        PROBE_REPORT,
        {
            "stage": 5,
            "version":
                "fullpdf_v2",
            "status":
                "PASS_RESOLVER_COMPATIBLE_SCHEMA",
            "selection_semantics":
                "first parser-valid interface in "
                "fixed deterministic priority; "
                "no scientific metric used",
            "chosen_spec":
                chosen_spec,
            "probe_count_until_pass":
                len(
                    probe_results
                ),
            "probe_results":
                probe_results,
            "truth_map_hash":
                normal_map_hash,
            "formal_metrics_computed":
                False,
            "controller_reexecuted":
                False,
            "Stage6_allowed":
                False,
        },
    )

    formal_manifest_report_sha = (
        sha(
            FORMAL_MANIFEST_REPORT
        )
        if FORMAL_MANIFEST_REPORT.is_file()
        else None
    )

    amendment = {
        "stage": 5,
        "version":
            "fullpdf_v2",
        "status":
            "FROZEN_POSTSEAL_EVALUATOR_TRUTH_BINDING",
        "scope":
            "evaluator truth only",
        "decision_ledger": {
            "path":
                str(
                    SEALED
                ),
            "sha256":
                sha(
                    SEALED
                ),
            "rows":
                len(
                    sealed_rows
                ),
            "unique_receiver_keys":
                len(
                    decision_keys
                ),
            "read_only":
                True,
        },
        "truth_binding": {
            "path":
                str(
                    FINAL_TRUTH
                ),
            "sha256":
                final_truth_sha,
            "rows":
                len(
                    chosen_rows
                ),
            "normal_resolver_unique":
                True,
            "truth_map_hash":
                normal_map_hash,
            "interface_spec":
                chosen_spec,
        },
        "truth_construction": {
            "historical_runner":
                str(
                    HISTORICAL
                ),
            "historical_runner_sha256":
                sha(
                    HISTORICAL
                ),
            "historical_config_assignment_line":
                config_line,
            "receiver_offset_source":
                offset_source,
            "geometries":
                geometries,
            "horizons_s":
                horizons,
            "future_offsets":
                offsets,
            "uncertain_truth_seed":
                'stable_seed(scenario_id, '
                '"constructed_receiver_truth")',
            "selected_receiver_source":
                "frozen primary_receiver_binding_fullpdf_v2",
            "receiver_selection_reexecuted":
                False,
            "tracks_to_predict_receiver_selection":
                False,
        },
        "population": {
            "formal_manifest":
                str(
                    FORMAL_MANIFEST
                ),
            "formal_manifest_sha256":
                sha(
                    FORMAL_MANIFEST
                ),
            "formal_manifest_report_sha256":
                formal_manifest_report_sha,
            "canonical_validation_manifest_sha256":
                sha(
                    VALIDATION_MANIFEST
                ),
            "scene_count":
                120,
            "valid_future_counts":
                valid_counts,
        },
        "schema_probe": {
            "report":
                str(
                    PROBE_REPORT
                ),
            "report_sha256":
                probe_sha,
            "scientific_metric_used":
                False,
        },
        "scientific_boundary": {
            "V3_decisions_already_sealed":
                True,
            "evaluator_truth_opened":
                True,
            "truth_opened_only_postseal":
                True,
            "formal_metrics_computed":
                False,
            "coverage_read":
                False,
            "BER_read":
                False,
            "effective_rate_read":
                False,
            "training":
                False,
            "retraining":
                False,
            "recalibration":
                False,
            "parameter_tuning":
                False,
            "post_outcome_tuning":
                False,
        },
        "Stage6_allowed":
            False,
    }

    amendment_sha = (
        write_once_json(
            AMENDMENT,
            amendment,
        )
    )

    seal_sha = write_once_json(
        SEAL,
        {
            "stage": 5,
            "version":
                "fullpdf_v2",
            "status":
                "SEALED_POST_V3_DECISION_"
                "EVALUATOR_TRUTH_BINDING_V1",
            "V3_decision_ledger_sha256":
                sha(
                    SEALED
                ),
            "evaluator_truth_binding_sha256":
                final_truth_sha,
            "formal_checker_sha256":
                sha(
                    CHECKER
                ),
            "checker2_sha256":
                sha(
                    CHECKER2
                ),
            "runtime_sha256":
                sha(
                    RUNTIME
                ),
            "protocol_sha256":
                sha(
                    PROTOCOL
                ),
            "Stage4_ledger_sha256":
                sha(
                    STAGE4_LEDGER
                ),
            "receiver_binding_sha256":
                sha(
                    BINDING
                ),
            "truth_binding_amendment_sha256":
                amendment_sha,
            "schema_probe_report_sha256":
                probe_sha,
            "formal_metrics_computed":
                False,
            "post_outcome_tuning":
                False,
            "Stage6_allowed":
                False,
        },
    )

    report_sha = write_once_json(
        REPORT,
        {
            **amendment,
            "amendment_sha256":
                amendment_sha,
            "seal_sha256":
                seal_sha,
            "status":
                "PASS_READY_FOR_FINAL_FORMAL_EVALUATION",
        },
    )

    print()
    print("=" * 76)
    print(
        "EVALUATOR-TRUTH BINDING MATERIALIZATION = PASS"
    )
    print("=" * 76)

    print(
        "V3_sealed_decision_sha256 =",
        sha(
            SEALED
        ),
    )

    print(
        "decision_rows =",
        len(
            sealed_rows
        ),
    )

    print(
        "receiver_keys =",
        len(
            decision_keys
        ),
    )

    print(
        "historical_truth_route = PASS"
    )

    print(
        "receiver_selection_reexecuted = false"
    )

    print(
        "chosen_parser_layout =",
        json.dumps(
            chosen_spec,
            sort_keys=True,
        ),
    )

    print(
        "evaluator_truth_rows =",
        len(
            chosen_rows
        ),
    )

    print(
        "normal_resolver_unique = true"
    )

    print(
        "evaluator_truth_sha256 =",
        final_truth_sha,
    )

    print(
        "truth_map_hash =",
        normal_map_hash,
    )

    print(
        "amendment_sha256 =",
        amendment_sha,
    )

    print(
        "truth_binding_seal_sha256 =",
        seal_sha,
    )

    print(
        "report_sha256 =",
        report_sha,
    )

    print(
        "formal_metrics_computed = false"
    )

    print(
        "post_outcome_tuning = false"
    )

    print(
        "Stage6_allowed = false"
    )

    print(
        "next = FINAL ONE-SHOT FORMAL EVALUATION"
    )

    print("=" * 76)


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print("=" * 76)
        print(
            "EVALUATOR-TRUTH BINDING "
            "MATERIALIZATION = FAIL-CLOSED"
        )
        print(
            "reason =",
            repr(
                exc
            ),
        )
        print(
            "formal_metrics_computed = false"
        )
        print(
            "Stage6_allowed = false"
        )
        print("=" * 76)
        raise
