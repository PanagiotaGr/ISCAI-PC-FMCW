#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path("/home/agni/waymo")
S9 = ROOT / "iscai_stage9"

for stage in (
    "iscai_stage1",
    "iscai_stage2",
    "iscai_stage3",
    "iscai_stage4",
    "iscai_stage5",
    "iscai_stage9",
):
    p = ROOT / stage / "src"
    if p.is_dir():
        sys.path.insert(0, str(p))

FORMAL_N = 26209
PREFIX_N = 388

FORMAL_IDS = (
    S9 / "artifacts/block91b5_opaque_scenario_id_cohort_freeze/"
    "stage9_FORMAL_scenario_ids.txt"
)

A_OLD = (
    S9 / "artifacts/"
    "block913a_v1_1_formal_c1_c2_prediction_side/"
    "scenario_cache"
)

A_NEW_ROOT = (
    S9 / "artifacts/"
    "block913a_v1_2_c1_c2_prediction_side_continuation"
)

A_NEW = A_NEW_ROOT / "scenario_cache_suffix"

A_MANIFEST = (
    A_NEW_ROOT /
    "formal_c1_c2_prediction_side_by_scenario.jsonl"
)

A_SEAL = (
    A_NEW_ROOT /
    "stage9_block913a_v1_2_continuation_seal.json"
)

BLOCKED_SEAL = (
    S9 / "artifacts/"
    "block913b_v1_2_R1_blocked_preoutcome_interface_inconsistency/"
    "stage9_block913b_v1_2_R1_blocked_preoutcome_interface_inconsistency_seal.json"
)

V12_912_SEAL = (
    S9 / "artifacts/"
    "block912_v1_2_c1_c2_primary_c3_secondary_scope_extension/"
    "stage9_block912_v1_2_c1_c2_primary_c3_secondary_scope_extension_seal.json"
)

V12_913_CONTRACT = (
    S9 / "configs/"
    "stage9_block913_v1_2_formal_c1_c2_primary_c3_secondary_execution_contract.json"
)

CARRIER = (
    S9 / "scripts/"
    "run_stage9_block913_v1_1_formal_womd_c1_c2_outcome_materialization.py"
)

OUT = (
    S9 / "artifacts/"
    "block913_v1_3_primary_pretruth_R4"
)

CACHE = OUT / "scenario_cache"

AMENDMENT = (
    OUT /
    "stage9_block913_v1_3_preoutcome_interface_amendment.json"
)

MANIFEST = (
    OUT /
    "stage9_block913_v1_3_pretruth_by_scenario.jsonl"
)

SUMMARY = (
    OUT /
    "stage9_block913_v1_3_pretruth_summary.json"
)

SEAL = (
    OUT /
    "stage9_block913_v1_3_pretruth_seal.json"
)

EXPECTED = {
    FORMAL_IDS:
        "7d3066f08c466cfd2cba64eecdd24474a7e83130a4e90d77e9cdee349e22f03f",

    A_MANIFEST:
        "bc6925256c42b6f67b0f2dcb075fa35a499b7ca15127071b6282d03b1a60e57b",

    A_SEAL:
        "a67b1ee879cdaf6dc4516834c439fbc0bdd78243281c30802f71d4b8ea0cea95",

    BLOCKED_SEAL:
        "337daee3bf4d3aae56144341e5e68aff12c150c14d5f0448c352a0e9932c2466",

    V12_912_SEAL:
        "eabde71a32a924bf2416c3605e8f7087c2bbfb9ad61ecf72ed3fefaf2a1ff9ac",

    V12_913_CONTRACT:
        "991349ff9029b197652411e7ac60767458debf15674fcd103bfa56cb16288001",

    CARRIER:
        "ac618eafeaa7884979834ef6dcce772bf8d24f59cb6632de993c3efb6bf76322",
}


class FailClosed(RuntimeError):
    pass


def req(x, msg):
    if not x:
        raise FailClosed(msg)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_bytes(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha(obj):
    return hashlib.sha256(
        canonical_bytes(obj)
    ).hexdigest()


def atomic_json(path: Path, obj):
    req(
        not path.exists(),
        f"refusing overwrite: {path}",
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".tmp.",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(fd, "wb") as f:
            f.write(
                json.dumps(
                    obj,
                    sort_keys=True,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            )
            f.write(b"\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)
        os.chmod(path, 0o444)

    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def a_path(ordinal, sid):
    base = A_OLD if ordinal < PREFIX_N else A_NEW
    return base / f"{ordinal:05d}_{sid}.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    req(
        spec is not None and spec.loader is not None,
        f"cannot load {path}",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def install_scipy_free_eval94(carrier):
    """
    Block9.4B imports scipy.optimize.minimize for CAL fitting.
    9.13 pretruth does not fit anything.  Load the exact source
    with only that unavailable optimizer dependency replaced by
    a fail-closed function.
    """
    path = carrier.BLOCK94B_CARRIER
    source = path.read_text(encoding="utf-8")

    needle = "from scipy.optimize import minimize"

    req(
        source.count(needle) == 1,
        "Block9.4B scipy import shape changed",
    )

    replacement = '''
def minimize(*args, **kwargs):
    raise RuntimeError(
        "scipy.optimize.minimize is forbidden in FORMAL pretruth"
    )
'''

    source = source.replace(
        needle,
        replacement,
        1,
    )

    mod = types.ModuleType(
        "_stage9_913_v13_eval94_scipy_free"
    )
    mod.__file__ = str(path)

    sys.modules[mod.__name__] = mod

    exec(
        compile(
            source,
            str(path),
            "exec",
        ),
        mod.__dict__,
    )

    original = carrier.load_module_from_path

    def loader(name, candidate):
        if (
            Path(candidate).resolve()
            ==
            Path(path).resolve()
        ):
            return mod

        return original(
            name,
            candidate,
        )

    carrier.load_module_from_path = loader


def current_histories(scene_inputs, current_index):
    out = []

    for h in tuple(scene_inputs.histories):
        latest = int(
            h.latest_observed_frame_index
        )

        if latest != current_index:
            continue

        steps = tuple(h.steps)

        req(
            0 <= latest < len(steps),
            "invalid current history index",
        )

        if bool(steps[latest].observed):
            out.append(h)

    out.sort(
        key=lambda h: str(h.prediction_id)
    )

    return out


def amended_receiver_projection(
    *,
    stack,
    s5,
    sup,
    scenario,
    built,
    current_index,
):
    """
    Stage9 9.13-v1.3 pre-outcome interface amendment.

    Full causally-current history universe is preserved.

    If a current history is deterministically matched by the exact
    frozen current-frame WOMD association:
        semantic class = static current track.object_type.

    If it is not matched:
        semantic class = OTHER.

    OTHER is a conservative ineligible class for the frozen
    nearest_causal_vehicle_ahead receiver policy.

    No future state, future validity, future trajectory,
    beam truth, critical outcome, or FORMAL metric is consumed.
    """

    scene_inputs = built["scene_inputs"]

    histories = tuple(
        scene_inputs.histories
    )

    history_by_id = {
        str(h.prediction_id): h
        for h in histories
    }

    req(
        len(history_by_id)
        ==
        len(histories),
        "Duplicate causal prediction IDs.",
    )

    current_ids = []

    for h in histories:

        latest = int(
            h.latest_observed_frame_index
        )

        if latest != int(current_index):
            continue

        steps = tuple(h.steps)

        req(
            0 <= current_index < len(steps),
            "current history index invalid",
        )

        if bool(
            steps[current_index].observed
        ):
            current_ids.append(
                str(h.prediction_id)
            )

    current_ids.sort()

    prediction_histories = []

    for pid in current_ids:

        h = history_by_id[pid]

        req(
            int(
                h.latest_observed_frame_index
            )
            ==
            int(current_index),
            f"Prediction {pid} is not current-available.",
        )

        steps = tuple(h.steps)

        req(
            (
                0
                <= current_index
                < len(steps)
                and
                bool(
                    steps[
                        current_index
                    ].observed
                )
            ),
            f"Prediction {pid} current step unavailable.",
        )

        prediction_histories.append(
            h
        )

    T_H0_from_W = (
        built["adapted"]
        .frames
        .T_H0_from_W
    )

    # Exact frozen current-frame matcher.
    truth_indices = tuple(
        sup._current_valid_truth_indices(
            scenario
        )
    )

    edges = []

    for history in prediction_histories:

        frame_index = int(
            history.latest_observed_frame_index
        )

        hx, hy, _ = (
            history.latest_position_H0_m
        )

        for truth_index in truth_indices:

            track = scenario.tracks[
                int(truth_index)
            ]

            if frame_index >= len(
                track.states
            ):
                continue

            state = track.states[
                frame_index
            ]

            if not bool(state.valid):
                continue

            tx, ty, _ = sup._point_H0(
                state,
                T_H0_from_W=
                    T_H0_from_W,
            )

            distance = math.hypot(
                float(tx) - float(hx),
                float(ty) - float(hy),
            )

            if distance <= 5.0:

                edges.append((
                    float(distance),
                    str(
                        history.prediction_id
                    ),
                    int(truth_index),
                    int(frame_index),
                ))

    edges.sort(
        key=lambda x:
            (x[0], x[1], x[2])
    )

    used_predictions = set()
    used_truth = set()
    matches = {}

    for (
        distance,
        pid,
        truth_index,
        frame_index,
    ) in edges:

        if (
            pid in used_predictions
            or
            truth_index in used_truth
        ):
            continue

        used_predictions.add(pid)
        used_truth.add(truth_index)

        matches[pid] = {
            "truth_track_index":
                int(truth_index),

            "historical_match_distance_m":
                float(distance),

            "historical_match_frame_index":
                int(frame_index),
        }

    matched_ids = set(matches)

    unmatched_ids = (
        set(current_ids)
        -
        matched_ids
    )

    rows = []
    provenance = {}

    for pid in current_ids:

        h = history_by_id[pid]

        if pid in matches:

            match = matches[pid]

            truth_index = int(
                match[
                    "truth_track_index"
                ]
            )

            track = scenario.tracks[
                truth_index
            ]

            semantic_class = str(
                sup._actor_class_name(
                    track.object_type
                )
            )

            provenance[pid] = {
                "bridge_status":
                    "MATCHED_CURRENT_WOMD_OBJECT_TYPE",

                "truth_track_index":
                    truth_index,

                "historical_match_distance_m":
                    float(
                        match[
                            "historical_match_distance_m"
                        ]
                    ),

                "historical_match_frame_index":
                    int(
                        match[
                            "historical_match_frame_index"
                        ]
                    ),

                "semantic_class":
                    semantic_class,
            }

        else:

            # Explicit v1.3 pre-outcome conservative fallback.
            # Keep the history in the denominator, but do not
            # fabricate VEHICLE/PED/CYCLIST identity.
            semantic_class = "OTHER"

            provenance[pid] = {
                "bridge_status":
                    "UNMATCHED_CURRENT_HISTORY_CONSERVATIVE_OTHER",

                "truth_track_index":
                    None,

                "historical_match_distance_m":
                    None,

                "historical_match_frame_index":
                    None,

                "semantic_class":
                    "OTHER",
            }

        rows.append(
            s5.ControllerSample(
                prediction_id=pid,
                semantic_class=
                    semantic_class,
                model_input=None,
                history=h,
                original_sample_index=None,
                current_available=True,
            )
        )

    rows.sort(
        key=lambda row:
            str(row.prediction_id)
    )

    # Full current candidate universe is retained.
    req(
        len(rows)
        ==
        len(current_ids),
        "receiver candidate universe changed",
    )

    projection = {
        "rows":
            rows,

        "causal_history_count":
            len(histories),

        "current_available_history_count":
            len(current_ids),

        "causal_class_observation_count":
            0,

        "class_resolved_candidate_count":
            len(matches),

        "unresolved_history_count":
            len(unmatched_ids),

        "unknown_class_count":
            len(unmatched_ids),

        "current_only_supervision_match_count":
            len(matches),

        "global_current_association_count":
            len(matches),

        "global_current_truth_candidate_count":
            len(truth_indices),

        "global_current_association_edge_count":
            len(edges),

        "conservative_OTHER_count":
            len(unmatched_ids),

        "_current_only_match_by_prediction":
            provenance,

        "future_GT_access":
            False,

        "semantic_class_bridge":
            (
                "V1_3_CURRENT_WOMD_OBJECT_TYPE_"
                "ELSE_CONSERVATIVE_OTHER"
            ),
    }

    result, selected = (
        s5.choose_primary_receiver(
            projection,
            stack["parameters"][
                "receiver_config"
            ],
        )
    )

    return projection, result, selected

def h1_and_baselines(
    *,
    carrier,
    stack,
    scenario,
    built,
    selected,
    actor,
):
    s5 = stack["s5"]
    sup = stack["sup"]
    ni = stack["ni"]
    h1 = stack["h1"]

    scene_inputs = built["scene_inputs"]
    T = built["adapted"].frames.T_H0_from_W

    map_index = sup.build_static_map_index_H0(
        scenario,
        T_H0_from_W=T,
    )

    ctx = sup.map_context_summary(
        map_index,
        target_position_H0_m=(
            selected.history.latest_position_H0_m
        ),
    )

    payload = ni.build_model_input_payload(
        scene_inputs,
        prediction_id=str(
            selected.prediction_id
        ),
        map_context=ctx,
    )

    req(
        payload.sha256()
        ==
        str(actor["payload_sha256"]),
        "selected receiver payload differs from sealed 9.13A",
    )

    selected_with_payload = (
        s5.ControllerSample(
            prediction_id=str(
                selected.prediction_id
            ),
            semantic_class=str(
                selected.semantic_class
            ),
            model_input=payload,
            history=selected.history,
            original_sample_index=0,
            current_available=True,
        )
    )

    (
        trajectory_mean,
        raw_cov,
        cal_cov,
        runtime_s,
    ) = h1.paired_receiver_trajectory(
        s5,
        stack["runtime"],
        selected_with_payload,
    )

    trajectory_mean = np.asarray(
        trajectory_mean,
        dtype=np.float64,
    )

    raw_cov = np.asarray(
        raw_cov,
        dtype=np.float64,
    )

    cal_cov = np.asarray(
        cal_cov,
        dtype=np.float64,
    )

    req(
        trajectory_mean.shape == (4, 3),
        "H1 trajectory mean shape drift",
    )

    req(
        raw_cov.shape == (4, 3, 3),
        "H1 raw covariance shape drift",
    )

    req(
        cal_cov.shape == (4, 3, 3),
        "H1 calibrated covariance shape drift",
    )

    # Final 1-s CAL arm must reproduce the already sealed 9.13A
    # selected-actor result.
    cached_mean = np.asarray(
        actor["mean_1s_H0_m"],
        dtype=np.float64,
    )

    cached_cov = np.asarray(
        actor["covariance_1s_H0_m2"],
        dtype=np.float64,
    )

    req(
        np.allclose(
            trajectory_mean[3],
            cached_mean,
            rtol=1e-10,
            atol=1e-10,
        ),
        "selected receiver mean does not reproduce 9.13A",
    )

    req(
        np.allclose(
            cal_cov[3],
            cached_cov,
            rtol=1e-10,
            atol=1e-10,
        ),
        "selected receiver calibrated covariance does not reproduce 9.13A",
    )

    current_heading = (
        s5.latest_history_heading(
            selected.history,
            stack["parameters"][
                "low_speed_threshold_mps"
            ],
        )
    )

    _, angular_offset = s5.offset_objects(
        stack["parameters"]
    )

    current_position = tuple(
        float(v)
        for v in (
            selected.history.latest_position_H0_m
        )
    )

    h1_sample_id = (
        f"{scenario.scenario_id}|"
        f"{selected.prediction_id}|"
        f"{carrier.H1_GEOMETRY_MODE}"
    )

    h1_seed = int(
        h1.stage9_mc_seed(
            h1_sample_id
        )
    )

    arms = {}

    for arm, covariance in (
        ("RAW", raw_cov),
        ("CALIBRATED", cal_cov),
    ):

        posterior = (
            s5.deterministic_receiver_angular_posterior(
                trajectory_mean_h0_m=
                    trajectory_mean,

                raw_trajectory_covariance_h0_m2=
                    covariance,

                current_position_h0_m=
                    current_position,

                current_heading_h0_rad=
                    current_heading,

                receiver_geometry_mode=
                    carrier.H1_GEOMETRY_MODE,

                receiver_offset=
                    angular_offset,

                sample_count=
                    carrier.MC_SAMPLES,

                seed=
                    h1_seed,

                low_speed_threshold_mps=
                    float(
                        stack["parameters"][
                            "low_speed_threshold_mps"
                        ]
                    ),
            )
        )

        azimuth_samples = np.asarray(
            posterior.azimuth_samples_rad,
            dtype=np.float64,
        )

        plans = []

        for B in carrier.CODEBOOK_SIZES:
            codebook = stack["codebooks"][B]

            state = (
                s5.initial_adaptive_temporal_state()
            )

            for horizon_index, horizon_s in enumerate(
                carrier.H1_HORIZONS_S
            ):

                probability = (
                    s5.beam_probability_mass_from_samples(
                        azimuth_samples_rad=
                            azimuth_samples[
                                :,
                                horizon_index
                            ],

                        codebook=codebook,
                    )
                )

                temporal = (
                    s5.adaptive_temporal_step(
                        state=state,
                        probability=probability,
                        codebook=codebook,
                        requested_coverage=
                            carrier.H1_Q,
                    )
                )

                state, decision = (
                    s5.parse_temporal_result(
                        temporal,
                        state,
                    )
                )

                adaptive_indices = tuple(
                    int(v)
                    for v in
                    decision
                    .adaptive_selection
                    .beam_indices
                )

                physical_indices = tuple(
                    int(v)
                    for v in
                    s5.adaptive_probe_indices(
                        decision
                    )
                )

                reported_probe_count = int(
                    s5.actual_probe_count_for_adaptive_decision(
                        decision
                    )
                )

                unique_physical_probe_count = len(
                    set(physical_indices)
                )

                physical_probe_count = max(
                    reported_probe_count,
                    unique_physical_probe_count,
                )

                d = {
                    "adaptive_indices":
                        adaptive_indices,

                    "physical_indices":
                        physical_indices,

                    "selected_k":
                        int(
                            decision
                            .adaptive_selection
                            .k
                        ),

                    "posterior_mass_selected":
                        float(
                            decision
                            .adaptive_selection
                            .selected_in_support_mass
                        ),

                    "physical_probe_count":
                        int(physical_probe_count),

                    "physical_probe_count_frozen_reported":
                        int(reported_probe_count),

                    "unique_physical_probe_count":
                        int(unique_physical_probe_count),

                    "physical_probe_accounting_repaired":
                        bool(
                            reported_probe_count
                            <
                            unique_physical_probe_count
                        ),

                    "physical_probe_accounting_rule":
                        (
                            "MAX_FROZEN_REPORTED_COUNT_"
                            "AND_UNIQUE_PHYSICAL_INDEX_COUNT"
                        ),

                    "primary_beam_index":
                        int(
                            decision
                            .primary_beam_index
                        ),

                    "primary_switched":
                        bool(
                            decision
                            .primary_switched
                        ),

                    "loss_of_lock":
                        bool(
                            decision
                            .loss_of_lock
                        ),

                    "exhaustive_reacquisition":
                        bool(
                            decision
                            .exhaustive_fallback_active
                        ),
                }

                plans.append({
                    "arm":
                        arm,

                    "codebook_size":
                        int(B),

                    "horizon_index":
                        int(horizon_index),

                    "horizon_s":
                        float(horizon_s),

                    "azimuth_std_rad":
                        float(
                            s5.circular_std(
                                azimuth_samples[
                                    :,
                                    horizon_index
                                ]
                            )
                        ),

                    **d,
                })

        arms[arm] = plans

    codebooks = {}

    for B in carrier.CODEBOOK_SIZES:

        cached = actor["codebooks"][str(B)]

        beam_counts = [
            int(x)
            for x in cached[
                "beam_sample_counts"
            ]
        ]

        critical_counts = [
            int(x)
            for x in cached[
                "critical_beam_sample_counts"
            ]
        ]

        req(
            sum(beam_counts)
            ==
            carrier.MC_SAMPLES,
            f"9.13A beam-count drift B={B}",
        )

        req(
            sum(critical_counts)
            ==
            int(actor["critical_count"]),
            f"9.13A critical-count drift B={B}",
        )

        final_cal = next(
            p
            for p in arms["CALIBRATED"]
            if (
                int(p["codebook_size"])
                ==
                int(B)
                and
                int(p["horizon_index"])
                ==
                3
            )
        )

        heuristic_q = (
            carrier.heuristic_q_from_raw_pcrit(
                float(actor["Pcrit_raw"])
            )
        )

        c2 = dict(
            cached["C2"]
        )

        # Runtime was not retained in 9.13A; no solver is rerun.
        c2["solver_runtime_s"] = None
        c2["reused_from_9_13A"] = True

        codebooks[str(B)] = {
            "beam_count":
                int(B),

            "beam_sample_counts":
                beam_counts,

            "critical_beam_sample_counts":
                critical_counts,

            "fixed_top_k": {
                "1":
                    list(
                        carrier.select_fixed_top_k(
                            beam_counts,
                            1,
                        )
                    ),

                "3":
                    list(
                        carrier.select_fixed_top_k(
                            beam_counts,
                            3,
                        )
                    ),

                "5":
                    list(
                        carrier.select_fixed_top_k(
                            beam_counts,
                            5,
                        )
                    ),
            },

            "fixed_q": {
                "0.95":
                    list(
                        carrier.select_probability_mass_q(
                            beam_counts,
                            0.95,
                        )
                    ),

                "0.99":
                    list(
                        carrier.select_probability_mass_q(
                            beam_counts,
                            0.99,
                        )
                    ),
            },

            "exhaustive":
                list(range(B)),

            "uncertainty_only_adaptive_topk": {
                "selected_beam_indices":
                    [
                        int(x)
                        for x in
                        final_cal["adaptive_indices"]
                    ],

                "physical_probe_indices":
                    [
                        int(x)
                        for x in
                        final_cal["physical_indices"]
                    ],

                "K":
                    int(
                        final_cal["selected_k"]
                    ),

                "physical_probe_count":
                    int(
                        final_cal[
                            "physical_probe_count"
                        ]
                    ),

                "posterior_mass_selected":
                    float(
                        final_cal[
                            "posterior_mass_selected"
                        ]
                    ),

                "source":
                    (
                        "v1.3 selected-receiver "
                        "frozen H1 CAL q0.95 final horizon"
                    ),
            },

            "heuristic_q":
                float(heuristic_q),

            "heuristic_q_selected_beam_indices":
                list(
                    carrier.select_probability_mass_q(
                        beam_counts,
                        heuristic_q,
                    )
                ),

            "proposed_C2":
                c2,
        }

    return {
        "predictor_inference_runtime_s":
            float(runtime_s),

        "trajectory_mean_sha256":
            hashlib.sha256(
                trajectory_mean.tobytes()
            ).hexdigest(),

        "raw_covariance_sha256":
            hashlib.sha256(
                raw_cov.tobytes()
            ).hexdigest(),

        "calibrated_covariance_sha256":
            hashlib.sha256(
                cal_cov.tobytes()
            ).hexdigest(),

        "H1_paired_ablation": {
            "q":
                float(carrier.H1_Q),

            "geometry_mode":
                carrier.H1_GEOMETRY_MODE,

            "sample_id":
                h1_sample_id,

            "seed":
                h1_seed,

            "single_predictor_forward":
                True,

            "common_random_numbers":
                True,

            "arms":
                arms,
        },

        "codebooks":
            codebooks,
    }


def main():
    print("=" * 96)
    print("STAGE 9 — 9.13-v1.3 PRIMARY PRETRUTH R4 C1+C2/H1")
    print("REUSE SEALED 9.13A | CURRENT-ONLY RECEIVER BRIDGE")
    print("NO C1/MC/C2 RERUN | NO GT | NO METRICS | NO C3 YET")
    print("=" * 96)

    for p, expected in EXPECTED.items():
        req(
            p.is_file(),
            f"missing authority: {p}",
        )

        actual = sha(p)

        req(
            actual == expected,
            (
                f"authority SHA drift: {p}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            ),
        )

    req(
        not OUT.exists(),
        f"fresh v1.3 namespace exists: {OUT}",
    )

    formal_ids = [
        x.strip()
        for x in FORMAL_IDS
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]

    req(
        len(formal_ids) == FORMAL_N,
        "FORMAL scenario count drift",
    )

    carrier = load_module(
        "_stage9_913_v13_carrier",
        CARRIER,
    )

    # Exact current association is embedded above.
    # Block9.4B / scipy are not imported in this PRETRUTH pass.
    _original_loader = carrier.load_module_from_path

    class _UnusedEval94:
        pass

    def _primary_pretruth_loader(name, path):
        if (
            Path(path).resolve()
            ==
            Path(carrier.BLOCK94B_CARRIER).resolve()
        ):
            return _UnusedEval94()

        return _original_loader(
            name,
            path,
        )

    carrier.load_module_from_path = (
        _primary_pretruth_loader
    )

    stack = (
        carrier.load_frozen_controller_stack()
    )

    s5 = stack["s5"]
    sup = stack["sup"]
    g93 = stack["g93"]

    OUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    CACHE.mkdir(
        parents=True,
        exist_ok=False,
    )

    amendment = {
        "schema":
            "stage9_block913_v1_3_preoutcome_interface_amendment_v1",

        "status":
            "FROZEN_PREOUTCOME_INTERFACE_AMENDMENT",

        "reason":
            (
                "Frozen Stage2 FORMAL degraded anchor is "
                "UnlabeledDetectionFrame; frozen Stage5 class "
                "collector therefore returns zero semantic-class "
                "observations. Amendment uses exact current-frame "
                "WOMD association for static object_type when matched; "
                "unmatched current histories remain in the candidate "
                "universe with conservative semantic class OTHER. "
                "No future-GT information is used."
            ),

        "future_GT_seen":
            False,

        "FORMAL_outcomes_seen":
            False,

        "policy_retuning":
            False,

        "C1_definition_changed":
            False,

        "C2_policy_changed":
            False,

        "thresholds_changed":
            False,

        "receiver_selector_changed":
            False,

        "FORMAL_cohort_changed":
            False,

        "reuse": {
            "9_13A_manifest_sha256":
                sha(A_MANIFEST),

            "9_13A_seal_sha256":
                sha(A_SEAL),

            "blocked_state_sha256":
                sha(BLOCKED_SEAL),
        },

        "C3":
            "REMAINS_OPEN_PREOUTCOME_NOT_EXECUTED_BY_THIS_RUNNER",
    }

    atomic_json(
        AMENDMENT,
        amendment,
    )

    selected_n = 0
    no_receiver_n = 0
    selected_prediction_unavailable_n = 0
    h1_forward_n = 0

    started = time.time()

    for ordinal, sid in enumerate(
        formal_ids
    ):

        ap = a_path(
            ordinal,
            sid,
        )

        req(
            ap.is_file(),
            f"missing sealed 9.13A cache: {ap}",
        )

        a = json.loads(
            ap.read_text(
                encoding="utf-8"
            )
        )

        req(
            a["scenario_id"] == sid,
            f"9.13A SID mismatch: {sid}",
        )

        req(
            int(a["formal_ordinal"])
            ==
            ordinal,
            f"9.13A ordinal mismatch: {sid}",
        )

        req(
            a["future_GT_access"] is False,
            f"9.13A GT flag drift: {sid}",
        )

        validation_row = (
            stack["validation_by_id"]
            .get(sid)
        )

        req(
            validation_row is not None,
            f"FORMAL ID missing validation row: {sid}",
        )

        scenario, read_route = (
            g93.read_cal_scenario(
                s5,
                validation_row,
            )
        )

        current_index = int(
            scenario.current_time_index
        )

        req(
            current_index == 10,
            f"current index drift: {sid}",
        )

        built = (
            stack[
                "build_real_causal_inputs"
            ](
                scenario,
                clean_config=
                    stack["clean_config"],

                degraded_config=
                    stack["degraded_config"],
            )
        )

        (
            projection,
            receiver_result,
            selected,
        ) = amended_receiver_projection(
            stack=stack,
            s5=s5,
            sup=sup,
            scenario=scenario,
            built=built,
            current_index=current_index,
        )

        pretruth = {
            "schema":
                "stage9_block913_v1_3_pretruth_scenario_v1",

            "scenario_id":
                sid,

            "formal_ordinal":
                ordinal,

            "read_route":
                str(read_route),

            "current_index":
                current_index,

            "controller_phase":
                "COMPLETE_PRETRUTH",

            "future_GT_access":
                False,

            "FORMAL_outcomes_computed":
                False,

            "tracks_to_predict_receiver_selection":
                False,

            "future_truth_receiver_selection":
                False,

            "receiver_interface_amendment":
                "V1_3_CURRENT_ONLY_STATIC_OBJECT_TYPE",

            "receiver": {
                "status":
                    str(
                        receiver_result.status
                    ),

                "eligible_candidate_count":
                    int(
                        receiver_result
                        .eligible_candidate_count
                    ),

                "selected":
                    bool(
                        selected is not None
                    ),

                "prediction_id":
                    (
                        str(
                            selected.prediction_id
                        )
                        if selected is not None
                        else None
                    ),

                "semantic_class":
                    (
                        str(
                            selected.semantic_class
                        )
                        if selected is not None
                        else None
                    ),

                "planar_range_m":
                    (
                        float(
                            receiver_result
                            .planar_range_m
                        )
                        if (
                            receiver_result
                            .planar_range_m
                            is not None
                        )
                        else None
                    ),
            },

            "causal_population": {
                "causal_history_count":
                    int(
                        projection[
                            "causal_history_count"
                        ]
                    ),

                "current_available_history_count":
                    int(
                        projection[
                            "current_available_history_count"
                        ]
                    ),

                "current_only_supervision_match_count":
                    int(
                        projection[
                            "current_only_supervision_match_count"
                        ]
                    ),
            },

            "reuse_9_13A": {
                "cache_sha256":
                    sha(ap),

                "C1_MC_recomputed":
                    False,

                "C2_solver_recomputed":
                    False,
            },

            "C3":
                {
                    "executed":
                        False,

                    "status":
                        "PENDING_NEXT_PREOUTCOME_STEP",
                },
        }

        if selected is None:
            no_receiver_n += 1

            pretruth[
                "controller_phase"
            ] = (
                "COMPLETE_PRETRUTH_"
                "NO_ELIGIBLE_RECEIVER"
            )

        else:
            selected_n += 1

            pid = str(
                selected.prediction_id
            )

            actors = [
                x
                for x in a["actors"]
                if (
                    str(
                        x["prediction_id"]
                    )
                    ==
                    pid
                )
            ]

            if len(actors) == 0:

                selected_prediction_unavailable_n += 1

                pretruth[
                    "controller_phase"
                ] = (
                    "COMPLETE_PRETRUTH_"
                    "SELECTED_RECEIVER_"
                    "PREDICTION_UNAVAILABLE"
                )

                pretruth[
                    "receiver"
                ][
                    "selected_receiver_prediction_available"
                ] = False

                pretruth[
                    "receiver"
                ][
                    "current_only_prediction_payload_association"
                ] = projection[
                    "_current_only_match_by_prediction"
                ][pid]

            else:
                req(
                    len(actors) == 1,
                    (
                        "selected receiver does not map "
                        f"uniquely to sealed 9.13A actor: {sid}|{pid}"
                    ),
                )

                actor = actors[0]

                extra = h1_and_baselines(
                    carrier=carrier,
                    stack=stack,
                    scenario=scenario,
                    built=built,
                    selected=selected,
                    actor=actor,
                )

                h1_forward_n += 1

                pretruth[
                    "receiver"
                ][
                    "selected_receiver_prediction_available"
                ] = True

                pretruth[
                    "receiver"
                ][
                    "current_only_prediction_payload_association"
                ] = projection[
                    "_current_only_match_by_prediction"
                ][pid]

                pretruth["C1_C2"] = {
                    "prediction_id":
                        pid,

                    "predictor_inference_runtime_s":
                        extra[
                            "predictor_inference_runtime_s"
                        ],

                    "trajectory_mean_sha256":
                        extra[
                            "trajectory_mean_sha256"
                        ],

                    "raw_covariance_sha256":
                        extra[
                            "raw_covariance_sha256"
                        ],

                    "calibrated_covariance_sha256":
                        extra[
                            "calibrated_covariance_sha256"
                        ],

                    "H1_paired_ablation":
                        extra[
                            "H1_paired_ablation"
                        ],

                    "ego_prediction_1s":
                        a["ego_prediction_1s"],

                    "sample_id":
                        actor["sample_id"],

                    "seed":
                        int(
                            actor["seed"]
                        ),

                    "samples_sha256":
                        actor[
                            "samples_sha256"
                        ],

                    "critical_bitset_sha256":
                        actor[
                            "critical_bitset_sha256"
                        ],

                    "critical_count":
                        int(
                            actor[
                                "critical_count"
                            ]
                        ),

                    "Pcrit_raw":
                        float(
                            actor[
                                "Pcrit_raw"
                            ]
                        ),

                    "Pcrit_cal":
                        float(
                            actor[
                                "Pcrit_cal_descriptive_only"
                            ]
                        ),

                    "Pcrit_cal_solver_input":
                        False,

                    "u_b_cal_defined":
                        False,

                    "same_MC_sample_reuse":
                        True,

                    "codebooks":
                        extra["codebooks"],
                }

        pretruth[
            "pretruth_decision_sha256"
        ] = canonical_sha(
            pretruth
        )

        atomic_json(
            CACHE /
            f"{ordinal:05d}_{sid}.json",
            pretruth,
        )

        done = ordinal + 1

        if (
            done % 100 == 0
            or
            done == FORMAL_N
        ):
            print(
                f"PRETRUTH progress = "
                f"{done}/{FORMAL_N} ; "
                f"selected={selected_n} ; "
                f"no_receiver={no_receiver_n} ; "
                f"H1_forwards={h1_forward_n} ; "
                f"elapsed_s="
                f"{time.time()-started:.1f}",
                flush=True,
            )

    # Freeze ordered manifest.
    fd, tmp = tempfile.mkstemp(
        prefix=MANIFEST.name + ".tmp.",
        dir=str(OUT),
    )

    try:
        with os.fdopen(fd, "wb") as f:
            for ordinal, sid in enumerate(
                formal_ids
            ):
                p = (
                    CACHE /
                    f"{ordinal:05d}_{sid}.json"
                )

                obj = json.loads(
                    p.read_text(
                        encoding="utf-8"
                    )
                )

                f.write(
                    canonical_bytes(obj)
                    +
                    b"\n"
                )

            f.flush()
            os.fsync(f.fileno())

        os.replace(
            tmp,
            MANIFEST,
        )

        os.chmod(
            MANIFEST,
            0o444,
        )

    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    summary = {
        "schema":
            "stage9_block913_v1_3_pretruth_summary_v1",

        "status":
            "FROZEN_COMPLETE_STAGE9_BLOCK913_V1_3_C1_C2_PRETRUTH_NO_GT",

        "FORMAL_scenario_count":
            FORMAL_N,

        "primary_receiver_selected":
            selected_n,

        "no_eligible_receiver":
            no_receiver_n,

        "selected_receiver_prediction_unavailable":
            selected_prediction_unavailable_n,

        "H1_selected_receiver_forward_count":
            h1_forward_n,

        "9_13A_C1_MC_recomputed":
            False,

        "9_13A_C2_recomputed":
            False,

        "future_GT_access":
            False,

        "FORMAL_outcomes_computed":
            False,

        "training":
            False,

        "CAL_refit":
            False,

        "tuning":
            False,

        "C3":
            "PENDING_PREOUTCOME",

        "future_GT_may_open":
            False,
    }

    atomic_json(
        SUMMARY,
        summary,
    )

    seal = {
        "schema":
            "stage9_block913_v1_3_pretruth_seal_v1",

        "status":
            summary["status"],

        "amendment_sha256":
            sha(AMENDMENT),

        "9_13A_manifest_sha256":
            sha(A_MANIFEST),

        "9_13A_seal_sha256":
            sha(A_SEAL),

        "blocked_interface_seal_sha256":
            sha(BLOCKED_SEAL),

        "manifest_sha256":
            sha(MANIFEST),

        "summary_sha256":
            sha(SUMMARY),

        "future_GT_access":
            False,

        "FORMAL_outcomes_computed":
            False,

        "C3_pretruth_complete":
            False,

        "future_GT_may_open":
            False,

        "next":
            "RHO2_C3_PRETRUTH_ONLY_THEN_COMBINED_PREOUTCOME_SEAL",
    }

    atomic_json(
        SEAL,
        seal,
    )

    print()
    print("=" * 96)
    print("9.13-v1.3 C1+C2 PRETRUTH = COMPLETE")
    print(
        "FORMAL_SCENARIO_COUNT =",
        FORMAL_N,
    )
    print(
        "PRIMARY_RECEIVER_SELECTED =",
        selected_n,
    )
    print(
        "NO_ELIGIBLE_RECEIVER =",
        no_receiver_n,
    )
    print(
        "SELECTED_RECEIVER_PREDICTION_UNAVAILABLE =",
        selected_prediction_unavailable_n,
    )
    print(
        "H1_SELECTED_RECEIVER_FORWARDS =",
        h1_forward_n,
    )
    print(
        "C1_MC_RECOMPUTED = FALSE"
    )
    print(
        "C2_RECOMPUTED = FALSE"
    )
    print(
        "FUTURE_GT_ACCESSED = FALSE"
    )
    print(
        "FORMAL_OUTCOMES_COMPUTED = FALSE"
    )
    print(
        "C3_PRETRUTH_COMPLETE = FALSE"
    )
    print(
        "FUTURE_GT_MAY_OPEN = FALSE"
    )
    print(
        "MANIFEST_SHA256 =",
        sha(MANIFEST),
    )
    print(
        "SUMMARY_SHA256 =",
        sha(SUMMARY),
    )
    print(
        "SEAL_SHA256 =",
        sha(SEAL),
    )
    print(
        "NEXT = RHO2 C3 PRETRUTH + COMBINED SEAL"
    )
    print("=" * 96)


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print("!" * 96)
        print(
            "9.13-v1.3 PRETRUTH FAIL-CLOSED"
        )
        print(
            type(exc).__name__ + ":",
            exc,
        )
        print(
            "PRESERVE THIS ATTEMPT."
        )
        print(
            "DO NOT RERUN SAME NAMESPACE."
        )
        print(
            "DO NOT OPEN FUTURE GT."
        )
        print("!" * 96)
        raise
