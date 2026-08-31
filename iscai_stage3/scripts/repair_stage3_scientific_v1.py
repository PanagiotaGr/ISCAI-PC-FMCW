from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/agni/waymo/iscai_stage3').resolve()
AUDIT = ROOT.parent / 'audits' / 'stage3_repair_v1'
BACKUP = AUDIT / 'backup'
REPORT = AUDIT / 'repair_report.json'

EXPECTED = {
    'src/iscai_stage3/association/contracts.py': 'e735089cb4374afbf075349bc7cc6887ae2b4cbe32cfcc29676ca6b35b6c3f98',
    'src/iscai_stage3/association/nearest_neighbor.py': '968c19b8fb4b1c98c62e5a64e9b88a8dcf3b36d6bfecf6d63158c545e84e9a0b',
    'configs/stage3_association.json': 'cf0ae6f1e68cb0708115e9d51b397f9febc3e44bfa6915b533f8c9fdf1f0b08a',
    'configs/stage3_evaluation.json': '344416c3e5de80d85ef23e41cd80448d7a101bb7aad96458c9dd1f4be447d88f',
    'scripts/run_block31_association_gate.py': '74162858bb54aac9f24e8db3e789168473a28f2d6309d6567f6817d281400dcb',
    'src/iscai_stage3/evaluation/contracts.py': 'cc879d707121873fd6ebf7049fcf575884840a3d7416547dc4294890eb68e291',
    'src/iscai_stage3/evaluation/metrics.py': '5c9685cf9c73ccad50f16507efcefb8211648b2a744587d5a53154b8c6403842',
    'scripts/run_block38f_formal_evaluation.py': '7cf1db9ca4907b32a160c6c410f716ec98a7a16e813f669695a0160157dfa871',
    'scripts/run_block38g_reproducibility_gate.py': 'f0da9ace3d36a07c9e6a69d89fe8d75b5406379306dae97d3c3056197db274ef',
    'scripts/run_stage3_final_closure.py': '575ab0dc22671965facd0db058d628fafb0ed1bc302099c8a9c5da9a7abfbbe1',
}

def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()

def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')

def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding='utf-8')
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{path}: expected exactly one replacement target, got {count}')
    path.write_text(text.replace(old, new), encoding='utf-8')

def replace_all_required(path: Path, old: str, new: str, minimum: int = 1) -> None:
    text = path.read_text(encoding='utf-8')
    count = text.count(old)
    if count < minimum:
        raise RuntimeError(f'{path}: expected at least {minimum} occurrences of {old!r}, got {count}')
    path.write_text(text.replace(old, new), encoding='utf-8')

if not ROOT.is_dir():
    raise SystemExit(f'ERROR: Stage3 root not found: {ROOT}')

# Idempotence: if the repair marker exists and all new semantic files exist, stop cleanly.
if REPORT.is_file() and (ROOT / 'src/iscai_stage3/evaluation/anchor_alignment.py').is_file():
    data = json.loads(REPORT.read_text(encoding='utf-8'))
    if data.get('status') == 'APPLIED':
        print('STAGE3_REPAIR_V1 = ALREADY_APPLIED')
        print('report =', REPORT)
        raise SystemExit(0)

AUDIT.mkdir(parents=True, exist_ok=True)
BACKUP.mkdir(parents=True, exist_ok=True)

# Verify exact audited pre-repair state. stage3_evaluation.json is also backed up,
# but its SHA was not needed by the prior defect proof.
for rel, expected in EXPECTED.items():
    p = ROOT / rel
    if not p.is_file():
        raise SystemExit(f'ERROR: missing expected file: {p}')
    actual = digest(p)
    if expected is not None and actual != expected:
        raise SystemExit(
            f'ERROR: audited pre-repair SHA mismatch for {rel}\n'
            f'expected={expected}\nactual  ={actual}\n'
            'No files were modified.'
        )

# Backup every file that will be modified.
for rel in EXPECTED:
    src = ROOT / rel
    dst = BACKUP / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

# ------------------------------------------------------------------
# 1) Association nomenclature: implementation is deterministic gated
#    greedy NN, not a global-optimal Hungarian assignment.
# ------------------------------------------------------------------
replace_once(
    ROOT / 'src/iscai_stage3/association/contracts.py',
    'Deterministic gated global nearest-neighbor\n    association in Stage2 measurement space.',
    'Deterministic gated greedy nearest-neighbor\n    association in Stage2 measurement space.',
)
replace_once(
    ROOT / 'src/iscai_stage3/association/contracts.py',
    '"deterministic_gated_global_"\n        "nearest_neighbor"',
    '"deterministic_gated_greedy_"\n        "nearest_neighbor"',
)
replace_once(
    ROOT / 'src/iscai_stage3/association/nearest_neighbor.py',
    'Deterministic gated global nearest-neighbor\n    association.',
    'Deterministic gated greedy nearest-neighbor\n    association.',
)
replace_once(
    ROOT / 'configs/stage3_association.json',
    '"method": "deterministic_gated_global_nearest_neighbor"',
    '"method": "deterministic_gated_greedy_nearest_neighbor"',
)
replace_once(
    ROOT / 'scripts/run_block31_association_gate.py',
    '"gated global nearest-neighbor"',
    '"gated greedy nearest-neighbor"',
)
replace_once(
    ROOT / 'scripts/run_block31_association_gate.py',
    '"deterministic_gated_global_"\n        "nearest_neighbor"',
    '"deterministic_gated_greedy_"\n        "nearest_neighbor"',
)

# Evaluation config: freeze the repaired evaluator semantics explicitly.
eval_cfg_path = ROOT / 'configs/stage3_evaluation.json'
eval_cfg = json.loads(eval_cfg_path.read_text(encoding='utf-8'))
eval_cfg['temporal_alignment'] = {
    'prediction_anchor': 'global_WOMD_current_time_anchor',
    'active_stale_track_policy': 'causal_open_loop_propagation_to_global_anchor',
    'terminated_track_policy': 'exclude_from_anchor_forecast',
    'prediction_timestamp_rule': 'timestamp_s ~= anchor_timestamp_s + horizon_s',
    'timestamp_tolerance_s': 0.002,
}
eval_cfg['primary_target_ignore_policy'] = {
    'primary_truth': 'WOMD_tracks_to_predict_evaluator_only',
    'ignored_truth': 'current_valid_non_SDC_not_in_tracks_to_predict',
    'ignored_real_actor_prediction_is_false_positive': False,
    'clutter_or_unmatched_prediction_is_false_positive': True,
}
write(eval_cfg_path, json.dumps(eval_cfg, indent=2, sort_keys=False) + '\n')

# ------------------------------------------------------------------
# 2) Enforce timestamp alignment at the common evaluation contract.
# ------------------------------------------------------------------
contracts = ROOT / 'src/iscai_stage3/evaluation/contracts.py'
replace_once(
    contracts,
    '''        if len(ids) != len(set(ids)):\n            raise ValueError(\n                "Prediction IDs must "\n                "be unique."\n            )\n\n        if (\n            self.runtime_ms is not None''',
    '''        if len(ids) != len(set(ids)):\n            raise ValueError(\n                "Prediction IDs must "\n                "be unique."\n            )\n\n        # Every prediction point must refer to the same\n        # global anchor carried by EvaluationPredictionSet.\n        # This prevents stale-track forecasts from being\n        # relabeled as if they started at the WOMD anchor.\n        timestamp_tolerance_s = 2e-3\n\n        for track in self.tracks:\n            for point in track.points:\n                expected_timestamp_s = (\n                    float(self.anchor_timestamp_s)\n                    + float(point.horizon_s)\n                )\n\n                if abs(\n                    float(point.timestamp_s)\n                    - expected_timestamp_s\n                ) > timestamp_tolerance_s:\n                    raise ValueError(\n                        "Prediction point timestamp is not "\n                        "aligned to the prediction-set "\n                        "global anchor."\n                    )\n\n        if (\n            self.runtime_ms is not None''',
)

# ------------------------------------------------------------------
# 3) Add evaluator ignore semantics for real non-target actors.
# ------------------------------------------------------------------
metrics = ROOT / 'src/iscai_stage3/evaluation/metrics.py'
replace_once(
    metrics,
    '''    assignment_future_truth_used: bool = False\n\n\ndef _planar_distance''',
    '''    assignment_future_truth_used: bool = False\n\n    # Predictions that match a real current-valid actor outside\n    # the benchmark target subset are evaluator-ignored, not FP.\n    ignored_prediction_count: int = 0\n    ignored_prediction_ids: tuple[str, ...] = ()\n\n\ndef _planar_distance''',
)
replace_once(
    metrics,
    '''def evaluate_prediction_set(\n    predictions: EvaluationPredictionSet,\n    truth: EvaluationTruth,\n    *,\n    config: EvaluationConfig | None = None,\n) -> EvaluationReport:''',
    '''def evaluate_prediction_set(\n    predictions: EvaluationPredictionSet,\n    truth: EvaluationTruth,\n    *,\n    config: EvaluationConfig | None = None,\n    ignored_truth: EvaluationTruth | None = None,\n) -> EvaluationReport:''',
)
replace_once(
    metrics,
    '''    assignment = (\n        assign_predictions_to_truth(\n            predictions,\n            truth,\n            config=config,\n        )\n    )\n\n    prediction_by_id = {''',
    '''    if ignored_truth is not None:\n        if ignored_truth.scenario_id != truth.scenario_id:\n            raise ValueError(\n                "Ignored-truth scenario ID does not match."\n            )\n\n        if abs(\n            ignored_truth.anchor_timestamp_s\n            - truth.anchor_timestamp_s\n        ) > config.anchor_timestamp_tolerance_s:\n            raise ValueError(\n                "Ignored-truth anchor timestamp does not match."\n            )\n\n        primary_truth_ids = {\n            track.truth_id\n            for track in truth.tracks\n        }\n        ignored_truth_ids = {\n            track.truth_id\n            for track in ignored_truth.tracks\n        }\n\n        if primary_truth_ids & ignored_truth_ids:\n            raise ValueError(\n                "Primary and ignored truth sets must be disjoint."\n            )\n\n    assignment = (\n        assign_predictions_to_truth(\n            predictions,\n            truth,\n            config=config,\n        )\n    )\n\n    prediction_by_id = {''',
)
replace_once(
    metrics,
    '''    truth_count = len(\n        truth.tracks\n    )\n\n    prediction_count = len(\n        predictions.tracks\n    )\n\n    match_count = len(\n        assignment.matches\n    )\n\n    recall = (''',
    '''    # Primary assignment is intentionally performed first.\n    # Only predictions still unmatched after primary-target\n    # assignment may be absorbed by the ignored real-actor set.\n    ignored_prediction_ids: tuple[str, ...] = ()\n\n    if ignored_truth is not None and assignment.unmatched_prediction_ids:\n        unmatched_ids = set(\n            assignment.unmatched_prediction_ids\n        )\n\n        residual_predictions = EvaluationPredictionSet(\n            method=predictions.method,\n            scenario_id=predictions.scenario_id,\n            anchor_timestamp_s=predictions.anchor_timestamp_s,\n            tracks=tuple(\n                track\n                for track in predictions.tracks\n                if track.prediction_id in unmatched_ids\n            ),\n            runtime_ms=predictions.runtime_ms,\n        )\n\n        ignored_assignment = assign_predictions_to_truth(\n            residual_predictions,\n            ignored_truth,\n            config=config,\n        )\n\n        ignored_prediction_ids = tuple(\n            sorted(\n                match.prediction_id\n                for match in ignored_assignment.matches\n            )\n        )\n\n    ignored_id_set = set(ignored_prediction_ids)\n\n    false_prediction_ids = tuple(\n        prediction_id\n        for prediction_id in assignment.unmatched_prediction_ids\n        if prediction_id not in ignored_id_set\n    )\n\n    truth_count = len(\n        truth.tracks\n    )\n\n    prediction_count = len(\n        predictions.tracks\n    )\n\n    match_count = len(\n        assignment.matches\n    )\n\n    recall = (''',
)
replace_once(
    metrics,
    '''    precision = (\n        1.0\n        if prediction_count == 0\n        and truth_count == 0\n        else\n        (\n            0.0\n            if prediction_count == 0\n            else\n            match_count\n            /\n            prediction_count\n        )\n    )''',
    '''    precision_denominator = (\n        match_count\n        + len(false_prediction_ids)\n    )\n\n    precision = (\n        1.0\n        if precision_denominator == 0\n        and truth_count == 0\n        else\n        (\n            0.0\n            if precision_denominator == 0\n            else\n            match_count\n            /\n            precision_denominator\n        )\n    )''',
)
replace_once(
    metrics,
    '''        false_prediction_count=len(\n            assignment\n            .unmatched_prediction_ids\n        ),''',
    '''        false_prediction_count=len(\n            false_prediction_ids\n        ),''',
)
replace_once(
    metrics,
    '''        assignment_future_truth_used=False,\n    )''',
    '''        assignment_future_truth_used=False,\n        ignored_prediction_count=len(\n            ignored_prediction_ids\n        ),\n        ignored_prediction_ids=(\n            ignored_prediction_ids\n        ),\n    )''',
)

# ------------------------------------------------------------------
# 4) Add reusable anchor-alignment helpers.
# ------------------------------------------------------------------
anchor_module = '''from __future__ import annotations\n\nfrom math import isfinite\n\nfrom iscai_stage3.evaluation.contracts import (\n    EvaluationPredictionTrack,\n    PredictionTrajectoryPoint,\n)\nfrom iscai_stage3.filters.imm import (\n    combine_imm_states,\n    interact_mode_states,\n    predict_imm_mode_state,\n)\nfrom iscai_stage3.filters.kalman_ekf import (\n    predict_kalman_state,\n)\n\n\ndef _normalized_horizons(horizons_s):\n    result = tuple(float(x) for x in horizons_s)\n    if not result:\n        raise ValueError("At least one horizon is required.")\n    if any((not isfinite(x)) or x <= 0.0 for x in result):\n        raise ValueError("Horizons must be positive and finite.")\n    if any(b <= a for a, b in zip(result, result[1:])):\n        raise ValueError("Horizons must strictly increase.")\n    return result\n\n\ndef align_state_prediction_to_global_anchor(\n    state,\n    predictor,\n    *,\n    horizons_s,\n    anchor_timestamp_s: float,\n    tolerance_s: float = 2e-3,\n):\n    \"\"\"\n    Align CV/CA/CTRV-style state forecasts to the global WOMD anchor.\n\n    A still-active track may have its latest detection one or two frames\n    before the anchor.  We causally propagate from that last state to the\n    global anchor and then to anchor+horizon, without reading future truth.\n    \"\"\"\n    horizons = _normalized_horizons(horizons_s)\n    anchor = float(anchor_timestamp_s)\n    state_time = float(state.timestamp_s)\n    delta = anchor - state_time\n\n    if delta < -tolerance_s:\n        raise ValueError("State timestamp is after the global anchor.")\n\n    if abs(delta) <= 1e-12:\n        raw = predictor(state, horizons_s=horizons)\n        anchor_position = tuple(float(x) for x in state.position_H0_m)\n        future_points = tuple(raw.points)\n    else:\n        if delta <= 0.0:\n            raise ValueError("State/global-anchor ordering is invalid.")\n\n        extended = (\n            delta,\n            *tuple(delta + h for h in horizons),\n        )\n        raw = predictor(state, horizons_s=extended)\n\n        if len(raw.points) != len(horizons) + 1:\n            raise ValueError("Unexpected aligned prediction point count.")\n\n        anchor_position = tuple(\n            float(x)\n            for x in raw.points[0].position_H0_m\n        )\n        future_points = tuple(raw.points[1:])\n\n    if len(future_points) != len(horizons):\n        raise ValueError("Unexpected future prediction point count.")\n\n    converted = []\n    for horizon, point in zip(horizons, future_points):\n        expected_timestamp = anchor + horizon\n        if abs(float(point.timestamp_s) - expected_timestamp) > tolerance_s:\n            raise ValueError(\n                "State-model forecast did not land on global anchor+horizon."\n            )\n\n        converted.append(\n            PredictionTrajectoryPoint(\n                horizon_s=horizon,\n                timestamp_s=expected_timestamp,\n                position_H0_m=tuple(\n                    float(x)\n                    for x in point.position_H0_m\n                ),\n            )\n        )\n\n    return EvaluationPredictionTrack(\n        prediction_id=str(raw.track_id),\n        anchor_position_H0_m=anchor_position,\n        points=tuple(converted),\n    )\n\n\ndef align_kalman_state_to_global_anchor(\n    state,\n    *,\n    anchor_timestamp_s: float,\n    config,\n    tolerance_s: float = 2e-3,\n):\n    anchor = float(anchor_timestamp_s)\n    current = float(state.timestamp_s)\n\n    if current > anchor + tolerance_s:\n        raise ValueError("Kalman state timestamp is after global anchor.")\n\n    if abs(current - anchor) <= 1e-12:\n        return state\n\n    if current >= anchor:\n        raise ValueError("Kalman state/global-anchor ordering is invalid.")\n\n    return predict_kalman_state(\n        state,\n        target_timestamp_s=anchor,\n        config=config,\n    )\n\n\ndef align_imm_states_to_global_anchor(\n    states,\n    mode_probabilities,\n    *,\n    frame_timestamps_s,\n    anchor_timestamp_s: float,\n    config,\n    tolerance_s: float = 2e-3,\n):\n    \"\"\"Causal open-loop IMM propagation through missed anchor frames.\"\"\"\n    current_states = tuple(states)\n    current_probabilities = tuple(float(x) for x in mode_probabilities)\n\n    state_times = {float(state.timestamp_s) for state in current_states}\n    if len(state_times) != 1:\n        raise ValueError("IMM mode states are not time-aligned.")\n\n    current_time = float(current_states[0].timestamp_s)\n    anchor = float(anchor_timestamp_s)\n\n    if current_time > anchor + tolerance_s:\n        raise ValueError("IMM state timestamp is after global anchor.")\n\n    frame_times = tuple(float(x) for x in frame_timestamps_s)\n    subsequent = tuple(\n        t\n        for t in frame_times\n        if t > current_time + 1e-12\n        and t <= anchor + tolerance_s\n    )\n\n    for target_timestamp in subsequent:\n        mixed_states, predicted_probabilities, _ = interact_mode_states(\n            current_states,\n            current_probabilities,\n            config=config,\n        )\n\n        current_states = tuple(\n            predict_imm_mode_state(\n                state,\n                target_timestamp_s=target_timestamp,\n                config=config,\n            )\n            for state in mixed_states\n        )\n        current_probabilities = tuple(predicted_probabilities)\n\n    final_time = float(current_states[0].timestamp_s)\n    if abs(final_time - anchor) > tolerance_s:\n        raise ValueError(\n            "IMM open-loop propagation did not reach the global anchor."\n        )\n\n    combined = combine_imm_states(\n        current_states,\n        current_probabilities,\n    )\n    anchor_position = tuple(\n        float(combined.mean_10[i])\n        for i in range(3)\n    )\n\n    return current_states, current_probabilities, anchor_position\n'''
write(ROOT / 'src/iscai_stage3/evaluation/anchor_alignment.py', anchor_module)

# ------------------------------------------------------------------
# 5) Patch the formal runner: only non-terminated tracks forecast;
#    stale active tracks are propagated to exact global anchor;
#    primary reconstruction ignores real non-target actors.
# ------------------------------------------------------------------
runner = ROOT / 'scripts/run_block38f_formal_evaluation.py'
replace_once(
    runner,
    '''from iscai_stage3.evaluation import (\n    build_womd_truth_sidecar,\n)''',
    '''from iscai_stage3.evaluation import (\n    build_womd_truth_sidecar,\n)\nfrom iscai_stage3.evaluation.anchor_alignment import (\n    align_imm_states_to_global_anchor,\n    align_kalman_state_to_global_anchor,\n    align_state_prediction_to_global_anchor,\n)''',
)
replace_once(
    runner,
    '''            prediction = predictor(\n                state,\n                horizons_s=HORIZONS,\n            )\n\n            predictions.append(\n                adapted_prediction(\n                    prediction,\n                    observations[-1]\n                    .position_H0_m,\n                )\n            )''',
    '''            predictions.append(\n                align_state_prediction_to_global_anchor(\n                    state,\n                    predictor,\n                    horizons_s=HORIZONS,\n                    anchor_timestamp_s=anchor_timestamp,\n                )\n            )''',
)
replace_once(
    runner,
    '''            prediction = (\n                predict_kalman(\n                    track_id=(\n                        result.track_id\n                    ),\n                    state=(\n                        result.current_state\n                    ),\n                    horizons_s=HORIZONS,\n                    config=config,\n                )\n            )\n\n            predictions.append(\n                adapted_prediction(\n                    prediction,\n                    observations[-1]\n                    .position_H0_m,\n                )\n            )''',
    '''            aligned_state = (\n                align_kalman_state_to_global_anchor(\n                    result.current_state,\n                    anchor_timestamp_s=anchor_timestamp,\n                    config=config,\n                )\n            )\n\n            prediction = (\n                predict_kalman(\n                    track_id=(\n                        result.track_id\n                    ),\n                    state=aligned_state,\n                    horizons_s=HORIZONS,\n                    config=config,\n                )\n            )\n\n            predictions.append(\n                adapted_prediction(\n                    prediction,\n                    tuple(\n                        float(aligned_state.mean_6[i])\n                        for i in range(3)\n                    ),\n                )\n            )''',
)
replace_once(
    runner,
    '''            prediction = predict_imm(\n                track_id=(\n                    result.track_id\n                ),\n                states=(\n                    result\n                    .current_mode_states\n                ),\n                mode_probabilities=(\n                    result\n                    .current_mode_probabilities\n                ),\n                horizons_s=HORIZONS,\n                config=config,\n            )\n\n            predictions.append(\n                adapted_prediction(\n                    prediction,\n                    observations[-1]\n                    .position_H0_m,\n                )\n            )''',
    '''            (\n                aligned_states,\n                aligned_probabilities,\n                aligned_anchor_position,\n            ) = align_imm_states_to_global_anchor(\n                result.current_mode_states,\n                result.current_mode_probabilities,\n                frame_timestamps_s=(\n                    context.frame_timestamps_s\n                ),\n                anchor_timestamp_s=anchor_timestamp,\n                config=config,\n            )\n\n            prediction = predict_imm(\n                track_id=(\n                    result.track_id\n                ),\n                states=aligned_states,\n                mode_probabilities=(\n                    aligned_probabilities\n                ),\n                horizons_s=HORIZONS,\n                config=config,\n            )\n\n            predictions.append(\n                adapted_prediction(\n                    prediction,\n                    aligned_anchor_position,\n                )\n            )''',
)
replace_once(
    runner,
    '''def evaluate_view(\n    prediction_sets,\n    truth,\n):''',
    '''def evaluate_view(\n    prediction_sets,\n    truth,\n    *,\n    ignored_truth=None,\n):''',
)
replace_once(
    runner,
    '''            evaluate_prediction_set(\n                predictions,\n                truth=truth,\n                config=EVAL_CONFIG,\n            )''',
    '''            evaluate_prediction_set(\n                predictions,\n                truth=truth,\n                config=EVAL_CONFIG,\n                ignored_truth=ignored_truth,\n            )''',
)
replace_once(
    runner,
    '''        tracks = tuple(\n            associated.tracks\n        )\n\n        anchor_timestamp = (\n            timestamps[-1]\n        )''',
    '''        anchor_timestamp = (\n            timestamps[-1]\n        )\n\n        all_associated_tracks = tuple(\n            associated.tracks\n        )\n\n        # A track may exist historically but already be terminated\n        # before the global anchor.  Such a track must not emit an\n        # anchor forecast.  Still-active tracks with a recent miss\n        # are retained and causally propagated to the anchor.\n        tracks = tuple(\n            track\n            for track in all_associated_tracks\n            if not track.terminated\n        )\n\n        stale_active_tracks = tuple(\n            track\n            for track in tracks\n            if track.detections\n            and track.detections[-1].timestamp_s\n                < anchor_timestamp - 2e-3\n        )''',
)
replace_once(
    runner,
    '''        supplementary_truth = (\n            build_womd_truth_sidecar(\n                scenario,\n                T_H0_from_W=(\n                    adapted.frames\n                    .T_H0_from_W\n                ),\n                eligible_track_indices=(\n                    all_current_indices\n                ),\n                horizons_s=HORIZONS,\n            )\n        )\n\n        primary_reports = (\n            evaluate_view(\n                prediction_sets,\n                primary_truth,\n            )\n        )''',
    '''        supplementary_truth = (\n            build_womd_truth_sidecar(\n                scenario,\n                T_H0_from_W=(\n                    adapted.frames\n                    .T_H0_from_W\n                ),\n                eligible_track_indices=(\n                    all_current_indices\n                ),\n                horizons_s=HORIZONS,\n            )\n        )\n\n        ignored_indices = tuple(\n            sorted(\n                set(all_current_indices)\n                - set(primary_indices)\n            )\n        )\n\n        ignored_truth = (\n            build_womd_truth_sidecar(\n                scenario,\n                T_H0_from_W=(\n                    adapted.frames\n                    .T_H0_from_W\n                ),\n                eligible_track_indices=(\n                    ignored_indices\n                ),\n                horizons_s=HORIZONS,\n            )\n        )\n\n        primary_reports = (\n            evaluate_view(\n                prediction_sets,\n                primary_truth,\n                ignored_truth=ignored_truth,\n            )\n        )''',
)
replace_once(
    runner,
    '''            "association_runtime_ms":\n                association_ms,\n\n            "scenario_runtime_ms":''',
    '''            "association_runtime_ms":\n                association_ms,\n\n            "association_track_count_all":\n                len(all_associated_tracks),\n\n            "anchor_active_track_count":\n                len(tracks),\n\n            "terminated_track_count_excluded":\n                (\n                    len(all_associated_tracks)\n                    - len(tracks)\n                ),\n\n            "stale_active_track_count_propagated":\n                len(stale_active_tracks),\n\n            "scenario_runtime_ms":''',
)
replace_once(
    runner,
    '''            "primary_target_policy":\n                "WOMD_tracks_to_predict_evaluator_only",\n\n            "primary_target_count":\n                len(primary_indices),''',
    '''            "primary_target_policy":\n                "WOMD_tracks_to_predict_evaluator_only_"\n                "with_current_real_nontarget_ignore",\n\n            "primary_target_count":\n                len(primary_indices),\n\n            "primary_ignored_real_nontarget_count":\n                len(ignored_indices),''',
)
replace_once(
    runner,
    '''        "false_predictions": 0,\n        "unmatched_truth": 0,''',
    '''        "false_predictions": 0,\n        "ignored_predictions": 0,\n        "unmatched_truth": 0,''',
)
replace_once(
    runner,
    '''    agg["false_predictions"] += (\n        report[\n            "false_prediction_count"\n        ]\n    )\n    agg["unmatched_truth"] += (''',
    '''    agg["false_predictions"] += (\n        report[\n            "false_prediction_count"\n        ]\n    )\n    agg["ignored_predictions"] += (\n        report.get(\n            "ignored_prediction_count",\n            0,\n        )\n    )\n    agg["unmatched_truth"] += (''',
)
replace_once(
    runner,
    '''        "false_prediction_count":\n            fp,\n        "unmatched_truth_count":''',
    '''        "false_prediction_count":\n            fp,\n        "ignored_prediction_count":\n            agg["ignored_predictions"],\n        "unmatched_truth_count":''',
)
replace_once(
    runner,
    '''        "target_policy":\n            "WOMD tracks_to_predict; "\n            "evaluator metadata only; "\n            "accessed after prediction",''',
    '''        "target_policy":\n            "WOMD tracks_to_predict; evaluator metadata only; "\n            "accessed after prediction; current-valid real "\n            "non-target actors are evaluator-ignored rather "\n            "than counted as false predictions",''',
)
replace_once(
    runner,
    '''        "reconstruction": (\n            "micro TP/FP/FN pooled over "\n            "the 120 frozen scenarios"\n        ),''',
    '''        "reconstruction": (\n            "micro TP/FP/FN pooled over the 120 frozen scenarios; "\n            "for the primary tracks_to_predict view, predictions "\n            "matched to current-valid real non-target actors are "\n            "ignored rather than counted as false positives"\n        ),''',
)
replace_once(
    runner,
    '''    "future_truth": {''',
    '''    "temporal_alignment": {\n        "prediction_anchor":\n            "global_WOMD_current_time_anchor",\n        "terminated_tracks_forecast":\n            False,\n        "active_stale_tracks":\n            "causally_open_loop_propagated_to_global_anchor",\n        "prediction_timestamp_contract":\n            "timestamp_s ~= anchor_timestamp_s + horizon_s",\n    },\n\n    "future_truth": {''',
)

# ------------------------------------------------------------------
# 6) Reproducibility gate: the fresh repaired reference run becomes the
#    reference; no stale pre-repair run hash may be hard-coded.
# ------------------------------------------------------------------
repro = ROOT / 'scripts/run_block38g_reproducibility_gate.py'
replace_once(
    repro,
    '''EXPECTED_RUN_SHA = (\n    "5fbb4642f87ab8e62cf1e1b3ef4214d9"\n    "2df5b5a1423bd9fb72b9733ced02e433"\n)\n\n''',
    '',
)
replace_once(
    repro,
    '''if (\n    reference_run_sha\n    !=\n    EXPECTED_RUN_SHA\n):\n    raise SystemExit(\n        "FAIL: frozen reference run SHA "\n        "does not match expected."\n    )\n\nif (\n    repeat_run_sha''',
    '''if (\n    len(reference_run_sha) != 64\n    or any(\n        char not in "0123456789abcdef"\n        for char in reference_run_sha\n    )\n):\n    raise SystemExit(\n        "FAIL: repaired reference run SHA is invalid."\n    )\n\nif (\n    repeat_run_sha''',
)

# ------------------------------------------------------------------
# 7) Final closure: derive repaired run SHA from the fresh reference and
#    expect the new regression count (old 106 + 3 repair tests = 109).
# ------------------------------------------------------------------
closure = ROOT / 'scripts/run_stage3_final_closure.py'
replace_once(
    closure,
    '''EXPECTED_RUN_SHA = (\n    "5fbb4642f87ab8e62cf1e1b3ef4214d9"\n    "2df5b5a1423bd9fb72b9733ced02e433"\n)\n\n''',
    'EXPECTED_TEST_COUNT = 109\n\n',
)
replace_once(
    closure,
    '''print(\n    "Blocks 3.8B–3.8G reports    = PASS"\n)\n\n\n# ============================================================\n# 4. Formal evaluation invariants''',
    '''print(\n    "Blocks 3.8B–3.8G reports    = PASS"\n)\n\nEXPECTED_RUN_SHA = (\n    loaded_reports["3.8G"]\n    ["reference"]\n    ["run_sha256"]\n)\n\nif (\n    not isinstance(EXPECTED_RUN_SHA, str)\n    or len(EXPECTED_RUN_SHA) != 64\n):\n    fail("invalid repaired reference run SHA")\n\n\n# ============================================================\n# 4. Formal evaluation invariants''',
)
replace_once(
    closure,
    '''if test_count != 106:\n    print(combined)\n\n    fail(\n        f"expected 106 tests, "\n        f"got {test_count}"\n    )\n\nprint(\n    "full Stage3 regression      = "\n    "106 / 106 PASS"\n)''',
    '''if test_count != EXPECTED_TEST_COUNT:\n    print(combined)\n\n    fail(\n        f"expected {EXPECTED_TEST_COUNT} tests, "\n        f"got {test_count}"\n    )\n\nprint(\n    "full Stage3 regression      = "\n    f"{test_count} / {test_count} PASS"\n)''',
)
replace_once(
    closure,
    '''    "regression": {\n        "tests_passed": 106,\n        "tests_total": 106,\n    },''',
    '''    "regression": {\n        "tests_passed": test_count,\n        "tests_total": test_count,\n    },''',
)
replace_once(
    closure,
    '''        "measurement_uncertainty_consumed":\n            True,\n\n        "future_truth_algorithm_input":''',
    '''        "measurement_uncertainty_consumed":\n            True,\n\n        "global_anchor_alignment_enforced":\n            True,\n\n        "terminated_tracks_forecast":\n            False,\n\n        "primary_real_nontarget_predictions_ignored":\n            True,\n\n        "estimated_association_semantics":\n            "deterministic_gated_greedy_nearest_neighbor",\n\n        "future_truth_algorithm_input":''',
)
replace_once(
    closure,
    '''print(\n    "full regression             = 106 / 106"\n)''',
    '''print(\n    "full regression             =",\n    f"{test_count} / {test_count}",\n)''',
)

# ------------------------------------------------------------------
# 8) Add 3 focused regression tests.
# ------------------------------------------------------------------
test_source = '''from __future__ import annotations\n\nfrom dataclasses import dataclass\nimport unittest\n\nfrom iscai_stage3.association.contracts import (\n    EstimatedAssociationResult,\n)\nfrom iscai_stage3.evaluation.anchor_alignment import (\n    align_state_prediction_to_global_anchor,\n)\nfrom iscai_stage3.evaluation.contracts import (\n    EvaluationConfig,\n    EvaluationPredictionSet,\n    EvaluationPredictionTrack,\n    EvaluationTruth,\n    EvaluationTruthTrack,\n    PredictionTrajectoryPoint,\n    TruthTrajectoryPoint,\n)\nfrom iscai_stage3.evaluation.metrics import (\n    evaluate_prediction_set,\n)\n\n\nclass TestStage3ScientificRepair(unittest.TestCase):\n    def _truth_track(self, truth_id, x):\n        return EvaluationTruthTrack(\n            truth_id=truth_id,\n            actor_class="TYPE_VEHICLE",\n            anchor_position_H0_m=(x, 0.0, 0.0),\n            points=(\n                TruthTrajectoryPoint(\n                    horizon_s=0.1,\n                    timestamp_s=1.1,\n                    position_H0_m=(x, 0.0, 0.0),\n                ),\n            ),\n        )\n\n    def _prediction_track(self, prediction_id, x):\n        return EvaluationPredictionTrack(\n            prediction_id=prediction_id,\n            anchor_position_H0_m=(x, 0.0, 0.0),\n            points=(\n                PredictionTrajectoryPoint(\n                    horizon_s=0.1,\n                    timestamp_s=1.1,\n                    position_H0_m=(x, 0.0, 0.0),\n                ),\n            ),\n        )\n\n    def test_prediction_set_rejects_stale_timestamp_relabel(self):\n        track = EvaluationPredictionTrack(\n            prediction_id="stale",\n            anchor_position_H0_m=(0.0, 0.0, 0.0),\n            points=(\n                PredictionTrajectoryPoint(\n                    horizon_s=0.1,\n                    timestamp_s=0.9,\n                    position_H0_m=(0.0, 0.0, 0.0),\n                ),\n            ),\n        )\n\n        with self.assertRaises(ValueError):\n            EvaluationPredictionSet(\n                method="CV",\n                scenario_id="s",\n                anchor_timestamp_s=1.0,\n                tracks=(track,),\n            )\n\n    def test_real_non_target_prediction_is_ignored_not_false_positive(self):\n        predictions = EvaluationPredictionSet(\n            method="CV",\n            scenario_id="s",\n            anchor_timestamp_s=1.0,\n            tracks=(\n                self._prediction_track("target", 0.0),\n                self._prediction_track("real-nontarget", 10.0),\n                self._prediction_track("clutter", 100.0),\n            ),\n        )\n\n        primary = EvaluationTruth(\n            scenario_id="s",\n            anchor_timestamp_s=1.0,\n            tracks=(self._truth_track("truth-target", 0.0),),\n        )\n\n        ignored = EvaluationTruth(\n            scenario_id="s",\n            anchor_timestamp_s=1.0,\n            tracks=(self._truth_track("truth-other", 10.0),),\n        )\n\n        report = evaluate_prediction_set(\n            predictions,\n            primary,\n            config=EvaluationConfig(\n                horizons_s=(0.1,),\n                anchor_assignment_gate_m=5.0,\n                endpoint_miss_threshold_m=2.0,\n            ),\n            ignored_truth=ignored,\n        )\n\n        self.assertEqual(report.matched_track_count, 1)\n        self.assertEqual(report.ignored_prediction_count, 1)\n        self.assertEqual(report.false_prediction_count, 1)\n        self.assertEqual(report.ignored_prediction_ids, ("real-nontarget",))\n        self.assertAlmostEqual(report.reconstruction_precision, 0.5)\n        self.assertAlmostEqual(report.reconstruction_recall, 1.0)\n\n    def test_stale_active_state_is_propagated_to_global_anchor(self):\n        @dataclass(frozen=True)\n        class State:\n            track_id: str = "x"\n            timestamp_s: float = 0.8\n            position_H0_m: tuple[float, float, float] = (0.0, 0.0, 0.0)\n\n        @dataclass(frozen=True)\n        class Point:\n            horizon_s: float\n            timestamp_s: float\n            position_H0_m: tuple[float, float, float]\n\n        @dataclass(frozen=True)\n        class Prediction:\n            track_id: str\n            points: tuple[Point, ...]\n\n        def predictor(state, *, horizons_s):\n            return Prediction(\n                track_id=state.track_id,\n                points=tuple(\n                    Point(\n                        horizon_s=float(tau),\n                        timestamp_s=state.timestamp_s + float(tau),\n                        position_H0_m=(float(tau), 0.0, 0.0),\n                    )\n                    for tau in horizons_s\n                ),\n            )\n\n        result = align_state_prediction_to_global_anchor(\n            State(),\n            predictor,\n            horizons_s=(0.1, 0.3),\n            anchor_timestamp_s=1.0,\n        )\n\n        self.assertAlmostEqual(result.anchor_position_H0_m[0], 0.2)\n        self.assertEqual(tuple(p.horizon_s for p in result.points), (0.1, 0.3))\n        self.assertEqual(tuple(p.timestamp_s for p in result.points), (1.1, 1.3))\n        self.assertEqual(\n            EstimatedAssociationResult(scenario_id="s", frames=(), tracks=()).method,\n            "deterministic_gated_greedy_nearest_neighbor",\n        )\n\n\nif __name__ == "__main__":\n    unittest.main()\n'''
write(ROOT / 'tests/test_stage3_repair_semantics.py', test_source)

# ------------------------------------------------------------------
# 9) Syntax compile only. Scientific checking is deliberately separate.
# ------------------------------------------------------------------
import py_compile
changed_paths = [ROOT / rel for rel in EXPECTED]
changed_paths += [
    ROOT / 'src/iscai_stage3/evaluation/anchor_alignment.py',
    ROOT / 'tests/test_stage3_repair_semantics.py',
]
for p in changed_paths:
    if p.suffix == '.py':
        py_compile.compile(str(p), doraise=True)

post = {str(p.relative_to(ROOT)): digest(p) for p in changed_paths if p.is_file()}
report = {
    'stage': 3,
    'repair': 'stage3_scientific_audit_v1',
    'status': 'APPLIED',
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'changes': [
        'rename estimated association semantics from global to greedy nearest-neighbor',
        'enforce global-anchor prediction timestamp contract',
        'exclude terminated tracks and propagate still-active stale tracks causally to global anchor',
        'ignore current-valid real non-target actors in primary reconstruction false-positive accounting',
        'remove stale pre-repair run SHA from reproducibility/closure gates',
        'add three focused regression tests',
    ],
    'pre_repair_backup': str(BACKUP),
    'post_sha256': post,
    'scientific_evaluation_run': False,
    'formal_metrics_recomputed': False,
}
write(REPORT, json.dumps(report, indent=2, sort_keys=True) + '\n')

print('============================================================')
print('STAGE 3 SCIENTIFIC REPAIR V1 = APPLIED')
print('============================================================')
print('root   =', ROOT)
print('backup =', BACKUP)
print('report =', REPORT)
print('formal evaluation run = NO')
print('next = run independent Stage3 repair check')
