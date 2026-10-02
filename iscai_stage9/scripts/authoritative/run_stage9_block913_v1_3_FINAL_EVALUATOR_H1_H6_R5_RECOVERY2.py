#!/home/agni/waymo/iscai_stage1/.venv_lidar/bin/python
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path('/home/agni/waymo')
S9 = ROOT / 'iscai_stage9'
FORMAL_N = 26209
FORMAL_IDS_SHA = '7d3066f08c466cfd2cba64eecdd24474a7e83130a4e90d77e9cdee349e22f03f'

# Exact successful combined pre-outcome closure from the user's R3.1 run.
PHASE3_ROOT = S9 / 'artifacts/block913_v1_3_c3_phase3_DIRECT_FAST_SHARDED_R3_EXACT_OCC'
COMBINED = PHASE3_ROOT / 'COMBINED_PRE_OUTCOME_SEAL.json'
COMBINED_SHA = '1dcf287cd7a947579b7d63b69364bfb32a93d13a7846b9156092cf9042222af7'
C3_MANIFEST = PHASE3_ROOT / 'stage9_block913_v1_3_c3_phase3_DIRECT_manifest.jsonl'
C3_MANIFEST_SHA = '53e2f47dd218e3fd5dea2fb18e714ac68c4f4c8fbefd02fd4b1aa753774e32a5'
C3_SEAL = PHASE3_ROOT / 'stage9_block913_v1_3_c3_phase3_DIRECT_seal.json'
C3_SEAL_SHA = '33068597382dfe853b73b0180f44f74d586171fc0212f7e381774de0ff6ba0d3'

R4_DIR = S9 / 'artifacts/block913_v1_3_primary_pretruth_R4'
R4_MANIFEST = R4_DIR / 'stage9_block913_v1_3_pretruth_by_scenario.jsonl'
R4_MANIFEST_SHA = '54ad640432c648155d0f3ee1a5dc60d085f74dc4385b8d71a34a047a84a4ea60'
R4_SEAL = R4_DIR / 'stage9_block913_v1_3_pretruth_seal.json'
R4_SEAL_SHA = 'ab3c20b7282c8e3a2978f25e3c16e7d1ffe700354a27b8a96f7049f45cb3090f'
R4_SOURCE = S9 / 'scripts/run_stage9_block913_v1_3_primary_pretruth_R4.py'
R4_SOURCE_SHA = '5b1dd54f8be489c6c727842b719eca5da76c564e44070960b9eb799ab9970050'

P2A_DIR = S9 / 'artifacts/block913_v1_3_c3_pretruth_phase2A_R1_phase1_seal_binding_repair'
P2A_MANIFEST = P2A_DIR / 'stage9_block913_v1_3_c3_phase2A_R1_gaussian_carrier_manifest.jsonl'
P2A_MANIFEST_SHA = '62b1e8d090614cf3ef8d20f4cb7d69296207c3f171d90647ec010f99e4c1ca0a'

ORIGINAL_EVALUATOR = S9 / 'scripts/run_stage9_block913_v1_1_formal_womd_c1_c2_outcome_materialization.py'
ORIGINAL_EVALUATOR_SHA = 'ac618eafeaa7884979834ef6dcce772bf8d24f59cb6632de993c3efb6bf76322'
BLOCK94B_SHA = 'f4d303168770fcfc7fb3745617cd95284d0f5d3fba610117f419e92c0106ece7'

P0_HELPER = S9 / 'scripts/run_stage9_v1_2_block911b_r3s5r2_phase0_resume.py'
P0_HELPER_SHA = 'e101fd7ef6cea006a6a97e7db802f399cc455fa11ff9ff2dc299755ae4907c3a'
R31_SOURCE = S9 / 'scripts/run_stage9_block913_phase3_DIRECT_FAST_SHARDED_R3_1_EXACT_OCC_ZERO_CARRIER.py'
R31_SOURCE_SHA = '4c2d965acae31e240fb8a45021ee28253e4242b6fdbfaefbee8166ddeb8e3139'
BLOCK94B_REPAIR_SOURCE = S9 / 'scripts/run_stage9_block913c_R3_reachable_block94b_evaluator_repair.py'
BLOCK94B_REPAIR_SOURCE_SHA = '943078fc1c460453391b39e0a58b3ff774df6f97a51bbf16417c9659c5d56300'

FORMAL_IDS = S9 / 'artifacts/block91b5_opaque_scenario_id_cohort_freeze/stage9_FORMAL_scenario_ids.txt'
OUT_DEFAULT = S9 / 'artifacts/block913_v1_3_FINAL_EVALUATOR_H1_H6_R3'

# Documented continuation of the failed R3 evaluator attempt in the SAME
# artifact namespace. R3 opened evaluator-only future GT for exactly ordinals
# 0..11, produced zero completed outcomes, and then failed in a legacy H1
# transform adapter. Those 12 scenes are authorized for ONE recovery re-read.
# A third future-GT read remains fail-closed.
FAILED_R3_RUNNER_SHA = 'b90c39e6947f94c05552c27569b5f785a854b7db2d8d5085235ff85360a85a0a'
RECOVERY_AMENDMENT = 'FINAL_EVALUATOR_TECHNICAL_RECOVERY_AMENDMENT.json'
RECOVERY1_AMENDMENT_SHA = '021f9e1b200fb8b4fed2b5fdbf22fc224740f670d9aff647b16c4c721cfc0756'
RECOVERY2_AMENDMENT = 'FINAL_EVALUATOR_TECHNICAL_RECOVERY_AMENDMENT_R2.json'
RECOVERY_EXPECTED = (
    (0, '10040e572b831a04'),
    (1, '10067cf7cc2506c7'),
    (2, '10083669957ee5f8'),
    (3, '1008b7b63e2d60'),
    (4, '1008f05c233dd975'),
    (5, '100d033b60683a9f'),
    (6, '100f370df1797a88'),
    (7, '100f9b9f8af6036f'),
    (8, '1010cc7e3a91ebc5'),
    (9, '10195df1c4a2c3ad'),
    (10, '101a844960d63c3f'),
    (11, '101b00dd28e01037'),
)

POLICY_SELECTION_SHA = '57efdd41127dbb7c6987cdb5cdf438c637ccdb179da082b4b796a65af347c4bc'
SCOPE_AMEND_SHA = '9ee691b255a60be24c8c9be6de833fd187cdd0a61941e801cb64b0b2a325d849'
SCOPE_SEAL_SHA = 'eabde71a32a924bf2416c3605e8f7087c2bbfb9ad61ecf72ed3fefaf2a1ff9ac'
EXEC_CONTRACT_SHA = '991349ff9029b197652411e7ac60767458debf15674fcd103bfa56cb16288001'

class FailClosed(RuntimeError):
    pass

def req(c: bool, m: str) -> None:
    if not c:
        raise FailClosed(m)

def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def canonical_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=True) + '\n').encode('utf-8')

def native(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [native(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x

def atomic_json(path: Path, obj: Any, *, allow_existing_equal: bool = False) -> None:
    path = Path(path)
    data = canonical_bytes(native(obj))
    if path.exists():
        if allow_existing_equal:
            req(path.read_bytes() == data, f'existing JSON differs: {path}')
            return
        raise FailClosed(f'refusing overwrite: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.tmp.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        req(not path.exists(), f'target appeared before commit: {path}')
        os.replace(tmp, path)
        os.chmod(path, 0o444)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def load_module(path: Path, name: str):
    path = Path(path)
    req(path.is_file(), f'module missing: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    req(spec is not None and spec.loader is not None, f'cannot spec {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())

def iter_jsonl(path: Path):
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def action_paths(ordinal: int, sid: str) -> tuple[Path, Path]:
    stem = f'{ordinal:05d}_{sid}'
    d = PHASE3_ROOT / 'scenario_actions'
    return d / f'{stem}.json', d / f'{stem}.npz'

def outcome_path(out: Path, ordinal: int, sid: str) -> Path:
    return out / 'scenario_outcomes' / f'{ordinal:05d}_{sid}.json'

def opened_path(out: Path, ordinal: int, sid: str) -> Path:
    return out / 'gt_open_intent' / f'{ordinal:05d}_{sid}.json'

def recovery_reopen_path(out: Path, ordinal: int, sid: str) -> Path:
    return out / 'gt_reopen_recovery' / f'{ordinal:05d}_{sid}.json'

def recovery_amendment_path(out: Path) -> Path:
    return out / RECOVERY_AMENDMENT

def recovery2_reopen_path(out: Path, ordinal: int, sid: str) -> Path:
    return out / 'gt_reopen_recovery2' / f'{ordinal:05d}_{sid}.json'

def recovery2_amendment_path(out: Path) -> Path:
    return out / RECOVERY2_AMENDMENT

def resolve_p2a_scene(row: dict) -> Path:
    p = P2A_DIR / str(row['scenario_cache_relpath'])
    req(p.is_file(), f'Phase2A carrier missing: {p}')
    req(sha256_path(p) == str(row['scenario_cache_sha256']), f'Phase2A carrier SHA drift: {p}')
    return p

def formal_ids() -> list[str]:
    req(FORMAL_IDS.is_file() and sha256_path(FORMAL_IDS) == FORMAL_IDS_SHA, 'FORMAL ID authority drift')
    ids = [x.strip() for x in FORMAL_IDS.read_text().splitlines() if x.strip()]
    req(len(ids) == FORMAL_N and len(set(ids)) == FORMAL_N, 'FORMAL IDs count/uniqueness drift')
    return ids

def verify_combined_gate() -> dict:
    req(COMBINED.is_file() and sha256_path(COMBINED) == COMBINED_SHA, 'combined pre-outcome seal drift')
    req(C3_MANIFEST.is_file() and sha256_path(C3_MANIFEST) == C3_MANIFEST_SHA, 'C3 manifest drift')
    req(C3_SEAL.is_file() and sha256_path(C3_SEAL) == C3_SEAL_SHA, 'C3 seal drift')
    req(R4_MANIFEST.is_file() and sha256_path(R4_MANIFEST) == R4_MANIFEST_SHA, 'R4 manifest drift')
    req(R4_SEAL.is_file() and sha256_path(R4_SEAL) == R4_SEAL_SHA, 'R4 seal drift')
    req(P2A_MANIFEST.is_file() and sha256_path(P2A_MANIFEST) == P2A_MANIFEST_SHA, 'Phase2A manifest drift')
    j = load_json(COMBINED)
    req(j.get('FUTURE_GT_MAY_OPEN') is True, 'combined seal does not authorize future GT')
    req(j.get('H5_H6_tested') is False, 'H5/H6 already tested before final evaluator')
    req(j.get('future_GT_access') is False, 'future GT already accessed before final evaluator')
    req(j.get('FORMAL_outcomes_computed') is False, 'FORMAL outcomes already computed before final evaluator')
    req(j.get('FORMAL_scene_count') == FORMAL_N, 'combined FORMAL scene count drift')
    req(j.get('FORMAL_ids_sha256') == FORMAL_IDS_SHA, 'combined FORMAL IDs drift')
    req(j.get('primary_confirmatory') == 'C1+C2', 'primary scope drift')
    req(j.get('secondary_FORMAL') == 'C3_PROFILE_SENSITIVITY', 'secondary scope drift')
    req(j.get('selected_C3_policy_sha256') == POLICY_SELECTION_SHA, 'C3 policy binding drift')
    req(j.get('stage9_block912_v1_2_scope_amendment_sha256') == SCOPE_AMEND_SHA, 'scope amendment drift')
    req(j.get('stage9_block912_v1_2_scope_seal_sha256') == SCOPE_SEAL_SHA, 'scope seal drift')
    req(j.get('stage9_block913_secondary_C3_execution_contract_sha256') == EXEC_CONTRACT_SHA, 'C3 execution contract drift')
    req(j.get('R4_primary_pretruth_manifest_sha256') == R4_MANIFEST_SHA, 'combined->R4 manifest drift')
    req(j.get('R4_primary_pretruth_seal_sha256') == R4_SEAL_SHA, 'combined->R4 seal drift')
    req(j.get('phase2A_manifest_sha256') == P2A_MANIFEST_SHA, 'combined->Phase2A drift')
    req(j.get('C3_phase3_DIRECT_manifest_sha256') == C3_MANIFEST_SHA, 'combined->C3 manifest drift')
    req(j.get('C3_phase3_DIRECT_seal_sha256') == C3_SEAL_SHA, 'combined->C3 seal drift')
    return j

def verify_r4_row(row: dict, ordinal: int, sid: str) -> None:
    req(int(row.get('formal_ordinal', -1)) == ordinal and str(row.get('scenario_id')) == sid, f'R4 identity drift {ordinal}/{sid}')
    req(row.get('future_GT_access') is False and row.get('FORMAL_outcomes_computed') is False, f'R4 crossed outcome boundary {sid}')
    req(row.get('future_truth_receiver_selection') is False, f'R4 future receiver selection {sid}')
    req(row.get('tracks_to_predict_receiver_selection') is False, f'R4 tracks-to-predict receiver selection {sid}')
    rec = row.get('receiver', {})
    req(rec.get('status') in ('SELECTED', 'NO_ELIGIBLE_RECEIVER'), f'R4 receiver status drift {sid}')
    if rec.get('status') == 'SELECTED':
        req(rec.get('selected') is True and rec.get('prediction_id') is not None, f'R4 selected receiver malformed {sid}')
    else:
        req(rec.get('selected') is False and rec.get('prediction_id') is None, f'R4 no-receiver malformed {sid}')
    # R4 hashes the object before this field is added.
    stored = str(row.get('pretruth_decision_sha256'))
    tmp = dict(row); tmp.pop('pretruth_decision_sha256', None)
    actual = hashlib.sha256(canonical_bytes(tmp)[:-1]).hexdigest()
    req(actual == stored, f'R4 pretruth decision SHA drift {sid}')

def verify_action_meta(meta: dict, ordinal: int, sid: str, npz: Path, *, hash_npz: bool = False) -> None:
    req(meta.get('schema') == 'stage9_block913_v1_3_c3_phase3_DIRECT_action_scene_v2', f'C3 action schema drift {sid}')
    req(meta.get('status') == 'FROZEN_C3_DIRECT_CAUSAL_ACTION_PREOUTCOME', f'C3 action status drift {sid}')
    req(int(meta.get('formal_ordinal', -1)) == ordinal and str(meta.get('scenario_id')) == sid, f'C3 action identity drift {sid}')
    req(meta.get('candidate_id') == 'V5_RHO2_DIRECT_RAW_PRIORITY', f'C3 candidate drift {sid}')
    req(float(meta.get('rho_dim_per_s')) == 2.0 and float(meta.get('rho_bright_per_s')) == 1.0, f'C3 rates drift {sid}')
    req(meta.get('receiver_role_binding_policy') == 'CURRENT_RECIPROCAL_ASSOCIATION_TYPE_VEHICLE', f'C3 role policy drift {sid}')
    req(meta.get('H5_H6_tested') is False and meta.get('future_GT_access') is False and meta.get('FORMAL_outcomes_computed') is False, f'C3 action crossed outcome boundary {sid}')
    req(meta.get('structural', {}).get('pass') is True, f'C3 structural fail {sid}')
    req(npz.is_file(), f'C3 action NPZ missing {sid}')
    if hash_npz:
        req(meta.get('npz_sha256') == sha256_path(npz), f'C3 action NPZ SHA drift {sid}')



def prepare_recovery2(out: Path) -> None:
    """Freeze the second technical-recovery amendment without reading a scenario.

    Expected state after R4-RECOVERY:
      * 42 original GT-intent markers
      * 12 first-recovery reread markers
      * 30 completed outcomes
      * 12 pending scenes
      * among pending: two already have a first-recovery marker (next read #3),
        ten have only the original intent marker (next read #2)
    """
    verify_combined_gate()
    prev = recovery_amendment_path(out)
    req(prev.is_file(), 'first technical recovery amendment missing')
    req(sha256_path(prev) == RECOVERY1_AMENDMENT_SHA,
        'first technical recovery amendment SHA drift')

    pre = out / 'FINAL_EVALUATOR_PREFLIGHT_SEAL.json'
    req(pre.is_file(), 'R3 preflight seal missing')
    pobj = load_json(pre)
    req(str(pobj.get('final_evaluator_runner_sha256')) == FAILED_R3_RUNNER_SHA,
        'preflight is not bound to failed R3 runner')

    ids = formal_ids()
    outcome_dir = out / 'scenario_outcomes'
    intent_dir = out / 'gt_open_intent'
    r1dir = out / 'gt_reopen_recovery'
    r2dir = out / 'gt_reopen_recovery2'
    r2dir.mkdir(parents=True, exist_ok=True)

    outcomes = sorted(outcome_dir.glob('*.json'))
    intents = sorted(intent_dir.glob('*.json'))
    r1markers = sorted(r1dir.glob('*.json')) if r1dir.is_dir() else []
    r2markers = sorted(r2dir.glob('*.json'))

    req(len(outcomes) == 30, f'R4 recovery outcome cardinality drift: {len(outcomes)} != 30')
    req(len(intents) == 42, f'R4 recovery original-intent cardinality drift: {len(intents)} != 42')
    req(len(r1markers) == 12, f'R4 recovery reread-marker cardinality drift: {len(r1markers)} != 12')
    req(len(r2markers) == 0, 'second-recovery markers already exist; do not authorize another attempt')

    pending = []
    for ordinal, sid in enumerate(ids):
        gp = opened_path(out, ordinal, sid)
        op = outcome_path(out, ordinal, sid)
        if not gp.exists():
            continue
        if op.exists():
            continue
        r1p = recovery_reopen_path(out, ordinal, sid)
        prior_reads = 1 + int(r1p.exists())
        req(prior_reads in (1, 2), f'unexpected prior physical-read count {ordinal}/{sid}/{prior_reads}')
        pending.append({
            'formal_ordinal': int(ordinal),
            'scenario_id': str(sid),
            'prior_physical_future_GT_read_count': int(prior_reads),
            'authorized_next_physical_future_GT_read_number': int(prior_reads + 1),
            'prior_gt_open_intent_sha256': sha256_path(gp),
            'prior_recovery1_marker_sha256': sha256_path(r1p) if r1p.exists() else None,
        })

    req(len(pending) == 12, f'pending technical-failure cardinality drift: {len(pending)} != 12')
    n_prior2 = sum(int(x['prior_physical_future_GT_read_count'] == 2) for x in pending)
    n_prior1 = sum(int(x['prior_physical_future_GT_read_count'] == 1) for x in pending)
    req((n_prior1, n_prior2) == (10, 2),
        f'pending recovery-depth drift: prior1={n_prior1} prior2={n_prior2}, expected 10/2')

    amendment = {
        'schema': 'stage9_block913_v1_3_final_evaluator_technical_recovery_amendment_v2',
        'status': 'FROZEN_TECHNICAL_RECOVERY2_AUTHORIZATION_BEFORE_REREAD',
        'combined_preoutcome_seal_sha256': COMBINED_SHA,
        'failed_original_runner_sha256': FAILED_R3_RUNNER_SHA,
        'previous_recovery_amendment_sha256': RECOVERY1_AMENDMENT_SHA,
        'recovery2_runner_sha256': sha256_path(Path(__file__).resolve()),
        'completed_outcome_count_at_amendment': len(outcomes),
        'original_gt_intent_count_at_amendment': len(intents),
        'recovery1_reread_marker_count_at_amendment': len(r1markers),
        'authorized_recovery2_scene_count': len(pending),
        'authorized_recovery2_scenes': pending,
        'root_cause': (
            'final evaluator incorrectly required every C3 target to belong to the '
            'Stage6/H6 matched-eligible set. Phase3 C3 target authority is broader: '
            'CURRENT_RECIPROCAL_ASSOCIATION_TYPE_VEHICLE. H6 historical metric authority '
            'remains the matched-eligible set. The two role sets must be validated '
            'separately rather than forced equal.'
        ),
        'H6_metric_definition_changed': False,
        'H6_matched_eligible_universe_changed': False,
        'C3_target_universe_changed': False,
        'validation_repair_only': True,
        'policy_retuned': False,
        'thresholds_retuned': False,
        'baselines_changed': False,
        'C3_reselected': False,
        'outcome_based_tuning': False,
        'future_GT_controller_access': False,
        'literal_one_physical_read_per_scene_invariant_preserved': False,
        'max_authorized_physical_future_GT_read_number': 3,
        'fourth_read_authorized': False,
    }
    atomic_json(recovery2_amendment_path(out), amendment, allow_existing_equal=True)
    print('=' * 118)
    print('FINAL_EVALUATOR_TECHNICAL_RECOVERY2_AMENDMENT = FROZEN')
    print('COMPLETED_OUTCOMES_AT_AMENDMENT =', len(outcomes))
    print('AUTHORIZED_RECOVERY2_SCENES =', len(pending))
    print('NEXT_READ_2_SCENES =', n_prior1)
    print('NEXT_READ_3_SCENES =', n_prior2)
    print('H6_METRIC_DEFINITION_CHANGED = FALSE')
    print('POLICY_THRESHOLD_BASELINE_CHANGE = FALSE')
    print('RECOVERY2_AMENDMENT_SHA256 =', sha256_path(recovery2_amendment_path(out)))
    print('=' * 118, flush=True)

def prepare_recovery(out: Path) -> None:
    """Freeze a technical-recovery amendment without reading any scenario."""
    verify_combined_gate()
    pre = out / 'FINAL_EVALUATOR_PREFLIGHT_SEAL.json'
    req(pre.is_file(), 'R3 preflight seal missing')
    pobj = load_json(pre)
    req(pobj.get('status') == 'FROZEN_COMPLETE_FINAL_EVALUATOR_PRE_GT_PREFLIGHT',
        'R3 preflight status drift')
    req(pobj.get('future_GT_access') is False and pobj.get('FORMAL_outcomes_computed') is False,
        'R3 preflight crossed GT/outcome boundary')
    req(str(pobj.get('final_evaluator_runner_sha256')) == FAILED_R3_RUNNER_SHA,
        'preflight is not bound to the failed R3 runner')

    outcome_dir = out / 'scenario_outcomes'
    intent_dir = out / 'gt_open_intent'
    req(outcome_dir.is_dir() and intent_dir.is_dir(), 'R3 evaluator directories missing')
    outcomes = sorted(outcome_dir.glob('*.json'))
    intents = sorted(intent_dir.glob('*.json'))
    req(len(outcomes) == 0, f'R3 recovery requires zero completed outcomes, found {len(outcomes)}')
    req(len(intents) == len(RECOVERY_EXPECTED),
        f'R3 recovery intent cardinality drift: {len(intents)} != {len(RECOVERY_EXPECTED)}')

    ids = formal_ids()
    authorized = []
    for ordinal, sid in RECOVERY_EXPECTED:
        req(ids[int(ordinal)] == str(sid), f'R3 recovery FORMAL identity drift {ordinal}/{sid}')
        gp = opened_path(out, int(ordinal), str(sid))
        req(gp.is_file(), f'expected stale GT intent missing {ordinal}/{sid}')
        j = load_json(gp)
        req(int(j.get('formal_ordinal', -1)) == int(ordinal) and str(j.get('scenario_id')) == str(sid),
            f'stale GT intent identity drift {ordinal}/{sid}')
        req(str(j.get('combined_preoutcome_seal_sha256')) == COMBINED_SHA,
            f'stale GT intent combined-seal drift {sid}')
        req(j.get('future_GT_controller_access') is False,
            f'stale GT intent controller-access drift {sid}')
        authorized.append({
            'formal_ordinal': int(ordinal),
            'scenario_id': str(sid),
            'prior_gt_open_intent_sha256': sha256_path(gp),
        })

    expected_names = {f'{int(o):05d}_{sid}.json' for o, sid in RECOVERY_EXPECTED}
    req({p.name for p in intents} == expected_names, 'unexpected R3 GT-intent files present')

    reread_dir = out / 'gt_reopen_recovery'
    reread_dir.mkdir(parents=True, exist_ok=True)
    req(not any(reread_dir.glob('*.json')),
        'recovery re-read markers already exist; do not authorize another recovery attempt')

    amendment = {
        'schema': 'stage9_block913_v1_3_final_evaluator_technical_recovery_amendment_v1',
        'status': 'FROZEN_TECHNICAL_RECOVERY_AUTHORIZATION_BEFORE_REREAD',
        'failed_runner_sha256': FAILED_R3_RUNNER_SHA,
        'recovery_runner_sha256': sha256_path(Path(__file__).resolve()),
        'failed_attempt_preflight_seal_sha256': sha256_path(pre),
        'combined_preoutcome_seal_sha256': COMBINED_SHA,
        'stale_gt_intent_count': len(authorized),
        'completed_outcome_count_at_amendment': 0,
        'authorized_recovery_reread_count': len(authorized),
        'authorized_recovery_scenes': authorized,
        'root_cause': (
            'legacy_H1_anchor_transform_expected_numeric_matrix_but_current_real_pipeline_'
            'provides_RigidTransform; failure occurred after evaluator-only scenario read '
            'and before any completed outcome artifact'
        ),
        'H1_transform_repair': (
            'use the already-authoritative RigidTransform T_H0_from_W with '
            'iscai_stage4.data.supervision._point_H0 and '
            'criticality_geometry_v1.transform_heading_with_rigid_transform; '
            'receiver-offset seed/distribution/geometry remain unchanged'
        ),
        'policy_retuned': False,
        'thresholds_retuned': False,
        'baselines_changed': False,
        'C3_reselected': False,
        'outcome_based_tuning': False,
        'future_GT_controller_access': False,
        'literal_one_physical_read_per_scene_invariant_preserved': False,
        'documented_physical_reread_scene_count': len(authorized),
        'third_read_authorized': False,
    }
    atomic_json(recovery_amendment_path(out), amendment, allow_existing_equal=True)
    print('=' * 118)
    print('FINAL_EVALUATOR_TECHNICAL_RECOVERY_AMENDMENT = FROZEN')
    print('AUTHORIZED_REREAD_SCENES =', len(authorized))
    print('COMPLETED_OUTCOMES_AT_AMENDMENT = 0')
    print('POLICY_THRESHOLD_BASELINE_CHANGE = FALSE')
    print('RECOVERY_AMENDMENT_SHA256 =', sha256_path(recovery_amendment_path(out)))
    print('=' * 118, flush=True)

def preflight(out: Path) -> None:
    print('=' * 118)
    print('STAGE 9.13 FINAL EVALUATOR R5-RECOVERY2 — GLOBAL PRE-GT PREFLIGHT')
    print('NO FUTURE GT / NO OUTCOMES / NO MODEL FORWARD / NO MC / NO C2 SOLVER')
    print('=' * 118, flush=True)
    verify_combined_gate()
    ids = formal_ids()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'scenario_outcomes').mkdir(exist_ok=True)
    (out / 'gt_open_intent').mkdir(exist_ok=True)
    pre = out / 'FINAL_EVALUATOR_PREFLIGHT_SEAL.json'
    final = out / 'stage9_block913_v1_3_FINAL_H1_H6_seal.json'
    req(not final.exists(), 'final seal already exists')

    r4_it = iter_jsonl(R4_MANIFEST)
    p2a_it = iter_jsonl(P2A_MANIFEST)
    c3_it = iter_jsonl(C3_MANIFEST)
    selected = 0; no_receiver = 0
    for i, sid in enumerate(ids):
        try:
            r4 = next(r4_it); pa = next(p2a_it); cm = next(c3_it)
        except StopIteration:
            raise FailClosed(f'manifest ended before ordinal {i}')
        verify_r4_row(r4, i, sid)
        req(int(pa.get('formal_ordinal', -1)) == i and str(pa.get('scenario_id')) == sid, f'Phase2A order drift {sid}')
        req(int(cm.get('formal_ordinal', -1)) == i and str(cm.get('scenario_id')) == sid, f'C3 manifest order drift {sid}')
        jp, npz = action_paths(i, sid)
        req(jp.is_file() and npz.is_file(), f'C3 action pair missing {i}/{sid}')
        req(str(cm.get('json_sha256')) == sha256_path(jp), f'C3 action JSON SHA drift {sid}')
        meta = load_json(jp)
        verify_action_meta(meta, i, sid, npz, hash_npz=False)
        # The sealed manifest must bind the NPZ hash recorded in action metadata.
        req(str(cm.get('npz_sha256')) == str(meta.get('npz_sha256')), f'C3 manifest/action NPZ binding drift {sid}')
        p2ap = P2A_DIR / str(pa['scenario_cache_relpath'])
        req(p2ap.is_file(), f'Phase2A carrier missing {sid}')
        if r4['receiver']['status'] == 'SELECTED': selected += 1
        else: no_receiver += 1
        if (i + 1) % 2000 == 0 or i + 1 == FORMAL_N:
            print(f'PREFLIGHT {i+1}/{FORMAL_N} selected={selected} no_receiver={no_receiver}', flush=True)
    try: next(r4_it); raise FailClosed('R4 manifest has extra rows')
    except StopIteration: pass
    try: next(p2a_it); raise FailClosed('Phase2A manifest has extra rows')
    except StopIteration: pass
    try: next(c3_it); raise FailClosed('C3 manifest has extra rows')
    except StopIteration: pass
    req(selected == 25239 and no_receiver == 970, f'R4 selected/no-receiver drift {selected}/{no_receiver}')
    obj = {
        'schema': 'stage9_block913_v1_3_final_evaluator_preflight_seal_v1',
        'status': 'FROZEN_COMPLETE_FINAL_EVALUATOR_PRE_GT_PREFLIGHT',
        'FORMAL_scene_count': FORMAL_N,
        'FORMAL_ids_sha256': FORMAL_IDS_SHA,
        'combined_preoutcome_seal_sha256': COMBINED_SHA,
        'R4_manifest_sha256': R4_MANIFEST_SHA,
        'Phase2A_manifest_sha256': P2A_MANIFEST_SHA,
        'C3_manifest_sha256': C3_MANIFEST_SHA,
        'C3_seal_sha256': C3_SEAL_SHA,
        'final_evaluator_runner_sha256': sha256_path(Path(__file__).resolve()),
        'block94b_evaluator_source_sha256': BLOCK94B_SHA,
        'block94b_reachable_repair_source_sha256': BLOCK94B_REPAIR_SOURCE_SHA,
        'primary_receiver_selected': selected,
        'no_eligible_receiver': no_receiver,
        'future_GT_access': False,
        'FORMAL_outcomes_computed': False,
        'H1_H6_tested': False,
        'policy_or_threshold_change': False,
        'FUTURE_GT_MAY_OPEN': True,
    }
    atomic_json(pre, obj, allow_existing_equal=True)
    print('FINAL_EVALUATOR_PREFLIGHT = PASS')
    print('PREFLIGHT_SHA256 =', sha256_path(pre), flush=True)

BLOCK94B_REACHABLE_EVALUATOR_FUNCTIONS = (
    'actor_label_row',
    'current_prediction_histories',
    'evaluator_matches',
    'future_ego_geometry',
    'req',
)

def load_block94b_reachable_evaluator(path: Path):
    """Load only the exact evaluator-reachable Block94B source closure.

    This mirrors the already-audited 9.13C-R3 dependency repair: no SciPy
    import is executed, no fitting/calibration code is loaded, and the exact
    source AST nodes for the evaluator-only closure are compiled unchanged.
    """
    path = Path(path)
    req(path.is_file(), f'Block94B source missing: {path}')
    req(sha256_path(path) == BLOCK94B_SHA, 'Block94B evaluator drift')
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    funcs = {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    classes = {n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
    roots = ('current_prediction_histories','evaluator_matches','future_ego_geometry','actor_label_row')
    for name in roots:
        req(name in funcs, f'Block94B evaluator function absent: {name}')
    req('FailClosed' in classes, 'Block94B FailClosed class absent')

    def direct_local_calls(node):
        out=set()
        for x in ast.walk(node):
            if isinstance(x,ast.Call) and isinstance(x.func,ast.Name):
                out.add(x.func.id)
        return out

    reachable=set(roots); frontier=list(roots)
    while frontier:
        name=frontier.pop()
        for called in direct_local_calls(funcs[name]):
            if called in funcs and called not in reachable:
                reachable.add(called); frontier.append(called)
    expected=set(BLOCK94B_REACHABLE_EVALUATOR_FUNCTIONS)
    req(reachable == expected, f'Block94B reachable evaluator closure drift: actual={sorted(reachable)} expected={sorted(expected)}')

    loaded_names=set(); called_names=set()
    for name in reachable:
        node=funcs[name]
        for x in ast.walk(node):
            if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load): loaded_names.add(x.id)
            if isinstance(x,ast.Call):
                if isinstance(x.func,ast.Name): called_names.add(x.func.id)
                elif isinstance(x.func,ast.Attribute): called_names.add(x.func.attr)
    req('minimize' not in loaded_names and 'minimize' not in called_names, 'scipy optimizer became evaluator-reachable')
    req('scipy' not in loaded_names, 'scipy became evaluator-reachable')

    minimize_import_present=False
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module == 'scipy.optimize':
            if any(a.name == 'minimize' for a in node.names): minimize_import_present=True
    req(minimize_import_present, 'expected Block94B top-level minimize import absent')

    math_import=None
    for node in tree.body:
        if isinstance(node,ast.Import) and any(a.name == 'math' for a in node.names):
            math_import=node; break
    req(math_import is not None, 'Block94B math import absent')

    selected_body=[math_import, classes['FailClosed']]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in expected:
            selected_body.append(node)
    selected_names={n.name for n in selected_body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    req(selected_names == expected, 'Block94B extracted function set drift')
    mod_ast=ast.Module(body=selected_body,type_ignores=[]); ast.fix_missing_locations(mod_ast)
    ns={'__builtins__':__builtins__}
    exec(compile(mod_ast, filename=str(path)+'::<reachable-evaluator-only>', mode='exec'), ns, ns)

    class FrozenBlock94BEvaluator: pass
    ev=FrozenBlock94BEvaluator()
    for name in expected:
        req(name in ns and callable(ns[name]), f'extracted evaluator symbol unavailable: {name}')
        setattr(ev,name,ns[name])
    ev._source_sha256=sha256_path(path)
    ev._reachable_function_names=tuple(sorted(expected))
    ev._scipy_import_executed=False
    return ev

def install_reachable_block94b(orig):
    req(BLOCK94B_REPAIR_SOURCE.is_file() and sha256_path(BLOCK94B_REPAIR_SOURCE) == BLOCK94B_REPAIR_SOURCE_SHA,
        'audited Block94B reachable-evaluator repair source drift')
    block_path=Path(orig.BLOCK94B_CARRIER).resolve()
    evaluator=load_block94b_reachable_evaluator(block_path)
    original_loader=orig.load_module_from_path
    def loader(name, candidate):
        if Path(candidate).resolve() == block_path:
            return evaluator
        return original_loader(name, candidate)
    orig.load_module_from_path=loader
    return evaluator

def bootstrap_primary():
    req(ORIGINAL_EVALUATOR.is_file() and sha256_path(ORIGINAL_EVALUATOR) == ORIGINAL_EVALUATOR_SHA, 'original primary evaluator drift')
    req(R4_SOURCE.is_file() and sha256_path(R4_SOURCE) == R4_SOURCE_SHA, 'R4 source drift')
    orig = load_module(ORIGINAL_EVALUATOR, '_stage913_final_orig_eval')
    req(sha256_path(Path(orig.BLOCK94B_CARRIER)) == BLOCK94B_SHA, 'Block94B evaluator drift')
    eval94 = install_reachable_block94b(orig)
    req(getattr(eval94, '_scipy_import_executed', None) is False, 'Block94B scipy-free extractor attestation missing')
    req(tuple(getattr(eval94, '_reachable_function_names', ())) == tuple(sorted(BLOCK94B_REACHABLE_EVALUATOR_FUNCTIONS)),
        'Block94B reachable-function attestation drift')
    g93 = orig.load_module_from_path('_stage913_final_g93', orig.BLOCK93G_CARRIER)
    m95 = orig.load_module_from_path('_stage913_final_m95', orig.BLOCK95_CARRIER)
    eval94_again = orig.load_module_from_path('_stage913_final_eval94', orig.BLOCK94B_CARRIER)
    req(eval94_again is eval94, 'Block94B loader did not return frozen extracted evaluator')
    geom = orig.load_module_from_path('_stage913_final_geom', orig.CRITICALITY_GEOMETRY)
    s5 = g93.load_stage5()
    joint_stage5 = m95.load_stage5_authority(ROOT)
    parameters = s5.frozen_stage5_parameters()
    codebooks = s5.build_codebooks(parameters)
    req(tuple(sorted(int(x) for x in codebooks)) == tuple(orig.CODEBOOK_SIZES), 'primary codebook family drift')
    validation_rows = s5.read_validation_manifest(orig.VALIDATION_MANIFEST)
    validation_by_id = {str(r.scenario_id): r for r in validation_rows}
    req(len(validation_by_id) == 44097, 'validation manifest population drift')
    import iscai_stage4.data.supervision as sup
    import iscai_stage4.data.real_pipeline as real_pipeline
    clean_cfg, degraded_cfg = real_pipeline.load_frozen_stage2_configs()
    stack = {
        's5': s5, 'g93': g93, 'eval94': eval94, 'geom': geom, 'sup': sup,
        'joint_stage5': joint_stage5, 'parameters': parameters, 'codebooks': codebooks,
        'real_pipeline': real_pipeline, 'clean_cfg': clean_cfg, 'degraded_cfg': degraded_cfg,
        'validation_by_id': validation_by_id,
    }
    return orig, stack

def bootstrap_c3():
    req(P0_HELPER.is_file() and sha256_path(P0_HELPER) == P0_HELPER_SHA, 'P0 helper drift')
    req(R31_SOURCE.is_file() and sha256_path(R31_SOURCE) == R31_SOURCE_SHA, 'R3.1 source drift')
    r31 = load_module(R31_SOURCE, '_stage913_final_r31')
    p0 = r31.import_file(P0_HELPER, '_stage913_final_p0')
    deps = r31.bootstrap_stage9_runtime(p0)
    from iscai_stage6.adb import metric_semantics as metrics_mod
    metric_path = Path(metrics_mod.__file__).resolve()
    expected = p0.EXPECTED.get(str(ROOT / 'iscai_stage6/src/iscai_stage6/adb/metric_semantics.py'))
    req(expected is not None and sha256_path(metric_path) == expected, 'Stage6 metric semantics drift')
    p0.build_h6_regions.orientation_source = deps['recv_mod'].resolve_future_orientation_source()
    return r31, p0, deps, metrics_mod

def current_ids_and_matches_for_h6(*, scenario, built, pred_ids: set[str], deps: dict) -> list[dict]:
    rt = deps['rt']; causal_mod = deps['causal_mod']
    actor_boxes = rt['build_causal_adb_actor_boxes'](scenario=scenario, adapted=built['adapted'])
    association = causal_mod.association_for_scene(
        built['scene_inputs'], actor_boxes, rt['PredictorAnchor'], rt['reciprocal_unique_nearest_anchor_association']
    )
    eligible = set(int(x) for x in causal_mod.eligible_box_indices(actor_boxes, rt['box_corners_headlamp']))
    out = []
    for m in association.matches:
        bix = int(m.box_index)
        if bix not in eligible:
            continue
        actor = actor_boxes[bix]
        cls = str(actor.object_type)
        if cls not in causal_mod.CORE_CLASSES:
            continue
        pid = str(m.prediction_id)
        # Exact R3/R3.1 predictive branch filters current reciprocal matches to
        # actors having a sealed Phase2A carrier before gaussian_v1_components.
        if pid not in pred_ids:
            continue
        out.append({'prediction_id': pid, 'actor_box_index': bix, 'actor_class': cls})
    return out



def validate_h6_role_sets(*, scenario, built, pred_ids: set[str], targets: list[dict],
                          matched: list[dict], deps: dict) -> dict:
    """Validate C3 target authority and H6 matched-eligible authority separately.

    C3 Phase3 target authority:
      CURRENT_RECIPROCAL_ASSOCIATION_TYPE_VEHICLE

    H6 historical metric authority:
      reciprocal association ∩ Stage6 eligible ∩ CORE_CLASSES ∩ sealed Phase2A carrier

    Therefore a C3 target is not required to appear in the H6 matched-eligible set.
    If absent, this validator proves the absence is exactly explained by the frozen
    Stage6 eligibility filter (not by identity/class/box drift).
    """
    rt = deps['rt']; causal_mod = deps['causal_mod']
    actor_boxes = rt['build_causal_adb_actor_boxes'](
        scenario=scenario, adapted=built['adapted']
    )
    association = causal_mod.association_for_scene(
        built['scene_inputs'], actor_boxes,
        rt['PredictorAnchor'], rt['reciprocal_unique_nearest_anchor_association']
    )
    eligible = set(
        int(x) for x in causal_mod.eligible_box_indices(
            actor_boxes, rt['box_corners_headlamp']
        )
    )

    assoc_by_pid = {}
    for m in association.matches:
        pid = str(m.prediction_id); bix = int(m.box_index)
        req(pid not in assoc_by_pid, f'duplicate reciprocal association in H6 role validator {pid}')
        req(0 <= bix < len(actor_boxes), f'H6 role-validator box index outside actor boxes {pid}/{bix}')
        a = actor_boxes[bix]
        assoc_by_pid[pid] = {
            'actor_box_index': bix,
            'actor_class': str(a.object_type),
            'eligible': bool(bix in eligible),
        }

    mb = {str(x['prediction_id']): x for x in matched}
    overlap = 0
    absent = 0
    absent_due_ineligible = 0
    for t in targets:
        pid = str(t['prediction_id'])
        req(pid in assoc_by_pid, f'C3 target absent from reciprocal association {pid}')
        a = assoc_by_pid[pid]
        req(str(a['actor_class']) == 'TYPE_VEHICLE', f'C3 target reciprocal class drift {pid}')
        req(int(a['actor_box_index']) == int(t['actor_box_index']),
            f'C3 target reciprocal box-index drift {pid}')
        req(pid in pred_ids, f'C3 target absent from sealed Phase2A carrier set {pid}')

        expected_in_h6 = bool(
            a['eligible'] and
            str(a['actor_class']) in causal_mod.CORE_CLASSES and
            pid in pred_ids
        )
        actual_in_h6 = pid in mb
        req(actual_in_h6 == expected_in_h6,
            f'H6 matched-eligible membership drift {pid}: actual={actual_in_h6} expected={expected_in_h6}')

        if actual_in_h6:
            overlap += 1
            req(str(mb[pid]['actor_class']) == 'TYPE_VEHICLE', f'C3/H6 overlap class drift {pid}')
            req(int(mb[pid]['actor_box_index']) == int(t['actor_box_index']),
                f'C3/H6 overlap box-index drift {pid}')
        else:
            absent += 1
            req(not a['eligible'],
                f'C3 target absent from H6 matched-eligible for unexplained reason {pid}')
            absent_due_ineligible += 1

    return {
        'C3_target_count': int(len(targets)),
        'H6_matched_eligible_count': int(len(matched)),
        'C3_H6_overlap_count': int(overlap),
        'C3_absent_from_H6_matched_eligible_count': int(absent),
        'C3_absent_due_frozen_stage6_ineligible_count': int(absent_due_ineligible),
        'role_sets_forced_equal': False,
        'H6_metric_universe_changed': False,
    }


def h1_truth_by_horizon_rigid(stack: dict, scenario, built, current_index: int,
                              truth_index: int, scenario_id: str, orig) -> dict:
    """Preserve H1 truth semantics while using the current RigidTransform carrier."""
    s5 = stack['s5']
    sup = stack['sup']
    geom = stack['geom']
    T = built['adapted'].frames.T_H0_from_W
    truth_track = scenario.tracks[int(truth_index)]
    mean_offset = np.asarray(stack['parameters']['offset']['mean_body_m'], dtype=np.float64)
    covariance_offset = np.asarray(stack['parameters']['offset']['covariance_body_m2'], dtype=np.float64)
    truth_rng = np.random.default_rng(
        s5.stable_seed(str(scenario_id), 'formal_receiver_offset_truth')
    )
    uncertain_realization = truth_rng.multivariate_normal(
        mean_offset, covariance_offset, check_valid='raise'
    )
    out = {}
    for horizon_index, (horizon_s, frame_offset) in enumerate(
        zip(orig.H1_HORIZONS_S, s5.HORIZON_OFFSETS)
    ):
        state_index = int(current_index) + int(frame_offset)
        if state_index >= len(truth_track.states) or not bool(truth_track.states[state_index].valid):
            continue
        state = truth_track.states[state_index]
        center = sup._point_H0(state, T_H0_from_W=T)
        heading = geom.transform_heading_with_rigid_transform(float(state.heading), T)
        position = s5.receiver_truth_position(
            center,
            heading,
            mode=orig.H1_GEOMETRY_MODE,
            known_offset=mean_offset,
            uncertain_realization=uncertain_realization,
        )
        radius, azimuth, elevation = s5.spherical_angles(position)
        out[int(horizon_index)] = {
            'horizon_s': float(horizon_s),
            'range_m': float(radius),
            'azimuth_rad': float(azimuth),
            'elevation_rad': float(elevation),
        }
    return out

def evaluate_primary(pretruth: dict, scenario, built, orig, stack: dict) -> dict:
    pretruth_hash = str(pretruth['pretruth_decision_sha256'])
    rec = pretruth['receiver']
    if rec['selected'] is not True:
        return {
            'status': 'NOT_EVALUATED_NO_ELIGIBLE_RECEIVER', 'pretruth_decision_sha256': pretruth_hash,
            'H1_records': [], 'communication_records': [], 'criticality_label': None,
            'association': None, 'actor_center_truth_1s': None,
        }
    if rec.get('selected_receiver_prediction_available') is not True:
        return {
            'status': 'NOT_EVALUATED_SELECTED_RECEIVER_PREDICTION_UNAVAILABLE', 'pretruth_decision_sha256': pretruth_hash,
            'H1_records': [], 'communication_records': [], 'criticality_label': None,
            'association': None, 'actor_center_truth_1s': None,
        }
    s5 = stack['s5']; sup = stack['sup']; eval94 = stack['eval94']; geom = stack['geom']
    current_index = int(pretruth['current_index'])
    T = built['adapted'].frames.T_H0_from_W
    scene_inputs = built['scene_inputs']
    selected_pid = str(rec['prediction_id'])
    current_ids = []
    for h in tuple(scene_inputs.histories):
        latest = int(h.latest_observed_frame_index)
        if latest != current_index: continue
        steps = tuple(h.steps)
        if 0 <= latest < len(steps) and bool(steps[latest].observed):
            current_ids.append(str(h.prediction_id))
    current_ids.sort()
    ph = eval94.current_prediction_histories(scene_inputs, current_ids, current_index)
    matches, truth_candidate_count, edge_count = eval94.evaluator_matches(ph, scenario, T, sup)
    req(selected_pid in matches, f'primary selected prediction absent from evaluator association {pretruth["scenario_id"]}|{selected_pid}')
    match = matches[selected_pid]
    truth_index = int(match['truth_track_index'])
    frozen_ego = pretruth['C1_C2']['ego_prediction_1s']
    ego_geom, ego_reason = eval94.future_ego_geometry(scenario, current_index, T, frozen_ego, sup, geom)
    crit = eval94.actor_label_row(
        {'scenario_id': str(pretruth['scenario_id']), 'prediction_id': selected_pid, 'pcrit_raw': float(pretruth['C1_C2']['Pcrit_raw'])},
        match, scenario, current_index, T, ego_geom, ego_reason, sup, geom,
    )
    crit['Pcrit_cal'] = float(pretruth['C1_C2']['Pcrit_cal'])

    # H1: exact frozen receiver-geometry truth route, evaluated against already-frozen RAW/CAL plans.
    h1_truth = h1_truth_by_horizon_rigid(stack, scenario, built, current_index, truth_index, pretruth['scenario_id'], orig)
    h1_records = []
    spec = pretruth['C1_C2']['H1_paired_ablation']
    for arm in orig.H1_ARMS:
        for plan in spec['arms'][arm]:
            truth = h1_truth.get(int(plan['horizon_index']))
            if truth is None: continue
            B = int(plan['codebook_size'])
            hit = s5.selected_set_hit(float(truth['azimuth_rad']), tuple(int(x) for x in plan['adaptive_indices']), stack['codebooks'][B])
            h1_records.append({
                'scenario_id': str(pretruth['scenario_id']), 'prediction_id': selected_pid, 'arm': str(arm),
                'geometry_mode': orig.H1_GEOMETRY_MODE, 'codebook_size': B, 'horizon_s': float(plan['horizon_s']),
                'requested_q': float(orig.H1_Q), 'containment_hit': bool(hit), 'selected_k': int(plan['selected_k']),
                'physical_probe_count': int(plan['physical_probe_count']), 'posterior_mass_selected': float(plan['posterior_mass_selected']),
                'azimuth_std_rad': float(plan['azimuth_std_rad']), 'mc_seed': int(spec['seed']),
                'receiver_distance_m': float(truth['range_m']),
            })

    idx_1s = current_index + 10
    truth_track = scenario.tracks[truth_index]
    actor_future_valid = idx_1s < len(truth_track.states) and bool(truth_track.states[idx_1s].valid)
    communication_records = []
    actor_center_truth = None
    if actor_future_valid:
        center = tuple(float(x) for x in sup._point_H0(truth_track.states[idx_1s], T_H0_from_W=T))
        radius, truth_azimuth, truth_elevation = s5.spherical_angles(center)
        actor_center_truth = {'truth_track_index': truth_index, 'center_H0_m': list(center), 'range_m': float(radius), 'azimuth_rad': float(truth_azimuth), 'elevation_rad': float(truth_elevation)}
        y_valid = bool(crit['label_valid']); ycrit = int(crit['Y_crit_1s']) if y_valid else None
        Tbeam_s = float(stack['parameters']['Tbeam_s']); Tframe_s = float(stack['parameters']['Tframe_s'])
        for B in orig.CODEBOOK_SIZES:
            B = int(B); codebook = stack['codebooks'][B]; cb = pretruth['C1_C2']['codebooks'][str(B)]
            decision_azimuth = orig._clip_truth_azimuth_for_complete_partition(stack, truth_azimuth, codebook)
            oracle_selection = s5.oracle_best_gain_beam(realized_azimuth_rad=float(truth_azimuth), codebook=codebook)
            oracle_indices = tuple(int(x) for x in s5.selection_indices(oracle_selection))
            oracle_link = s5.best_optical_link(indices=oracle_indices, azimuth=float(truth_azimuth), elevation=float(truth_elevation), codebook=codebook, probing_beam_count=1, Tbeam_s=Tbeam_s, Tframe_s=Tframe_s)
            p_c2 = cb['proposed_C2']
            solver_runtime = p_c2.get('solver_runtime_s')
            if solver_runtime is not None:
                solver_runtime = float(solver_runtime)
            decisions = {
                'fixed_top_1': (tuple(cb['fixed_top_k']['1']), tuple(cb['fixed_top_k']['1']), len(cb['fixed_top_k']['1']), len(cb['fixed_top_k']['1']), 'stage9_complete_partition', False, None),
                'fixed_top_3': (tuple(cb['fixed_top_k']['3']), tuple(cb['fixed_top_k']['3']), len(cb['fixed_top_k']['3']), len(cb['fixed_top_k']['3']), 'stage9_complete_partition', False, None),
                'fixed_top_5': (tuple(cb['fixed_top_k']['5']), tuple(cb['fixed_top_k']['5']), len(cb['fixed_top_k']['5']), len(cb['fixed_top_k']['5']), 'stage9_complete_partition', False, None),
                'fixed_q_0.95': (tuple(cb['fixed_q']['0.95']), tuple(cb['fixed_q']['0.95']), len(cb['fixed_q']['0.95']), len(cb['fixed_q']['0.95']), 'stage9_complete_partition', False, None),
                'fixed_q_0.99': (tuple(cb['fixed_q']['0.99']), tuple(cb['fixed_q']['0.99']), len(cb['fixed_q']['0.99']), len(cb['fixed_q']['0.99']), 'stage9_complete_partition', False, None),
                'exhaustive': (tuple(cb['exhaustive']), tuple(cb['exhaustive']), B, B, 'stage9_complete_partition', False, None),
                'heuristic_q': (tuple(cb['heuristic_q_selected_beam_indices']), tuple(cb['heuristic_q_selected_beam_indices']), len(cb['heuristic_q_selected_beam_indices']), len(cb['heuristic_q_selected_beam_indices']), 'stage9_complete_partition', False, None),
                'uncertainty_only_adaptive_topk': (tuple(cb['uncertainty_only_adaptive_topk']['selected_beam_indices']), tuple(cb['uncertainty_only_adaptive_topk']['physical_probe_indices']), int(cb['uncertainty_only_adaptive_topk']['K']), int(cb['uncertainty_only_adaptive_topk']['physical_probe_count']), 'frozen_stage5_adaptive_in_support', False, None),
                'proposed_C2': (tuple(p_c2['selected_beam_indices']), tuple(p_c2['selected_beam_indices']), int(p_c2['K']), int(p_c2['K']), 'stage9_complete_partition', bool(p_c2.get('fallback_used', False)), solver_runtime),
            }
            for policy, (coverage_indices, physical_indices, K, probe_count, coverage_semantics, fallback, runtime) in decisions.items():
                coverage_azimuth = float(truth_azimuth) if coverage_semantics == 'frozen_stage5_adaptive_in_support' else float(decision_azimuth)
                hit = s5.selected_set_hit(coverage_azimuth, coverage_indices, codebook)
                achieved = s5.best_optical_link(indices=physical_indices, azimuth=float(truth_azimuth), elevation=float(truth_elevation), codebook=codebook, probing_beam_count=int(probe_count), Tbeam_s=Tbeam_s, Tframe_s=Tframe_s)
                metrics = s5.link_metrics(achieved, oracle_link)
                communication_records.append({
                    'scenario_id': str(pretruth['scenario_id']), 'prediction_id': selected_pid, 'policy': policy, 'codebook_size': B,
                    'selected_beam_indices': [int(x) for x in coverage_indices], 'physical_probe_indices': [int(x) for x in physical_indices],
                    'K': int(K), 'physical_probe_count': int(probe_count), 'probing_overhead_fraction': float(probe_count * Tbeam_s / Tframe_s),
                    'coverage_hit': bool(hit), 'beam_outage': bool(not hit), 'critical_label_valid': y_valid, 'Y_crit_1s': ycrit,
                    'joint_critical_miss': bool(y_valid and ycrit == 1 and not hit), 'coverage_semantics': coverage_semantics,
                    'fallback_used': fallback, 'solver_runtime_s': runtime, **metrics,
                })
            om = s5.link_metrics(oracle_link, oracle_link)
            communication_records.append({
                'scenario_id': str(pretruth['scenario_id']), 'prediction_id': selected_pid, 'policy': 'oracle_eval_only', 'codebook_size': B,
                'selected_beam_indices': [int(x) for x in oracle_indices], 'physical_probe_indices': [int(x) for x in oracle_indices],
                'K': 1, 'physical_probe_count': 1, 'probing_overhead_fraction': float(Tbeam_s / Tframe_s), 'coverage_hit': True,
                'beam_outage': False, 'critical_label_valid': y_valid, 'Y_crit_1s': ycrit, 'joint_critical_miss': False,
                'coverage_semantics': 'evaluator_only_oracle', 'fallback_used': False, 'solver_runtime_s': None, **om,
            })
    return {
        'status': 'EVALUATED_FUTURE_GT' if actor_future_valid else 'EVALUATED_FUTURE_GT_ACTOR_1S_INVALID',
        'pretruth_decision_sha256': pretruth_hash,
        'association': {
            'selected_prediction_id': selected_pid, 'truth_track_index': truth_index,
            'historical_match_distance_m': float(match['historical_match_distance_m']),
            'historical_match_frame_index': int(match['historical_match_frame_index']),
            'truth_candidate_count': int(truth_candidate_count), 'association_edge_count': int(edge_count),
        },
        'criticality_label': native(crit), 'actor_center_truth_1s': actor_center_truth,
        'actor_future_valid_1s': bool(actor_future_valid), 'H1_records': h1_records, 'communication_records': communication_records,
    }

def h5_fast_pair_stats(Gb: np.ndarray, Gr: np.ndarray, first: np.ndarray, A: float, W: float, time_grid: np.ndarray) -> dict[str, int]:
    b = np.asarray(Gb, dtype=np.float64); r = np.asarray(Gr, dtype=np.float64); f = np.asarray(first, dtype=np.int64)
    req(b.shape == r.shape and b.ndim == 2 and b.shape[1] == 11, 'H5 event arrays invalid')
    req(f.shape == (b.shape[0],), 'H5 first-index shape mismatch')
    req(np.all((f >= 0) & (f <= 9)), 'H5 first-index outside 0..9')
    total_n = total_b = total_r = 0
    for fv in range(10):
        rows = np.flatnonzero(f == fv)
        if rows.size == 0: continue
        col = fv + 1
        t = float(time_grid[col])
        use = np.flatnonzero((time_grid <= t + 1e-12) & ((t - time_grid) <= float(W) + 1e-12))
        bb = b[rows][:, use]; rr = r[rows][:, use]
        bfin = np.isfinite(bb); rfin = np.isfinite(rr)
        bok = np.any(bfin, axis=1); rok = np.any(rfin, axis=1)
        bv = np.full(rows.size, np.nan, dtype=np.float64); rv = np.full(rows.size, np.nan, dtype=np.float64)
        if np.any(bok): bv[bok] = float(A) * np.max(np.where(bfin[bok], bb[bok], -np.inf), axis=1)
        if np.any(rok): rv[rok] = float(A) * np.max(np.where(rfin[rok], rr[rok], -np.inf), axis=1)
        common = np.isfinite(bv) & np.isfinite(rv)
        T = 0.1 * (fv + 1.0)
        total_n += int(np.count_nonzero(common))
        total_b += int(np.count_nonzero(common & (bv >= T)))
        total_r += int(np.count_nonzero(common & (rv >= T)))
    return {'n': total_n, 'baseline_violations': total_b, 'recovery_violations': total_r}

def h5_scene_counts(meta: dict, z, p0, *, parity_state: dict) -> dict:
    out = {prof: {base: {'n': 0, 'baseline_violations': 0, 'recovery_violations': 0} for base in p0.BASELINES} for prof in p0.PROFILES}
    tg = np.asarray(p0.TIME_GRID, dtype=np.float64)
    for t in meta.get('targets', []):
        pfx = str(t['array_key_prefix'])
        first = np.asarray(z[pfx + '__first_critical_index_u8'], dtype=np.uint8)
        Grec = np.asarray(z[pfx + '__Gcrit__RHO2_recovery_ADB'], dtype=np.float64)
        Greact = np.asarray(z[pfx + '__Gcrit__original_reactive_ADB'], dtype=np.float64)
        Gv3 = np.asarray(z[pfx + '__Gcrit__Stage6_V3_predictive_ADB'], dtype=np.float64)
        req(first.shape == (int(t['critical_count']),), f'H5 critical-count drift {meta["scenario_id"]}/{pfx}')
        req(Grec.shape == Greact.shape == Gv3.shape == (first.size, 11), f'H5 Gcrit shape drift {meta["scenario_id"]}/{pfx}')
        for prof, (A, W) in p0.PROFILES.items():
            for base in p0.BASELINES:
                Gb = Greact if base == 'original_reactive_ADB' else Gv3 if base == 'Stage6_V3_predictive_ADB' else None
                req(Gb is not None, f'unknown H5 baseline {base}')
                st = h5_fast_pair_stats(Gb, Grec, first, float(A), float(W), tg)
                a = out[prof][base]
                for k in a: a[k] += int(st[k])
                if not parity_state.get('done') and first.size:
                    Db = p0.max_hold_array(Gb, float(A), float(W)); Dr = p0.max_hold_array(Grec, float(A), float(W))
                    exact = p0.event_pair_stats(Db, Dr, first)
                    req(exact == st, f'H5 fast/canonical parity fail {meta["scenario_id"]}/{pfx}/{prof}/{base}: {exact} vs {st}')
        if not parity_state.get('done') and first.size:
            parity_state['done'] = True
            print(f'H5_EXACT_EVENT_PARITY PASS sid={meta["scenario_id"]} prefix={pfx}', flush=True)
    return out

def worker(out: Path, shard_count: int, shard_index: int) -> None:
    req(0 <= shard_index < shard_count, 'invalid shard index')
    pre = out / 'FINAL_EVALUATOR_PREFLIGHT_SEAL.json'
    req(pre.is_file(), 'global preflight seal missing')
    pobj = load_json(pre)
    req(pobj.get('status') == 'FROZEN_COMPLETE_FINAL_EVALUATOR_PRE_GT_PREFLIGHT' and pobj.get('future_GT_access') is False, 'preflight seal invalid')
    verify_combined_gate(); ids = formal_ids()
    print('=' * 118)
    print(f'STAGE 9.13 FINAL EVALUATOR R5-RECOVERY2 — WORKER {shard_index}/{shard_count}')
    print('ONE FUTURE-GT READ PER ASSIGNED FORMAL SCENE / EVALUATOR ONLY / H1-H6')
    print('=' * 118, flush=True)
    orig, pst = bootstrap_primary()
    r31, p0, deps, metrics_mod = bootstrap_c3()
    req('scipy' not in sys.modules, 'SciPy unexpectedly imported into final evaluator')
    parity_state = {'done': False}
    amendment_path = recovery2_amendment_path(out)
    req(amendment_path.is_file(), 'technical recovery2 amendment missing')
    amendment = load_json(amendment_path)
    req(amendment.get('status') == 'FROZEN_TECHNICAL_RECOVERY2_AUTHORIZATION_BEFORE_REREAD',
        'technical recovery2 amendment status drift')
    req(str(amendment.get('previous_recovery_amendment_sha256')) == RECOVERY1_AMENDMENT_SHA,
        'technical recovery2 previous-amendment binding drift')
    req(str(amendment.get('recovery2_runner_sha256')) == sha256_path(Path(__file__).resolve()),
        'technical recovery2 runner binding drift')
    recovery_auth = {
        (int(x['formal_ordinal']), str(x['scenario_id'])): dict(x)
        for x in amendment['authorized_recovery2_scenes']
    }
    req(len(recovery_auth) == 12, 'technical recovery2 authorization cardinality drift')
    processed = reused = 0; t0 = time.time()

    with R4_MANIFEST.open('r', encoding='utf-8') as fr, P2A_MANIFEST.open('r', encoding='utf-8') as fa:
        ri = (json.loads(x) for x in fr if x.strip()); ai = (json.loads(x) for x in fa if x.strip())
        for ordinal, sid in enumerate(ids):
            try: r4 = next(ri); pa = next(ai)
            except StopIteration: raise FailClosed(f'R4/Phase2A ended before {ordinal}')
            if ordinal % shard_count != shard_index:
                continue
            verify_r4_row(r4, ordinal, sid)
            req(int(pa.get('formal_ordinal', -1)) == ordinal and str(pa.get('scenario_id')) == sid, f'Phase2A identity drift {sid}')
            op = outcome_path(out, ordinal, sid); gp = opened_path(out, ordinal, sid)
            if op.exists():
                j = load_json(op)
                req(j.get('status') == 'COMPLETE_FINAL_EVALUATOR_H1_H6' and int(j.get('formal_ordinal', -1)) == ordinal and str(j.get('scenario_id')) == sid, f'existing outcome invalid {sid}')
                req(gp.is_file(), f'existing outcome lacks GT-open marker {sid}')
                reused += 1; processed += 1
                continue

            recovery_reread = False
            prior_intent_sha = None
            physical_read_number = 1
            key = (int(ordinal), str(sid))
            r1p = recovery_reopen_path(out, ordinal, sid)
            r2p = recovery2_reopen_path(out, ordinal, sid)
            if gp.exists():
                req(key in recovery_auth,
                    f'GT-open marker exists without completed outcome for {sid}; not recovery2-authorized')
                auth = recovery_auth[key]
                prior_intent_sha = sha256_path(gp)
                req(prior_intent_sha == str(auth['prior_gt_open_intent_sha256']),
                    f'recovery2 GT-intent SHA drift {sid}')
                prior_reads = int(auth['prior_physical_future_GT_read_count'])
                req(prior_reads == 1 + int(r1p.exists()),
                    f'recovery2 prior-read marker/count drift {sid}')
                physical_read_number = int(auth['authorized_next_physical_future_GT_read_number'])
                req(physical_read_number == prior_reads + 1,
                    f'recovery2 next-read number drift {sid}')
                req(not r2p.exists(),
                    f'recovery2 marker exists without completed outcome for {sid}; further reread forbidden')
                recovery_reread = True
            else:
                req(key not in recovery_auth,
                    f'recovery2-authorized original GT intent unexpectedly missing {sid}')
                req(not r1p.exists() and not r2p.exists(),
                    f'recovery markers exist without original GT intent {sid}')

            jp, npz = action_paths(ordinal, sid)
            req(jp.is_file() and npz.is_file(), f'C3 action pair missing {sid}')
            meta = load_json(jp); verify_action_meta(meta, ordinal, sid, npz, hash_npz=True)
            p2ap = resolve_p2a_scene(pa); p2a_scene = load_json(p2ap)
            pred_by_id = r31.normalize_predictions(p2a_scene)
            req(len(pred_by_id) == int(pa.get('actor_prediction_count', -1)), f'Phase2A carrier count drift {sid}')

            with np.load(npz, allow_pickle=False) as z:
                current = np.asarray(z['current_reactive_illumination'], dtype=np.float64)
                v3 = np.asarray(z['stage6_v3_schedule'], dtype=np.float64)
                recovery = np.asarray(z['recovery_schedule'], dtype=np.float64)
                req(current.shape == (501, 301) and v3.shape == recovery.shape == (4, 501, 301), f'C3 action schedule shape drift {sid}')

            if not recovery_reread:
                marker = {
                    'schema': 'stage9_block913_v1_3_final_evaluator_gt_open_intent_v1',
                    'formal_ordinal': ordinal, 'scenario_id': sid,
                    'pretruth_decision_sha256': r4['pretruth_decision_sha256'],
                    'combined_preoutcome_seal_sha256': COMBINED_SHA,
                    'future_GT_controller_access': False,
                    'reason': 'evaluator-only H1-H6 pass after successful combined pre-outcome seal',
                }
                atomic_json(gp, marker)
                physical_read_number = 1
            else:
                reread_marker = {
                    'schema': 'stage9_block913_v1_3_final_evaluator_gt_reopen_recovery_v2',
                    'formal_ordinal': ordinal,
                    'scenario_id': sid,
                    'prior_gt_open_intent_sha256': prior_intent_sha,
                    'prior_recovery1_marker_sha256': sha256_path(r1p) if r1p.exists() else None,
                    'technical_recovery2_amendment_sha256': sha256_path(amendment_path),
                    'recovery2_runner_sha256': sha256_path(Path(__file__).resolve()),
                    'physical_future_GT_read_number': int(physical_read_number),
                    'future_GT_controller_access': False,
                    'policy_retuned': False,
                    'thresholds_retuned': False,
                    'baselines_changed': False,
                    'fourth_read_authorized': False,
                }
                atomic_json(r2p, reread_marker)

            scenario, read_route = pst['g93'].read_cal_scenario(
                pst['s5'], pst['validation_by_id'][sid]
            )
            req(int(scenario.current_time_index) == int(r4['current_index']), f'current index drift {sid}')
            req(str(read_route) == str(r4['read_route']), f'read route drift {sid}')
            built = pst['real_pipeline'].build_real_causal_inputs(scenario, clean_config=pst['clean_cfg'], degraded_config=pst['degraded_cfg'])

            # H5 is materialized only after the evaluator-only future-GT boundary
            # has opened for this scene. It consumes frozen causal Gcrit arrays only;
            # no future-GT value is used in H5.
            with np.load(npz, allow_pickle=False) as z:
                h5 = h5_scene_counts(meta, z, p0, parity_state=parity_state)

            primary = evaluate_primary(r4, scenario, built, orig, pst)
            matched = current_ids_and_matches_for_h6(
                scenario=scenario, built=built, pred_ids=set(pred_by_id), deps=deps
            )
            h6_role_diag = validate_h6_role_sets(
                scenario=scenario,
                built=built,
                pred_ids=set(pred_by_id),
                targets=list(meta.get('targets', [])),
                matched=matched,
                deps=deps,
            )
            ev = p0.build_h6_regions(scenario, matched, deps['recv_mod'])
            acts = {
                'reactive': np.broadcast_to(current[None, :, :], (4, 501, 301)),
                'Stage6_V3': v3,
                'recovery': recovery,
            }
            h6 = {name: p0.h6_metrics(schedule, current, ev, metrics_mod) for name, schedule in acts.items()}
            class_counts = defaultdict(int)
            for m in matched: class_counts[str(m['actor_class'])] += 1
            result = {
                'schema': 'stage9_block913_v1_3_final_evaluator_H1_H6_scene_v1',
                'status': 'COMPLETE_FINAL_EVALUATOR_H1_H6',
                'formal_ordinal': ordinal, 'scenario_id': sid,
                'combined_preoutcome_seal_sha256': COMBINED_SHA,
                'R4_pretruth_decision_sha256': r4['pretruth_decision_sha256'],
                'R4_manifest_sha256': R4_MANIFEST_SHA,
                'C3_action_json_sha256': sha256_path(jp),
                'C3_action_npz_sha256': meta['npz_sha256'],
                'Phase2A_scene_json_sha256': pa['scenario_cache_sha256'],
                'primary': primary,
                'H5_raw_counts': h5,
                'H6_metrics': native(h6),
                'H6_matched_eligible_count': len(matched),
                'H6_matched_eligible_class_counts': dict(sorted(class_counts.items())),
                'H6_role_binding_diagnostics': native(h6_role_diag),
                'future_GT_access': True,
                'future_GT_evaluator_access': True,
                'future_GT_controller_access': False,
                'technical_recovery_reread': bool(recovery_reread),
                'technical_recovery_generation': 2 if recovery_reread else 0,
                'future_GT_physical_read_number': int(physical_read_number),
                'prior_gt_open_intent_sha256': prior_intent_sha if recovery_reread else None,
                'technical_recovery_amendment_sha256': sha256_path(amendment_path) if recovery_reread else None,
                'model_forward_executed': False,
                'C1_MC_recomputed': False,
                'C2_solver_recomputed': False,
                'Phase2B_endpoint_MC_replayed': False,
                'policy_retuned': False,
                'thresholds_retuned': False,
                'baseline_changed': False,
                'hypothesis_decision_made_here': False,
            }
            atomic_json(op, result)
            processed += 1
            if processed % 25 == 0:
                print(f'[W{shard_index:02d}] done={processed} reuse={reused} ordinal={ordinal:05d} elapsed_s={time.time()-t0:.1f}', flush=True)
    expected = sum(1 for i in range(FORMAL_N) if i % shard_count == shard_index)
    req(processed == expected, f'worker processed count drift {processed}!={expected}')
    print(f'WORKER_DONE shard={shard_index}/{shard_count} scenes={processed} reuse={reused} elapsed_s={time.time()-t0:.1f}', flush=True)

def _new_policy_acc():
    return {
        'trials': 0, 'hits': 0, 'critical_label_valid': 0, 'critical_events': 0, 'critical_hits': 0,
        'joint_critical_misses': 0, 'sum_K': 0.0, 'sum_probe': 0.0, 'sum_overhead': 0.0,
        'fallback_count': 0, 'solver_runtime_sum_s': 0.0, 'solver_runtime_n': 0,
        'metric_sums': defaultdict(float), 'metric_counts': defaultdict(int),
    }

def _update_policy(a: dict, r: dict) -> None:
    a['trials'] += 1; a['hits'] += int(bool(r['coverage_hit'])); a['sum_K'] += float(r['K'])
    a['sum_probe'] += float(r['physical_probe_count']); a['sum_overhead'] += float(r['probing_overhead_fraction'])
    a['fallback_count'] += int(bool(r.get('fallback_used', False)))
    if r.get('solver_runtime_s') is not None:
        a['solver_runtime_sum_s'] += float(r['solver_runtime_s']); a['solver_runtime_n'] += 1
    if bool(r['critical_label_valid']):
        a['critical_label_valid'] += 1
        if int(r['Y_crit_1s']) == 1:
            a['critical_events'] += 1; a['critical_hits'] += int(bool(r['coverage_hit'])); a['joint_critical_misses'] += int(bool(r['joint_critical_miss']))
    for name in ('optical_gain','received_power_normalized','snr_db','ber','effective_rate_bps','beam_gain_loss_db','received_power_loss_db','snr_loss_db','effective_rate_loss_bps'):
        v = r.get(name)
        if v is not None and math.isfinite(float(v)):
            a['metric_sums'][name] += float(v); a['metric_counts'][name] += 1

def _finish_policy(a: dict) -> dict:
    n = a['trials']; cv = a['critical_label_valid']; ce = a['critical_events']
    out = {
        'trials': n, 'coverage_hit_count': a['hits'], 'beam_outage_count': n-a['hits'],
        'coverage_rate': (a['hits']/n if n else None), 'beam_outage_rate': ((n-a['hits'])/n if n else None),
        'critical_label_valid_count': cv, 'critical_event_count': ce, 'critical_hit_count': a['critical_hits'],
        'critical_event_coverage': (a['critical_hits']/ce if ce else None),
        'critical_conditioned_outage': (a['joint_critical_misses']/ce if ce else None),
        'joint_critical_miss_numerator': a['joint_critical_misses'], 'joint_critical_miss_denominator': cv,
        'joint_critical_miss_rate': (a['joint_critical_misses']/cv if cv else None),
        'mean_K': (a['sum_K']/n if n else None), 'mean_physical_probe_count': (a['sum_probe']/n if n else None),
        'mean_probing_overhead_fraction': (a['sum_overhead']/n if n else None), 'fallback_count': a['fallback_count'],
        'mean_solver_runtime_s': (a['solver_runtime_sum_s']/a['solver_runtime_n'] if a['solver_runtime_n'] else None),
    }
    for k, total in a['metric_sums'].items():
        c = a['metric_counts'][k]; out['mean_' + k] = total/c if c else None
    return out

def bootstrap_only(out: Path) -> None:
    pre = out / 'FINAL_EVALUATOR_PREFLIGHT_SEAL.json'
    req(pre.is_file(), 'global preflight seal missing')
    pobj = load_json(pre)
    req(pobj.get('status') == 'FROZEN_COMPLETE_FINAL_EVALUATOR_PRE_GT_PREFLIGHT', 'preflight seal invalid')
    req(pobj.get('future_GT_access') is False and pobj.get('FORMAL_outcomes_computed') is False, 'preflight GT/outcome flag drift')
    verify_combined_gate()
    print('=' * 118)
    print('STAGE 9.13 FINAL EVALUATOR R5-RECOVERY2 — BOOTSTRAP-ONLY SMOKE')
    print('NO FUTURE GT / NO OUTCOMES / NO SCENARIO READ')
    print('=' * 118, flush=True)
    _, pst = bootstrap_primary()
    bootstrap_c3()
    req('scipy' not in sys.modules, 'SciPy unexpectedly imported into final evaluator bootstrap')
    ev = pst['eval94']
    req(getattr(ev, '_scipy_import_executed', None) is False, 'Block94B scipy-free attestation missing')
    for name in BLOCK94B_REACHABLE_EVALUATOR_FUNCTIONS:
        req(callable(getattr(ev, name, None)), f'Block94B evaluator callable missing: {name}')
    print('BLOCK94B_REACHABLE_EVALUATOR = PASS')
    print('SCIPY_IMPORTED = FALSE')
    print('FINAL_EVALUATOR_BOOTSTRAP = PASS', flush=True)

def finalize(out: Path) -> None:
    pre = out / 'FINAL_EVALUATOR_PREFLIGHT_SEAL.json'
    req(pre.is_file(), 'preflight seal missing')
    verify_combined_gate(); ids = formal_ids()
    amendment1_path = recovery_amendment_path(out)
    amendment2_path = recovery2_amendment_path(out)
    req(amendment1_path.is_file() and sha256_path(amendment1_path) == RECOVERY1_AMENDMENT_SHA,
        'first technical recovery amendment drift at finalization')
    req(amendment2_path.is_file(), 'second technical recovery amendment missing at finalization')
    amendment2 = load_json(amendment2_path)
    req(str(amendment2.get('recovery2_runner_sha256')) == sha256_path(Path(__file__).resolve()),
        'technical recovery2 runner binding drift at finalization')
    req(not (out / 'stage9_block913_v1_3_FINAL_H1_H6_seal.json').exists(), 'final seal already exists')
    orig = load_module(ORIGINAL_EVALUATOR, '_stage913_finalizer_orig')
    h1_acc = {(arm, int(B), float(h)): orig._new_h1_accumulator() for arm in orig.H1_ARMS for B in orig.CODEBOOK_SIZES for h in orig.H1_HORIZONS_S}
    pol_acc: dict[tuple[int,str],dict] = {}
    h5_tot = None
    h6_sum = defaultdict(float); h6_n = defaultdict(int)
    h6_actions = ('reactive','Stage6_V3','recovery')
    metric_names = ('vehicle_shadow_zone_violation','glare_risk_exposure','over_masking_area','road_illumination_retention','pedestrian_visibility_proxy','cyclist_visibility_proxy')
    selected = no_receiver = primary_eval = 0
    unique_reread_scenes = 0
    extra_physical_reread_events = 0
    max_physical_read_number = 1
    h6_c3_absent_due_ineligible_total = 0
    manifest_tmp = out / '.stage9_block913_v1_3_FINAL_H1_H6_manifest.partial.jsonl'
    req(not manifest_tmp.exists(), 'finalizer partial manifest exists')
    with manifest_tmp.open('w', encoding='utf-8') as mf:
        for i, sid in enumerate(ids):
            op = outcome_path(out, i, sid); gp = opened_path(out, i, sid)
            req(op.is_file() and gp.is_file(), f'final outcome/GT marker missing {i}/{sid}')
            j = load_json(op)
            req(j.get('status') == 'COMPLETE_FINAL_EVALUATOR_H1_H6' and int(j.get('formal_ordinal', -1)) == i and str(j.get('scenario_id')) == sid, f'outcome identity/status drift {sid}')
            req(j.get('future_GT_evaluator_access') is True and j.get('future_GT_controller_access') is False, f'future GT role drift {sid}')
            r1p = recovery_reopen_path(out, i, sid)
            r2p = recovery2_reopen_path(out, i, sid)
            expected_reads = 1 + int(r1p.exists()) + int(r2p.exists())
            req(int(j.get('future_GT_physical_read_number', -1)) == expected_reads,
                f'physical future-GT read provenance drift {sid}: outcome={j.get("future_GT_physical_read_number")} markers={expected_reads}')
            if expected_reads > 1:
                req(j.get('technical_recovery_reread') is True,
                    f'technical recovery flag missing {sid}')
                unique_reread_scenes += 1
                extra_physical_reread_events += expected_reads - 1
                max_physical_read_number = max(max_physical_read_number, expected_reads)
            else:
                req(j.get('technical_recovery_reread') is False,
                    f'unexpected technical recovery flag {sid}')
            diag = j.get('H6_role_binding_diagnostics')
            if diag is not None:
                req(diag.get('role_sets_forced_equal') is False and diag.get('H6_metric_universe_changed') is False,
                    f'H6 role-diagnostic semantic drift {sid}')
                h6_c3_absent_due_ineligible_total += int(diag.get('C3_absent_due_frozen_stage6_ineligible_count', 0))
            p = j['primary']
            if p['status'] == 'NOT_EVALUATED_NO_ELIGIBLE_RECEIVER': no_receiver += 1
            else:
                selected += 1
                if p['status'].startswith('EVALUATED_FUTURE_GT'): primary_eval += 1
            for r in p.get('H1_records', []):
                key = (str(r['arm']), int(r['codebook_size']), float(r['horizon_s']))
                req(key in h1_acc, f'H1 stratum drift {sid}/{key}')
                orig._update_h1_accumulator(h1_acc[key], r)
            for r in p.get('communication_records', []):
                key = (int(r['codebook_size']), str(r['policy']))
                if key not in pol_acc: pol_acc[key] = _new_policy_acc()
                _update_policy(pol_acc[key], r)
            if h5_tot is None:
                h5_tot = {prof: {base: {'n':0,'baseline_violations':0,'recovery_violations':0} for base in bases} for prof, bases in ((prof, list(d.keys())) for prof,d in j['H5_raw_counts'].items())}
            for prof, dd in j['H5_raw_counts'].items():
                req(prof in h5_tot, f'H5 profile drift {prof}')
                for base, a in dd.items():
                    req(base in h5_tot[prof], f'H5 baseline drift {base}')
                    for k in ('n','baseline_violations','recovery_violations'): h5_tot[prof][base][k] += int(a[k])
            for a in h6_actions:
                for m in metric_names:
                    v = j['H6_metrics'][a][m]
                    if v is not None and math.isfinite(float(v)):
                        h6_sum[(a,m)] += float(v); h6_n[(a,m)] += 1
            row = {'formal_ordinal': i, 'scenario_id': sid, 'outcome_json': op.name, 'outcome_json_sha256': sha256_path(op), 'gt_open_intent_sha256': sha256_path(gp)}
            mf.write(canonical_bytes(row).decode('utf-8'))
            if (i+1) % 2000 == 0 or i+1 == FORMAL_N:
                print(f'FINALIZE {i+1}/{FORMAL_N}', flush=True)
        mf.flush(); os.fsync(mf.fileno())
    manifest = out / 'stage9_block913_v1_3_FINAL_H1_H6_manifest.jsonl'
    os.replace(manifest_tmp, manifest); os.chmod(manifest, 0o444)
    req(selected == 25239 and no_receiver == 970, f'final primary selected/no-receiver drift {selected}/{no_receiver}')
    r1_count = len(list((out / 'gt_reopen_recovery').glob('*.json')))
    r2_count = len(list((out / 'gt_reopen_recovery2').glob('*.json')))
    req(r1_count == 12 and r2_count == 12,
        f'technical recovery marker cardinality drift r1={r1_count} r2={r2_count}')
    req(unique_reread_scenes == 22,
        f'unique reread-scene count drift {unique_reread_scenes} != 22')
    req(extra_physical_reread_events == 24,
        f'extra physical reread-event count drift {extra_physical_reread_events} != 24')
    req(max_physical_read_number == 3,
        f'max physical future-GT read number drift {max_physical_read_number} != 3')
    req(h5_tot is not None and len(h5_tot) == 3 and all(len(dd) == 2 for dd in h5_tot.values()),
        'H5 comparison cardinality drift; expected exactly 3 profiles x 2 baselines')
    h1_summary = orig.finalize_h1(h1_acc)
    comm = {str(B): {} for B in sorted(set(B for B,_ in pol_acc))}
    for (B,p), a in sorted(pol_acc.items()): comm[str(B)][p] = _finish_policy(a)
    h5_raw = {}
    for prof, dd in h5_tot.items():
        h5_raw[prof] = {}
        for base, a in dd.items():
            n = int(a['n']); bv = int(a['baseline_violations']); rv = int(a['recovery_violations'])
            h5_raw[prof][base] = {
                **a, 'baseline_violation_rate': (bv/n if n else None), 'recovery_violation_rate': (rv/n if n else None),
                'recovery_minus_baseline_point_difference': ((rv-bv)/n if n else None),
            }
    h6_macro = {a:{m:{'n_finite':h6_n[(a,m)], 'mean': (h6_sum[(a,m)]/h6_n[(a,m)] if h6_n[(a,m)] else None)} for m in metric_names} for a in h6_actions}
    def hm(a,m):
        v = h6_macro[a][m]['mean']; req(v is not None, f'H6 empty support {a}/{m}'); return float(v)
    gates = {
        'vehicle_shadow_strict_improvement': hm('recovery','vehicle_shadow_zone_violation') < hm('reactive','vehicle_shadow_zone_violation'),
        'overmask_delta_le_0_02': hm('recovery','over_masking_area') - hm('reactive','over_masking_area') <= 0.02 + 1e-15,
        'pedestrian_visibility_delta_ge_minus_0_05': hm('recovery','pedestrian_visibility_proxy') - hm('reactive','pedestrian_visibility_proxy') >= -0.05 - 1e-15,
        'cyclist_visibility_delta_ge_minus_0_05': hm('recovery','cyclist_visibility_proxy') - hm('reactive','cyclist_visibility_proxy') >= -0.05 - 1e-15,
    }
    h6 = {
        'macro': h6_macro,
        'frozen_core_gates': gates,
        'frozen_core_gate_status': 'PASS_FROZEN_STAGE6_CORE_GATES' if all(gates.values()) else 'FAIL_H6_CORE',
        'road_illumination_retention': {
            'reactive_mean': hm('reactive','road_illumination_retention'),
            'stage6_v3_mean': hm('Stage6_V3','road_illumination_retention'),
            'recovery_mean': hm('recovery','road_illumination_retention'),
            'recovery_minus_reactive': hm('recovery','road_illumination_retention') - hm('reactive','road_illumination_retention'),
            'recovery_minus_stage6_v3': hm('recovery','road_illumination_retention') - hm('Stage6_V3','road_illumination_retention'),
            'formal_NI_threshold': None,
        },
    }
    summary = {
        'schema': 'stage9_block913_v1_3_FINAL_H1_H6_summary_v1',
        'status': 'FORMAL_RAW_OUTCOMES_H1_H6_COMPLETE',
        'FORMAL_scene_count': FORMAL_N, 'FORMAL_ids_sha256': FORMAL_IDS_SHA,
        'final_evaluator_runner_sha256': sha256_path(Path(__file__).resolve()),
        'block94b_evaluator_source_sha256': BLOCK94B_SHA,
        'block94b_reachable_repair_source_sha256': BLOCK94B_REPAIR_SOURCE_SHA,
        'primary_receiver_selected': selected, 'no_eligible_receiver': no_receiver, 'primary_future_GT_evaluated': primary_eval,
        'H1': native(h1_summary),
        'H2_H3_H4_communication_point_estimates': native(comm),
        'H5_six_raw_comparisons': native(h5_raw),
        'H5_bootstrap_performed_here': False,
        'H6': native(h6),
        'future_GT_opened': True, 'future_GT_evaluator_only': True, 'future_GT_controller_access': False,
        'model_forward_executed': False, 'C1_MC_recomputed': False, 'C2_solver_recomputed': False, 'Phase2B_endpoint_MC_replayed': False,
        'policy_retuned': False, 'thresholds_retuned': False, 'baseline_changed': False,
        'hypothesis_significance_testing_performed_here': False,
        'statistical_inference_deferred_to_9_15': True,
        'technical_recovery': {
            'documented': True,
            'recovery1_amendment_sha256': sha256_path(amendment1_path),
            'recovery2_amendment_sha256': sha256_path(amendment2_path),
            'failed_original_runner_sha256': FAILED_R3_RUNNER_SHA,
            'unique_scenes_with_physical_reread': unique_reread_scenes,
            'extra_physical_future_GT_read_events': extra_physical_reread_events,
            'max_physical_future_GT_read_number': max_physical_read_number,
            'literal_one_physical_read_per_scene_invariant_preserved': False,
            'H6_metric_definition_changed': False,
            'H6_role_equality_assertion_removed': True,
            'C3_targets_absent_due_frozen_stage6_ineligible_total': h6_c3_absent_due_ineligible_total,
            'outcome_based_tuning': False,
            'policy_threshold_baseline_change': False,
        },
        'next_block': '9.15_STATS_BOOTSTRAP_HYPOTHESES',
    }
    summary_path = out / 'stage9_block913_v1_3_FINAL_H1_H6_summary.json'
    atomic_json(summary_path, summary)
    seal = {
        'schema': 'stage9_block913_v1_3_FINAL_H1_H6_seal_v1',
        'status': 'FROZEN_COMPLETE_STAGE9_BLOCK913_V1_3_H1_H6_FORMAL_OUTCOMES',
        'FORMAL_scene_count': FORMAL_N, 'FORMAL_ids_sha256': FORMAL_IDS_SHA,
        'combined_preoutcome_seal_sha256': COMBINED_SHA,
        'R4_primary_pretruth_manifest_sha256': R4_MANIFEST_SHA,
        'C3_phase3_DIRECT_manifest_sha256': C3_MANIFEST_SHA,
        'C3_phase3_DIRECT_seal_sha256': C3_SEAL_SHA,
        'final_evaluator_preflight_seal_sha256': sha256_path(pre),
        'final_evaluator_runner_sha256': sha256_path(Path(__file__).resolve()),
        'block94b_evaluator_source_sha256': BLOCK94B_SHA,
        'block94b_reachable_repair_source_sha256': BLOCK94B_REPAIR_SOURCE_SHA,
        'final_outcome_manifest_sha256': sha256_path(manifest),
        'final_summary_sha256': sha256_path(summary_path),
        'H1_materialized': True, 'H2_materialized': True, 'H3_materialized': True, 'H4_materialized': True,
        'H5_all_six_comparisons_materialized': True, 'H6_materialized': True,
        'future_GT_access': True, 'future_GT_evaluator_only': True, 'future_GT_controller_access': False,
        'FORMAL_outcomes_computed': True,
        'FAST_excluded': False, 'H5_definition_changed': False, 'H6_definition_changed': False,
        'baselines_changed': False, 'thresholds_retuned': False, 'C3_reselected': False, 'post_FORMAL_retuning': False,
        'bootstrap_or_significance_performed_in_9_13': False,
        'technical_recovery_documented': True,
        'technical_recovery1_amendment_sha256': sha256_path(amendment1_path),
        'technical_recovery2_amendment_sha256': sha256_path(amendment2_path),
        'future_GT_unique_reread_scene_count': unique_reread_scenes,
        'future_GT_extra_physical_reread_event_count': extra_physical_reread_events,
        'future_GT_max_physical_read_number': max_physical_read_number,
        'literal_one_physical_read_per_scene_invariant_preserved': False,
        'H6_metric_definition_changed_by_recovery': False,
        'H6_role_equality_validation_repaired': True,
        'outcome_based_tuning_after_failed_attempt': False,
        'next_block': '9.15_STATS_BOOTSTRAP_HYPOTHESES',
    }
    seal_path = out / 'stage9_block913_v1_3_FINAL_H1_H6_seal.json'
    atomic_json(seal_path, seal)
    print('=' * 118)
    print('STAGE 9.13 FINAL H1-H6 = COMPLETE')
    print('MANIFEST_SHA256 =', sha256_path(manifest))
    print('SUMMARY_SHA256  =', sha256_path(summary_path))
    print('FINAL_SEAL_SHA256 =', sha256_path(seal_path))
    print('H6_CORE =', h6['frozen_core_gate_status'])
    print('STATISTICAL_INFERENCE_DEFERRED_TO_9_15 = TRUE')
    print('=' * 118, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--prepare-recovery-only', action='store_true')
    ap.add_argument('--prepare-recovery2-only', action='store_true')
    ap.add_argument('--bootstrap-only', action='store_true')
    ap.add_argument('--finalize-only', action='store_true')
    ap.add_argument('--shard-count', type=int, default=None)
    ap.add_argument('--shard-index', type=int, default=None)
    args = ap.parse_args(); out = args.out.resolve()
    modes = int(args.preflight_only) + int(args.prepare_recovery_only) + int(args.prepare_recovery2_only) + int(args.bootstrap_only) + int(args.finalize_only) + int(args.shard_count is not None or args.shard_index is not None)
    req(modes == 1, 'choose exactly one mode: --preflight-only, --prepare-recovery-only, --prepare-recovery2-only, --bootstrap-only, --finalize-only, or --shard-count/--shard-index')
    if args.preflight_only: preflight(out); return
    if args.prepare_recovery_only: prepare_recovery(out); return
    if args.prepare_recovery2_only: prepare_recovery2(out); return
    if args.bootstrap_only: bootstrap_only(out); return
    if args.finalize_only: finalize(out); return
    req(args.shard_count is not None and args.shard_index is not None, 'worker requires both shard args')
    worker(out, args.shard_count, args.shard_index)

if __name__ == '__main__':
    try:
        main()
    except FailClosed as e:
        print('FAIL_CLOSED:', e, file=sys.stderr, flush=True)
        raise SystemExit(2)
