#!/usr/bin/env python3
"""Independent read-only checker for Stage 7 Block 7.5C1.

This checker never opens raw formal WOMD scenarios and never accesses future GT.
It verifies only the durable causal/non-oracle lock produced by C1.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path('/home/agni/waymo')
S3 = ROOT / 'iscai_stage3'
S7 = ROOT / 'iscai_stage7'

RUNNER = S7 / 'scripts/run_block75c1_formal_causal_output_lock.py'
CONTRACT = S7 / 'configs/stage7_block75b_final_formal_evaluator_contract.json'
FORMAL = S3 / 'artifacts/block38e/formal_validation_120.jsonl'
MANIFEST = S7 / 'artifacts/block75c1_nonoracle_lock_manifest.json'
REPORT = S7 / 'reports/stage7_block75c1_nonoracle_lock_report.json'

EXPECTED_CONTRACT_SHA = '5f79169138653b8223a795fae625adc5859a7a046a67c79fa98d357e256d7029'
EXPECTED_FORMAL_SHA = '2208e7287ddf6439fda4597c435a9cba1d1b9d0e4c4547bc5dd92e56e8124e46'
EXPECTED_ORDER_SHA = '4ba67f109c67f1306374423429654415ef884217dd93371f33e3d8fe27d948b7'
SYSTEMS = (
    'shared_trajectory_posterior',
    'independent_models',
    'direct_beam_classifier',
    'direct_ADB_predictor',
    'deterministic_shared_trajectory',
)
NPZ_ARRAYS = (
    'current_reactive_t0',
    'shared_trajectory_posterior',
    'independent_models',
    'direct_ADB_predictor',
    'deterministic_shared_trajectory',
)

class FailClosed(RuntimeError):
    pass

def require(cond: bool, message: str) -> None:
    if not cond:
        raise FailClosed(message)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')

def read_json(path: Path) -> Any:
    require(path.is_file(), f'Missing JSON: {path}')
    return json.loads(path.read_text(encoding='utf-8'))

def main() -> int:
    print('=' * 78)
    print('STAGE 7 — BLOCK 7.5C1 INDEPENDENT NON-ORACLE LOCK CHECK')
    print('READ ONLY / NO RAW WOMD / NO FUTURE GT')
    print('=' * 78)

    for path in (RUNNER, CONTRACT, FORMAL, MANIFEST, REPORT):
        require(path.is_file(), f'Missing required file: {path}')

    require(sha256_file(CONTRACT) == EXPECTED_CONTRACT_SHA, 'Block7.5B contract SHA mismatch.')
    require(sha256_file(FORMAL) == EXPECTED_FORMAL_SHA, 'Formal manifest SHA mismatch.')
    runner_sha = sha256_file(RUNNER)

    rows = [json.loads(x) for x in FORMAL.read_text(encoding='utf-8').splitlines() if x.strip()]
    ids = [str(x['scenario_id']) for x in rows]
    require(len(ids) == 120 and len(set(ids)) == 120, 'Formal manifest is not 120 unique scenarios.')
    order_sha = hashlib.sha256(json.dumps(ids, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()
    require(order_sha == EXPECTED_ORDER_SHA, 'Formal scenario order SHA mismatch.')

    manifest = read_json(MANIFEST)
    report = read_json(REPORT)

    require(manifest.get('status') == 'FROZEN_COMPLETE_NONORACLE_OUTPUT_LOCK', 'Manifest status invalid.')
    require(manifest.get('runner_sha256') == runner_sha, 'Manifest runner SHA does not match installed runner.')
    require(manifest.get('block75b_contract_sha256') == EXPECTED_CONTRACT_SHA, 'Manifest contract SHA mismatch.')
    require(manifest.get('scenario_lock_count') == 120, 'Manifest scenario lock count !=120.')
    require(manifest.get('systems') == list(SYSTEMS), 'Manifest five-system order changed.')
    require(manifest.get('future_GT_accessed') is False, 'Manifest says future GT accessed.')
    require(manifest.get('formal_metrics_computed') is False, 'Manifest says formal metrics computed.')
    require(manifest.get('post_outcome_tuning') is False, 'Manifest says post-outcome tuning occurred.')
    require(manifest.get('Stage4_modified') is False, 'Manifest says Stage4 modified.')
    require(manifest.get('Stage5_modified') is False, 'Manifest says Stage5 modified.')
    require(manifest.get('Stage6_modified') is False, 'Manifest says Stage6 modified.')

    formal_meta = manifest.get('formal_manifest', {})
    require(formal_meta.get('sha256') == EXPECTED_FORMAL_SHA, 'Manifest formal SHA mismatch.')
    require(formal_meta.get('scenario_count') == 120, 'Manifest formal count mismatch.')
    require(formal_meta.get('scenario_order_sha256') == EXPECTED_ORDER_SHA, 'Manifest formal order mismatch.')

    locks = manifest.get('scenario_locks')
    require(isinstance(locks, list) and len(locks) == 120, 'scenario_locks must contain 120 rows.')
    require([str(x.get('scenario_id')) for x in locks] == ids, 'Manifest lock scenario order mismatch.')
    require([int(x.get('formal_index')) for x in locks] == list(range(120)), 'Formal indices are not 0..119.')

    lock_digests: list[str] = []
    no_receiver = 0
    checked_npz_bytes = 0

    for expected_index, entry in enumerate(locks):
        sid = ids[expected_index]
        require(str(entry.get('scenario_id')) == sid, f'Entry {expected_index}: scenario mismatch.')

        lock_json = Path(str(entry['lock_json']))
        lock_npz = Path(str(entry['adb_schedules_npz']))
        require(lock_json.is_file(), f'Missing lock JSON: {lock_json}')
        require(lock_npz.is_file(), f'Missing lock NPZ: {lock_npz}')
        require(sha256_file(lock_json) == entry['lock_json_sha256'], f'{sid}: JSON SHA mismatch.')
        require(sha256_file(lock_npz) == entry['adb_schedules_npz_sha256'], f'{sid}: NPZ SHA mismatch.')

        doc = read_json(lock_json)
        require(doc.get('status') == 'LOCKED_NONORACLE_CAUSAL_OUTPUTS', f'{sid}: lock status invalid.')
        require(doc.get('formal_index') == expected_index, f'{sid}: formal index mismatch.')
        require(str(doc.get('scenario_id')) == sid, f'{sid}: JSON scenario mismatch.')
        require(doc.get('runner_sha256') == runner_sha, f'{sid}: runner SHA mismatch.')
        require(doc.get('block75b_contract_sha256') == EXPECTED_CONTRACT_SHA, f'{sid}: contract SHA mismatch.')
        require(doc.get('formal_manifest_sha256') == EXPECTED_FORMAL_SHA, f'{sid}: formal SHA mismatch.')
        require(doc.get('formal_order_sha256') == EXPECTED_ORDER_SHA, f'{sid}: order SHA mismatch.')
        require(doc.get('future_GT_accessed') is False, f'{sid}: future GT accessed.')
        require(doc.get('formal_metrics_computed') is False, f'{sid}: formal metrics computed.')
        require(doc.get('post_outcome_tuning') is False, f'{sid}: post-outcome tuning flag true.')
        require(doc.get('scientific_parameters_modified') is False, f'{sid}: scientific parameters modified.')

        causal = doc.get('causal_input', {})
        require(causal.get('tracks_to_predict_accessed') is False, f'{sid}: tracks_to_predict accessed.')
        require(causal.get('objects_of_interest_accessed') is False, f'{sid}: objects_of_interest accessed.')
        require(causal.get('future_state_accessed') is False, f'{sid}: future state accessed.')

        comm = doc.get('communication_outputs')
        require(isinstance(comm, dict), f'{sid}: communication_outputs missing.')
        # communication_outputs is serialized by the C1 runner with canonical
        # JSON sort_keys=True, so object key insertion order is intentionally
        # not preserved on disk. The frozen system ORDER is already checked
        # above via manifest['systems'] (a JSON list). Here the correct
        # independent invariant is the exact five-system SET with no extras.
        require(
            len(comm) == len(SYSTEMS) and set(comm.keys()) == set(SYSTEMS),
            f'{sid}: communication system set changed: {tuple(comm.keys())}',
        )

        aliases = doc.get('ADB_outputs', {}).get('aliases')
        require(aliases == {'direct_beam_classifier': 'shared_trajectory_posterior'}, f'{sid}: ADB alias changed.')
        require(comm['direct_ADB_predictor'].get('status') in ('REUSE_SHARED_COMMUNICATION_BRANCH', 'NO_ELIGIBLE_RECEIVER'), f'{sid}: direct-ADB communication alias status invalid.')

        proof = doc.get('shared_posterior_runtime_identity', {})
        require(proof.get('same_content_identity') is True, f'{sid}: shared posterior identity failed.')
        require(proof.get('shared_predictor_forward_count_for_scene') == 1, f'{sid}: shared predictor forward count !=1.')
        rid = proof.get('selected_receiver_prediction_id')
        if rid is None:
            no_receiver += 1
            require(proof.get('shared_posterior_digest_sha256') is None, f'{sid}: no-receiver shared digest should be null.')
            require(proof.get('communication_branch_input_digest_sha256') is None, f'{sid}: no-receiver comm digest should be null.')
            require(proof.get('ADB_branch_input_digest_sha256') is None, f'{sid}: no-receiver ADB digest should be null.')
        else:
            d = proof.get('shared_posterior_digest_sha256')
            require(isinstance(d, str) and len(d) == 64, f'{sid}: shared digest malformed.')
            require(proof.get('communication_branch_input_digest_sha256') == d, f'{sid}: communication digest differs.')
            require(proof.get('ADB_branch_input_digest_sha256') == d, f'{sid}: ADB digest differs.')
            require(comm['shared_trajectory_posterior'].get('posterior_digest_sha256') == d, f'{sid}: communication output digest differs.')

        core = dict(doc)
        stored_nonoracle = core.pop('nonoracle_output_digest_sha256', None)
        recomputed_nonoracle = hashlib.sha256(canonical_bytes(core)).hexdigest()
        require(stored_nonoracle == recomputed_nonoracle, f'{sid}: nonoracle output digest mismatch.')
        require(entry.get('nonoracle_output_digest_sha256') == stored_nonoracle, f'{sid}: manifest nonoracle digest mismatch.')
        lock_digests.append(stored_nonoracle)

        with np.load(lock_npz, allow_pickle=False) as z:
            require(tuple(z.files) == NPZ_ARRAYS, f'{sid}: NPZ field set/order changed: {z.files}')
            current = z['current_reactive_t0']
            require(current.shape == (501, 301), f'{sid}: current t0 shape {current.shape}')
            require(current.dtype == np.float64, f'{sid}: current t0 dtype {current.dtype}')
            require(np.all(np.isfinite(current)), f'{sid}: current t0 non-finite.')
            require(np.all((current >= 0.0) & (current <= 1.0)), f'{sid}: current t0 outside [0,1].')
            for key in NPZ_ARRAYS[1:]:
                arr = z[key]
                require(arr.shape == (4, 501, 301), f'{sid}: {key} shape {arr.shape}')
                require(arr.dtype == np.float64, f'{sid}: {key} dtype {arr.dtype}')
                require(np.all(np.isfinite(arr)), f'{sid}: {key} non-finite.')
                require(np.all((arr >= 0.0) & (arr <= 1.0)), f'{sid}: {key} outside [0,1].')
        checked_npz_bytes += lock_npz.stat().st_size

        if expected_index % 10 == 0 or expected_index == 119:
            print(f'PASS lock {expected_index:03d}/119 {sid}')

    require(len(set(lock_digests)) == 120, 'Non-oracle scenario lock digests are not unique.')

    manifest_sha = sha256_file(MANIFEST)
    require(report.get('status') == 'PASS', 'C1 report status invalid.')
    require(report.get('manifest_path') == str(MANIFEST), 'C1 report manifest path mismatch.')
    require(report.get('manifest_sha256') == manifest_sha, 'C1 report manifest SHA mismatch.')
    require(report.get('runner_sha256') == runner_sha, 'C1 report runner SHA mismatch.')
    require(report.get('formal_causal_scenarios') == 120, 'C1 report formal count mismatch.')
    require(report.get('five_systems_locked') is True, 'C1 report five-system lock false.')
    require(report.get('shared_posterior_runtime_identity_proof') is True, 'C1 report identity proof false.')
    require(report.get('nonoracle_lock_complete') is True, 'C1 report lock incomplete.')
    require(report.get('future_GT_accessed') is False, 'C1 report future GT accessed.')
    require(report.get('formal_metrics_computed') is False, 'C1 report metrics computed.')
    require(report.get('Stage4_5_6_modified') is False, 'C1 report upstream modification.')
    require(report.get('post_outcome_tuning') is False, 'C1 report post-outcome tuning.')

    print('\n===== FINAL =====')
    print('runner SHA256 =', runner_sha)
    print('manifest SHA256 =', manifest_sha)
    print('formal locks verified = 120/120')
    print('unique nonoracle digests = 120/120')
    print('no-eligible-receiver scenarios =', no_receiver)
    print('verified NPZ bytes =', checked_npz_bytes)
    print('raw formal WOMD opened by checker = NO')
    print('future GT opened by checker = NO')
    print('formal metrics computed by checker = NO')
    print('INDEPENDENT C1 CHECK = PASS_100_PERCENT')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
