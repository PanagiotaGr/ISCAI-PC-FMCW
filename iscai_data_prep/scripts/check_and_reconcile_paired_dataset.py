from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from waymo_open_dataset.protos import scenario_pb2


WORK_ROOT = Path('/waymo/iscai_data_prep')
SOURCE_ROOT = Path('/waymo/data/v1_3_0_scenario')
PAIRED_ROOT = Path('/waymo/data/paired_womd_lidar_v1_3_0')

INVENTORY_DB = WORK_ROOT / 'reports/paired_inventory.sqlite'
SELECTION_CONFIG = WORK_ROOT / 'manifests/selection_config.json'
CLASS_AUDIT = WORK_ROOT / 'reports/class_distribution_audit.json'
DELETION_REPORT = WORK_ROOT / 'reports/approved_branch_deletion_report.json'
SELECTION_MANIFESTS = {
    'training': WORK_ROOT / 'manifests/selected_training.jsonl',
    'validation': WORK_ROOT / 'manifests/selected_validation.jsonl',
}

GLOBAL_REPORT = WORK_ROOT / 'reports/global_reconciliation.json'
DATASET_MANIFEST = PAIRED_ROOT / 'manifests/dataset_manifest.json'
HARD_RESERVE_BYTES = 250 * 1024**3
SPLITS = ('training', 'validation')


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def id_set_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def expected_source_shards(split: str) -> list[str]:
    connection = sqlite3.connect(INVENTORY_DB)
    try:
        rows = connection.execute(
            '''
            SELECT shard_path
            FROM local_shards
            WHERE split = ? AND scan_complete = 1
            ORDER BY shard_path
            ''',
            (split,),
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def load_reports(split: str) -> list[dict[str, Any]]:
    report_dir = PAIRED_ROOT / 'reports' / 'shards' / split
    reports: list[dict[str, Any]] = []
    if not report_dir.exists():
        return reports
    for path in sorted(report_dir.glob('*.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        report['_report_path'] = str(path)
        reports.append(report)
    return reports


def count_lidar_files(split: str) -> int:
    root = PAIRED_ROOT / split / 'lidar'
    if not root.exists():
        return 0
    count = 0
    for prefix in root.iterdir():
        if prefix.is_dir():
            count += sum(
                1 for item in prefix.iterdir()
                if item.is_file() and item.suffix == '.tfrecord'
            )
    return count


def compact_motion_files(split: str) -> list[Path]:
    root = PAIRED_ROOT / split / 'motion'
    return sorted(root.glob('paired-*')) if root.exists() else []


def quick_summary() -> tuple[dict[str, Any], bool]:
    config = json.loads(SELECTION_CONFIG.read_text(encoding='utf-8'))
    summary: dict[str, Any] = {
        'selection_status': config.get('status'),
        'class_preservation_gate_pass': config.get(
            'class_preservation_gate_pass'
        ),
        'splits': {},
    }

    all_complete = True

    for split in SPLITS:
        expected_sources = expected_source_shards(split)
        expected_selected = int(config[f'{split}_scenarios'])
        reports = load_reports(split)
        statuses = Counter(
            str(report.get('status', 'MISSING')) for report in reports
        )

        selected_total = sum(
            int(report.get('selected_scenarios', 0)) for report in reports
        )
        motion_records = sum(
            int(report.get('motion', {}).get('records', 0))
            for report in reports
        )
        lidar_entries = sum(len(report.get('lidar', [])) for report in reports)
        source_deleted_true = sum(
            report.get('source_deleted') is True for report in reports
        )
        remaining_sources = sum(Path(path).exists() for path in expected_sources)
        actual_lidar_files = count_lidar_files(split)
        actual_motion_shards = len(compact_motion_files(split))

        complete = (
            len(reports) == len(expected_sources)
            and statuses == Counter({'COMMITTED_SOURCE_DELETED': len(expected_sources)})
            and source_deleted_true == len(expected_sources)
            and remaining_sources == 0
            and selected_total == expected_selected
            and motion_records == expected_selected
            and lidar_entries == expected_selected
            and actual_lidar_files == expected_selected
        )
        all_complete = all_complete and complete

        summary['splits'][split] = {
            'complete': complete,
            'expected_source_shards': len(expected_sources),
            'reports': len(reports),
            'statuses': dict(statuses),
            'source_deleted_true': source_deleted_true,
            'remaining_original_source_shards': remaining_sources,
            'expected_selected_scenarios': expected_selected,
            'reported_selected_scenarios': selected_total,
            'reported_motion_records': motion_records,
            'reported_lidar_entries': lidar_entries,
            'actual_lidar_files': actual_lidar_files,
            'compact_motion_shards': actual_motion_shards,
        }

    disk = shutil.disk_usage('/waymo')
    summary['storage'] = {
        'free_bytes': disk.free,
        'free_gib': disk.free / 1024**3,
        'hard_reserve_bytes': HARD_RESERVE_BYTES,
        'hard_reserve_gate_pass': disk.free >= HARD_RESERVE_BYTES,
    }
    summary['all_splits_complete'] = all_complete
    summary['global_reconciliation_ready'] = (
        all_complete
        and summary['storage']['hard_reserve_gate_pass']
        and summary['selection_status'] == 'frozen_class_audited'
        and summary['class_preservation_gate_pass'] is True
    )
    return summary, bool(summary['global_reconciliation_ready'])


def read_compact_motion_ids(path: Path) -> list[str]:
    scenario_ids: list[str] = []
    with path.open('rb') as stream:
        while True:
            header = stream.read(12)
            if not header:
                break
            if len(header) != 12:
                raise RuntimeError(f'Truncated TFRecord header: {path}')
            payload_length = struct.unpack('<Q', header[:8])[0]
            payload = stream.read(payload_length)
            data_crc = stream.read(4)
            if len(payload) != payload_length or len(data_crc) != 4:
                raise RuntimeError(f'Truncated TFRecord record: {path}')
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(payload)
            if not scenario.scenario_id:
                raise RuntimeError(f'Empty motion scenario_id in {path}')
            scenario_ids.append(scenario.scenario_id)
    return scenario_ids


def report_ledger_sha256(report_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(report_paths):
        digest.update(str(path.relative_to(PAIRED_ROOT)).encode('utf-8'))
        digest.update(b'\0')
        digest.update(sha256_file(path).encode('ascii'))
        digest.update(b'\n')
    return digest.hexdigest()


def reconcile() -> dict[str, Any]:
    quick, ready = quick_summary()
    if not ready:
        blocked = {
            'status': 'BLOCKED_INCOMPLETE',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'quick_completion': quick,
            'message': (
                'Global reconciliation requires completed training and '
                'validation migrations.'
            ),
        }
        atomic_json(GLOBAL_REPORT, blocked)
        return blocked

    config = json.loads(SELECTION_CONFIG.read_text(encoding='utf-8'))
    class_audit = json.loads(CLASS_AUDIT.read_text(encoding='utf-8'))

    result: dict[str, Any] = {
        'status': 'RUNNING',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'splits': {},
        'errors': [],
    }

    split_manifest_ids: dict[str, set[str]] = {}
    all_report_paths: list[Path] = []

    for split in SPLITS:
        rows = load_jsonl(SELECTION_MANIFESTS[split])
        manifest_ids_list = [str(row['scenario_id']) for row in rows]
        manifest_ids = set(manifest_ids_list)
        split_manifest_ids[split] = manifest_ids

        if len(manifest_ids) != len(manifest_ids_list):
            result['errors'].append(f'{split}: duplicate IDs in selection manifest')

        expected_by_source: dict[str, list[str]] = {}
        expected_lidar_bytes: dict[str, int] = {}
        for row in rows:
            expected_by_source.setdefault(str(row['source_shard']), []).append(
                str(row['scenario_id'])
            )
            expected_lidar_bytes[str(row['scenario_id'])] = int(row['lidar_bytes'])

        for source, ids in expected_by_source.items():
            source_rows = [row for row in rows if str(row['source_shard']) == source]
            source_rows.sort(key=lambda row: int(row['record_offset']))
            expected_by_source[source] = [str(row['scenario_id']) for row in source_rows]

        reports = load_reports(split)
        reports_by_source = {str(report['source_shard']): report for report in reports}
        all_report_paths.extend(Path(report['_report_path']) for report in reports)

        report_lidar_ids: list[str] = []
        parsed_motion_ids: list[str] = []
        expected_motion_paths: set[Path] = set()
        size_mismatches = 0
        missing_lidar_files = 0
        motion_mapping_failures = 0
        report_mapping_failures = 0

        for source in expected_source_shards(split):
            report = reports_by_source.get(source)
            if report is None:
                result['errors'].append(f'{split}: missing report for {source}')
                continue

            expected_ids = expected_by_source.get(source, [])
            report_ids = [str(item['scenario_id']) for item in report.get('lidar', [])]
            report_lidar_ids.extend(report_ids)

            if report.get('status') != 'COMMITTED_SOURCE_DELETED':
                report_mapping_failures += 1
            if int(report.get('selected_scenarios', -1)) != len(expected_ids):
                report_mapping_failures += 1
            if int(report.get('motion', {}).get('records', -1)) != len(expected_ids):
                report_mapping_failures += 1
            if report_ids != expected_ids:
                report_mapping_failures += 1

            motion_output = report.get('motion_output')
            if expected_ids:
                if not motion_output:
                    motion_mapping_failures += 1
                else:
                    motion_path = Path(motion_output)
                    expected_motion_paths.add(motion_path)
                    if not motion_path.is_file():
                        motion_mapping_failures += 1
                    else:
                        ids = read_compact_motion_ids(motion_path)
                        parsed_motion_ids.extend(ids)
                        if ids != expected_ids:
                            motion_mapping_failures += 1
            elif motion_output:
                motion_mapping_failures += 1

            for item in report.get('lidar', []):
                scenario_id = str(item['scenario_id'])
                path = (
                    PAIRED_ROOT / split / 'lidar' / scenario_id[:2]
                    / f'{scenario_id}.tfrecord'
                )
                if not path.is_file():
                    missing_lidar_files += 1
                    continue
                expected_size = expected_lidar_bytes[scenario_id]
                if path.stat().st_size != expected_size:
                    size_mismatches += 1

        actual_motion_paths = set(compact_motion_files(split))
        actual_lidar_ids = {
            path.stem
            for prefix in (PAIRED_ROOT / split / 'lidar').iterdir()
            if prefix.is_dir()
            for path in prefix.iterdir()
            if path.is_file() and path.suffix == '.tfrecord'
        }

        report_lidar_set = set(report_lidar_ids)
        parsed_motion_set = set(parsed_motion_ids)

        split_pass = all([
            report_mapping_failures == 0,
            motion_mapping_failures == 0,
            missing_lidar_files == 0,
            size_mismatches == 0,
            len(report_lidar_ids) == len(report_lidar_set),
            len(parsed_motion_ids) == len(parsed_motion_set),
            report_lidar_set == manifest_ids,
            parsed_motion_set == manifest_ids,
            actual_lidar_ids == manifest_ids,
            actual_motion_paths == expected_motion_paths,
        ])

        if not split_pass:
            result['errors'].append(f'{split}: exact global reconciliation failed')

        result['splits'][split] = {
            'pass': split_pass,
            'selected_scenarios': len(manifest_ids),
            'selection_id_set_sha256': id_set_sha256(manifest_ids),
            'report_lidar_ids': len(report_lidar_ids),
            'parsed_motion_ids': len(parsed_motion_ids),
            'actual_lidar_files': len(actual_lidar_ids),
            'compact_motion_files': len(actual_motion_paths),
            'report_mapping_failures': report_mapping_failures,
            'motion_mapping_failures': motion_mapping_failures,
            'missing_lidar_files': missing_lidar_files,
            'lidar_size_mismatches': size_mismatches,
            'duplicate_report_lidar_ids': len(report_lidar_ids) - len(report_lidar_set),
            'duplicate_parsed_motion_ids': len(parsed_motion_ids) - len(parsed_motion_set),
            'report_ids_match_manifest': report_lidar_set == manifest_ids,
            'motion_ids_match_manifest': parsed_motion_set == manifest_ids,
            'actual_lidar_ids_match_manifest': actual_lidar_ids == manifest_ids,
            'motion_file_set_matches_reports': actual_motion_paths == expected_motion_paths,
        }

    cross_split_overlap = split_manifest_ids['training'] & split_manifest_ids['validation']
    disk = shutil.disk_usage('/waymo')
    reserve_pass = disk.free >= HARD_RESERVE_BYTES
    overall_pass = (
        not result['errors']
        and not cross_split_overlap
        and reserve_pass
        and config.get('status') == 'frozen_class_audited'
        and config.get('class_preservation_gate_pass') is True
        and class_audit.get('status') == 'PASS_FROZEN'
    )

    result.update({
        'status': 'PASS_COMPLETE' if overall_pass else 'FAIL',
        'cross_split_overlap_count': len(cross_split_overlap),
        'free_bytes': disk.free,
        'free_gib': disk.free / 1024**3,
        'hard_reserve_bytes': HARD_RESERVE_BYTES,
        'hard_reserve_gate_pass': reserve_pass,
        'selection_config_sha256': sha256_file(SELECTION_CONFIG),
        'class_audit_sha256': sha256_file(CLASS_AUDIT),
        'training_selection_manifest_sha256': sha256_file(
            SELECTION_MANIFESTS['training']
        ),
        'validation_selection_manifest_sha256': sha256_file(
            SELECTION_MANIFESTS['validation']
        ),
        'deletion_report_sha256': (
            sha256_file(DELETION_REPORT) if DELETION_REPORT.is_file() else None
        ),
        'shard_report_ledger_sha256': report_ledger_sha256(all_report_paths),
        'per_shard_data_checksums_reused': True,
        'full_data_rehash_performed': False,
    })

    atomic_json(GLOBAL_REPORT, result)

    if overall_pass:
        dataset_manifest = {
            'status': 'COMPLETE_FROZEN',
            'dataset_release': config['dataset_release'],
            'created_utc': result['created_utc'],
            'association_key': config['association_key'],
            'hard_free_space_reserve_gib': config['hard_free_space_reserve_gib'],
            'selection': {
                'training_policy': config['training_selection_policy'],
                'validation_policy': config['validation_selection_policy'],
                'training_scenarios': config['training_scenarios'],
                'validation_scenarios': config['validation_scenarios'],
                'total_scenarios': config['total_scenarios'],
                'total_pair_bytes': config['total_pair_bytes'],
            },
            'class_preservation_gate_pass': True,
            'splits': result['splits'],
            'cross_split_overlap_count': 0,
            'storage': {
                'free_bytes_at_freeze': disk.free,
                'hard_reserve_gate_pass': True,
            },
            'provenance_sha256': {
                'selection_config': result['selection_config_sha256'],
                'class_audit': result['class_audit_sha256'],
                'selected_training': result['training_selection_manifest_sha256'],
                'selected_validation': result['validation_selection_manifest_sha256'],
                'approved_deletion_report': result['deletion_report_sha256'],
                'shard_report_ledger': result['shard_report_ledger_sha256'],
                'global_reconciliation': sha256_file(GLOBAL_REPORT),
            },
            'verification_scope': {
                'exact_motion_id_reconciliation': True,
                'exact_lidar_filename_id_reconciliation': True,
                'lidar_size_reconciliation': True,
                'per_shard_checksums_reused': True,
                'full_data_rehash_performed': False,
            },
        }
        atomic_json(DATASET_MANIFEST, dataset_manifest)

    return result


def print_quick(summary: dict[str, Any]) -> None:
    print('=== QUICK COMPLETION CHECK ===')
    for split in SPLITS:
        item = summary['splits'][split]
        print(f'\n{split.upper()}')
        print('  complete:', item['complete'])
        print(
            '  reports:',
            f"{item['reports']}/{item['expected_source_shards']}",
        )
        print('  statuses:', item['statuses'])
        print('  remaining original shards:', item['remaining_original_source_shards'])
        print(
            '  selected:',
            f"{item['reported_selected_scenarios']}/"
            f"{item['expected_selected_scenarios']}",
        )
        print(
            '  motion records / LiDAR entries / actual LiDAR:',
            item['reported_motion_records'],
            item['reported_lidar_entries'],
            item['actual_lidar_files'],
        )
    print('\nFree GiB:', f"{summary['storage']['free_gib']:.2f}")
    print('Hard reserve gate:', summary['storage']['hard_reserve_gate_pass'])
    print('GLOBAL READY:', summary['global_reconciliation_ready'])


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--quick', action='store_true')
    mode.add_argument('--reconcile', action='store_true')
    args = parser.parse_args()

    if args.quick:
        summary, ready = quick_summary()
        print_quick(summary)
        raise SystemExit(0 if ready else 2)

    result = reconcile()
    print('=== GLOBAL RECONCILIATION ===')
    print('Status:', result['status'])
    if result['status'] == 'BLOCKED_INCOMPLETE':
        print_quick(result['quick_completion'])
    else:
        for split in SPLITS:
            item = result['splits'][split]
            print(
                f"{split}: pass={item['pass']} "
                f"scenarios={item['selected_scenarios']} "
                f"motion={item['parsed_motion_ids']} "
                f"lidar={item['actual_lidar_files']}"
            )
        print('Cross-split overlap:', result['cross_split_overlap_count'])
        print('Hard reserve gate:', result['hard_reserve_gate_pass'])
        print('Report:', GLOBAL_REPORT)
        if result['status'] == 'PASS_COMPLETE':
            print('Dataset manifest:', DATASET_MANIFEST)
    raise SystemExit(0 if result['status'] == 'PASS_COMPLETE' else 2)


if __name__ == '__main__':
    main()
