#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path('/home/agni/waymo')
S8 = ROOT / 'iscai_stage8'
DATA_ROOT = S8 / 'data/block82b_deepsense_primary'

SCRIPT = S8 / 'scripts/run_stage8_block85b_train_frozen_encoder.py'
CONTRACT85A = S8 / 'configs/stage8_block85a_encoder_preregistration_contract.json'
MODEL_SRC = S8 / 'src/deepsense_lidar_encoder.py'
LOADER_SRC = S8 / 'src/mat5_lidar_loader.py'

B85A_ROOT = S8 / 'artifacts/block85a_encoder_preregistration'
B85A_ARCH = B85A_ROOT / 'frozen_encoder_architecture.json'
B85A_RECIPE = B85A_ROOT / 'frozen_training_recipe.json'
B85A_MANIFEST = B85A_ROOT / 'stage8_block85a_manifest.json'
B85A_SEAL = B85A_ROOT / 'stage8_block85a_encoder_preregistration_seal.json'
B85A_REPORT = S8 / 'reports/stage8_block85a_encoder_preregistration_report.json'
B85A_RUNNER = S8 / 'scripts/run_stage8_block85a_encoder_preregistration_freeze.py'

B82B_FILES = S8 / 'artifacts/block82b_canonical_corpus_schema_split/canonical_extracted_file_manifest.json'
B82B_SEAL = S8 / 'artifacts/block82b_canonical_corpus_schema_split/stage8_block82b_canonical_corpus_schema_split_seal.json'

OUT = S8 / 'artifacts/block85b_encoder_training'
NORM = OUT / 'train_only_normalization.json'
TRAIN_INDEX = OUT / 'frozen_train_sample_index.json'
LOG = OUT / 'training_log.json'
CHECKPOINT = OUT / 'duallidarconv1d64_epoch080.pt'
MANIFEST = OUT / 'stage8_block85b_manifest.json'
REPORT = S8 / 'reports/stage8_block85b_encoder_training_report.json'
SEAL = OUT / 'stage8_block85b_encoder_training_seal.json'

EXPECTED = {
    str(CONTRACT85A): 'b0bab29b655aca9e8f29b77fba73d1aeaa1dc4ac444269d254fc7d3b0d4fc45d',
    str(MODEL_SRC): 'adb943f12490612a6b92ffa3e70f69b96b1002fa77e4e769f5ea26ed6d25c417',
    str(LOADER_SRC): 'b4b3d9e09a20a5ee790a4ad7fd22f53dad1d6f4b953e70f824ade80f9b86299c',
    str(B85A_ARCH): '28f8332aa19173ec0267a5efa4590ecd5e44fd1db8b84384784fb0a4354848ad',
    str(B85A_RECIPE): '326bb570ed7cb65a819fcd2cc14d3371246c1678aaf7e3ede4b3e30113c3f446',
    str(B85A_MANIFEST): 'a1e813719b0e93b6e6fba42cd95bf194059cf0525f1088109de974c915474cfa',
    str(B85A_REPORT): '698cd6923756c643b0328b502b3563c6f46bc101b3fce984bb2a905433def970',
    str(B85A_SEAL): '2d43c234dbe42b86f3f3f58a57d582c9ea92253fbe62dd3091d4fd91cff98fb4',
    str(B85A_RUNNER): 'be1f3b49a84047a426748ff7e1577c5238805c2d3b8b658e36294a595263093c',
    str(B82B_FILES): '79dc92fb6a90acd36c4598de436517ebac6c7e7904334dd76a520e1c90a0b895',
    str(B82B_SEAL): 'c84eb7ef45d16a1feb3ef068bdaf498e5e4298514f3c4807ffe10db9b1112ba3',
}

TRAIN_SPECS = [
    (8, DATA_ROOT / 'LiDAR/Scenario8/development_dataset/scenario8_dev_train.csv'),
    (9, DATA_ROOT / 'LiDAR/Scenario9/development_dataset/scenario9_dev_train.csv'),
]
SEED = 20260902

class FailClosed(RuntimeError):
    pass

def require(c, m):
    if not c:
        raise FailClosed(m)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f'Refusing overwrite: {path}')
    data = (json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.tmp.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def atomic_torch_save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f'Refusing overwrite: {path}')
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.tmp.', dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(obj, tmp)
        with open(tmp, 'rb') as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def resolve_ref(csv_path: Path, raw_ref: str) -> Path:
    s = (raw_ref or '').strip().replace('\\', '/')
    while s.startswith('./'):
        s = s[2:]
    if s.startswith('LiDAR/'):
        p = DATA_ROOT / s
    else:
        p = csv_path.parent / s
    return p.resolve()

def rel(p: Path) -> str:
    return str(p.relative_to(DATA_ROOT))

def stable_sample_id(scenario, csv_path, row_number, raw_path, power_ref):
    payload = '|'.join([
        str(scenario),
        'TRAIN',
        rel(csv_path),
        str(row_number),
        rel(raw_path),
        power_ref,
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def build_train_records():
    records = []
    scenario_counts = {}
    expected_headers = ['index','unit1_lidar_1','unit1_lidar_SCR_1','unit1_pwr_1','beam_index_1']
    for scenario, csv_path in TRAIN_SPECS:
        require(csv_path.is_file(), f'Missing TRAIN CSV: {csv_path}')
        count = 0
        with csv_path.open(newline='', encoding='utf-8-sig') as f:
            r = csv.DictReader(f)
            require(r.fieldnames == expected_headers,
                    f'Unexpected TRAIN CSV headers in {csv_path}: {r.fieldnames}')
            for row_number, row in enumerate(r, start=2):
                raw_path = resolve_ref(csv_path, row['unit1_lidar_1'])
                scr_path = resolve_ref(csv_path, row['unit1_lidar_SCR_1'])
                power_ref = (row['unit1_pwr_1'] or '').strip().replace('\\','/')
                # Important: power_ref is retained as text identity only. Payload is never opened here.
                try:
                    label = int(float(row['beam_index_1']))
                except Exception:
                    raise FailClosed(f'Invalid TRAIN beam label {csv_path}:{row_number}')
                require(0 <= label <= 63, f'TRAIN beam label outside 0..63 {csv_path}:{row_number}')
                require(raw_path.is_file() and scr_path.is_file(),
                        f'TRAIN LiDAR reference missing {csv_path}:{row_number}')
                require('/development_dataset/' in ('/' + rel(raw_path)), 'Raw LiDAR escaped development_dataset')
                require('/development_dataset/' in ('/' + rel(scr_path)), 'SCR LiDAR escaped development_dataset')
                records.append({
                    'scenario': scenario,
                    'csv': csv_path,
                    'row_number': row_number,
                    'raw': raw_path,
                    'scr': scr_path,
                    'power_ref_text_only': power_ref,
                    'label': label,
                    'sample_id': stable_sample_id(scenario, csv_path, row_number, raw_path, power_ref),
                })
                count += 1
        scenario_counts[str(scenario)] = count
    require(scenario_counts == {'8': 2831, '9': 4199}, f'Unexpected TRAIN counts: {scenario_counts}')
    require(len(records) == 7030, f'Expected 7030 TRAIN rows, got {len(records)}')
    require(len({r['sample_id'] for r in records}) == len(records), 'Duplicate TRAIN sample IDs')
    return records, scenario_counts

def load_frozen_hash_map():
    doc = json.loads(B82B_FILES.read_text())
    require(doc.get('status') == 'FROZEN_CANONICAL_EXTRACTED_FILE_MANIFEST', '8.2B file manifest status changed')
    return {x['relative_path']: x['sha256'] for x in doc['files']}

def verify_accessed_input_hashes(records, frozen_hashes):
    paths = set()
    for _, csv_path in TRAIN_SPECS:
        paths.add(csv_path)
    for r in records:
        paths.add(r['raw'])
        paths.add(r['scr'])

    def verify_one(p):
        rp = rel(p)
        expected = frozen_hashes.get(rp)
        require(expected is not None, f'Input not present in frozen 8.2B manifest: {rp}')
        actual = sha256_file(p)
        require(actual == expected, f'Frozen TRAIN input SHA changed: {rp}')
        return rp

    workers = max(1, min(os.cpu_count() or 1, 12))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        checked = list(ex.map(verify_one, sorted(paths)))
    return len(checked), workers

def load_arrays(records, loader):
    n = len(records)
    raw = np.empty((n, 460, 2), dtype=np.float32)
    scr = np.empty((n, 216, 2), dtype=np.float32)
    labels = np.empty((n,), dtype=np.int64)

    def one(item):
        i, rec = item
        a = loader.load_mat5_double_matrix(rec['raw'], expected_name='data', expected_shape=(460, 2))
        b = loader.load_mat5_double_matrix(rec['scr'], expected_name='data', expected_shape=(216, 2))
        return i, a.astype(np.float32, copy=False), b.astype(np.float32, copy=False), rec['label']

    workers = max(1, min(os.cpu_count() or 1, 12))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, a, b, y in ex.map(one, enumerate(records), chunksize=16):
            raw[i] = a
            scr[i] = b
            labels[i] = y
    return raw, scr, labels, workers

def channel_stats(x):
    xd = x.astype(np.float64, copy=False)
    mean = xd.mean(axis=(0,1))
    std = xd.std(axis=(0,1), ddof=0)
    require(np.isfinite(mean).all() and np.isfinite(std).all(), 'Non-finite TRAIN normalization stats')
    require((std > 0).all(), f'Zero TRAIN normalization std: {std}')
    return mean, std

def worker_init_fn(worker_id):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

def main():
    print('='*88)
    print('STAGE 8 — BLOCK 8.5B')
    print('FIXED-RECIPE DEEPSENSE ENCODER TRAINING')
    print('TRAIN SPLIT ONLY — NO CALIBRATION / FORMAL / MMWAVE POWER ACCESS')
    print('='*88)

    require(Path(__file__).resolve() == SCRIPT.resolve(), 'Runner path mismatch')
    for raw, expected in EXPECTED.items():
        p = Path(raw)
        require(p.is_file(), f'Missing frozen input: {p}')
        require(sha256_file(p) == expected, f'Frozen input SHA changed: {p}')
    for p in (NORM, TRAIN_INDEX, LOG, CHECKPOINT, MANIFEST, REPORT, SEAL):
        require(not p.exists(), f'8.5B output exists: {p}')

    seal85a = json.loads(B85A_SEAL.read_text())
    require(seal85a.get('status') == 'FROZEN_COMPLETE_BLOCK85A', '8.5A status changed')
    require(seal85a.get('Block85B_may_start') is True, '8.5B not authorized')
    require(seal85a.get('formal_power_access') is False, 'Formal-power boundary already crossed')

    print('\n===== A. FROZEN PREREGISTRATION GATE =====')
    print('Block8.5A exact identities = PASS')
    print('Block8.2B canonical corpus identity = PASS')
    print('CALIBRATION / FORMAL POWER ACCESS BEFORE TRAINING = NO')

    model_mod = load_module('deepsense_lidar_encoder', MODEL_SRC)
    loader_mod = load_module('mat5_lidar_loader', LOADER_SRC)

    print('\n===== B. TRAIN SAMPLE INDEX + FROZEN BYTE GATE =====')
    records, scenario_counts = build_train_records()
    frozen_hashes = load_frozen_hash_map()
    checked, hash_workers = verify_accessed_input_hashes(records, frozen_hashes)
    print('TRAIN rows =', len(records))
    print('Scenario8 TRAIN rows =', scenario_counts['8'])
    print('Scenario9 TRAIN rows =', scenario_counts['9'])
    print('TRAIN CSV/LiDAR files SHA-verified against Block8.2B =', checked)
    print('parallel input-hash workers =', hash_workers)
    print('unit1_pwr_1 payload files opened = NO')

    atomic_json(TRAIN_INDEX, {
        'stage': 8,
        'block': '8.5B',
        'status': 'FROZEN_TRAIN_SAMPLE_INDEX',
        'row_count': len(records),
        'scenario_counts': scenario_counts,
        'samples': [{
            'sample_id': r['sample_id'],
            'scenario': r['scenario'],
            'csv_relative_path': rel(r['csv']),
            'row_number': r['row_number'],
            'raw_relative_path': rel(r['raw']),
            'scr_relative_path': rel(r['scr']),
            'beam_label': r['label'],
        } for r in records],
        'mmwave_power_payloads_opened': False,
    })

    print('\n===== C. TRAIN-ONLY LiDAR PRELOAD + NORMALIZATION =====')
    raw, scr, labels, load_workers = load_arrays(records, loader_mod)
    raw_mean, raw_std = channel_stats(raw)
    scr_mean, scr_std = channel_stats(scr)

    raw = (raw - raw_mean.astype(np.float32)) / np.maximum(raw_std, 1e-8).astype(np.float32)
    scr = (scr - scr_mean.astype(np.float32)) / np.maximum(scr_std, 1e-8).astype(np.float32)
    require(np.isfinite(raw).all() and np.isfinite(scr).all(), 'Non-finite normalized TRAIN LiDAR')

    atomic_json(NORM, {
        'stage': 8,
        'block': '8.5B',
        'status': 'FROZEN_TRAIN_ONLY_NORMALIZATION',
        'source': 'TRAIN only, pooled Scenario8+9',
        'raw_mean': [float(x) for x in raw_mean],
        'raw_std': [float(x) for x in raw_std],
        'scr_mean': [float(x) for x in scr_mean],
        'scr_std': [float(x) for x in scr_std],
        'formula': '(x-mean)/max(std,1e-8)',
        'calibration_used': False,
        'formal_used': False,
    })
    print('raw mean =', [float(x) for x in raw_mean])
    print('raw std  =', [float(x) for x in raw_std])
    print('SCR mean =', [float(x) for x in scr_mean])
    print('SCR std  =', [float(x) for x in scr_std])
    print('parallel MAT load workers =', load_workers)

    print('\n===== D. FIXED 80-EPOCH TRAINING =====')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    require(torch.cuda.is_available(), 'Frozen execution environment gate expected CUDA, but CUDA is unavailable')
    require(torch.cuda.device_count() >= 1, 'No CUDA device available')
    device = torch.device('cuda:0')
    device_name = torch.cuda.get_device_name(0)

    cpu_threads = max(1, min(os.cpu_count() or 1, 12))
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    raw_t = torch.from_numpy(np.ascontiguousarray(raw.transpose(0,2,1)))
    scr_t = torch.from_numpy(np.ascontiguousarray(scr.transpose(0,2,1)))
    y_t = torch.from_numpy(labels)
    ds = TensorDataset(raw_t, scr_t, y_t)

    g = torch.Generator()
    g.manual_seed(SEED)
    dl_workers = min(8, max(1, os.cpu_count() or 1))
    dl = DataLoader(
        ds,
        batch_size=128,
        shuffle=True,
        generator=g,
        num_workers=dl_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
    )

    model = model_mod.DualLiDARConv1D64().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(params == 18208, f'Runtime parameter count changed: {params}')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9,0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)

    epoch_log = []
    for epoch in range(1, 81):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0

        for xb_raw, xb_scr, yb in dl:
            xb_raw = xb_raw.to(device, non_blocking=True)
            xb_scr = xb_scr.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb_raw, xb_scr)
            require(torch.isfinite(logits).all().item(), f'Non-finite logits at epoch {epoch}')
            loss = criterion(logits, yb)
            require(math.isfinite(float(loss.item())), f'Non-finite loss at epoch {epoch}')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            require(math.isfinite(float(grad_norm)), f'Non-finite gradient norm at epoch {epoch}')
            optimizer.step()

            bs = int(yb.shape[0])
            loss_sum += float(loss.item()) * bs
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            seen += bs

        require(seen == 7030, f'Epoch {epoch} saw {seen} samples instead of 7030')
        mean_loss = loss_sum / seen
        train_acc = correct / seen
        lr_before_step = float(optimizer.param_groups[0]['lr'])
        scheduler.step()

        rec = {
            'epoch': epoch,
            'train_cross_entropy': mean_loss,
            'train_top1_accuracy': train_acc,
            'learning_rate': lr_before_step,
        }
        epoch_log.append(rec)

        if epoch == 1 or epoch % 10 == 0 or epoch == 80:
            print(
                f'epoch={epoch:03d}/080 '
                f'train_ce={mean_loss:.8f} train_top1={train_acc:.6f} lr={lr_before_step:.8g}'
            )

    require(len(epoch_log) == 80, 'Training log is not exactly 80 epochs')
    torch.cuda.synchronize(device)

    atomic_json(LOG, {
        'stage': 8,
        'block': '8.5B',
        'status': 'FROZEN_FIXED_80_EPOCH_TRAINING_LOG',
        'device': 'cuda:0',
        'device_name': device_name,
        'cpu_threads': cpu_threads,
        'dataloader_workers': dl_workers,
        'seed': SEED,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'epochs': epoch_log,
        'checkpoint_selection': 'fixed final epoch 80 only',
        'validation_model_selection': False,
        'calibration_data_used': False,
        'formal_data_used': False,
        'measured_power_payloads_opened': False,
    })

    print('\n===== E. FINAL-EPOCH CHECKPOINT FREEZE =====')
    checkpoint_obj = {
        'stage': 8,
        'block': '8.5B',
        'model_name': 'DualLiDARConv1D64',
        'epoch': 80,
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'normalization': {
            'raw_mean': torch.tensor(raw_mean, dtype=torch.float64),
            'raw_std': torch.tensor(raw_std, dtype=torch.float64),
            'scr_mean': torch.tensor(scr_mean, dtype=torch.float64),
            'scr_std': torch.tensor(scr_std, dtype=torch.float64),
        },
        'training': {
            'seed': SEED,
            'optimizer': 'AdamW',
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'epochs': 80,
            'batch_size': 128,
            'scheduler': 'CosineAnnealingLR(T_max=80,eta_min=1e-5)',
            'parameter_count': params,
            'device': 'cuda:0',
            'device_name': device_name,
        },
        'upstream': {
            'block85a_seal_sha256': EXPECTED[str(B85A_SEAL)],
            'block82b_files_manifest_sha256': EXPECTED[str(B82B_FILES)],
        },
    }
    atomic_torch_save(CHECKPOINT, checkpoint_obj)
    checkpoint_sha = sha256_file(CHECKPOINT)
    print('checkpoint epoch = 80')
    print('checkpoint parameter count =', params)
    print('checkpoint SHA256 =', checkpoint_sha)

    manifest = {
        'stage': 8,
        'block': '8.5B',
        'status': 'FROZEN_COMPLETE_BLOCK85B',
        'Block85A_seal_sha256': EXPECTED[str(B85A_SEAL)],
        'Block82B_files_manifest_sha256': EXPECTED[str(B82B_FILES)],
        'model_source_sha256': EXPECTED[str(MODEL_SRC)],
        'loader_source_sha256': EXPECTED[str(LOADER_SRC)],
        'runner': {'path': str(SCRIPT), 'sha256': sha256_file(SCRIPT)},
        'train_index': {'path': str(TRAIN_INDEX), 'sha256': sha256_file(TRAIN_INDEX)},
        'normalization': {'path': str(NORM), 'sha256': sha256_file(NORM)},
        'training_log': {'path': str(LOG), 'sha256': sha256_file(LOG)},
        'checkpoint': {'path': str(CHECKPOINT), 'sha256': checkpoint_sha},
        'train_rows': 7030,
        'epochs_completed': 80,
        'calibration_data_accessed': False,
        'formal_data_accessed': False,
        'measured_power_payloads_opened': False,
    }
    atomic_json(MANIFEST, manifest)

    final = epoch_log[-1]
    report = {
        'stage': 8,
        'block': '8.5B',
        'status': 'PASS_BLOCK85B_FIXED_RECIPE_ENCODER_TRAINING',
        'train_rows': 7030,
        'scenario8_train_rows': 2831,
        'scenario9_train_rows': 4199,
        'epochs_completed': 80,
        'parameter_count': params,
        'device': 'cuda:0',
        'device_name': device_name,
        'final_train_cross_entropy': final['train_cross_entropy'],
        'final_train_top1_accuracy': final['train_top1_accuracy'],
        'checkpoint_epoch': 80,
        'checkpoint_sha256': checkpoint_sha,
        'hyperparameter_search': False,
        'validation_model_selection': False,
        'calibration_data_accessed': False,
        'formal_data_accessed': False,
        'measured_power_payloads_opened': False,
        'Block86_authorized': True,
        'normalization_sha256': sha256_file(NORM),
        'train_index_sha256': sha256_file(TRAIN_INDEX),
        'training_log_sha256': sha256_file(LOG),
        'manifest_sha256': sha256_file(MANIFEST),
        'runner_sha256': sha256_file(SCRIPT),
    }
    atomic_json(REPORT, report)

    seal = {
        'stage': 8,
        'block': '8.5B',
        'status': 'FROZEN_COMPLETE_BLOCK85B',
        'Block85A_seal_sha256': EXPECTED[str(B85A_SEAL)],
        'Block82B_seal_sha256': EXPECTED[str(B82B_SEAL)],
        'model_source_sha256': EXPECTED[str(MODEL_SRC)],
        'loader_source_sha256': EXPECTED[str(LOADER_SRC)],
        'runner_sha256': sha256_file(SCRIPT),
        'train_index_sha256': sha256_file(TRAIN_INDEX),
        'normalization_sha256': sha256_file(NORM),
        'training_log_sha256': sha256_file(LOG),
        'checkpoint_sha256': checkpoint_sha,
        'manifest_sha256': sha256_file(MANIFEST),
        'report_sha256': sha256_file(REPORT),
        'training_completed': True,
        'calibration_started': False,
        'formal_power_access': False,
        'Block86_may_start': True,
    }
    atomic_json(SEAL, seal)

    for raw, expected in EXPECTED.items():
        require(sha256_file(Path(raw)) == expected, f'Frozen upstream modified during 8.5B: {raw}')

    print('\n' + '='*88)
    print('BLOCK 8.5B = FULLY VERIFIED / FROZEN')
    print('TRAIN ROWS = 7,030')
    print('SCENARIO8/9 TRAIN = 2,831 / 4,199')
    print('TRAIN-ONLY NORMALIZATION = PASS')
    print('MODEL = DualLiDARConv1D64 / 18,208 PARAMETERS')
    print('DEVICE = CUDA:0 / ' + device_name)
    print('FIXED TRAINING EPOCHS = 80/80 COMPLETE')
    print('CHECKPOINT SELECTION = FIXED FINAL EPOCH 80')
    print('HYPERPARAMETER SEARCH / EARLY STOPPING = NO')
    print('CALIBRATION DATA ACCESSED = NO')
    print('FORMAL DATA ACCESSED = NO')
    print('MMWAVE POWER PAYLOADS OPENED = NO')
    print('BLOCK 8.6 MAY START = YES')
    print('final TRAIN CE =', final['train_cross_entropy'])
    print('final TRAIN Top-1 =', final['train_top1_accuracy'])
    print('8.5B train-index SHA256 =', sha256_file(TRAIN_INDEX))
    print('8.5B normalization SHA256 =', sha256_file(NORM))
    print('8.5B training-log SHA256 =', sha256_file(LOG))
    print('8.5B checkpoint SHA256 =', checkpoint_sha)
    print('8.5B manifest SHA256 =', sha256_file(MANIFEST))
    print('8.5B report SHA256 =', sha256_file(REPORT))
    print('8.5B seal SHA256 =', sha256_file(SEAL))
    print('8.5B runner SHA256 =', sha256_file(SCRIPT))
    print('STATUS = FROZEN_COMPLETE_BLOCK85B')
    print('='*88)
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print('\n' + '!'*88)
        print('BLOCK 8.5B FAIL-CLOSED')
        print(f'{type(exc).__name__}: {exc}')
        print('DO NOT DELETE/MODIFY FROZEN TRAINING INPUTS OR PARTIAL OUTPUTS.')
        print('DO NOT START CALIBRATION OR FORMAL EVALUATION.')
        print('!'*88)
        raise
