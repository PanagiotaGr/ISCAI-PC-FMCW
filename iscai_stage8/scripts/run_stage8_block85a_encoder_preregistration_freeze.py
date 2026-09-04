#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path("/home/agni/waymo")
S8 = ROOT / "iscai_stage8"

SCRIPT = S8 / "scripts/run_stage8_block85a_encoder_preregistration_freeze.py"
CONTRACT = S8 / "configs/stage8_block85a_encoder_preregistration_contract.json"
MODEL = S8 / "src/deepsense_lidar_encoder.py"
LOADER = S8 / "src/mat5_lidar_loader.py"

B84_SEAL = S8 / "artifacts/block84_beam_policy_interface/stage8_block84_beam_policy_interface_seal.json"
B84R_SEAL = S8 / "artifacts/block84r_reporting_correction/stage8_block84r_reporting_correction_seal.json"
B82B_SEAL = S8 / "artifacts/block82b_canonical_corpus_schema_split/stage8_block82b_canonical_corpus_schema_split_seal.json"
B82B_FILES = S8 / "artifacts/block82b_canonical_corpus_schema_split/canonical_extracted_file_manifest.json"

OUT = S8 / "artifacts/block85a_encoder_preregistration"
ARCH = OUT / "frozen_encoder_architecture.json"
TRAIN = OUT / "frozen_training_recipe.json"
MANIFEST = OUT / "stage8_block85a_manifest.json"
REPORT = S8 / "reports/stage8_block85a_encoder_preregistration_report.json"
SEAL = OUT / "stage8_block85a_encoder_preregistration_seal.json"

EXPECTED_CONTRACT = "b0bab29b655aca9e8f29b77fba73d1aeaa1dc4ac444269d254fc7d3b0d4fc45d"
EXPECTED_MODEL = "adb943f12490612a6b92ffa3e70f69b96b1002fa77e4e769f5ea26ed6d25c417"
EXPECTED_LOADER = "b4b3d9e09a20a5ee790a4ad7fd22f53dad1d6f4b953e70f824ade80f9b86299c"
EXPECTED_UPSTREAM = {
    str(B84_SEAL): "c5dcbf5b1bf8d7965fd32bf794f095b4495d30d98e35bddef22ce3abffe554e2",
    str(B84R_SEAL): "87f8776d2a5bc3bfb2e3f265e1c791aebd20597eb9da91ba591db71953d208a5",
    str(B82B_SEAL): "c84eb7ef45d16a1feb3ef068bdaf498e5e4298514f3c4807ffe10db9b1112ba3",
    str(B82B_FILES): "79dc92fb6a90acd36c4598de436517ebac6c7e7904334dd76a520e1c90a0b895",
}

class FailClosed(RuntimeError): pass

def require(c,m):
    if not c: raise FailClosed(m)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    require(not path.exists(),f"Refusing overwrite: {path}")
    data=(json.dumps(obj,sort_keys=True,separators=(",",":"))+"\n").encode()
    fd,tmp=tempfile.mkstemp(prefix=path.name+".tmp.",dir=str(path.parent))
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def literal_assignment(path, name):
    tree=ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for t in node.targets:
                if isinstance(t,ast.Name) and t.id==name:
                    return ast.literal_eval(node.value)
    raise FailClosed(f"literal assignment {name} not found in {path}")

def param_count_from_spec(s):
    # Two independent Conv1d branches + LayerNorm + Linear + Linear.
    c1 = s["conv1"]["out_channels"]*s["conv1"]["in_channels"]*s["conv1"]["kernel_size"] + s["conv1"]["out_channels"]
    c2 = s["conv2"]["out_channels"]*s["conv2"]["in_channels"]*s["conv2"]["kernel_size"] + s["conv2"]["out_channels"]
    branches = 2*(c1+c2)
    ln = 2*s["layer_norm_features"]
    fc1 = s["fusion_features"]*s["hidden_features"] + s["hidden_features"]
    fc2 = s["hidden_features"]*s["output_logits"] + s["output_logits"]
    return branches+ln+fc1+fc2

def main():
    print("="*88)
    print("STAGE 8 — BLOCK 8.5A")
    print("DEEPSENSE LIGHTWEIGHT ENCODER / INPUT / TRAINING PREREGISTRATION")
    print("PREREGISTRATION ONLY — NO TRAINING")
    print("="*88)

    require(Path(__file__).resolve()==SCRIPT.resolve(),"runner path mismatch")
    require(CONTRACT.is_file() and sha(CONTRACT)==EXPECTED_CONTRACT,"contract SHA mismatch")
    require(MODEL.is_file() and sha(MODEL)==EXPECTED_MODEL,"model SHA mismatch")
    require(LOADER.is_file() and sha(LOADER)==EXPECTED_LOADER,"loader SHA mismatch")
    for raw,expected in EXPECTED_UPSTREAM.items():
        p=Path(raw); require(p.is_file(),f"missing upstream: {p}"); require(sha(p)==expected,f"upstream SHA changed: {p}")
    for p in (ARCH,TRAIN,MANIFEST,REPORT,SEAL):
        require(not p.exists(),f"8.5A output exists: {p}")

    print("\n===== A. UPSTREAM PRE-TRAINING GATE =====")
    b84r=json.loads(B84R_SEAL.read_text())
    require(b84r.get("status")=="FROZEN_COMPLETE_BLOCK84R","8.4R status changed")
    require(b84r.get("Block85_may_start") is True,"8.5 not authorized")
    b82=json.loads(B82B_SEAL.read_text())
    require(b82.get("formal_power_vectors_decoded") is False,"formal power boundary crossed")
    print("Block8.4 + 8.4R + 8.2B exact identities = PASS")
    print("formal measured-power vectors decoded = NO")

    print("\n===== B. ACTUAL INPUT REPRESENTATION FREEZE =====")
    c=json.loads(CONTRACT.read_text())
    rep=c["representation"]
    require(rep["raw_shape_required"]==[460,2],"raw shape changed")
    require(rep["scr_shape_required"]==[216,2],"SCR shape changed")
    require(rep["use_raw_lidar"] and rep["use_scr_lidar"],"dual input changed")
    require(rep["measured_power_feature"] is False,"measured power leaked into features")
    print("raw LiDAR = 460x2 double -> float32 = FROZEN")
    print("SCR LiDAR = 216x2 double -> float32 = FROZEN")
    print("raw + SCR dual branch = FROZEN")
    print("scenario ID / measured power as features = NO")

    print("\n===== C. ARCHITECTURE SOURCE / PARAMETER GATE =====")
    spec=literal_assignment(MODEL,"ARCH_SPEC")
    calc=param_count_from_spec(spec)
    require(calc==18208,f"computed parameter count {calc} != 18208")
    require(spec["expected_parameter_count"]==18208,"source expected parameter count changed")
    require(spec["output_logits"]==64,"output logits changed")
    print("DualLiDARConv1D64 source = PASS")
    print("calculated trainable parameter count = 18,208 PASS")
    print("output logits = 64 PASS")

    print("\n===== D. TRAINING RECIPE FREEZE =====")
    tr=c["training"]
    require(tr["train_rows_expected"]==7030,"train row count changed")
    require(tr["epochs"]==80 and tr["batch_size"]==128,"epoch/batch recipe changed")
    require(tr["learning_rate"]==0.001 and tr["weight_decay"]==0.0001,"optimizer recipe changed")
    require(tr["early_stopping"] is False and tr["validation_for_model_selection"] is False,"validation/model selection boundary changed")
    require(tr["checkpoint_selection"]=="fixed final epoch 80 only","checkpoint selection changed")
    require(tr["hyperparameter_search"] is False,"hyperparameter search allowed")
    print("TRAIN split only = 7,030 rows")
    print("AdamW lr=1e-3 wd=1e-4 / 80 epochs / batch=128 = FROZEN")
    print("CosineAnnealing eta_min=1e-5 = FROZEN")
    print("early stopping / validation model selection = NO")
    print("hyperparameter search = NO")
    print("final checkpoint = fixed epoch 80")

    arch_doc={
        "stage":8,"block":"8.5A","status":"FROZEN_ENCODER_ARCHITECTURE",
        "architecture":spec,
        "calculated_parameter_count":calc,
        "model_sha256":EXPECTED_MODEL,
        "loader_sha256":EXPECTED_LOADER,
        "input_shapes":{"raw":[460,2],"scr":[216,2]},
    }
    atomic_json(ARCH,arch_doc)

    atomic_json(TRAIN,{
        "stage":8,"block":"8.5A","status":"FROZEN_TRAINING_RECIPE",
        "normalization":c["normalization"],
        "training":c["training"],
        "label_access":c["training_label_access"],
        "fallback":c["inference_fail_closed_fallback"],
        "reproducibility":c["reproducibility"],
    })

    atomic_json(MANIFEST,{
        "stage":8,"block":"8.5A","status":"FROZEN_COMPLETE_BLOCK85A",
        "contract":{"path":str(CONTRACT),"sha256":EXPECTED_CONTRACT},
        "model":{"path":str(MODEL),"sha256":EXPECTED_MODEL},
        "loader":{"path":str(LOADER),"sha256":EXPECTED_LOADER},
        "runner":{"path":str(SCRIPT),"sha256":sha(SCRIPT)},
        "architecture":{"path":str(ARCH),"sha256":sha(ARCH)},
        "training_recipe":{"path":str(TRAIN),"sha256":sha(TRAIN)},
        "training_started":False,
        "calibration_started":False,
        "formal_power_access":False,
    })

    atomic_json(REPORT,{
        "stage":8,"block":"8.5A","status":"PASS_BLOCK85A_ENCODER_PREREGISTRATION",
        "raw_input_shape":[460,2],
        "scr_input_shape":[216,2],
        "dual_branch":True,
        "parameter_count":calc,
        "output_logits":64,
        "train_rows":7030,
        "fixed_epochs":80,
        "hyperparameter_search":False,
        "validation_model_selection":False,
        "formal_power_access":False,
        "training_started":False,
        "Block85B_authorized":True,
        "architecture_sha256":sha(ARCH),
        "training_recipe_sha256":sha(TRAIN),
        "manifest_sha256":sha(MANIFEST),
        "runner_sha256":sha(SCRIPT),
    })

    atomic_json(SEAL,{
        "stage":8,"block":"8.5A","status":"FROZEN_COMPLETE_BLOCK85A",
        "Block84R_seal_sha256":EXPECTED_UPSTREAM[str(B84R_SEAL)],
        "Block82B_seal_sha256":EXPECTED_UPSTREAM[str(B82B_SEAL)],
        "contract_sha256":EXPECTED_CONTRACT,
        "model_sha256":EXPECTED_MODEL,
        "loader_sha256":EXPECTED_LOADER,
        "runner_sha256":sha(SCRIPT),
        "architecture_sha256":sha(ARCH),
        "training_recipe_sha256":sha(TRAIN),
        "manifest_sha256":sha(MANIFEST),
        "report_sha256":sha(REPORT),
        "training_started":False,
        "formal_power_access":False,
        "Block85B_may_start":True,
    })

    for raw,expected in EXPECTED_UPSTREAM.items():
        require(sha(Path(raw))==expected,f"upstream modified during 8.5A: {raw}")

    print("\n"+"="*88)
    print("BLOCK 8.5A = FULLY VERIFIED / FROZEN")
    print("ACTUAL RAW LiDAR INPUT = 460x2")
    print("ACTUAL SCR LiDAR INPUT = 216x2")
    print("DUAL RAW+SCR ENCODER = FROZEN")
    print("MODEL = DualLiDARConv1D64")
    print("TRAINABLE PARAMETERS = 18,208")
    print("OUTPUT = 64 BEAM LOGITS")
    print("TRAIN SPLIT ONLY = 7,030 ROWS")
    print("FIXED TRAINING = 80 EPOCHS / BATCH 128 / AdamW")
    print("EARLY STOPPING / HYPERPARAMETER SEARCH = NO")
    print("CALIBRATION/FORMAL DATA FOR MODEL SELECTION = NO")
    print("MEASURED POWER TRAINING FEATURE/TARGET = NO")
    print("TRAINING STARTED = NO")
    print("BLOCK 8.5B MAY START = YES")
    print("8.5A contract SHA256 =",EXPECTED_CONTRACT)
    print("8.5A model SHA256    =",EXPECTED_MODEL)
    print("8.5A loader SHA256   =",EXPECTED_LOADER)
    print("8.5A arch SHA256     =",sha(ARCH))
    print("8.5A recipe SHA256   =",sha(TRAIN))
    print("8.5A manifest SHA256 =",sha(MANIFEST))
    print("8.5A report SHA256   =",sha(REPORT))
    print("8.5A seal SHA256     =",sha(SEAL))
    print("8.5A runner SHA256   =",sha(SCRIPT))
    print("STATUS = FROZEN_COMPLETE_BLOCK85A")
    print("="*88)
    return 0

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("\n"+"!"*88)
        print("BLOCK 8.5A FAIL-CLOSED")
        print(f"{type(e).__name__}: {e}")
        print("DO NOT START TRAINING/CALIBRATION/FORMAL EVALUATION.")
        print("!"*88)
        raise
