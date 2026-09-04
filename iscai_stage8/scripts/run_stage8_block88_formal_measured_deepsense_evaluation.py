#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/agni/waymo")
S8 = ROOT / "iscai_stage8"
DATA_ROOT = S8 / "data/block82b_deepsense_primary"

SCRIPT = S8 / "scripts/run_stage8_block88_formal_measured_deepsense_evaluation.py"
CONTRACT = S8 / "configs/stage8_block88_formal_measured_deepsense_evaluation_contract.json"

MODEL_SRC = S8 / "src/deepsense_lidar_encoder.py"
LOADER_SRC = S8 / "src/mat5_lidar_loader.py"
POLICY_SRC = S8 / "src/deepsense_beam_policy.py"
RUNTIME_SRC = S8 / "src/deepsense_policy_runtime.py"
EVALUATOR_SRC = S8 / "src/deepsense_mmwave_evaluator.py"

B85B = S8 / "artifacts/block85b_encoder_training_cpu_repair"
CHECKPOINT = B85B / "duallidarconv1d64_epoch080.pt"
NORM = B85B / "train_only_normalization.json"
SEAL85B = B85B / "stage8_block85b_encoder_training_cpu_repair_seal.json"

B86 = S8 / "artifacts/block86_probability_calibration_final"
CAL = B86 / "frozen_temperature_calibrator.json"
SEAL86 = B86 / "stage8_block86_probability_calibration_seal.json"

B87 = S8 / "artifacts/block87_preformal_policy_fallback_replay"
SEAL87 = B87 / "stage8_block87_preformal_policy_fallback_replay_seal.json"
REPLAY87 = B87 / "calibration_policy_replay_summary.json"

B84_SEAL = S8 / "artifacts/block84_beam_policy_interface/stage8_block84_beam_policy_interface_seal.json"
B83_SEAL = S8 / "artifacts/block83_measured_mmwave_evaluator_contract/stage8_block83_measured_mmwave_evaluator_contract_seal.json"
B82B_SEAL = S8 / "artifacts/block82b_canonical_corpus_schema_split/stage8_block82b_canonical_corpus_schema_split_seal.json"
B82B_FILES = S8 / "artifacts/block82b_canonical_corpus_schema_split/canonical_extracted_file_manifest.json"
B82B_SPLITS = S8 / "artifacts/block82b_canonical_corpus_schema_split/frozen_split_roles_and_disjointness.json"

OUT = S8 / "artifacts/block88_formal_measured_deepsense_evaluation"
TRAIN_GATE = OUT / "train_power_encoding_preformal_gate.json"
CONTROLLER_JSONL = OUT / "formal_controller_decisions.jsonl"
CONTROLLER_SEAL = OUT / "formal_controller_preoutcome_seal.json"
FORMAL_ACCESS_AUTH = OUT / "formal_power_access_authorization.json"
EVAL_JSONL = OUT / "formal_measured_power_records.jsonl"
SUMMARY = OUT / "formal_metrics_summary.json"
MANIFEST = OUT / "stage8_block88_manifest.json"
REPORT = S8 / "reports/stage8_block88_formal_measured_deepsense_evaluation_report.json"
SEAL = OUT / "stage8_block88_formal_measured_deepsense_evaluation_seal.json"

EXPECTED_CONTRACT = "8af6c990e8e7c0b5863709ae4a2a05e1abfe6a760be5582bb18fea84b60aaf99"
EXPECTED = {
 str(MODEL_SRC): "adb943f12490612a6b92ffa3e70f69b96b1002fa77e4e769f5ea26ed6d25c417",
 str(LOADER_SRC): "b4b3d9e09a20a5ee790a4ad7fd22f53dad1d6f4b953e70f824ade80f9b86299c",
 str(POLICY_SRC): "5b7e6c3c4ff7eca4d1db86df90edfa19f0bbfc551255779833485df074d87cb8",
 str(RUNTIME_SRC): "5171a69ebe491027088934075c2fe2c8c74ef4a5e10823f2ff636237056ff3f3",
 str(EVALUATOR_SRC): "7e2c4ffe589e3d6f58e5b6daf1d8fb9e6e44b2eef074bdf1624a3334f8cbe650",
 str(CHECKPOINT): "a31e411a4c729b9ff99c20295f849112417420d301296d3f6e6f73d1a1ac99b5",
 str(NORM): "8e12e7b5bccf57c31075569a575507cd95e4b1710f5bb7327cbdd203c36575fd",
 str(SEAL85B): "ee9e5a8d7f4d630e5334f5f5089484e6016f35e38316346edb5a3194af74ec3b",
 str(CAL): "ddac88b72b6fc08953ca5dcbf899c5bacaa646a252986147389d079bcaea8d82",
 str(SEAL86): "6bd11ccb198ee67f6aa15895951e69a663d5474e1630c27643e7d7f8343ed444",
 str(SEAL87): "b1160dd3f4b9736b3f661baff628dd3242f3887a3642aee3bba2e146d7c207ea",
 str(REPLAY87): "c53bc9cc7ca1474ce60ec1cfc0f29243204a13bf2a5b6aed685e1dbb14929953",
 str(B84_SEAL): "c5dcbf5b1bf8d7965fd32bf794f095b4495d30d98e35bddef22ce3abffe554e2",
 str(B83_SEAL): "056371994fa598c6c58622b0161febde17f0e75231b4ba6e1d596d93db4e575f",
 str(B82B_SEAL): "c84eb7ef45d16a1feb3ef068bdaf498e5e4298514f3c4807ffe10db9b1112ba3",
 str(B82B_FILES): "79dc92fb6a90acd36c4598de436517ebac6c7e7904334dd76a520e1c90a0b895",
 str(B82B_SPLITS): "67e7624cdb633bf83c0fedab19b6d4c5b41d70ccf0d69c9875873c24b3fe20ce",
}

TRAIN_SPECS = [
 (8, DATA_ROOT / "LiDAR/Scenario8/development_dataset/scenario8_dev_train.csv"),
 (9, DATA_ROOT / "LiDAR/Scenario9/development_dataset/scenario9_dev_train.csv"),
]
FORMAL_SPECS = [
 (8, DATA_ROOT / "LiDAR/Scenario8/development_dataset/scenario8_dev_test.csv"),
 (9, DATA_ROOT / "LiDAR/Scenario9/development_dataset/scenario9_dev_test.csv"),
]
POLICIES = [
 "adaptive_topk_q090","adaptive_topk_q095","adaptive_topk_q0975","adaptive_topk_q099",
 "fixed_top1","fixed_top3","fixed_top5","exhaustive_64",
]
ORACLE = "oracle_gt_best_beam_k1"
ALL_METHODS = POLICIES + [ORACLE]

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
    data=(json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()
    fd,tmp=tempfile.mkstemp(prefix=path.name+".tmp.",dir=str(path.parent))
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
        dfd=os.open(path.parent,os.O_DIRECTORY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def atomic_jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    require(not path.exists(),f"Refusing overwrite: {path}")
    fd,tmp=tempfile.mkstemp(prefix=path.name+".tmp.",dir=str(path.parent))
    try:
        with os.fdopen(fd,"wb") as f:
            for row in rows:
                f.write((json.dumps(row,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode())
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
        dfd=os.open(path.parent,os.O_DIRECTORY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def normalize_ref(csv_path, raw_ref):
    s=(raw_ref or "").strip().replace("\\","/")
    while s.startswith("./"): s=s[2:]
    p=(DATA_ROOT/s) if s.startswith("LiDAR/") else (csv_path.parent/s)
    return p.resolve(), s

def load_frozen_hash_map():
    d=json.loads(B82B_FILES.read_text())
    require(d.get("status")=="FROZEN_CANONICAL_EXTRACTED_FILE_MANIFEST","8.2B file manifest status changed")
    return {x["relative_path"]:x["sha256"] for x in d["files"]}

def require_frozen_hash(path, fmap):
    rel=str(path.relative_to(DATA_ROOT))
    expected=fmap.get(rel)
    require(expected is not None,f"Path absent from frozen 8.2B manifest: {rel}")
    require(sha(path)==expected,f"Frozen corpus SHA mismatch: {rel}")
    return rel

def parse_power(path):
    toks=path.read_text(encoding="utf-8",errors="strict").replace(","," ").split()
    vals=[float(x) for x in toks]
    require(len(vals)==64,f"Expected 64 measured powers in {path}, got {len(vals)}")
    require(all(math.isfinite(x) for x in vals),f"Non-finite measured power in {path}")
    return vals

def read_train_power_rows():
    out=[]; counts={}
    for sc,csv_path in TRAIN_SPECS:
        n=0
        with csv_path.open(newline="",encoding="utf-8-sig") as f:
            r=csv.DictReader(f)
            require(r.fieldnames==['index','unit1_lidar_1','unit1_lidar_SCR_1','unit1_pwr_1','beam_index_1'],
                    f"Unexpected TRAIN headers: {r.fieldnames}")
            for row in r:
                p,_=normalize_ref(csv_path,row["unit1_pwr_1"])
                y=int(float(row["beam_index_1"]))
                require(0<=y<64,"TRAIN label outside 0..63")
                out.append((sc,p,y))
                n+=1
        counts[str(sc)]=n
    require(counts=={"8":2831,"9":4199} and len(out)==7030,f"TRAIN counts changed: {counts}")
    return out,counts

def controller_formal_rows(policy_mod):
    rows=[]; counts={}
    for sc,csv_path in FORMAL_SPECS:
        csv_rel=str(csv_path.relative_to(DATA_ROOT))
        n=0
        with csv_path.open(newline="",encoding="utf-8-sig") as f:
            r=csv.DictReader(f)
            require(r.fieldnames==['index','unit1_lidar_1','unit1_lidar_SCR_1','unit1_pwr_1','beam_index_1'],
                    f"Unexpected FORMAL headers: {r.fieldnames}")
            for row in r:
                official_index=int(float(row["index"]))
                raw,raw_ref=normalize_ref(csv_path,row["unit1_lidar_1"])
                scr,scr_ref=normalize_ref(csv_path,row["unit1_lidar_SCR_1"])
                power,power_ref=normalize_ref(csv_path,row["unit1_pwr_1"])
                require(raw.is_file() and scr.is_file() and power.is_file(),"Missing formal referenced file")
                sid=policy_mod.stable_sample_id(
                    sc,"FORMAL_EVALUATION",csv_rel,official_index,raw_ref,power_ref
                )
                rows.append({
                    "sample_id":sid,"scenario":sc,"csv_path":csv_path,"csv_relative_path":csv_rel,
                    "official_csv_index":official_index,
                    "raw":raw,"raw_ref":raw_ref,"scr":scr,"scr_ref":scr_ref,
                    "power":power,"power_ref":power_ref,
                })
                n+=1
        counts[str(sc)]=n
    require(counts=={"8":437,"9":593} and len(rows)==1030,f"FORMAL counts changed: {counts}")
    require(len({x["sample_id"] for x in rows})==1030,"Duplicate formal sample IDs")
    return rows,counts

def evaluator_formal_labels():
    labels={}
    for sc,csv_path in FORMAL_SPECS:
        csv_rel=str(csv_path.relative_to(DATA_ROOT))
        with csv_path.open(newline="",encoding="utf-8-sig") as f:
            r=csv.DictReader(f)
            for row in r:
                official_index=int(float(row["index"]))
                raw,raw_ref=normalize_ref(csv_path,row["unit1_lidar_1"])
                _,power_ref=normalize_ref(csv_path,row["unit1_pwr_1"])
                sid=policy_mod.stable_sample_id(
                    sc,"FORMAL_EVALUATION",csv_rel,official_index,raw_ref,power_ref
                )
                y=int(float(row["beam_index_1"]))
                require(0<=y<64,f"Formal label outside 0..63 for {sid}")
                labels[sid]=y
    require(len(labels)==1030,"Formal label map count changed")
    return labels

def load_formal_lidar(rows,loader,fmap):
    # Verify only formal CSV/LiDAR bytes here; formal power bytes remain unopened.
    for _,csv_path in FORMAL_SPECS:
        require_frozen_hash(csv_path,fmap)
    paths=sorted({r["raw"] for r in rows} | {r["scr"] for r in rows})
    workers=max(1,min(os.cpu_count() or 1,12))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda p: require_frozen_hash(p,fmap),paths))

    n=len(rows)
    raw=np.empty((n,460,2),np.float32); scr=np.empty((n,216,2),np.float32)
    ok=np.ones(n,dtype=bool); errors=[None]*n

    def one(item):
        i,r=item
        try:
            a=loader.load_mat5_double_matrix(r["raw"],expected_name="data",expected_shape=(460,2))
            b=loader.load_mat5_double_matrix(r["scr"],expected_name="data",expected_shape=(216,2))
            return i,a.astype(np.float32,copy=False),b.astype(np.float32,copy=False),None
        except Exception as e:
            return i,None,None,f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i,a,b,err in ex.map(one,enumerate(rows),chunksize=16):
            if err is None:
                raw[i]=a; scr[i]=b
            else:
                ok[i]=False; errors[i]=err
                raw[i].fill(0.0); scr[i].fill(0.0)
    return raw,scr,ok,errors,workers

def mean(vals):
    return float(sum(vals)/len(vals)) if vals else float("nan")

def aggregate_records(records, method, scenario=None):
    subset=[r for r in records if (scenario is None or r["scenario"]==scenario)]
    n=len(subset)
    require(n>0,"Empty aggregation subset")
    ev=[r["methods"][method] for r in subset]
    return {
        "n":n,
        "coverage":mean([x["coverage"] for x in ev]),
        "mean_K":mean([x["K"] for x in ev]),
        "mean_probing_overhead":mean([x["probing_overhead"] for x in ev]),
        "mean_overhead_reduction_vs_exhaustive":mean([x["overhead_reduction_vs_exhaustive"] for x in ev]),
        "mean_measured_power_gap_db":mean([x["measured_power_gap_db"] for x in ev]),
        "mean_measured_power_loss_db":mean([x["measured_power_loss_db"] for x in ev]),
        "outage_rate":{
            "1dB":mean([int(x["outage"]["1dB"]) for x in ev]),
            "3dB":mean([int(x["outage"]["3dB"]) for x in ev]),
            "6dB":mean([int(x["outage"]["6dB"]) for x in ev]),
        },
        "mean_normalized_se_gap_bps_per_hz":{
            "0dB":mean([x["normalized_spectral_efficiency"]["0dB"]["gap_bps_per_hz"] for x in ev]),
            "10dB":mean([x["normalized_spectral_efficiency"]["10dB"]["gap_bps_per_hz"] for x in ev]),
            "20dB":mean([x["normalized_spectral_efficiency"]["20dB"]["gap_bps_per_hz"] for x in ev]),
        },
    }

def main():
    global policy_mod
    print("="*96)
    print("STAGE 8 — BLOCK 8.8")
    print("FORMAL MEASURED DEEPSENSE EVALUATION")
    print("FIRST FORMAL POWER OPENING — FROZEN CONTROLLER BEFORE EVALUATOR ACCESS")
    print("="*96)

    require(Path(__file__).resolve()==SCRIPT.resolve(),"Runner path mismatch")
    require(CONTRACT.is_file() and sha(CONTRACT)==EXPECTED_CONTRACT,"8.8 contract SHA mismatch")
    for raw,expected in EXPECTED.items():
        p=Path(raw); require(p.is_file(),f"Missing frozen input: {p}"); require(sha(p)==expected,f"Frozen SHA changed: {p}")
    for p in (TRAIN_GATE,CONTROLLER_JSONL,CONTROLLER_SEAL,FORMAL_ACCESS_AUTH,EVAL_JSONL,SUMMARY,MANIFEST,REPORT,SEAL):
        require(not p.exists(),f"8.8 output already exists: {p}")

    s87=json.loads(SEAL87.read_text())
    require(s87.get("status")=="FROZEN_COMPLETE_BLOCK87","8.7 status changed")
    require(s87.get("Block88_may_start") is True,"8.8 not authorized")
    require(s87.get("formal_data_access") is False and s87.get("formal_power_access") is False,
            "8.7 formal boundary changed")

    model_mod=load_module("ds_model",MODEL_SRC)
    loader_mod=load_module("mat5",LOADER_SRC)
    policy_mod=load_module("beam_policy",POLICY_SRC)
    runtime_mod=load_module("policy_runtime",RUNTIME_SRC)
    evaluator=load_module("mmwave_eval",EVALUATOR_SRC)
    fmap=load_frozen_hash_map()

    print("\n===== A. PRE-FORMAL TRAIN POWER ENCODING / EVALUATOR GATE =====")
    train_rows,train_counts=read_train_power_rows()

    def train_one(item):
        sc,p,y=item
        require_frozen_hash(p,fmap)
        vals=parse_power(p)
        require(all(x>0.0 for x in vals),f"Frozen evaluator requires positive linear powers; TRAIN mismatch: {p}")
        require(evaluator.label_consistent_with_power(y,vals),
                f"TRAIN label/power-max mismatch under frozen evaluator: {p}")
        return 1

    workers=max(1,min(os.cpu_count() or 1,12))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        passed=sum(ex.map(train_one,train_rows,chunksize=32))
    require(passed==7030,"TRAIN power preformal gate incomplete")
    atomic_json(TRAIN_GATE,{
        "stage":8,"block":"8.8","status":"PASS_PREFORMAL_TRAIN_POWER_ENCODING_GATE",
        "rows_checked":7030,"scenario_counts":train_counts,
        "power_vector_length":64,"all_finite_and_strictly_positive":True,
        "all_labels_consistent_with_measured_power_max":True,
        "evaluator_sha256":EXPECTED[str(EVALUATOR_SRC)],
        "formal_power_numeric_values_accessed":False,
        "post_outcome_tuning":False,
    })
    print("TRAIN measured-power vectors checked = 7,030/7,030 PASS")
    print("64 values / finite / strictly positive = PASS")
    print("TRAIN label == measured-power max = 7,030/7,030 PASS")
    print("FORMAL measured-power numeric decode = NO")

    print("\n===== B. PHASE A — FORMAL CONTROLLER DECISION FREEZE =====")
    formal_rows,formal_counts=controller_formal_rows(policy_mod)
    raw,scr,input_ok,input_errors,load_workers=load_formal_lidar(formal_rows,loader_mod,fmap)

    norm=json.loads(NORM.read_text())
    rm=np.asarray(norm["raw_mean"],np.float32); rs=np.asarray(norm["raw_std"],np.float32)
    sm=np.asarray(norm["scr_mean"],np.float32); ss=np.asarray(norm["scr_std"],np.float32)
    raw=(raw-rm)/np.maximum(rs,1e-8); scr=(scr-sm)/np.maximum(ss,1e-8)

    cal=json.loads(CAL.read_text())
    T=float(cal["temperature"])
    require(abs(T-1.1038547007246138)<1e-15,"Frozen temperature changed")

    ckpt=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
    model=model_mod.DualLiDARConv1D64()
    model.load_state_dict(ckpt["state_dict"],strict=True); model.eval()
    torch.set_num_threads(max(1,min(os.cpu_count() or 1,12)))

    logits=np.full((1030,64),np.nan,dtype=np.float64)
    valid_idx=np.where(input_ok)[0]
    with torch.inference_mode():
        for start in range(0,len(valid_idx),256):
            ix=valid_idx[start:start+256]
            a=torch.from_numpy(np.ascontiguousarray(raw[ix].transpose(0,2,1)))
            b=torch.from_numpy(np.ascontiguousarray(scr[ix].transpose(0,2,1)))
            z=model(a,b).cpu().numpy().astype(np.float64)
            require(z.shape==(len(ix),64),"Formal logit batch shape changed")
            logits[ix]=z

    controller_records=[]
    fallback_total=0
    for i,r in enumerate(formal_rows):
        if input_ok[i] and np.isfinite(logits[i]).all():
            probs=runtime_mod.calibrated_probabilities_from_logits(logits[i],T)
            prob_sha=hashlib.sha256(
                json.dumps(probs,separators=(",",":"),ensure_ascii=False).encode()
            ).hexdigest()
            decisions={}
            for pname in POLICIES:
                out=runtime_mod.safe_select(policy_mod,pname,logits[i],T)
                fallback_total += int(out["fallback_triggered"])
                decisions[pname]={
                    "selected_beams":out["selected_beams"],
                    "K":out["K"],
                    "requested_mass":out.get("requested_mass"),
                    "achieved_probability_mass":out.get("achieved_probability_mass"),
                    "fallback_triggered":out["fallback_triggered"],
                    "fallback_reason":out["fallback_reason"],
                    "measured_power_accessed":False,
                }
            rec={
                "sample_id":r["sample_id"],"scenario":r["scenario"],
                "csv_relative_path":r["csv_relative_path"],
                "official_csv_index":r["official_csv_index"],
                "raw_lidar_ref":r["raw_ref"],"scr_lidar_ref":r["scr_ref"],
                "power_ref_identifier_only":r["power_ref"],
                "temperature":T,
                "probabilities_sha256":prob_sha,
                "probabilities":probs,
                "input_valid":True,"input_error":None,
                "policies":decisions,
                "formal_label_consumed_by_controller":False,
                "formal_power_numeric_values_consumed_by_controller":False,
            }
        else:
            decisions={}
            reason=input_errors[i] or "nonfinite formal logits"
            for pname in POLICIES:
                out=runtime_mod.exhaustive_fallback(reason)
                out["requested_policy_name"]=pname
                fallback_total += 1
                decisions[pname]={
                    "selected_beams":out["selected_beams"],"K":64,
                    "requested_mass":None,"achieved_probability_mass":None,
                    "fallback_triggered":True,"fallback_reason":out["fallback_reason"],
                    "measured_power_accessed":False,
                }
            rec={
                "sample_id":r["sample_id"],"scenario":r["scenario"],
                "csv_relative_path":r["csv_relative_path"],
                "official_csv_index":r["official_csv_index"],
                "raw_lidar_ref":r["raw_ref"],"scr_lidar_ref":r["scr_ref"],
                "power_ref_identifier_only":r["power_ref"],
                "temperature":T,
                "probabilities_sha256":None,"probabilities":None,
                "input_valid":False,"input_error":reason,
                "policies":decisions,
                "formal_label_consumed_by_controller":False,
                "formal_power_numeric_values_consumed_by_controller":False,
            }
        controller_records.append(rec)

    atomic_jsonl(CONTROLLER_JSONL,controller_records)
    controller_sha=sha(CONTROLLER_JSONL)
    atomic_json(CONTROLLER_SEAL,{
        "stage":8,"block":"8.8","status":"FROZEN_PREOUTCOME_FORMAL_CONTROLLER_DECISIONS",
        "rows":1030,"scenario_counts":formal_counts,
        "controller_decisions_sha256":controller_sha,
        "checkpoint_sha256":EXPECTED[str(CHECKPOINT)],
        "calibrator_sha256":EXPECTED[str(CAL)],
        "policy_sha256":EXPECTED[str(POLICY_SRC)],
        "runtime_sha256":EXPECTED[str(RUNTIME_SRC)],
        "formal_label_consumed_by_controller":False,
        "formal_power_numeric_values_consumed_by_controller":False,
        "fallback_decision_count":fallback_total,
        "formal_power_decode_authorized_next":True,
    })
    controller_seal_sha=sha(CONTROLLER_SEAL)
    print("formal controller rows frozen =",len(controller_records))
    print("Scenario8/9 formal controller rows =",formal_counts["8"],"/",formal_counts["9"])
    print("formal label consumed by controller = NO")
    print("formal measured-power numeric values consumed by controller = NO")
    print("controller decisions SHA256 =",controller_sha)
    print("controller pre-outcome seal SHA256 =",controller_seal_sha)

    # Explicit causal barrier: free model-side arrays before evaluator phase.
    del raw,scr,logits,model,ckpt
    gc.collect()

    atomic_json(FORMAL_ACCESS_AUTH,{
        "stage":8,"block":"8.8","status":"FORMAL_MEASURED_POWER_ACCESS_AUTHORIZED",
        "controller_preoutcome_seal_sha256":controller_seal_sha,
        "controller_decisions_sha256":controller_sha,
        "evaluator_sha256":EXPECTED[str(EVALUATOR_SRC)],
        "model_calibrator_policy_frozen":True,
        "controller_reexecution_after_this_marker":False,
    })
    print("FORMAL POWER ACCESS BARRIER = SEALED")

    print("\n===== C. PHASE B — FIRST FORMAL MEASURED-POWER OPENING =====")
    labels=evaluator_formal_labels()
    decision_by_id={r["sample_id"]:r for r in controller_records}
    eval_records=[]
    label_consistency=0

    for r in formal_rows:
        sid=r["sample_id"]
        require_frozen_hash(r["power"],fmap)
        powers=parse_power(r["power"])
        require(all(x>0.0 for x in powers),f"Formal measured powers not strictly positive: {sid}")
        y=labels[sid]
        require(evaluator.label_consistent_with_power(y,powers),
                f"FORMAL label/power-max consistency failed: {sid}")
        label_consistency += 1

        methods={}
        d=decision_by_id[sid]
        for pname in POLICIES:
            sel=d["policies"][pname]["selected_beams"]
            methods[pname]=evaluator.evaluate_selected_set(sel,powers,y)

        # Evaluator-only GT/power-derived best-beam reference. Never controller input.
        methods[ORACLE]=evaluator.evaluate_selected_set([y],powers,y)

        eval_records.append({
            "sample_id":sid,"scenario":r["scenario"],
            "csv_relative_path":r["csv_relative_path"],
            "official_csv_index":r["official_csv_index"],
            "power_ref":r["power_ref"],
            "formal_ground_truth_beam":y,
            "power_file_sha256":sha(r["power"]),
            "label_consistent_with_measured_power_max":True,
            "methods":methods,
            "controller_decisions_sha256":controller_sha,
            "measured_power_used_by_controller":False,
        })

    require(label_consistency==1030,"Formal label consistency count changed")
    atomic_jsonl(EVAL_JSONL,eval_records)
    eval_sha=sha(EVAL_JSONL)
    print("formal measured-power vectors opened/evaluated = 1,030")
    print("formal label == measured-power max = 1,030/1,030 PASS")
    print("measured power used by controller = NO")
    print("formal evaluator records SHA256 =",eval_sha)

    print("\n===== D. FORMAL POINT-ESTIMATE SUMMARY =====")
    summary={
        "stage":8,"block":"8.8","status":"FORMAL_MEASURED_DEEPSENSE_POINT_ESTIMATES",
        "formal_rows":1030,"scenario_counts":formal_counts,
        "primary_policy":"adaptive_topk_q095",
        "methods":{},
        "bootstrap_CI":"DEFERRED_TO_BLOCK8.9",
        "absolute_measured_spectral_efficiency_claim":False,
        "normalized_SE_primary_reference_snr_db":10,
        "formal_power_accessed":True,
        "post_outcome_tuning":False,
    }
    for m in ALL_METHODS:
        summary["methods"][m]={
            "Scenario8":aggregate_records(eval_records,m,8),
            "Scenario9":aggregate_records(eval_records,m,9),
            "pooled":aggregate_records(eval_records,m,None),
        }
    atomic_json(SUMMARY,summary)

    for m in ALL_METHODS:
        p=summary["methods"][m]["pooled"]
        print(
            f"{m}: coverage={p['coverage']:.6f} meanK={p['mean_K']:.6f} "
            f"loss_dB={p['mean_measured_power_loss_db']:.6f} "
            f"outage3dB={p['outage_rate']['3dB']:.6f} "
            f"SEgap10={p['mean_normalized_se_gap_bps_per_hz']['10dB']:.6f}"
        )

    atomic_json(MANIFEST,{
        "stage":8,"block":"8.8","status":"FROZEN_COMPLETE_BLOCK88",
        "contract_sha256":EXPECTED_CONTRACT,
        "Block87_seal_sha256":EXPECTED[str(SEAL87)],
        "Block86_seal_sha256":EXPECTED[str(SEAL86)],
        "Block85B_seal_sha256":EXPECTED[str(SEAL85B)],
        "Block84_seal_sha256":EXPECTED[str(B84_SEAL)],
        "Block83_seal_sha256":EXPECTED[str(B83_SEAL)],
        "Block82B_seal_sha256":EXPECTED[str(B82B_SEAL)],
        "runner_sha256":sha(SCRIPT),
        "train_power_gate_sha256":sha(TRAIN_GATE),
        "controller_decisions_sha256":controller_sha,
        "controller_preoutcome_seal_sha256":controller_seal_sha,
        "formal_access_authorization_sha256":sha(FORMAL_ACCESS_AUTH),
        "formal_records_sha256":eval_sha,
        "formal_summary_sha256":sha(SUMMARY),
        "formal_power_accessed":True,
        "controller_reexecuted_after_power_open":False,
        "model_update":False,"calibrator_refit":False,"policy_update":False,"evaluator_update":False,
    })
    atomic_json(REPORT,{
        "stage":8,"block":"8.8","status":"PASS_BLOCK88_FORMAL_MEASURED_DEEPSENSE_EVALUATION",
        "formal_rows":1030,"scenario8_rows":437,"scenario9_rows":593,
        "controller_frozen_before_formal_power_decode":True,
        "formal_label_power_consistency_passed":1030,
        "controller_fallback_decision_count":fallback_total,
        "primary_policy_pooled":summary["methods"]["adaptive_topk_q095"]["pooled"],
        "formal_power_accessed":True,
        "measured_power_controller_input":False,
        "post_outcome_tuning":False,
        "bootstrap_CI":"DEFERRED_TO_BLOCK8.9",
        "Block89_authorized":True,
        "controller_decisions_sha256":controller_sha,
        "formal_records_sha256":eval_sha,
        "formal_summary_sha256":sha(SUMMARY),
        "manifest_sha256":sha(MANIFEST),
        "runner_sha256":sha(SCRIPT),
    })
    atomic_json(SEAL,{
        "stage":8,"block":"8.8","status":"FROZEN_COMPLETE_BLOCK88",
        "Block87_seal_sha256":EXPECTED[str(SEAL87)],
        "Block83_evaluator_sha256":EXPECTED[str(EVALUATOR_SRC)],
        "contract_sha256":EXPECTED_CONTRACT,
        "runner_sha256":sha(SCRIPT),
        "controller_decisions_sha256":controller_sha,
        "controller_preoutcome_seal_sha256":controller_seal_sha,
        "formal_records_sha256":eval_sha,
        "formal_summary_sha256":sha(SUMMARY),
        "manifest_sha256":sha(MANIFEST),
        "report_sha256":sha(REPORT),
        "formal_power_accessed":True,
        "controller_reexecution_after_power_open":False,
        "measured_power_controller_input":False,
        "post_outcome_tuning":False,
        "Block89_may_start":True,
    })

    # After formal outcome access, only identity verification; no upstream mutation.
    for raw,expected in EXPECTED.items():
        require(sha(Path(raw))==expected,f"Frozen upstream modified during formal evaluation: {raw}")

    p=summary["methods"]["adaptive_topk_q095"]["pooled"]
    print("\n"+"="*96)
    print("BLOCK 8.8 = FULLY VERIFIED / FROZEN")
    print("FORMAL dev_test ROWS = 1,030 (437 + 593)")
    print("PREOUTCOME CONTROLLER DECISIONS FROZEN BEFORE POWER DECODE = PASS")
    print("FORMAL LABEL/MEASURED-POWER MAX CONSISTENCY = 1,030/1,030 PASS")
    print("PRIMARY POLICY = ADAPTIVE TOP-K q=0.95")
    print(f"PRIMARY POOLED COVERAGE = {p['coverage']:.12f}")
    print(f"PRIMARY POOLED MEAN K = {p['mean_K']:.12f}")
    print(f"PRIMARY POOLED MEAN POWER LOSS dB = {p['mean_measured_power_loss_db']:.12f}")
    print(f"PRIMARY POOLED OUTAGE@3dB = {p['outage_rate']['3dB']:.12f}")
    print(f"PRIMARY POOLED NORMALIZED SE GAP@10dB = {p['mean_normalized_se_gap_bps_per_hz']['10dB']:.12f}")
    print("FIXED TOP-1/3/5 + q SWEEP + EXHAUSTIVE-64 = EVALUATED")
    print("EVALUATOR-ONLY GT-BEST-BEAM ORACLE = EVALUATED")
    print("MEASURED POWER CONTROLLER INPUT = NO")
    print("MODEL / CALIBRATOR / POLICY / EVALUATOR UPDATE = NO")
    print("POST-OUTCOME TUNING = NO")
    print("BOOTSTRAP CIs = DEFERRED TO BLOCK8.9")
    print("BLOCK 8.9 MAY START = YES")
    print("8.8 train gate SHA256 =",sha(TRAIN_GATE))
    print("8.8 controller decisions SHA256 =",controller_sha)
    print("8.8 controller seal SHA256 =",controller_seal_sha)
    print("8.8 formal records SHA256 =",eval_sha)
    print("8.8 summary SHA256 =",sha(SUMMARY))
    print("8.8 manifest SHA256 =",sha(MANIFEST))
    print("8.8 report SHA256 =",sha(REPORT))
    print("8.8 seal SHA256 =",sha(SEAL))
    print("8.8 runner SHA256 =",sha(SCRIPT))
    print("STATUS = FROZEN_COMPLETE_BLOCK88")
    print("="*96)
    return 0

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("\n"+"!"*96)
        print("BLOCK 8.8 FAIL-CLOSED")
        print(f"{type(e).__name__}: {e}")
        print("AFTER FORMAL POWER ACCESS, DO NOT MODIFY MODEL/CALIBRATOR/POLICY/EVALUATOR.")
        print("DO NOT START BLOCK8.9 UNLESS BLOCK8.8 SEAL EXISTS.")
        print("!"*96)
        raise
