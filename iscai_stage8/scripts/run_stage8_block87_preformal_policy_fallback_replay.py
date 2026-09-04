#!/usr/bin/env python3
from __future__ import annotations

import csv
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

SCRIPT = S8 / "scripts/run_stage8_block87_preformal_policy_fallback_replay.py"
CONTRACT = S8 / "configs/stage8_block87_preformal_policy_fallback_replay_contract.json"
RUNTIME = S8 / "src/deepsense_policy_runtime.py"

MODEL_SRC = S8 / "src/deepsense_lidar_encoder.py"
LOADER_SRC = S8 / "src/mat5_lidar_loader.py"
POLICY_SRC = S8 / "src/deepsense_beam_policy.py"

B85B = S8 / "artifacts/block85b_encoder_training_cpu_repair"
CHECKPOINT = B85B / "duallidarconv1d64_epoch080.pt"
NORM = B85B / "train_only_normalization.json"
SEAL85B = B85B / "stage8_block85b_encoder_training_cpu_repair_seal.json"

B86 = S8 / "artifacts/block86_probability_calibration_final"
CAL = B86 / "frozen_temperature_calibrator.json"
INDEX86 = B86 / "calibration_sample_index.json"
DIAG86 = B86 / "calibration_diagnostics.json"
MANIFEST86 = B86 / "stage8_block86_manifest.json"
SEAL86 = B86 / "stage8_block86_probability_calibration_seal.json"
REPORT86 = S8 / "reports/stage8_block86_probability_calibration_final_report.json"
RUNNER86 = S8 / "scripts/run_stage8_block86_probability_calibration_final.py"

B82B_FILES = S8 / "artifacts/block82b_canonical_corpus_schema_split/canonical_extracted_file_manifest.json"
B84_SEAL = S8 / "artifacts/block84_beam_policy_interface/stage8_block84_beam_policy_interface_seal.json"

OUT = S8 / "artifacts/block87_preformal_policy_fallback_replay"
REPLAY = OUT / "calibration_policy_replay_summary.json"
SYNTH = OUT / "synthetic_fallback_tests.json"
MANIFEST = OUT / "stage8_block87_manifest.json"
REPORT = S8 / "reports/stage8_block87_preformal_policy_fallback_replay_report.json"
SEAL = OUT / "stage8_block87_preformal_policy_fallback_replay_seal.json"

EXPECTED_CONTRACT = "635cc64fb9b468f4c6760c5d13370b600b911b3c0f3eaab2866e7ef975b17e75"
EXPECTED_RUNTIME = "5171a69ebe491027088934075c2fe2c8c74ef4a5e10823f2ff636237056ff3f3"
EXPECTED = {
 str(CHECKPOINT): "a31e411a4c729b9ff99c20295f849112417420d301296d3f6e6f73d1a1ac99b5",
 str(NORM): "8e12e7b5bccf57c31075569a575507cd95e4b1710f5bb7327cbdd203c36575fd",
 str(SEAL85B): "ee9e5a8d7f4d630e5334f5f5089484e6016f35e38316346edb5a3194af74ec3b",
 str(CAL): "ddac88b72b6fc08953ca5dcbf899c5bacaa646a252986147389d079bcaea8d82",
 str(INDEX86): "dc8a3229b1d103eee281ed824e9b0621b67e8013f0c8fef3b1308a0cf152aa36",
 str(DIAG86): "ec12d83799659005cacb14ed306e862a86f259a62e5b9dfea98677674b878bc6",
 str(MANIFEST86): "52aecb66146ee99f56cbcc17dc244f14c9e9f06fedd5758449feb3bafb7368f8",
 str(REPORT86): "9310098c7dba3c0457e33284f2b736425c2ac293a47556447bfc69bc70769fe8",
 str(SEAL86): "6bd11ccb198ee67f6aa15895951e69a663d5474e1630c27643e7d7f8343ed444",
 str(RUNNER86): "39a2b3aaf409881931ca84532807dc7b7a88de13687e175a7475c421b8f0b218",
 str(B82B_FILES): "79dc92fb6a90acd36c4598de436517ebac6c7e7904334dd76a520e1c90a0b895",
 str(POLICY_SRC): "5b7e6c3c4ff7eca4d1db86df90edfa19f0bbfc551255779833485df074d87cb8",
 str(B84_SEAL): "c5dcbf5b1bf8d7965fd32bf794f095b4495d30d98e35bddef22ce3abffe554e2",
 str(MODEL_SRC): "adb943f12490612a6b92ffa3e70f69b96b1002fa77e4e769f5ea26ed6d25c417",
 str(LOADER_SRC): "b4b3d9e09a20a5ee790a4ad7fd22f53dad1d6f4b953e70f824ade80f9b86299c",
}

VAL_SPECS = [
 (8, DATA_ROOT / "LiDAR/Scenario8/development_dataset/scenario8_dev_val.csv"),
 (9, DATA_ROOT / "LiDAR/Scenario9/development_dataset/scenario9_dev_val.csv"),
]
POLICIES = [
 "adaptive_topk_q090","adaptive_topk_q095","adaptive_topk_q0975","adaptive_topk_q099",
 "fixed_top1","fixed_top3","fixed_top5","exhaustive_64",
]
QMAP = {
 "adaptive_topk_q090":0.90,
 "adaptive_topk_q095":0.95,
 "adaptive_topk_q0975":0.975,
 "adaptive_topk_q099":0.99,
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

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def resolve_ref(csv_path, raw_ref):
    s=(raw_ref or "").strip().replace("\\","/")
    while s.startswith("./"): s=s[2:]
    return (DATA_ROOT/s).resolve() if s.startswith("LiDAR/") else (csv_path.parent/s).resolve()

def build_records():
    out=[]; counts={}
    for sc,csv_path in VAL_SPECS:
        n=0
        with csv_path.open(newline="",encoding="utf-8-sig") as f:
            r=csv.DictReader(f)
            require(r.fieldnames==['index','unit1_lidar_1','unit1_lidar_SCR_1','unit1_pwr_1','beam_index_1'],
                    f"Unexpected val headers: {r.fieldnames}")
            for rn,row in enumerate(r,start=2):
                raw=resolve_ref(csv_path,row["unit1_lidar_1"])
                scr=resolve_ref(csv_path,row["unit1_lidar_SCR_1"])
                require(raw.is_file() and scr.is_file(),f"Missing val LiDAR: {csv_path}:{rn}")
                out.append({"scenario":sc,"csv":csv_path,"row":rn,"raw":raw,"scr":scr})
                n+=1
        counts[str(sc)]=n
    require(counts=={"8":775,"9":1172} and len(out)==1947,f"Val counts changed: {counts}, n={len(out)}")
    return out,counts

def load_arrays(records,loader):
    n=len(records)
    raw=np.empty((n,460,2),np.float32); scr=np.empty((n,216,2),np.float32)
    def one(item):
        i,r=item
        a=loader.load_mat5_double_matrix(r["raw"],expected_name="data",expected_shape=(460,2))
        b=loader.load_mat5_double_matrix(r["scr"],expected_name="data",expected_shape=(216,2))
        return i,a.astype(np.float32,copy=False),b.astype(np.float32,copy=False)
    workers=max(1,min(os.cpu_count() or 1,32))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i,a,b in ex.map(one,enumerate(records),chunksize=16):
            raw[i]=a; scr[i]=b
    return raw,scr,workers

def validate_policy_output(policy_name,out,probs):
    require(out["fallback_triggered"] is False,f"Unexpected fallback for valid replay: {policy_name}")
    beams=out["selected_beams"]; k=out["K"]
    require(k==len(beams) and 1<=k<=64,f"Invalid K for {policy_name}")
    require(len(set(beams))==len(beams),"Duplicate selected beams")
    require(all(isinstance(x,int) and 0<=x<64 for x in beams),"Beam outside 0..63")
    if policy_name=="fixed_top1": require(k==1,"fixed_top1 K changed")
    if policy_name=="fixed_top3": require(k==3,"fixed_top3 K changed")
    if policy_name=="fixed_top5": require(k==5,"fixed_top5 K changed")
    if policy_name=="exhaustive_64": require(k==64 and beams==list(range(64)),"exhaustive semantics changed")
    if policy_name in QMAP:
        q=QMAP[policy_name]
        mass=sum(probs[i] for i in beams)
        require(mass+1e-12>=q,f"Adaptive mass below q for {policy_name}")
        if len(beams)>1:
            prev=sum(probs[i] for i in beams[:-1])
            require(prev < q+1e-12,f"Adaptive set not minimal for {policy_name}")
    return k

def main():
    print("="*88)
    print("STAGE 8 — BLOCK 8.7")
    print("PRE-FORMAL CALIBRATED POLICY / FALLBACK REPLAY")
    print("DEV_VAL ONLY — NO FORMAL / NO MMWAVE POWER ACCESS")
    print("="*88)

    require(Path(__file__).resolve()==SCRIPT.resolve(),"Runner path mismatch")
    require(CONTRACT.is_file() and sha(CONTRACT)==EXPECTED_CONTRACT,"8.7 contract SHA mismatch")
    require(RUNTIME.is_file() and sha(RUNTIME)==EXPECTED_RUNTIME,"8.7 runtime SHA mismatch")
    for raw,expected in EXPECTED.items():
        p=Path(raw); require(p.is_file(),f"Missing frozen input: {p}"); require(sha(p)==expected,f"Frozen SHA changed: {p}")
    for p in (REPLAY,SYNTH,MANIFEST,REPORT,SEAL):
        require(not p.exists(),f"8.7 output exists: {p}")

    s86=json.loads(SEAL86.read_text())
    require(s86.get("status")=="FROZEN_COMPLETE_BLOCK86_FINAL","8.6 FINAL status changed")
    require(s86.get("Block87_may_start") is True,"8.7 not authorized")
    require(s86.get("formal_data_access") is False and s86.get("formal_power_access") is False,
            "8.6 FINAL formal boundary changed")

    cal=json.loads(CAL.read_text())
    T=float(cal["temperature"])
    require(abs(T-1.1038547007246138)<1e-15,"Frozen temperature changed")

    print("\n===== A. FROZEN PIPELINE GATE =====")
    print("Block8.6 FINAL exact identities = PASS")
    print("temperature =",T)
    print("formal dev_test / formal power access = NO")

    model_mod=load_module("ds_model",MODEL_SRC)
    loader_mod=load_module("mat5",LOADER_SRC)
    policy_mod=load_module("beam_policy",POLICY_SRC)
    runtime_mod=load_module("policy_runtime",RUNTIME)

    print("\n===== B. DEV_VAL PIPELINE REPLAY =====")
    records,counts=build_records()
    raw,scr,workers=load_arrays(records,loader_mod)
    norm=json.loads(NORM.read_text())
    rm=np.asarray(norm["raw_mean"],np.float32); rs=np.asarray(norm["raw_std"],np.float32)
    sm=np.asarray(norm["scr_mean"],np.float32); ss=np.asarray(norm["scr_std"],np.float32)
    raw=(raw-rm)/np.maximum(rs,1e-8); scr=(scr-sm)/np.maximum(ss,1e-8)
    require(np.isfinite(raw).all() and np.isfinite(scr).all(),"Non-finite normalized replay LiDAR")

    ckpt=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
    model=model_mod.DualLiDARConv1D64()
    model.load_state_dict(ckpt["state_dict"],strict=True); model.eval()
    torch.set_num_threads(max(1,min(os.cpu_count() or 1,12)))

    logits=[]
    with torch.inference_mode():
        for i in range(0,1947,256):
            a=torch.from_numpy(np.ascontiguousarray(raw[i:i+256].transpose(0,2,1)))
            b=torch.from_numpy(np.ascontiguousarray(scr[i:i+256].transpose(0,2,1)))
            z=model(a,b).cpu().numpy().astype(np.float64)
            require(np.isfinite(z).all(),"Non-finite replay logits")
            logits.append(z)
    logits=np.concatenate(logits,axis=0)
    require(logits.shape==(1947,64),"Replay logit shape changed")

    stats={p:{"fallback_count":0,"K_sum":0,"K_min":65,"K_max":0} for p in POLICIES}
    monotonic_ok=0

    for z in logits:
        probs=runtime_mod.calibrated_probabilities_from_logits(z,T)
        ks={}
        for pname in POLICIES:
            out=runtime_mod.safe_select(policy_mod,pname,z,T)
            k=validate_policy_output(pname,out,probs)
            stats[pname]["fallback_count"] += int(out["fallback_triggered"])
            stats[pname]["K_sum"] += k
            stats[pname]["K_min"] = min(stats[pname]["K_min"],k)
            stats[pname]["K_max"] = max(stats[pname]["K_max"],k)
            ks[pname]=k
        require(ks["adaptive_topk_q090"]<=ks["adaptive_topk_q095"]<=ks["adaptive_topk_q0975"]<=ks["adaptive_topk_q099"],
                f"Adaptive K monotonicity failed: {ks}")
        monotonic_ok += 1

    for pname in POLICIES:
        stats[pname]["mean_K"]=stats[pname].pop("K_sum")/1947.0
        require(stats[pname]["fallback_count"]==0,f"Valid replay fallback occurred for {pname}")

    require(monotonic_ok==1947,"Not all samples passed q monotonicity")
    print("dev_val rows replayed = 1,947")
    print("Scenario8/9 =",counts["8"],"/",counts["9"])
    print("parallel MAT load workers =",workers)
    print("valid-data fallback count = 0 for all policies")
    print("adaptive q monotonicity = 1,947/1,947 PASS")
    for pname in POLICIES:
        s=stats[pname]
        print(f"{pname}: meanK={s['mean_K']:.6f} minK={s['K_min']} maxK={s['K_max']}")

    print("\n===== C. SYNTHETIC FAIL-CLOSED FALLBACK TESTS =====")
    good=np.zeros(64,dtype=np.float64); good[3]=2.0
    cases=[
        ("wrong_logit_length",np.zeros(63),T,"adaptive_topk_q095"),
        ("nan_logit",np.where(np.arange(64)==2,np.nan,0.0),T,"adaptive_topk_q095"),
        ("inf_logit",np.where(np.arange(64)==2,np.inf,0.0),T,"adaptive_topk_q095"),
        ("nonpositive_temperature",good,0.0,"adaptive_topk_q095"),
        ("nan_temperature",good,float("nan"),"adaptive_topk_q095"),
        ("unsupported_policy",good,T,"not_a_policy"),
    ]
    synth=[]
    for name,z,t,pname in cases:
        out=runtime_mod.safe_select(policy_mod,pname,z,t)
        require(out["fallback_triggered"] is True,f"Fallback did not trigger: {name}")
        require(out["K"]==64 and out["selected_beams"]==list(range(64)),f"Fallback not exhaustive: {name}")
        require(out["measured_power_accessed"] is False,f"Fallback measured-power boundary failed: {name}")
        synth.append({"name":name,"pass":True,"K":64})
    print("synthetic fallback tests = 6/6 PASS")
    print("fallback = exhaustive_64 / measured-power access = NO")

    atomic_json(REPLAY,{
        "stage":8,"block":"8.7","status":"PASS_CALIBRATION_POLICY_REPLAY",
        "rows":1947,"scenario_counts":counts,"temperature":T,
        "policy_stats":stats,
        "adaptive_K_monotonic_samples_passed":monotonic_ok,
        "valid_data_fallback_total":sum(x["fallback_count"] for x in stats.values()),
        "formal_data_access":False,"mmwave_power_payload_access":False,
    })
    atomic_json(SYNTH,{
        "stage":8,"block":"8.7","status":"PASS_SYNTHETIC_FAIL_CLOSED_FALLBACK_TESTS",
        "tests":synth,"all_fallbacks_exhaustive_64":True,"measured_power_access":False,
    })
    atomic_json(MANIFEST,{
        "stage":8,"block":"8.7","status":"FROZEN_COMPLETE_BLOCK87",
        "contract_sha256":EXPECTED_CONTRACT,"runtime_sha256":EXPECTED_RUNTIME,
        "Block86_seal_sha256":EXPECTED[str(SEAL86)],
        "checkpoint_sha256":EXPECTED[str(CHECKPOINT)],
        "calibrator_sha256":EXPECTED[str(CAL)],
        "policy_sha256":EXPECTED[str(POLICY_SRC)],
        "runner_sha256":sha(SCRIPT),
        "replay_sha256":sha(REPLAY),"synthetic_fallback_sha256":sha(SYNTH),
        "formal_data_access":False,"formal_power_access":False,
    })
    atomic_json(REPORT,{
        "stage":8,"block":"8.7","status":"PASS_BLOCK87_PREFORMAL_POLICY_FALLBACK_REPLAY",
        "replay_rows":1947,"policy_count":len(POLICIES),
        "all_valid_replay_fallback_counts_zero":True,
        "adaptive_K_monotonicity_passed_rows":1947,
        "synthetic_fallback_tests_passed":6,
        "fallback_policy":"exhaustive_64",
        "training":False,"calibration_refit":False,
        "formal_data_access":False,"formal_power_access":False,
        "Block88_authorized":True,
        "replay_sha256":sha(REPLAY),"synthetic_sha256":sha(SYNTH),
        "manifest_sha256":sha(MANIFEST),"runner_sha256":sha(SCRIPT),
    })
    atomic_json(SEAL,{
        "stage":8,"block":"8.7","status":"FROZEN_COMPLETE_BLOCK87",
        "Block86_seal_sha256":EXPECTED[str(SEAL86)],
        "Block85B_seal_sha256":EXPECTED[str(SEAL85B)],
        "Block84_seal_sha256":EXPECTED[str(B84_SEAL)],
        "contract_sha256":EXPECTED_CONTRACT,"runtime_sha256":EXPECTED_RUNTIME,
        "runner_sha256":sha(SCRIPT),"replay_sha256":sha(REPLAY),
        "synthetic_fallback_sha256":sha(SYNTH),"manifest_sha256":sha(MANIFEST),
        "report_sha256":sha(REPORT),
        "formal_data_access":False,"formal_power_access":False,
        "Block88_may_start":True,
    })

    for raw,expected in EXPECTED.items():
        require(sha(Path(raw))==expected,f"Frozen upstream modified during 8.7: {raw}")

    print("\n"+"="*88)
    print("BLOCK 8.7 = FULLY VERIFIED / FROZEN")
    print("END-TO-END CALIBRATED POLICY REPLAY = 1,947 dev_val ROWS PASS")
    print("PRIMARY POLICY q=0.95 + q SWEEP = PASS")
    print("FIXED TOP-1/3/5 + EXHAUSTIVE-64 = PASS")
    print("ADAPTIVE K MONOTONICITY = 1,947/1,947 PASS")
    print("VALID-DATA FALLBACKS = 0")
    print("SYNTHETIC FAIL-CLOSED FALLBACK TESTS = 6/6 PASS")
    print("FALLBACK = EXHAUSTIVE-64")
    print("MEASURED POWER CONTROLLER INPUT = NO")
    print("TRAINING / CALIBRATION REFIT = NO")
    print("FORMAL dev_test / FORMAL POWER ACCESS = NO")
    print("BLOCK 8.8 MAY START = YES")
    print("8.7 replay SHA256 =",sha(REPLAY))
    print("8.7 synthetic SHA256 =",sha(SYNTH))
    print("8.7 manifest SHA256 =",sha(MANIFEST))
    print("8.7 report SHA256 =",sha(REPORT))
    print("8.7 seal SHA256 =",sha(SEAL))
    print("8.7 runner SHA256 =",sha(SCRIPT))
    print("STATUS = FROZEN_COMPLETE_BLOCK87")
    print("="*88)
    return 0

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("\n"+"!"*88)
        print("BLOCK 8.7 FAIL-CLOSED")
        print(f"{type(e).__name__}: {e}")
        print("DO NOT ACCESS FORMAL dev_test OR FORMAL MMWAVE POWER.")
        print("DO NOT START BLOCK8.8.")
        print("!"*88)
        raise
