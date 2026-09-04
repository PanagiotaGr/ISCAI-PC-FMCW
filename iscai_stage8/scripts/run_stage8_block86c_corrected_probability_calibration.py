#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, importlib.util, json, math, os, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch

ROOT=Path("/home/agni/waymo")
S8=ROOT/"iscai_stage8"
DATA_ROOT=S8/"data/block82b_deepsense_primary"

SCRIPT=S8/"scripts/run_stage8_block86c_corrected_probability_calibration.py"
CONTRACT=S8/"configs/stage8_block86c_corrected_probability_calibration_contract.json"
MODEL_SRC=S8/"src/deepsense_lidar_encoder.py"
LOADER_SRC=S8/"src/mat5_lidar_loader.py"

B85C=S8/"artifacts/block85c_label_index_corrected_training"
CKPT=B85C/"duallidarconv1d64_label0based_epoch080.pt"
MAP85C=B85C/"train_label_index_mapping_gate.json"
SEAL85C=B85C/"stage8_block85c_label_index_corrected_training_seal.json"
REPORT85C=S8/"reports/stage8_block85c_label_index_corrected_training_report.json"
NORM=S8/"artifacts/block85b_encoder_training_cpu_repair/train_only_normalization.json"
B82B_FILES=S8/"artifacts/block82b_canonical_corpus_schema_split/canonical_extracted_file_manifest.json"
B84_SEAL=S8/"artifacts/block84_beam_policy_interface/stage8_block84_beam_policy_interface_seal.json"

OUT=S8/"artifacts/block86c_corrected_probability_calibration"
CAL=OUT/"frozen_temperature_calibrator.json"
INDEX=OUT/"calibration_sample_index.json"
DIAG=OUT/"calibration_diagnostics.json"
MANIFEST=OUT/"stage8_block86c_manifest.json"
REPORT=S8/"reports/stage8_block86c_corrected_probability_calibration_report.json"
SEAL=OUT/"stage8_block86c_corrected_probability_calibration_seal.json"

EXPECTED_CONTRACT="e630a81585c0355f12766c67dc721596eaa563bd5bc4bf06ec760bcd447bb806"
EXPECTED={
 str(MODEL_SRC):"adb943f12490612a6b92ffa3e70f69b96b1002fa77e4e769f5ea26ed6d25c417",
 str(LOADER_SRC):"b4b3d9e09a20a5ee790a4ad7fd22f53dad1d6f4b953e70f824ade80f9b86299c",
 str(CKPT):"7aa785741a2929eedc106e77b0e701f02412f5dbf434823e51eed6d0f4eb6417",
 str(MAP85C):"36b7610ebb1cdf7e8004c10baad794d6e074554fa5805e34eac020eb681eac75",
 str(SEAL85C):"b0dae15111baae6a0cbe132e402459148474c6bad87c58c5a71eccba737de91e",
 str(REPORT85C):"78b8f3a57550f489ed8979f7b839f9e3869eb221f59a80bc841c658260174dcb",
 str(NORM):"8e12e7b5bccf57c31075569a575507cd95e4b1710f5bb7327cbdd203c36575fd",
 str(B82B_FILES):"79dc92fb6a90acd36c4598de436517ebac6c7e7904334dd76a520e1c90a0b895",
 str(B84_SEAL):"c5dcbf5b1bf8d7965fd32bf794f095b4495d30d98e35bddef22ce3abffe554e2",
}

VAL_SPECS=[
 (8,DATA_ROOT/"LiDAR/Scenario8/development_dataset/scenario8_dev_val.csv"),
 (9,DATA_ROOT/"LiDAR/Scenario9/development_dataset/scenario9_dev_val.csv"),
]

class FailClosed(RuntimeError): pass
def req(c,m):
    if not c: raise FailClosed(m)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def atom_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    req(not path.exists(),f"Refusing overwrite: {path}")
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

def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def resolve(csvp,s):
    s=(s or "").strip().replace("\\","/")
    while s.startswith("./"): s=s[2:]
    return (DATA_ROOT/s).resolve() if s.startswith("LiDAR/") else (csvp.parent/s).resolve()

def records():
    out=[]; counts={}
    for sc,csvp in VAL_SPECS:
        n=0
        with csvp.open(newline="",encoding="utf-8-sig") as f:
            r=csv.DictReader(f)
            req(r.fieldnames==["index","unit1_lidar_1","unit1_lidar_SCR_1","unit1_pwr_1","beam_index_1"],"dev_val headers changed")
            for rn,row in enumerate(r,start=2):
                raw=resolve(csvp,row["unit1_lidar_1"])
                scr=resolve(csvp,row["unit1_lidar_SCR_1"])
                y1=int(float(row["beam_index_1"]))
                req(1<=y1<=64,f"official dev_val label outside 1..64: {y1}")
                y0=y1-1
                req(raw.is_file() and scr.is_file(),"missing dev_val LiDAR")
                sid=hashlib.sha256(
                    f"{sc}|CALIBRATION_CORRECTED|{rn}|{raw}|{scr}|{y1}|{y0}".encode()
                ).hexdigest()[:16]
                out.append((sc,csvp,rn,raw,scr,y1,y0,sid))
                n+=1
        counts[str(sc)]=n
    req(counts=={"8":775,"9":1172} and len(out)==1947,f"calibration counts changed: {counts}")
    return out,counts

def lse(x):
    m=x.max(axis=1,keepdims=True)
    return (m+np.log(np.exp(x-m).sum(axis=1,keepdims=True))).squeeze(1)

def nll(logits,y,T):
    z=logits/float(T)
    return float(np.mean(lse(z)-z[np.arange(len(y)),y]))

def softmax(logits,T):
    z=logits/float(T); z-=z.max(axis=1,keepdims=True)
    e=np.exp(z); return e/e.sum(axis=1,keepdims=True)

def brier(p,y):
    q=np.zeros_like(p); q[np.arange(len(y)),y]=1.0
    return float(np.mean(np.sum((p-q)**2,axis=1)))

def ece15(p,y):
    conf=p.max(axis=1); pred=p.argmax(axis=1); cor=(pred==y).astype(np.float64)
    s=0.0
    for b in range(15):
        lo=b/15; hi=(b+1)/15
        mask=(conf>=lo)&(conf<(hi) if b<14 else conf<=hi)
        if mask.any():
            s+=(mask.sum()/len(y))*abs(float(cor[mask].mean())-float(conf[mask].mean()))
    return float(s)

def fitT(logits,y):
    a=math.log(.05); b=math.log(20.0); phi=(1+math.sqrt(5))/2
    c=b-(b-a)/phi; d=a+(b-a)/phi
    fc=nll(logits,y,math.exp(c)); fd=nll(logits,y,math.exp(d))
    for _ in range(200):
        if fc<=fd:
            b=d; d=c; fd=fc; c=b-(b-a)/phi; fc=nll(logits,y,math.exp(c))
        else:
            a=c; c=d; fc=fd; d=a+(b-a)/phi; fd=nll(logits,y,math.exp(d))
    x=(a+b)/2
    return math.exp(x),nll(logits,y,math.exp(x))

def main():
    print("="*92)
    print("STAGE 8 — BLOCK 8.6C")
    print("CORRECTED PROBABILITY CALIBRATION — DEV_VAL ONLY")
    print("LABEL ADAPTER: beam_index_1 - 1")
    print("="*92)

    req(Path(__file__).resolve()==SCRIPT.resolve(),"runner path mismatch")
    req(CONTRACT.is_file() and sha(CONTRACT)==EXPECTED_CONTRACT,"contract SHA mismatch")
    for raw,e in EXPECTED.items():
        p=Path(raw); req(p.is_file() and sha(p)==e,f"frozen upstream changed: {p}")
    for p in [CAL,INDEX,DIAG,MANIFEST,REPORT,SEAL]:
        req(not p.exists(),f"8.6C output exists: {p}")

    s85=json.loads(SEAL85C.read_text())
    req(s85.get("status")=="FROZEN_COMPLETE_BLOCK85C","8.5C status changed")
    req(s85.get("Block86C_may_start") is True,"8.6C not authorized")
    req(s85.get("formal_data_access") is False and s85.get("formal_power_access") is False,"formal boundary changed")
    mp=json.loads(MAP85C.read_text())
    req(mp.get("label_minus_1_matches_power_max")==7030,"frozen mapping evidence changed")

    print("\n===== A. CORRECTED CHECKPOINT / LABEL ADAPTER GATE =====")
    print("corrected checkpoint identity = PASS")
    print("TRAIN mapping y-1 evidence = 7,030/7,030 PASS")
    print("formal data/power access = NO")

    recs,counts=records()
    fm=json.loads(B82B_FILES.read_text())
    hmap={x["relative_path"]:x["sha256"] for x in fm["files"]}

    # Verify only CSV + LiDAR bytes. Do not open power payloads.
    paths={r[1] for r in recs} | {r[3] for r in recs} | {r[4] for r in recs}
    def chk(p):
        rel=str(p.relative_to(DATA_ROOT))
        req(hmap.get(rel)==sha(p),f"frozen calibration byte mismatch: {rel}")
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(chk,sorted(paths)))

    print("\n===== B. DEV_VAL CORRECTED INDEX / LiDAR LOAD =====")
    loader=mod("mat5",LOADER_SRC); modelm=mod("enc",MODEL_SRC)
    raw=np.empty((1947,460,2),np.float32)
    scr=np.empty((1947,216,2),np.float32)
    y=np.empty(1947,np.int64)

    def one(t):
        i,r=t
        a=loader.load_mat5_double_matrix(r[3],"data",(460,2)).astype(np.float32)
        b=loader.load_mat5_double_matrix(r[4],"data",(216,2)).astype(np.float32)
        return i,a,b,r[6]

    with ThreadPoolExecutor(max_workers=12) as ex:
        for i,a,b,yy in ex.map(one,enumerate(recs),chunksize=16):
            raw[i]=a; scr[i]=b; y[i]=yy

    atom_json(INDEX,{
        "stage":8,"block":"8.6C","status":"FROZEN_CORRECTED_CALIBRATION_SAMPLE_INDEX",
        "row_count":1947,"scenario_counts":counts,
        "label_adapter":"beam_index_1 - 1",
        "samples":[{
            "sample_id":r[7],"scenario":r[0],"row_number":r[2],
            "official_label_1based":r[5],"model_target_0based":r[6],
            "raw_relative_path":str(r[3].relative_to(DATA_ROOT)),
            "scr_relative_path":str(r[4].relative_to(DATA_ROOT)),
        } for r in recs],
        "mmwave_power_payloads_opened":False,
        "formal_data_access":False,
    })
    print("Scenario8 dev_val =",counts["8"])
    print("Scenario9 dev_val =",counts["9"])
    print("pooled calibration rows =",len(recs))
    print("calibration mmWave power payloads opened = NO")

    norm=json.loads(NORM.read_text())
    rm=np.asarray(norm["raw_mean"],np.float32); rs=np.asarray(norm["raw_std"],np.float32)
    sm=np.asarray(norm["scr_mean"],np.float32); ss=np.asarray(norm["scr_std"],np.float32)
    raw=(raw-rm)/np.maximum(rs,1e-8)
    scr=(scr-sm)/np.maximum(ss,1e-8)
    req(np.isfinite(raw).all() and np.isfinite(scr).all(),"nonfinite calibrated inputs")

    print("\n===== C. FROZEN CORRECTED MODEL INFERENCE =====")
    ck=torch.load(CKPT,map_location="cpu",weights_only=False)
    model=modelm.DualLiDARConv1D64()
    model.load_state_dict(ck["state_dict"],strict=True); model.eval()
    torch.set_num_threads(12)
    logits=[]
    with torch.inference_mode():
        for i in range(0,1947,256):
            a=torch.from_numpy(np.ascontiguousarray(raw[i:i+256].transpose(0,2,1)))
            b=torch.from_numpy(np.ascontiguousarray(scr[i:i+256].transpose(0,2,1)))
            z=model(a,b).cpu().numpy().astype(np.float64)
            req(np.isfinite(z).all(),"nonfinite calibration logits")
            logits.append(z)
    logits=np.concatenate(logits,axis=0)
    req(logits.shape==(1947,64),"logit shape changed")
    print("logits shape = 1947x64 PASS")
    print("model parameters modified = NO")

    print("\n===== D. FIXED SCALAR TEMPERATURE FIT =====")
    T,after=fitT(logits,y)
    p0=softmax(logits,1.0); p1=softmax(logits,T)
    acc0=float(np.mean(p0.argmax(1)==y)); acc1=float(np.mean(p1.argmax(1)==y))
    req(acc0==acc1,"positive scalar temperature changed Top-1")
    d={
        "temperature":T,
        "nll_before":nll(logits,y,1.0),"nll_after":after,
        "brier_before":brier(p0,y),"brier_after":brier(p1,y),
        "ece15_before":ece15(p0,y),"ece15_after":ece15(p1,y),
        "top1_before":acc0,"top1_after":acc1,
    }
    print("fitted temperature =",T)
    print("calibration NLL before/after =",d["nll_before"],d["nll_after"])
    print("calibration ECE15 before/after =",d["ece15_before"],d["ece15_after"])
    print("calibration Top-1 before/after =",acc0,acc1)

    atom_json(CAL,{
        "stage":8,"block":"8.6C","status":"FROZEN_CORRECTED_SCALAR_TEMPERATURE_CALIBRATOR",
        "temperature":T,"formula":"softmax(logits / temperature)",
        "label_adapter":"beam_index_1 - 1",
        "fit_split":"official dev_val pooled Scenario8+9","fit_rows":1947,
        "objective":"mean multiclass NLL","optimizer":"golden-section log(T), 200 iterations, T in [0.05,20]",
        "checkpoint_sha256":EXPECTED[str(CKPT)],
        "model_parameters_modified":False,"mmwave_power_payloads_opened":False,"formal_data_used":False,
    })
    atom_json(DIAG,{
        "stage":8,"block":"8.6C","status":"CORRECTED_CALIBRATION_DIAGNOSTICS",
        **d,"ece_bins":15,"diagnostics_used_to_change_method":False,
        "mmwave_power_payloads_opened":False,"formal_data_used":False,
    })
    atom_json(MANIFEST,{
        "stage":8,"block":"8.6C","status":"FROZEN_COMPLETE_BLOCK86C",
        "contract_sha256":EXPECTED_CONTRACT,"Block85C_seal_sha256":EXPECTED[str(SEAL85C)],
        "checkpoint_sha256":EXPECTED[str(CKPT)],"mapping_sha256":EXPECTED[str(MAP85C)],
        "normalization_sha256":EXPECTED[str(NORM)],"runner_sha256":sha(SCRIPT),
        "calibrator_sha256":sha(CAL),"calibration_index_sha256":sha(INDEX),"diagnostics_sha256":sha(DIAG),
        "formal_data_access":False,"formal_power_access":False,
    })
    atom_json(REPORT,{
        "stage":8,"block":"8.6C","status":"PASS_BLOCK86C_CORRECTED_PROBABILITY_CALIBRATION",
        "calibration_rows":1947,"scenario8_rows":775,"scenario9_rows":1172,
        "label_adapter":"beam_index_1 - 1","temperature":T,
        "nll_before":d["nll_before"],"nll_after":d["nll_after"],
        "ece15_before":d["ece15_before"],"ece15_after":d["ece15_after"],"top1":acc1,
        "model_retrained":False,"calibration_mmwave_power_access":False,"formal_data_access":False,
        "Block87C_authorized":True,
        "calibrator_sha256":sha(CAL),"index_sha256":sha(INDEX),"diagnostics_sha256":sha(DIAG),
        "manifest_sha256":sha(MANIFEST),"runner_sha256":sha(SCRIPT),
    })
    atom_json(SEAL,{
        "stage":8,"block":"8.6C","status":"FROZEN_COMPLETE_BLOCK86C",
        "Block85C_seal_sha256":EXPECTED[str(SEAL85C)],"Block84_seal_sha256":EXPECTED[str(B84_SEAL)],
        "contract_sha256":EXPECTED_CONTRACT,"checkpoint_sha256":EXPECTED[str(CKPT)],
        "mapping_sha256":EXPECTED[str(MAP85C)],"runner_sha256":sha(SCRIPT),
        "calibrator_sha256":sha(CAL),"calibration_index_sha256":sha(INDEX),
        "diagnostics_sha256":sha(DIAG),"manifest_sha256":sha(MANIFEST),"report_sha256":sha(REPORT),
        "formal_data_access":False,"formal_power_access":False,"Block87C_may_start":True,
    })

    for raw,e in EXPECTED.items():
        req(sha(Path(raw))==e,f"upstream modified during 8.6C: {raw}")

    print("\n"+"="*92)
    print("BLOCK 8.6C = FULLY VERIFIED / FROZEN")
    print("CORRECTED LABEL ADAPTER = beam_index_1 - 1")
    print("CALIBRATION SPLIT = OFFICIAL dev_val ONLY")
    print("CALIBRATION ROWS = 1,947 (775 + 1,172)")
    print("CALIBRATOR = SCALAR TEMPERATURE SCALING")
    print("MODEL RETRAINING / PARAMETER UPDATE = NO")
    print("CALIBRATION MMWAVE POWER PAYLOADS OPENED = NO")
    print("FORMAL dev_test / FORMAL POWER ACCESS = NO")
    print("FROZEN TEMPERATURE =",T)
    print("OLD 8.6 / 8.7 PRESERVED APPEND-ONLY")
    print("BLOCK 8.7C MAY START = YES")
    print("8.6C calibrator SHA256 =",sha(CAL))
    print("8.6C index SHA256 =",sha(INDEX))
    print("8.6C diagnostics SHA256 =",sha(DIAG))
    print("8.6C manifest SHA256 =",sha(MANIFEST))
    print("8.6C report SHA256 =",sha(REPORT))
    print("8.6C seal SHA256 =",sha(SEAL))
    print("8.6C runner SHA256 =",sha(SCRIPT))
    print("STATUS = FROZEN_COMPLETE_BLOCK86C")
    print("="*92)

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print("\n"+"!"*92)
        print("BLOCK 8.6C FAIL-CLOSED")
        print(f"{type(e).__name__}: {e}")
        print("DO NOT START 8.7C OR FORMAL EVALUATION.")
        print("!"*92)
        raise
