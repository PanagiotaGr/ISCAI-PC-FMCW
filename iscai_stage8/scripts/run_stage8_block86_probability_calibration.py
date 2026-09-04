#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, importlib.util, json, math, os, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/agni/waymo")
S8 = ROOT / "iscai_stage8"
DATA_ROOT = S8 / "data/block82b_deepsense_primary"

SCRIPT = S8 / "scripts/run_stage8_block86_probability_calibration.py"
CONTRACT = S8 / "configs/stage8_block86_probability_calibration_contract.json"
MODEL_SRC = S8 / "src/deepsense_lidar_encoder.py"
LOADER_SRC = S8 / "src/mat5_lidar_loader.py"

B85B = S8 / "artifacts/block85b_encoder_training_cpu_repair"
CHECKPOINT = B85B / "duallidarconv1d64_epoch080.pt"
NORM85B = B85B / "train_only_normalization.json"
TRAINIDX85B = B85B / "frozen_train_sample_index.json"
LOG85B = B85B / "training_log.json"
SEAL85B = B85B / "stage8_block85b_encoder_training_cpu_repair_seal.json"
REPORT85B = S8 / "reports/stage8_block85b_encoder_training_cpu_repair_report.json"
RUNNER85B = S8 / "scripts/run_stage8_block85b_cpu_deterministic_repair.py"

B82B_FILES = S8 / "artifacts/block82b_canonical_corpus_schema_split/canonical_extracted_file_manifest.json"
B84_POLICY = S8 / "src/deepsense_beam_policy.py"
B84_SEAL = S8 / "artifacts/block84_beam_policy_interface/stage8_block84_beam_policy_interface_seal.json"

OUT = S8 / "artifacts/block86_probability_calibration"
CAL = OUT / "frozen_temperature_calibrator.json"
INDEX = OUT / "calibration_sample_index.json"
DIAG = OUT / "calibration_diagnostics.json"
MANIFEST = OUT / "stage8_block86_manifest.json"
REPORT = S8 / "reports/stage8_block86_probability_calibration_report.json"
SEAL = OUT / "stage8_block86_probability_calibration_seal.json"

EXPECTED_CONTRACT = "4a1d99ceb55e5a6a02d951cf1034e6c416246d50dcc7d77b2a59b44d74cdd6f2"
EXPECTED = {
 str(CHECKPOINT): "a31e411a4c729b9ff99c20295f849112417420d301296d3f6e6f73d1a1ac99b5",
 str(NORM85B): "8e12e7b5bccf57c31075569a575507cd95e4b1710f5bb7327cbdd203c36575fd",
 str(TRAINIDX85B): "cf2905af5711d633a0e6f9e277c03c25028c3aede2a392f09efc9e9f56a75a21",
 str(LOG85B): "b16a3217e9ec2c54862b0390540681c21b2de52c3e0a760d74799680309ecae5",
 str(REPORT85B): "619b81a0bc425006c0856263f1e4b7cb552763433103bdfe84da486377c82aba",
 str(SEAL85B): "ee9e5a8d7f4d630e5334f5f5089484e6016f35e38316346edb5a3194af74ec3b",
 str(RUNNER85B): "b7e727048a2afcaf0fadb3c488c4e8562b33a5fd5cba6fa63a14b705ef4dd368",
 str(B82B_FILES): "79dc92fb6a90acd36c4598de436517ebac6c7e7904334dd76a520e1c90a0b895",
 str(B84_POLICY): "5b7e6c3c4ff7eca4d1db86df90edfa19f0bbfc551255779833485df074d87cb8",
 str(B84_SEAL): "c5dcbf5b1bf8d7965fd32bf794f095b4495d30d98e35bddef22ce3abffe554e2",
 str(MODEL_SRC): "adb943f12490612a6b92ffa3e70f69b96b1002fa77e4e769f5ea26ed6d25c417",
 str(LOADER_SRC): "b4b3d9e09a20a5ee790a4ad7fd22f53dad1d6f4b953e70f824ade80f9b86299c",
}

VAL_SPECS = [
 (8, DATA_ROOT / "LiDAR/Scenario8/development_dataset/scenario8_dev_val.csv"),
 (9, DATA_ROOT / "LiDAR/Scenario9/development_dataset/scenario9_dev_val.csv"),
]

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

def stable_id(sc,csv_path,row_number,raw,scr):
    payload="|".join([str(sc),"CALIBRATION",str(csv_path.relative_to(DATA_ROOT)),str(row_number),
                      str(raw.relative_to(DATA_ROOT)),str(scr.relative_to(DATA_ROOT))])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def build_records():
    rows=[]; counts={}
    for sc,csv_path in VAL_SPECS:
        require(csv_path.is_file(),f"Missing calibration CSV: {csv_path}")
        n=0
        with csv_path.open(newline="",encoding="utf-8-sig") as f:
            r=csv.DictReader(f)
            require(r.fieldnames==['index','unit1_lidar_1','unit1_lidar_SCR_1','unit1_pwr_1','beam_index_1'],
                    f"Unexpected headers: {r.fieldnames}")
            for rn,row in enumerate(r,start=2):
                raw=resolve_ref(csv_path,row["unit1_lidar_1"])
                scr=resolve_ref(csv_path,row["unit1_lidar_SCR_1"])
                y=int(float(row["beam_index_1"]))
                require(0<=y<=63,f"Calibration label outside 0..63 at {csv_path}:{rn}")
                require(raw.is_file() and scr.is_file(),f"Missing calibration LiDAR at {csv_path}:{rn}")
                rows.append({"scenario":sc,"csv":csv_path,"row":rn,"raw":raw,"scr":scr,"label":y,
                             "sample_id":stable_id(sc,csv_path,rn,raw,scr)})
                n+=1
        counts[str(sc)]=n
    require(counts=={"8":775,"9":1172},f"Calibration counts changed: {counts}")
    require(len(rows)==1947,"Expected 1947 calibration rows")
    require(len({r["sample_id"] for r in rows})==1947,"Duplicate calibration IDs")
    return rows,counts

def frozen_hash_map():
    d=json.loads(B82B_FILES.read_text())
    return {x["relative_path"]:x["sha256"] for x in d["files"]}

def verify_inputs(records,frozen):
    paths={p for _,p in VAL_SPECS}
    for r in records: paths.update([r["raw"],r["scr"]])
    def one(p):
        rel=str(p.relative_to(DATA_ROOT))
        require(frozen.get(rel)==sha(p),f"Frozen calibration input SHA mismatch: {rel}")
        return rel
    workers=max(1,min(os.cpu_count() or 1,32))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        checked=list(ex.map(one,sorted(paths)))
    return len(checked),workers

def load_arrays(records,loader):
    n=len(records)
    raw=np.empty((n,460,2),np.float32); scr=np.empty((n,216,2),np.float32); y=np.empty(n,np.int64)
    def one(item):
        i,r=item
        a=loader.load_mat5_double_matrix(r["raw"],expected_name="data",expected_shape=(460,2))
        b=loader.load_mat5_double_matrix(r["scr"],expected_name="data",expected_shape=(216,2))
        return i,a.astype(np.float32,copy=False),b.astype(np.float32,copy=False),r["label"]
    workers=max(1,min(os.cpu_count() or 1,32))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i,a,b,yy in ex.map(one,enumerate(records),chunksize=16):
            raw[i]=a; scr[i]=b; y[i]=yy
    return raw,scr,y,workers

def logsumexp_np(x,axis=1):
    m=np.max(x,axis=axis,keepdims=True)
    return (m + np.log(np.exp(x-m).sum(axis=axis,keepdims=True))).squeeze(axis)

def nll(logits,y,T):
    z=logits/float(T)
    return float(np.mean(logsumexp_np(z)-z[np.arange(len(y)),y]))

def softmax(logits,T):
    z=logits/float(T); z=z-z.max(axis=1,keepdims=True)
    e=np.exp(z); return e/e.sum(axis=1,keepdims=True)

def brier(probs,y):
    one=np.zeros_like(probs); one[np.arange(len(y)),y]=1.0
    return float(np.mean(np.sum((probs-one)**2,axis=1)))

def ece15(probs,y):
    conf=probs.max(axis=1); pred=probs.argmax(axis=1); corr=(pred==y).astype(np.float64)
    total=len(y); out=0.0
    for b in range(15):
        lo=b/15; hi=(b+1)/15
        mask=(conf>=lo)&(conf<hi if b<14 else conf<=hi)
        if mask.any():
            out += (mask.sum()/total)*abs(float(corr[mask].mean())-float(conf[mask].mean()))
    return float(out)

def golden_fit(logits,y):
    # Fixed 200-iteration golden-section search over log(T) in [ln(.05), ln(20)].
    a=math.log(0.05); b=math.log(20.0)
    phi=(1+math.sqrt(5.0))/2.0
    c=b-(b-a)/phi; d=a+(b-a)/phi
    fc=nll(logits,y,math.exp(c)); fd=nll(logits,y,math.exp(d))
    for _ in range(200):
        if fc <= fd:
            b=d; d=c; fd=fc; c=b-(b-a)/phi; fc=nll(logits,y,math.exp(c))
        else:
            a=c; c=d; fc=fd; d=a+(b-a)/phi; fd=nll(logits,y,math.exp(d))
    x=(a+b)/2.0
    return math.exp(x), nll(logits,y,math.exp(x))

def main():
    print("="*88)
    print("STAGE 8 — BLOCK 8.6")
    print("DEEPSENSE PROBABILITY CALIBRATION — DEV_VAL ONLY")
    print("NO RETRAINING / NO FORMAL / NO MMWAVE POWER ACCESS")
    print("="*88)

    require(Path(__file__).resolve()==SCRIPT.resolve(),"Runner path mismatch")
    require(CONTRACT.is_file() and sha(CONTRACT)==EXPECTED_CONTRACT,"8.6 contract SHA mismatch")
    for raw,expected in EXPECTED.items():
        p=Path(raw); require(p.is_file(),f"Missing frozen input: {p}"); require(sha(p)==expected,f"Frozen SHA changed: {p}")
    for p in (CAL,INDEX,DIAG,MANIFEST,REPORT,SEAL):
        require(not p.exists(),f"8.6 output exists: {p}")

    s85=json.loads(SEAL85B.read_text())
    require(s85.get("status")=="FROZEN_COMPLETE_BLOCK85B","8.5B canonical status changed")
    require(s85.get("Block86_may_start") is True,"8.6 not authorized")
    require(s85.get("formal_power_access") is False,"Formal power boundary crossed")

    print("\n===== A. CANONICAL 8.5B / PRE-FORMAL GATE =====")
    print("canonical CPU-deterministic checkpoint identity = PASS")
    print("formal data/power access = NO")

    model_mod=load_module("ds_model",MODEL_SRC)
    loader_mod=load_module("mat5",LOADER_SRC)

    print("\n===== B. CALIBRATION INDEX / BYTE GATE =====")
    records,counts=build_records()
    checked,workers=verify_inputs(records,frozen_hash_map())
    print("Scenario8 dev_val =",counts["8"])
    print("Scenario9 dev_val =",counts["9"])
    print("pooled calibration rows =",len(records))
    print("calibration CSV/LiDAR files SHA-verified =",checked)
    print("parallel hash workers =",workers)
    print("unit1_pwr_1 payload files opened = NO")

    atomic_json(INDEX,{
        "stage":8,"block":"8.6","status":"FROZEN_CALIBRATION_SAMPLE_INDEX",
        "counts":counts,"row_count":len(records),
        "samples":[{"sample_id":r["sample_id"],"scenario":r["scenario"],
                    "csv_relative_path":str(r["csv"].relative_to(DATA_ROOT)),
                    "row_number":r["row"],
                    "raw_relative_path":str(r["raw"].relative_to(DATA_ROOT)),
                    "scr_relative_path":str(r["scr"].relative_to(DATA_ROOT)),
                    "beam_label":r["label"]} for r in records],
        "mmwave_power_payloads_opened":False,
    })

    print("\n===== C. FROZEN MODEL INFERENCE ON DEV_VAL =====")
    raw,scr,y,load_workers=load_arrays(records,loader_mod)
    norm=json.loads(NORM85B.read_text())
    rm=np.asarray(norm["raw_mean"],np.float32); rs=np.asarray(norm["raw_std"],np.float32)
    sm=np.asarray(norm["scr_mean"],np.float32); ss=np.asarray(norm["scr_std"],np.float32)
    raw=(raw-rm)/np.maximum(rs,1e-8); scr=(scr-sm)/np.maximum(ss,1e-8)
    require(np.isfinite(raw).all() and np.isfinite(scr).all(),"Non-finite normalized calibration LiDAR")

    ckpt=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
    model=model_mod.DualLiDARConv1D64()
    model.load_state_dict(ckpt["state_dict"],strict=True)
    model.eval()
    torch.set_num_threads(max(1,min(os.cpu_count() or 1,12)))
    logits=[]
    with torch.inference_mode():
        for i in range(0,len(y),256):
            a=torch.from_numpy(np.ascontiguousarray(raw[i:i+256].transpose(0,2,1)))
            b=torch.from_numpy(np.ascontiguousarray(scr[i:i+256].transpose(0,2,1)))
            z=model(a,b).cpu().numpy().astype(np.float64)
            require(np.isfinite(z).all(),"Non-finite calibration logits")
            logits.append(z)
    logits=np.concatenate(logits,axis=0)
    require(logits.shape==(1947,64),f"Logit shape changed: {logits.shape}")
    print("logits shape = 1947x64 PASS")
    print("parallel MAT load workers =",load_workers)
    print("model parameters modified = NO")

    print("\n===== D. FIXED SCALAR TEMPERATURE FIT =====")
    T,fit_nll=golden_fit(logits,y)
    require(math.isfinite(T) and 0.05 <= T <= 20.0,f"Invalid fitted T={T}")
    p0=softmax(logits,1.0); p1=softmax(logits,T)
    acc0=float(np.mean(p0.argmax(1)==y)); acc1=float(np.mean(p1.argmax(1)==y))
    require(acc0==acc1,"Positive scalar temperature changed Top-1 predictions")
    diag={
        "temperature":T,
        "nll_before":nll(logits,y,1.0),"nll_after":fit_nll,
        "brier_before":brier(p0,y),"brier_after":brier(p1,y),
        "ece15_before":ece15(p0,y),"ece15_after":ece15(p1,y),
        "top1_before":acc0,"top1_after":acc1,
    }
    print("fitted temperature =",T)
    print("calibration NLL before/after =",diag["nll_before"],diag["nll_after"])
    print("calibration ECE15 before/after =",diag["ece15_before"],diag["ece15_after"])
    print("calibration Top-1 before/after =",acc0,acc1)

    atomic_json(CAL,{
        "stage":8,"block":"8.6","status":"FROZEN_SCALAR_TEMPERATURE_CALIBRATOR",
        "temperature":T,"formula":"softmax(logits / temperature)",
        "fit_split":"official dev_val pooled Scenario8+9",
        "fit_rows":1947,"objective":"mean multiclass NLL",
        "optimizer":"golden-section over log(T), 200 iterations, T in [0.05,20]",
        "model_checkpoint_sha256":EXPECTED[str(CHECKPOINT)],
        "model_parameters_modified":False,
        "formal_data_used":False,"mmwave_power_payloads_opened":False,
    })
    atomic_json(DIAG,{
        "stage":8,"block":"8.6","status":"CALIBRATION_DIAGNOSTICS_DESCRIPTIVE_ONLY",
        **diag,"ece_bins":15,
        "diagnostics_used_to_change_method":False,
        "formal_data_used":False,
    })
    atomic_json(MANIFEST,{
        "stage":8,"block":"8.6","status":"FROZEN_COMPLETE_BLOCK86",
        "contract_sha256":EXPECTED_CONTRACT,
        "checkpoint_sha256":EXPECTED[str(CHECKPOINT)],
        "runner_sha256":sha(SCRIPT),
        "calibrator_sha256":sha(CAL),
        "calibration_index_sha256":sha(INDEX),
        "diagnostics_sha256":sha(DIAG),
        "retraining":False,"formal_data_access":False,"mmwave_power_access":False,
    })
    atomic_json(REPORT,{
        "stage":8,"block":"8.6","status":"PASS_BLOCK86_PROBABILITY_CALIBRATION",
        "calibration_rows":1947,"scenario8_rows":775,"scenario9_rows":1172,
        "calibrator":"scalar temperature scaling",
        "temperature":T,
        "nll_before":diag["nll_before"],"nll_after":diag["nll_after"],
        "ece15_before":diag["ece15_before"],"ece15_after":diag["ece15_after"],
        "top1":acc1,
        "model_retrained":False,
        "calibration_mmwave_power_access":False,
        "formal_data_access":False,
        "Block87_authorized":True,
        "calibrator_sha256":sha(CAL),"diagnostics_sha256":sha(DIAG),
        "manifest_sha256":sha(MANIFEST),"runner_sha256":sha(SCRIPT),
    })
    atomic_json(SEAL,{
        "stage":8,"block":"8.6","status":"FROZEN_COMPLETE_BLOCK86",
        "Block85B_seal_sha256":EXPECTED[str(SEAL85B)],
        "Block84_seal_sha256":EXPECTED[str(B84_SEAL)],
        "contract_sha256":EXPECTED_CONTRACT,
        "checkpoint_sha256":EXPECTED[str(CHECKPOINT)],
        "runner_sha256":sha(SCRIPT),
        "calibrator_sha256":sha(CAL),
        "calibration_index_sha256":sha(INDEX),
        "diagnostics_sha256":sha(DIAG),
        "manifest_sha256":sha(MANIFEST),
        "report_sha256":sha(REPORT),
        "formal_data_access":False,"formal_power_access":False,
        "Block87_may_start":True,
    })

    for raw,expected in EXPECTED.items():
        require(sha(Path(raw))==expected,f"Frozen upstream modified during 8.6: {raw}")

    print("\n"+"="*88)
    print("BLOCK 8.6 = FULLY VERIFIED / FROZEN")
    print("CALIBRATION SPLIT = OFFICIAL dev_val ONLY")
    print("CALIBRATION ROWS = 1,947 (775 + 1,172)")
    print("CALIBRATOR = SCALAR TEMPERATURE SCALING")
    print("MODEL RETRAINING / PARAMETER UPDATE = NO")
    print("CALIBRATION MMWAVE POWER PAYLOADS OPENED = NO")
    print("FORMAL dev_test / POWER ACCESS = NO")
    print("FROZEN TEMPERATURE =",T)
    print("BLOCK 8.7 MAY START = YES")
    print("8.6 calibrator SHA256 =",sha(CAL))
    print("8.6 index SHA256      =",sha(INDEX))
    print("8.6 diagnostics SHA256 =",sha(DIAG))
    print("8.6 manifest SHA256   =",sha(MANIFEST))
    print("8.6 report SHA256     =",sha(REPORT))
    print("8.6 seal SHA256       =",sha(SEAL))
    print("8.6 runner SHA256     =",sha(SCRIPT))
    print("STATUS = FROZEN_COMPLETE_BLOCK86")
    print("="*88)
    return 0

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("\n"+"!"*88)
        print("BLOCK 8.6 FAIL-CLOSED")
        print(f"{type(e).__name__}: {e}")
        print("DO NOT ACCESS FORMAL dev_test OR FORMAL MMWAVE POWER.")
        print("DO NOT START BLOCK8.7/8.8.")
        print("!"*88)
        raise
