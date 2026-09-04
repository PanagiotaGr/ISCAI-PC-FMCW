#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, os, random, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path("/home/agni/waymo")
S8=ROOT/"iscai_stage8"
DATA_ROOT=S8/"data/block82b_deepsense_primary"

SCRIPT=S8/"scripts/run_stage8_block85c_label_index_corrected_training.py"
CONTRACT=S8/"configs/stage8_block85c_label_index_correction_contract.json"
MODEL_SRC=S8/"src/deepsense_lidar_encoder.py"
LOADER_SRC=S8/"src/mat5_lidar_loader.py"

B85A_SEAL=S8/"artifacts/block85a_encoder_preregistration/stage8_block85a_encoder_preregistration_seal.json"
B85B=S8/"artifacts/block85b_encoder_training_cpu_repair"
OLD_NORM=B85B/"train_only_normalization.json"
OLD_CKPT=B85B/"duallidarconv1d64_epoch080.pt"
OLD_SEAL=B85B/"stage8_block85b_encoder_training_cpu_repair_seal.json"
B82B_FILES=S8/"artifacts/block82b_canonical_corpus_schema_split/canonical_extracted_file_manifest.json"
B82B_SEAL=S8/"artifacts/block82b_canonical_corpus_schema_split/stage8_block82b_canonical_corpus_schema_split_seal.json"
B88_FAILED_RUNNER=S8/"scripts/run_stage8_block88_formal_measured_deepsense_evaluation.py"

OUT=S8/"artifacts/block85c_label_index_corrected_training"
MAPPING=OUT/"train_label_index_mapping_gate.json"
INDEX=OUT/"corrected_train_sample_index.json"
LOG=OUT/"training_log.json"
CKPT=OUT/"duallidarconv1d64_label0based_epoch080.pt"
MANIFEST=OUT/"stage8_block85c_manifest.json"
REPORT=S8/"reports/stage8_block85c_label_index_corrected_training_report.json"
SEAL=OUT/"stage8_block85c_label_index_corrected_training_seal.json"

EXPECTED_CONTRACT="1a9474d9e1fcd248d522a51d9e03dbee8d843b91feaca9df837cf608615c7c58"
EXPECTED={
 str(MODEL_SRC):"adb943f12490612a6b92ffa3e70f69b96b1002fa77e4e769f5ea26ed6d25c417",
 str(LOADER_SRC):"b4b3d9e09a20a5ee790a4ad7fd22f53dad1d6f4b953e70f824ade80f9b86299c",
 str(B85A_SEAL):"2d43c234dbe42b86f3f3f58a57d582c9ea92253fbe62dd3091d4fd91cff98fb4",
 str(OLD_NORM):"8e12e7b5bccf57c31075569a575507cd95e4b1710f5bb7327cbdd203c36575fd",
 str(OLD_CKPT):"a31e411a4c729b9ff99c20295f849112417420d301296d3f6e6f73d1a1ac99b5",
 str(OLD_SEAL):"ee9e5a8d7f4d630e5334f5f5089484e6016f35e38316346edb5a3194af74ec3b",
 str(B82B_FILES):"79dc92fb6a90acd36c4598de436517ebac6c7e7904334dd76a520e1c90a0b895",
 str(B82B_SEAL):"c84eb7ef45d16a1feb3ef068bdaf498e5e4298514f3c4807ffe10db9b1112ba3",
 str(B88_FAILED_RUNNER):"d03582aba683b11a5f2e7c8d88d2ca2a742b36af2cf4596fae0d3c1ea2795cd5",
}
SEED=20260902

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

def atom_torch(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    req(not path.exists(),f"Refusing overwrite: {path}")
    fd,tmp=tempfile.mkstemp(prefix=path.name+".tmp.",dir=str(path.parent)); os.close(fd)
    try:
        torch.save(obj,tmp)
        with open(tmp,"rb") as f: os.fsync(f.fileno())
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

def power_vec(p):
    vals=[float(x) for x in p.read_text(encoding="utf-8",errors="strict").replace(","," ").split()]
    req(len(vals)==64,f"Expected 64 powers: {p}")
    req(all(math.isfinite(x) for x in vals),f"Nonfinite power: {p}")
    return vals

def records():
    out=[]; counts={}
    for sc,name in [(8,"scenario8_dev_train.csv"),(9,"scenario9_dev_train.csv")]:
        csvp=DATA_ROOT/f"LiDAR/Scenario{sc}/development_dataset/{name}"
        n=0
        with csvp.open(newline="",encoding="utf-8-sig") as f:
            r=csv.DictReader(f)
            req(r.fieldnames==["index","unit1_lidar_1","unit1_lidar_SCR_1","unit1_pwr_1","beam_index_1"],"TRAIN headers changed")
            for rn,row in enumerate(r,start=2):
                raw=resolve(csvp,row["unit1_lidar_1"])
                scr=resolve(csvp,row["unit1_lidar_SCR_1"])
                pwr=resolve(csvp,row["unit1_pwr_1"])
                y1=int(float(row["beam_index_1"]))
                req(1<=y1<=64,f"official TRAIN label outside 1..64: {y1}")
                y0=y1-1
                req(0<=y0<64,"corrected target outside 0..63")
                req(raw.is_file() and scr.is_file() and pwr.is_file(),"missing TRAIN referenced file")
                sid=hashlib.sha256(
                    f"{sc}|TRAIN_CORRECTED|{rn}|{raw}|{scr}|{pwr}|{y1}|{y0}".encode()
                ).hexdigest()[:16]
                out.append((sc,csvp,rn,raw,scr,pwr,y1,y0,sid))
                n+=1
        counts[str(sc)]=n
    req(counts=={"8":2831,"9":4199} and len(out)==7030,f"TRAIN counts changed: {counts}")
    return out,counts

def main():
    print("="*92)
    print("STAGE 8 — BLOCK 8.5C")
    print("1-BASED OFFICIAL LABEL -> 0-BASED MODEL/POWER INDEX CORRECTION")
    print("PREFORMAL CORRECTED FIXED-RECIPE RETRAINING")
    print("="*92)

    req(Path(__file__).resolve()==SCRIPT.resolve(),"runner path mismatch")
    req(CONTRACT.is_file() and sha(CONTRACT)==EXPECTED_CONTRACT,"contract SHA mismatch")
    for raw,e in EXPECTED.items():
        p=Path(raw); req(p.is_file() and sha(p)==e,f"frozen upstream changed: {p}")
    for p in [MAPPING,INDEX,LOG,CKPT,MANIFEST,REPORT,SEAL]:
        req(not p.exists(),f"8.5C output exists: {p}")

    s85=json.loads(OLD_SEAL.read_text())
    req(s85.get("status")=="FROZEN_COMPLETE_BLOCK85B","old 8.5B status changed")
    print("\n===== A. TRAIN-ONLY LABEL INDEX MAPPING FREEZE =====")
    recs,counts=records()
    fm=json.loads(B82B_FILES.read_text())
    hmap={x["relative_path"]:x["sha256"] for x in fm["files"]}

    # Verify all referenced TRAIN bytes including powers before using the mapping.
    paths={r[1] for r in recs} | {r[3] for r in recs} | {r[4] for r in recs} | {r[5] for r in recs}
    def chk(p):
        rel=str(p.relative_to(DATA_ROOT))
        req(hmap.get(rel)==sha(p),f"frozen TRAIN byte mismatch: {rel}")
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(chk,sorted(paths)))

    def mapcheck(r):
        vals=power_vec(r[5]); y1=r[6]; y0=r[7]
        vmax=max(vals)
        maxima={i for i,x in enumerate(vals) if math.isclose(x,vmax,rel_tol=1e-12,abs_tol=1e-15)}
        return int(y0 in maxima), int(y1 in maxima if 0<=y1<64 else False)

    with ThreadPoolExecutor(max_workers=12) as ex:
        checks=list(ex.map(mapcheck,recs,chunksize=32))
    minus1=sum(x[0] for x in checks); identity=sum(x[1] for x in checks)
    req(minus1==7030 and identity==0,f"mapping audit changed: minus1={minus1} identity={identity}")

    atom_json(MAPPING,{
        "stage":8,"block":"8.5C","status":"FROZEN_LABEL_INDEX_MAPPING",
        "rows_checked":7030,"scenario_counts":counts,
        "identity_label_matches_power_max":identity,
        "label_minus_1_matches_power_max":minus1,
        "adapter":"class_index_0based = beam_index_1 - 1",
        "official_label_range_observed":[min(r[6] for r in recs),max(r[6] for r in recs)],
        "corrected_class_range_observed":[min(r[7] for r in recs),max(r[7] for r in recs)],
        "formal_data_access":False,"formal_power_access":False,
    })
    print("identity y -> power index =",identity,"/7030")
    print("y-1 -> power index =",minus1,"/7030 PASS")
    print("FROZEN ADAPTER = class_index_0based = beam_index_1 - 1")
    print("FORMAL DATA/POWER ACCESS = NO")

    print("\n===== B. CORRECTED TRAIN DATA PRELOAD =====")
    loader=mod("mat5",LOADER_SRC); modelm=mod("enc",MODEL_SRC)
    raw=np.empty((7030,460,2),np.float32)
    scr=np.empty((7030,216,2),np.float32)
    y=np.empty(7030,np.int64)

    def one(t):
        i,r=t
        a=loader.load_mat5_double_matrix(r[3],"data",(460,2)).astype(np.float32)
        b=loader.load_mat5_double_matrix(r[4],"data",(216,2)).astype(np.float32)
        return i,a,b,r[7]

    with ThreadPoolExecutor(max_workers=12) as ex:
        for i,a,b,yy in ex.map(one,enumerate(recs),chunksize=16):
            raw[i]=a; scr[i]=b; y[i]=yy

    norm=json.loads(OLD_NORM.read_text())
    rm=np.asarray(norm["raw_mean"],np.float32); rs=np.asarray(norm["raw_std"],np.float32)
    sm=np.asarray(norm["scr_mean"],np.float32); ss=np.asarray(norm["scr_std"],np.float32)
    raw=(raw-rm)/np.maximum(rs,1e-8)
    scr=(scr-sm)/np.maximum(ss,1e-8)
    req(np.isfinite(raw).all() and np.isfinite(scr).all(),"nonfinite normalized TRAIN data")

    atom_json(INDEX,{
        "stage":8,"block":"8.5C","status":"FROZEN_CORRECTED_TRAIN_SAMPLE_INDEX",
        "row_count":7030,"scenario_counts":counts,
        "label_adapter":"beam_index_1 - 1",
        "samples":[{
            "sample_id":r[8],"scenario":r[0],"row_number":r[2],
            "official_label_1based":r[6],"model_target_0based":r[7],
            "raw_relative_path":str(r[3].relative_to(DATA_ROOT)),
            "scr_relative_path":str(r[4].relative_to(DATA_ROOT)),
        } for r in recs],
        "normalization_sha256":EXPECTED[str(OLD_NORM)],
        "formal_data_access":False,
    })
    print("TRAIN rows =",len(y))
    print("corrected target min/max =",int(y.min()),int(y.max()))
    print("normalization reused exact SHA =",EXPECTED[str(OLD_NORM)])

    print("\n===== C. FIXED 80-EPOCH CORRECTED TRAINING =====")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(12)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass

    ds=TensorDataset(
        torch.from_numpy(np.ascontiguousarray(raw.transpose(0,2,1))),
        torch.from_numpy(np.ascontiguousarray(scr.transpose(0,2,1))),
        torch.from_numpy(y),
    )
    g=torch.Generator().manual_seed(SEED)
    dl=DataLoader(ds,batch_size=128,shuffle=True,generator=g,num_workers=0,drop_last=False)

    model=modelm.DualLiDARConv1D64()
    req(sum(p.numel() for p in model.parameters() if p.requires_grad)==18208,"parameter count changed")
    lossfn=torch.nn.CrossEntropyLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4,betas=(0.9,0.999))
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80,eta_min=1e-5)

    elog=[]
    for ep in range(1,81):
        model.train(); ls=0.0; cor=0; seen=0
        for a,b,yy in dl:
            opt.zero_grad(set_to_none=True)
            z=model(a,b); loss=lossfn(z,yy)
            req(torch.isfinite(z).all().item() and math.isfinite(float(loss.item())),"nonfinite training state")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            n=int(yy.shape[0]); ls+=float(loss.item())*n; cor+=int((z.argmax(1)==yy).sum()); seen+=n
        req(seen==7030,f"epoch sample count changed: {seen}")
        lr=float(opt.param_groups[0]["lr"]); sch.step()
        rec={"epoch":ep,"train_cross_entropy":ls/seen,"train_top1_accuracy":cor/seen,"learning_rate":lr}
        elog.append(rec)
        if ep==1 or ep%10==0 or ep==80:
            print(f"epoch={ep:03d}/080 train_ce={rec['train_cross_entropy']:.8f} train_top1={rec['train_top1_accuracy']:.6f} lr={lr:.8g}")

    atom_json(LOG,{
        "stage":8,"block":"8.5C","status":"FROZEN_FIXED_80_EPOCH_CORRECTED_TRAINING_LOG",
        "device":"cpu","seed":SEED,"epochs":elog,
        "label_adapter":"beam_index_1 - 1",
        "checkpoint_selection":"fixed final epoch 80 only",
        "validation_model_selection":False,
        "formal_data_used":False,
        "formal_power_access":False,
    })

    atom_torch(CKPT,{
        "stage":8,"block":"8.5C","epoch":80,
        "model_name":"DualLiDARConv1D64",
        "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "normalization_sha256":EXPECTED[str(OLD_NORM)],
        "label_adapter":"beam_index_1 - 1",
        "class_semantics":"0-based measured-power/codebook index",
        "training_recipe":"identical to frozen 8.5A/8.5B except corrected target indexing",
    })

    atom_json(MANIFEST,{
        "stage":8,"block":"8.5C","status":"FROZEN_COMPLETE_BLOCK85C",
        "contract_sha256":EXPECTED_CONTRACT,
        "mapping_sha256":sha(MAPPING),"corrected_index_sha256":sha(INDEX),
        "training_log_sha256":sha(LOG),"checkpoint_sha256":sha(CKPT),
        "normalization_sha256":EXPECTED[str(OLD_NORM)],
        "model_source_sha256":EXPECTED[str(MODEL_SRC)],
        "loader_source_sha256":EXPECTED[str(LOADER_SRC)],
        "runner_sha256":sha(SCRIPT),
        "formal_data_access":False,"formal_power_access":False,
    })

    atom_json(REPORT,{
        "stage":8,"block":"8.5C","status":"PASS_BLOCK85C_LABEL_INDEX_CORRECTED_TRAINING",
        "mapping_train_rows_passed":7030,
        "mapping":"beam_index_1 - 1",
        "scientific_recipe_change":"target-index correction only",
        "architecture_change":False,"optimizer_change":False,"normalization_change":False,
        "epochs_completed":80,"train_rows":7030,
        "final_train_cross_entropy":elog[-1]["train_cross_entropy"],
        "final_train_top1_accuracy":elog[-1]["train_top1_accuracy"],
        "checkpoint_sha256":sha(CKPT),
        "formal_data_access":False,"formal_power_access":False,
        "Block86C_authorized":True,
    })

    atom_json(SEAL,{
        "stage":8,"block":"8.5C","status":"FROZEN_COMPLETE_BLOCK85C",
        "Block85A_seal_sha256":EXPECTED[str(B85A_SEAL)],
        "old_Block85B_seal_sha256":EXPECTED[str(OLD_SEAL)],
        "old_Block85B_checkpoint_sha256":EXPECTED[str(OLD_CKPT)],
        "failed_Block88_runner_sha256":EXPECTED[str(B88_FAILED_RUNNER)],
        "contract_sha256":EXPECTED_CONTRACT,
        "mapping_sha256":sha(MAPPING),
        "corrected_index_sha256":sha(INDEX),
        "training_log_sha256":sha(LOG),
        "checkpoint_sha256":sha(CKPT),
        "manifest_sha256":sha(MANIFEST),
        "report_sha256":sha(REPORT),
        "runner_sha256":sha(SCRIPT),
        "formal_data_access":False,"formal_power_access":False,
        "Block86C_may_start":True,
    })

    for raw,e in EXPECTED.items():
        req(sha(Path(raw))==e,f"upstream modified during 8.5C: {raw}")

    print("\n"+"="*92)
    print("BLOCK 8.5C = FULLY VERIFIED / FROZEN")
    print("TRAIN LABEL MAPPING y-1 = 7,030/7,030 PASS")
    print("MODEL TARGET SEMANTICS = 0-BASED 64-WAY POWER/CODEBOOK INDEX")
    print("ARCHITECTURE / NORMALIZATION / OPTIMIZER / EPOCHS = UNCHANGED")
    print("FIXED CORRECTED TRAINING = 80/80 COMPLETE")
    print("CALIBRATION DATA ACCESSED = NO")
    print("FORMAL DATA / FORMAL POWER ACCESSED = NO")
    print("OLD 8.5B / 8.6 / 8.7 PRESERVED APPEND-ONLY")
    print("BLOCK 8.6C MAY START = YES")
    print("8.5C mapping SHA256 =",sha(MAPPING))
    print("8.5C index SHA256 =",sha(INDEX))
    print("8.5C training log SHA256 =",sha(LOG))
    print("8.5C checkpoint SHA256 =",sha(CKPT))
    print("8.5C manifest SHA256 =",sha(MANIFEST))
    print("8.5C report SHA256 =",sha(REPORT))
    print("8.5C seal SHA256 =",sha(SEAL))
    print("8.5C runner SHA256 =",sha(SCRIPT))
    print("STATUS = FROZEN_COMPLETE_BLOCK85C")
    print("="*92)

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print("\n"+"!"*92)
        print("BLOCK 8.5C FAIL-CLOSED")
        print(f"{type(e).__name__}: {e}")
        print("DO NOT START 8.6C OR FORMAL EVALUATION.")
        print("!"*92)
        raise
