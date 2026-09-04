#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, importlib.util, json, os, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch

ROOT=Path("/home/agni/waymo")
S8=ROOT/"iscai_stage8"
DATA_ROOT=S8/"data/block82b_deepsense_primary"

SCRIPT=S8/"scripts/run_stage8_block87c_corrected_preformal_policy_fallback_replay.py"
CONTRACT=S8/"configs/stage8_block87c_corrected_preformal_policy_fallback_replay_contract.json"

MODEL_SRC=S8/"src/deepsense_lidar_encoder.py"
LOADER_SRC=S8/"src/mat5_lidar_loader.py"
POLICY_SRC=S8/"src/deepsense_beam_policy.py"
RUNTIME_SRC=S8/"src/deepsense_policy_runtime.py"

B85C=S8/"artifacts/block85c_label_index_corrected_training"
CKPT=B85C/"duallidarconv1d64_label0based_epoch080.pt"
MAP85C=B85C/"train_label_index_mapping_gate.json"
SEAL85C=B85C/"stage8_block85c_label_index_corrected_training_seal.json"

B86C=S8/"artifacts/block86c_corrected_probability_calibration"
CAL=B86C/"frozen_temperature_calibrator.json"
INDEX86C=B86C/"calibration_sample_index.json"
DIAG86C=B86C/"calibration_diagnostics.json"
MANIFEST86C=B86C/"stage8_block86c_manifest.json"
SEAL86C=B86C/"stage8_block86c_corrected_probability_calibration_seal.json"
REPORT86C=S8/"reports/stage8_block86c_corrected_probability_calibration_report.json"

NORM=S8/"artifacts/block85b_encoder_training_cpu_repair/train_only_normalization.json"
B82B_FILES=S8/"artifacts/block82b_canonical_corpus_schema_split/canonical_extracted_file_manifest.json"
B84_SEAL=S8/"artifacts/block84_beam_policy_interface/stage8_block84_beam_policy_interface_seal.json"

OUT=S8/"artifacts/block87c_corrected_preformal_policy_fallback_replay"
REPLAY=OUT/"calibration_policy_replay_summary.json"
SYNTH=OUT/"synthetic_fallback_tests.json"
MANIFEST=OUT/"stage8_block87c_manifest.json"
REPORT=S8/"reports/stage8_block87c_corrected_preformal_policy_fallback_replay_report.json"
SEAL=OUT/"stage8_block87c_corrected_preformal_policy_fallback_replay_seal.json"

EXPECTED_CONTRACT="039b7f998335814d35732c4d420f88048b889cb972924687fae3cad0e6910a68"
EXPECTED={
 str(MODEL_SRC):"adb943f12490612a6b92ffa3e70f69b96b1002fa77e4e769f5ea26ed6d25c417",
 str(LOADER_SRC):"b4b3d9e09a20a5ee790a4ad7fd22f53dad1d6f4b953e70f824ade80f9b86299c",
 str(POLICY_SRC):"5b7e6c3c4ff7eca4d1db86df90edfa19f0bbfc551255779833485df074d87cb8",
 str(RUNTIME_SRC):"5171a69ebe491027088934075c2fe2c8c74ef4a5e10823f2ff636237056ff3f3",
 str(CKPT):"7aa785741a2929eedc106e77b0e701f02412f5dbf434823e51eed6d0f4eb6417",
 str(MAP85C):"36b7610ebb1cdf7e8004c10baad794d6e074554fa5805e34eac020eb681eac75",
 str(SEAL85C):"b0dae15111baae6a0cbe132e402459148474c6bad87c58c5a71eccba737de91e",
 str(CAL):"25e7b666a3b8803723e5ade05c3c9c6c1d943995401ef108bf36ffc7c65085d1",
 str(INDEX86C):"4deef5e69cd899f5edef7bb58f96e864b0d60a84f485274021791100e42b76a0",
 str(DIAG86C):"569c4b747c385e31ed37bcc046b7b6182ffb75425da4f63538939f150788e6ec",
 str(MANIFEST86C):"8a34dda411a0f8fb1fbe0809c545b2008b1f3188e8811d037204d9180be014cb",
 str(REPORT86C):"b1b83b2577c99789621635c07ac2fec8a7a9799263c0899cf3e10977247da09d",
 str(SEAL86C):"ff895d100129e7f3f1e8c67e75c5d067fe7f0b3aaf40f29717c730e01a6ed60b",
 str(NORM):"8e12e7b5bccf57c31075569a575507cd95e4b1710f5bb7327cbdd203c36575fd",
 str(B82B_FILES):"79dc92fb6a90acd36c4598de436517ebac6c7e7904334dd76a520e1c90a0b895",
 str(B84_SEAL):"c5dcbf5b1bf8d7965fd32bf794f095b4495d30d98e35bddef22ce3abffe554e2",
}

VAL_SPECS=[
 (8,DATA_ROOT/"LiDAR/Scenario8/development_dataset/scenario8_dev_val.csv"),
 (9,DATA_ROOT/"LiDAR/Scenario9/development_dataset/scenario9_dev_val.csv"),
]
POLICIES=[
 "adaptive_topk_q090","adaptive_topk_q095","adaptive_topk_q0975","adaptive_topk_q099",
 "fixed_top1","fixed_top3","fixed_top5","exhaustive_64",
]
QMAP={
 "adaptive_topk_q090":0.90,
 "adaptive_topk_q095":0.95,
 "adaptive_topk_q0975":0.975,
 "adaptive_topk_q099":0.99,
}

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
                req(raw.is_file() and scr.is_file(),"missing dev_val LiDAR")
                out.append((sc,csvp,rn,raw,scr))
                n+=1
        counts[str(sc)]=n
    req(counts=={"8":775,"9":1172} and len(out)==1947,f"dev_val counts changed: {counts}")
    return out,counts

def validate(policy_name,out,probs):
    req(out["fallback_triggered"] is False,f"unexpected fallback on valid replay: {policy_name}")
    beams=out["selected_beams"]; k=out["K"]
    req(k==len(beams) and 1<=k<=64,f"bad K for {policy_name}")
    req(len(set(beams))==len(beams),"duplicate beams")
    req(all(isinstance(x,int) and 0<=x<64 for x in beams),"beam outside 0..63")
    if policy_name=="fixed_top1": req(k==1,"top1 K changed")
    if policy_name=="fixed_top3": req(k==3,"top3 K changed")
    if policy_name=="fixed_top5": req(k==5,"top5 K changed")
    if policy_name=="exhaustive_64": req(k==64 and beams==list(range(64)),"exhaustive semantics changed")
    if policy_name in QMAP:
        q=QMAP[policy_name]
        mass=sum(probs[i] for i in beams)
        req(mass+1e-12>=q,f"mass below q for {policy_name}")
        if k>1:
            req(sum(probs[i] for i in beams[:-1]) < q+1e-12,f"adaptive set not minimal for {policy_name}")
    return k

def main():
    print("="*92)
    print("STAGE 8 — BLOCK 8.7C")
    print("CORRECTED PRE-FORMAL CALIBRATED POLICY / FALLBACK REPLAY")
    print("DEV_VAL ONLY — NO FORMAL / NO MMWAVE POWER ACCESS")
    print("="*92)

    req(Path(__file__).resolve()==SCRIPT.resolve(),"runner path mismatch")
    req(CONTRACT.is_file() and sha(CONTRACT)==EXPECTED_CONTRACT,"contract SHA mismatch")
    for raw,e in EXPECTED.items():
        p=Path(raw); req(p.is_file() and sha(p)==e,f"frozen upstream changed: {p}")
    for p in [REPLAY,SYNTH,MANIFEST,REPORT,SEAL]:
        req(not p.exists(),f"8.7C output exists: {p}")

    s85=json.loads(SEAL85C.read_text())
    s86=json.loads(SEAL86C.read_text())
    req(s85.get("status")=="FROZEN_COMPLETE_BLOCK85C","8.5C status changed")
    req(s86.get("status")=="FROZEN_COMPLETE_BLOCK86C","8.6C status changed")
    req(s86.get("Block87C_may_start") is True,"8.7C not authorized")
    req(s85.get("formal_data_access") is False and s85.get("formal_power_access") is False,"8.5C formal boundary changed")
    req(s86.get("formal_data_access") is False and s86.get("formal_power_access") is False,"8.6C formal boundary changed")
    mp=json.loads(MAP85C.read_text())
    req(mp.get("label_minus_1_matches_power_max")==7030,"8.5C mapping evidence changed")

    cal=json.loads(CAL.read_text())
    T=float(cal["temperature"])
    req(abs(T-1.139078692545004)<1e-15,"corrected frozen temperature changed")

    print("\n===== A. CORRECTED FROZEN PIPELINE GATE =====")
    print("Block8.5C corrected checkpoint identity = PASS")
    print("Block8.6C corrected calibrator identity = PASS")
    print("label adapter y-1 evidence = 7,030/7,030 PASS")
    print("temperature =",T)
    print("formal dev_test / formal power access = NO")

    loader=mod("mat5",LOADER_SRC)
    modelm=mod("enc",MODEL_SRC)
    policy=mod("policy",POLICY_SRC)
    runtime=mod("runtime",RUNTIME_SRC)

    recs,counts=records()
    fm=json.loads(B82B_FILES.read_text())
    hmap={x["relative_path"]:x["sha256"] for x in fm["files"]}

    # Verify only dev_val CSV and LiDAR bytes. Never open unit1_pwr_1 payloads.
    paths={r[1] for r in recs} | {r[3] for r in recs} | {r[4] for r in recs}
    def chk(p):
        rel=str(p.relative_to(DATA_ROOT))
        req(hmap.get(rel)==sha(p),f"frozen replay byte mismatch: {rel}")
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(chk,sorted(paths)))

    print("\n===== B. CORRECTED DEV_VAL PIPELINE REPLAY =====")
    raw=np.empty((1947,460,2),np.float32)
    scr=np.empty((1947,216,2),np.float32)
    def one(t):
        i,r=t
        a=loader.load_mat5_double_matrix(r[3],"data",(460,2)).astype(np.float32)
        b=loader.load_mat5_double_matrix(r[4],"data",(216,2)).astype(np.float32)
        return i,a,b
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i,a,b in ex.map(one,enumerate(recs),chunksize=16):
            raw[i]=a; scr[i]=b

    norm=json.loads(NORM.read_text())
    rm=np.asarray(norm["raw_mean"],np.float32); rs=np.asarray(norm["raw_std"],np.float32)
    sm=np.asarray(norm["scr_mean"],np.float32); ss=np.asarray(norm["scr_std"],np.float32)
    raw=(raw-rm)/np.maximum(rs,1e-8)
    scr=(scr-sm)/np.maximum(ss,1e-8)
    req(np.isfinite(raw).all() and np.isfinite(scr).all(),"nonfinite replay LiDAR")

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
            req(np.isfinite(z).all(),"nonfinite replay logits")
            logits.append(z)
    logits=np.concatenate(logits,axis=0)
    req(logits.shape==(1947,64),"replay logit shape changed")

    stats={p:{"fallback_count":0,"K_sum":0,"K_min":65,"K_max":0} for p in POLICIES}
    mono=0
    for z in logits:
        probs=runtime.calibrated_probabilities_from_logits(z,T)
        ks={}
        for pname in POLICIES:
            out=runtime.safe_select(policy,pname,z,T)
            k=validate(pname,out,probs)
            stats[pname]["fallback_count"]+=int(out["fallback_triggered"])
            stats[pname]["K_sum"]+=k
            stats[pname]["K_min"]=min(stats[pname]["K_min"],k)
            stats[pname]["K_max"]=max(stats[pname]["K_max"],k)
            ks[pname]=k
        req(ks["adaptive_topk_q090"]<=ks["adaptive_topk_q095"]<=ks["adaptive_topk_q0975"]<=ks["adaptive_topk_q099"],
            f"adaptive K monotonicity failed: {ks}")
        mono+=1

    for p in POLICIES:
        stats[p]["mean_K"]=stats[p].pop("K_sum")/1947.0
        req(stats[p]["fallback_count"]==0,f"valid-data fallback for {p}")

    print("dev_val rows replayed = 1,947")
    print("Scenario8/9 =",counts["8"],"/",counts["9"])
    print("valid-data fallback count = 0 for all policies")
    print("adaptive q monotonicity = 1,947/1,947 PASS")
    for p in POLICIES:
        s=stats[p]
        print(f"{p}: meanK={s['mean_K']:.6f} minK={s['K_min']} maxK={s['K_max']}")

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
    tests=[]
    for name,z,t,pname in cases:
        out=runtime.safe_select(policy,pname,z,t)
        req(out["fallback_triggered"] is True,f"fallback did not trigger: {name}")
        req(out["K"]==64 and out["selected_beams"]==list(range(64)),f"fallback not exhaustive: {name}")
        req(out["measured_power_accessed"] is False,f"fallback power boundary failed: {name}")
        tests.append({"name":name,"pass":True,"K":64})
    print("synthetic fallback tests = 6/6 PASS")
    print("fallback = exhaustive_64 / measured-power access = NO")

    atom_json(REPLAY,{
        "stage":8,"block":"8.7C","status":"PASS_CORRECTED_CALIBRATION_POLICY_REPLAY",
        "rows":1947,"scenario_counts":counts,"temperature":T,
        "checkpoint_sha256":EXPECTED[str(CKPT)],
        "calibrator_sha256":EXPECTED[str(CAL)],
        "label_adapter_bound":"beam_index_1 - 1",
        "policy_stats":stats,
        "adaptive_K_monotonic_samples_passed":mono,
        "valid_data_fallback_total":sum(x["fallback_count"] for x in stats.values()),
        "formal_data_access":False,"mmwave_power_payload_access":False,
    })
    atom_json(SYNTH,{
        "stage":8,"block":"8.7C","status":"PASS_CORRECTED_SYNTHETIC_FAIL_CLOSED_TESTS",
        "tests":tests,"all_fallbacks_exhaustive_64":True,"measured_power_access":False,
    })
    atom_json(MANIFEST,{
        "stage":8,"block":"8.7C","status":"FROZEN_COMPLETE_BLOCK87C",
        "contract_sha256":EXPECTED_CONTRACT,
        "Block85C_seal_sha256":EXPECTED[str(SEAL85C)],
        "Block86C_seal_sha256":EXPECTED[str(SEAL86C)],
        "mapping_sha256":EXPECTED[str(MAP85C)],
        "checkpoint_sha256":EXPECTED[str(CKPT)],
        "calibrator_sha256":EXPECTED[str(CAL)],
        "policy_sha256":EXPECTED[str(POLICY_SRC)],
        "runtime_sha256":EXPECTED[str(RUNTIME_SRC)],
        "runner_sha256":sha(SCRIPT),
        "replay_sha256":sha(REPLAY),"synthetic_sha256":sha(SYNTH),
        "formal_data_access":False,"formal_power_access":False,
    })
    atom_json(REPORT,{
        "stage":8,"block":"8.7C","status":"PASS_BLOCK87C_CORRECTED_PREFORMAL_REPLAY",
        "replay_rows":1947,"policy_count":8,
        "label_adapter_bound":"beam_index_1 - 1",
        "all_valid_replay_fallback_counts_zero":True,
        "adaptive_K_monotonicity_passed_rows":1947,
        "synthetic_fallback_tests_passed":6,
        "training":False,"calibration_refit":False,
        "formal_data_access":False,"formal_power_access":False,
        "Block88C_authorized":True,
        "replay_sha256":sha(REPLAY),"synthetic_sha256":sha(SYNTH),
        "manifest_sha256":sha(MANIFEST),"runner_sha256":sha(SCRIPT),
    })
    atom_json(SEAL,{
        "stage":8,"block":"8.7C","status":"FROZEN_COMPLETE_BLOCK87C",
        "Block85C_seal_sha256":EXPECTED[str(SEAL85C)],
        "Block86C_seal_sha256":EXPECTED[str(SEAL86C)],
        "Block84_seal_sha256":EXPECTED[str(B84_SEAL)],
        "contract_sha256":EXPECTED_CONTRACT,
        "mapping_sha256":EXPECTED[str(MAP85C)],
        "checkpoint_sha256":EXPECTED[str(CKPT)],
        "calibrator_sha256":EXPECTED[str(CAL)],
        "runtime_sha256":EXPECTED[str(RUNTIME_SRC)],
        "runner_sha256":sha(SCRIPT),
        "replay_sha256":sha(REPLAY),"synthetic_sha256":sha(SYNTH),
        "manifest_sha256":sha(MANIFEST),"report_sha256":sha(REPORT),
        "formal_data_access":False,"formal_power_access":False,
        "Block88C_may_start":True,
    })

    for raw,e in EXPECTED.items():
        req(sha(Path(raw))==e,f"upstream modified during 8.7C: {raw}")

    print("\n"+"="*92)
    print("BLOCK 8.7C = FULLY VERIFIED / FROZEN")
    print("CORRECTED CHECKPOINT + CALIBRATOR REPLAY = 1,947 dev_val ROWS PASS")
    print("LABEL ADAPTER BOUND = beam_index_1 - 1")
    print("PRIMARY POLICY q=0.95 + q SWEEP = PASS")
    print("FIXED TOP-1/3/5 + EXHAUSTIVE-64 = PASS")
    print("ADAPTIVE K MONOTONICITY = 1,947/1,947 PASS")
    print("VALID-DATA FALLBACKS = 0")
    print("SYNTHETIC FAIL-CLOSED FALLBACK TESTS = 6/6 PASS")
    print("MEASURED POWER CONTROLLER INPUT = NO")
    print("TRAINING / CALIBRATION REFIT = NO")
    print("FORMAL dev_test / FORMAL POWER ACCESS = NO")
    print("OLD 8.7 PRESERVED APPEND-ONLY")
    print("BLOCK 8.8C MAY START = YES")
    print("8.7C replay SHA256 =",sha(REPLAY))
    print("8.7C synthetic SHA256 =",sha(SYNTH))
    print("8.7C manifest SHA256 =",sha(MANIFEST))
    print("8.7C report SHA256 =",sha(REPORT))
    print("8.7C seal SHA256 =",sha(SEAL))
    print("8.7C runner SHA256 =",sha(SCRIPT))
    print("STATUS = FROZEN_COMPLETE_BLOCK87C")
    print("="*92)

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print("\n"+"!"*92)
        print("BLOCK 8.7C FAIL-CLOSED")
        print(f"{type(e).__name__}: {e}")
        print("DO NOT ACCESS FORMAL dev_test OR FORMAL MMWAVE POWER.")
        print("DO NOT START BLOCK8.8C.")
        print("!"*92)
        raise
