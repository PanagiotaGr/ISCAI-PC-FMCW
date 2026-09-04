#!/usr/bin/env python3
from __future__ import annotations

import hashlib, json, math, os, tempfile
from pathlib import Path
import numpy as np

ROOT=Path("/home/agni/waymo")
S8=ROOT/"iscai_stage8"

SCRIPT=S8/"scripts/run_stage8_block89_statistics_failure_tradeoff.py"
CONTRACT=S8/"configs/stage8_block89_statistics_failure_tradeoff_contract.json"

B88=S8/"artifacts/block88c_corrected_formal_measured_deepsense_evaluation"
RECORDS=B88/"formal_measured_power_records.jsonl"
SUMMARY88=B88/"formal_metrics_summary.json"
CONTROLLER=B88/"formal_controller_decisions.jsonl"
CONTROLLER_SEAL=B88/"formal_controller_preoutcome_seal.json"
ACCESS_AUTH=B88/"formal_power_access_authorization.json"
MANIFEST88=B88/"stage8_block88c_manifest.json"
SEAL88=B88/"stage8_block88c_corrected_formal_measured_deepsense_evaluation_seal.json"
REPORT88=S8/"reports/stage8_block88c_corrected_formal_measured_deepsense_evaluation_report.json"
RUNNER88=S8/"scripts/run_stage8_block88c_corrected_formal_measured_deepsense_evaluation.py"

EVALUATOR_SRC=S8/"src/deepsense_mmwave_evaluator.py"
B83_SEAL=S8/"artifacts/block83_measured_mmwave_evaluator_contract/stage8_block83_measured_mmwave_evaluator_contract_seal.json"
B87C_SEAL=S8/"artifacts/block87c_corrected_preformal_policy_fallback_replay/stage8_block87c_corrected_preformal_policy_fallback_replay_seal.json"

OUT=S8/"artifacts/block89_statistics_failure_tradeoff"
BOOT=OUT/"paired_bootstrap_10000_cis.json"
PAIRED=OUT/"paired_comparisons_vs_primary.json"
QTRADE=OUT/"q_policy_tradeoff.json"
FAIL=OUT/"descriptive_failure_slices.json"
MANIFEST=OUT/"stage8_block89_manifest.json"
REPORT=S8/"reports/stage8_block89_statistics_failure_tradeoff_report.json"
SEAL=OUT/"stage8_block89_statistics_failure_tradeoff_seal.json"

EXPECTED_CONTRACT="2da032d367df1d897efcc8f1b071c7e9c8b759238d0e42ad0b9d69204e98181b"
EXPECTED={
 str(RECORDS):"1984d777f8e649089c7f14ed80df8cd105838496ca03c70adce54fe124dd164b",
 str(SUMMARY88):"8f29fd2b3882ffb28b66fdef87a53490eeaf64fe3b501d788495710a989fa0ac",
 str(CONTROLLER):"8632cfbca75cedc1479730c9eba576492f09fb2fbff3ec4c7042a2f5a9cfb48f",
 str(CONTROLLER_SEAL):"b1f99a850dfb78edf08f4132db9617d947038a029703c55075ff35d51ca3b113",
 str(ACCESS_AUTH):"da3e945dff15d22e896e61ee4fedc0ccd0e4c6f4d07c59e847e5fea9bfe3af45",
 str(MANIFEST88):"eb32fdf7e0d6b95e5fa942b49f6dd8ecf861452c36f472edb199a52117de46d4",
 str(REPORT88):"7ebab989e03a987a3a7b04f6149c0975afee18f32910a7f2f9867506a704fd00",
 str(SEAL88):"f287399f7ddc6070c45f5c42502ae90fc4db0622f6cf42a845aa94d2b8ff8cb4",
 str(RUNNER88):"c94dea394cecdc8e74f01b5d7a24c69799da51f6881906bcb8052414be152b09",
 str(EVALUATOR_SRC):"7e2c4ffe589e3d6f58e5b6daf1d8fb9e6e44b2eef074bdf1624a3334f8cbe650",
 str(B83_SEAL):"056371994fa598c6c58622b0161febde17f0e75231b4ba6e1d596d93db4e575f",
 str(B87C_SEAL):"01b0cb67f10cae247fab56bd8214d32a2ea815c7ff9ef88589d4d2f11676e67a",
}

METHODS=[
 "adaptive_topk_q090","adaptive_topk_q095","adaptive_topk_q0975","adaptive_topk_q099",
 "fixed_top1","fixed_top3","fixed_top5","exhaustive_64","oracle_gt_best_beam_k1",
]
PRIMARY="adaptive_topk_q095"
QMETHODS=[
 ("0.90","adaptive_topk_q090"),("0.95","adaptive_topk_q095"),
 ("0.975","adaptive_topk_q0975"),("0.99","adaptive_topk_q099"),
]
METRICS=[
 "coverage","mean_K","probing_overhead","measured_power_loss_db",
 "outage_1db","outage_3db","outage_6db","normalized_se_gap_10db",
]
DIRECTION={
 "coverage":"higher_is_better",
 "mean_K":"lower_is_better",
 "probing_overhead":"lower_is_better",
 "measured_power_loss_db":"lower_is_better",
 "outage_1db":"lower_is_better",
 "outage_3db":"lower_is_better",
 "outage_6db":"lower_is_better",
 "normalized_se_gap_10db":"lower_is_better",
}
R=10000
SEED=20260902
CHUNK=250

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

def load_jsonl(p):
    rows=[]
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def metric_value(x,m):
    if m=="coverage": return float(x["coverage"])
    if m=="mean_K": return float(x["K"])
    if m=="probing_overhead": return float(x["probing_overhead"])
    if m=="measured_power_loss_db": return float(x["measured_power_loss_db"])
    if m=="outage_1db": return float(bool(x["outage"]["1dB"]))
    if m=="outage_3db": return float(bool(x["outage"]["3dB"]))
    if m=="outage_6db": return float(bool(x["outage"]["6dB"]))
    if m=="normalized_se_gap_10db":
        return float(x["normalized_spectral_efficiency"]["10dB"]["gap_bps_per_hz"])
    raise KeyError(m)

def ci(v):
    q=np.percentile(v,[2.5,50.0,97.5])
    return {"lower":float(q[0]),"median":float(q[1]),"upper":float(q[2])}

def point(arr,ix):
    return float(arr[ix].mean())

def kbin(k):
    if k<=3:return "1-3"
    if k<=5:return "4-5"
    if k<=8:return "6-8"
    return "9+"

def main():
    print("="*96)
    print("STAGE 8 — BLOCK 8.9")
    print("POST-FORMAL STATISTICS / FAILURE SLICES / POLICY TRADE-OFFS")
    print("ANALYSIS ONLY ON FROZEN 8.8C RECORDS — NO MODEL/CONTROLLER/EVALUATOR REEXECUTION")
    print("="*96)

    req(Path(__file__).resolve()==SCRIPT.resolve(),"runner path mismatch")
    req(CONTRACT.is_file() and sha(CONTRACT)==EXPECTED_CONTRACT,"8.9 contract SHA mismatch")
    for raw,e in EXPECTED.items():
        p=Path(raw); req(p.is_file() and sha(p)==e,f"frozen upstream changed: {p}")
    for p in [BOOT,PAIRED,QTRADE,FAIL,MANIFEST,REPORT,SEAL]:
        req(not p.exists(),f"8.9 output exists: {p}")

    s88=json.loads(SEAL88.read_text())
    req(s88.get("status")=="FROZEN_COMPLETE_BLOCK88C","8.8C status changed")
    req(s88.get("formal_power_accessed") is True,"8.8C formal access status changed")
    req(s88.get("controller_reexecution_after_power_open") is False,"causal barrier status changed")
    req(s88.get("measured_power_controller_input") is False,"controller power boundary changed")
    req(s88.get("post_outcome_tuning") is False,"8.8C post-outcome tuning status changed")
    req(s88.get("Block89_may_start") is True,"8.9 not authorized")

    print("\n===== A. IMMUTABLE FORMAL RECORD GATE =====")
    rows=load_jsonl(RECORDS)
    req(len(rows)==1030,"formal record count changed")
    req(len({r["sample_id"] for r in rows})==1030,"duplicate formal sample IDs")
    scenarios=np.array([int(r["scenario"]) for r in rows],dtype=np.int16)
    i8=np.where(scenarios==8)[0]; i9=np.where(scenarios==9)[0]
    req(len(i8)==437 and len(i9)==593,f"formal scenario counts changed: {len(i8)}/{len(i9)}")
    req(all(r.get("measured_power_used_by_controller") is False for r in rows),"controller power boundary changed in records")
    print("formal immutable records = 1,030 PASS")
    print("Scenario8/9 = 437 / 593")
    print("raw measured-power files reopened = NO")
    print("model/controller/evaluator reexecution = NO")

    # Build N x M arrays for all frozen metrics.
    arrays={}
    for metric in METRICS:
        a=np.empty((1030,len(METHODS)),dtype=np.float64)
        for n,r in enumerate(rows):
            req(set(METHODS).issubset(r["methods"].keys()),f"missing method at row {n}")
            for j,m in enumerate(METHODS):
                a[n,j]=metric_value(r["methods"][m],metric)
        req(np.isfinite(a).all(),f"nonfinite metric array: {metric}")
        arrays[metric]=a

    print("\n===== B. POINT-ESTIMATE REPRODUCTION GATE =====")
    summ=json.loads(SUMMARY88.read_text())
    agg_ix={"Scenario8":i8,"Scenario9":i9,"pooled":np.arange(1030)}
    for method in METHODS:
        j=METHODS.index(method)
        for agg,ix in agg_ix.items():
            frozen=summ["methods"][method][agg]
            checks={
                "coverage":float(frozen["coverage"]),
                "mean_K":float(frozen["mean_K"]),
                "probing_overhead":float(frozen["mean_probing_overhead"]),
                "measured_power_loss_db":float(frozen["mean_measured_power_loss_db"]),
                "outage_1db":float(frozen["outage_rate"]["1dB"]),
                "outage_3db":float(frozen["outage_rate"]["3dB"]),
                "outage_6db":float(frozen["outage_rate"]["6dB"]),
                "normalized_se_gap_10db":float(frozen["mean_normalized_se_gap_bps_per_hz"]["10dB"]),
            }
            for metric,val in checks.items():
                got=float(arrays[metric][ix,j].mean())
                req(abs(got-val)<=1e-12,f"point estimate mismatch {method}/{agg}/{metric}: {got} != {val}")
    print("all 9 methods x 3 aggregations x 8 metrics reproduce frozen 8.8C summary = PASS")

    print("\n===== C. PAIRED STRATIFIED BOOTSTRAP 10,000 =====")
    # Same bootstrap indices are reused across every method and metric.
    rng=np.random.default_rng(SEED)
    boot={agg:{metric:np.empty((R,len(METHODS)),dtype=np.float64) for metric in METRICS}
          for agg in ["Scenario8","Scenario9","pooled"]}

    pos=0
    while pos<R:
        b=min(CHUNK,R-pos)
        draw8=i8[rng.integers(0,len(i8),size=(b,len(i8)))]
        draw9=i9[rng.integers(0,len(i9),size=(b,len(i9)))]
        for metric,a in arrays.items():
            m8=a[draw8].mean(axis=1)
            m9=a[draw9].mean(axis=1)
            mp=(m8*len(i8)+m9*len(i9))/1030.0
            boot["Scenario8"][metric][pos:pos+b]=m8
            boot["Scenario9"][metric][pos:pos+b]=m9
            boot["pooled"][metric][pos:pos+b]=mp
        pos+=b
        if pos in (2500,5000,7500,10000):
            print(f"bootstrap replicates completed = {pos}/10000")

    method_cis={
        "stage":8,"block":"8.9","status":"PAIRED_STRATIFIED_BOOTSTRAP_10000_COMPLETE",
        "replicates":R,"seed":SEED,"interval":"95% percentile",
        "scenario_draws_per_replicate":{"Scenario8":437,"Scenario9":593},
        "same_resample_indices_for_all_methods_and_metrics":True,
        "methods":{},
    }
    for m in METHODS:
        j=METHODS.index(m); method_cis["methods"][m]={}
        for agg,ix in agg_ix.items():
            method_cis["methods"][m][agg]={}
            for metric in METRICS:
                method_cis["methods"][m][agg][metric]={
                    "point_estimate":point(arrays[metric],ix)[()] if False else float(arrays[metric][ix,j].mean()),
                    "ci95":ci(boot[agg][metric][:,j]),
                    "direction":DIRECTION[metric],
                }
    atom_json(BOOT,method_cis)

    primary_j=METHODS.index(PRIMARY)
    paired={
        "stage":8,"block":"8.9","status":"PAIRED_COMPARISONS_VS_PRIMARY",
        "primary":PRIMARY,
        "delta_definition":"method_minus_primary",
        "replicates":R,"seed":SEED,
        "comparisons":{},
        "common_overhead_comparison":{},
    }
    for m in METHODS:
        j=METHODS.index(m); paired["comparisons"][m]={}
        for agg,ix in agg_ix.items():
            paired["comparisons"][m][agg]={}
            for metric in METRICS:
                delta_arr=arrays[metric][ix,j]-arrays[metric][ix,primary_j]
                bd=boot[agg][metric][:,j]-boot[agg][metric][:,primary_j]
                paired["comparisons"][m][agg][metric]={
                    "point_delta":float(delta_arr.mean()),
                    "ci95":ci(bd),
                    "metric_direction":DIRECTION[metric],
                }

    common="fixed_top5"; cj=METHODS.index(common)
    for agg,ix in agg_ix.items():
        paired["common_overhead_comparison"][agg]={
            "definition":"fixed_top5 minus adaptive_topk_q095",
            "fixed_top5_mean_K":float(arrays["mean_K"][ix,cj].mean()),
            "adaptive_q095_mean_K":float(arrays["mean_K"][ix,primary_j].mean()),
            "metrics":{},
        }
        for metric in ["coverage","measured_power_loss_db","outage_3db","normalized_se_gap_10db","probing_overhead"]:
            d=boot[agg][metric][:,cj]-boot[agg][metric][:,primary_j]
            paired["common_overhead_comparison"][agg]["metrics"][metric]={
                "point_delta":float((arrays[metric][ix,cj]-arrays[metric][ix,primary_j]).mean()),
                "ci95":ci(d),
                "metric_direction":DIRECTION[metric],
            }
    atom_json(PAIRED,paired)
    print("paired bootstrap CIs = PASS")
    print("common-overhead comparison fixed_top5 vs adaptive q=.95 = PASS")

    print("\n===== D. PREREGISTERED q TRADE-OFF =====")
    qout={
        "stage":8,"block":"8.9","status":"DESCRIPTIVE_Q_POLICY_TRADEOFF",
        "selection_or_retuning":False,
        "primary_q":"0.95",
        "q_points":{},
    }
    for q,m in QMETHODS:
        j=METHODS.index(m)
        qout["q_points"][q]={}
        for agg,ix in agg_ix.items():
            qout["q_points"][q][agg]={}
            for metric in ["coverage","mean_K","probing_overhead","measured_power_loss_db","outage_3db","normalized_se_gap_10db"]:
                qout["q_points"][q][agg][metric]={
                    "point_estimate":float(arrays[metric][ix,j].mean()),
                    "ci95":ci(boot[agg][metric][:,j]),
                }
    atom_json(QTRADE,qout)
    for q,m in QMETHODS:
        j=METHODS.index(m)
        print(
            f"q={q} pooled: coverage={arrays['coverage'][:,j].mean():.6f} "
            f"meanK={arrays['mean_K'][:,j].mean():.6f} "
            f"loss_dB={arrays['measured_power_loss_db'][:,j].mean():.6f} "
            f"outage3={arrays['outage_3db'][:,j].mean():.6f}"
        )

    print("\n===== E. DESCRIPTIVE FAILURE SLICES — NO RETUNING =====")
    pj=primary_j
    pdata=[r["methods"][PRIMARY] for r in rows]
    flags={
        "coverage_miss":np.array([int(not bool(x["coverage"])) for x in pdata],dtype=np.int8),
        "power_loss_gt_0.1dB":np.array([int(float(x["measured_power_loss_db"])>0.1) for x in pdata],dtype=np.int8),
        "power_loss_gt_0.5dB":np.array([int(float(x["measured_power_loss_db"])>0.5) for x in pdata],dtype=np.int8),
        "outage_1dB":arrays["outage_1db"][:,pj].astype(np.int8),
        "outage_3dB":arrays["outage_3db"][:,pj].astype(np.int8),
    }
    kbins=np.array([kbin(int(x["K"])) for x in pdata],dtype=object)

    fout={
        "stage":8,"block":"8.9","status":"DESCRIPTIVE_FAILURE_SLICES_ONLY",
        "primary_policy":PRIMARY,
        "thresholds_frozen_in_contract":True,
        "used_for_retuning":False,
        "overall":{},
        "by_scenario":{},
        "by_K_bin":{},
        "worst_20_by_primary_measured_power_loss_db":[],
    }
    for name,f in flags.items():
        fout["overall"][name]={"count":int(f.sum()),"rate":float(f.mean())}
    for sc,ix in [("Scenario8",i8),("Scenario9",i9)]:
        fout["by_scenario"][sc]={}
        for name,f in flags.items():
            fout["by_scenario"][sc][name]={"count":int(f[ix].sum()),"rate":float(f[ix].mean())}
        fout["by_scenario"][sc]["mean_K"]=float(arrays["mean_K"][ix,pj].mean())
        fout["by_scenario"][sc]["mean_power_loss_db"]=float(arrays["measured_power_loss_db"][ix,pj].mean())

    for bname in ["1-3","4-5","6-8","9+"]:
        ix=np.where(kbins==bname)[0]
        rec={"n":int(len(ix))}
        if len(ix):
            rec.update({
                "coverage":float(arrays["coverage"][ix,pj].mean()),
                "mean_K":float(arrays["mean_K"][ix,pj].mean()),
                "mean_power_loss_db":float(arrays["measured_power_loss_db"][ix,pj].mean()),
                "outage_1db":float(arrays["outage_1db"][ix,pj].mean()),
                "outage_3db":float(arrays["outage_3db"][ix,pj].mean()),
            })
        fout["by_K_bin"][bname]=rec

    loss=arrays["measured_power_loss_db"][:,pj]
    worst=np.argsort(-loss)[:20]
    for rank,ix in enumerate(worst,start=1):
        r=rows[int(ix)]; x=r["methods"][PRIMARY]
        fout["worst_20_by_primary_measured_power_loss_db"].append({
            "rank":rank,"sample_id":r["sample_id"],"scenario":int(r["scenario"]),
            "official_csv_index":r["official_csv_index"],
            "K":int(x["K"]),"coverage":int(x["coverage"]),
            "measured_power_loss_db":float(x["measured_power_loss_db"]),
            "outage_1db":bool(x["outage"]["1dB"]),"outage_3db":bool(x["outage"]["3dB"]),
        })
    atom_json(FAIL,fout)

    print("primary coverage misses =",fout["overall"]["coverage_miss"]["count"])
    print("primary loss >0.1 dB =",fout["overall"]["power_loss_gt_0.1dB"]["count"])
    print("primary loss >0.5 dB =",fout["overall"]["power_loss_gt_0.5dB"]["count"])
    print("primary outage@1dB =",fout["overall"]["outage_1dB"]["count"])
    print("primary outage@3dB =",fout["overall"]["outage_3dB"]["count"])
    print("failure slices used for retuning = NO")

    atom_json(MANIFEST,{
        "stage":8,"block":"8.9","status":"FROZEN_COMPLETE_BLOCK89",
        "contract_sha256":EXPECTED_CONTRACT,
        "Block88C_seal_sha256":EXPECTED[str(SEAL88)],
        "formal_records_sha256":EXPECTED[str(RECORDS)],
        "formal_summary_sha256":EXPECTED[str(SUMMARY88)],
        "controller_decisions_sha256":EXPECTED[str(CONTROLLER)],
        "Block83_evaluator_sha256":EXPECTED[str(EVALUATOR_SRC)],
        "runner_sha256":sha(SCRIPT),
        "bootstrap_sha256":sha(BOOT),"paired_comparisons_sha256":sha(PAIRED),
        "q_tradeoff_sha256":sha(QTRADE),"failure_slices_sha256":sha(FAIL),
        "analysis_only":True,"raw_power_files_reopened":False,
        "controller_reexecuted":False,"evaluator_reexecuted":False,
        "post_outcome_tuning":False,
    })

    p95=qout["q_points"]["0.95"]["pooled"]
    atom_json(REPORT,{
        "stage":8,"block":"8.9","status":"PASS_BLOCK89_STATISTICS_FAILURE_TRADEOFF",
        "bootstrap_replicates":R,"bootstrap_seed":SEED,
        "paired_scenario_stratified":True,
        "primary_policy":PRIMARY,
        "primary_pooled":{k:v["point_estimate"] for k,v in p95.items()},
        "primary_pooled_ci95":{k:v["ci95"] for k,v in p95.items()},
        "common_overhead_comparison":"fixed_top5 minus adaptive_topk_q095",
        "failure_slices_descriptive_only":True,
        "raw_power_files_reopened":False,
        "controller_reexecuted":False,"evaluator_reexecuted":False,
        "post_outcome_tuning":False,
        "Block810_authorized":True,
        "bootstrap_sha256":sha(BOOT),"paired_sha256":sha(PAIRED),
        "q_tradeoff_sha256":sha(QTRADE),"failure_sha256":sha(FAIL),
        "manifest_sha256":sha(MANIFEST),"runner_sha256":sha(SCRIPT),
    })

    atom_json(SEAL,{
        "stage":8,"block":"8.9","status":"FROZEN_COMPLETE_BLOCK89",
        "Block88C_seal_sha256":EXPECTED[str(SEAL88)],
        "formal_records_sha256":EXPECTED[str(RECORDS)],
        "contract_sha256":EXPECTED_CONTRACT,
        "runner_sha256":sha(SCRIPT),
        "bootstrap_sha256":sha(BOOT),"paired_comparisons_sha256":sha(PAIRED),
        "q_tradeoff_sha256":sha(QTRADE),"failure_slices_sha256":sha(FAIL),
        "manifest_sha256":sha(MANIFEST),"report_sha256":sha(REPORT),
        "bootstrap_replicates":R,"bootstrap_seed":SEED,
        "analysis_only":True,"raw_power_files_reopened":False,
        "model_controller_evaluator_reexecution":False,
        "post_outcome_tuning":False,
        "Block810_may_start":True,
    })

    for raw,e in EXPECTED.items():
        req(sha(Path(raw))==e,f"frozen upstream modified during 8.9: {raw}")

    pci=p95["coverage"]["ci95"]
    kci=p95["mean_K"]["ci95"]
    lci=p95["measured_power_loss_db"]["ci95"]
    oci=p95["outage_3db"]["ci95"]
    sci=p95["normalized_se_gap_10db"]["ci95"]

    print("\n"+"="*96)
    print("BLOCK 8.9 = FULLY VERIFIED / FROZEN")
    print("ANALYSIS INPUT = IMMUTABLE BLOCK8.8C FORMAL RECORDS ONLY")
    print("PAIRED STRATIFIED BOOTSTRAP = 10,000 / SEED 20260902 PASS")
    print("SCENARIO8/9 DRAWS PER REPLICATE = 437 / 593")
    print("SAME RESAMPLES FOR ALL METHODS/METRICS = PASS")
    print(f"PRIMARY q=.95 COVERAGE CI95 = [{pci['lower']:.12f}, {pci['upper']:.12f}]")
    print(f"PRIMARY q=.95 MEAN K CI95 = [{kci['lower']:.12f}, {kci['upper']:.12f}]")
    print(f"PRIMARY q=.95 POWER LOSS dB CI95 = [{lci['lower']:.12f}, {lci['upper']:.12f}]")
    print(f"PRIMARY q=.95 OUTAGE@3dB CI95 = [{oci['lower']:.12f}, {oci['upper']:.12f}]")
    print(f"PRIMARY q=.95 NORM-SE GAP@10dB CI95 = [{sci['lower']:.12f}, {sci['upper']:.12f}]")
    print("q=.90/.95/.975/.99 TRADE-OFF = PASS")
    print("PAIRED COMPARISONS VS PRIMARY = PASS")
    print("COMMON-OVERHEAD fixed_top5 vs adaptive q=.95 = PASS")
    print("FAILURE SLICES = DESCRIPTIVE ONLY / NO RETUNING")
    print("RAW LiDAR / RAW MMWAVE POWER FILES REOPENED = NO")
    print("MODEL / CONTROLLER / EVALUATOR REEXECUTION = NO")
    print("POST-OUTCOME TUNING = NO")
    print("BLOCK 8.10 MAY START = YES")
    print("8.9 bootstrap SHA256 =",sha(BOOT))
    print("8.9 paired SHA256 =",sha(PAIRED))
    print("8.9 q tradeoff SHA256 =",sha(QTRADE))
    print("8.9 failure SHA256 =",sha(FAIL))
    print("8.9 manifest SHA256 =",sha(MANIFEST))
    print("8.9 report SHA256 =",sha(REPORT))
    print("8.9 seal SHA256 =",sha(SEAL))
    print("8.9 runner SHA256 =",sha(SCRIPT))
    print("STATUS = FROZEN_COMPLETE_BLOCK89")
    print("="*96)

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print("\n"+"!"*96)
        print("BLOCK 8.9 FAIL-CLOSED")
        print(f"{type(e).__name__}: {e}")
        print("DO NOT MODIFY ANY FROZEN BLOCK8.8C OUTCOME OR UPSTREAM SCIENTIFIC ARTIFACT.")
        print("DO NOT START BLOCK8.10 UNLESS BLOCK8.9 SEAL EXISTS.")
        print("!"*96)
        raise
