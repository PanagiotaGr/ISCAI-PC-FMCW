#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, html, json, math, os, tempfile
from pathlib import Path

ROOT=Path("/home/agni/waymo")
S5=ROOT/"iscai_stage5"
S7=ROOT/"iscai_stage7"
S8=ROOT/"iscai_stage8"

SCRIPT=S8/"scripts/run_stage8_block810s_literal_pdf_deliverable_closure.py"
CONTRACT=S8/"configs/stage8_block810s_literal_pdf_deliverable_closure_contract.json"

B810=S8/"artifacts/block810_final_reporting_reproducibility"
SEAL810=B810/"stage8_block810_final_reproducibility_seal.json"
SUPER810=B810/"stage8_pdf_compliance_complete_superseding_seal.json"
TABLE810=B810/"deepsense_measured_mmwave_final_table.json"
SEP810=B810/"optical_mmwave_reporting_separation.json"
MATRIX810=B810/"stage8_completion_matrix.json"

B89=S8/"artifacts/block89_statistics_failure_tradeoff"
SEAL89=B89/"stage8_block89_statistics_failure_tradeoff_seal.json"
BOOT89=B89/"paired_bootstrap_10000_cis.json"
PAIRED89=B89/"paired_comparisons_vs_primary.json"
Q89=B89/"q_policy_tradeoff.json"
FAIL89=B89/"descriptive_failure_slices.json"

B88=S8/"artifacts/block88c_corrected_formal_measured_deepsense_evaluation"
SEAL88=B88/"stage8_block88c_corrected_formal_measured_deepsense_evaluation_seal.json"
SUMMARY88=B88/"formal_metrics_summary.json"

OUT=S8/"artifacts/block810s_literal_pdf_deliverable_closure"
FIG1=OUT/"deepsense_q_coverage_overhead_curve.svg"
FIG2=OUT/"deepsense_q_powerloss_overhead_curve.svg"
FIG3=OUT/"deepsense_outage_overhead_curve.svg"
FIG4=OUT/"deepsense_failure_case_summary.svg"
CSV_FINAL=OUT/"deepsense_measured_mmwave_final_table.csv"
CSV_Q=OUT/"deepsense_q_tradeoff_table.csv"
BASELINE=OUT/"upstream_previous_nearest_beam_crosscheck.json"
FIGMAN=OUT/"paper_ready_figure_manifest.json"
MATRIX=OUT/"literal_pdf_deliverable_closure_matrix.json"
MANIFEST=OUT/"stage8_block810s_manifest.json"
REPORT=S8/"reports/stage8_block810s_literal_pdf_deliverable_closure_report.json"
SEAL=OUT/"stage8_block810s_literal_pdf_deliverable_closure_seal.json"
SUPER=OUT/"stage8_literal_pdf_deliverable_closure_superseding_seal.json"

EXPECTED_CONTRACT="a64b84bbfee867ae8a1702a9d84d2e32bcfd4fd76fd78acc605526407ed62565"
EXPECTED={
 str(SEAL810):"c1c9c06f7ccc80160247ca9eeadd8d64312201168ae1eae26fa6b924ca8b2c17",
 str(SUPER810):"497d94565019238e71afa3d1ff05c45d42ba56a507aa7f61706f97bd4df313cc",
 str(TABLE810):"afb4736f2cb9fdf780a2662d37e41a368b0f5dbe54dc451c6c534b1d26359da5",
 str(SEP810):"19170f933ddf604009cf1a1587eb2c0d1dc0785a8eeafba0eb5033e245ff0fff",
 str(MATRIX810):"8f2a778a302ed3ab16546affeba6ec2596eb01c53271c5185d44fe9e5cabd3ed",
 str(SEAL89):"73c74faab50ff8b52d76089e093439414894a3e95d87e32348da845e05798a29",
 str(BOOT89):"a60faacc0c193fad5f8324aefc2dc80036053cf17d574b31790fce104f2bc1ae",
 str(PAIRED89):"3fe13ed2fa61b850b2aa799e0718768c8ec769b61be942577e7caab9d85ed9a0",
 str(Q89):"e0d037deff7c955552005a14da247fa9866996a7558d59b6ef8ea90cdc41d8bc",
 str(FAIL89):"99821b0a0b17e4f1d2dad2b20e6284509be07ea13962a06a95285f3d346a6838",
 str(SEAL88):"f287399f7ddc6070c45f5c42502ae90fc4db0622f6cf42a845aa94d2b8ff8cb4",
 str(SUMMARY88):"8f29fd2b3882ffb28b66fdef87a53490eeaf64fe3b501d788495710a989fa0ac",
}

METHOD_LABEL={
 "adaptive_topk_q090":"Adaptive q=.90",
 "adaptive_topk_q095":"Adaptive q=.95",
 "adaptive_topk_q0975":"Adaptive q=.975",
 "adaptive_topk_q099":"Adaptive q=.99",
 "fixed_top1":"Fixed Top-1",
 "fixed_top3":"Fixed Top-3",
 "fixed_top5":"Fixed Top-5",
 "exhaustive_64":"Exhaustive-64",
 "oracle_gt_best_beam_k1":"GT-best oracle",
}

class FailClosed(RuntimeError): pass
def req(c,m):
    if not c: raise FailClosed(m)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def atomic_bytes(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    req(not path.exists(),f"Refusing overwrite: {path}")
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

def atomic_json(path,obj):
    atomic_bytes(path,(json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode())

def atomic_text(path,text):
    atomic_bytes(path,text.encode("utf-8"))

def svg_escape(x): return html.escape(str(x),quote=True)

def nice_range(vals, pad=0.08, floor0=False):
    lo=min(vals); hi=max(vals)
    if lo==hi:
        d=1.0 if lo==0 else abs(lo)*0.1
        lo-=d; hi+=d
    d=hi-lo
    lo-=pad*d; hi+=pad*d
    if floor0: lo=max(0.0,lo)
    return lo,hi

def make_scatter_svg(title,xlabel,ylabel,points,xkey,ykey,out_path,
                     x_ci_key=None,y_ci_key=None,x_floor0=False,y_floor0=False,
                     footnote=None):
    W,H=980,650
    L,R,T,B=110,45,70,105
    xs=[float(p[xkey]) for p in points]; ys=[float(p[ykey]) for p in points]
    xlo,xhi=nice_range(xs,floor0=x_floor0); ylo,yhi=nice_range(ys,floor0=y_floor0)
    def X(x): return L+(float(x)-xlo)/(xhi-xlo)*(W-L-R)
    def Y(y): return T+(yhi-float(y))/(yhi-ylo)*(H-T-B)
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    out.append('<rect width="100%" height="100%" fill="white"/>')
    out.append(f'<text x="{W/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="700">{svg_escape(title)}</text>')
    # grid/ticks
    for i in range(6):
        tx=xlo+(xhi-xlo)*i/5; xx=X(tx)
        ty=ylo+(yhi-ylo)*i/5; yy=Y(ty)
        out.append(f'<line x1="{xx:.2f}" y1="{T}" x2="{xx:.2f}" y2="{H-B}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{xx:.2f}" y="{H-B+24}" text-anchor="middle" font-family="sans-serif" font-size="13">{tx:.4g}</text>')
        out.append(f'<line x1="{L}" y1="{yy:.2f}" x2="{W-R}" y2="{yy:.2f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{L-12}" y="{yy+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{ty:.4g}</text>')
    out.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="black" stroke-width="1.5"/>')
    out.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="black" stroke-width="1.5"/>')
    out.append(f'<text x="{(L+W-R)/2}" y="{H-45}" text-anchor="middle" font-family="sans-serif" font-size="16">{svg_escape(xlabel)}</text>')
    out.append(f'<text transform="translate(27 {(T+H-B)/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="16">{svg_escape(ylabel)}</text>')
    for idx,p in enumerate(points):
        x=float(p[xkey]); y=float(p[ykey]); xx=X(x); yy=Y(y)
        if x_ci_key and p.get(x_ci_key):
            c=p[x_ci_key]; x1=X(c["lower"]); x2=X(c["upper"])
            out.append(f'<line x1="{x1:.2f}" y1="{yy:.2f}" x2="{x2:.2f}" y2="{yy:.2f}" stroke="#6b7280" stroke-width="1.5"/>')
        if y_ci_key and p.get(y_ci_key):
            c=p[y_ci_key]; y1=Y(c["lower"]); y2=Y(c["upper"])
            out.append(f'<line x1="{xx:.2f}" y1="{y1:.2f}" x2="{xx:.2f}" y2="{y2:.2f}" stroke="#6b7280" stroke-width="1.5"/>')
        out.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="6.5" fill="#111827"/>')
        dx=8; dy=-8 if idx%2==0 else 18
        out.append(f'<text x="{xx+dx:.2f}" y="{yy+dy:.2f}" font-family="sans-serif" font-size="13">{svg_escape(p["label"])}</text>')
    if footnote:
        out.append(f'<text x="{L}" y="{H-15}" font-family="sans-serif" font-size="12">{svg_escape(footnote)}</text>')
    out.append('</svg>')
    atomic_text(out_path,"\n".join(out)+"\n")

def make_bar_svg(title, labels, values, ylabel, out_path, footnote=None):
    W,H=980,620; L,R,T,B=105,45,70,135
    vmax=max(values) if values else 1
    ymax=max(1.0,vmax*1.15)
    def Y(v): return T+(ymax-v)/ymax*(H-T-B)
    plotW=W-L-R
    slot=plotW/max(1,len(values)); bw=slot*0.56
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="{W/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="700">{svg_escape(title)}</text>']
    for i in range(6):
        v=ymax*i/5; yy=Y(v)
        out.append(f'<line x1="{L}" y1="{yy:.2f}" x2="{W-R}" y2="{yy:.2f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{L-12}" y="{yy+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{v:.4g}</text>')
    out.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="black" stroke-width="1.5"/>')
    out.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="black" stroke-width="1.5"/>')
    out.append(f'<text transform="translate(27 {(T+H-B)/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="16">{svg_escape(ylabel)}</text>')
    for i,(lab,v) in enumerate(zip(labels,values)):
        x=L+slot*(i+.5)-bw/2; y=Y(v); h=(H-B)-y
        out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bw:.2f}" height="{h:.2f}" fill="#374151"/>')
        out.append(f'<text x="{x+bw/2:.2f}" y="{y-8:.2f}" text-anchor="middle" font-family="sans-serif" font-size="13">{v}</text>')
        out.append(f'<text transform="translate({x+bw/2:.2f} {H-B+22}) rotate(22)" text-anchor="start" font-family="sans-serif" font-size="13">{svg_escape(lab)}</text>')
    if footnote:
        out.append(f'<text x="{L}" y="{H-15}" font-family="sans-serif" font-size="12">{svg_escape(footnote)}</text>')
    out.append('</svg>')
    atomic_text(out_path,"\n".join(out)+"\n")

def crosscheck_baselines():
    patterns={
        "previous_beam":["previous beam","previous-beam","previous_beam","persistence"],
        "nearest_beam":["nearest beam","nearest-beam","nearest_beam","geometry nearest"],
    }
    exts={".json",".txt",".md",".py",".yaml",".yml",".csv",".sh"}
    max_bytes=8_000_000
    result={"stage":8,"block":"8.10S","status":"READ_ONLY_UPSTREAM_BEAM_BASELINE_CROSSCHECK",
            "roots":[str(S5),str(S7)],"patterns":patterns,"matches":{"previous_beam":[],"nearest_beam":[]},
            "files_scanned":0,"files_skipped_large":0,"read_only":True}
    for base in [S5,S7]:
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in exts: continue
            try:
                size=p.stat().st_size
            except OSError:
                continue
            if size>max_bytes:
                result["files_skipped_large"]+=1; continue
            result["files_scanned"]+=1
            try:
                txt=p.read_text(encoding="utf-8",errors="ignore")
            except Exception:
                continue
            low=txt.lower()
            lines=txt.splitlines()
            for key,ps in patterns.items():
                if not any(x in low for x in ps): continue
                hits=[]
                for ln,line in enumerate(lines,1):
                    l=line.lower()
                    if any(x in l for x in ps):
                        hits.append({"line":ln,"snippet":line[:260]})
                        if len(hits)>=12: break
                if hits:
                    result["matches"][key].append({"path":str(p),"sha256":sha(p),"hits":hits})
    for key in ["previous_beam","nearest_beam"]:
        found=bool(result["matches"][key])
        result[key+"_status"]="FOUND_UPSTREAM_FROZEN_EVIDENCE" if found else "NOT_FOUND_UPSTREAM"
    result["stage8_section32_gate_impact"]="NONE"
    result["stage8_completion_criterion_impact"]="NONE"
    result["interpretation"]=(
        "The full experimental protocol lists previous-beam and nearest-beam as broader beam baselines. "
        "Section 32 DeepSense validation and the explicit Stage8 completion gate do not require every broader "
        "baseline to be rerun on DeepSense. Missing upstream evidence, if any, is therefore recorded rather than "
        "repaired post-outcome."
    )
    return result

def write_final_csv(table):
    rows=[]
    for method,aggs in table["methods"].items():
        for agg,x in aggs.items():
            row={"method":method,"aggregation":agg,"n":x["n"]}
            for metric in ["coverage","mean_K","probing_overhead","measured_power_loss_db","outage_1db","outage_3db","outage_6db","normalized_se_gap_10db"]:
                row[metric]=x[metric]["point"]
                row[metric+"_ci95_lower"]=x[metric]["ci95"]["lower"]
                row[metric+"_ci95_upper"]=x[metric]["ci95"]["upper"]
            rows.append(row)
    fields=list(rows[0].keys())
    lines=[]
    import io
    s=io.StringIO()
    w=csv.DictWriter(s,fieldnames=fields,lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    atomic_text(CSV_FINAL,s.getvalue())

def write_q_csv(q):
    rows=[]
    for qv,aggs in q["q_points"].items():
        for agg,mets in aggs.items():
            row={"q":qv,"aggregation":agg}
            for metric,v in mets.items():
                row[metric]=v["point_estimate"]
                row[metric+"_ci95_lower"]=v["ci95"]["lower"]
                row[metric+"_ci95_upper"]=v["ci95"]["upper"]
            rows.append(row)
    import io
    s=io.StringIO()
    fields=list(rows[0].keys())
    w=csv.DictWriter(s,fieldnames=fields,lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    atomic_text(CSV_Q,s.getvalue())

def main():
    print("="*104)
    print("STAGE 8 — BLOCK 8.10S")
    print("LITERAL PDF DELIVERABLE CLOSURE SUPPLEMENT")
    print("PAPER-READY DEEPSENSE OUTPUTS + READ-ONLY UPSTREAM BEAM-BASELINE CROSS-CHECK")
    print("="*104)

    req(Path(__file__).resolve()==SCRIPT.resolve(),"runner path mismatch")
    req(CONTRACT.is_file() and sha(CONTRACT)==EXPECTED_CONTRACT,"8.10S contract SHA mismatch")
    for raw,e in EXPECTED.items():
        p=Path(raw); req(p.is_file(),f"missing frozen input: {p}"); req(sha(p)==e,f"frozen input changed: {p}")
    for p in [FIG1,FIG2,FIG3,FIG4,CSV_FINAL,CSV_Q,BASELINE,FIGMAN,MATRIX,MANIFEST,REPORT,SEAL,SUPER]:
        req(not p.exists(),f"8.10S output already exists: {p}")

    s810=json.loads(SEAL810.read_text())
    sup=json.loads(SUPER810.read_text())
    s89=json.loads(SEAL89.read_text())
    s88=json.loads(SEAL88.read_text())
    req(s810.get("status")=="FROZEN_COMPLETE_BLOCK810","8.10 status changed")
    req(s810.get("stage8_core_status")=="STAGE8_PDF_CORE_COMPLETE","8.10 Stage8 core status changed")
    req(sup.get("status")=="STAGE8_PDF_CORE_COMPLETE","8.10 superseding seal status changed")
    req(s89.get("status")=="FROZEN_COMPLETE_BLOCK89","8.9 status changed")
    req(s88.get("status")=="FROZEN_COMPLETE_BLOCK88C","8.8C status changed")

    print("\n===== A. FROZEN POST-CORE INPUT GATE =====")
    print("Stage8 core superseding seal = PASS")
    print("Block8.8C formal outcomes = immutable PASS")
    print("Block8.9 bootstrap/failure analysis = immutable PASS")
    print("scientific reexecution = NO")

    table=json.loads(TABLE810.read_text())
    q=json.loads(Q89.read_text())
    fail=json.loads(FAIL89.read_text())
    boot=json.loads(BOOT89.read_text())

    print("\n===== B. PAPER-READY DEEPSENSE TABLES =====")
    write_final_csv(table)
    write_q_csv(q)
    print("DeepSense measured-mmWave final table CSV = PASS")
    print("DeepSense q-tradeoff table CSV = PASS")

    qpoints=[]
    for qv,m in [("0.90","adaptive_topk_q090"),("0.95","adaptive_topk_q095"),("0.975","adaptive_topk_q0975"),("0.99","adaptive_topk_q099")]:
        x=table["methods"][m]["pooled"]
        qpoints.append({
            "label":"q="+qv,
            "overhead":x["probing_overhead"]["point"],
            "overhead_ci":x["probing_overhead"]["ci95"],
            "coverage":x["coverage"]["point"],
            "coverage_ci":x["coverage"]["ci95"],
            "loss":x["measured_power_loss_db"]["point"],
            "loss_ci":x["measured_power_loss_db"]["ci95"],
        })

    make_scatter_svg(
        "DeepSense adaptive Top-K: coverage–overhead trade-off",
        "Probing overhead (K/64)","Top-K empirical coverage",
        qpoints,"overhead","coverage",FIG1,
        x_ci_key="overhead_ci",y_ci_key="coverage_ci",x_floor0=True,y_floor0=True,
        footnote="95% paired scenario-stratified bootstrap CIs; 10,000 replicates; formal dev_test only."
    )
    make_scatter_svg(
        "DeepSense adaptive Top-K: measured-power-loss–overhead trade-off",
        "Probing overhead (K/64)","Mean measured power loss (dB)",
        qpoints,"overhead","loss",FIG2,
        x_ci_key="overhead_ci",y_ci_key="loss_ci",x_floor0=True,y_floor0=True,
        footnote="Lower-left is better; 95% paired scenario-stratified bootstrap CIs."
    )

    methods=["adaptive_topk_q090","adaptive_topk_q095","adaptive_topk_q0975","adaptive_topk_q099","fixed_top1","fixed_top3","fixed_top5"]
    outpts=[]
    for m in methods:
        x=table["methods"][m]["pooled"]
        outpts.append({
            "label":METHOD_LABEL[m],
            "overhead":x["probing_overhead"]["point"],
            "overhead_ci":x["probing_overhead"]["ci95"],
            "outage":x["outage_1db"]["point"],
            "outage_ci":x["outage_1db"]["ci95"],
        })
    make_scatter_svg(
        "DeepSense beam policy: outage–overhead curve",
        "Probing overhead (K/64)","Outage probability @ 1 dB",
        outpts,"overhead","outage",FIG3,
        x_ci_key="overhead_ci",y_ci_key="outage_ci",x_floor0=True,y_floor0=True,
        footnote="1 dB shown because primary @3 dB is zero; frozen 3 dB results remain in final table."
    )

    flag_order=["coverage_miss","power_loss_gt_0.1dB","power_loss_gt_0.5dB","outage_1dB","outage_3dB"]
    labels=["Coverage miss","Loss >0.1 dB","Loss >0.5 dB","Outage @1 dB","Outage @3 dB"]
    vals=[int(fail["overall"][k]["count"]) for k in flag_order]
    make_bar_svg(
        "DeepSense primary q=.95: descriptive formal failure counts",
        labels,vals,"Formal sample count",FIG4,
        footnote="Descriptive only; thresholds were frozen before this reporting supplement and are not used for retuning."
    )
    print("coverage–overhead SVG = PASS")
    print("power-loss–overhead SVG = PASS")
    print("outage–overhead SVG = PASS")
    print("failure-case summary SVG = PASS")

    print("\n===== C. READ-ONLY UPSTREAM PREVIOUS/NEAREST-BEAM CROSS-CHECK =====")
    base=crosscheck_baselines()
    atomic_json(BASELINE,base)
    print("files scanned =",base["files_scanned"])
    print("previous-beam upstream status =",base["previous_beam_status"])
    print("nearest-beam upstream status =",base["nearest_beam_status"])
    print("Stage8 §32 / completion-gate impact = NONE")
    print("post-outcome baseline implementation = NO")

    figman={
        "stage":8,"block":"8.10S","status":"PAPER_READY_DEEPSENSE_REPORTING_COMPLETE",
        "figures":[
            {"path":FIG1.name,"sha256":sha(FIG1),"type":"vector_svg","source":"frozen 8.10 final table / 8.9 bootstrap"},
            {"path":FIG2.name,"sha256":sha(FIG2),"type":"vector_svg","source":"frozen 8.10 final table / 8.9 bootstrap"},
            {"path":FIG3.name,"sha256":sha(FIG3),"type":"vector_svg","source":"frozen 8.10 final table / 8.9 bootstrap"},
            {"path":FIG4.name,"sha256":sha(FIG4),"type":"vector_svg","source":"frozen 8.9 failure slices"},
        ],
        "tables":[
            {"path":CSV_FINAL.name,"sha256":sha(CSV_FINAL)},
            {"path":CSV_Q.name,"sha256":sha(CSV_Q)},
        ],
        "scientific_reexecution":False,
        "raw_data_reopened":False,
    }
    atomic_json(FIGMAN,figman)

    matrix={
        "stage":8,"block":"8.10S","status":"LITERAL_PDF_DELIVERABLE_CLOSURE_MATRIX",
        "stage8_scientific_mandatory_core":"PASS_PREVIOUSLY_FROZEN",
        "stage8_acceptance_criteria":"PASS_PREVIOUSLY_FROZEN",
        "paper_ready_deepsense_figures":True,
        "paper_ready_deepsense_tables":True,
        "outage_overhead_curve":True,
        "failure_case_figure":True,
        "q_tradeoff_figure_and_table":True,
        "optical_mmwave_separation":"PASS_PREVIOUSLY_FROZEN",
        "statistical_confidence_intervals":"PASS_PREVIOUSLY_FROZEN",
        "failure_case_analysis":"PASS_PREVIOUSLY_FROZEN",
        "broader_previous_beam_baseline_upstream":base["previous_beam_status"],
        "broader_nearest_beam_baseline_upstream":base["nearest_beam_status"],
        "broader_baseline_missing_would_block_stage8_section32":False,
        "broader_baseline_missing_would_block_explicit_stage8_completion_gate":False,
        "zero_shot_WOMD_to_DeepSense":"OPTIONAL_NONBLOCKING_NOT_RUN",
        "post_outcome_scientific_changes":False,
        "final_literal_stage8_status":"STAGE8_LITERAL_PDF_DELIVERABLE_CLOSURE_COMPLETE",
    }
    atomic_json(MATRIX,matrix)

    atomic_json(MANIFEST,{
        "stage":8,"block":"8.10S","status":"FROZEN_COMPLETE_BLOCK810S",
        "contract_sha256":EXPECTED_CONTRACT,
        "Block810_seal_sha256":EXPECTED[str(SEAL810)],
        "Stage8_pdf_core_superseding_seal_sha256":EXPECTED[str(SUPER810)],
        "Block89_seal_sha256":EXPECTED[str(SEAL89)],
        "Block88C_seal_sha256":EXPECTED[str(SEAL88)],
        "paper_ready_manifest_sha256":sha(FIGMAN),
        "baseline_crosscheck_sha256":sha(BASELINE),
        "closure_matrix_sha256":sha(MATRIX),
        "runner_sha256":sha(SCRIPT),
        "scientific_reexecution":False,
        "raw_data_reopened":False,
        "post_outcome_tuning":False,
    })

    atomic_json(REPORT,{
        "stage":8,"block":"8.10S","status":"PASS_BLOCK810S_LITERAL_PDF_DELIVERABLE_CLOSURE",
        "stage8_core_status":"STAGE8_PDF_CORE_COMPLETE",
        "literal_deliverable_status":"STAGE8_LITERAL_PDF_DELIVERABLE_CLOSURE_COMPLETE",
        "paper_ready_figures":4,
        "paper_ready_csv_tables":2,
        "previous_beam_upstream_status":base["previous_beam_status"],
        "nearest_beam_upstream_status":base["nearest_beam_status"],
        "stage8_section32_blocked_by_broader_baselines":False,
        "explicit_stage8_completion_gate_blocked_by_broader_baselines":False,
        "scientific_reexecution":False,
        "post_outcome_tuning":False,
        "paper_ready_manifest_sha256":sha(FIGMAN),
        "baseline_crosscheck_sha256":sha(BASELINE),
        "closure_matrix_sha256":sha(MATRIX),
        "manifest_sha256":sha(MANIFEST),
        "runner_sha256":sha(SCRIPT),
    })

    atomic_json(SEAL,{
        "stage":8,"block":"8.10S","status":"FROZEN_COMPLETE_BLOCK810S",
        "stage8_core_status":"STAGE8_PDF_CORE_COMPLETE",
        "literal_deliverable_status":"STAGE8_LITERAL_PDF_DELIVERABLE_CLOSURE_COMPLETE",
        "Block810_seal_sha256":EXPECTED[str(SEAL810)],
        "Stage8_pdf_core_superseding_seal_sha256":EXPECTED[str(SUPER810)],
        "Block89_seal_sha256":EXPECTED[str(SEAL89)],
        "Block88C_seal_sha256":EXPECTED[str(SEAL88)],
        "contract_sha256":EXPECTED_CONTRACT,
        "runner_sha256":sha(SCRIPT),
        "paper_ready_manifest_sha256":sha(FIGMAN),
        "baseline_crosscheck_sha256":sha(BASELINE),
        "closure_matrix_sha256":sha(MATRIX),
        "manifest_sha256":sha(MANIFEST),
        "report_sha256":sha(REPORT),
        "scientific_reexecution":False,
        "raw_data_reopened":False,
        "post_outcome_tuning":False,
    })

    atomic_json(SUPER,{
        "project":"Agni","stage":8,
        "status":"STAGE8_LITERAL_PDF_DELIVERABLE_CLOSURE_COMPLETE",
        "does_not_supersede_scientific_outcomes":True,
        "scientific_core_superseding_seal_sha256":EXPECTED[str(SUPER810)],
        "final_Block810S_seal_sha256":sha(SEAL),
        "paper_ready_manifest_sha256":sha(FIGMAN),
        "baseline_crosscheck_sha256":sha(BASELINE),
        "closure_matrix_sha256":sha(MATRIX),
        "zero_shot_WOMD_to_DeepSense":"OPTIONAL_NONBLOCKING_NOT_RUN",
        "scientific_reexecution":False,
        "post_outcome_tuning":False,
    })

    for raw,e in EXPECTED.items():
        req(sha(Path(raw))==e,f"frozen upstream modified during 8.10S: {raw}")

    print("\n"+"="*104)
    print("BLOCK 8.10S = FULLY VERIFIED / FROZEN")
    print("STAGE8 SCIENTIFIC CORE = UNCHANGED / PASS")
    print("PAPER-READY DEEPSENSE FIGURES = 4/4 PASS")
    print("PAPER-READY DEEPSENSE TABLES = 2/2 PASS")
    print("OUTAGE–OVERHEAD CURVE = PASS")
    print("FAILURE-CASE FIGURE = PASS")
    print("UPSTREAM PREVIOUS-BEAM CROSS-CHECK =",base["previous_beam_status"])
    print("UPSTREAM NEAREST-BEAM CROSS-CHECK =",base["nearest_beam_status"])
    print("BROADER BASELINE STATUS BLOCKS STAGE8 §32 / EXPLICIT COMPLETION GATE = NO")
    print("RAW LiDAR / RAW MMWAVE POWER REOPENED = NO")
    print("MODEL / CONTROLLER / EVALUATOR REEXECUTION = NO")
    print("POST-OUTCOME TUNING = NO")
    print("ZERO-SHOT WOMD->DEEPSENSE = OPTIONAL / NON-BLOCKING / NOT RUN")
    print("STAGE8 LITERAL PDF DELIVERABLE CLOSURE = PASS")
    print("8.10S figure manifest SHA256 =",sha(FIGMAN))
    print("8.10S baseline crosscheck SHA256 =",sha(BASELINE))
    print("8.10S closure matrix SHA256 =",sha(MATRIX))
    print("8.10S manifest SHA256 =",sha(MANIFEST))
    print("8.10S report SHA256 =",sha(REPORT))
    print("8.10S seal SHA256 =",sha(SEAL))
    print("8.10S superseding literal-closure seal SHA256 =",sha(SUPER))
    print("8.10S runner SHA256 =",sha(SCRIPT))
    print("STATUS = STAGE8_LITERAL_PDF_DELIVERABLE_CLOSURE_COMPLETE")
    print("="*104)

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print("\n"+"!"*104)
        print("BLOCK 8.10S FAIL-CLOSED")
        print(f"{type(e).__name__}: {e}")
        print("DO NOT MODIFY ANY FROZEN STAGE8 SCIENTIFIC OR FORMAL OUTCOME ARTIFACT.")
        print("!"*104)
        raise
