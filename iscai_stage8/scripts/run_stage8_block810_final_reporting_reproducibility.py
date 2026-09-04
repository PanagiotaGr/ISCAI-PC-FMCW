#!/usr/bin/env python3
from __future__ import annotations

import hashlib, json, math, os, tempfile
from pathlib import Path

ROOT=Path("/home/agni/waymo")
S7=ROOT/"iscai_stage7"
S8=ROOT/"iscai_stage8"

SCRIPT=S8/"scripts/run_stage8_block810_final_reporting_reproducibility.py"
CONTRACT=S8/"configs/stage8_block810_final_reporting_reproducibility_contract.json"

S7_PDF=S7/"artifacts/stage7_pdf_compliance_complete_superseding_seal.json"
S7_C3=S7/"artifacts/stage7_block75c3_final_reproducibility_seal.json"

B88=S8/"artifacts/block88c_corrected_formal_measured_deepsense_evaluation"
S88=B88/"stage8_block88c_corrected_formal_measured_deepsense_evaluation_seal.json"
REC88=B88/"formal_measured_power_records.jsonl"
SUM88=B88/"formal_metrics_summary.json"
CTRL88=B88/"formal_controller_decisions.jsonl"
CTRLSEAL88=B88/"formal_controller_preoutcome_seal.json"

B89=S8/"artifacts/block89_statistics_failure_tradeoff"
BOOT89=B89/"paired_bootstrap_10000_cis.json"
PAIRED89=B89/"paired_comparisons_vs_primary.json"
Q89=B89/"q_policy_tradeoff.json"
FAIL89=B89/"descriptive_failure_slices.json"
MAN89=B89/"stage8_block89_manifest.json"
SEAL89=B89/"stage8_block89_statistics_failure_tradeoff_seal.json"
REPORT89=S8/"reports/stage8_block89_statistics_failure_tradeoff_report.json"
RUNNER89=S8/"scripts/run_stage8_block89_statistics_failure_tradeoff.py"

SEAL85C=S8/"artifacts/block85c_label_index_corrected_training/stage8_block85c_label_index_corrected_training_seal.json"
SEAL86C=S8/"artifacts/block86c_corrected_probability_calibration/stage8_block86c_corrected_probability_calibration_seal.json"
SEAL87C=S8/"artifacts/block87c_corrected_preformal_policy_fallback_replay/stage8_block87c_corrected_preformal_policy_fallback_replay_seal.json"

OUT=S8/"artifacts/block810_final_reporting_reproducibility"
SEP=OUT/"optical_mmwave_reporting_separation.json"
TABLE=OUT/"deepsense_measured_mmwave_final_table.json"
MATRIX=OUT/"stage8_completion_matrix.json"
MANIFEST=OUT/"stage8_block810_manifest.json"
REPORT=S8/"reports/stage8_block810_final_reporting_reproducibility_report.json"
SEAL=OUT/"stage8_block810_final_reproducibility_seal.json"
SUPER=OUT/"stage8_pdf_compliance_complete_superseding_seal.json"

EXPECTED_CONTRACT="ca5fe9d4a132b88af297f822ba48c25ff7b153bc2fb3f42068ced5ce5f197b74"
EXPECTED={
 str(S7_PDF):"9f483cc19546b93ed5a28e045965eba39494d7aca9b2ba949fc49584965b0b0a",
 str(S7_C3):"e01ba2e735d4d6403a584f8f1a304e3b3acb926d36fb3adb3cb0f62c5cc44d8a",
 str(S88):"f287399f7ddc6070c45f5c42502ae90fc4db0622f6cf42a845aa94d2b8ff8cb4",
 str(REC88):"1984d777f8e649089c7f14ed80df8cd105838496ca03c70adce54fe124dd164b",
 str(SUM88):"8f29fd2b3882ffb28b66fdef87a53490eeaf64fe3b501d788495710a989fa0ac",
 str(CTRL88):"8632cfbca75cedc1479730c9eba576492f09fb2fbff3ec4c7042a2f5a9cfb48f",
 str(CTRLSEAL88):"b1f99a850dfb78edf08f4132db9617d947038a029703c55075ff35d51ca3b113",
 str(BOOT89):"a60faacc0c193fad5f8324aefc2dc80036053cf17d574b31790fce104f2bc1ae",
 str(PAIRED89):"3fe13ed2fa61b850b2aa799e0718768c8ec769b61be942577e7caab9d85ed9a0",
 str(Q89):"e0d037deff7c955552005a14da247fa9866996a7558d59b6ef8ea90cdc41d8bc",
 str(FAIL89):"99821b0a0b17e4f1d2dad2b20e6284509be07ea13962a06a95285f3d346a6838",
 str(MAN89):"b0e1f4eb0f7517d888e0da561f3da1530865550986e72975aabf76ce67f80fe4",
 str(REPORT89):"363cb799bfd1d824bbf846ac67545f60f603b8a85f018d17d3de57191527a748",
 str(SEAL89):"73c74faab50ff8b52d76089e093439414894a3e95d87e32348da845e05798a29",
 str(RUNNER89):"38a1be31ae995217e3e55a59f4a370bde53a44ac53e860e5b91a1e9ce9ce47cd",
 str(SEAL85C):"b0dae15111baae6a0cbe132e402459148474c6bad87c58c5a71eccba737de91e",
 str(SEAL86C):"ff895d100129e7f3f1e8c67e75c5d067fe7f0b3aaf40f29717c730e01a6ed60b",
 str(SEAL87C):"01b0cb67f10cae247fab56bd8214d32a2ea815c7ff9ef88589d4d2f11676e67a",
}

METHOD_ORDER=[
 "adaptive_topk_q090","adaptive_topk_q095","adaptive_topk_q0975","adaptive_topk_q099",
 "fixed_top1","fixed_top3","fixed_top5","exhaustive_64","oracle_gt_best_beam_k1",
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

def finite_tree(x):
    if isinstance(x,float): return math.isfinite(x)
    if isinstance(x,dict): return all(finite_tree(v) for v in x.values())
    if isinstance(x,list): return all(finite_tree(v) for v in x)
    return True

def main():
    print("="*100)
    print("STAGE 8 — BLOCK 8.10")
    print("OPTICAL/mmWAVE REPORTING SEPARATION + FINAL STAGE8 REPRODUCIBILITY SEAL")
    print("ANALYSIS/REPORTING ONLY — NO SCIENTIFIC REEXECUTION")
    print("="*100)

    req(Path(__file__).resolve()==SCRIPT.resolve(),"runner path mismatch")
    req(CONTRACT.is_file() and sha(CONTRACT)==EXPECTED_CONTRACT,"8.10 contract SHA mismatch")
    for raw,e in EXPECTED.items():
        p=Path(raw); req(p.is_file(),f"missing frozen input: {p}"); req(sha(p)==e,f"frozen input SHA changed: {p}")
    for p in [SEP,TABLE,MATRIX,MANIFEST,REPORT,SEAL,SUPER]:
        req(not p.exists(),f"8.10 output exists: {p}")

    print("\n===== A. FROZEN STAGE7 / STAGE8 PROVENANCE GATE =====")
    s88=json.loads(S88.read_text())
    s89=json.loads(SEAL89.read_text())
    s85=json.loads(SEAL85C.read_text())
    s86=json.loads(SEAL86C.read_text())
    s87=json.loads(SEAL87C.read_text())

    req(s85.get("status")=="FROZEN_COMPLETE_BLOCK85C","8.5C status changed")
    req(s86.get("status")=="FROZEN_COMPLETE_BLOCK86C","8.6C status changed")
    req(s87.get("status")=="FROZEN_COMPLETE_BLOCK87C","8.7C status changed")
    req(s88.get("status")=="FROZEN_COMPLETE_BLOCK88C","8.8C status changed")
    req(s89.get("status")=="FROZEN_COMPLETE_BLOCK89","8.9 status changed")
    req(s88.get("formal_power_accessed") is True,"8.8C formal access flag changed")
    req(s88.get("controller_reexecution_after_power_open") is False,"8.8C causal barrier changed")
    req(s88.get("measured_power_controller_input") is False,"8.8C controller power boundary changed")
    req(s88.get("post_outcome_tuning") is False,"8.8C post-outcome tuning flag changed")
    req(s89.get("analysis_only") is True,"8.9 analysis-only status changed")
    req(s89.get("raw_power_files_reopened") is False,"8.9 raw-power reopening flag changed")
    req(s89.get("model_controller_evaluator_reexecution") is False,"8.9 reexecution flag changed")
    req(s89.get("post_outcome_tuning") is False,"8.9 post-outcome tuning flag changed")
    req(s89.get("Block810_may_start") is True,"8.10 not authorized")

    print("Stage7 final optical provenance hashes = PASS")
    print("Stage8 corrected pre-formal chain 8.5C/8.6C/8.7C = PASS")
    print("Stage8 formal Block8.8C = PASS")
    print("Stage8 statistics Block8.9 = PASS")
    print("scientific reexecution in Block8.10 = NO")

    print("\n===== B. OPTICAL / DEEPSENSE mmWAVE REPORTING SEPARATION =====")
    sep={
        "stage":8,"block":"8.10","status":"FROZEN_OPTICAL_MMWAVE_REPORTING_SEPARATION",
        "required_statement":"DeepSense validates the policy, not the optical headlamp.",
        "optical_pc_fmcw":{
            "provenance_stage":7,
            "pdf_compliance_complete_superseding_seal_sha256":EXPECTED[str(S7_PDF)],
            "final_c3_reproducibility_seal_sha256":EXPECTED[str(S7_C3)],
            "metrics_family":["pointing loss","optical SNR","DPSK BER","effective optical rate"],
            "result_type":"model-based optical/PC-FMCW",
        },
        "deepsense_mmwave":{
            "provenance_stage":8,
            "formal_records_sha256":EXPECTED[str(REC88)],
            "formal_summary_sha256":EXPECTED[str(SUM88)],
            "bootstrap_sha256":EXPECTED[str(BOOT89)],
            "metrics_family":["measured power loss","Top-K coverage","normalized spectral-efficiency gap","outage","probing overhead"],
            "result_type":"measured DeepSense mmWave beam-policy validation",
            "absolute_measured_spectral_efficiency_claim":False,
        },
        "cross_modal_pooling":False,
        "cross_modal_averaging":False,
        "cross_modal_joint_confidence_interval":False,
        "direct_optical_hardware_validation_claim_from_deepsense":False,
        "tables_must_remain_separate":True,
    }
    atom_json(SEP,sep)
    print("OPTICAL/PC-FMCW results = separate Stage7 provenance")
    print("DEEPSENSE measured mmWave results = separate Stage8 provenance")
    print("cross-modal pooling/averaging = NO")
    print('required statement = "DeepSense validates the policy, not the optical headlamp."')

    print("\n===== C. FINAL DEEPSENSE MEASURED-mmWAVE TABLE =====")
    summ=json.loads(SUM88.read_text())
    boot=json.loads(BOOT89.read_text())
    req(summ.get("formal_rows")==1030,"8.8C formal row count changed")
    req(boot.get("replicates")==10000 and boot.get("seed")==20260902,"8.9 bootstrap design changed")

    table={
        "stage":8,"block":"8.10","status":"FROZEN_DEEPSENSE_MEASURED_MMWAVE_FINAL_TABLE",
        "dataset":"DeepSense primary LiDAR-aided beam prediction, Scenario8/9 official dev_test",
        "formal_rows":1030,
        "scenario_counts":summ["scenario_counts"],
        "label_adapter":"beam_index_1 - 1",
        "primary_policy":"adaptive_topk_q095",
        "metrics_note":"spectral-efficiency values are normalized proxy gaps under frozen reference-SNR semantics; not absolute measured SE",
        "bootstrap":{"replicates":10000,"seed":20260902,"ci":"95% percentile, paired and scenario-stratified"},
        "methods":{},
    }

    for m in METHOD_ORDER:
        req(m in summ["methods"] and m in boot["methods"],f"missing method: {m}")
        table["methods"][m]={}
        for agg in ["Scenario8","Scenario9","pooled"]:
            s=summ["methods"][m][agg]
            b=boot["methods"][m][agg]
            table["methods"][m][agg]={
                "n":s["n"],
                "coverage":{"point":s["coverage"],"ci95":b["coverage"]["ci95"]},
                "mean_K":{"point":s["mean_K"],"ci95":b["mean_K"]["ci95"]},
                "probing_overhead":{"point":s["mean_probing_overhead"],"ci95":b["probing_overhead"]["ci95"]},
                "measured_power_loss_db":{"point":s["mean_measured_power_loss_db"],"ci95":b["measured_power_loss_db"]["ci95"]},
                "outage_1db":{"point":s["outage_rate"]["1dB"],"ci95":b["outage_1db"]["ci95"]},
                "outage_3db":{"point":s["outage_rate"]["3dB"],"ci95":b["outage_3db"]["ci95"]},
                "outage_6db":{"point":s["outage_rate"]["6dB"],"ci95":b["outage_6db"]["ci95"]},
                "normalized_se_gap_10db":{
                    "point":s["mean_normalized_se_gap_bps_per_hz"]["10dB"],
                    "ci95":b["normalized_se_gap_10db"]["ci95"],
                },
            }
    req(finite_tree(table),"nonfinite final table")
    atom_json(TABLE,table)

    p=table["methods"]["adaptive_topk_q095"]["pooled"]
    print(f"primary q=.95 coverage = {p['coverage']['point']:.12f} "
          f"CI95 [{p['coverage']['ci95']['lower']:.12f}, {p['coverage']['ci95']['upper']:.12f}]")
    print(f"primary q=.95 mean K = {p['mean_K']['point']:.12f} "
          f"CI95 [{p['mean_K']['ci95']['lower']:.12f}, {p['mean_K']['ci95']['upper']:.12f}]")
    print(f"primary q=.95 power loss dB = {p['measured_power_loss_db']['point']:.12f} "
          f"CI95 [{p['measured_power_loss_db']['ci95']['lower']:.12f}, {p['measured_power_loss_db']['ci95']['upper']:.12f}]")
    print(f"primary q=.95 outage@3dB = {p['outage_3db']['point']:.12f} "
          f"CI95 [{p['outage_3db']['ci95']['lower']:.12f}, {p['outage_3db']['ci95']['upper']:.12f}]")

    print("\n===== D. STAGE8 PDF-CORE COMPLETION MATRIX =====")
    matrix={
        "stage":8,"block":"8.10","status":"STAGE8_PDF_CORE_COMPLETION_MATRIX",
        "requirements":{
            "DeepSense primary measured-power interface frozen":True,
            "official train/calibration/formal split frozen":True,
            "measured-mmWave evaluator semantics frozen pre-outcome":True,
            "beam-policy interface and fail-closed fallback frozen pre-outcome":True,
            "DeepSense-specific lightweight encoder trained":True,
            "1-based official label to 0-based power/codebook mapping frozen pre-formal":True,
            "probability calibration frozen before formal evaluation":True,
            "pre-formal end-to-end policy/fallback replay passed":True,
            "formal controller decisions frozen before measured-power decode":True,
            "Top-K policy evaluated on measured DeepSense powers":True,
            "q=.90/.95/.975/.99 evaluated":True,
            "fixed Top-1/3/5 and exhaustive-64 evaluated":True,
            "evaluator-only GT-best-beam reference evaluated":True,
            "paired scenario-stratified bootstrap 10000 CIs completed":True,
            "descriptive failure-case analysis completed":True,
            "common-overhead comparison completed":True,
            "optical and measured-mmWave reporting separated":True,
            "DeepSense interpreted as policy validation, not optical-headlamp validation":True,
            "no cross-modal pooling":True,
            "no post-outcome tuning":True,
            "zero-shot WOMD-to-DeepSense required for core closure":False,
        },
        "optional_nonblocking":{
            "zero_shot_WOMD_to_DeepSense":"NOT_REQUIRED_FOR_STAGE8_PDF_CORE_COMPLETE"
        },
        "status_summary":"STAGE8_PDF_CORE_COMPLETE",
    }
    req(all(v is True for k,v in matrix["requirements"].items() if k!="zero-shot WOMD-to-DeepSense required for core closure"),
        "mandatory Stage8 completion requirement false")
    req(matrix["requirements"]["zero-shot WOMD-to-DeepSense required for core closure"] is False,
        "zero-shot optionality changed")
    atom_json(MATRIX,matrix)
    print("mandatory Stage8 DeepSense policy validation = COMPLETE")
    print("statistical CIs / failure analysis = COMPLETE")
    print("optical/mmWave separation = COMPLETE")
    print("zero-shot WOMD→DeepSense = OPTIONAL / NON-BLOCKING")

    atom_json(MANIFEST,{
        "stage":8,"block":"8.10","status":"FROZEN_COMPLETE_BLOCK810",
        "contract_sha256":EXPECTED_CONTRACT,
        "stage7_pdf_seal_sha256":EXPECTED[str(S7_PDF)],
        "stage7_c3_seal_sha256":EXPECTED[str(S7_C3)],
        "Block85C_seal_sha256":EXPECTED[str(SEAL85C)],
        "Block86C_seal_sha256":EXPECTED[str(SEAL86C)],
        "Block87C_seal_sha256":EXPECTED[str(SEAL87C)],
        "Block88C_seal_sha256":EXPECTED[str(S88)],
        "Block89_seal_sha256":EXPECTED[str(SEAL89)],
        "formal_records_sha256":EXPECTED[str(REC88)],
        "bootstrap_sha256":EXPECTED[str(BOOT89)],
        "reporting_separation_sha256":sha(SEP),
        "deepsense_final_table_sha256":sha(TABLE),
        "completion_matrix_sha256":sha(MATRIX),
        "runner_sha256":sha(SCRIPT),
        "analysis_only":True,
        "raw_power_files_reopened":False,
        "scientific_reexecution":False,
        "post_outcome_tuning":False,
    })

    atom_json(REPORT,{
        "stage":8,"block":"8.10","status":"PASS_BLOCK810_FINAL_REPORTING_REPRODUCIBILITY",
        "stage8_core_status":"STAGE8_PDF_CORE_COMPLETE",
        "primary_policy":"adaptive_topk_q095",
        "primary_pooled":p,
        "optical_mmwave_separate":True,
        "cross_modal_pooling":False,
        "deepSense_validates_policy_not_optical_headlamp":True,
        "zero_shot_optional_nonblocking":True,
        "formal_outcomes_modified":False,
        "raw_power_files_reopened":False,
        "scientific_reexecution":False,
        "post_outcome_tuning":False,
        "reporting_separation_sha256":sha(SEP),
        "deepsense_final_table_sha256":sha(TABLE),
        "completion_matrix_sha256":sha(MATRIX),
        "manifest_sha256":sha(MANIFEST),
        "runner_sha256":sha(SCRIPT),
    })

    atom_json(SEAL,{
        "stage":8,"block":"8.10","status":"FROZEN_COMPLETE_BLOCK810",
        "stage8_core_status":"STAGE8_PDF_CORE_COMPLETE",
        "stage7_pdf_seal_sha256":EXPECTED[str(S7_PDF)],
        "stage7_c3_seal_sha256":EXPECTED[str(S7_C3)],
        "Block85C_seal_sha256":EXPECTED[str(SEAL85C)],
        "Block86C_seal_sha256":EXPECTED[str(SEAL86C)],
        "Block87C_seal_sha256":EXPECTED[str(SEAL87C)],
        "Block88C_seal_sha256":EXPECTED[str(S88)],
        "Block89_seal_sha256":EXPECTED[str(SEAL89)],
        "formal_records_sha256":EXPECTED[str(REC88)],
        "formal_summary_sha256":EXPECTED[str(SUM88)],
        "bootstrap_sha256":EXPECTED[str(BOOT89)],
        "paired_comparisons_sha256":EXPECTED[str(PAIRED89)],
        "q_tradeoff_sha256":EXPECTED[str(Q89)],
        "failure_slices_sha256":EXPECTED[str(FAIL89)],
        "contract_sha256":EXPECTED_CONTRACT,
        "runner_sha256":sha(SCRIPT),
        "reporting_separation_sha256":sha(SEP),
        "deepsense_final_table_sha256":sha(TABLE),
        "completion_matrix_sha256":sha(MATRIX),
        "manifest_sha256":sha(MANIFEST),
        "report_sha256":sha(REPORT),
        "optical_mmwave_separate":True,
        "cross_modal_pooling":False,
        "raw_power_files_reopened":False,
        "scientific_reexecution":False,
        "post_outcome_tuning":False,
    })

    atom_json(SUPER,{
        "project":"Agni","stage":8,
        "status":"STAGE8_PDF_CORE_COMPLETE",
        "supersedes_for_stage8_reporting":"all prior partial/failed Stage8 completion attempts; all remain append-only audit evidence",
        "final_Block810_seal_sha256":sha(SEAL),
        "Block89_seal_sha256":EXPECTED[str(SEAL89)],
        "Block88C_seal_sha256":EXPECTED[str(S88)],
        "formal_records_sha256":EXPECTED[str(REC88)],
        "stage7_pdf_compliance_seal_sha256":EXPECTED[str(S7_PDF)],
        "optical_mmwave_reporting_separation_sha256":sha(SEP),
        "completion_matrix_sha256":sha(MATRIX),
        "zero_shot_WOMD_to_DeepSense":"OPTIONAL_NONBLOCKING_NOT_RUN",
        "scientific_reexecution":False,
        "post_outcome_tuning":False,
    })

    for raw,e in EXPECTED.items():
        req(sha(Path(raw))==e,f"frozen upstream modified during 8.10: {raw}")

    print("\n"+"="*100)
    print("BLOCK 8.10 = FULLY VERIFIED / FROZEN")
    print("STAGE7 OPTICAL/PC-FMCW PROVENANCE = BOUND")
    print("STAGE8 DEEPSENSE MEASURED-mmWAVE PROVENANCE = BOUND")
    print("OPTICAL/mmWAVE TABLE SEPARATION = PASS")
    print("CROSS-MODAL POOLING / AVERAGING = NO")
    print('DEEPSENSE VALIDATES THE POLICY, NOT THE OPTICAL HEADLAMP = PASS')
    print("FORMAL OUTCOMES MODIFIED = NO")
    print("RAW LiDAR / RAW MMWAVE POWER REOPENED = NO")
    print("MODEL / CONTROLLER / EVALUATOR REEXECUTION = NO")
    print("POST-OUTCOME TUNING = NO")
    print("ZERO-SHOT WOMD->DEEPSENSE = OPTIONAL / NON-BLOCKING / NOT RUN")
    print("STAGE8 PDF CORE COMPLETION = PASS")
    print("8.10 separation SHA256 =",sha(SEP))
    print("8.10 DeepSense table SHA256 =",sha(TABLE))
    print("8.10 completion matrix SHA256 =",sha(MATRIX))
    print("8.10 manifest SHA256 =",sha(MANIFEST))
    print("8.10 report SHA256 =",sha(REPORT))
    print("8.10 seal SHA256 =",sha(SEAL))
    print("8.10 superseding PDF seal SHA256 =",sha(SUPER))
    print("8.10 runner SHA256 =",sha(SCRIPT))
    print("STATUS = STAGE8_PDF_CORE_COMPLETE")
    print("="*100)

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print("\n"+"!"*100)
        print("BLOCK 8.10 FAIL-CLOSED")
        print(f"{type(e).__name__}: {e}")
        print("DO NOT MODIFY FROZEN STAGE7 OR STAGE8 SCIENTIFIC/FORMAL OUTCOME ARTIFACTS.")
        print("!"*100)
        raise
