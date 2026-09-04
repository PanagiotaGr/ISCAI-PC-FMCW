#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, os, platform, statistics, sys, tempfile
from pathlib import Path
from zipfile import ZipFile

ROOT=Path('/home/agni/waymo'); S7=ROOT/'iscai_stage7'
SCRIPT=S7/'scripts/run_block75c3c_final_latency_resources_and_seal.py'
C1_RUNNER=S7/'scripts/run_block75c1_formal_causal_output_lock.py'; C1_CHECKER=S7/'scripts/check_block75c1_nonoracle_lock.py'; C1_MANIFEST=S7/'artifacts/block75c1_nonoracle_lock_manifest.json'; C1_ROOT=S7/'artifacts/block75c1_nonoracle_lock'
C2_RUNNER=S7/'scripts/run_block75c2_final_formal_evaluator.py'; C2_MANIFEST=S7/'artifacts/block75c2_final_formal_raw_manifest.json'; C2_ROWS=S7/'artifacts/block75c2_final_joint_rows.jsonl'; C2_REPORT=S7/'reports/stage7_block75c2_final_formal_evaluation_report.json'; C2_SEAL=S7/'artifacts/stage7_block75c2_final_frozen_seal.json'
C3A_RUNNER=S7/'scripts/run_block75c3a_core_statistics.py'; C3A_ROOT=S7/'artifacts/block75c3a_core_statistics'; C3A_AGG=C3A_ROOT/'five_system_aggregate.json'; C3A_BOOT=C3A_ROOT/'paired_bootstrap_10000.json'; C3A_SLICES=C3A_ROOT/'failure_slices.json'; C3A_TRADE=C3A_ROOT/'common_tradeoff_core.json'; C3A_MANIFEST=C3A_ROOT/'block75c3a_manifest.json'; C3A_REPORT=S7/'reports/stage7_block75c3a_core_statistics_report.json'
C3B_RUNNER=S7/'scripts/run_block75c3b_final_frozen_sweeps.py'; C3B_ROOT=S7/'artifacts/block75c3b_final_frozen_sweeps'; C3B_CODEBOOK=C3B_ROOT/'codebook_sweep.json'; C3B_COVERAGE=C3B_ROOT/'coverage_sweep.json'; C3B_UNCERTAINTY=C3B_ROOT/'uncertainty_sweep.json'; C3B_MANIFEST=C3B_ROOT/'block75c3b_manifest.json'; C3B_REPORT=S7/'reports/stage7_block75c3b_final_frozen_sweeps_report.json'
SOURCE_BUNDLE=ROOT/'audits/stage7_pdf_alignment/block75_c2c3_exact_source_bundle.zip'
OUT_ROOT=S7/'artifacts/block75c3c_final_latency_resources'; RESOURCE_OUT=OUT_ROOT/'resource_latency_analysis.json'; MULT_OUT=OUT_ROOT/'latency_multiplier_sweep.json'; COMPLETENESS_OUT=OUT_ROOT/'methodology_completeness.json'; MANIFEST_OUT=OUT_ROOT/'block75c3c_manifest.json'; REPORT_OUT=S7/'reports/stage7_block75c3c_final_latency_resources_report.json'; FINAL_SEAL=S7/'artifacts/stage7_block75c3_final_reproducibility_seal.json'

EXPECTED={
str(C1_RUNNER):'077237d85e62468f90ebf21a70fe6cdf3ad095fe83dba2985d0b4cb799379275', str(C1_CHECKER):'b9bfb6447fbfe1982c6aa46e0cb5053d527112207bfc68be5acea29de095725e', str(C1_MANIFEST):'d6792d2bcf304bd51260e98d99e8f8fa1e0aa470849a2d1989bdeac5ad74bfe5',
str(C2_RUNNER):'3ccbed33cc66f89a2b366e50b923e06af7c71f0d9ad2baeedf691eef099af454', str(C2_MANIFEST):'adae0abde4377e0e6a7c9e993540b22ad53b7b0ebff038b4313d946d57684175', str(C2_ROWS):'6498e4bfe78597bca43aea7b7a7e72d2ea13befeadd1eaee4216917c26f16830', str(C2_REPORT):'616a7ae86574889ded6213b2b5f4363b80f20b45e183040282d658de65b244e7', str(C2_SEAL):'743541e5fb6b7718ef6d6634e09f13c6f7b879d106a5bcd9c0e7e8dad7d5d82c',
str(C3A_RUNNER):'3ca0f4921d09dbb97cc13ea9eee30d03afe8eb90f3437495212f7a6afbf89d73', str(C3A_AGG):'bec76ea0ed08c78623319356c89c43877fbe6e73ffe3abe7fda4835399c2eeef', str(C3A_BOOT):'5e19036f2376da5f449cdacf79ff7c4dd037412f65832380e5d56055558371b6', str(C3A_SLICES):'a07c57012bab0b74abd7ee633f1f0b818390a9ffc1b3361114e9d6b22fe2e7bb', str(C3A_TRADE):'c3a31d4a3550482a3875984d33fa3080d7b4e35fc4e3e395970886b570ecaa64', str(C3A_MANIFEST):'4c7da7e404c227414cc9e31053b9f46f7440e0ad207515addc801f7fb002fc4b', str(C3A_REPORT):'be51ebc83feef4fde84c7209864abb609e24c23cbde3fced9b437407ab570b54',
str(C3B_RUNNER):'77d7b60021d367404ca371aad89e9cc5ed8ed4af19fa9bf8bc76559f7fc7c88c', str(C3B_CODEBOOK):'829458108c3a4c1ee014274130c84905e76e939e448abaf5558e573b0f1521ad', str(C3B_COVERAGE):'a2236a7c89388fe9112d90467ee73564810836642d31de94df43ce0a529f4e64', str(C3B_UNCERTAINTY):'cf7a0c71428baebb010566ec4ddecf2404453197fcd8af9e60499cb92af23995', str(C3B_MANIFEST):'eda5ea7c9c4b1e62f3196482959cd142ee1ab9684d278b1b8ab4fc02e9ba411b', str(C3B_REPORT):'d258495a2db314e94ba8d15f96242c4a86d86e9d644e1893fceb02756bc68b5a',
str(SOURCE_BUNDLE):'8ce6594f1fe53a3f1fbb1bfa210ad475598818d75a4c82fe1fc31e19b2971282'}
SYSTEMS=['shared_trajectory_posterior','independent_models','direct_beam_classifier','direct_ADB_predictor','deterministic_shared_trajectory']; LATENCY_MULTIPLIERS=[0.5,1.0,2.0]
BUNDLE_MEMBERS={'beam_latency_policy':'iscai_stage5/configs/beam_latency_policy.json','beam_latency_runtime':'iscai_stage5/src/iscai_stage5/beam_latency.py','stage5_formal_runner':'iscai_stage5/scripts/run_block58_part2_formal_evaluation.py','stage6_resources':'iscai_stage6/reports/stage6_experiment5_latency_resources.json','stage6_metric_semantics':'iscai_stage6/src/iscai_stage6/adb/metric_semantics.py','stage7_handoff':'iscai_stage7/configs/stage7_block74_MINIMAL_HANDOFF.json','stage7_eval_contract':'iscai_stage7/configs/stage7_block75b_final_formal_evaluator_contract.json'}

class FailClosed(RuntimeError): pass
def require(c,m):
    if not c: raise FailClosed(m)
def sha256_path(p):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
def cbytes(o): return (json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
def atomic_json(p,o):
    p.parent.mkdir(parents=True,exist_ok=True); require(not p.exists(),f'Refusing overwrite: {p}'); data=cbytes(o); fd,tmp=tempfile.mkstemp(prefix=p.name+'.tmp.',dir=str(p.parent))
    try:
        with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,p); dfd=os.open(p.parent,os.O_DIRECTORY); os.fsync(dfd); os.close(dfd)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
def jload(p): return json.loads(p.read_text(encoding='utf-8'))
def finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(float(x))
def walk(o,p=()):
    if isinstance(o,dict):
        for k in sorted(o): yield from walk(o[k],p+(str(k),))
    elif isinstance(o,list):
        for i,v in enumerate(o): yield from walk(v,p+(str(i),))
    else: yield p,o
def unit(k):
    k=k.lower()
    if k.endswith('_ms') or 'milliseconds' in k: return 'ms'
    if k.endswith('_us') or 'microseconds' in k: return 'us'
    if k.endswith('_ns') or 'nanoseconds' in k: return 'ns'
    if k.endswith('_s') or 'seconds' in k: return 's'
    if 'mib' in k:return 'MiB'
    if 'gib' in k:return 'GiB'
    if 'kib' in k:return 'KiB'
    if 'bytes' in k:return 'bytes'
    if 'flop' in k:return 'FLOP'
    if 'parameter' in k or 'param_count' in k or 'num_params' in k:return 'count'
    return None
def sec(v,u): return float(v) if u=='s' else float(v)/1e3 if u=='ms' else float(v)/1e6 if u=='us' else float(v)/1e9 if u=='ns' else None
def evidence(doc,label):
    out=[]
    for parts,v in walk(doc):
        if not finite(v): continue
        s='.'.join(parts).lower(); u=unit(parts[-1] if parts else ''); cls=[]
        if any(t in s for t in ('latency','runtime','elapsed','duration','time_s','time_ms')): cls.append('latency_or_runtime')
        if any(t in s for t in ('memory','rss','resident','footprint')): cls.append('memory')
        if any(t in s for t in ('parameter','param_count','num_params')): cls.append('parameters')
        if 'flop' in s: cls.append('flops')
        if not cls: continue
        r={'source':label,'path':'.'.join(parts),'value':float(v),'unit_from_explicit_key':u,'classes':cls}; ss=sec(v,u)
        if ss is not None and 'latency_or_runtime' in cls:r['seconds']=ss
        out.append(r)
    return out
def comp(rows,alts):
    hits={}
    for terms in alts:
        for r in rows:
            if 'seconds' in r and all(t.lower() in r['path'].lower() for t in terms): hits[r['path']]=r
    vals=[hits[k] for k in sorted(hits)]
    if len(vals)==1:return {'status':'EVALUABLE_FROZEN_AUTHORITY','seconds':vals[0]['seconds'],'source_path':vals[0]['path'],'source':vals[0]['source']}
    if not vals:return {'status':'NOT_EVALUABLE','reason':'no exact frozen scalar matched component semantics'}
    return {'status':'NOT_EVALUABLE','reason':'multiple frozen scalars matched; refusing arbitrary choice','candidates':vals}
def c1diag():
    rows=[]; js=sorted(C1_ROOT.glob('*/lock.json')) if C1_ROOT.is_dir() else []
    for p in js:
        try:d=jload(p)
        except:continue
        for parts,v in walk(d):
            if not finite(v):continue
            s='.'.join(parts).lower(); u=unit(parts[-1] if parts else '')
            if any(t in s for t in ('latency','runtime','elapsed','duration','timing')):
                r={'record':p.parent.name,'path':'.'.join(parts),'value':float(v),'unit_from_key':u}; ss=sec(v,u)
                if ss is not None:r['seconds']=ss
                rows.append(r)
    return {'status':'DIAGNOSTIC_ONLY_EXCLUDED_FROM_FORMAL_LATENCY','lock_json_files_seen':len(js),'numeric_timing_like_scalars':len(rows),'rows':rows}

def main():
    print('='*88); print('STAGE 7 — BLOCK 7.5C3C FINAL'); print('LATENCY / RESOURCES + LATENCY SENSITIVITY + FINAL REPRODUCIBILITY SEAL'); print('ANALYSIS-ONLY; NO MODEL FORWARD; NO C1/C2/C3A/C3B MODIFICATION'); print('='*88)
    require(Path(__file__).resolve()==SCRIPT.resolve(),f'Runner must be installed at {SCRIPT}; actual={Path(__file__).resolve()}')
    for p in [OUT_ROOT,RESOURCE_OUT,MULT_OUT,COMPLETENESS_OUT,MANIFEST_OUT,REPORT_OUT,FINAL_SEAL]: require(not p.exists(),f'Final C3C target already exists: {p}')
    print('\n===== A. FROZEN IDENTITY GATE =====')
    for raw,exp in EXPECTED.items():
        p=Path(raw); require(p.is_file(),f'Missing frozen input: {p}'); act=sha256_path(p); require(act==exp,f'Frozen SHA mismatch: {p}: {act} != {exp}')
    print('C1/C2/C3A/C3B/source-bundle exact hashes = PASS')
    c1m=jload(C1_MANIFEST); c2m=jload(C2_MANIFEST); require(len(c1m.get('scenario_locks',[]))==120,'C1 N120 changed.'); require(len(c2m.get('scenario_records',[]))==120,'C2 N120 changed.'); require(sum(1 for _ in C2_ROWS.open('r',encoding='utf-8'))==600,'C2 N600 changed.'); print('120 scenarios / 600 rows = PASS')
    print('\n===== B. EXACT LATENCY / RESOURCE AUTHORITIES =====')
    with ZipFile(SOURCE_BUNDLE,'r') as z:
        names=set(z.namelist()); [require(m in names,f'Missing authority member {m}') for m in BUNDLE_MEMBERS.values()]; bp=z.read(BUNDLE_MEMBERS['beam_latency_policy']); sr=z.read(BUNDLE_MEMBERS['stage6_resources'])
    beam=json.loads(bp.decode()); adb=json.loads(sr.decode()); hbp=ROOT/BUNDLE_MEMBERS['beam_latency_policy']; hsr=ROOT/BUNDLE_MEMBERS['stage6_resources']; require(hbp.read_bytes()==bp,'Host beam policy differs from frozen bundle.'); require(hsr.read_bytes()==sr,'Host Stage6 resources differs from frozen bundle.'); print('Stage5/Stage6 byte identity = PASS')
    be=evidence(beam,'frozen_stage5_beam_latency_policy'); ae=evidence(adb,'frozen_stage6_latency_resources'); rows=be+ae
    print('\n===== C. FORMAL LATENCY / RESOURCE EXTRACTION ====='); print('Stage5 resource-like scalars =',len(be)); print('Stage6 resource-like scalars =',len(ae))
    comps={'data_loading_latency':comp(rows,[('data','load','latency'),('loading','latency')]),'preprocessing_latency':comp(rows,[('preprocess','latency'),('preprocessing','runtime')]),'predictor_inference_latency':comp(rows,[('predictor','inference'),('inference','latency')]),'beam_selection_latency':comp(rows,[('beam','selection','latency'),('beam','selection','runtime')]),'adb_generation_latency':comp(rows,[('adb','generation','latency'),('actuation','latency')])}
    ev={k:v for k,v in comps.items() if v['status']=='EVALUABLE_FROZEN_AUTHORITY'}; missing=[k for k in comps if k not in ev]
    full={'status':'EVALUABLE_COMPONENT_SUM','seconds':sum(v['seconds'] for v in ev.values()),'formula':'data_loading + preprocessing + predictor_inference + beam_selection + ADB_generation','components':comps} if not missing else {'status':'NOT_EVALUABLE_FROM_EXACT_FROZEN_C3_AUTHORITIES','reason':'refusing diagnostic promotion/imputation','missing_components':missing,'components':comps}
    five={s:{'status':'EVALUABLE_FROZEN_AUTHORITY' if (rr:=[r for r in rows if s.lower() in r['path'].lower()]) else 'NOT_EVALUABLE','evidence':rr,'reason':None if rr else 'no explicit per-system resource scalar in exact C3 authority bundle'} for s in SYSTEMS}
    rdoc={'block':'7.5C3C','analysis_semantics':{'new_model_forward':False,'retraining':False,'recalibration':False,'C1_diagnostic_timings_promoted_to_formal':False,'resource_values_fabricated':False},'authority':{'source_bundle':str(SOURCE_BUNDLE),'source_bundle_sha256':EXPECTED[str(SOURCE_BUNDLE)],'beam_latency_policy_sha256':hashlib.sha256(bp).hexdigest(),'stage6_resource_sha256':hashlib.sha256(sr).hexdigest()},'formal_resource_evidence':rows,'full_joint_end_to_end_latency':full,'five_system_resources':five,'C1_frozen_timing_telemetry':c1diag(),'C1_telemetry_scientific_role':'DIAGNOSTIC_ONLY_EXCLUDED_FROM_FORMAL_LATENCY_CLAIMS'}; atomic_json(RESOURCE_OUT,rdoc); print('resource/latency publication = PASS')
    print('\n===== D. LATENCY MULTIPLIER SWEEP ====='); base=[r for r in rows if 'seconds' in r and 'latency_or_runtime' in r['classes']]; ents=[]
    for m in LATENCY_MULTIPLIERS:
        e={'multiplier':m,'semantics':'offline deterministic sensitivity transform; no rerun','rows':[{'source':r['source'],'path':r['path'],'baseline_seconds':r['seconds'],'latency_multiplier':m,'sensitivity_seconds':r['seconds']*m} for r in base]}; e['full_joint_sensitivity_seconds']=full['seconds']*m if full['status']=='EVALUABLE_COMPONENT_SUM' else None; ents.append(e)
    atomic_json(MULT_OUT,{'latency_multipliers':LATENCY_MULTIPLIERS,'baseline_multiplier':1.0,'new_model_forward':False,'post_outcome_tuning':False,'entries':ents}); print('latency multipliers {0.5,1,2} = PASS')
    completeness={'paired_bootstrap_10000':'PASS_C3A','failure_slice_analysis':'PASS_C3A','common_tradeoff':'PASS_C3A','codebook_sweep_16_32_64':'PASS_C3B','coverage_sweep':'PASS_C3B','uncertainty_sweep':'PASS_C3B','latency_multiplier_sweep':'PASS_C3C','formal_latency_resource_evidence':'PASS_NO_FABRICATION','full_joint_end_to_end_latency':{'status':full['status'],'missing_components':missing,'policy':'NOT_EVALUABLE preferred over imputation'}}; atomic_json(COMPLETENESS_OUT,completeness); print('methodology completeness accounting = PASS')
    manifest={'block':'7.5C3C','status':'FROZEN_COMPLETE_C3C','inputs':{p:EXPECTED[p] for p in sorted(EXPECTED)},'outputs':[{'path':str(p),'sha256':sha256_path(p)} for p in [RESOURCE_OUT,MULT_OUT,COMPLETENESS_OUT]],'causality':{'new_model_forward':False,'future_GT_reopened':False,'controllers_rerun':False,'post_outcome_tuning':False,'C1_modified':False,'C2_modified':False,'C3A_modified':False,'C3B_modified':False}}; atomic_json(MANIFEST_OUT,manifest)
    report={'block':'7.5C3C','status':'FROZEN_COMPLETE_C3C','resource_latency_analysis':str(RESOURCE_OUT),'latency_multiplier_sweep':str(MULT_OUT),'methodology_completeness':str(COMPLETENESS_OUT),'full_joint_end_to_end_latency_status':full['status'],'declared_missing_full_joint_components':missing,'scientific_integrity':'NO_FABRICATION_NO_DIAGNOSTIC_PROMOTION','runner_sha256':sha256_path(SCRIPT),'manifest_sha256':sha256_path(MANIFEST_OUT)}; atomic_json(REPORT_OUT,report); print('C3C manifest/report = PASS')
    print('\n===== E. UPSTREAM IMMUTABILITY RECHECK =====')
    for raw,exp in EXPECTED.items(): require(sha256_path(Path(raw))==exp,f'Upstream modified during C3C: {raw}')
    print('C1/C2/C3A/C3B/source bundle unchanged = PASS')
    seal_inputs=[C2_SEAL,C3A_MANIFEST,C3A_REPORT,C3B_MANIFEST,C3B_REPORT,MANIFEST_OUT,REPORT_OUT,RESOURCE_OUT,MULT_OUT,COMPLETENESS_OUT]
    seal={'stage':'Stage7','block':'7.5C3','status':'FROZEN_COMPLETE_C3','five_systems':SYSTEMS,'bootstrap':{'replicates':10000,'seed':2338514766,'status':'PASS_C3A'},'sweeps':{'codebook_sizes':[16,32,64],'coverage':[0.90,0.95,0.975,0.99],'uncertainty_alpha':[0.5,1.0,1.5,2.0],'latency_multipliers':LATENCY_MULTIPLIERS},'full_joint_end_to_end_latency_status':full['status'],'non_evaluable_is_not_imputed':True,'causality':{'future_GT_controller_input':False,'new_model_forward_in_C3':False,'controllers_rerun_after_C2_future_GT':False,'post_outcome_tuning':False},'immutable_artifacts':[{'path':str(p),'sha256':sha256_path(p)} for p in seal_inputs],'runner':{'path':str(SCRIPT),'sha256':sha256_path(SCRIPT)},'environment_descriptor':{'python':sys.version.split()[0],'platform':platform.platform()}}; atomic_json(FINAL_SEAL,seal)
    print('\n'+'='*88); print('BLOCK 7.5C3C = FULLY VERIFIED / FROZEN'); print('FROZEN LATENCY/RESOURCE AUTHORITIES = PASS'); print('LATENCY MULTIPLIER SWEEP .5/1/2 = PASS'); print('NO DIAGNOSTIC TIMING PROMOTED TO FORMAL = PASS'); print('NO FABRICATED MEMORY/PARAMETER/FLOP VALUES = PASS'); print('FULL JOINT END-TO-END LATENCY =',full['status']); print('C1 MODIFIED = NO'); print('C2 MODIFIED = NO'); print('C3A MODIFIED = NO'); print('C3B MODIFIED = NO'); print('NEW MODEL FORWARD = NO'); print('POST-OUTCOME TUNING = NO'); print('C3 FINAL SEAL = PASS'); print('C3C manifest SHA256 =',sha256_path(MANIFEST_OUT)); print('C3C report SHA256   =',sha256_path(REPORT_OUT)); print('C3 final seal SHA256 =',sha256_path(FINAL_SEAL)); print('C3C runner SHA256   =',sha256_path(SCRIPT)); print('STATUS = FROZEN_COMPLETE_C3'); print('='*88); return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print('\n'+'!'*88,file=sys.stderr); print('BLOCK 7.5C3C FAIL-CLOSED',file=sys.stderr); print(f'{type(exc).__name__}: {exc}',file=sys.stderr); print('DO NOT DELETE/MODIFY ANY C3C OUTPUT THAT EXISTS.',file=sys.stderr); print('C3 FINAL SEAL = NO',file=sys.stderr); print('!'*88,file=sys.stderr); raise
