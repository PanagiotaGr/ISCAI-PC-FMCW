#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, os, random, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
ROOT=Path('/home/agni/waymo'); S8=ROOT/'iscai_stage8'; DATA_ROOT=S8/'data/block82b_deepsense_primary'
SCRIPT=S8/'scripts/run_stage8_block85b_cpu_deterministic_repair.py'
MODEL_SRC=S8/'src/deepsense_lidar_encoder.py'; LOADER_SRC=S8/'src/mat5_lidar_loader.py'
B85A=S8/'artifacts/block85a_encoder_preregistration/stage8_block85a_encoder_preregistration_seal.json'
B82B_FILES=S8/'artifacts/block82b_canonical_corpus_schema_split/canonical_extracted_file_manifest.json'
FAILED_RUNNER=S8/'scripts/run_stage8_block85b_train_frozen_encoder.py'
OUT=S8/'artifacts/block85b_encoder_training_cpu_repair'; NORM=OUT/'train_only_normalization.json'; IDX=OUT/'frozen_train_sample_index.json'; LOG=OUT/'training_log.json'; CKPT=OUT/'duallidarconv1d64_epoch080.pt'; REPORT=S8/'reports/stage8_block85b_encoder_training_cpu_repair_report.json'; SEAL=OUT/'stage8_block85b_encoder_training_cpu_repair_seal.json'
EXPECTED={str(B85A):'2d43c234dbe42b86f3f3f58a57d582c9ea92253fbe62dd3091d4fd91cff98fb4',str(B82B_FILES):'79dc92fb6a90acd36c4598de436517ebac6c7e7904334dd76a520e1c90a0b895',str(MODEL_SRC):'adb943f12490612a6b92ffa3e70f69b96b1002fa77e4e769f5ea26ed6d25c417',str(LOADER_SRC):'b4b3d9e09a20a5ee790a4ad7fd22f53dad1d6f4b953e70f824ade80f9b86299c',str(FAILED_RUNNER):'73c62d52cd3d28edbf85f08b8e046be40c079214fd0baf17e3a4d0b1a194347b'}
SEED=20260902
class FailClosed(RuntimeError): pass
def req(c,m):
    if not c: raise FailClosed(m)
def sha(p):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def atom_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); req(not path.exists(),f'overwrite {path}')
    data=(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n').encode(); fd,tmp=tempfile.mkstemp(prefix=path.name+'.tmp.',dir=str(path.parent))
    try:
        with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
def atom_torch(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); req(not path.exists(),f'overwrite {path}')
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.tmp.',dir=str(path.parent)); os.close(fd)
    try:
        torch.save(obj,tmp)
        with open(tmp,'rb') as f: os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def resolve(csvp,s):
    s=s.strip().replace('\\','/').lstrip('./'); return (DATA_ROOT/s).resolve() if s.startswith('LiDAR/') else (csvp.parent/s).resolve()
def records():
    out=[]
    for sc,name in [(8,'scenario8_dev_train.csv'),(9,'scenario9_dev_train.csv')]:
        csvp=DATA_ROOT/f'LiDAR/Scenario{sc}/development_dataset/{name}'
        with csvp.open(newline='',encoding='utf-8-sig') as f:
            r=csv.DictReader(f)
            for rn,row in enumerate(r,start=2):
                a=resolve(csvp,row['unit1_lidar_1']); b=resolve(csvp,row['unit1_lidar_SCR_1']); y=int(float(row['beam_index_1']))
                req(0<=y<64 and a.is_file() and b.is_file(),'bad train row')
                sid=hashlib.sha256(f'{sc}|TRAIN|{rn}|{a}|{b}|{y}'.encode()).hexdigest()[:16]
                out.append((sc,csvp,rn,a,b,y,sid))
    req(len(out)==7030,'train count'); return out
def main():
    print('='*88); print('STAGE 8 — BLOCK 8.5B CPU DETERMINISTIC REPAIR'); print('SAME FROZEN RECIPE / CPU BACKEND ONLY'); print('='*88)
    req(Path(__file__).resolve()==SCRIPT.resolve(),'runner path')
    for raw,e in EXPECTED.items(): p=Path(raw); req(p.is_file() and sha(p)==e,f'upstream changed {p}')
    for p in [NORM,IDX,LOG,CKPT,REPORT,SEAL]: req(not p.exists(),f'output exists {p}')
    recs=records(); fm=json.loads(B82B_FILES.read_text()); hmap={x['relative_path']:x['sha256'] for x in fm['files']}
    paths={r[1] for r in recs}|{r[3] for r in recs}|{r[4] for r in recs}
    def chk(p): rel=str(p.relative_to(DATA_ROOT)); req(hmap.get(rel)==sha(p),f'hash mismatch {rel}')
    with ThreadPoolExecutor(max_workers=12) as ex: list(ex.map(chk,sorted(paths)))
    loader=mod('mat5',LOADER_SRC); modelm=mod('enc',MODEL_SRC)
    raw=np.empty((7030,460,2),np.float32); scr=np.empty((7030,216,2),np.float32); y=np.empty(7030,np.int64)
    def one(t):
        i,r=t; return i,loader.load_mat5_double_matrix(r[3],'data',(460,2)).astype(np.float32),loader.load_mat5_double_matrix(r[4],'data',(216,2)).astype(np.float32),r[5]
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i,a,b,yy in ex.map(one,enumerate(recs),chunksize=16): raw[i]=a; scr[i]=b; y[i]=yy
    rm=raw.astype(np.float64).mean((0,1)); rs=raw.astype(np.float64).std((0,1)); sm=scr.astype(np.float64).mean((0,1)); ss=scr.astype(np.float64).std((0,1))
    raw=(raw-rm.astype(np.float32))/np.maximum(rs,1e-8).astype(np.float32); scr=(scr-sm.astype(np.float32))/np.maximum(ss,1e-8).astype(np.float32)
    atom_json(NORM,{'status':'FROZEN_TRAIN_ONLY_NORMALIZATION','raw_mean':rm.tolist(),'raw_std':rs.tolist(),'scr_mean':sm.tolist(),'scr_std':ss.tolist(),'source':'TRAIN only'})
    atom_json(IDX,{'status':'FROZEN_TRAIN_SAMPLE_INDEX','row_count':7030,'scenario8':2831,'scenario9':4199,'mmwave_power_payloads_opened':False})
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.use_deterministic_algorithms(True); torch.set_num_threads(12)
    device=torch.device('cpu')
    ds=TensorDataset(torch.from_numpy(np.ascontiguousarray(raw.transpose(0,2,1))),torch.from_numpy(np.ascontiguousarray(scr.transpose(0,2,1))),torch.from_numpy(y))
    g=torch.Generator().manual_seed(SEED); dl=DataLoader(ds,batch_size=128,shuffle=True,generator=g,num_workers=0,drop_last=False)
    model=modelm.DualLiDARConv1D64().to(device); req(sum(p.numel() for p in model.parameters() if p.requires_grad)==18208,'param count')
    lossfn=torch.nn.CrossEntropyLoss(); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4,betas=(0.9,0.999)); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80,eta_min=1e-5)
    elog=[]
    for ep in range(1,81):
        model.train(); ls=0.; cor=0; seen=0
        for a,b,yy in dl:
            opt.zero_grad(set_to_none=True); z=model(a,b); loss=lossfn(z,yy); req(torch.isfinite(z).all().item() and math.isfinite(float(loss.item())),'nonfinite'); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); n=yy.shape[0]; ls+=float(loss.item())*n; cor+=int((z.argmax(1)==yy).sum()); seen+=n
        req(seen==7030,'epoch sample count'); lr=float(opt.param_groups[0]['lr']); sch.step(); rec={'epoch':ep,'train_cross_entropy':ls/seen,'train_top1_accuracy':cor/seen,'learning_rate':lr}; elog.append(rec)
        if ep==1 or ep%10==0 or ep==80: print(f"epoch={ep:03d}/080 train_ce={rec['train_cross_entropy']:.8f} train_top1={rec['train_top1_accuracy']:.6f} lr={lr:.8g}")
    atom_json(LOG,{'status':'FROZEN_FIXED_80_EPOCH_TRAINING_LOG','device':'cpu','seed':SEED,'epochs':elog,'checkpoint_selection':'fixed final epoch 80 only','validation_model_selection':False,'formal_data_used':False,'measured_power_payloads_opened':False})
    atom_torch(CKPT,{'stage':8,'block':'8.5B-cpu-repair','epoch':80,'model_name':'DualLiDARConv1D64','state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'normalization':{'raw_mean':rm,'raw_std':rs,'scr_mean':sm,'scr_std':ss},'execution_backend':'cpu_deterministic_repair'})
    atom_json(REPORT,{'status':'PASS_BLOCK85B_CPU_DETERMINISTIC_REPAIR','scientific_recipe_change':False,'execution_backend_change':'CUDA->CPU only','epochs_completed':80,'train_rows':7030,'final_train_cross_entropy':elog[-1]['train_cross_entropy'],'final_train_top1_accuracy':elog[-1]['train_top1_accuracy'],'checkpoint_sha256':sha(CKPT),'formal_data_accessed':False,'measured_power_payloads_opened':False,'Block86_authorized':True})
    atom_json(SEAL,{'stage':8,'block':'8.5B-cpu-repair','status':'FROZEN_COMPLETE_BLOCK85B','Block85A_seal_sha256':EXPECTED[str(B85A)],'failed_cuda_runner_sha256':EXPECTED[str(FAILED_RUNNER)],'runner_sha256':sha(SCRIPT),'normalization_sha256':sha(NORM),'training_log_sha256':sha(LOG),'checkpoint_sha256':sha(CKPT),'report_sha256':sha(REPORT),'scientific_recipe_change':False,'execution_backend':'cpu','Block86_may_start':True})
    print('='*88); print('BLOCK 8.5B CPU REPAIR = FULLY VERIFIED / FROZEN'); print('SCIENTIFIC RECIPE CHANGE = NO'); print('EXECUTION BACKEND = CPU DETERMINISTIC'); print('TRAIN ROWS = 7,030'); print('FIXED TRAINING EPOCHS = 80/80 COMPLETE'); print('CHECKPOINT = FIXED FINAL EPOCH 80'); print('CALIBRATION/FORMAL DATA ACCESSED = NO'); print('MMWAVE POWER PAYLOADS OPENED = NO'); print('BLOCK 8.6 MAY START = YES'); print('8.5B checkpoint SHA256 =',sha(CKPT)); print('8.5B report SHA256 =',sha(REPORT)); print('8.5B seal SHA256 =',sha(SEAL)); print('8.5B runner SHA256 =',sha(SCRIPT)); print('STATUS = FROZEN_COMPLETE_BLOCK85B'); print('='*88)
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as e: print('BLOCK 8.5B CPU REPAIR FAIL-CLOSED',type(e).__name__,e); raise
