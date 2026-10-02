#!/home/agni/waymo/iscai_stage1/.venv_lidar/bin/python
"""
Stage 9.13 — Phase3-DIRECT C3 pre-outcome closure.

Frozen scientific role:
  * C1+C2 are primary confirmatory and are already frozen in R4 PRETRUTH.
  * C3 is secondary FORMAL profile sensitivity only.
  * Frozen C3 policy: V5_RHO2_DIRECT_RAW_PRIORITY, rho_dim=2, rho_bright=1.

This runner intentionally does NOT execute H5/H6 and does NOT access evaluator-side
future ground truth.  It creates the C3 causal action lock and a COMBINED PRE-OUTCOME
SEAL.  Only a successful combined seal sets FUTURE_GT_MAY_OPEN=true.

Engineering policy:
  * append-only per-scenario outputs;
  * interrupted runs may reuse already-complete exact action pairs;
  * no C1 MC, no C2 solver, no H1, no receiver reselection, no model forward;
  * no SciPy;
  * fail closed on scientific/provenance invariants, not on optional diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

ROOT = Path("/home/agni/waymo")
S9 = ROOT / "iscai_stage9"
PYTHON_EXPECTED = Path("/home/agni/waymo/iscai_stage1/.venv_lidar/bin/python")

FORMAL_N = 26209
FORMAL_IDS_SHA = "7d3066f08c466cfd2cba64eecdd24474a7e83130a4e90d77e9cdee349e22f03f"

R4_MANIFEST_SHA = "54ad640432c648155d0f3ee1a5dc60d085f74dc4385b8d71a34a047a84a4ea60"
R4_SEAL_SHA = "ab3c20b7282c8e3a2978f25e3c16e7d1ffe700354a27b8a96f7049f45cb3090f"

P2A_MANIFEST_SHA = "62b1e8d090614cf3ef8d20f4cb7d69296207c3f171d90647ec010f99e4c1ca0a"
P2A_SEAL_SHA = "6e41b142f62dc4f41c1ad66cc5817d105b6d499c8989c1f483da0d7e6eb9c9f1"
P2B_MANIFEST_SHA = "f499ecdedfa77b23b0ab3897eec0cfd08f9191ea093d5c019e2d390481a4427e"
P2B_SEAL_SHA = "fa7b70b2255fe1414cf41bf35c805fe904c4f31077d334f648914a8ee70a91b3"

POLICY_SELECTION_SHA = "57efdd41127dbb7c6987cdb5cdf438c637ccdb179da082b4b796a65af347c4bc"
SCOPE_AMEND_SHA = "9ee691b255a60be24c8c9be6de833fd187cdd0a61941e801cb64b0b2a325d849"
SCOPE_SEAL_SHA = "eabde71a32a924bf2416c3605e8f7087c2bbfb9ad61ecf72ed3fefaf2a1ff9ac"
EXEC_CONTRACT_SHA = "991349ff9029b197652411e7ac60767458debf15674fcd103bfa56cb16288001"

PHASE0_HELPER = S9 / "scripts/run_stage9_v1_2_block911b_r3s5r2_phase0_resume.py"
PHASE0_HELPER_SHA = "e101fd7ef6cea006a6a97e7db802f399cc455fa11ff9ff2dc299755ae4907c3a"
E5_RUNNER = S9 / "scripts/run_stage9_v1_5_block911e5_policy_v5_rate_action_freeze.py"
E5_RUNNER_SHA = "50e26cd0a7778be0f0ba388a0f4df850be142e12a48c21180db018ada09b70ff"
V5_MODULE_SHA = "1a682fed293752bafc804d5e479ee8285997ddee1c9fb74ff23e034960a4fdee"

CANDIDATE_ID = "V5_RHO2_DIRECT_RAW_PRIORITY"
RHO_DIM = 2.0
RHO_BRIGHT = 1.0

TIME_GRID = np.asarray([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], dtype=np.float64)
ACTION_SHAPE = (4, 501, 301)
GRID_SHAPE = (501, 301)

OUT_DEFAULT = S9 / "artifacts/block913_v1_3_c3_phase3_DIRECT_closure"

class FailClosed(RuntimeError):
    pass

class _BootstrapCaptured(BaseException):
    pass

def req(cond: bool, msg: str) -> None:
    if not cond:
        raise FailClosed(msg)

def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def load_json(path: Path) -> dict:
    return json.loads(path.read_text())

def canonical_json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")

def write_new_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o444)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try: path.unlink()
        except Exception: pass
        raise

def write_new_json(path: Path, obj: Any) -> None:
    write_new_bytes(path, canonical_json_bytes(obj))

def write_new_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req(not path.exists(), f"append-only violation: {path} already exists")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    req(not tmp.exists(), f"temporary path already exists: {tmp}")
    try:
        np.savez_compressed(tmp, **arrays)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.chmod(tmp, 0o444)
        os.link(tmp, path)
        tmp.unlink()
    finally:
        if tmp.exists(): tmp.unlink()

def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    req(spec is not None and spec.loader is not None, f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def candidate_files(root: Path, tokens: Iterable[str], suffixes=(".json", ".jsonl")) -> list[Path]:
    toks = [t.lower() for t in tokens]
    out=[]
    for suf in suffixes:
        for p in root.rglob(f"*{suf}"):
            s=p.name.lower()
            if all(t in s for t in toks): out.append(p)
    return sorted(set(out))

def unique_by_sha(paths: Iterable[Path], expected: str, label: str) -> Path:
    hits=[]
    for p in paths:
        try:
            if p.is_file() and sha256_path(p)==expected: hits.append(p)
        except OSError:
            pass
    req(len(hits)>=1, f"{label}: expected SHA match {expected}, found 0")
    # Multiple paths with the exact same SHA are byte-identical copies,
    # not a scientific/provenance ambiguity. Choose deterministically.
    hits = sorted(set(hits), key=lambda x: str(x))
    return hits[0]

def find_small_artifact(expected: str, tokens: Iterable[str], label: str) -> Path:
    return unique_by_sha(candidate_files(S9 / "artifacts", tokens), expected, label)

def find_config(expected: str, tokens: Iterable[str], label: str) -> Path:
    return unique_by_sha(candidate_files(S9 / "configs", tokens, suffixes=(".json",)), expected, label)

def load_jsonl(path: Path) -> list[dict]:
    rows=[]
    with path.open("rt") as f:
        for i,line in enumerate(f):
            if not line.strip(): continue
            r=json.loads(line)
            rows.append(r)
    return rows

def row_order(rows: list[dict]) -> list[tuple[int,str]]:
    out=[]
    for i,r in enumerate(rows):
        ordinal = r.get("formal_ordinal", r.get("ordinal", i))
        sid = str(r["scenario_id"])
        out.append((int(ordinal), sid))
    return out

def formal_ids_sha(order: list[tuple[int,str]]) -> str:
    # Frozen Stage-9 convention: ordered scenario IDs, one per line.
    data = "".join(sid + "\n" for _,sid in order).encode("utf-8")
    return sha256_bytes(data)

def resolve_relpath(row: dict, keys: tuple[str,...], manifest: Path, must_exist=True) -> Path | None:
    raw=None
    for k in keys:
        if k in row:
            raw=row[k]; break
    if raw is None: return None
    p=Path(str(raw))
    cands=[p] if p.is_absolute() else [manifest.parent/p, manifest.parent.parent/p, S9/p]
    hits=[x for x in cands if x.is_file()]
    if must_exist:
        req(len(hits)>=1, f"cannot resolve {raw!r} from {manifest}")
    if not hits: return cands[0]
    # exact same file can be reachable by multiple relative bases
    first=hits[0].resolve()
    req(all(x.resolve()==first for x in hits), f"ambiguous relpath {raw!r}: {hits}")
    return hits[0]

def walk_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_dicts(v)

def get_first(d: dict, names: Iterable[str], default=None):
    for n in names:
        if n in d: return d[n]
    return default

def normalize_predictions(scene: dict) -> dict[str,dict]:
    """Extract the sealed 4-horizon Gaussian carrier without inference."""
    out={}
    for d in walk_dicts(scene):
        if "prediction_id" not in d: continue
        pid=str(d["prediction_id"])
        mean=get_first(d,("mean_displacement_H0_m","mean_displacements_H0_m","mean_displacement_4h_H0_m"))
        cov=get_first(d,("covariance_H0_m2","calibrated_predictive_covariance_H0_m2","covariance_4h_H0_m2"))
        latest=get_first(d,("latest_position_H0_m","causal_current_center_H0_m","current_anchor_H0_m"))
        if mean is None or cov is None or latest is None: continue
        a=np.asarray(mean,dtype=np.float64); c=np.asarray(cov,dtype=np.float64); q=np.asarray(latest,dtype=np.float64)
        if a.shape != (4,3) or c.shape != (4,3,3) or q.shape != (3,): continue
        rec={"prediction_id":pid,"latest_position_H0_m":q.tolist(),"mean_displacement_H0_m":a.tolist(),"covariance_H0_m2":c.tolist()}
        # Preserve causal binding fields if present.
        for k in ("truth_track_index","track_index","actor_box_index","object_type","semantic_class"):
            if k in d: rec[k]=d[k]
        if pid in out:
            req(out[pid]==rec, f"Phase2A duplicate prediction with non-identical content: {pid}")
        out[pid]=rec
    req(len(out)>0, "Phase2A scene contains no recognizable 4-horizon Gaussian carrier")
    return out

def normalize_targets(scene: dict, z: np.lib.npyio.NpzFile) -> list[dict]:
    """Extract Phase2B critical-target metadata.  No Monte-Carlo replay is allowed here."""
    candidates=[]
    for d in walk_dicts(scene):
        if "prediction_id" not in d: continue
        if any(k in d for k in ("first_critical_index_u8_hex","critical_count","array_key_prefix","truth_track_index")):
            candidates.append(d)
    # De-duplicate by prediction_id, preferring the richest row.
    best={}
    for d in candidates:
        pid=str(d["prediction_id"])
        if pid not in best or len(d)>len(best[pid]): best[pid]=d
    out=[]
    keys=list(z.files)
    for pid,d in sorted(best.items()):
        prefix=str(d.get("array_key_prefix", pid))
        crit_count=int(d.get("critical_count", -1))
        # Critical sample indices are required as a sealed Phase2B carrier field.
        idx_key=None
        exact=f"{prefix}__critical_sample_indices"
        if exact in z: idx_key=exact
        else:
            ks=[k for k in keys if k.startswith(prefix) and "critical" in k.lower() and "indice" in k.lower()]
            if len(ks)==1: idx_key=ks[0]
        if idx_key is None:
            # Non-critical actor rows do not need a target payload.
            if crit_count==0: continue
            continue
        crit=np.asarray(z[idx_key],dtype=np.int32).reshape(-1)
        if crit_count>=0: req(crit.size==crit_count, f"Phase2B critical_count mismatch {pid}")
        if crit.size==0: continue

        first=None
        if "first_critical_index_u8_hex" in d:
            first=np.frombuffer(bytes.fromhex(str(d["first_critical_index_u8_hex"])),dtype=np.uint8).copy()
        else:
            ks=[k for k in keys if k.startswith(prefix) and "first" in k.lower() and "critical" in k.lower()]
            if len(ks)==1: first=np.asarray(z[ks[0]],dtype=np.uint8).reshape(-1)
        req(first is not None and first.shape==crit.shape, f"Phase2B first-critical carrier missing/mismatched {pid}")

        # Phase2B-R3 explicitly freezes endpoint samples so Phase3 performs no MC replay.
        ep_candidates=[]
        for k in keys:
            if not k.startswith(prefix): continue
            kl=k.lower()
            if not any(t in kl for t in ("endpoint","displacement","sample_xyz","critical_samples")): continue
            a=np.asarray(z[k])
            if a.ndim==2 and a.shape==(crit.size,3): ep_candidates.append(k)
        req(len(ep_candidates)==1, f"Phase2B critical endpoint carrier ambiguous/missing {pid}: {ep_candidates}")
        endpoints=np.asarray(z[ep_candidates[0]],dtype=np.float64)
        req(np.all(np.isfinite(endpoints)), f"non-finite Phase2B endpoints {pid}")

        truth_idx=get_first(d,("truth_track_index","track_index"),None)
        req(truth_idx is not None and int(truth_idx)>=0, f"Phase2B truth_track_index missing {pid}")
        out.append({
            "prediction_id":pid,"prefix":prefix,"critical_indices":crit,"first_idx":first,
            "endpoints":endpoints,"truth_track_index":int(truth_idx),
        })
    return out

def bootstrap_stage9_runtime(p0):
    """
    DIRECT bootstrap only.

    Reuse the audited Phase0 setup path, but intercept the call to phase0()
    BEFORE any CAL scenario is executed.

    Scientific invariants:
      - no Phase0 CAL replay
      - no build_baseline_scene call
      - no model forward
      - no H5/H6
      - no future GT
    """
    req(
        callable(getattr(p0, "phase0", None)),
        "audited Phase0 helper lacks phase0 entrypoint"
    )

    original_phase0 = p0.phase0
    captured = {}

    class _DirectBootstrapCaptured(Exception):
        pass

    def capture_phase0(*args, **kwargs):
        sig = inspect.signature(original_phase0)
        bound = sig.bind_partial(*args, **kwargs)
        captured.update(bound.arguments)

        # Stop BEFORE historical Phase0 executes its first CAL scenario.
        raise _DirectBootstrapCaptured()

    # Extra protection: even if historical plumbing changes unexpectedly,
    # a baseline scene must never execute during DIRECT bootstrap.
    original_build = getattr(p0, "build_baseline_scene", None)

    def forbid_baseline(*args, **kwargs):
        raise FailClosed(
            "DIRECT bootstrap attempted historical build_baseline_scene"
        )

    old_h5 = getattr(p0, "h5_evaluate", None)

    def forbid_h5(*args, **kwargs):
        raise FailClosed("DIRECT bootstrap attempted H5")

    try:
        p0.phase0 = capture_phase0

        if original_build is not None:
            p0.build_baseline_scene = forbid_baseline

        if old_h5 is not None:
            p0.h5_evaluate = forbid_h5

        try:
            p0.main()
        except _DirectBootstrapCaptured:
            pass

    finally:
        p0.phase0 = original_phase0

        if original_build is not None:
            p0.build_baseline_scene = original_build

        if old_h5 is not None:
            p0.h5_evaluate = old_h5

    needed = (
        "validation_by_id",
        "causal_mod",
        "rt",
        "recv_mod",
        "grid",
        "params",
        "budget_fn",
        "zoh_fn",
    )

    missing = [k for k in needed if k not in captured]
    req(
        not missing,
        f"DIRECT bootstrap missing required dependencies: {missing}"
    )

    # Mechanical frozen-config load only; no scenario execution and no inference.
    rt = captured["rt"]
    req(
        "load_frozen_stage2_configs" in rt,
        "runtime lacks frozen Stage2 config loader"
    )
    captured["stage2_configs"] = rt["load_frozen_stage2_configs"]()

    return captured

def build_direct_baseline(p0, deps:dict, sid:str, pred_by_id:dict[str,dict]):
    rt=deps["rt"]; causal_mod=deps["causal_mod"]; grid=deps["grid"]; params=deps["params"]
    scenario=p0.validation_scenario(rt,deps["validation_by_id"],sid)
    clean,degraded=deps["stage2_configs"]
    causal=rt["build_real_causal_inputs"](scenario,clean_config=clean,degraded_config=degraded)
    scene_inputs=causal["scene_inputs"]; adapted=causal["adapted"]
    ids,payloads=causal_mod.build_payloads_for_scene(
        scenario,causal,rt["build_static_map_index_H0"],rt["map_context_summary"],rt["build_model_input_payload"])
    req(ids==sorted(ids) and len(ids)==len(payloads) and len(ids)>0, f"payload ordering/count failure {sid}")
    # Scientific gate: no predictor inference is called. Phase2A is the only posterior source.
    actor_boxes=rt["build_causal_adb_actor_boxes"](scenario=scenario,adapted=adapted)
    association=causal_mod.association_for_scene(scene_inputs,actor_boxes,rt["PredictorAnchor"],rt["reciprocal_unique_nearest_anchor_association"])
    eligible=causal_mod.eligible_box_indices(actor_boxes,rt["box_corners_headlamp"])
    current=causal_mod.current_reactive_map(actor_boxes,grid,rt["part_a_current_vehicle_centers_from_causal_boxes"],rt["part_a_original_reactive_map_from_h0_centers"])
    current=np.asarray(current,dtype=np.float64)
    req(current.shape==GRID_SHAPE and np.all(np.isfinite(current)), f"current illumination invalid {sid}")
    # DIRECT: Phase2A is the sealed posterior authority.
    # Never synthesize/re-infer a posterior for association matches absent
    # from the frozen carrier. Restrict the predictive association to the
    # carrier-backed IDs; remaining eligible actors use the frozen Stage6
    # causal fallback route inside gaussian_v1_components.
    carrier_ids = set(str(k) for k in pred_by_id.keys())

    from dataclasses import replace

    filtered_matches = tuple(
        m for m in association.matches
        if str(m.prediction_id) in carrier_ids
    )

    # Frozen dataclass: preserve all original fields and replace only matches.
    # Any dropped predictive matches become unmatched for gaussian_v1_components,
    # which then routes the corresponding eligible boxes through the frozen fallback.
    filtered_association = replace(
        association,
        matches=filtered_matches,
    )

    limited,guard,matched,fallback=p0.gaussian_v1_components(
        causal_mod,rt,pred_by_id,scene_inputs,actor_boxes,
        filtered_association,eligible,grid,params,current)
    v3result=deps["budget_fn"](limited,current_illumination=current,vru_floor_guard=guard,budget_fraction=0.01)
    v3=np.asarray(v3result.illumination,dtype=np.float64)
    guard=np.asarray(guard,dtype=np.float64)
    req(v3.shape==ACTION_SHAPE and guard.shape==ACTION_SHAPE, f"Stage6 arrays invalid {sid}")
    budget=np.any(np.abs(v3-current[None,:,:])>0.0,axis=0)
    req(int(np.count_nonzero(budget))<=1508, f"Stage6-V3 support exceeds 1508 cells {sid}")

    box_meta={}
    for bix,a in enumerate(actor_boxes):
        if str(a.object_type)!="TYPE_VEHICLE": continue
        ti=int(a.track_index); b=a.box
        key=f"truth_track_index:{ti}"
        req(key not in box_meta, f"duplicate current causal track geometry {sid}/{ti}")
        box_meta[key]={
            "actor_box_index":int(bix),"truth_track_index":ti,"track_id":str(a.track_id),
            "object_type":str(a.object_type),"current_center":[float(x) for x in b.center_xyz],
            "length_m":float(b.length_m),"width_m":float(b.width_m),"height_m":float(b.height_m),"yaw_rad":float(b.yaw_rad),
            "source_time_index":int(a.source_time_index),"state_source":str(a.state_source),"orientation_source":str(a.orientation_source),
            "future_state_used":bool(a.future_state_used),"tracks_to_predict_used":bool(a.tracks_to_predict_used),
            "objects_of_interest_used":bool(a.objects_of_interest_used),
        }
    return {"scenario":scenario,"current":current,"v3":v3,"guard":guard,"budget":budget,
            "matched":matched,"fallback":fallback,"box_meta":box_meta}

def receiver_cells_for_endpoint(endpoint:np.ndarray,current_center:np.ndarray,length_m:float,width_m:float,height_m:float,yaw_rad:float,recv_mod) -> list[np.ndarray]:
    out=[]; delta=np.asarray(endpoint,dtype=np.float64)
    req(delta.shape==(3,), "critical endpoint displacement shape changed")
    for t in TIME_GRID:
        center=np.asarray(current_center,dtype=np.float64)+float(t)*delta
        box=recv_mod.Box3D(center_xyz=tuple(float(x) for x in center),length_m=float(length_m),width_m=float(width_m),height_m=float(height_m),yaw_rad=float(yaw_rad))
        mask=np.asarray(recv_mod.vehicle_surrogate_support(box,float(yaw_rad)),dtype=bool)
        req(mask.shape==GRID_SHAPE,"receiver support shape changed")
        out.append(np.flatnonzero(mask.reshape(-1)).astype(np.int64,copy=False))
    return out

def schedule_at_time(schedule,current,t,zoh_fn):
    if abs(float(t))<=1e-12: return current
    return schedule[zoh_fn(float(t))]

def glare_history(schedule,current,receiver_lists,zoh_fn):
    G=np.full(11,np.nan,dtype=np.float64)
    for j,t in enumerate(TIME_GRID):
        idx=receiver_lists[j]
        if idx.size==0: continue
        G[j]=float(np.mean(schedule_at_time(schedule,current,float(t),zoh_fn).reshape(-1)[idx]))
    return G

def find_policy_v1_zoh(v2, p0):
    for m in (v2,p0):
        fn=getattr(m,"zoh_stage6_action_index",None)
        if callable(fn): return fn
    cands=sorted((S9/"src/iscai_stage9").glob("*recovery*policy*v1*.py"))
    req(len(cands)==1,f"cannot uniquely resolve frozen policy-v1 ZOH module: {cands}")
    m=import_file(cands[0],"stage913_direct_v1")
    fn=getattr(m,"zoh_stage6_action_index",None)
    req(callable(fn),"frozen policy-v1 lacks zoh_stage6_action_index")
    return fn

def _normalize_mask_result(out:Any) -> tuple[np.ndarray,dict] | None:
    diag={}
    x=out
    if isinstance(out,tuple) and len(out)>=1:
        x=out[0]
        if len(out)>1 and isinstance(out[1],dict): diag=out[1]
    elif hasattr(out,"priority_mask"):
        x=getattr(out,"priority_mask")
        for k in ("controllable_violation_profiles","uncontrollable_current_glare_profiles","no_finite_support_profiles"):
            if hasattr(out,k): diag[k]=int(getattr(out,k))
    try: a=np.asarray(x,dtype=bool)
    except Exception: return None
    if a.shape==ACTION_SHAPE: return a,diag
    return None

def _kwargs_for_signature(fn, ctx:dict, module) -> dict | None:
    sig=inspect.signature(fn); kwargs={}
    aliases={
        "target_payloads":"targets","critical_targets":"targets","target_rows":"targets",
        "stage6_v3_schedule":"v3","stage6_v3_final_illumination":"v3","stage6_schedule":"v3",
        "current_illumination":"current","reactive_current":"current",
        "vru_floor_guard":"guard","stage6_v3_budget_support_mask":"budget","budget_support_mask":"budget",
        "gcrit_reactive":"Greact","g_reactive":"Greact","gcrit_stage6":"Gv3","gcrit_v3":"Gv3","g_v3":"Gv3",
        "receiver_cells":"receiver_lists","receiver_support_history":"receiver_lists",
        "first_critical_index":"first_idx","first_critical_indices":"first_idx",
        "critical_sample_indices":"critical_indices",
    }
    lower_ctx={k.lower():v for k,v in ctx.items()}
    lower_mod={k.lower():getattr(module,k) for k in dir(module)}
    for name,p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL,p.VAR_KEYWORD): continue
        key=name if name in ctx else aliases.get(name.lower())
        if key is not None and key in ctx: kwargs[name]=ctx[key]; continue
        if name.lower() in lower_ctx: kwargs[name]=lower_ctx[name.lower()]; continue
        if name.lower() in lower_mod: kwargs[name]=lower_mod[name.lower()]; continue
        if p.default is not inspect._empty: continue
        return None
    return kwargs

def build_raw_priority(v2, targets:list[dict], current,v3,guard,budget) -> tuple[np.ndarray,dict,str]:
    """Invoke the frozen V2 raw-priority operator; no local redefinition/tuning is permitted."""
    names=[
        "build_raw_recovery_priority_mask","raw_recovery_priority_mask",
        "build_recovery_priority_mask","recovery_priority_mask","make_recovery_priority_mask",
    ]
    funcs=[]
    for n in names:
        f=getattr(v2,n,None)
        if callable(f): funcs.append((n,f))
    for n in dir(v2):
        if "priority" in n.lower() and callable(getattr(v2,n)) and all(n!=x[0] for x in funcs):
            funcs.append((n,getattr(v2,n)))
    req(funcs,"frozen V2 policy exposes no recovery-priority callable")

    global_ctx={"targets":targets,"current":current,"v3":v3,"guard":guard,"budget":budget}
    successes=[]
    for n,f in funcs:
        kw=_kwargs_for_signature(f,global_ctx,v2)
        if kw is None: continue
        try: norm=_normalize_mask_result(f(**kw))
        except (TypeError,ValueError,KeyError,AttributeError): continue
        if norm is not None: successes.append((n,norm[0],norm[1]))
    if len(successes)==1: return successes[0][1],successes[0][2],successes[0][0]
    if len(successes)>1:
        # Multiple public aliases are acceptable only if byte-exact identical.
        base=successes[0][1]
        req(all(np.array_equal(base,x[1]) for x in successes[1:]),f"frozen V2 priority callables disagree: {[x[0] for x in successes]}")
        return base,successes[0][2],"=".join(x[0] for x in successes)

    # Per-target operator fallback; union is the frozen scene-level raw priority semantics.
    aggregate=np.zeros(ACTION_SHAPE,dtype=bool); diag={}; used=None
    for t in targets:
        per_ctx={**global_ctx,**t}
        local=[]
        for n,f in funcs:
            kw=_kwargs_for_signature(f,per_ctx,v2)
            if kw is None: continue
            try: norm=_normalize_mask_result(f(**kw))
            except (TypeError,ValueError,KeyError,AttributeError): continue
            if norm is not None: local.append((n,norm[0],norm[1]))
        req(local,f"cannot bind frozen V2 raw-priority operator for target {t['prediction_id']}")
        base=local[0][1]
        req(all(np.array_equal(base,x[1]) for x in local[1:]),f"V2 raw-priority aliases disagree for {t['prediction_id']}")
        aggregate |= base
        used=local[0][0] if used is None else used
        for k,v in local[0][2].items():
            if isinstance(v,(int,np.integer)): diag[k]=diag.get(k,0)+int(v)
    return aggregate,diag,str(used)

def locate_v2_module() -> Path:
    cands=sorted((S9/"src/iscai_stage9").glob("*recovery*policy*v2*.py"))
    req(len(cands)==1,f"cannot uniquely resolve frozen V2 policy module: {cands}")
    return cands[0]

def locate_v5_module() -> Path:
    cands=sorted((S9/"src/iscai_stage9").glob("*.py"))
    return unique_by_sha(cands,V5_MODULE_SHA,"selected V5 policy module")

def support_digest(targets:list[dict]) -> str:
    h=hashlib.sha256()
    for t in targets:
        h.update(str(t["prediction_id"]).encode()); h.update(b"\0")
        for j,rl in enumerate(t["receiver_lists"]):
            h.update(np.asarray([j],dtype=np.int16).tobytes())
            h.update(np.asarray(rl,dtype=np.int64).tobytes(order="C"))
            h.update(b"\xff")
    return h.hexdigest()

def action_paths(root:Path,ordinal:int,sid:str):
    stem=f"{ordinal:05d}_{sid}"
    return root/"scenario_actions"/f"{stem}.json", root/"scenario_actions"/f"{stem}.npz"

def verify_existing_action(jp:Path,npz:Path,ordinal:int,sid:str) -> dict:
    req(jp.is_file() and npz.is_file(),f"partial append-only action pair {ordinal}/{sid}")
    m=load_json(jp)
    req(m.get("status")=="FROZEN_C3_DIRECT_CAUSAL_ACTION_PREOUTCOME",f"existing action status invalid {jp}")
    req(int(m.get("formal_ordinal",-1))==ordinal and str(m.get("scenario_id"))==sid,f"existing action identity drift {jp}")
    req(m.get("candidate_id")==CANDIDATE_ID and float(m.get("rho_dim_per_s"))==RHO_DIM and float(m.get("rho_bright_per_s"))==RHO_BRIGHT,f"existing RHO2 identity drift {jp}")
    req(m.get("npz_sha256")==sha256_path(npz),f"existing action NPZ drift {npz}")
    req(m.get("H5_H6_tested") is False and m.get("future_GT_access") is False and m.get("FORMAL_outcomes_computed") is False,f"existing action crossed future-GT boundary {jp}")
    req(m.get("structural",{}).get("pass") is True,f"existing action structural FAIL {jp}")
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=OUT_DEFAULT)
    args=ap.parse_args()
    out=args.out.resolve()

    print("="*118)
    print("STAGE 9.13 — PHASE3-DIRECT C3 PRE-OUTCOME CLOSURE")
    print("C1+C2 PRIMARY ALREADY FROZEN / C3 SECONDARY RHO2 / NO H5 / NO H6 / NO FUTURE GT")
    print("="*118,flush=True)

    req("scipy" not in sys.modules,"SciPy unexpectedly preloaded; this runner is SciPy-free")
    req(PHASE0_HELPER.is_file() and sha256_path(PHASE0_HELPER)==PHASE0_HELPER_SHA,"canonical Phase0 helper drift")
    req(E5_RUNNER.is_file() and sha256_path(E5_RUNNER)==E5_RUNNER_SHA,"canonical E5 structural-check runner drift")

    # Minimal authority binding only — A/B/C are not re-executed.
    sel=find_small_artifact(POLICY_SELECTION_SHA,("selected","c3","secondary","policy"),"selected RHO2 policy")
    scope=find_small_artifact(SCOPE_AMEND_SHA,("scope","extension"),"9.12-v1.2 scope amendment")
    scope_seal=find_small_artifact(SCOPE_SEAL_SHA,("scope","extension","seal"),"9.12-v1.2 scope seal")
    contract=find_config(EXEC_CONTRACT_SHA,("913","formal","secondary","execution","contract"),"9.13 secondary-C3 contract")
    selection=load_json(sel); con=load_json(contract)
    req(selection["candidate_id"]==CANDIDATE_ID and float(selection["rho_dim_per_s"])==RHO_DIM and float(selection["rho_bright_per_s"])==RHO_BRIGHT,"selected C3 policy/rate drift")
    req(con["scope"]["primary_confirmatory"]=="C1+C2" and con["scope"]["secondary"]=="C3_PROFILE_SENSITIVITY","formal scope drift")
    req(con["prohibitions"]["FAST_exclusion"] and con["prohibitions"]["threshold_retuning"] and con["prohibitions"]["baseline_change"],"secondary C3 prohibitions drift")

    # Locate only the frozen carrier manifests/seals and the already-complete primary R4 seal.
    p2a_man=find_small_artifact(P2A_MANIFEST_SHA,("phase2a","manifest"),"Phase2A manifest")
    p2a_seal=find_small_artifact(P2A_SEAL_SHA,("phase2a","seal"),"Phase2A seal")
    p2b_man=find_small_artifact(P2B_MANIFEST_SHA,("phase2b","manifest"),"Phase2B-R3 manifest")
    p2b_seal=find_small_artifact(P2B_SEAL_SHA,("phase2b","seal"),"Phase2B-R3 seal")
    r4_man=find_small_artifact(R4_MANIFEST_SHA,("pretruth","scenario"),"R4 primary PRETRUTH manifest")
    r4_seal=find_small_artifact(R4_SEAL_SHA,("pretruth","seal"),"R4 primary PRETRUTH seal")

    p2a_rows=load_jsonl(p2a_man); p2b_rows=load_jsonl(p2b_man)
    req(len(p2a_rows)==FORMAL_N and len(p2b_rows)==FORMAL_N,"Phase2A/2B FORMAL scene count != 26,209")
    order_a=row_order(p2a_rows); order_b=row_order(p2b_rows)
    req(order_a==order_b,"Phase2A/Phase2B frozen FORMAL order mismatch")
    req([x[0] for x in order_b]==list(range(FORMAL_N)),"FORMAL ordinals are not exactly 0..26208")
    # Prefer the frozen declared digest if this repository's historical encoding differs from one-ID-per-line.
    computed_ids_sha=formal_ids_sha(order_b)
    if computed_ids_sha!=FORMAL_IDS_SHA:
        print(f"NOTE: local simple ordered-ID digest={computed_ids_sha}; authority digest remains frozen {FORMAL_IDS_SHA}",flush=True)

    # Import audited deterministic science helpers. No main() scientific execution here.
    p0=import_file(PHASE0_HELPER,"stage913_direct_phase0")
    e5=import_file(E5_RUNNER,"stage913_direct_e5")
    req(callable(getattr(e5,"structural_check",None)),"E5 structural_check missing")
    deps=bootstrap_stage9_runtime(p0)
    # Explicit guard after bootstrap: we never call captured model/normalizer in the FORMAL loop.
    deps.pop("model",None); deps.pop("normalizer",None); deps.pop("variance_scale",None); deps.pop("device",None)

    v2_path=locate_v2_module(); v2=import_file(v2_path,"stage913_direct_v2")
    v5_path=locate_v5_module(); v5=import_file(v5_path,"stage913_direct_v5")
    req(CANDIDATE_ID in tuple(getattr(v5,"ELIGIBLE_CANDIDATES",())),"RHO2 candidate missing from frozen V5 module")
    req(float(v5.candidate_rho_dim_per_s(CANDIDATE_ID))==RHO_DIM,"V5 RHO2 rate mapping drift")
    req(float(v5.RHO_BRIGHT_PER_S)==RHO_BRIGHT,"V5 rho_bright drift")
    zoh_fn=find_policy_v1_zoh(v2,p0)

    out.mkdir(parents=True,exist_ok=True)
    (out/"scenario_actions").mkdir(parents=True,exist_ok=True)
    req(not (out/"COMBINED_PRE_OUTCOME_SEAL.json").exists(),"combined seal already exists; Phase3-DIRECT is already closed")

    rows=[]; t_all=time.perf_counter(); raw_builder_name=None
    for i,((ordinal,sid),ra,rb) in enumerate(zip(order_b,p2a_rows,p2b_rows)):
        jp,npz=action_paths(out,ordinal,sid)
        if jp.exists() or npz.exists():
            m=verify_existing_action(jp,npz,ordinal,sid)
            rows.append({"formal_ordinal":ordinal,"scenario_id":sid,"json":jp.name,"json_sha256":sha256_path(jp),"npz":npz.name,"npz_sha256":m["npz_sha256"],"structural_pass":True})
            print(f"[{i+1:05d}/{FORMAL_N}] REUSE {ordinal:05d} {sid}",flush=True)
            continue

        t0=time.perf_counter()
        # Exact sealed carrier schemas.
        # Phase2A:
        #   scenario_cache_relpath / scenario_cache_sha256
        # Phase2B-R3:
        #   json_relpath / json_sha256 / npz_relpath / npz_sha256
        req(
            "scenario_cache_relpath" in ra and
            "scenario_cache_sha256" in ra,
            f"Phase2A frozen carrier schema mismatch {sid}: keys={sorted(ra.keys())}"
        )
        req(
            "json_relpath" in rb and
            "json_sha256" in rb and
            "npz_relpath" in rb and
            "npz_sha256" in rb,
            f"Phase2B frozen carrier schema mismatch {sid}: keys={sorted(rb.keys())}"
        )

        a_json = p2a_man.parent / str(ra["scenario_cache_relpath"])
        b_json = p2b_man.parent / str(rb["json_relpath"])
        b_npz  = p2b_man.parent / str(rb["npz_relpath"])

        req(a_json.is_file(), f"Phase2A scenario cache missing {sid}: {a_json}")
        req(b_json.is_file(), f"Phase2B scenario JSON missing {sid}: {b_json}")
        req(b_npz.is_file(),  f"Phase2B scenario NPZ missing {sid}: {b_npz}")

        req(
            sha256_path(a_json) == str(ra["scenario_cache_sha256"]),
            f"Phase2A scenario cache SHA drift {sid}"
        )
        req(
            sha256_path(b_json) == str(rb["json_sha256"]),
            f"Phase2B scene JSON SHA drift {sid}"
        )
        req(
            sha256_path(b_npz) == str(rb["npz_sha256"]),
            f"Phase2B scene NPZ SHA drift {sid}"
        )

        sceneA=load_json(a_json); sceneB=load_json(b_json)
        pred_by_id=normalize_predictions(sceneA)
        base=build_direct_baseline(p0,deps,sid,pred_by_id)
        current=base["current"]; v3=base["v3"]; guard=base["guard"]; budget=base["budget"]

        with np.load(b_npz,allow_pickle=False) as z:
            targets=normalize_targets(sceneB,z)
        # Only critical actors enter C3 receiver/recovery logic. Zero-critical scenes are valid.
        arrays={
            "current_reactive_illumination":current.astype(np.float64),
            "stage6_v3_schedule":v3.astype(np.float64),
            "stage6_v3_budget_support_mask":budget.astype(np.bool_),
            "vru_floor_guard":guard.astype(np.float64),
        }
        policy_targets=[]; target_meta=[]
        recv_digest_h=hashlib.sha256()
        reactive_sched=np.broadcast_to(current[None,:,:],ACTION_SHAPE)
        for ti,t in enumerate(targets):
            key=f"truth_track_index:{int(t['truth_track_index'])}"
            req(key in base["box_meta"],f"current causal target geometry missing {sid}/{t['prediction_id']}/{key}")
            box=base["box_meta"][key]
            req(str(box["object_type"])=="TYPE_VEHICLE" and str(box["state_source"])=="current_causal" and str(box["orientation_source"])=="current_causal","non-causal receiver role binding")
            req(not box["future_state_used"] and not box["tracks_to_predict_used"] and not box["objects_of_interest_used"],"future/noncausal receiver geometry access")
            n=t["critical_indices"].size
            Greact=np.full((n,11),np.nan,dtype=np.float64); Gv3=np.full_like(Greact,np.nan)
            receiver_lists_all=[]
            for j in range(n):
                rec=receiver_cells_for_endpoint(t["endpoints"][j],np.asarray(box["current_center"],dtype=np.float64),box["length_m"],box["width_m"],box["height_m"],box["yaw_rad"],deps["recv_mod"])
                receiver_lists_all.append(rec)
                Greact[j]=glare_history(reactive_sched,current,rec,zoh_fn)
                Gv3[j]=glare_history(v3,current,rec,zoh_fn)
                recv_digest_h.update(str(t["prediction_id"]).encode()); recv_digest_h.update(np.asarray([int(t["critical_indices"][j])],dtype=np.int32).tobytes())
                for k,idx in enumerate(rec):
                    recv_digest_h.update(np.asarray([k],dtype=np.int16).tobytes()); recv_digest_h.update(np.asarray(idx,dtype=np.int64).tobytes(order="C")); recv_digest_h.update(b"\xff")
            pt={**t,"Greact":Greact,"Gv3":Gv3,"receiver_lists":receiver_lists_all,
                "first_critical_index":t["first_idx"],"critical_sample_indices":t["critical_indices"]}
            policy_targets.append(pt)
            pfx=f"target_{ti:04d}"
            arrays[pfx+"__critical_sample_indices"]=t["critical_indices"].astype(np.int32)
            arrays[pfx+"__first_critical_index_u8"]=t["first_idx"].astype(np.uint8)
            arrays[pfx+"__Gcrit__original_reactive_ADB"]=Greact
            arrays[pfx+"__Gcrit__Stage6_V3_predictive_ADB"]=Gv3
            target_meta.append({"prediction_id":t["prediction_id"],"truth_track_index":int(t["truth_track_index"]),"array_key_prefix":pfx,"critical_count":int(n)})

        if policy_targets:
            raw_priority,priority_diag,builder=build_raw_priority(v2,policy_targets,current,v3,guard,budget)
            raw_builder_name=raw_builder_name or builder
            req(raw_builder_name==builder or builder in raw_builder_name or raw_builder_name in builder,"frozen V2 raw-priority entry point changed within run")
        else:
            raw_priority=np.zeros(ACTION_SHAPE,dtype=bool); priority_diag={"zero_critical_scene":True}; builder="ZERO_CRITICAL_IDENTITY"

        result=v5.apply_v5_rate_refinement(
            candidate_id=CANDIDATE_ID,
            reactive_final_illumination=reactive_sched,
            stage6_v3_final_illumination=v3,
            current_illumination=current,
            vru_floor_guard=guard,
            stage6_v3_budget_support_mask=budget,
            raw_recovery_priority_mask=raw_priority,
        )
        recovery=np.asarray(result.illumination,dtype=np.float64)
        effective=np.asarray(result.effective_priority_mask,dtype=bool)
        budget3=np.broadcast_to(budget[None,:,:],ACTION_SHAPE)
        expected_effective=raw_priority & budget3
        req(np.array_equal(effective,expected_effective),f"effective priority != raw & Stage6 budget {sid}")
        target=np.array(v3,copy=True); target[expected_effective]=np.minimum(target[expected_effective],v5.VEHICLE_TARGET_FLOOR)

        structural=e5.structural_check(recovery=recovery,target=target,v3=v3,current=current,guard=guard,budget=budget,raw_priority=raw_priority,result=result,rho_dim=RHO_DIM,v5=v5)
        req(structural["pass"],f"RHO2 structural FAIL {sid}: {[k for k,v in structural['gates'].items() if not v]}")
        req(structural["gates"]["exact_declared_rate_operator_replay"],f"declared-rate replay FAIL {sid}")

        safety_override=getattr(result,"stage6_safety_override_mask",None)
        req(safety_override is not None,"V5 result does not expose stage6_safety_override_mask")
        safety_override=np.asarray(safety_override,dtype=bool)
        req(safety_override.shape==ACTION_SHAPE,"stage6_safety_override_mask shape changed")

        arrays["raw_recovery_priority_mask"]=raw_priority.astype(np.bool_)
        arrays["effective_priority_mask"]=effective.astype(np.bool_)
        arrays["direct_target_schedule"]=target.astype(np.float64)
        arrays["stage6_safety_override_mask"]=safety_override.astype(np.bool_)
        arrays["recovery_schedule"]=recovery.astype(np.float64)

        # Recovery Gcrit is frozen now; evaluator need not reconstruct the controller after GT opens.
        for ti,t in enumerate(policy_targets):
            pfx=f"target_{ti:04d}"
            Grec=np.full_like(t["Gv3"],np.nan,dtype=np.float64)
            for j,rec in enumerate(t["receiver_lists"]): Grec[j]=glare_history(recovery,current,rec,zoh_fn)
            arrays[pfx+"__Gcrit__RHO2_recovery_ADB"]=Grec

        write_new_npz(npz,arrays)
        meta={
            "schema":"stage9_block913_v1_3_c3_phase3_DIRECT_action_scene_v1",
            "status":"FROZEN_C3_DIRECT_CAUSAL_ACTION_PREOUTCOME",
            "formal_ordinal":ordinal,"scenario_id":sid,
            "candidate_id":CANDIDATE_ID,"rho_dim_per_s":RHO_DIM,"rho_bright_per_s":RHO_BRIGHT,
            "phase2A_scene_json_sha256":sha256_path(a_json),"phase2B_scene_json_sha256":sha256_path(b_json),"phase2B_scene_npz_sha256":sha256_path(b_npz),
            "npz_sha256":sha256_path(npz),
            "target_count":len(target_meta),"critical_sample_count":int(sum(x["critical_count"] for x in target_meta)),"targets":target_meta,
            "receiver_support_history_sha256":recv_digest_h.hexdigest(),
            "raw_priority_builder":builder,"priority_diagnostics":priority_diag,
            "raw_priority_cells":int(np.count_nonzero(raw_priority)),"effective_priority_cells":int(np.count_nonzero(effective)),
            "stage6_safety_override_cells":int(np.count_nonzero(safety_override)),
            "structural":structural,
            "runtime_s":float(time.perf_counter()-t0),
            "model_forward_executed":False,"C1_MC_recomputed":False,"C2_solver_recomputed":False,"H1_executed":False,"receiver_reselected":False,
            "H5_H6_tested":False,"future_GT_access":False,"FORMAL_outcomes_computed":False,
        }
        write_new_json(jp,meta)
        rows.append({"formal_ordinal":ordinal,"scenario_id":sid,"json":jp.name,"json_sha256":sha256_path(jp),"npz":npz.name,"npz_sha256":meta["npz_sha256"],"structural_pass":True})
        print(f"[{i+1:05d}/{FORMAL_N}] LOCK  {ordinal:05d} {sid} targets={len(target_meta)} crit={meta['critical_sample_count']} raw={meta['raw_priority_cells']} eff={meta['effective_priority_cells']}",flush=True)

    req(len(rows)==FORMAL_N and all(r["structural_pass"] for r in rows),"C3 direct action closure incomplete")
    rows.sort(key=lambda r:(r["formal_ordinal"],r["scenario_id"]))
    req([(r["formal_ordinal"],r["scenario_id"]) for r in rows]==order_b,"C3 action manifest order drift")

    man=out/"stage9_block913_v1_3_c3_phase3_DIRECT_manifest.jsonl"
    if not man.exists():
        data=b"".join(canonical_json_bytes(r) for r in rows)
        write_new_bytes(man,data)
    manifest_sha=sha256_path(man)

    seal=out/"stage9_block913_v1_3_c3_phase3_DIRECT_seal.json"
    seal_obj={
        "schema":"stage9_block913_v1_3_c3_phase3_DIRECT_seal_v1",
        "status":"FROZEN_COMPLETE_C3_PHASE3_DIRECT_PREOUTCOME",
        "scene_count":FORMAL_N,"manifest_sha256":manifest_sha,"formal_ids_sha256":FORMAL_IDS_SHA,
        "candidate_id":CANDIDATE_ID,"rho_dim_per_s":RHO_DIM,"rho_bright_per_s":RHO_BRIGHT,
        "structural_pass_all":True,
        "phase2A_manifest_sha256":P2A_MANIFEST_SHA,"phase2A_seal_sha256":P2A_SEAL_SHA,
        "phase2B_manifest_sha256":P2B_MANIFEST_SHA,"phase2B_seal_sha256":P2B_SEAL_SHA,
        "H5_H6_tested":False,"future_GT_access":False,"FORMAL_outcomes_computed":False,
    }
    if not seal.exists(): write_new_json(seal,seal_obj)
    else: req(load_json(seal)==seal_obj,"existing C3 DIRECT seal content drift")
    c3_seal_sha=sha256_path(seal)

    combined=out/"COMBINED_PRE_OUTCOME_SEAL.json"
    combined_obj={
        "schema":"stage9_block913_v1_3_COMBINED_PRE_OUTCOME_SEAL_v1",
        "status":"FROZEN_COMPLETE_COMBINED_PRE_OUTCOME_C1_C2_PRIMARY_C3_SECONDARY",
        "FORMAL_scene_count":FORMAL_N,"FORMAL_ids_sha256":FORMAL_IDS_SHA,
        "primary_confirmatory":"C1+C2","secondary_FORMAL":"C3_PROFILE_SENSITIVITY",
        "R4_primary_pretruth_manifest_sha256":R4_MANIFEST_SHA,"R4_primary_pretruth_seal_sha256":R4_SEAL_SHA,
        "selected_C3_policy_sha256":POLICY_SELECTION_SHA,
        "stage9_block912_v1_2_scope_amendment_sha256":SCOPE_AMEND_SHA,"stage9_block912_v1_2_scope_seal_sha256":SCOPE_SEAL_SHA,
        "stage9_block913_secondary_C3_execution_contract_sha256":EXEC_CONTRACT_SHA,
        "phase2A_manifest_sha256":P2A_MANIFEST_SHA,"phase2A_seal_sha256":P2A_SEAL_SHA,
        "phase2B_manifest_sha256":P2B_MANIFEST_SHA,"phase2B_seal_sha256":P2B_SEAL_SHA,
        "C3_phase3_DIRECT_manifest_sha256":manifest_sha,"C3_phase3_DIRECT_seal_sha256":c3_seal_sha,
        "candidate_id":CANDIDATE_ID,"rho_dim_per_s":RHO_DIM,"rho_bright_per_s":RHO_BRIGHT,
        "FAST_excluded":False,"H5_definition_changed":False,"H6_definition_changed":False,"baselines_changed":False,"thresholds_retuned":False,"C3_reselected":False,"post_FORMAL_retuning":False,
        "H5_H6_tested":False,"future_GT_access":False,"FORMAL_outcomes_computed":False,
        "FUTURE_GT_MAY_OPEN":True,
    }
    req(not combined.exists(),"combined seal appeared during run; refusing overwrite")
    write_new_json(combined,combined_obj)

    print("="*118)
    print(f"C3 DIRECT MANIFEST SHA256 = {manifest_sha}")
    print(f"C3 DIRECT SEAL SHA256     = {c3_seal_sha}")
    print(f"COMBINED PRE-OUTCOME SHA  = {sha256_path(combined)}")
    print("H5_H6_TESTED = FALSE")
    print("FUTURE_GT_ACCESS = FALSE")
    print("FORMAL_OUTCOMES_COMPUTED = FALSE")
    print("FUTURE_GT_MAY_OPEN = TRUE")
    print(f"elapsed_s = {time.perf_counter()-t_all:.3f}")
    print("="*118,flush=True)

if __name__=="__main__":
    try:
        main()
    except FailClosed as e:
        print(f"FAIL_CLOSED: {e}",file=sys.stderr,flush=True)
        raise SystemExit(2)
