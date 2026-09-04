#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path

ROOT = Path("/home/agni/waymo")
S8 = ROOT / "iscai_stage8"

SCRIPT = S8 / "scripts/run_stage8_block84r_reporting_correction.py"
TESTS = S8 / "artifacts/block84_beam_policy_interface/synthetic_policy_unit_tests.json"
REPORT84 = S8 / "reports/stage8_block84_beam_policy_interface_report.json"
SEAL84 = S8 / "artifacts/block84_beam_policy_interface/stage8_block84_beam_policy_interface_seal.json"

OUT = S8 / "artifacts/block84r_reporting_correction"
REPORT = S8 / "reports/stage8_block84r_reporting_correction_report.json"
SEAL = OUT / "stage8_block84r_reporting_correction_seal.json"

EXPECTED = {
    str(TESTS): "1a8c8f5d35abfad0b05f74eef97ae9929aa6b74ba6c5a88a941ca6cb0a7ee356",
    str(REPORT84): "5edd76e61dc81cfeb562c80ec1002ce14554d5867b44f859f8f8bc035ff68b0c",
    str(SEAL84): "c5dcbf5b1bf8d7965fd32bf794f095b4495d30d98e35bddef22ce3abffe554e2",
}

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
    data=(json.dumps(obj,sort_keys=True,separators=(",",":"))+"\n").encode()
    fd,tmp=tempfile.mkstemp(prefix=path.name+".tmp.",dir=str(path.parent))
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def main():
    print("="*80)
    print("STAGE 8 — BLOCK 8.4R REPORTING CORRECTION")
    print("NO POLICY RE-EXECUTION / NO SCIENTIFIC ARTIFACT MODIFICATION")
    print("="*80)

    require(Path(__file__).resolve()==SCRIPT.resolve(),"runner path mismatch")
    for raw,expected in EXPECTED.items():
        p=Path(raw)
        require(p.is_file(),f"missing frozen 8.4 artifact: {p}")
        require(sha(p)==expected,f"frozen 8.4 SHA changed: {p}")
    require(not REPORT.exists() and not SEAL.exists(),"8.4R already exists")

    tests=json.loads(TESTS.read_text())
    report84=json.loads(REPORT84.read_text())
    arr=tests.get("tests",[])
    require(len(arr)==8,f"expected 8 frozen tests, got {len(arr)}")
    require(all(x.get("pass") is True for x in arr),"not all frozen tests passed")
    require(report84.get("synthetic_tests_passed")==8,
            f"frozen JSON report does not record 8 tests: {report84.get('synthetic_tests_passed')}")

    correction={
        "stage":8,"block":"8.4R",
        "status":"PASS_REPORTING_CORRECTION_ONLY",
        "scientific_change":False,
        "policy_reexecution":False,
        "formal_power_access":False,
        "frozen_tests_actual_count":8,
        "frozen_tests_passed":8,
        "incorrect_console_strings":["synthetic policy tests = 8/7 PASS","SYNTHETIC POLICY TESTS = 7/7 PASS"],
        "canonical_statement":"SYNTHETIC POLICY TESTS = 8/8 PASS",
        "Block84_tests_sha256":EXPECTED[str(TESTS)],
        "Block84_report_sha256":EXPECTED[str(REPORT84)],
        "Block84_seal_sha256":EXPECTED[str(SEAL84)],
    }
    atomic_json(REPORT,correction)
    atomic_json(SEAL,{
        "stage":8,"block":"8.4R","status":"FROZEN_COMPLETE_BLOCK84R",
        "Block84_seal_sha256":EXPECTED[str(SEAL84)],
        "correction_report_sha256":sha(REPORT),
        "scientific_change":False,
        "Block85_may_start":True,
    })

    print("FROZEN 8.4 TEST JSON COUNT = 8")
    print("FROZEN 8.4 REPORT synthetic_tests_passed = 8")
    print("CANONICAL SYNTHETIC POLICY TEST RESULT = 8/8 PASS")
    print("SCIENTIFIC CHANGE = NO")
    print("POLICY RE-EXECUTION = NO")
    print("FORMAL POWER ACCESS = NO")
    print("BLOCK 8.5 MAY START = YES")
    print("8.4R report SHA256 =",sha(REPORT))
    print("8.4R seal SHA256   =",sha(SEAL))
    print("STATUS = FROZEN_COMPLETE_BLOCK84R")
    return 0

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("BLOCK 8.4R FAIL-CLOSED:",type(e).__name__,e)
        raise
