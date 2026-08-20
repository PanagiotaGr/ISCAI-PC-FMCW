#!/bin/bash

set -e


export PYTHONPATH="\
/home/agni/waymo/iscai_stage3/src:\
/home/agni/waymo/iscai_stage2/src:\
/home/agni/waymo/iscai_stage1/src:\
/home/agni/waymo/iscai_stage0/src"


PY=/home/agni/waymo/iscai_stage1/.venv_lidar/bin/python


echo "===== Stage3 complete validation ====="


echo ""
echo "[1/5] Unit tests"

"$PY" -m unittest discover \
-s tests \
-p 'test_*.py'


echo ""
echo "[2/5] Real temporal closed loop"

"$PY" scripts/run_stage3_real_temporal_closed_loop_gate.py


echo ""
echo "[3/5] Multi scenario temporal"

"$PY" scripts/run_stage3_multiscenario_temporal_gate.py


echo ""
echo "[4/5] Classical baselines"

"$PY" scripts/run_stage3_classical_baseline_gate.py


echo ""
echo "[5/5] Baseline metrics"

"$PY" scripts/run_stage3_baseline_metrics_gate.py


echo ""
echo "===== STAGE3 COMPLETE ====="
