#!/bin/bash

set -e


export PYTHONPATH="\
/home/agni/waymo/iscai_stage4/src:\
/home/agni/waymo/iscai_stage3/src:\
/home/agni/waymo/iscai_stage2/src:\
/home/agni/waymo/iscai_stage1/src:\
/home/agni/waymo/iscai_stage0/src"


PY=/home/agni/waymo/iscai_stage1/.venv_lidar/bin/python


echo "===== Stage4 complete validation ====="


echo ""
echo "[1/6] Probabilistic smoke"

"$PY" scripts/run_stage4_smoke.py


echo ""
echo "[2/6] Real WOMD probabilistic gate"

"$PY" scripts/run_stage4_real_womd_calibration_gate.py


echo ""
echo "[3/6] Multi scenario probabilistic gate"

"$PY" scripts/run_stage4_multiscenario_real_gate.py


echo ""
echo "[4/6] Calibration metrics"

"$PY" scripts/run_stage4_calibration_metrics_gate.py


echo ""
echo "[5/6] Covariance sweep"

"$PY" scripts/run_stage4_covariance_sweep_gate.py


echo ""
echo "[6/6] Calibration comparison"

"$PY" scripts/run_stage4_calibration_comparison_gate.py


echo ""
echo "===== STAGE4 COMPLETE ====="
