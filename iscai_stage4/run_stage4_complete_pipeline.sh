#!/bin/bash

set -e


echo "=============================="
echo "STAGE4 COMPLETE PIPELINE START"
echo "=============================="


export PY=/home/agni/waymo/iscai_stage5/.venv_stage5/bin/python

export PYTHONPATH="/home/agni/waymo/iscai_stage4/src:/home/agni/waymo/iscai_stage1/src:$PYTHONPATH"



echo ""
echo "=============================="
echo "1) DETERMINISTIC FINAL"
echo "=============================="


$PY scripts/run_stage4_deterministic_final.py



echo ""
echo "=============================="
echo "2) GAUSSIAN FINAL"
echo "=============================="


$PY scripts/run_stage4_gaussian_experiments.py



echo ""
echo "=============================="
echo "3) HYPERPARAMETER SEARCH"
echo "=============================="


$PY scripts/run_stage4_hyperparameter_search.py



echo ""
echo "=============================="
echo "4) FINAL MODEL EVALUATION"
echo "=============================="


$PY scripts/evaluate_stage4_all_models.py



echo ""
echo "=============================="
echo "STAGE4 PIPELINE FINISHED"
echo "=============================="


date

