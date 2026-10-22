#!/bin/bash
# Wall-clock benchmark (reviewer R2.m2). Run on an otherwise idle GPU.
REPO=/home/tico/Desktop/master_research/DeiDDPG
W=$REPO/exp/robustness_work
PY=/home/tico/anaconda3/envs/gddiff/bin/python
ARGS="--config celeba_hq.yml --deg deblur_gauss --sigma_y 0.05 --inject_noise 1 --gamma 8 --zeta 0.5 --eta_tilde 0.7"
cd $REPO
mkdir -p $W/timing; rm -f $W/timing/*.csv
rm -rf exp/image_samples/timing_*
for PER in 0.1 0.2 0.5; do
  GDDIFF_TIMING_LOG=$W/timing/gddiff_per$PER.csv $PY main.py $ARGS --per $PER --path_y $W/bench20 \
    -i timing_gddiff_$PER/run < /dev/null > $W/logs/timing_gddiff_$PER.log 2>&1 || echo "FAILED gddiff $PER"
done
GDDIFF_SINGLE_BRANCH=1 GDDIFF_TIMING_LOG=$W/timing/single_branch.csv $PY main.py $ARGS --per 0.5 --path_y $W/bench20 \
  -i timing_single/run < /dev/null > $W/logs/timing_single.log 2>&1 || echo "FAILED single"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > $W/timing/gpu.txt
echo TIMING DONE
