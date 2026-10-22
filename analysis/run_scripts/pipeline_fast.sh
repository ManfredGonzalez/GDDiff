#!/bin/bash
# Reduced pipeline for the submission deadline (lambda = 0.5, same command and seed as the paper runs).
#   worker 1: ablation from_G   on the 300-image clean subset (paired with robust_base = full Mean-GDDiff)
#   worker 2: ablation no_zmask on the 300-image clean subset
#   worker 3: occlusion (eyes, lower face, patch) on 100 images each (50 RaFD + 50 LFW)
REPO=/home/tico/Desktop/master_research/DeiDDPG
W=$REPO/exp/robustness_work
PY=/home/tico/anaconda3/envs/gddiff/bin/python
ARGS="--config celeba_hq.yml --deg deblur_gauss --sigma_y 0.05 --inject_noise 1 --gamma 8 --zeta 0.5 --eta_tilde 0.7 --per 0.5"
cd $REPO
log() { echo "$(date '+%F %T') $*" >> $W/logs/status_fast.txt; }
run() {  # name inputset ablation
  log "start $1"
  GDDIFF_AVG_ABLATION=$3 $PY main.py $ARGS --path_y $W/robust_inputs/$2 -i $1/run < /dev/null > $W/logs/$1.log 2>&1 \
    && log "done $1" || log "FAILED $1"
}
log "pipeline_fast start"
run abl300_from_G clean from_G &
run abl300_no_zmask clean no_zmask &
( run robust_occl_eyes occl_eyes100 avg; run robust_occl_lower occl_lower100 avg; run robust_occl_patch occl_patch100 avg ) &
wait
log "PIPELINE DONE"
