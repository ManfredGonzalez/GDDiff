#!/bin/bash
REPO=/home/tico/Desktop/master_research/DeiDDPG
W=$REPO/exp/robustness_work
PY=/home/tico/anaconda3/envs/gddiff/bin/python
ARGS="--config celeba_hq.yml --deg deblur_gauss --sigma_y 0.05 --inject_noise 1 --gamma 8 --zeta 0.5 --eta_tilde 0.7 --per 0.5"
cd $REPO
log() { echo "$(date '+%F %T') $*" >> $W/logs/status_fast.txt; }
for o in occl_eyes occl_lower occl_patch; do
  log "start robust_$o (100 images)"
  $PY main.py $ARGS --path_y $W/robust_inputs/${o}100 -i robust_$o/run < /dev/null > $W/logs/robust_$o.log 2>&1 \
    && log "done robust_$o" || log "FAILED robust_$o"
done
log "OCCLUSION DONE"
