#!/bin/bash
# Simpler-mask baselines (reviewer R1.4) + full ablation metrics (detached; safe to disconnect).
REPO=/home/tico/Desktop/master_research/DeiDDPG
TK=/home/tico/Desktop/master_research/deid-toolkit
W=$REPO/exp/robustness_work
E=/home/tico/anaconda3/envs
ARGS="--config celeba_hq.yml --deg deblur_gauss --sigma_y 0.05 --inject_noise 1 --gamma 8 --zeta 0.5 --eta_tilde 0.7 --per 0.5"
log() { echo "$(date '+%F %T') $*" >> $W/logs/status_kernels.txt; }
log "start"
cd $REPO
for M in uniform zero_all; do
  ( rm -rf exp/image_samples/kern300_$M
    GDDIFF_KERNEL_MODE=$M $E/gddiff/bin/python main.py $ARGS --path_y $W/robust_inputs/clean -i kern300_$M/run \
      < /dev/null > $W/logs/kern300_$M.log 2>&1 && log "done generation $M" || log "FAILED generation $M" ) &
done
wait
cd $TK
$E/adaface/bin/python $REPO/analysis/ablation_metrics.py --part identity < /dev/null > $W/logs/kern_identity.log 2>&1 \
  && log "done identity" || log "FAILED identity"
$E/toolkit/bin/python $REPO/analysis/ablation_metrics.py --part quality < /dev/null > $W/logs/kern_quality.log 2>&1 \
  && log "done quality" || log "FAILED quality"
$E/gddiff/bin/python $REPO/analysis/ablation_metrics.py --part summary < /dev/null > $W/logs/kern_summary.txt 2>&1 \
  && log "done summary" || log "FAILED summary"
cd $REPO
for M in gaze3d gazetr semiuhpe 6drepnet360 qcs fmae hycnxtvit fairface; do
  $E/gaze/bin/python analysis/ablation_utility.py --part predict --model $M < /dev/null > $W/logs/kern_utility_$M.log 2>&1 \
    && log "done utility $M" || log "FAILED utility $M"
done
$E/mivolo/bin/python analysis/ablation_utility.py --part predict --model mivolo < /dev/null > $W/logs/kern_utility_mivolo.log 2>&1 \
  && log "done utility mivolo" || log "FAILED utility mivolo"
$E/gaze/bin/python analysis/ablation_utility.py --part compare < /dev/null > $W/logs/kern_utility_compare.txt 2>&1 \
  && log "done utility compare" || log "FAILED utility compare"
log "KERNELS DONE"
