#!/bin/bash
# t_mu (lambda) sweep of Mean-GDDiff on the 300-image subset + all metrics (detached; safe to disconnect).
REPO=/home/tico/Desktop/master_research/DeiDDPG
TK=/home/tico/Desktop/master_research/deid-toolkit
W=$REPO/exp/robustness_work
E=/home/tico/anaconda3/envs
ARGS="--config celeba_hq.yml --deg deblur_gauss --sigma_y 0.05 --inject_noise 1 --gamma 8 --zeta 0.5 --eta_tilde 0.7"
log() { echo "$(date '+%F %T') $*" >> $W/logs/status_sweep.txt; }
gen() {
  for P in "$@"; do
    rm -rf $REPO/exp/image_samples/sweep300_$P
    (cd $REPO && $E/gddiff/bin/python main.py $ARGS --per $P --path_y $W/robust_inputs/clean -i sweep300_$P/run \
      < /dev/null > $W/logs/sweep300_$P.log 2>&1) && log "done generation $P" || log "FAILED generation $P"
  done
}
log "start"
gen 0.0 0.3 0.8 &
gen 0.1 0.4 0.9 &
gen 0.2 0.6 0.7 &
wait
cd $TK
$E/adaface/bin/python $REPO/analysis/ablation_metrics.py --part identity < /dev/null > $W/logs/sweep_identity.log 2>&1 \
  && log "done identity" || log "FAILED identity"
$E/toolkit/bin/python $REPO/analysis/ablation_metrics.py --part quality < /dev/null > $W/logs/sweep_quality.log 2>&1 \
  && log "done quality" || log "FAILED quality"
$E/gddiff/bin/python $REPO/analysis/ablation_metrics.py --part summary < /dev/null > $W/logs/sweep_summary.txt 2>&1 \
  && log "done summary" || log "FAILED summary"
cd $REPO
for M in gaze3d gazetr semiuhpe 6drepnet360 qcs fmae hycnxtvit fairface; do
  $E/gaze/bin/python analysis/ablation_utility.py --part predict --model $M < /dev/null > $W/logs/sweep_utility_$M.log 2>&1 \
    && log "done utility $M" || log "FAILED utility $M"
done
$E/mivolo/bin/python analysis/ablation_utility.py --part predict --model mivolo < /dev/null > $W/logs/sweep_utility_mivolo.log 2>&1 \
  && log "done utility mivolo" || log "FAILED utility mivolo"
$E/gaze/bin/python analysis/ablation_utility.py --part compare < /dev/null > $W/logs/sweep_utility_compare.txt 2>&1 \
  && log "done utility compare" || log "FAILED utility compare"
$E/gaze/bin/python analysis/tmu_sweep.py < /dev/null > $W/logs/sweep_table.txt 2>&1 \
  && log "done sweep table" || log "FAILED sweep table"
log "SWEEP DONE"
