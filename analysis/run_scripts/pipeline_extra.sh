#!/bin/bash
# Extra revision experiments (detached; safe to disconnect). Starts after the t_mu sweep.
#   (1) wall-clock of selected baselines on the 20-image benchmark (needs an idle GPU)
#   (2) reconstruction attack: de-identified outputs re-restored by the plain single-branch sampler
#   (3) eyeglasses preservation on GlassesGAN images: GDDiff (lambda 0.1, 0.5) and FAMS
REPO=/home/tico/Desktop/master_research/DeiDDPG
TK=/home/tico/Desktop/master_research/deid-toolkit
TECH=$TK/root_dir/techniques
FAMS=/home/tico/Desktop/master_research/face_anon_simple
W=$REPO/exp/robustness_work
X=$W/extra
E=/home/tico/anaconda3/envs
ARGS="--config celeba_hq.yml --deg deblur_gauss --sigma_y 0.05 --inject_noise 1 --gamma 8 --zeta 0.5 --eta_tilde 0.7"
ST=$W/logs/status_extra.txt
log() { echo "$(date '+%F %T') $*" >> $ST; }
log "waiting for sweep"
until grep -q "SWEEP DONE" $W/logs/status_sweep.txt 2>/dev/null; do sleep 60; done

# (1) timing: GDDiff and the baselines timed back to back under the same GPU conditions (shared server; never stop
#     other users' processes such as the lab's Ollama server); the processes present are logged
OTHER=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')
log "timing start; other GPU processes: ${OTHER:-none}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > $X/timing_gpu.txt
echo "method,n_images,seconds" > $X/timing_baselines.csv
tm() {  # tm <method> <n> <command...>
  local m=$1 n=$2; shift 2
  local t0=$(date +%s.%N)
  ( "$@" ) < /dev/null >> $W/logs/timing_$m.log 2>&1 || log "FAILED timing $m $n"
  echo "$m,$n,$(echo "$(date +%s.%N) - $t0" | bc)" >> $X/timing_baselines.csv
}
for N in 2 20; do
  IN=$X/bench${N}_flat; O=$X/timing_out
  rm -rf $O; mkdir -p $O
  tm FAMS $N bash -c "cd $FAMS && $E/face-anon-simple/bin/python infer_referencenet.py --dataset_loading_script_path $IN --output_dir $O/fams_$N"
  tm LDFA $N $E/face-anon-simple/bin/python $X/run_ldfa_folder.py $IN $O/ldfa_$N
  tm AMT-GAN $N bash -c "mkdir -p $O/amt_$N && cd $TECH/AMT-GAN && $E/AMT-GAN/bin/python -u test.py --source_dir $IN --save_path $O/amt_$N"
  tm CLEANIR $N bash -c "mkdir -p $O/cleanir_$N && cd $TECH && $E/cleanir/bin/python cleanir.py $IN $O/cleanir_$N"
  mkdir -p $O/gd_$N/in && cp $IN/* $O/gd_$N/in/  # same wall-clock protocol for GDDiff (lambda 0.5, three outputs)
  tm GDDiff $N bash -c "cd $REPO && $E/gddiff/bin/python main.py $ARGS --per 0.5 --path_y $O/gd_$N -i timing_wall_$N/run"
  rm -rf $REPO/exp/image_samples/timing_wall_$N
done
log "done timing"

# (2) + (3) generation in parallel
attack() {
  for V in atk_full atk_lam0.1 atk_gauss atk_lapl atk_orig; do
    rm -rf $REPO/exp/image_samples/attack300_$V
    (cd $REPO && GDDIFF_SINGLE_BRANCH=1 $E/gddiff/bin/python main.py $ARGS --per 0.5 --path_y $X/attack_in/$V \
      -i attack300_$V/run < /dev/null > $W/logs/attack300_$V.log 2>&1) && log "done attack $V" || log "FAILED attack $V"
  done
}
glasses() {
  rm -rf $REPO/exp/image_samples/glasses300_$1
  (cd $REPO && $E/gddiff/bin/python main.py $ARGS --per $1 --path_y $X/glasses_in/glasses -i glasses300_$1/run \
    < /dev/null > $W/logs/glasses300_$1.log 2>&1) && log "done glasses gddiff $1" || log "FAILED glasses gddiff $1"
}
fams() {
  for S in rafd xm2vts; do
    rm -rf $X/fams_glasses/$S
    (cd $FAMS && $E/face-anon-simple/bin/python infer_referencenet.py --dataset_loading_script_path $X/glasses_in/glasses/$S \
      --output_dir $X/fams_glasses/$S < /dev/null > $W/logs/fams_glasses_$S.log 2>&1) && log "done glasses fams $S" || log "FAILED glasses fams $S"
  done
}
attack & glasses 0.1 & glasses 0.5 & fams &
wait

# metrics
cd $TK
$E/adaface/bin/python $REPO/analysis/ablation_metrics.py --part identity < /dev/null > $W/logs/extra_identity.log 2>&1 \
  && log "done identity" || log "FAILED identity"
$E/toolkit/bin/python $REPO/analysis/ablation_metrics.py --part quality < /dev/null > $W/logs/extra_quality.log 2>&1 \
  && log "done quality" || log "FAILED quality"
$E/gddiff/bin/python $REPO/analysis/ablation_metrics.py --part summary < /dev/null > $W/logs/extra_summary.txt 2>&1 \
  && log "done summary" || log "FAILED summary"
cd $REPO
$E/face-anon-simple/bin/python analysis/glasses_study.py < /dev/null > $W/logs/glasses_summary.txt 2>&1 \
  && log "done glasses study" || log "FAILED glasses study"
log "EXTRA DONE"
