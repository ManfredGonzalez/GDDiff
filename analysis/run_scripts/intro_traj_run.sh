#!/bin/bash
# Intro figure: per-step x0 trajectories of GDDiff on one RaFD face (waits for free GPU memory; shared server).
T=/home/tico/Desktop/master_research/DeiDDPG/exp/robustness_work/intro_traj
cd /home/tico/Desktop/master_research/DeiDDPG
for P in 0.1 0.3 0.5; do
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 4500 ]; do sleep 30; done
  rm -rf exp/image_samples/intro_traj_$P
  GDDIFF_TRAJ_LOG=$T/traj /home/tico/anaconda3/envs/gddiff/bin/python main.py --config celeba_hq.yml --deg deblur_gauss \
    --sigma_y 0.05 --inject_noise 1 --gamma 8 --zeta 0.5 --eta_tilde 0.7 --per $P --path_y $T/in -i intro_traj_$P/run \
    < /dev/null > $T/log_$P.txt 2>&1 && echo "$(date '+%T') done $P" >> $T/status.txt || echo "$(date '+%T') FAILED $P" >> $T/status.txt
done
echo "$(date '+%T') INTRO DONE" >> $T/status.txt
