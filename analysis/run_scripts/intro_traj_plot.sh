#!/bin/bash
T=/home/tico/Desktop/master_research/DeiDDPG/exp/robustness_work/intro_traj
until grep -q "INTRO DONE" $T/status.txt 2>/dev/null; do sleep 60; done
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 30; done
cd /home/tico/Desktop/master_research/deid-toolkit
/home/tico/anaconda3/envs/adaface/bin/python /home/tico/Desktop/master_research/DeiDDPG/analysis/intro_trajectories.py \
  --traj $T/traj --orig $T/in/rafd/Rafd090_29_Moroccan_male_neutral_frontal.jpg \
  --outputs /home/tico/Desktop/master_research/DeiDDPG/exp/image_samples \
  --out /home/tico/Desktop/master_research/DeiDDPG/analysis/results_intro/GauDeiDiff_intro_measured.pdf > $T/plot_log.txt 2>&1 \
  && echo "$(date '+%T') PLOT DONE" >> $T/status.txt || echo "$(date '+%T') PLOT FAILED" >> $T/status.txt
# method overview figure (weighting fields and x0 estimates per branch; same face and trajectories)
/home/tico/anaconda3/envs/adaface/bin/python /home/tico/Desktop/master_research/DeiDDPG/analysis/method_overview.py \
  --traj $T/traj --orig $T/in/rafd/Rafd090_29_Moroccan_male_neutral_frontal.jpg \
  --outputs /home/tico/Desktop/master_research/DeiDDPG/exp/image_samples --bbox 40 49 175 175 \
  --out /home/tico/Desktop/master_research/DeiDDPG/analysis/results_intro/GDDiff_method_overview.pdf
