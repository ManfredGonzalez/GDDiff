#!/bin/bash
# WORK: folder with the 64 probe images (probe_data/) and outputs (fusion/, fusion_logs/)
WORK=${WORK:-exp/variance_work}
# worker $1 of 2: runs its share of (per, seed) combinations
cd /home/tico/Desktop/master_research/DeiDDPG
n=0
for PER in 0.1 0.2 0.5; do for SEED in 1 2 3 4 5 6 7 8; do
  n=$((n+1)); [ $((n % 2)) -ne $1 ] && continue
  GDDIFF_FUSION_LOG=$WORK/fusion/per$PER ~/anaconda3/envs/gddiff/bin/python main.py --config celeba_hq.yml     --path_y $WORK/probe_data --deg deblur_gauss --sigma_y 0.05 --inject_noise 1 --gamma 8 --zeta 0.5     --eta_tilde 0.7 --per $PER --seed $SEED -i variance_probe/per${PER}_seed${SEED} > $WORK/fusion_logs/per${PER}_seed${SEED}.log 2>&1     || echo "FAILED per=$PER seed=$SEED"
  echo "done per=$PER seed=$SEED"
done; done
