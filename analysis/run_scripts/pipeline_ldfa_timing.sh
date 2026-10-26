#!/bin/bash
# LDFA wall-clock (rerun with the fixed runner) once the extra pipeline has released the GPU.
W=/home/tico/Desktop/master_research/DeiDDPG/exp/robustness_work
X=$W/extra
E=/home/tico/anaconda3/envs
until grep -q "EXTRA DONE" $W/logs/status_extra.txt 2>/dev/null; do sleep 60; done
OTHER=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')
echo "$(date '+%F %T') ldfa timing start; other GPU processes: ${OTHER:-none}" >> $W/logs/status_extra.txt
sed -i '/^LDFA,/d' $X/timing_baselines.csv
for N in 2 20; do
  rm -rf $X/timing_out/ldfa_$N
  t0=$(date +%s.%N)
  $E/face-anon-simple/bin/python $X/run_ldfa_folder.py $X/bench${N}_flat $X/timing_out/ldfa_$N < /dev/null > $W/logs/timing_LDFA_$N.log 2>&1 \
    || echo "$(date '+%F %T') FAILED ldfa timing $N" >> $W/logs/status_extra.txt
  echo "LDFA,$N,$(echo "$(date +%s.%N) - $t0" | bc)" >> $X/timing_baselines.csv
done
echo "$(date '+%F %T') LDFA TIMING DONE" >> $W/logs/status_extra.txt
