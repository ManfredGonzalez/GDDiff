#!/bin/bash

# Base command
BASE_COMMAND="python main.py --config celeba_hq.yml --path_y /home/tico/Desktop/master_research/deid-toolkit/root_dir/datasets/aligned/processingMDDPG --deg deblur_gauss --sigma_y 0.05 --inject_noise 1 --gamma 8 --zeta 0.5 --eta_tilde 0.7"

# Run the command for guidance_Q 0.1, 0.3, and 0.4
for GUIDANCE_Q in 0.0; do
    echo "Running with --per $GUIDANCE_Q"
    $BASE_COMMAND --per $GUIDANCE_Q
done
