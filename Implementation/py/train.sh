#!/bin/sh
python3 neural_network_v2_code/train.py \
--dataset "fixed_camera_diffuse_wall" \
--name "test5_only_two_final_conv" \
--lr "0.0001" \
--lambda "0.2" \
--n-out-channels "8" \
--n-epochs "5000" \
--slurm "False"