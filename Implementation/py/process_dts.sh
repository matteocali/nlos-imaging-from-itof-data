#!/bin/sh
python3 neural_network_v2_code/process_dts.py \
--name "full_dts_add_layer" \
--input "/media/matteocali/shared_ssd/NLoS imaging using iToF/mirror_dts/full_dts" \
--bg-value "0" \
--add-layer "True" \
--multi_freqs "False" \
--data-augment-size "780" \
--slurm "False"
