#!/bin/bash

cd ..

pydir=~/miniconda3/envs/pytorch_nn/bin

$pydir/python3 test.py\
    --dts-name="fixed_camera_full_add_layer_20"\
    --model="nlos_nn_v2_ablation_20_only_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_403"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
    --dts-name="fixed_camera_full_add_layer_50"\
    --model="nlos_nn_v2_ablation_50_only_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_403"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
    --dts-name="fixed_camera_full_add_layer_60"\
    --model="nlos_nn_v2_ablation_60_only_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_403"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
    --dts-name="fixed_camera_full_add_layer"\
    --model="nlos_nn_v2_ablation_mae_only_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_403"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
    --dts-name="fixed_camera_full_add_layer"\
    --model="nlos_nn_v2_ablation_mse_only_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_403"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
    --dts-name="fixed_camera_full_add_layer"\
    --model="nlos_nn_v2_ablation_mse+iou_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_403"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"
