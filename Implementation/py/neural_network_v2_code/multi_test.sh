#!/bin/bash

pydir=~/miniconda3/envs/pytorch_nn/bin

$pydir/python3 test.py\
	--dts-name="fixed_camera_diffuse_wall_add_layer"\
	--model="nlos_nn_v2_itof_out_add_layer_aug_l_0.3_loss_grad-mae_model_lr_0.0001_ochannel_16_l_0.3_addlayers_0_aug_202"\
    --lr="0.0001"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
	--dts-name="fixed_camera_diffuse_wall_add_layer"\
	--model="nlos_nn_v2_itof_out_add_layer_aug_l_0.3_loss_grad-mse_model_lr_0.0001_ochannel_16_l_0.3_addlayers_0_aug_202"\
    --lr="0.0001"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
	--dts-name="fixed_camera_diffuse_wall_add_layer"\
	--model="nlos_nn_v2_itof_out_add_layer_aug_l_0.3_loss_ssim_model_lr_0.0001_ochannel_16_l_0.3_addlayers_0_aug_202"\
    --lr="0.0001"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
	--dts-name="fixed_camera_diffuse_wall_add_layer"\
	--model="nlos_nn_v2_itof_out_add_layer_aug_l_0.5_loss_grad-mae_model_lr_0.0001_ochannel_16_l_0.5_addlayers_0_aug_202"\
    --lr="0.0001"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
	--dts-name="fixed_camera_diffuse_wall_add_layer"\
	--model="nlos_nn_v2_itof_out_add_layer_aug_l_0.5_loss_grad-mse_model_lr_0.0001_ochannel_16_l_0.5_addlayers_0_aug_202"\
    --lr="0.0001"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
	--dts-name="fixed_camera_diffuse_wall_add_layer"\
	--model="nlos_nn_v2_itof_out_add_layer_aug_l_0.7_loss_grad-mae_model_lr_0.0001_ochannel_16_l_0.7_addlayers_0_aug_202"\
    --lr="0.0001"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
	--dts-name="fixed_camera_diffuse_wall_add_layer"\
	--model="nlos_nn_v2_itof_out_add_layer_aug_l_0.7_loss_grad-mse_model_lr_0.0001_ochannel_16_l_0.7_addlayers_0_aug_202"\
    --lr="0.0001"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
	--dts-name="fixed_camera_diffuse_wall_add_layer"\
	--model="nlos_nn_v2_itof_out_add_layer_aug_l_1.0_loss_grad-mse_model_lr_0.0001_ochannel_16_l_1.0_addlayers_0_aug_202"\
    --lr="0.0001"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
