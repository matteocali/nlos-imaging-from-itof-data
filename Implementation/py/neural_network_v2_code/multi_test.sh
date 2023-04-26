#!/bin/bash

pydir=~/miniconda3/envs/pytorch_nn/bin

$pydir/python3 test.py\
    --dts-name="full_dts_add_layer"\
    --model="nlos_nn_v2_itof_out_add_layer_aug_l_0.3_loss_grad-mae_dts_full_model_lr_0.0001_ochannel_16_l_0.3_addlayers_0_aug_806"\
    --lr="0.0001"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
    --dts-name="full_dts_add_layer"\
    --model="nlos_nn_v2_itof_out_add_layer_aug_l_0.3_loss_grad-mse_dts_full_model_lr_0.0001_ochannel_16_l_0.3_addlayers_0_aug_806"\
    --lr="0.0001"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
    --dts-name="full_dts_add_layer"\
    --model="nlos_nn_v2_itof_out_add_layer_aug_l_0.3_loss_ssim_dts_full_model_lr_0.0001_ochannel_16_l_0.3_addlayers_0_aug_806"\
    --lr="0.0001"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
    &\
$pydir/python3 test.py\
    --dts-name="full_dts_add_layer"\
    --model="nlos_nn_v2_itof_out_add_layer_aug_l_0.7_loss_grad-mae_dts_full_model_lr_0.0001_ochannel_16_l_0.7_addlayers_0_aug_806"\
    --lr="0.0001"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
    
