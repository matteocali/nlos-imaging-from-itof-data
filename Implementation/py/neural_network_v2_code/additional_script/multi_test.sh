#!/bin/bash

cd ..

pydir=~/miniconda3/envs/pytorch_nn/bin

$pydir/python3 test.py\
    --dts-name="full_dts_add_layer"\
    --model="nlos_nn_v2_itof_out_add_layer_aug_STE_full_dts_1_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_806"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
    --dts-name="full_dts_add_layer"\
    --model="nlos_nn_v2_itof_out_add_layer_aug_STE_full_dts_2_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_806"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
    --dts-name="full_dts_add_layer"\
    --model="nlos_nn_v2_itof_out_add_layer_aug_STE_full_dts_3_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_806"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
    --dts-name="full_dts_add_layer"\
    --model="nlos_nn_v2_itof_out_add_layer_aug_STE_full_dts_4_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_806"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
    --dts-name="full_dts_add_layer"\
    --model="nlos_nn_v2_itof_out_add_layer_aug_STE_full_dts_5_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_806"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
    --dts-name="full_dts_add_layer"\
    --model="nlos_nn_v2_itof_out_add_layer_aug_STE_full_dts_6_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_806"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
    --dts-name="full_dts_add_layer"\
    --model="nlos_nn_v2_itof_out_add_layer_aug_STE_full_dts_7_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_806"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 test.py\
    --dts-name="full_dts_add_layer"\
    --model="nlos_nn_v2_itof_out_add_layer_aug_STE_full_dts_8_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_806"\
    --lr="0.0001"\
    --encoder-channels="32, 64, 128, 256, 512"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"
#&\
#$pydir/python3 test.py\
#    --dts-name="full_dts_add_layer"\
#    --model="nlos_nn_v2_itof_out_add_layer_aug_l_5.0_l2_0_loss_grad-mse_dts_fixed_full_model_lr_0.0001_ochannel_16_l_5.0_addlayers_0_aug_403"\
#    --lr="0.0001"\
#    --encoder-channels="32, 64, 128, 256, 512"\
#    --n-out-channels="16"\
#    --additional-layers="0"\
#    --bg-value="0"\

