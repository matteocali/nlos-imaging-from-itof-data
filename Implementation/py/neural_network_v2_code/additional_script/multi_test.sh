#!/bin/bash

pydir=~/miniconda3/envs/pytorch_nn/bin

$pydir/python3 ../test.py\
    --dts-name="fixed_camera_full_add_layer"\
    --model="nlos_nn_v2_itof_out_add_layer_aug_scale_0.14_iou_1.0_small_lr_big_net_model_lr_5e-05_ochannel_16_l_0.0_addlayers_0_aug_403"\
    --encoder-channels="40, 80, 160, 320, 640"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"\
&\
$pydir/python3 ../test.py\
    --dts-name="fixed_camera_full_add_layer"\
    --model="nlos_nn_v2_itof_out_add_layer_aug_scale_0.14_iou_1.0_small_lr_double_net_model_lr_5e-05_ochannel_16_l_0.0_addlayers_0_aug_403"\
    --encoder-channels="64, 128, 256, 512, 1024"\
    --n-out-channels="16"\
    --additional-layers="0"\
    --bg-value="0"
#&\
#$pydir/python3 ../test.py\
#    --dts-name="fixed_camera_full_add_layer"\
#    --model="tmp_itof_out_add_layer_aug_0.14_0.14_iou_1.0_BCE_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_403"\
#    --encoder-channels="32, 64, 128, 256, 512"\
#    --n-out-channels="16"\
#    --additional-layers="0"\
#    --bg-value="0"\
#&\
#$pydir/python3 ../test.py\
#    --dts-name="fixed_camera_full_add_layer"\
#    --model="tmp_itof_out_add_layer_aug_0.14_0.14_iou_1.0_itof_lovasz_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_403"\
#    --encoder-channels="32, 64, 128, 256, 512"\
#    --n-out-channels="16"\
#    --additional-layers="0"\
#    --bg-value="0"\
#&\
#$pydir/python3 ../test.py\
#    --dts-name="fixed_camera_full_add_layer"\
#    --model="tmp_itof_out_add_layer_aug_0.14_0.14_iou_1.0_softmax_lovasz_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_403"\
#    --encoder-channels="32, 64, 128, 256, 512"\
#    --n-out-channels="16"\
#    --additional-layers="0"\
#    --bg-value="0"\
#&\
#$pydir/python3 ../test.py\
#    --dts-name="fixed_camera_full_add_layer"\
#    --model="nlos_nn_v2_itof_out_add_layer_aug_scale_0.14_iou_3.0_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_403"\
#    --encoder-channels="32, 64, 128, 256, 512"\
#    --n-out-channels="16"\
#    --additional-layers="0"\
#    --bg-value="0"\
#&\
#$pydir/python3 ../test.py\
#    --dts-name="fixed_camera_full_add_layer"\
#    --model="nlos_nn_v2_itof_out_add_layer_aug_scale_0.14_iou_5.0_model_lr_0.0001_ochannel_16_l_0.0_addlayers_0_aug_403"\
#    --encoder-channels="32, 64, 128, 256, 512"\
#    --n-out-channels="16"\
#    --additional-layers="0"\
#    --bg-value="0"\
#&\
#$pydir/python3 ../test.py\
#    --dts-name="fixed_camera_full_add_layer"\
#    --model="nlos_nn_v2_itof_out_add_layer_aug_l_5.0_l2_0_loss_grad-mae_dts_fixed_full_model_lr_0.0001_ochannel_16_l_5.0_addlayers_0_aug_403"\
#    --lr="0.0001"\
#    --encoder-channels="32, 64, 128, 256, 512"\
#    --n-out-channels="16"\
#    --additional-layers="0"\
#    --bg-value="0"\
#&\
#$pydir/python3 ../test.py\
#    --dts-name="fixed_camera_full_add_layer"\
#    --model="nlos_nn_v2_itof_out_add_layer_aug_l_5.0_l2_0_loss_grad-mse_dts_fixed_full_model_lr_0.0001_ochannel_16_l_5.0_addlayers_0_aug_403"\
#    --lr="0.0001"\
#    --encoder-channels="32, 64, 128, 256, 512"\
#    --n-out-channels="16"\
#    --additional-layers="0"\
#    --bg-value="0"\

