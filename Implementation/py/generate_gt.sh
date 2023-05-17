#!/bin/sh

pydir=~/miniconda3/envs/mitsuba2/bin

$pydir/python3 general_purposes_code/src/ground_truth_gen.py \
-g "/media/matteocali/shared_hdd/mitsuba_renders/nlos_scenes/datasets/depth_map_ground_truth_close/renders" \
-i "/media/matteocali/shared_hdd/mitsuba_renders/nlos_scenes/datasets/dataset_random_close/renders/" \
-o "/media/matteocali/shared_hdd/mitsuba_renders/nlos_scenes/datasets/test/gt_out" \
-d "/media/matteocali/shared_hdd/mitsuba_renders/nlos_scenes/datasets/test/dts_out" \
-f "/media/matteocali/shared_hdd/mitsuba_renders/nlos_scenes/datasets/test/final_out" \
-t "mirror"