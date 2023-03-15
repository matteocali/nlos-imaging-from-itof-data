import numpy as np
import torch
from utils.utils import depth2itof, itof2depth
from matplotlib import pyplot as plt


if __name__ == '__main__':
    train_dts = torch.load('neural_network_v2_code/datasets/fixed_camera_diffuse_wall_add_layer/processed_data/processed_train_dts.pt')
    sample = train_dts[0]
    gt_depth = sample['gt_depth']

    conv_itof = depth2itof(gt_depth, 20e06)
    conv_depth = itof2depth(conv_itof, 20e06)  # type: ignore

    pass