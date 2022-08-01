import numpy as np
import os
import h5py
import matplotlib
import scipy
import scipy.signal
import tensorflow as tf
import pandas as pd
import sys
from tqdm import tqdm

sys.path.append("../training/src/")
sys.path.append("../utils/")  # Adds higher directory to python modules path
import utils

font = {'size': 6}
import PredictiveModel_hidden as PredictiveModel

# import PredictiveModel_2freq_big as PredictiveModel
matplotlib.rc('font', **font)


def phi_remapping(v, d_max=4):
    # Function used to map the phasor to a desired range
    fn = v.shape[-1]
    ampl = np.sqrt(v[..., :fn] ** 2 + v[..., fn:] ** 2)
    phi = np.arctan2(v[..., fn:], v[..., :fn])
    phi = (phi + 2 * utils.pi()) % (2 * utils.pi())
    phi = phi / 7.5 * d_max
    phi = (phi - utils.pi()) % (2 * utils.pi()) - utils.pi()
    v_new = np.copy(v)
    v_new[..., :fn] = ampl * np.cos(phi)
    v_new[..., fn:] = ampl * np.sin(phi)

    return v_new


def test_img(weight_names, data_path, P, freqs, fl_scale, fl_norm_perpixel, fil_dir, fil_den, fil_auto, test_files=None, dim_t=2000):
    ff = freqs.shape[0]
    dim_encoding = ff * 4
    test_names = pd.read_csv(test_files).to_numpy()
    names = [file for file in os.listdir(data_path) if file.endswith(".h5")]
    load_names = []
    for name in names:
        if os.path.basename(name) in test_names:
            load_names.append(name)

    dim_dataset = len(load_names)

    # Define the network and load the corresponding weights
    net = PredictiveModel.PredictiveModel(name='test_result_01', dim_b=dim_dataset, freqs=freqs, P=P,
                                          saves_path='./saves', dim_t=dim_t, fil_size=fil_dir, fil_denoise_size=fil_den,
                                          dim_encoding=dim_encoding, fil_encoder=fil_auto)

    for name in weight_names:
        if name.find("v_e") != -1:
            direct_cnn_weights = name
    net.DirectCNN.load_weights(direct_cnn_weights)

    for name in tqdm(load_names, desc="Testing"):
        with h5py.File(f"{data_path}/{name}", "r") as f:
            tr = f["data"][:]
            gt_depth = f["depth_map"][:].T
            gt_alpha = f["alpha_map"][:].T

        phi = np.transpose(utils.phi(freqs, dim_t, 0.01))
        tr = np.swapaxes(tr, 0, 1)

        v_in = np.matmul(tr, phi)

        # Direct part of the transient
        x_d = np.copy(tr)
        for j in range(x_d.shape[0]):
            for k in range(x_d.shape[1]):
                peaks = np.nanargmax(x_d[j, k, :], axis=0)
                zeros_pos = np.where(x_d[j, k, :] == 0)[0]
                valid_zero_indexes = zeros_pos[np.where(zeros_pos > peaks)]
                if valid_zero_indexes.size == 0:
                    x_d[j, k, :] = 0
                else:
                    x_d[j, k, int(valid_zero_indexes[0]):] = 0

        v_d_gt = np.matmul(x_d, phi)
        v_g_gt = v_in - v_d_gt
        s_pad = int((P - 1) / 2)
        (dim_x, dim_y, dim_t) = tr.shape
        if fl_scale:
            # Compute the scaling kernel for the image pixels
            ampl = np.sqrt(v_in[..., :ff] ** 2 + v_in[..., ff:] ** 2)
            norm_fact = np.ones((P, P)) / P ** 2
            ampl20 = np.squeeze(ampl[..., 0])
            norm_fact = scipy.signal.convolve2d(ampl20, norm_fact, mode="same")
            if fl_norm_perpixel:
                norm_fact = ampl[..., 0]
            v_in = v_in[np.newaxis, :, :, :]
            norm_fact = norm_fact[np.newaxis, ..., np.newaxis]

            # Give the correct number of dimensions to each matrix and then scale them
            v_in /= norm_fact
            v_in[np.isnan(v_in)] = 0
            v_d_gt = v_d_gt[np.newaxis, ...]
            v_d_gt /= norm_fact[0, ...]
            v_d_gt[np.isnan(v_d_gt)] = 0
            v_g_gt = v_g_gt[np.newaxis, ...]
            v_g_gt /= norm_fact[0, ...]
            v_g_gt[np.isnan(v_g_gt)] = 0
            norm_fact = np.squeeze(norm_fact)
            norm_fact = norm_fact[..., np.newaxis]
            x_d /= norm_fact
            x_d[np.isnan(x_d)] = 0
            tr /= norm_fact
            tr[np.isnan(tr)] = 0
        else:
            v_in = v_in[np.newaxis, :, :, :]

        fl_denoise = not (net.P == net.out_win)  # If the two values are different, then the denoising network has been used
        # Make prediction
        v_input = np.pad(v_in, pad_width=[[0, 0], [s_pad, s_pad], [s_pad, s_pad], [0, 0]], mode="edge")
        if fl_denoise:
            v_in_v = net.SpatialNet(v_input)
        else:
            v_in_v = v_input

        [pred_depth, pred_alpha] = net.DirectCNN(v_in_v)
        pred_depth = np.squeeze(pred_depth)
        pred_alpha = np.squeeze(pred_alpha)

        with h5py.File(f"C:/Users/DECaligM/Desktop/New folder/{name}_TEST.h5", "w") as f:
            f.create_dataset("depth_map", data=pred_depth)
            f.create_dataset("alpha_map", data=pred_alpha)
            f.create_dataset("depth_map_gt", data=gt_depth)
            f.create_dataset("alpha_map_gt", data=gt_alpha)
