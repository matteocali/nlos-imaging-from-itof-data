import numpy as np
import os
import tensorflow as tf
import h5py
import pandas as pd
import matplotlib
from datetime import date
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append("../training/src/")
sys.path.append("../utils/")  # Adds higher directory to python modules path
import utils
import PredictiveModel_hidden as PredictiveModel
import csv


# Function for transient data prediction on single pixels
# It also presents the comparison with the pixels from iToF2dToF


def test_synth(weight_names, P, freqs, lr, fl_scale=False, fil_dir=8, dim_t=2000, fl_test_img=False, test_files=None, dts_path=None, processed_dts_path=None):
    ff = freqs.shape[0]
    mid = int((P - 1) / 2)  # index pointing at the middle element of the patch

    if processed_dts_path is not None and not fl_test_img:
        with h5py.File(processed_dts_path, "r") as f:
            v_in = f["raw_itof"][:]
            gt_alpha = f["gt_alpha"][:]
            gt_depth = f["gt_depth"][:]

        dim_dataset = v_in.shape[0]

        # SCALING
        if fl_scale:
            v_a = np.sqrt(v_in[..., 0]**2 + v_in[..., ff]**2)
            v_a = v_a[..., np.newaxis]

            # Scale all factors
            v_in /= v_a

        # Load the model
        net = PredictiveModel.PredictiveModel(name=f'test_patches_{date.today()}', dim_b=dim_dataset, freqs=freqs, P=P,
                                              saves_path='./saves', dim_t=dim_t, fil_size=fil_dir, lr=lr)

        # Load the weights
        net.DirectCNN.load_weights(weight_names[0])


        # Make prediction
        #v_input = tf.convert_to_tensor(v_in,dtype="float32")
        v_in_v = np.copy(v_in)
        [pred_depth, pred_alpha] = net.DirectCNN(v_in_v)

        # Compute metrics
        print("METRICS:\n")
        print("Mae computation")

        # Extract just the single pixel from the gt
        gt_alpha = tf.convert_to_tensor(gt_alpha, dtype="float32")
        gt_depth = tf.convert_to_tensor(gt_depth, dtype="float32")
        gt_depth = tf.slice(gt_depth, begin=[0, mid, mid], size=[-1, 1, 1])
        gt_alpha = tf.slice(gt_alpha, begin=[0, mid, mid], size=[-1, 1, 1])

        # Process the output with the Direct CNN
        pred_depth = tf.squeeze(pred_depth, axis=-1)
        pred_alpha = tf.squeeze(pred_alpha, axis=-1)

        alpha_mae = np.mean(np.abs(pred_alpha - gt_alpha))
        depth_mae = np.mean(np.abs((pred_depth * gt_alpha) - gt_depth))

        print("  - Alpha mae: ", alpha_mae)
        print("  - Depth mae: ", depth_mae)


    # Perform the inference on full images
    if dts_path is not None and test_files is not None and fl_test_img:
        files_names = pd.read_csv(test_files, header=None).to_numpy()
        names = [file for file in os.listdir(dts_path) if file.endswith(".h5")]
        load_names = []
        for name in names:
            if os.path.basename(name) in files_names:
                load_names.append(name)

        dim_dataset = len(load_names)

        # Define the network and load the corresponding weights
        net = PredictiveModel.PredictiveModel(name=f'test_img_as_patches_{date.today()}', dim_b=dim_dataset,
                                              freqs=freqs, P=P, saves_path='./saves', dim_t=dim_t,
                                              fil_size=fil_dir, lr=lr)

        # Load the weights
        net.DirectCNN.load_weights(weight_names[0])

        for name in tqdm(load_names, desc="Testing"):
            with h5py.File(f"{dts_path}/{name}", "r") as f:
                tr = f["data"][:]
                tr = np.swapaxes(tr, 0, 1)
                gt_depth = f["depth_map"][:]
                gt_depth = np.swapaxes(gt_depth, 0, 1)
                gt_alpha = f["alpha_map"][:]
                gt_alpha = np.swapaxes(gt_alpha, 0, 1)

            # Fix first peak before the computation
            tr_dim_0 = tr.shape[0]
            tr_dim_1 = tr.shape[1]
            tr = tr.reshape((tr_dim_0 * tr_dim_1, tr.shape[2]))
            ind_maxima = np.argmax(tr, axis=-1)
            val_maxima = np.zeros(ind_maxima.shape, dtype=np.float32)
            ind_end_direct = np.zeros(ind_maxima.shape, dtype=np.int32)

            for j in range(tr.shape[0]):
                zeros_pos = np.where(tr[j] == 0)[0]  # find the index of the zeros
                ind_end_direct[j] = zeros_pos[np.where(zeros_pos > ind_maxima[j])][0]  # find the index of the zeros after the first peak
            for j in range(ind_maxima.shape[0]):
                val_maxima[j] = np.sum(tr[j, :ind_end_direct[j]])  # compute the value of the first peak considering the sum of the values before the global
            for j in range(tr.shape[0]):
                tr[j, :ind_end_direct[j]] = 0  # set the values before the global to zero
            for j in range(tr.shape[0]):
                tr[j, ind_maxima[j]] = val_maxima[j]  # set the value of the first peak to the value computed before

            tr = tr.reshape((tr_dim_0, tr_dim_1, tr.shape[1]))

            phi = np.transpose(utils.phi(freqs, dim_t, 0.01))
            v_in = np.matmul(tr, phi)

            # SCALING
            if fl_scale:
                v_a = np.sqrt(v_in[..., 0] ** 2 + v_in[..., ff] ** 2)
                v_a = v_a[..., np.newaxis]

                # Scale all factors
                v_in /= v_a

            # Add padding to the image
            v_in = np.pad(v_in, pad_width=[[mid, mid], [mid, mid], [0, 0]], mode="reflect")

            # Split the images into patches
            pred_alpha_img = np.zeros((v_in.shape[0] - P + 1, v_in.shape[1] - P + 1), dtype=np.float32)
            pred_depth_img = np.zeros((v_in.shape[0] - P + 1, v_in.shape[1] - P + 1), dtype=np.float32)
            for i in range(mid, v_in.shape[0] - mid):
                for j in range(mid, v_in.shape[1] - mid):
                    pred_depth, pred_alpha = tf.squeeze(net.DirectCNN(v_in[np.newaxis, i - mid:i + mid + 1, j - mid:j + mid + 1, ...])).numpy()
                    pred_alpha_img[i - mid, j - mid] = pred_alpha
                    pred_depth_img[i - mid, j - mid] = pred_depth
