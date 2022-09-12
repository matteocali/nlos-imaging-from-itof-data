import numpy as np
import os
import h5py
from matplotlib import pyplot as plt
import pandas as pd
import sys
from tqdm import tqdm
import tensorflow as tf
sys.path.append("../training/src/")
sys.path.append("../utils/")  # Adds higher directory to python modules path
import utils
import PredictiveModel_hidden as PredictiveModel


def test_img(weight_names, data_path, out_path, P, freqs, fl_scale, fil_dir, lr, loss_fn="mae", n_single_layers=None, test_files=None,
             dim_t=2000, return_vals=False, plot_results=False):

    ff = freqs.shape[0]
    test_names = pd.read_csv(test_files, header=None).to_numpy()
    names = [file for file in os.listdir(data_path) if file.endswith(".h5")]
    load_names = []
    for name in names:
        if os.path.basename(name) in test_names:
            load_names.append(name)

    dim_dataset = len(load_names)

    # Define the network and load the corresponding weights
    net = PredictiveModel.PredictiveModel(name='test_result_01', dim_b=dim_dataset, freqs=freqs, P=P,
                                          saves_path='./saves', dim_t=dim_t, fil_size=fil_dir, lr=lr,
                                          loss_name=loss_fn, single_layers=n_single_layers)

    for name in weight_names:
        if name.find("v_e") != -1:
            direct_cnn_weights = name
    net.DirectCNN.load_weights(direct_cnn_weights)

    for name in tqdm(load_names, desc="Testing"):
        with h5py.File(f"{data_path}/{name}", "r") as f:
            tr = f["data"][:]
            gt_depth = f["depth_map"][:]
            gt_depth = np.swapaxes(gt_depth, 0, 1)
            gt_alpha = f["alpha_map"][:]
            gt_alpha = np.swapaxes(gt_alpha, 0, 1)

        s_pad = int((P - 1) / 2)
        dim_x, dim_y, dim_t = tr.shape

        # Fix first peak before the computation
        tr = np.reshape(tr, ((dim_x * dim_y), dim_t))

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

        # Compute the iToF data
        phi = np.transpose(utils.phi(freqs=freqs, dim_t=dim_t, exp_time=0.01))
        v_in = np.matmul(tr, phi)
        tr = np.reshape(tr, (dim_x, dim_y, dim_t))
        v_in = np.reshape(v_in, (dim_x, dim_y, ff*2))

        if fl_scale:
            # Compute the scaling kernel for the image pixels
            ampl20 = np.sqrt(v_in[..., 0] ** 2 + v_in[..., ff] ** 2)
            norm_fact = ampl20[..., np.newaxis]

            # Give the correct number of dimensions to each matrix and then scale them
            v_in /= norm_fact
            v_in[np.isnan(v_in)] = 0
            v_in = v_in[np.newaxis, ...]
        else:
            v_in = v_in[np.newaxis, :, :, :]

        # Make prediction
        v_in = np.swapaxes(v_in, 1, 2)
        v_in = np.pad(v_in, pad_width=[[0, 0], [s_pad, s_pad], [s_pad, s_pad], [0, 0]], mode="reflect")

        [pred_depth, pred_alpha] = net.DirectCNN(tf.convert_to_tensor(v_in))
        pred_depth = np.squeeze(pred_depth)
        pred_alpha = np.squeeze(pred_alpha)

        if plot_results:  # Plot the results
            pred_depth_masked = pred_depth * gt_alpha
            gt_depth_masked = gt_depth * gt_alpha

            pred_alpha_masked_ones = pred_alpha * gt_alpha
            num_ones = np.sum(gt_alpha)
            alpha_mae_obj = np.sum(np.abs(pred_alpha_masked_ones - gt_alpha)) / num_ones
            pred_alpha_masked_zeros = pred_alpha * (1 - gt_alpha)
            num_zeros = np.sum(1 - gt_alpha)
            alpha_mae_bkg = np.sum(np.abs(pred_alpha_masked_zeros - np.zeros(gt_alpha.shape, dtype=np.float32))) / num_zeros
            alpha_mae = np.sum(alpha_mae_obj + alpha_mae_bkg) / 2

            pred_depth_masked_ones = pred_depth * gt_alpha
            depth_mae_obj = np.sum(np.abs(pred_depth_masked_ones - gt_alpha)) / num_ones
            pred_depth_masked_zeros = pred_depth * (1 - gt_alpha)
            depth_mae_bkg = np.sum(np.abs(pred_depth_masked_zeros - np.zeros(gt_alpha.shape, dtype=np.float32))) / num_zeros
            depth_mae = np.sum(depth_mae_obj + depth_mae_bkg) / 2

            fig, ax = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
            fig.suptitle(name[:-3])
            img0 = ax[0, 0].matshow(gt_depth_masked, cmap='jet')
            img0.set_clim(np.min(gt_depth), np.max(gt_depth))
            cax = fig.add_axes([ax[0, 0].get_position().x1 + 0.005, ax[0, 0].get_position().y0, 0.015, ax[0, 0].get_position().height])
            fig.colorbar(img0, cax=cax)
            ax[0, 0].set_title("Ground truth depth map")
            ax[0, 0].set_xlabel("Column pixel")
            ax[0, 0].set_ylabel("Row pixel")
            img1 = ax[0, 1].matshow(pred_depth_masked, cmap='jet')
            cax = fig.add_axes([ax[0, 1].get_position().x1 + 0.005, ax[0, 1].get_position().y0, 0.015, ax[0, 1].get_position().height])
            fig.colorbar(img1, cax=cax)
            img1.set_clim(np.min(gt_depth), np.max(gt_depth))
            ax[0, 1].set_title("Predicted depth map")
            ax[0, 1].set_xlabel("Column pixel")
            ax[0, 1].set_ylabel("Row pixel")
            img2 = ax[0, 2].matshow(np.abs(gt_depth_masked - pred_depth_masked), cmap='jet')
            cax = fig.add_axes([ax[0, 2].get_position().x1 + 0.005, ax[0, 2].get_position().y0, 0.015, ax[0, 2].get_position().height])
            fig.colorbar(img1, cax=cax)
            img2.set_clim(np.min(gt_depth), np.max(gt_depth))
            ax[0, 2].text(25, 25, "MAE: " + str(round(depth_mae, 3)), ha='left', va='center', fontsize=6, color='white')
            ax[0, 2].set_title("Absolute difference of depth map")
            ax[0, 2].set_xlabel("Column pixel")
            ax[0, 2].set_ylabel("Row pixel")
            img3 = ax[1, 0].matshow(gt_alpha, cmap='jet')
            cax = fig.add_axes([ax[1, 0].get_position().x1 + 0.005, ax[1, 0].get_position().y0, 0.015, ax[1, 0].get_position().height])
            fig.colorbar(img3, cax=cax)
            ax[1, 0].set_title("Ground truth alpha map")
            ax[1, 0].set_xlabel("Column pixel")
            ax[1, 0].set_ylabel("Row pixel")
            img4 = ax[1, 1].matshow(pred_alpha, cmap='jet')
            cax = fig.add_axes([ax[1, 1].get_position().x1 + 0.005, ax[1, 1].get_position().y0, 0.015, ax[1, 1].get_position().height])
            fig.colorbar(img4, cax=cax)
            img4.set_clim(np.min(gt_alpha), np.max(gt_alpha))
            ax[1, 1].set_title("Predicted alpha map")
            ax[1, 1].set_xlabel("Column pixel")
            ax[1, 1].set_ylabel("Row pixel")
            img5 = ax[1, 2].matshow(np.abs(gt_alpha - pred_alpha), cmap='jet')
            ax[1, 2].text(25, 25, "MAE: " + str(round(alpha_mae, 3)), ha='left', va='center', fontsize=6, color='white')
            cax = fig.add_axes([ax[1, 2].get_position().x1 + 0.005, ax[1, 2].get_position().y0, 0.015, ax[1, 2].get_position().height])
            fig.colorbar(img5, cax=cax)
            ax[1, 2].set_title("Absolute difference of alpha map")
            ax[1, 2].set_xlabel("Column pixel")
            ax[1, 2].set_ylabel("Row pixel")
            #fig.tight_layout()
            plt.savefig(f"{out_path}/{name[:-3]}_PLOTS.svg")
            plt.close()

        if not return_vals:
            with h5py.File(f"{out_path}/{name[:-3]}_TEST.h5", "w") as f:
                f.create_dataset("depth_map", data=pred_depth)
                f.create_dataset("alpha_map", data=pred_alpha)
                f.create_dataset("depth_map_gt", data=gt_depth)
                f.create_dataset("alpha_map_gt", data=gt_alpha)
        else:
            return pred_depth, pred_alpha, gt_depth, gt_alpha
