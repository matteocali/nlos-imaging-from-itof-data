import numpy as np
import os
import tensorflow as tf
import h5py
import math
import matplotlib
import scipy
import scipy.signal
import glob
import matplotlib.pyplot as plt
import time
import pandas as pd
import cv2
import sys

sys.path.append("../training/src/")
sys.path.append("../utils/")  # Adds higher directory to python modules path
import DataLoader
import GenerativeModel
import utils
import depth_estimation

font = {'size': 6}
# import PredictiveModel_itof2dtof as PredictiveModel
import PredictiveModel_hidden as PredictiveModel
# import PredictiveModel_hidden_2 as PredictiveModel
import PredictiveModel_old
from fun_metrics_computation_transient_images import metrics_computation_transient_images

# import PredictiveModel_2freq_big as PredictiveModel
matplotlib.rc('font', **font)


def phi_remapping(v, d_min=0, d_max=4):
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


def test_img(name, weights_path, attempt_name, Sname, P, freqs, fl_scale, fl_norm_perpixel, fil_dir, fil_den, fil_auto,
             fl_test_old=False, old_weights=None, Pold=3, fl_newhidden=True, dim_t=2000):
    ff = freqs.shape[0]
    dim_encoding = ff * 4
    path = "../dataset_rec/"
    test_files = "../dataset_creation/data_split/test_images.csv"
    test_names = pd.read_csv(test_files).to_numpy()
    basenames = []
    for name in test_names:
        name = os.path.basename(name[0])
        basenames.append(name)
    names = glob.glob(path + "*.h5*")
    loadnames = []
    for name in names:
        if os.path.basename(name) in basenames:
            loadnames.append(name)

    dim_dataset = 1

    # If we are testing also the old network, load the weights and needed data to test that one too
    if fl_test_old:
        net_old = PredictiveModel_old.PredictiveModel(name='DeepBVE_nf_std', dim_b=dim_dataset, freqs=freqs, P=Pold,
                                                      saves_path='./saves', fil_size=128)
        gen_old = GenerativeModel.GenerativeModel(dim_b=dim_dataset,
                                                  dim_x=1,
                                                  dim_y=1,
                                                  dim_t=2000)

        if old_weights is None:
            print("Using old network, but no weights were provided")
            sys.exit()
        if P != 3:
            net_old.SpatialNet.load_weights(old_weights[0])
        net_old.DirectCNN.load_weights(old_weights[1])
        net_old.TransientNet.load_weights(old_weights[2])

    # Define the network and load the corresponding weights
    net = PredictiveModel.PredictiveModel(name='DeepBVE_nf_std', dim_b=dim_dataset, freqs=freqs, P=P,
                                          saves_path='./saves', dim_t=dim_t, fil_size=fil_dir, fil_denoise_size=fil_den,
                                          dim_encoding=dim_encoding, fil_encoder=fil_auto)

    if P != 3:
        net.SpatialNet.load_weights(weights_path[0])
    net.decoder.load_weights(weights_path[1])
    net.encoder.load_weights(weights_path[2])
    net.predv_encoding.load_weights(weights_path[3])
    net.DirectCNN.load_weights(weights_path[4])

    avg_s = []
    avg_sm = []
    avg_b = []
    avg_mae = []
    avg_emd = []
    avg_pdf = []
    avg_cdf = []
    fracs = []
    all_names = []
    avg_s_old = []
    avg_sm_old = []
    avg_b_old = []
    avg_mae_old = []
    avg_emd_old = []
    avg_pdf_old = []
    avg_cdf_old = []
    fracs_old = []
    all_names_old = []

    #  Direct metrics
    dir_amp_b = []
    dir_amp_p = []
    dir_pos_b = []
    dir_pos_p = []
    dir_amp_b_old = []
    dir_amp_p_old = []
    dir_pos_b_old = []
    dir_pos_p_old = []

    for xx, name in enumerate(loadnames):
        st1 = time.time()
        print(os.path.basename(name))
        # maxv=8
        # query_name = "3walls_1_TOF_119_rec.h5"
        # if os.path.basename(name) !=  query_name:
        #    continue
        # if os.path.basename(name)[:2] == "ow":
        #    maxv+=1
        #    continue
        # if xx < maxv:
        #    continue
        with h5py.File(name, "r") as f:  # 3,10,13,    8 has high values
            for key in f.keys():
                tr = f[key][:]
        dim_t_data = tr.shape[-1]
        if dim_t_data != dim_t:
            tr1 = np.zeros((tr.shape[0], tr.shape[1], dim_t), dtype=np.float32)
            tr1[..., :dim_t_data] = tr
            tr = tr1

        phi = np.transpose(utils.phi(freqs, dim_t))
        tr = np.swapaxes(tr, 0, 1)
        print("Mean magnitude", np.nanmean(np.nansum(tr, axis=-1)))
        dep = np.nanargmax(tr, axis=-1) / 2000 * 5
        plt.figure()
        plt.imshow(dep, cmap="jet")

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
                #x_d[j, k, np.nanargmax(x_d[j, k, :]) + 5:] = 0

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

        fl_denoise = not (
                    net.P == net.out_win)  # If the two values are different, then the denoising network has been used
        # Make prediction
        v_input = np.pad(v_in, pad_width=[[0, 0], [s_pad, s_pad], [s_pad, s_pad], [0, 0]], mode="edge")
        if fl_denoise:
            v_in_v = net.SpatialNet(v_input)
        else:
            v_in_v = v_input

        if fl_newhidden:
            [v_out_g, v_out_d, v_free] = net.DirectCNN(v_in_v)
            if v_out_g.shape[-1] != v_out_d.shape[-1]:
                v_out_g = v_in - v_out_d
            phid = np.arctan2(v_out_d[:, :, :, ff:], v_out_d[:, :, :, :ff])
            phid = np.where(np.isnan(phid), 0.,
                            phid)  # Needed for the first epochs to correct the output in case of a 0/0
            phid = phid % (2 * math.pi)
            v_out_d = np.squeeze(v_out_d)
            v_out_g = np.squeeze(v_out_g)
            val = np.nanmean(np.abs(v_out_g - np.squeeze(v_g_gt))) / 2
            v_free = np.squeeze(v_free)
            v_out_encoding = np.concatenate((v_free, v_out_g), axis=-1)
        else:
            [v_out_g, v_out_d, phid] = net.DirectCNN(v_in_v)
            v_out_d = np.squeeze(v_out_d)
            v_out_g = np.squeeze(v_out_g)
            v_out_encoding = np.concatenate((v_out_d, v_out_g), axis=-1)  # If the middle network is not used
        v_out_encoding = np.reshape(v_out_encoding, (-1, dim_encoding))  # If the middle network is not used
        x_out_g = np.squeeze(net.decoder(v_out_encoding))
        x_out_g = np.reshape(x_out_g, (dim_x, dim_y, -1))

        x_g = tr - x_d

        shape_diff = x_g.shape[-1] - x_out_g.shape[-1]
        missing = np.zeros((x_out_g.shape[0], x_out_g.shape[1], shape_diff), dtype=np.float32)
        x_out_g = np.concatenate((missing, x_out_g), axis=-1)

        # Older network
        if fl_test_old:
            s_pad_old = int((Pold - 1) / 2)
            v_input_old = np.pad(v_in, pad_width=[[0, 0], [s_pad_old, s_pad_old], [s_pad_old, s_pad_old], [0, 0]],
                                 mode="edge")
            if Pold != 3:
                v_in_v_old = net_old.SpatialNet(v_input_old)
            else:
                v_in_v_old = v_input_old

            [v_out_g_old, v_out_d_old, phid_old] = net_old.DirectCNN(v_in_v_old)
            z_old = net_old.TransientNet([v_out_g_old, v_out_d_old])
            # original network predicts the cumulative. We need to get back to the original space
            xg_old = gen_old(z_old)
            xg_old_cum = np.squeeze(xg_old)
            xg_old = xg_old_cum[..., 1:] - xg_old_cum[..., :-1]
            all_z = np.zeros((xg_old_cum.shape[0], xg_old_cum.shape[1], 1), dtype=np.float32)
            xg_old = np.concatenate((all_z, xg_old), axis=-1)

            print("TEST ON OLD NETWORK")
            err_so, err_som, err_bo, err_maeo, err_emdo, mean_pdfo, mean_cdfo, frac_starto, dir_amp_baseo, dir_amp_predo, dir_pos_baseo, dir_pos_predo = metrics_computation_transient_images(
                tr, x_g, xg_old, v_out_d_old, np.squeeze(v_in), freqs=freqs)
            if not np.isnan(err_so):
                avg_s_old.append(err_so)
            if not np.isnan(err_som):
                avg_sm_old.append(err_som)
            if not np.isnan(err_so):
                avg_b_old.append(err_bo)
            if not np.isnan(err_maeo):
                avg_mae_old.append(err_maeo)
            if not np.isnan(err_emdo):
                avg_emd_old.append(err_emdo)
            if not np.isnan(mean_pdfo):
                avg_pdf_old.append(mean_pdfo)
            if not np.isnan(mean_cdfo):
                avg_cdf_old.append(mean_cdfo)
            if not np.isnan(err_so):
                fracs_old.append(frac_starto)
            if not np.isnan(err_so):
                all_names_old.append(os.path.basename(name))

            dir_amp_b_old.append(dir_amp_baseo)
            dir_amp_p_old.append(dir_amp_predo)
            dir_pos_b_old.append(dir_pos_baseo)
            dir_pos_p_old.append(dir_pos_predo)

        v_in = tf.squeeze(v_in)

        err_s, err_sm, err_b, err_mae, err_emd, mean_pdf, mean_cdf, frac_start, dir_amp_base, dir_amp_pred, dir_pos_base, dir_pos_pred = metrics_computation_transient_images(
            tr, x_g, x_out_g, v_out_d, v_in, freqs=freqs)

        # AUTOENCODER PART
        fl_auto = False
        if fl_auto:
            v_enc = net.encoder(x_g.reshape(-1, 2000))
            if fl_newhidden:
                v_g_gt = np.reshape(v_g_gt, (-1, 2 * ff))
                v_enc = np.squeeze(v_enc)
                print(v_enc.shape)
                print(v_g_gt.shape)
                v_enc = np.concatenate((v_enc, v_g_gt), axis=-1)
            x_dec = net.decoder(np.squeeze(v_enc))
            x_dec = np.reshape(x_dec, (dim_x, dim_y, -1))
            x_dec = np.concatenate((missing, x_dec), axis=-1)
            err_s, err_sm, err_b, err_mae, err_emd, mean_pdf, mean_cdf, frac_start, dir_amp_base, dir_amp_pred, dir_pos_base, dir_pos_pred = metrics_computation_transient_images(
                tr, x_g, x_dec, v_out_d, v_in, freqs=freqs)

        if not np.isnan(err_s):
            avg_s.append(err_s)
        if not np.isnan(err_sm):
            avg_sm.append(err_sm)
        if not np.isnan(err_s):
            avg_b.append(err_b)
        if not np.isnan(err_mae):
            avg_mae.append(err_mae)
        if not np.isnan(err_emd):
            avg_emd.append(err_emd)
        if not np.isnan(mean_pdf):
            avg_pdf.append(mean_pdf)
        if not np.isnan(mean_cdf):
            avg_cdf.append(mean_cdf)
        if not np.isnan(err_s):
            fracs.append(frac_start)
        if not np.isnan(err_s):
            all_names.append(os.path.basename(name))
        dir_amp_b.append(dir_amp_base)
        dir_amp_p.append(dir_amp_pred)
        dir_pos_b.append(dir_pos_base)
        dir_pos_p.append(dir_pos_pred)
        continue

    fracs_old = np.asarray(fracs_old)
    fracs_old /= np.nansum(fracs_old)
    mean_s_old = np.nansum(100 * np.asarray(avg_s_old) * np.asarray(fracs_old))
    mean_sm_old = np.nansum(100 * np.asarray(avg_sm_old) * np.asarray(fracs_old))
    mean_b_old = np.nansum(100 * np.asarray(avg_b_old) * np.asarray(fracs_old))
    print("OLD NETWORK")
    print("   ")
    print("DIRECT COMPONENT METRICS")
    print("Baseline error on the direct position (60 MHz)", 100 * np.round(np.nanmean(dir_pos_b_old), 4), " cm")
    print("Network error on the direct position ", 100 * np.round(np.nanmean(dir_pos_p_old), 4), " cm")
    print("Baseline error on the direct amplitude ", np.nanmean(dir_amp_b_old))
    print("Network error on the direct ampltiude ", np.nanmean(dir_amp_p_old))
    print("   ")
    print("GLOBAL COMPONENT METRICS")
    print("Error on the start is of ", mean_s_old, " cm")
    print("Error on the start (based on max) is of ", mean_sm_old, " cm")
    print("Error on the baricenter is of ", mean_b_old, " cm")
    print("MAE error is of ", np.nanmean(avg_mae_old))
    print("EMD error is of ", np.nanmean(avg_emd_old))
    print("The mean PDF is of ", np.nanmean(avg_pdf_old))
    print("The mean CDF is of ", np.nanmean(avg_cdf_old))
    print("start errors are ", np.round(avg_s_old, 2) * 100)
    print("fractions are ", fracs_old)
    print("names are ", all_names_old)

    plt.figure()
    plt.title("OLD histogram of start error")
    plt.hist(avg_s, bins=10)
    plt.figure()
    plt.title("OLD histogram of baricenter error")
    plt.hist(avg_b, bins=10)
    plt.figure()
    plt.title("OLD histogram of MAE")
    plt.hist(avg_mae, bins=10)
    plt.figure()
    plt.title("OLD histogram of EMD")
    plt.hist(avg_emd, bins=10)
    plt.figure()
    plt.title("OLD histogram of average pdf")
    plt.hist(avg_pdf, bins=10)
    plt.figure()
    plt.title("OLD histogram of average cdf")
    fracs = np.asarray(fracs)
    fracs /= np.nansum(fracs)
    mean_s = np.nansum(100 * np.asarray(avg_s) * np.asarray(fracs))
    mean_sm = np.nansum(100 * np.asarray(avg_sm) * np.asarray(fracs))
    mean_b = np.nansum(100 * np.asarray(avg_b) * np.asarray(fracs))
    print("NEW NETWORK")
    print("   ")
    print("DIRECT COMPONENT METRICS")
    print("Baseline error on the direct position (60 MHz)", 100 * np.round(np.nanmean(dir_pos_b), 4), " cm")
    print("Network error on the direct position ", 100 * np.round(np.nanmean(dir_pos_p), 4), " cm")
    print("Baseline error on the direct amplitude ", np.nanmean(dir_amp_b))
    print("Network error on the direct ampltiude ", np.nanmean(dir_amp_p))

    print("   ")
    print("GLOBAL COMPONENT METRICS")
    print("Error on the start is of ", mean_s, " cm")
    print("Error on the start (based on max) is of ", mean_sm, " cm")
    print("Error on the baricenter is of ", mean_b, " cm")
    print("MAE error is of ", np.nanmean(avg_mae))
    print("EMD error is of ", np.nanmean(avg_emd))
    print("The mean PDF is of ", np.nanmean(avg_pdf))
    print("The mean CDF is of ", np.nanmean(avg_cdf))
    print("start errors are ", np.round(avg_s, 2) * 100)
    print("fractions are ", fracs)
    print("names are ", all_names)

    plt.figure()
    plt.title("histogram of start error")
    plt.hist(avg_s, bins=10)
    plt.figure()
    plt.title("histogram of baricenter error")
    plt.hist(avg_b, bins=10)
    plt.figure()
    plt.title("histogram of MAE")
    plt.hist(avg_mae, bins=10)
    plt.figure()
    plt.title("histogram of EMD")
    plt.hist(avg_emd, bins=10)
    plt.figure()
    plt.title("histogram of average pdf")
    plt.hist(avg_pdf, bins=10)
    plt.figure()
    plt.title("histogram of average cdf")
    plt.hist(avg_cdf, bins=10)
    plt.show()
