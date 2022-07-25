import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append("../utils/")
import utils
import depth_estimation
import math
import time

"""
This function computes various metrics monitoring the accuracy of a model for what concerns its prediction of direct and global components 

"""


def metrics_computation_transient_images(trans, trans_nod, pred_trans_nod, pred_vd, v,
                                         freqs=np.array((20e06, 50e06, 60e06), dtype=np.float32), fl_barragan=False,
                                         mask=None, fl_matte=False):
    e1 = time.time()
    dim_x = trans.shape[0]
    dim_y = trans.shape[1]
    dim_t = trans.shape[2]
    phi = utils.phi(freqs, dim_t)
    nf = freqs.shape[0]
    max_d = 0.5 * utils.max_t(dim_t) * utils.c()
    # If we are using the simulated data with matte material with matching synthetic and real inputs
    if fl_matte:
        max_d /= 2
    if fl_barragan:
        phi = utils.phi(freqs, dim_t * 2)
        max_d *= 2
        phi = phi[:, ::2]

    # 0) Clean the vector
    ind_pmax = np.nanargmax(pred_trans_nod, axis=-1)
    pred_trans_nod = np.where(pred_trans_nod < 0, 0, pred_trans_nod)
    mean_shift = pred_trans_nod[..., 0]  # First element has to be 0
    mean_shift = mean_shift[..., np.newaxis]
    pred_trans_nod = pred_trans_nod - mean_shift
    for i in range(ind_pmax.shape[0]):
        for j in range(ind_pmax.shape[1]):
            if ind_pmax[i, j] - 100 > 0:
                temp_vec = pred_trans_nod[i, j, ind_pmax[i, j] - 100:ind_pmax[i, j]]
                min_ind = np.nanargmin(temp_vec)
                # pred_trans_nod[i,j,:ind_pmax[i,j]-min_ind] = 0

    e2 = time.time()
    # 1) Metrics computation for the direct component
    print("1) Metrics regarding the direct component")

    trans_d = trans - trans_nod  # Compute the transient composed of the direct component alone
    gt_ind = np.nanargmax(trans, axis=-1)  # Find the position of the maximum
    gt_pos = gt_ind / dim_t * max_d  # Convert in meters

    # Find the value of the magnitude by summing all bins of the direct
    gt_mag = np.nansum(trans_d, axis=-1)

    # Compute the predicted estimation as an average between the predictions at the various frequencies
    # N.B. the minimum operation instead improves the performance of the baseline and worsens the one of the network
    pred_base = np.zeros((nf, gt_pos.shape[0], gt_pos.shape[1]), dtype=np.float32)
    pred_pos = np.zeros((nf, gt_pos.shape[0], gt_pos.shape[1]), dtype=np.float32)
    for i in range(nf):
        temp_v = np.stack((v[..., i], v[..., i + nf]), axis=-1)
        temp_vd = np.stack((pred_vd[..., i], pred_vd[..., i + nf]), axis=-1)
        pred_base[i] = depth_estimation.freq1_depth_estimation(temp_v, freqs[i])
        pred_pos[i] = depth_estimation.freq1_depth_estimation(temp_vd, freqs[i])
        if i > 0:  # Take care of phase unwrapping for higher frequencies
            amb_range = utils.amb_range(freqs[i])
            pred_base[i] = np.where(pred_base[0] - pred_base[i] > amb_range / 2, pred_base[i] + amb_range, pred_base[i])
            pred_pos[i] = np.where(pred_pos[0] - pred_pos[i] > amb_range / 2, pred_pos[i] + amb_range, pred_pos[i])

    e3 = time.time()
    base_mean = np.mean(pred_base, axis=0)
    pred_mean = np.mean(pred_pos, axis=0)
    # pred_mean =  np.nanmin(pred_pos,axis=0)

    # MAE_depth_base = np.mean(np.abs(gt_pos-base_mean))
    if mask is None:
        MAE_depth_base = np.mean(np.abs(gt_pos - pred_base[-1]))  # Compute against prediction at 60 MHz
        MAE_depth_direct = np.mean(np.abs(gt_pos - pred_mean))
    else:
        MAE_depth_base = np.nansum(mask * np.abs(gt_pos - pred_base[-1])) / np.nansum(
            mask)  # Compute against prediction at 60 MHz
        MAE_depth_direct = np.nansum(mask * np.abs(gt_pos - pred_mean)) / np.nansum(mask)

    v60 = np.stack((v[..., nf], v[..., -1]), axis=-1)
    depth_mpi = depth_estimation.freq1_depth_estimation(v60, 60e06)
    depth_gt = np.nanargmax(trans, axis=-1) / 2000 * 5
    err_mpi = depth_mpi - depth_gt
    err_mpi = np.where(err_mpi > 1, err_mpi + 2.5, err_mpi)
    # plt.figure()
    # plt.imshow(err_mpi,cmap="jet")
    # cbar = plt.colorbar()
    # cbar.set_label("Depth error [m]")
    # plt.clim(-0.2,0.2)
    ##plt.figure()
    # plt.imshow(pred_mean-gt_pos,cmap="jet")
    # cbar = plt.colorbar()
    # cbar.set_label("Depth error [m]")
    # plt.clim(-0.2,0.2)
    # plt.show()

    for i in range(nf):
        print("freq", freqs[i])
        print("base", np.mean(np.abs(pred_base[i] - gt_pos)) * 100)
        print("net", np.mean(np.abs(pred_pos[i] - gt_pos)) * 100)
    print("DEPTH")
    print("Baseline MAE on the depth of the direct component consists of {} cm".format(MAE_depth_base * 100))
    print("MAE on the depth of the direct component consists of {} cm".format(MAE_depth_direct * 100))

    # Computation of the error on the magnitude of the direct component
    base_ampl = np.sqrt(v[..., :nf] ** 2 + v[..., nf:] ** 2)
    pred_ampl = np.sqrt(pred_vd[..., :nf] ** 2 + pred_vd[..., nf:] ** 2)

    ###
    print(gt_mag.shape)
    print(base_ampl.shape)
    print(pred_ampl.shape)
    for i in range(nf):
        print("freq", freqs[i])
        print("base", np.mean(np.abs(base_ampl[..., i] - gt_mag)))
        print("net", np.mean(np.abs(pred_ampl[..., i] - gt_mag)))

    ###
    base_ampl = np.mean(base_ampl, axis=-1)
    pred_ampl = np.mean(pred_ampl, axis=-1)

    err_base = gt_mag - base_ampl
    err_pred = gt_mag - pred_ampl

    if mask is None:
        MAE_ampl_base = np.mean(np.abs(err_base))
        MAE_ampl_direct = np.mean(np.abs(err_pred))
    else:
        MAE_ampl_base = np.nansum(mask * np.abs(err_base)) / np.nansum(mask)
        MAE_ampl_direct = np.nansum(mask * np.abs(err_pred)) / np.nansum(mask)

    print("MAGNITUDE")
    print("Baseline MAE on the magnitude of the direct component consists of {} ".format(MAE_ampl_base))
    print("MAE on the network prediction of the magnitude of the direct component consists of {} ".format(
        MAE_ampl_direct))

    e4 = time.time()
    ###################################################

    # Metrics computation regarding the global component
    print("2) Metrics regarding the global component")

    # Compute the cumulatives
    trans_nod_cum = np.cumsum(trans_nod, axis=-1)
    pred_trans_nod_cum = np.cumsum(pred_trans_nod, axis=-1)
    # pred_trans_nod_cum = pred_trans_nod

    # MAE between the distributions
    if mask is None:
        MAE = np.mean(np.abs(trans_nod - pred_trans_nod))
        # EMD (MAE between the cumulatives)
        EMD = np.mean(np.abs(trans_nod_cum - pred_trans_nod_cum))
        EMD_sign = np.mean((trans_nod_cum - pred_trans_nod_cum))
    else:
        mask = mask[..., np.newaxis]
        MAE = np.nansum(mask * np.abs(trans_nod - pred_trans_nod)) / np.nansum(mask) / trans_nod.shape[-1]
        # EMD (MAE between the cumulatives)
        EMD = np.nansum(mask * np.abs(trans_nod_cum - pred_trans_nod_cum)) / np.nansum(mask) / trans_nod.shape[-1]
        EMD_sign = np.nansum(mask * (trans_nod_cum - pred_trans_nod_cum)) / np.nansum(mask) / trans_nod.shape[-1]
        mask = np.squeeze(mask)

    print("The mean value of the gt distribution is: ", np.mean(trans_nod))
    print("The MAE between the distribution is: ", MAE)
    print("The mean value of the gt cumulative is: ", np.mean(trans_nod_cum))
    print("The MAE between the cumulatives is: ", EMD)
    print("The ME between the cumulatives is: ", EMD_sign)

    e5 = time.time()

    # Error on the starting point of the cumulative
    gt_start = np.nanargmax(np.where(trans_nod_cum > 0, 1, 0), axis=-1) / dim_t * max_d
    pred_start = np.nanargmax(np.where(pred_trans_nod_cum > 0, 1, 0), axis=-1) / dim_t * max_d
    pred_start_max = np.nanargmax(pred_trans_nod, axis=-1) / dim_t * max_d

    e6 = time.time()
    # Compute the error keeping out the elements with all 0 ground truth
    err_start = np.abs(gt_start - pred_start)
    err_start_max = np.abs(gt_start - pred_start_max)
    err_start_img = gt_start - pred_start
    mask_start = np.where((np.nanmax(trans_nod_cum, axis=-1) > 0) & (np.nanmax(pred_trans_nod_cum, axis=-1) > 0))
    frac_start = mask_start[0].shape[0] / np.nansum(dim_x * dim_y)
    err_start = err_start[mask_start[0], mask_start[1]]
    err_start_max = err_start_max[mask_start[0], mask_start[1]]
    e7 = time.time()

    print("The error on the start of the global consists of ", np.mean(err_start) * 100, "[cm]")
    print("The error on the start of the global (based on max) consists of ", np.mean(err_start_max) * 100, "[cm]")
    print("fraction of good values", frac_start)

    # Error on the baricenter of the predicted global
    pos = np.arange(dim_t)
    pos = pos[:, np.newaxis]
    weights_gt = np.nansum(trans_nod, axis=-1)
    weights_gt = np.where(weights_gt == 0, 1, weights_gt)
    weights_gt = weights_gt[..., np.newaxis]
    bar_pos = (trans_nod / weights_gt) @ pos / dim_t * max_d
    weights = np.nansum(pred_trans_nod, axis=-1)
    weights = np.where(weights == 0, 1, weights)
    weights = weights[..., np.newaxis]
    pred_bar_pos = (
                               pred_trans_nod / weights) @ pos / dim_t * max_d  # Baricenter computation making the global a distribution

    err_bar = np.abs(bar_pos - pred_bar_pos)
    err_bar_img = bar_pos - pred_bar_pos
    err_bar = err_bar[mask_start[0]]

    print("The error on the baricenter of the global is of ", np.mean(err_bar) * 100, "[cm]")
    mae_img = np.mean(np.abs(trans_nod - pred_trans_nod), axis=-1)
    # plt.figure()
    # plt.title("Error between the pdfs")
    # plt.imshow(np.squeeze(mae_img),cmap="jet")
    # plt.colorbar()
    # plt.clim(-0.0002,0.0002)

    emd_img = np.mean(np.abs(trans_nod_cum - pred_trans_nod_cum), axis=-1)
    # plt.figure()
    # plt.plot(phi[0,...])
    # plt.plot(phi[3,...])
    # plt.show()
    # sys.exit()
    v_nod = (trans_nod) @ np.transpose(phi)
    A_nod = np.sqrt(v_nod[..., :3] ** 2 + v_nod[..., 3:] ** 2)
    phi_nod = np.arctan2(v_nod[..., 3:], v_nod[..., :3])
    phi20 = depth_estimation.freq1_depth_estimation(v_nod[..., [0, nf]], 20e06)
    phi50 = depth_estimation.freq1_depth_estimation(v_nod[..., [1, nf + 1]], 50e06)
    if nf > 2:
        phi60 = depth_estimation.freq1_depth_estimation(v_nod[..., [2, nf + 2]], 60e06)
    phid = np.nanargmax(trans, axis=-1) / dim_t * max_d

    # print(np.mean(np.abs(v_nod[...,5]-v_nod[...,2])))
    # print(np.mean(np.abs(v_nod[...,4]-v_nod[...,1])))
    # print(np.mean(np.abs(v_nod[...,3]-v_nod[...,0])))
    emd_img = np.squeeze(emd_img)
    fl_plots = True
    if fl_plots:
        ###############
        dir_gt = np.nanargmax(trans, axis=-1) / 2000 * 5
        glo_gt = np.nanargmax(np.where(trans_nod > 0, 1, 0), axis=-1) / 2000 * 5
        diff_gt = glo_gt - dir_gt
        diff = np.where(diff_gt < 0, 0., diff_gt)

        dir_pos = pred_base[-1]
        glob_dir_pos = np.nanargmax(pred_trans_nod, axis=-1) / 2000 * 5
        diff = glob_dir_pos - dir_pos
        diff = np.where(diff < 0, 0., diff)
        plt.figure()
        plt.title("starting position of the global")
        plt.imshow(glob_dir_pos, cmap="jet")

        plt.figure()
        plt.title("Depth map")
        plt.imshow(dir_pos, cmap="jet")
        plt.colorbar()

        plt.figure()
        plt.plot(diff[:, 30], "r", label="predicted 1")
        # plt.plot(diff[:,60])
        plt.plot(diff[:, 100], "o", label="predicted 2")

        plt.plot(diff_gt[:, 30], "b", label="gt 1")
        # plt.plot(diff_gt[:,60])
        plt.plot(diff_gt[:, 100], "k", label="gt 2")
        plt.legend()

        plt.figure()
        plt.title("global - direct position GT")
        plt.imshow(glo_gt - dir_gt, cmap="jet")
        plt.colorbar()
        plt.clim(0.6, 1.5)

        plt.figure()
        plt.title("global - direct position")
        plt.imshow(glob_dir_pos - dir_pos, cmap="jet")
        plt.colorbar()
        plt.clim(0.6, 1.5)
        plt.show()

        ##############
        ind_sec = int(dim_x / 2)
        plt.figure()
        plt.title("cdf prediction and position")
        plt.plot(np.mean(trans_nod_cum[:, ind_sec, :], axis=-1))
        plt.plot(np.mean(pred_trans_nod_cum[:, ind_sec, :], axis=-1))
        plt.legend(["ground truth", "prediction"])

        plt.figure()
        plt.title("Depth_map")
        plt.imshow(gt_start, cmap="jet")
        cbar = plt.colorbar()
        cbar.set_label("Depth [m]")

        plt.figure()
        plt.title("distribution")
        plt.imshow(np.log(trans_nod[:, ind_sec, :] + 1), cmap="jet")
        plt.colorbar()
        plt.clim(0, 0.0002)

        cval = np.mean(trans_nod_cum)
        plt.figure()
        plt.title("Error between the cdfs")
        plt.imshow(np.squeeze(emd_img), cmap="jet")
        plt.colorbar()
        plt.clim(-cval, cval)

        cval = np.mean(trans_nod)
        cval = np.mean(trans_nod[trans_nod > 0])
        cval = 0.0005
        plt.figure()
        plt.title("Error between the pdfs")
        plt.imshow(np.squeeze(mae_img), cmap="jet")
        plt.colorbar()
        plt.clim(-cval, cval)

        plt.figure()
        plt.title("Prediction error of the start of the distribution")
        plt.imshow(err_start_img, cmap="jet")
        plt.colorbar()
        plt.clim(-0.5, 0.5)

        plt.figure()
        plt.title("Prediction error of the baricenter of the distribution")
        plt.imshow(err_bar_img, cmap="jet")
        plt.colorbar()
        plt.clim(-0.5, 0.5)
        # if np.mean(err_start>0.5):

        posy = [160, 17, 160, 160]
        posx = [121, 5, 128, 130]
        #xsteps = np.arange(0, 5, 5 / 2000)
        xsteps = range(0, trans_nod.shape[2])
        for ix, iy in zip(posx, posy):
            fig = plt.figure()
            plt_start_pos = 0
            plt_end_pos = trans_nod.shape[2]
            #plt_start_pos = np.where(trans_nod[ix, iy] != 0)[0][0] - 10
            #plt_end_pos = np.where(trans_nod[ix, iy] != 0)[0][-1] + 11
            xsteps = range(0, trans_nod[ix, iy, plt_start_pos:plt_end_pos].shape[0])
            plt.plot(xsteps, trans_nod[ix, iy, plt_start_pos:plt_end_pos], "g", label="Original global component")
            plt.plot(xsteps, pred_trans_nod[ix, iy, plt_start_pos:plt_end_pos], "b", label="Estimated global component")
            plt.xticks(range(0, trans_nod[ix, iy, plt_start_pos:plt_end_pos].shape[0], int(trans_nod[ix, iy, plt_start_pos:plt_end_pos].shape[0] / 13)),
                       ["{:.2f}".format(round(value * 0.01 / 3e8 * 1e9, 2)) for value in range(plt_start_pos, plt_end_pos, int(trans_nod[ix, iy, plt_start_pos:plt_end_pos].shape[0] / 13))], rotation=45)
            plt.xlabel(f"Time instants [ns]")
            plt.ylabel(r"Radiance value [$W/(m^{2}·sr)$]")
            plt.grid()
            savename = "../dataset_rec/plots/bar_transient_" + str(ix) + "_" + str(iy) + ".svg"
            fig.tight_layout()
            plt.savefig(savename, dpi=1200)
            plt.show()

    return np.mean(err_start), np.mean(err_start_max), np.mean(err_bar), MAE, EMD, np.mean(trans_nod), np.mean(
        trans_nod_cum), frac_start, MAE_ampl_base, MAE_ampl_direct, MAE_depth_base, MAE_depth_direct
