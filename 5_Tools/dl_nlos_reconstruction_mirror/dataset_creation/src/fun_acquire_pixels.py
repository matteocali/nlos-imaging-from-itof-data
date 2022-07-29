import numpy as np
import h5py
import sys
from utils import phi as phi_func
from fct_data_fitters import cumul_fitter
from fct_aux import find_ind_peaks_vec
from fct_losses import loss_mae, loss_mse, loss_EMD, loss_wEMD
import time
import os
import utils

"""
The script takes in input a transient dataset, chooses randomly or from a grid a set of pixels of the image and performs the fitting.
Finally, it produces in output a dataset ready to be used for training with the following information:
-Back = Original backscattering vector (scaled by the sum of the global values if the sum is > 0)
-Back_nod = Original backscattering vector without the direct component (scaled by the sum of its values)
-Fit_data = Fitted sum of weibulls giving a good prediction of the original backscattering vector. They are scaled in the same manner as Back
-Fit_Parameters = parameters of the fitted weibull functions (actually, cumulative functions of the weibull)
                    |
                    |___   -b1, b2: starting position of the (possibly) two functions. If the fitted function is just one, b2 = 5 (last possible value)
                           -a1_c, a2_c: estimated amplitude of the two functions (signal the total amount of noise due to the two peaks). a2_c = 0 if the second function does not exist.
                           -lam1, lam2, k1, k2: additional parameters determining the shapes of the two functions. Parameters are set to 0 if the second function does not exist.
-peak_ind = position of the first peak [0,2000]
-peak_val = intensity of the first peak (scaled in the same way as Back)
-v_real = raw measurements corresponding to phi*Back
-v_real_nod = raw measurements corresponding to phi * Back_nod
-v_real_d = raw measurements corresponding to phi * (Back - Back_nod)
-phi matrix used for the operations.
"""


def acquire_pixels(images, num_pixels=2000, max_img=1000, f_ran=1, s=3, flag_ow=True, flag_fit=True,
                   fl_normalize_transient=True, freqs=np.array((20e06, 50e06, 60e06), dtype=np.float32)):
    st = time.time()  # start time
    pad_s = int((s - 1) / 2)  # padding size

    phi = phi_func(freqs)  # compute the phi matrix (iToF data)
    nf = phi.shape[0]  # number of frequencies
    step_t = 0.0025  # time step
    max_t = 5  # maximum time
    wsize = 30  # window size for the average
    depth = np.arange(0, max_t, step_t)  # depth vector

    # Load the first image to get the size of the images
    num_images = len(images)
    with h5py.File(images[0], "r") as h:
        temp = h["data"][:]
    [dim_x, dim_y, dim_t] = temp.shape

    # Given the image size, build a mask to choose which pixels to get the ground truth from.
    # This can be done either with a grid or randomly

    # 1) GRID
    if not (f_ran):
        mask = np.zeros((dim_x - 2*pad_s, dim_y - 2*pad_s), dtype=np.int32)
        num_x = np.sqrt(num_pixels * dim_x / dim_y)
        num_y = np.sqrt(num_pixels * dim_y / dim_x)
        st_x = int(dim_x / num_x)
        st_y = int(dim_y / num_y)
        for i in range(0, dim_x, st_x):
            for j in range(0, dim_y, st_y):
                mask[i + pad_s, j + pad_s] = 1
        num_pixels = np.sum(mask)

    v_real = np.zeros((num_images, num_pixels, s, s, nf), dtype=np.float32)
    gt_depth_real = np.zeros((num_images, num_pixels, s, s), dtype=np.float32)
    gt_alpha_real = np.zeros((num_images, num_pixels, s, s), dtype=np.float32)
    v_real_no_d = np.zeros((num_images, num_pixels, s, s, nf), dtype=np.float32)
    v_real_d = np.zeros((num_images, num_pixels, s, s, nf), dtype=np.float32)
    peak_val = np.zeros((num_images, num_pixels, s, s), dtype=np.float32)
    peak_ind = np.zeros((num_images, num_pixels, s, s), dtype=np.int32)
    Back = np.zeros((num_images, num_pixels, dim_t), dtype=np.float32)
    Back_nod = np.zeros((num_images, num_pixels, dim_t), dtype=np.float32)
    if flag_fit:
        Fit_Data = np.zeros((num_images, num_pixels, dim_t), dtype=np.float32)
    Fit_Parameters = np.zeros((num_images, num_pixels, s, s, 8), dtype=np.float32)

    count = 0

    for k, image in enumerate(images):
        if f_ran:  # if the pixels are chosen randomly create the mask, different each time
            mask1 = np.zeros((dim_x, dim_y), dtype=np.int32)
            mask = np.zeros((dim_x - 2*pad_s, dim_y - 2*pad_s), dtype=np.int32)
            mask = mask.reshape(-1)
            mask[:num_pixels] = 1
            np.random.shuffle(mask)
            mask = mask.reshape(dim_x - 2*pad_s, dim_y - 2*pad_s)
            mask1[pad_s:-pad_s, pad_s:-pad_s] = mask
            mask = mask1

        start = time.time()
        # Load the transient data
        with h5py.File(image, "r") as h:
            temp = h["data"][:]  # load the transient data
            gt_depth = h["depth_map"][:]  # load the ground truth depth data
            gt_alpha = h["alpha_map"][:]  # load the ground truth alpha map data

        # Compute the v_values for the patch around each pixels
        ind = np.where(mask > 0)

        # 1) COMPUTATION OF v_real, v_real_no_d AND v_real_d
        for i in range(ind[0].shape[0]):
            tran_patch = temp[ind[0][i] - pad_s:ind[0][i] + pad_s + 1, ind[1][i] - pad_s:ind[1][i] + pad_s + 1]
            tran_patch = np.reshape(tran_patch, (s * s, dim_t))

            # Normalize the entire vector
            if fl_normalize_transient:
                norm_factor = np.sum(tran_patch, axis=-1, keepdims=True)
                tran_patch /= norm_factor

            # Fix first peak before the computation
            ind_maxima = np.argmax(tran_patch, axis=-1)
            val_maxima = np.zeros(ind_maxima.shape, dtype=np.float32)
            ind_end_direct = np.zeros(ind_maxima.shape, dtype=np.int32)

            for j in range(tran_patch.shape[0]):
                zeros_pos = np.where(tran_patch[j] == 0)[0]  # find the index of the zeros
                ind_end_direct[j] = zeros_pos[np.where(zeros_pos > ind_maxima[j])][0]  # find the index of the zeros after the first peak
            for j in range(ind_maxima.shape[0]):
                val_maxima[j] = np.sum(tran_patch[j, :ind_end_direct[j]])  # compute the value of the first peak considering the sum of the values before the global
            for j in range(tran_patch.shape[0]):
                tran_patch[j, :ind_end_direct[j]] = 0  # set the values before the global to zero
            for j in range(tran_patch.shape[0]):
                tran_patch[j, ind_maxima[j]] = val_maxima[j]  # set the value of the first peak to the value computed before
            # computation with the direct component
            v = np.matmul(tran_patch, np.transpose(phi))
            v = np.reshape(v, (s, s, phi.shape[0]))
            v_real[count, i, :, :, :] = v

            # remove first peak
            for j in range(tran_patch.shape[0]):
                tran_patch[j, :ind_end_direct[j]] = 0

            # computation without direct component
            v = np.matmul(tran_patch, np.transpose(phi))
            v = np.reshape(v, (s, s, phi.shape[0]))
            v_real_no_d[count, i, :, :, :] = v
            # computation with direct component
            for j in range(tran_patch.shape[0]):
                tran_patch[j, ind_maxima[j]] = val_maxima[j]
            # computation without global component
            for j in range(tran_patch.shape[0]):
                tran_patch[j, ind_end_direct[j]:] = 0
            v = np.matmul(tran_patch, np.transpose(phi))
            v = np.reshape(v, (s, s, phi.shape[0]))
            v_real_d[count, i, :, :, :] = v

        back = np.zeros((num_pixels, s, s, dim_t), dtype=np.float32)  # backscattering vector [num_pixels, s, s, dim_t]
        gt_depth_patches = np.zeros((num_pixels, s, s), dtype=np.float32)  # depthground truth vector [num_pixels, s, s]
        gt_alpha_patches = np.zeros((num_pixels, s, s), dtype=np.float32)  # alpha ground truth vector [num_pixels, s, s]
        for i in range(s):
            for j in range(s):
                back[:, i, j, :] = temp[ind[0][:] + i - pad_s, ind[1][:] + j - pad_s]
                gt_depth_patches[:, i, j] = gt_depth[ind[0][:] + i - pad_s, ind[1][:] + j - pad_s]
                gt_alpha_patches[:, i, j] = gt_alpha[ind[0][:] + i - pad_s, ind[1][:] + j - pad_s]

        # fix first peak before continuing
        # The first peak can be scattered over more than one bin, so we just sum the bins and put them all in the same one
        ind_maxima = np.argmax(back, axis=-1)
        val_maxima = np.zeros(ind_maxima.shape, dtype=np.float32)
        ind_end_direct = np.zeros(ind_maxima.shape, dtype=np.int32)

        for j in range(back.shape[0]):
            for l in range(s):
                for m in range(s):
                    zeros_pos = np.where(back[j, l, m, :] == 0)[0]  # find the index of the zeros
                    ind_end_direct[j, l, m] = zeros_pos[np.where(zeros_pos > ind_maxima[j, l, m])][0]  # find the index of the zeros after the first peak
        for j in range(ind_maxima.shape[0]):
            for l in range(s):
                for m in range(s):
                    val_maxima[j, l, m] = np.sum(back[j, l, m, :ind_end_direct[j, l, m]])
        for j in range(back.shape[0]):
            for l in range(s):
                for m in range(s):
                    back[j, l, m, :ind_end_direct[j, l, m]] = 0
        for j in range(back.shape[0]):
            for l in range(s):
                for m in range(s):
                    back[j, l, m, ind_maxima[j, l, m]] = val_maxima[j, l, m]
        back_nod = np.copy(back)

        if fl_normalize_transient:
            # Normalize everything again for the sum of elements 
            norm_values = np.sum(back, axis=-1, keepdims=True)
            back /= norm_values
            back_nod /= norm_values

        # find index of maximum and save its value for later
        indmax = np.argmax(back, axis=-1)
        max_val = np.max(back, axis=-1)
        peak_ind[count, ...] = indmax
        peak_val[count, ...] = max_val

        for i in range(back.shape[0]):
            for l in range(s):
                for m in range(s):
                    back_nod[i, l, m, :indmax[i, l, m] + 5] = 0
        cumv = np.cumsum(back_nod, axis=-1)
        scal_fact = cumv[..., -1]
        scal_fact = scal_fact[..., np.newaxis]

        # Find the starting point of the noise peaks
        der2 = np.zeros((num_pixels, s, s, dim_t), dtype=np.float32)
        indpeak1 = np.zeros((num_pixels, s, s), dtype=np.int32)
        indpeak2 = np.zeros((num_pixels, s, s), dtype=np.int32)
        for l in range(s):
            for m in range(s):
                der2s, indpeak1s, indpeak2s = find_ind_peaks_vec(cumv[:, l, m, :], wsize)
                der2[:, l, m, :] = der2s
                indpeak1[:, l, m] = indpeak1s
                indpeak2[:, l, m] = indpeak2s

        # starting parameters for curve fitting -> si puo' togliere
        b1_ = np.zeros((indpeak1.shape), dtype=np.float32)
        for l in range(s):
            for m in range(s):
                b1_[:, l, m] = depth[indpeak1[:, l, m]]
        a1_c = np.zeros((back_nod.shape[0], s, s), dtype=np.float32)
        b2_ = np.zeros((back_nod.shape[0], s, s), dtype=np.float32)

        for i in range(back_nod.shape[0]):
            for l in range(s):
                for m in range(s):
                    if indpeak2[i, l, m] == 2000:
                        b2_[i, l, m] = 0
                        a1_c[i, l, m] = np.sum(back_nod[i, l, m, :])
                    else:
                        b2_[i, l, m] = depth[indpeak2[i, l, m]]
                        ind = int(b2_[i, l, m] * 400)
                        a1_c[i, l, m] = np.sum(back_nod[i, l, m, :ind])

        if flag_fit:
            fit_data_cum = np.zeros((back.shape), dtype=np.float32)
            fit_par = np.zeros((a1_c.shape[0], s, s, 6), dtype=np.float32)
        mask_cum = np.zeros((back.shape[0]), dtype=np.int32)

        if flag_fit:
            for i in range(a1_c.shape[0]):
                for l in range(s):
                    for m in range(s):
                        # FIT DATA USING CUMULATIVE INFORMATION
                        try:
                            if not scal_fact[i, l, m] == 0:
                                fit_cum_s, fit_data_cum_s, params = cumul_fitter(a1_c[i, l, m], b1_[i, l, m],
                                                                                 b2_[i, l, m], depth,
                                                                                 cumv[i, l, m, :] / scal_fact[i, l, m])
                                fit_data_cum_s *= step_t * scal_fact[i, l, m]
                                fit_data_cum[i, l, m, :] = fit_data_cum_s
                                params[-1] = params[-1] * scal_fact[i, l, m]
                                params[-2] = params[-2] * scal_fact[i, l, m]
                                fit_par[i, l, m, :] = params
                        except RuntimeError:
                            mask_cum[i] = 1

        Back[count, ...] = back[:, pad_s, pad_s, :]
        gt_depth_real[count, ...] = gt_depth_patches
        gt_alpha_real[count, ...] = gt_alpha_patches
        Back_nod[count, ...] = back_nod[:, pad_s, pad_s, :]
        if flag_fit:
            Fit_Data[count, ...] = fit_data_cum
            Fit_Parameters[count, ..., :6] = fit_par
            Fit_Parameters[count, ..., 6] = b1_
        b2_[b2_ == 0] = 5
        Fit_Parameters[count, ..., 7] = b2_

        ending = time.time()
        minutes, seconds = divmod(ending - start, 60)
        hours, minutes = divmod(minutes, 60)
        print(" Total time for an image is %d:%02d:%02d" % (hours, minutes, seconds))
        count += 1
        if count == max_img:
            break
        if count > max_img:
            print("WRONG COUNT")
            sys.exit()
    max_ind = count

    Back = Back[:max_ind, ...]
    gt_depth_real = gt_depth_real[:max_ind, ...]
    gt_alpha_real = gt_alpha_real[:max_ind, ...]
    Back_nod = Back_nod[:max_ind, ...]
    if flag_fit:
        Fit_Data = Fit_Data[:max_ind, ...]
    peak_ind = peak_ind[:max_ind, ...]
    peak_val = peak_val[:max_ind, ...]
    Fit_Parameters = Fit_Parameters[:max_ind, ...]
    v_real = v_real[:max_ind, ...]
    v_real_no_d = v_real_no_d[:max_ind, ...]
    v_real_d = v_real_d[:max_ind, ...]

    Back = np.reshape(Back, (Back.shape[0] * Back.shape[1], dim_t))
    gt_depth_real = np.reshape(gt_depth_real, (gt_depth_real.shape[0] * gt_depth_real.shape[1], s, s))
    gt_alpha_real = np.reshape(gt_alpha_real, (gt_alpha_real.shape[0] * gt_alpha_real.shape[1], s, s))
    Back_nod = np.reshape(Back_nod, (Back_nod.shape[0] * Back_nod.shape[1], dim_t))
    if flag_fit:
        Fit_Data = np.reshape(Fit_Data, (Fit_Data.shape[0] * Fit_Data.shape[1], s, s, dim_t))
    else:
        Fit_Data = None
    peak_ind = np.reshape(peak_ind, (peak_ind.shape[0] * peak_ind.shape[1], s, s))
    peak_val = np.reshape(peak_val, (peak_val.shape[0] * peak_val.shape[1], s, s))
    Fit_Parameters = np.reshape(Fit_Parameters, (Fit_Parameters.shape[0] * Fit_Parameters.shape[1], s, s, Fit_Parameters.shape[-1]))
    v_real = np.reshape(v_real, (v_real.shape[0] * v_real.shape[1], v_real.shape[2], v_real.shape[3], v_real.shape[4]))
    v_real_no_d = np.reshape(v_real_no_d, (
    v_real_no_d.shape[0] * v_real_no_d.shape[1], v_real_no_d.shape[2], v_real_no_d.shape[3], v_real_no_d.shape[4]))
    v_real_d = np.reshape(v_real_d, (
    v_real_d.shape[0] * v_real_d.shape[1], v_real_d.shape[2], v_real_d.shape[3], v_real_d.shape[4]))

    # Random shuffling of all arrays
    ran_ind = np.random.permutation(v_real.shape[0])
    Back = Back[ran_ind, ...]
    gt_depth_real = gt_depth_real[ran_ind, ...]
    gt_alpha_real = gt_alpha_real[ran_ind, ...]
    Back_nod = Back_nod[ran_ind, ...]
    if flag_fit:
        Fit_Data = Fit_Data[ran_ind, ...]
        Fit_Parameters = Fit_Parameters[ran_ind, ...]
    peak_ind = peak_ind[ran_ind, ...]
    peak_val = peak_val[ran_ind, ...]
    v_real = v_real[ran_ind, ...]
    v_real_no_d = v_real_no_d[ran_ind, ...]
    v_real_d = v_real_d[ran_ind, ...]

    fi = time.time()
    minutes, seconds = divmod(fi - st, 60)
    hours, minutes = divmod(minutes, 60)
    print(" The overall computation time for the dataset is %d:%02d:%02d" % (hours, minutes, seconds))
    return Back, Back_nod, gt_depth_real, gt_alpha_real, Fit_Parameters, peak_ind, peak_val, v_real, v_real_no_d, v_real_d, phi, Fit_Data
