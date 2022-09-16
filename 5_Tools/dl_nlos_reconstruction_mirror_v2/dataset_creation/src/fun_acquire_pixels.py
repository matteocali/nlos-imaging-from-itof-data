import numpy as np
import h5py
import sys
from utils import phi as phi_func
import time
from tqdm import tqdm

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


def acquire_pixels(images, num_pixels=2000, max_img=1000, s=3, freqs=np.array((20e06, 50e06, 60e06), dtype=np.float32)):
    st = time.time()  # start time

    # Load the first image to get the size of the images
    num_images = len(images)
    with h5py.File(images[0], "r") as h:
        temp = h["data"][:]

    _, _, dim_t = temp.shape                                 # extract the number of temporal bins
    pad_s = int((s - 1) / 2)                                 # padding size
    phi = phi_func(freqs=freqs, dim_t=dim_t, exp_time=0.01)  # compute the phi matrix (iToF data)
    nf = phi.shape[0]                                        # number of frequencies


    # Given the image size, build a mask_out to choose which pixels to get the ground truth from
    v_real = np.zeros((num_images, num_pixels, s, s, nf), dtype=np.float32)     # initialize the output matrix (iToF data)
    gt_depth_real = np.zeros((num_images, num_pixels, s, s), dtype=np.float32)  # initialize the output matrix (depth data)
    gt_alpha_real = np.zeros((num_images, num_pixels, s, s), dtype=np.float32)  # initialize the output matrix (alpha data)

    count = 0  # counter for the number of images processed

    for image in tqdm(images, desc="Loading images", total=num_images):
        # Load the transient data
        with h5py.File(image, "r") as h:
            temp = h["data"][:]                       # load the transient data
            gt_depth = h["depth_map"][:]              # load the ground truth depth data
            gt_alpha = h["alpha_map"][:].astype(int)  # load the ground truth alpha map data

        # Build the matrix that define the sampling (different for each image)
        # sample at random the same number of patch where the ground truth is set to 1 and to 0
        num_sample = min(int(num_pixels / 2), int(np.sum(gt_alpha)))  # number of 0 and one to sample

        mask_out = np.invert(np.copy(gt_alpha).astype(bool)).astype(int)  # create the mask_out as the opposite of the alpha
        border = np.ones(mask_out.shape)  # define a border to avoid creating patches that span outside the image
        border[pad_s:-pad_s, pad_s:-pad_s] = 0  # set the border to 1
        mask_out[np.where(border == 1)] = 0  # set the border to 0 in the mask_out
        ones_pos = list(np.where(mask_out == 1))  # find the positions of the ones in the mask_out
        mask_out[np.where(border == 1)] = 1  # set the border back to 1 in the mask_out
        p = np.random.permutation(len(ones_pos[0]))  # permute the positions of the ones
        ones_pos[0] = ones_pos[0][p][:num_sample]  # select the first num_sample ones
        ones_pos[1] = ones_pos[1][p][:num_sample]  # select the first num_sample ones
        mask_out[tuple(ones_pos)] = 0  # set the ones to zero in the mask_out according to the permutation
        mask_out = np.invert(np.copy(mask_out).astype(bool)).astype(int)  # invert the mask_out

        mask_in = np.copy(gt_alpha)
        zeros_pos = list(np.where(mask_in == 1))
        p = np.random.permutation(len(zeros_pos[0]))  # permute the positions of the ones
        zeros_pos[0] = zeros_pos[0][p][:num_sample]  # select the first num_sample zeros
        zeros_pos[1] = zeros_pos[1][p][:num_sample]  # select the first num_sample zeros
        mask_in[tuple(zeros_pos)] = 0  # set the ones to zero in the mask_in according to the permutation
        mask_in = np.invert(np.copy(mask_in).astype(bool)).astype(int)  # invert the mask_in
        mask = np.multiply(mask_in, mask_out)  # multiply the two masks to get the final mask_in

        diff = int(num_pixels / 2) - num_sample  # difference between the number of pixels to sample and the number of pixels sampled
        if diff > 0:  # if the number of pixels sampled is less than the number of pixels to sample add the same number of zero valued pixels
            indexes = np.where(mask == 0)
            p = np.random.permutation(len(indexes[0]))  # permute the positions of the ones
            indexes[0] = indexes[0][p]      # permute the array
            indexes[0] = indexes[0][:diff]  # select the first diff ones
            indexes[1] = indexes[1][p]      # permute the array
            indexes[1] = indexes[1][:diff]  # select the first diff ones
            mask[tuple(indexes)] = 1  # set the ones to zero in the mask according to the permutation

        ind = list(np.where(mask > 0))  # find the positions of the pixels to sample

        # Computation of v_real (iToF data), gt_depth_real (depth data) and gt_alpha_real (alpha data)
        for i in range(ind[0].shape[0]):
            tran_patch = temp[ind[0][i] - pad_s:ind[0][i] + pad_s + 1, ind[1][i] - pad_s:ind[1][i] + pad_s + 1]
            tran_patch = np.reshape(tran_patch, (s * s, dim_t))

            # computation with the direct component
            v = np.matmul(tran_patch, np.transpose(phi))
            v = np.reshape(v, (s, s, phi.shape[0]))
            v_real[count, i, ...] = v

            # compute depth and alpha gt
            gt_depth_real[count, i, ...] = gt_depth[ind[0][i] - pad_s:ind[0][i] + pad_s + 1, ind[1][i] - pad_s:ind[1][i] + pad_s + 1]
            gt_alpha_real[count, i, ...] = gt_alpha[ind[0][i] - pad_s:ind[0][i] + pad_s + 1, ind[1][i] - pad_s:ind[1][i] + pad_s + 1]

        count += 1  # increment the counter
        if count == max_img:
            break
        if count > max_img:
            print("WRONG COUNT")
            sys.exit()
    max_ind = count  # save the number of images used for the training

    # remove all the exceeding data
    gt_depth_real = gt_depth_real[:max_ind, ...]
    gt_alpha_real = gt_alpha_real[:max_ind, ...]
    v_real = v_real[:max_ind, ...]

    # reshape the data
    gt_depth_real = np.reshape(gt_depth_real, (gt_depth_real.shape[0] * gt_depth_real.shape[1], s, s))
    gt_alpha_real = np.reshape(gt_alpha_real, (gt_alpha_real.shape[0] * gt_alpha_real.shape[1], s, s))
    v_real = np.reshape(v_real, (v_real.shape[0] * v_real.shape[1], s, s, v_real.shape[4]))

    # random shuffling of all arrays
    ran_ind = np.random.permutation(v_real.shape[0])
    gt_depth_real = gt_depth_real[ran_ind, ...]
    gt_alpha_real = gt_alpha_real[ran_ind, ...]
    v_real = v_real[ran_ind, ...]

    fi = time.time()
    minutes, seconds = divmod(fi - st, 60)
    hours, minutes = divmod(minutes, 60)
    print(" The overall computation time for the dataset is %d:%02d:%02d" % (hours, minutes, seconds))
    return v_real, gt_depth_real, gt_alpha_real


def acquire_pixels_test(images, max_img=1000, s=3, freqs=np.array((20e06, 50e06, 60e06), dtype=np.float32)):
    st = time.time()  # start time
    phi = phi_func(freqs)  # compute the phi matrix (iToF data)
    nf = phi.shape[0]  # number of frequencies

    # Load the first image to get the size of the images
    num_images = len(images)
    with h5py.File(images[0], "r") as h:
        temp = h["data"][:]
    [dim_x, dim_y, dim_t] = temp.shape

    v_real = np.zeros((num_images, dim_x, dim_y, nf), dtype=np.float32)
    gt_alpha_real = np.zeros((num_images, dim_x, dim_y), dtype=np.float32)
    gt_depth_real = np.zeros((num_images, dim_x, dim_y), dtype=np.float32)
    names = []

    count = 0

    for image in tqdm(images, desc="Loading images", total=num_images):
        file_name = image.split("/")[-1][:-3]
        names.append(file_name)

        # Load the transient data
        with h5py.File(image, "r") as h:
            temp = h["data"][:]  # load the transient data
            gt_depth = h["depth_map"][:]  # load the ground truth depth data
            gt_alpha = h["alpha_map"][:].astype(int)  # load the ground truth alpha map data

        temp_lin = np.reshape(temp, (dim_x * dim_y, dim_t))

        # computation with the direct component
        v = np.matmul(temp_lin, np.transpose(phi))
        v = np.reshape(v, (dim_x, dim_y, phi.shape[0]))
        v_real[count, ...] = v

        # add the gt depth and alpha map
        gt_depth_real[count, ...] = gt_depth
        gt_alpha_real[count, ...] = gt_alpha

        # print(" Total time for an image is %d:%02d:%02d" % (hours, minutes, seconds))
        count += 1
        if count == max_img:
            break
        if count > max_img:
            print("WRONG COUNT")
            sys.exit()
    max_ind = count

    v_real = v_real[:max_ind, ...]
    gt_depth_real = gt_depth_real[:max_ind, ...]
    gt_alpha_real = gt_alpha_real[:max_ind, ...]
    names = names[:max_ind]

    fi = time.time()
    minutes, seconds = divmod(fi - st, 60)
    hours, minutes = divmod(minutes, 60)
    print(" The overall computation time for the dataset is %d:%02d:%02d" % (hours, minutes, seconds))

    return gt_depth_real, gt_alpha_real, v_real, names