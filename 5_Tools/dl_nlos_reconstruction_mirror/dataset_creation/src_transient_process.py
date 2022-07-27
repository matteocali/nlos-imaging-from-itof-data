import os
from pathlib import Path
import sys
import h5py
import numpy as np
import glob
import csv
import time
sys.path.append("./src")
sys.path.append("../utils")
from fun_acquire_pixels import acquire_pixels
from fct_Aphi_compute import Aphi_compute


"""

Script for preparing training, validation and test set based on transient data.
For more details of the data used see fun_acquire_pixels where most of the computations are performed

The data is saved in the "data" folder under the training directory

Flags and variables:
    -out_dir:                   path to the output directory (by default the "data" folder in the training directory)
    -data_dir:                  location of the transient dataset
    -flag_new_shuffle:          Set to True to get a new random split of the dataset images. Otherwise the csv files in ./data_split/ will be used
    -flag_ow:                   Set to True also uses single walls images. Otherwise only images with MPI will be used 
    -flag_fit:                  Set to True performs the fitting of the Weibull functions on the training data 
    -f_ran:                     Set to True chooses the patches in random positions from all over the image, otherwise a grid will be employed
    -n_patches:                 Number of patches taken from each images
    -s:                         Patch size
    -max_imgs:                  Maximum number of images of each set. Can be used to make a quick run of the process. Otherwise set it to high value (higher than max dataset size)
    -train_slice:               Percentage of images belonging to training dataset expressed as a number between 0 and 1. Test and validation will take each half of what remains
    -fl_get_train:              If set to True we will acquire the training set
    -fl_get_val:                If set to True we will acquire the validation set
    -fl_get_test:               If set to True we will acquire the test set
    -fl_normalize_transient:    Whether to normalize all transient to sum to 1. Can be useful but it is better to do it later right before training
"""


start_time = time.time()

out_dir = "../training/data/"    # output directory
# data_dir = "../dataset_rec/"
data_dir = "Z:/decaligm/mitsuba_renders/nlos_scenes/dataset/depth_map_ground_truth_far/final_dataset/"  # dataset directory
rnd_seed = 2019283  # random seed
np.random.seed(rnd_seed)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

flag_new_shuffle = True  # whether to shuffle again the images and create new training validation and test datasets
flag_ow = True  # whether to use also one wall images or not
flag_fit = False  # whether to fit the data using weibull functions (slows down the code)
f_ran = True  # whether to use random or grid sampling

n_patches = 200  # number of pixels taken from each image
s = 3  # size of each input patch
add_str = f"_s{s}"  # part of the dataset name containing the patch size

max_imgs = 1000  # maximum number of images to be used (if grater than the actual number, all the dataset will be used)
train_slice = 0.6  # percentage of images belonging to training dataset
val_slice = round((1 - train_slice) / 2, 1)  # percentage of images belonging to validation dataset
freqs = np.array((20e06, 50e06, 60e06), dtype=np.float32)  # frequencies used by the iToF sensor
if freqs.shape[0] == 2:
    add_str += "_2freq"

# Choose which datasets you want to build
fl_get_train = True
fl_get_val = True
fl_get_test = True
fl_normalize_transient = False  # whether to normalize the transient information

# Grab all the names of the images of the dataset, shuffle them and save them in a csv file
if not fl_normalize_transient:
    add_str += "_nonorm"

if flag_new_shuffle:
    # Load all the images of the dataset
    path = data_dir + "*.h5"  # Path to the transient dataset
    filenames = glob.glob(path)  # Get all the names of the files

    # Shuffle the images
    num_img = len(filenames)  # Number of images in the dataset
    indexes = np.arange(num_img)  # Create an array with the indexes of the images
    indexes = np.random.permutation(indexes)  # Shuffle the indexes
    indexes = np.array(indexes)  # Convert the indexes to an array
    filenames = np.array(filenames)  # Convert the names to an array
    filenames = filenames[indexes]  # Shuffle the names according to the indexes

    for i, file in enumerate(filenames):
        filenames[i] = file.replace("\\", "/")

    with open("shuffled_images.csv", "w", newline="") as csvfile:  # Save the names of the images in a csv file
        wr = csv.writer(csvfile)
        for filename in filenames:
            wr.writerow([filename])

    images_list = []
    with open("shuffled_images.csv", "r") as csvfile:  # Load the names of the images from the csv file
        wr = csv.reader(csvfile)
        for row in wr:
            images_list.append(row)

    images_list = [item for sublist in images_list for item in sublist]  # Flatten the list of lists

    # Split the images in training, validation and test sets
    num_img = len(images_list)  # Number of images in the dataset
    n_train = int(np.round(train_slice * num_img))  # Number of images in the training set
    n_val = int(np.round(val_slice * num_img))  # Number of images in the validation set
    n_test = num_img - n_train - n_val  # Number of images in the test set
    train_files = images_list[:n_train]  # Names of the images in the training set
    val_files = images_list[n_train:n_train+n_val]  # Names of the images in the validation set
    test_files = images_list[n_train+n_val:]  # Names of the images in the test set
    np.random.shuffle(test_files)  # Shuffle the test set
    np.random.shuffle(val_files)  # Shuffle the validation set
    np.random.shuffle(train_files)  # Shuffle the training set

    with open("./data_split/train_images.csv", "w", newline="") as csvfile:  # Save the names of the images in the train_dataset in a csv file
        wr = csv.writer(csvfile)
        for filename in train_files:
            wr.writerow([os.path.basename(filename)])
    with open("./data_split/val_images.csv", "w", newline="") as csvfile:  # Save the names of the images in the val_dataset in a csv file
        wr = csv.writer(csvfile)
        for filename in val_files:
            wr.writerow([os.path.basename(filename)])
    with open("./data_split/test_images.csv", "w", newline="") as csvfile:  # Save the names of the images in the test_dataset in a csv file
        wr = csv.writer(csvfile)
        for filename in test_files:
            wr.writerow([os.path.basename(filename)])

# Load the names of the images from the csv file, considering the split in training, validation and test sets
train_files = []
val_files = []
test_files = []

with open("./data_split/train_images.csv", "r") as csvfile:
    wr = csv.reader(csvfile)
    for row in wr:
        train_files.append(os.path.basename(row[0]))
with open("./data_split/val_images.csv", "r") as csvfile:
    wr = csv.reader(csvfile)
    for row in wr:
        val_files.append(os.path.basename(row[0]))
with open("./data_split/test_images.csv", "r") as csvfile:
    wr = csv.reader(csvfile)
    for row in wr:
        test_files.append(os.path.basename(row[0]))

N_train = len(train_files)  # Number of images in the training set
N_val = len(val_files)  # Number of images in the validation set
N_test = len(test_files)  # Number of images in the test set

print(f"Training set size: {N_train}")
print(f"Validation set size: {N_val}")
print(f"Test set size: {N_test}\n")

# TRAINING IMAGES
datasetname = "test"    # Flag used to keep track of the used dataset

if fl_get_train:
    train_files = np.asarray(train_files)
    train_files = [data_dir + file for file in train_files]
    print("Training dataset:")
    Back, Back_nod, gt_depth_real, gt_alpha_real, _, _, _, v_real, v_real_no_d, v_real_d, _, Back_fit = acquire_pixels(train_files, n_patches, max_imgs, f_ran, s, flag_ow, flag_fit, fl_normalize_transient, freqs)
    num_elem = Back.shape[0]

    A_in, phi_in, A_g, phi_g, A_d, phi_d = Aphi_compute(v_real, v_real_no_d, v_real_d)

    file_train = out_dir + "train_n" + str(num_elem) + add_str + ".h5"
    with h5py.File(file_train, 'w') as f:
        f.create_dataset("name", data=datasetname)
        f.create_dataset("transient", data=Back, compression="gzip")
        f.create_dataset("transient_global", data=Back_nod, compression="gzip")
        f.create_dataset("gt_depth", data=gt_depth_real, compression="gzip")
        f.create_dataset("gt_alpha", data=gt_alpha_real, compression="gzip")
        f.create_dataset("raw_itof", data=v_real)
        f.create_dataset("global_itof", data=v_real_no_d)
        f.create_dataset("direct_itof", data=v_real_d)
        f.create_dataset("amplitude_raw", data=A_in)
        f.create_dataset("phase_raw", data=phi_in)
        f.create_dataset("amplitude_direct", data=A_d)
        f.create_dataset("phase_direct", data=phi_d)
        f.create_dataset("amplitude_global", data=A_g)
        f.create_dataset("phase_global", data=phi_g)
        f.create_dataset("freqs", data=freqs)
        if flag_fit:
            f.create_dataset("transient_fit", data=Back_fit)

# VALIDATION IMAGES
if fl_get_val:
    val_files = np.asarray(val_files)
    val_files = [data_dir + fil for fil in val_files]
    print("\nValidation dataset:")
    Back, Back_nod, gt_depth_real, gt_alpha_real, _, _, _, v_real, v_real_no_d, v_real_d, _, Back_fit = acquire_pixels(val_files, n_patches, max_imgs, f_ran, s, flag_ow, flag_fit, fl_normalize_transient, freqs)
    num_elem = Back.shape[0]

    A_in, phi_in, A_g, phi_g, A_d, phi_d = Aphi_compute(v_real, v_real_no_d, v_real_d)
    file_val = out_dir + "val_n" + str(num_elem) + add_str + ".h5"
    with h5py.File(file_val, 'w') as f:
        f.create_dataset("name", data=datasetname)
        f.create_dataset("transient", data=Back, compression="gzip")
        f.create_dataset("transient_global", data=Back_nod, compression="gzip")
        f.create_dataset("gt_depth", data=gt_depth_real, compression="gzip")
        f.create_dataset("gt_alpha", data=gt_alpha_real, compression="gzip")
        f.create_dataset("raw_itof", data=v_real)
        f.create_dataset("global_itof", data=v_real_no_d)
        f.create_dataset("direct_itof", data=v_real_d)
        f.create_dataset("amplitude_raw", data=A_in)
        f.create_dataset("phase_raw", data=phi_in)
        f.create_dataset("amplitude_direct", data=A_d)
        f.create_dataset("phase_direct", data=phi_d)
        f.create_dataset("amplitude_global", data=A_g)
        f.create_dataset("phase_global", data=phi_g)
        f.create_dataset("freqs", data=freqs)
        if flag_fit:
            f.create_dataset("transient_fit", data=Back_fit)

# TEST IMAGES
if fl_get_test:
    test_files = np.asarray(test_files)
    test_files = [data_dir + fil for fil in test_files]
    print("\nTest dataset:")
    Back, Back_nod, gt_depth_real, gt_alpha_real, _, _, _, v_real, v_real_no_d, v_real_d, _, Back_fit = acquire_pixels(test_files, n_patches, max_imgs, f_ran, s, flag_ow, flag_fit, fl_normalize_transient, freqs)
    num_elem = Back.shape[0]

    A_in, phi_in, A_g, phi_g, A_d, phi_d = Aphi_compute(v_real, v_real_no_d, v_real_d)
    file_test = out_dir + "test_n" + str(num_elem) + add_str + ".h5"
    with h5py.File(file_test, 'w') as f:
        f.create_dataset("name", data=datasetname)
        f.create_dataset("transient", data=Back, compression="gzip")
        f.create_dataset("transient_global", data=Back_nod, compression="gzip")
        f.create_dataset("gt_depth", data=gt_depth_real, compression="gzip")
        f.create_dataset("gt_alpha", data=gt_alpha_real, compression="gzip")
        f.create_dataset("raw_itof", data=v_real)
        f.create_dataset("global_itof", data=v_real_no_d)
        f.create_dataset("direct_itof", data=v_real_d)
        f.create_dataset("amplitude_raw", data=A_in)
        f.create_dataset("phase_raw", data=phi_in)
        f.create_dataset("amplitude_direct", data=A_d)
        f.create_dataset("phase_direct", data=phi_d)
        f.create_dataset("amplitude_global",data=A_g)
        f.create_dataset("phase_global", data=phi_g)
        f.create_dataset("freqs", data=freqs)
        if flag_fit:
            f.create_dataset("transient_fit", data=Back_fit)

end_time = time.time()
minutes, seconds = divmod(end_time - start_time, 60)
hours, minutes = divmod(minutes, 60)
print("\nTotal time to build the whole dataset is %d:%02d:%02d" % (hours, minutes, seconds))

