import random
import os, sys
sys.path.append("./src/")
sys.path.append("../utils/")
import h5py
import numpy as np
import glob
import csv
flag_aug = False
if flag_aug:
    from fun_acquire_pixels_kxk import acquire_pixels
else:
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
    -f_ran:                     Set to True chooses the patches in random positions from all over the image, otherwise a grid willl be employed
    -n_patches:                 Number of patches taken from each images
    -s:                         Patch size
    -max_imgs:                  Maximum number of images of each set. Can be used to make a quick run of the process. Otherwise set it to high value (higher than max dataset size)
    -train_slice:               Percentage of images belonging to training dataset expressed as a number between 0 and 1. Test and validation will take each half of what remains
    -fl_get_train:              If set to True we will acquire the training set. Same for the validation and test ones.
    -fl_get_val:
    -fl_get_test:
    -fl_normalize_transient:    Whether to normalize all transient to sum to 1. Can be useful but it is better to do it later right before training
    
"""


out_dir = "../training/data/"    # Output directory
data_dir = "../dataset_rec/"  # dataset directory

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

flag_new_shuffle = True#False  # whether to shuffle again the images and create new training validation and test datasets
flag_ow = True  # whether to use also one wall images or not
flag_fit = False  # whether to fit the data using weibull functions (slows down the code)
f_ran = True
n_patches = 200  # number of pixels taken from each image
s = 3  # size of each input patch
add_str = "_s" + str(s)  # Dataset name
if flag_aug:
    add_str += "_aug"
max_imgs=1000
train_slice = 0.6
val_slice = (1-train_slice)/2
freqs = np.array((20e06, 50e06, 60e06),dtype=np.float32)
#freqs = np.array((20e06,100e06),dtype=np.float32)
if freqs.shape[0] == 2:
    add_str += "_2freq"

# Choose which datasets you want to build
fl_get_train = True
fl_get_val = False#True
fl_get_test = False#True
fl_normalize_transient = False  # Whether to normalize the transient information 

# Grab all the names of the images of the dataset, shuffle them and save them in a csv file
if not fl_normalize_transient:
    add_str += "_nonorm"

if flag_new_shuffle:
    path = data_dir + "*.h5"    # Path to the transient dataset
    filenames = glob.glob(path)
    num_img = len(filenames)
    indexes = np.arange(num_img)
    indexes = np.random.permutation(indexes)
    indexes = np.array(indexes)
    filenames = np.array(filenames)
    filenames = filenames[indexes]
    for i,file in enumerate(filenames):
        filenames[i] = file.replace("\\", "/")
    with open("shuffled_images.csv", "w", newline="") as csvfile:
        wr = csv.writer(csvfile)
        for filename in filenames:
            wr.writerow([filename])

    file_list = []
    with open("shuffled_images.csv", "r") as csvfile:
        wr = csv.reader(csvfile)
        for row in wr:
            file_list.append(row)

    file_list = [item for sublist in file_list for item in sublist]
    owalls = [s for s in file_list if "cube" in s]
    twalls_ang = [s for s in file_list if "sphere" in s]
    twalls = [s for s in file_list if (("twalls" in s) and ("angle" not in s))]
    three_walls = [s for s in file_list if "3walls" in s]

    images_list = [file_list]#[owalls, twalls_ang, twalls, three_walls]


    train_files = []
    val_files = []
    test_files = []
    for images in images_list:
        num_img = len(images)
        n_train = int(np.round(train_slice*num_img))
        n_val = int(np.round(val_slice*num_img))
        n_test = num_img-n_train-n_val
        train_tmp = images[:n_train]
        val_tmp = images[n_train:n_train+n_val]
        test_tmp = images[n_train+n_val:]
        train_files += train_tmp
        val_files += val_tmp
        test_files += test_tmp
    np.random.shuffle(test_files)
    np.random.shuffle(val_files)
    np.random.shuffle(train_files)



    with open("./data_split/train_images.csv","w",newline="") as csvfile:
        wr = csv.writer(csvfile)
        for filename in train_files:
            wr.writerow([os.path.basename(filename)])
    with open("./data_split/val_images.csv","w",newline="") as csvfile:
        wr = csv.writer(csvfile)
        for filename in val_files:
            wr.writerow([os.path.basename(filename)])
    with open("./data_split/test_images.csv","w",newline="") as csvfile:
        wr = csv.writer(csvfile)
        for filename in test_files:
            wr.writerow([os.path.basename(filename)])


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

N_train = len(train_files)
N_val = len(val_files)
N_test = len(test_files)

print(N_train,N_val,N_test)

# TRAINING IMAGES
datasetname = 1    # Flag used to keep track of the used dataset



if fl_get_train:
    train_files = np.asarray(train_files)
    train_files = [data_dir + fil for fil in train_files]
    print("Training dataset")


    Back, Back_nod, Fit_Parameters, peak_ind, peak_val, v_real, v_real_no_d, v_real_d, phi, Back_fit = acquire_pixels(train_files,n_patches,max_imgs,f_ran,s,flag_ow,flag_fit,fl_normalize_transient,freqs)
    num_elem = Back.shape[0]

    A_in,phi_in,A_g,phi_g,A_d,phi_d = Aphi_compute(v_real,v_real_no_d,v_real_d)

    file_train = out_dir + "train_walls_"+str(num_elem)+add_str+".h5"
    with h5py.File(file_train,'w') as f:
        dset = f.create_dataset("name", data=datasetname)
        dset = f.create_dataset("transient", data=Back, compression="gzip") # Conviene togliere la compressione
        dset = f.create_dataset("transient_global", data=Back_nod, compression="gzip")
        dset = f.create_dataset("raw_itof", data=v_real)
        dset = f.create_dataset("global_itof", data=v_real_no_d)
        dset = f.create_dataset("direct_itof", data=v_real_d)
        dset = f.create_dataset("amplitude_raw",data = A_in)
        dset = f.create_dataset("phase_raw",data = phi_in)
        dset = f.create_dataset("amplitude_direct",data = A_d)
        dset = f.create_dataset("phase_direct",data = phi_d)
        dset = f.create_dataset("amplitude_global",data = A_g)
        dset = f.create_dataset("phase_global",data = phi_g)
        dset = f.create_dataset("freqs",data = freqs)
        if flag_fit:
            dset = f.create_dataset("transient_fit",data = Back_fit)



# VALIDATION IMAGES
if fl_get_val:
    val_files = np.asarray(val_files)
    val_files = [data_dir + fil for fil in val_files]
    print("Validation dataset")
    Back, Back_nod, Fit_Parameters, peak_ind, peak_val, v_real, v_real_no_d, v_real_d, phi, Back_fit = acquire_pixels(val_files,n_patches,max_imgs,f_ran,s,flag_ow,flag_fit,fl_normalize_transient,freqs)
    num_elem = Back.shape[0]

    A_in,phi_in,A_g,phi_g,A_d,phi_d = Aphi_compute(v_real,v_real_no_d,v_real_d)
    file_val = out_dir + "val_walls_"+str(num_elem)+add_str+".h5"
    with h5py.File(file_val,'w') as f:
        dset = f.create_dataset("name", data=datasetname)
        dset = f.create_dataset("transient", data=Back, compression="gzip")
        dset = f.create_dataset("transient_global", data=Back_nod, compression="gzip")
        dset = f.create_dataset("raw_itof", data=v_real)
        dset = f.create_dataset("global_itof", data=v_real_no_d)
        dset = f.create_dataset("direct_itof", data=v_real_d)
        dset = f.create_dataset("amplitude_raw",data = A_in)
        dset = f.create_dataset("phase_raw",data = phi_in)
        dset = f.create_dataset("amplitude_direct",data = A_d)
        dset = f.create_dataset("phase_direct",data = phi_d)
        dset = f.create_dataset("amplitude_global",data = A_g)
        dset = f.create_dataset("phase_global",data = phi_g)
        dset = f.create_dataset("freqs",data = freqs)
        if flag_fit:
            dset = f.create_dataset("transient_fit",data = Back_fit)

# TEST IMAGES
if fl_get_test:
    test_files = np.asarray(test_files)
    test_files = [data_dir + fil for fil in test_files]
    print("Test dataset")
    Back, Back_nod,  Fit_Parameters, peak_ind, peak_val, v_real, v_real_no_d, v_real_d, phi, Back_fit = acquire_pixels(test_files,n_patches,max_imgs,f_ran,s,flag_ow,flag_fit,fl_normalize_transient,freqs)
    num_elem = Back.shape[0]

    A_in,phi_in,A_g,phi_g,A_d,phi_d = Aphi_compute(v_real,v_real_no_d,v_real_d)
    file_test = out_dir + "test_walls_"+str(num_elem)+add_str+".h5"
    with h5py.File(file_test,'w') as f:
        dset = f.create_dataset("name", data=datasetname)
        dset = f.create_dataset("transient", data=Back, compression="gzip")
        dset = f.create_dataset("transient_global", data=Back_nod, compression="gzip")
        dset = f.create_dataset("raw_itof", data=v_real)
        dset = f.create_dataset("global_itof", data=v_real_no_d)
        dset = f.create_dataset("direct_itof", data=v_real_d)
        dset = f.create_dataset("amplitude_raw",data = A_in)
        dset = f.create_dataset("phase_raw",data = phi_in)
        dset = f.create_dataset("amplitude_direct",data = A_d)
        dset = f.create_dataset("phase_direct",data = phi_d)
        dset = f.create_dataset("amplitude_global",data = A_g)
        dset = f.create_dataset("phase_global",data = phi_g)
        dset = f.create_dataset("freqs",data = freqs)
        if flag_fit:
            dset = f.create_dataset("transient_fit",data = Back_fit)




