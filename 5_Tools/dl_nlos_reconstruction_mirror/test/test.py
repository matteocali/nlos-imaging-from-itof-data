import numpy as np
import sys
sys.path.append("./src/") 
sys.path.append("../utils/") 
sys.path.append("../itraining/src/")
from src.fun_test_img_transient import test_img
import fnmatch
import glob
import os

"""

Code for testing the approaches on synthetic and real images and on transient information. Works for 2 or 3 frequencies trainings.
In case of 2 frequencies, the name of the approach must finish with '2freq'

- P:                    Patch size used for training the network
- flag_nrom_perpixel:   Set to 'True' performs a normalization pixel per pixel. Wehn 'False', the input is normalized with a running mean patch by patch 
- flag_epoch:           Set to 'True' to test on a specific epoch. Otherwise the weights corresponding to the epoch with the lowest loss on the alidation set are going to be employed
- flag_plot:            Set to 'True' to plot and save the results  (Slows down the code)
- num_epoch:            if 'flag_epoch' is True, the weights corresponding to this epoch are going to be loaded for testing

"""

attempt_name = "2022-07-27_test_01"  # name of the stored approach weights
dim_t = 2000
P = 3
fl_newhidden = True
flag_norm_perpixel = True
flag_scale = True  # whether to apply scaling on the inputs
flag_plot = False
flag_epoch = False 
fl_test_img = False  # whether to test on synthetic images
fil_denoise = 32
fil_autoencoder = 32
fil_direct = 128

# Whether to test using the old network
fl_test_old = False
old_weights = []
Pold = 3
if fl_test_old:
    old_folder = "../../svn_backup/training/saves/2022-03-01_walls_s3_transient/checkpoints/"
    searchstr = "*best_weights.h5"
    searchstr = "*20000*"
    for fname in os.listdir(old_folder):
        if fnmatch.fnmatch(fname,searchstr):
            old_weights.append(old_folder+fname)
    old_weights.sort()
    print(old_weights)
else:
    old_weights = None

num_epoch = 40000
epoch_named = attempt_name + "_d_e" + str(num_epoch) + "_weights.h5"
epoch_namev = attempt_name + "_v_e" + str(num_epoch) + "_weights.h5"
epoch_namez = attempt_name + "_z_e" + str(num_epoch) + "_weights.h5"
epoch_name_enc = attempt_name + "_enc_e" + str(num_epoch) + "_weights.h5"
epoch_name_dec = attempt_name + "_dec_e" + str(num_epoch) + "_weights.h5"
epoch_name_predv_enc = attempt_name + "_predv_enc_e" + str(num_epoch) + "_weights.h5"

str_freqs = ""
if attempt_name[-5:] == "2freq":
    freqs = np.array((20e06, 50e06), dtype=np.float32)
    str_freqs = "_2freq"
else:
    freqs = np.array((20e06, 50e06, 60e06), dtype=np.float32)

# Create needed directories
if not os.path.exists("./out/"+attempt_name):
    os.mkdir("./out/"+attempt_name)
if not os.path.exists("./out/"+attempt_name+"/S1"):
    os.mkdir("./out/"+attempt_name+"/S1")
if not os.path.exists("./out/"+attempt_name+"/S3"):
    os.mkdir("./out/"+attempt_name+"/S3")
if not os.path.exists("./out/"+attempt_name+"/S4"):
    os.mkdir("./out/"+attempt_name+"/S4")
if not os.path.exists("./out/"+attempt_name+"/S5"):
    os.mkdir("./out/"+attempt_name+"/S5")

data_path_real = "../../Datasets/S3S4S5/*"                                 # Path of the real data used for validation and testing (S3,S4 and S5 datasets)
data_pathS1 = "../../Datasets/S1/synthetic_dataset/test_set/"              # Path of the synthetic dataset S1
if P == 3:
    data_path_synth = "../training/data/test_walls_6800_s3.h5"                          # Path of the synthetic test set (same patch size as training and validation)
    if attempt_name[-5:] == "2freq":
        data_path_synth = "../training/data/test_walls_6800_s3_2freq.h5"                          # Path of the synthetic test set (same patch size as training and validation)
    
elif P == 11:
    data_path_synth = "../training/data/test_walls_6800_s11.h5"                          # Path of the synthetic test set (same patch size as training and validation)
    if attempt_name[-5:] == "2freq":
        data_path_synth = "../training/data/test_walls_6800_s11_2freq.h5"                          # Path of the synthetic test set (same patch size as training and validation)
elif P == 31:
    data_path_synth = "../training/data/test_walls_4133_s31.h5"                          # Path of the synthetic test set (same patch size as training and validation)
weights_folder = "..\\training\\saves\\" + attempt_name + "\\checkpoints\\"
searchstr = "*best*"
weight_names = []
for fname in os.listdir(weights_folder):
    if fnmatch.fnmatch(fname,searchstr):
        weight_names.append(weights_folder+fname)
if flag_epoch:
    weight_names = weight_names[:5]
    weight_names[0] = weights_folder + epoch_named
    weight_names[1] = weights_folder + epoch_namev
    weight_names[2] = weights_folder + epoch_name_enc
    weight_names[3] = weights_folder + epoch_name_dec
    weight_names[4] = weights_folder + epoch_name_predv_enc

# Test on patches of the same dataset
print(" ")
names = glob.glob(data_path_real+"*.h5")
# Order the weights and datasets in alphabetical order
weight_names.sort()
print(weight_names)
names.sort()
test_img(weights_path=data_path_real,
         name=weight_names,
         attempt_name=attempt_name,
         Sname="",
         P=P,
         freqs=freqs,
         fl_scale=flag_scale,
         fl_norm_perpixel=flag_norm_perpixel,
         fil_dir=fil_direct,
         fil_den=fil_denoise,
         fil_auto=fil_autoencoder,
         fl_test_old=fl_test_old,
         old_weights=old_weights,
         Pold=Pold,
         fl_newhidden=fl_newhidden,
         dim_t=dim_t)  # test on transient images
