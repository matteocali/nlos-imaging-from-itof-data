import numpy as np
import os
import h5py
import sys
sys.path.append("./src/") 
sys.path.append("../utils/") 
sys.path.append("../itraining/src/")
from src.fun_test_real import test_real
from src.fun_test_S1 import testS1
from src.fun_test_synth import test_synth
from src.fun_test_img_transient import test_img
#from fun_test_img_transient import test_img
from src.fun_test_aligned import test_aligned
#from fun_test_aligned import test_aligned
import fnmatch
import glob
import os

os.environ["CUDA_VISIBLE_DEVICES"]=""     # whether to use a gpu and in case which one

"""

Code for testing the approaches on synthetic and real images and on transient information. Works for 2 or 3 frequencies trainings.
In case of 2 frequencies, the name of the approach must finish with '2freq'

- P:                    Patch size used for training the network
- flag_nrom_perpixel:   Set to 'True' performs a normalization pixel per pixel. Wehn 'False', the input is normalized with a running mean patch by patch 
- flag_epoch:           Set to 'True' to test on a specific epoch. Otherwise the weights corresponding to the epoch with the lowest loss on the alidation set are going to be employed
- flag_plot:            Set to 'True' to plot and save the results  (Slows down the code)
- num_epoch:            if 'flag_epoch' is True, the weights corresponding to this epoch are going to be loaded for testing

"""

attempt_name = "2022-02-28_s3_sameconv"    # Name of the stored approach weights
dim_t = 2000
P = 3
fl_newhidden = True
flag_norm_perpixel = True
flag_scale = True          # Whether to apply scaling on the inputs
flag_plot = False
flag_epoch = False 
fl_test_img = False         # Whether to test on synthetic images
fil_denoise = 32
fil_autoencoder = 32
fil_direct = 128

# Whether to test using the old network
fl_test_old = False
old_weights = []
Pold = 3
if fl_test_old:
    old_folder = "../../svn_backup/training/saves/2022-03-01_walls_s3_transient/checkpoints/"
    #old_folder = "../../svn_backup/training/saves/2021-12-06_walls_s3_vd/checkpoints/"
    searchstr = "*best_weights.h5"
    searchstr = "*20000*"
    for fname in os.listdir(old_folder):
        if fnmatch.fnmatch(fname,searchstr):
            old_weights.append(old_folder+fname)
    old_weights.sort()
    print(old_weights)
else:
    old_weights = None





num_epoch= 40000
epoch_named = attempt_name + "_d_e" +str(num_epoch) + "_weights.h5" 
epoch_namev = attempt_name + "_v_e" +str(num_epoch) + "_weights.h5" 
epoch_namez = attempt_name + "_z_e" +str(num_epoch) + "_weights.h5" 
epoch_name_enc = attempt_name + "_enc_e" +str(num_epoch) + "_weights.h5" 
epoch_name_dec = attempt_name + "_dec_e" +str(num_epoch) + "_weights.h5" 
epoch_name_predv_enc = attempt_name + "_predv_enc_e" +str(num_epoch) + "_weights.h5" 

str_freqs = ""
if attempt_name[-5:]=="2freq":
    freqs=np.array((20e06, 50e06),dtype=np.float32)
    str_freqs = "_2freq"
else:
    freqs=np.array((20e06, 50e06, 60e06),dtype=np.float32)

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
    if attempt_name[-5:]=="2freq":
        data_path_synth = "../training/data/test_walls_6800_s3_2freq.h5"                          # Path of the synthetic test set (same patch size as training and validation)
    
elif P == 11:
    data_path_synth = "../training/data/test_walls_6800_s11.h5"                          # Path of the synthetic test set (same patch size as training and validation)
    if attempt_name[-5:]=="2freq":
        data_path_synth = "../training/data/test_walls_6800_s11_2freq.h5"                          # Path of the synthetic test set (same patch size as training and validation)
elif P == 31:
    data_path_synth = "../training/data/test_walls_4133_s31.h5"                          # Path of the synthetic test set (same patch size as training and validation)
#data_path_synth = "../training/data/val_walls_4478_s11.h5"                          # Path of the synthetic test set (same patch size as training and validation)
#data_path_synth = "../training/data/test_aug_shot_13600_s11.h5"

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
#testS1(data_pathS1,weight_names,attempt_name,"S1",P,freqs,flag_norm_perpixel,flag_plot,fil_direct,fil_denoise)            # Test on S1
#test_real(names[0],weight_names,attempt_name,"S3",P,freqs,flag_norm_perpixel,flag_plot,fil_direct,fil_denoise)           # Test on S3
#test_real(names[1],weight_names,attempt_name,"S4",P,freqs,flag_norm_perpixel,flag_plot,fil_direct,fil_denoise)           # Test on S4
#test_real(names[2],weight_names,attempt_name,"S5",P,freqs,flag_norm_perpixel,flag_plot,fil_direct,fil_denoise)           # Test on S5
test_img(data_path_real,weight_names,attempt_name,"",P,freqs,flag_scale,flag_norm_perpixel,fil_direct,fil_denoise,fil_autoencoder,fl_test_old,old_weights,Pold,fl_newhidden,dim_t=dim_t)         # Test on transient images
#test_aligned(data_path_real,weight_names,attempt_name,"",P,freqs,flag_scale,flag_norm_perpixel,fil_direct,fil_denoise,fil_autoencoder,fl_test_old,old_weights,Pold,fl_newhidden,dim_t=dim_t)         # Test on transient images
#test_synth(data_path_synth,weight_names,attempt_name,"",P,freqs,fl_test_img,flag_scale,fil_direct,fil_denoise,fil_autoencoder,fl_newhidden,dim_t=dim_t)      # Test on synthetic patches/pixels

