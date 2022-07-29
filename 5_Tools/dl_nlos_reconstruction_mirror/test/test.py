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
- flag_nrom_perpixel:   Set to 'True' performs a normalization pixel per pixel. When 'False', the input is normalized with a running mean patch by patch 
- flag_epoch:           Set to 'True' to test on a specific epoch. Otherwise the weights corresponding to the epoch with the lowest loss on the validation set are going to be employed
- num_epoch:            if 'flag_epoch' is True, the weights corresponding to this epoch are going to be loaded for testing
- fl_test_img           Set to 'True' to test on synthetic images
- flag_plot:            Set to 'True' to plot and save the results  (Slows down the code)

"""

attempt_name = "2022-07-28_test_01"                                                                 # name of the stored approach weights
win_server_path = "Z:/decaligm"  # path to the server
git_folder_path = "thesis-nlos-for-itof/5_Tools/dl_nlos_reconstruction_mirror"                      # path to the git folder
dataset_folder = "mitsuba_renders/nlos_scenes/datasets/depth_map_ground_truth_far"                  # path to the dataset folder
data_path_real = "../../Datasets/S3S4S5/*"                                                          # path to the real images
data_path_synth = f"{win_server_path}/{dataset_folder}/final_dataset"                               # Path of the synthetic test set (same patch size as training and validation)
test_file_csv = f"{win_server_path}/{git_folder_path}/dataset_creation/data_split/test_images.csv"  # path to the test file
#weights_folder = f"../training/saves/{attempt_name}/checkpoints/"                                   # path to the weights
weights_folder = f"{win_server_path}/{git_folder_path}/training/saves/{attempt_name}/checkpoints/"  # path to the weights
dim_t = 2000                                                                                        # number of bins in the transient dimension
P = 3                                                                                               # patch size
flag_norm_perpixel = True                                                                           # normalization per pixel
flag_scale = True                                                                                   # whether to apply scaling on the inputs
flag_plot = False                                                                                   # whether to plot and save the results
flag_epoch = False                                                                                  # whether to test on a specific epoch
fl_test_img = True                                                                                 # whether to test on synthetic images
fil_denoise = 32
fil_autoencoder = 32
fil_direct = 32
num_epoch = 40000                                                                                   # epoch to test on
epoch_name_d = attempt_name + "_d_e" + str(num_epoch) + "_weights.h5"                               # name of the epoch to test on for the spatial net
epoch_name_v = attempt_name + "_v_e" + str(num_epoch) + "_weights.h5"                               # name of the epoch to test on for the direct net

# Check if the iToF data uses two or three frequencies and set their value accordingly
str_freqs = ""
if attempt_name[-5:] == "2freq":
    freqs = np.array((20e06, 50e06), dtype=np.float32)
    str_freqs = "_2freq"
else:
    freqs = np.array((20e06, 50e06, 60e06), dtype=np.float32)

# Create needed directories
if not os.path.exists("./out/" + attempt_name):
    os.mkdir("./out/" + attempt_name)

# Search for the weight files to test on, the one with the best loss on the validation set
search_str = "*best*"
weight_names = []
for f_name in os.listdir(weights_folder):
    if fnmatch.fnmatch(f_name, search_str):
        weight_names.append(weights_folder + f_name)

# Load the weights corresponding to the desired epoch
if flag_epoch:
    weight_names = weight_names[:5]
    weight_names[0] = weights_folder + epoch_name_d
    weight_names[1] = weights_folder + epoch_name_v

# Test on patches of the same dataset
print(" ")
names = glob.glob(data_path_real + "*.h5")

# Order the weights and datasets in alphabetical order
weight_names.sort()
print(weight_names)
names.sort()

if fl_test_img:
    data_path = data_path_synth
else:
    data_path = data_path_real

test_img(weight_names=weight_names,
         data_path=data_path,
         test_files=test_file_csv,
         P=P,
         freqs=freqs,
         fl_scale=flag_scale,
         fl_norm_perpixel=flag_norm_perpixel,
         fil_dir=fil_direct,
         fil_den=fil_denoise,
         fil_auto=fil_autoencoder,
         dim_t=dim_t)  # test on transient images
