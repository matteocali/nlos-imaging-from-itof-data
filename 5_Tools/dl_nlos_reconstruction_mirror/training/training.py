import numpy as np
from pathlib import Path
import sys
#sys.path.append("./src/")
sys.path.append("./data/")
sys.path.append("../utils/")
import src.DataLoader as DataLoader
import src.PredictiveModel_hidden as PredictiveModel
from datetime import date
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

"""
Main file for all trainings.
All flags for training and the number of feature maps for each network can be set from here

Flags:
-use_S1             Whether to use the S1 dataset for training
-use_data          Whether to use the walls dataset for training
-fl_scale_perpixel  Controls the normalization by the 20 MHz component. If the flag is True the normalization is done pixel per pixel, otherwise each patch is normalized by the mean 20 MHz value
-fl_2freq           If set, the training is done on only 2 frequencies (in this case 20 and 50 MHz)

Parameters:
-fil_spat_size      Number of feature maps for the Spatial Feature Extractor model
-fil_dir_size       Number of feature maps for the Direct_CNN model
-P                  Side of the input patches, which correspond to the receptive field of the network.
                    --> If P is set to 3 the Spatial feature extractor is not used
-dim_b              Batch size 
-dim_t              Number of bins in the transient dimension
"""

name_of_attempt = "test_01"  # String used to denominate the attempt.
name_of_attempt = str(date.today()) + "_" + name_of_attempt
fil_spat_size = 32  # Number of feature maps for the Spatial Feature Extractor model
fil_dir_size = 32  # Number of feature maps for the Direct_CNN model
fil_encoder = 32  # Number of feature maps of encoder and decoder

# Training and test set generators
use_data = True
fl_scale = True
fl_scale_perpixel = True
fl_2freq = False
P = 3
dim_b = 1024  # Batch dimension
dim_t = 2000  # Number of bins in the transient dimension

# Additional string used to highlight if the approach was trained on two frequencies
if fl_2freq:
    str_freqs = "_2freq"
    freqs = np.array((20e06, 50e06), dtype=np.float32)
    dim_encoding = 8  # Dimension in the encoding domain
else:
    str_freqs = ""
    freqs = np.array((20e06, 50e06, 60e06), dtype=np.float32)
    dim_encoding = 12  # Dimension in the encoding domain

# Training and validation data for dataset
print(f"CURRENT FOLDER: {os.getcwd()}")
train_filename = str(Path(f"/cig/common04nb/students/decaligm/thesis-nlos-for-itof/5_Tools/dl_nlos_reconstruction_mirror/training/data/train_n40200_s{str(P)}_nonorm{str_freqs}.h5"))
val_filename = str(Path(f"/cig/common04nb/students/decaligm/thesis-nlos-for-itof/5_Tools/dl_nlos_reconstruction_mirror/training/data/val_n13400_s{str(P)}_nonorm{str_freqs}.h5"))

# Put the loaded data in the right format for the network.
train_loader = DataLoader.DataLoader(filename=train_filename,
                                     freqs=freqs,
                                     dim_batch=dim_b,
                                     fl_scale=fl_scale,
                                     fl_scale_perpixel=fl_scale_perpixel,
                                     P=P)
val_loader = DataLoader.DataLoader(filename=val_filename,
                                   freqs=freqs,
                                   dim_batch=dim_b,
                                   fl_scale=fl_scale,
                                   fl_scale_perpixel=fl_scale_perpixel,
                                   P=P)

# Prepare the main model
net = PredictiveModel.PredictiveModel(name=name_of_attempt, dim_b=dim_b, freqs=freqs, P=P, saves_path='./saves',
                                      dim_t=dim_t, fil_size=fil_dir_size, fil_denoise_size=fil_spat_size,
                                      dim_encoding=dim_encoding, fil_encoder=fil_encoder)
# Summaries of the various networks
net.SpatialNet.summary()
net.DirectCNN.summary()

# Path of the weight in case we want to start from a pretrained network
pretrain_filenamed = None
pretrain_filenamev = None
pretrain_filenamez = None

# Training loop
net.training_loop(train_w_loader=train_loader, test_w_loader=val_loader, final_epochs=50000, print_freq=1, save_freq=25, pretrain_filenamed=pretrain_filenamed, pretrain_filenamev=pretrain_filenamev, pretrain_filenamez=pretrain_filenamez)

