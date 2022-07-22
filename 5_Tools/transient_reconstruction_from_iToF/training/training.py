import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import dirname, abspath
sys.path.append("./src/")
sys.path.append("./data/")
sys.path.append("../utils/")
import DataLoader
#import PredictiveModel
import PredictiveModel_hidden as PredictiveModel
#import PredictiveModel_hidden_2 as PredictiveModel
from datetime import date
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

"""
Main file for all trainings.
All flags for training and the number of feature maps for each network can be set from here

Flags:
-use_S1             Whether to use the S1 dataset for training
-use_walls          Whether to use the walls dataset for training
-use_transient      Whether to train our model also on transient data (backpropagation is done also on the Transient Reconstruction Module)
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
"""
#physical_devices = tf.config.experimental.list_physical_devices("GPU")
#assert len(physical_devices) > 0
#tf.config.experimental.set_memory_growth(physical_devices[0],True)
#import PredictiveModel_min as PredictiveModel
"""
name_of_attempt = "s3_samesize"     # String used to denominate the attempt. 
name_of_attempt = str(date.today()) + "_" + name_of_attempt
fil_spat_size = 32
fil_dir_size = 32
fil_encoder = 32     #Number of feature maps of encoder and decoder

# Training and test set generators
use_S1 = False
use_walls = True
use_transient = False#True (se funziona anche con False)
fl_scale = True
fl_scale_perpixel = True
fl_2freq = False
P = 3
dim_b = 1024 # Dimensione della batch
dim_t = 2000 # Number of bins in the transient dimension
# Additional string used to highlight if the approach was trained on two frequencies
str_freqs = ""
if fl_2freq:
    str_freqs = "_2freq"
    freqs = np.array((20e06,50e06),dtype=np.float32)
    dim_encoding = 8    #Dimension in the encoding domain
else:
    freqs = np.array((20e06,50e06,60e06),dtype=np.float32)
    dim_encoding = 12    #Dimension in the encoding domain

# Path of training and validation data of dataset S1
train_S1_filename = "./data/train_S1_400_s" + str(P) + str_freqs+".h5"
val_S1_filename = "./data/val_S1_140_s" + str(P) + str_freqs+".h5"
# Training and validation data for dataset walls
train_walls_filename = "./data/train_walls_200_s3_nonorm" + str(P) + str_freqs+".h5"
val_walls_filename = "./data/val_walls_6800_s" + str(P) + str_freqs+".h5"
#train_walls_filename = "./data/train_aug_shot_20200_s" +str(P)+ str_freqs+".h5"
#val_walls_filename = "./data/val_aug_shot_6800_s" + str(P) + str_freqs+".h5"
# Path of the real dataset S3, used for validation
val_S3_filename = "./data/val_S3_full_imgs" + str_freqs + ".h5"

# Put the loaded data in the right format for the network.
# Note that some datasets might not be used depending on the previously set flags
train_S1_loader = DataLoader.DataLoader(train_S1_filename, freqs, dim_batch=dim_b, fl_scale=fl_scale, fl_scale_perpixel=fl_scale_perpixel, P=P)
val_S1_loader = DataLoader.DataLoader(val_S1_filename, freqs, dim_batch=dim_b, fl_scale=fl_scale, fl_scale_perpixel=fl_scale_perpixel, P=P)
train_walls_loader = DataLoader.DataLoader(train_walls_filename, freqs, dim_batch=dim_b, fl_scale=fl_scale, fl_scale_perpixel=fl_scale_perpixel, P=P)
val_walls_loader = DataLoader.DataLoader(val_walls_filename, freqs, dim_batch=dim_b, fl_scale=fl_scale, fl_scale_perpixel=fl_scale_perpixel, P=P)
val_S3_loader = DataLoader.DataLoader(val_S3_filename, freqs, dim_batch=8, fl_scale=fl_scale, fl_scale_perpixel=fl_scale_perpixel, P=P)

# Prepare the main model
net = PredictiveModel.PredictiveModel(name=name_of_attempt,
                                      dim_b=dim_b,
                                      freqs=freqs,
                                      P=P,
                                      saves_path='./saves',
                                      use_S1=use_S1,
                                      use_transient=use_transient,
                                      dim_t=dim_t,
                                      fil_size=fil_dir_size,
                                      fil_denoise_size=fil_spat_size,
                                      dim_encoding=dim_encoding,
                                      fil_encoder=fil_encoder)
# Summaries of the various networks
net.SpatialNet.summary()
net.DirectCNN.summary()

# Path of the weight in case we want to start from a pretrained network
pretrain_filenamed = None
pretrain_filenamev = None
pretrain_filenamez = None
# Training loop
net.training_loop(train_S1_loader, val_S1_loader, train_walls_loader, val_walls_loader, val_S3_loader, final_epochs=50000, print_freq=1, save_freq=25, pretrain_filenamed=pretrain_filenamed, pretrain_filenamev=pretrain_filenamev, pretrain_filenamez=pretrain_filenamez, use_S1=use_S1, use_walls=use_walls)

