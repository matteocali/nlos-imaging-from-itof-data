import numpy as np
import os
from pathlib import Path
import getopt
from datetime import date
import sys
sys.path.append("./src/")
sys.path.append("./data/")
sys.path.append("../utils/")
import DataLoader as DataLoader
import PredictiveModel_hidden as PredictiveModel
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings


"""
Main file for all trainings.
All flags for training and the number of feature maps for each network can be set from here

Flags:
-fl_scale_perpixel  Controls the normalization by the 20 MHz component. If the flag is True the normalization is done pixel per pixel, otherwise each patch is normalized by the mean 20 MHz value
-fl_2freq           If set, the training is done on only 2 frequencies (in this case 20 and 50 MHz)

Parameters:
-fil_spat_size      Number of feature maps for the Spatial Feature Extractor model
-fil_dir_size       Number of feature maps for the Direct_CNN model
-P                  Side of the input patches, which correspond to the receptive field of the network.
                    --> If P is set to 3 the Spatial feature extractor is not used
-lr                 Learning rate for the optimizer
-dim_b              Batch size 
-dim_t              Number of bins in the transient dimension
"""


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line
    (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_name = "train"   # Argument containing the name of the training attempt
    arg_lr = 1e-04       # Argument containing the learning rate
    arg_n_layers = 4     # Argument containing the number of layers in the network
    arg_loss_scale = 1   # Argument containing the loss scale
    arg_kernel_size = 3  # Argument containing the kernel size
    arg_train_dts = ""   # Argument containing the name of the training dataset
    arg_val_dts = ""     # Argument containing the name of the validation dataset
    arg_help = "{0} -n <name> -r <lr> -l <layers> -s <scale> -k <kernel> -t <train>, -v <validation>".format(argv[0])  # Help string

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, _ = getopt.getopt(argv[1:], "hn:r:l:s:k:t:v:", ["help", "name=", "lr=", "layers=", "scale=", "kernel=", "train=", "validation="])
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-n", "--name"):
            arg_name = arg  # Set the attempt name
        elif opt in ("-r", "--lr"):
            arg_lr = float(arg)  # Set the learning rate
        elif opt in ("-l", "--layers"):
            arg_n_layers = int(arg)
        elif opt in ("-s", "--scale"):
            arg_loss_scale = int(arg)
        elif opt in ("-k", "--kernel"):
            arg_kernel_size = int(arg)
        elif opt in ("-t", "--train"):
            arg_train_dts = arg
        elif opt in ("-v", "--validation"):
            arg_val_dts = arg

    print("Attempt name: ", arg_name)
    print("Learning rate: ", arg_lr)
    print("Number of layers: ", arg_n_layers)
    print("Loss scale factor: ", arg_loss_scale)
    print("Kernel size: ", arg_kernel_size)
    print("Train dataset name: ", arg_train_dts)
    print("Validation dataset name: ", arg_val_dts)
    print()

    return [arg_name, arg_lr, arg_n_layers, arg_loss_scale, arg_kernel_size, arg_train_dts, arg_val_dts]


if __name__ == '__main__':
    args = arg_parser(sys.argv)  # Get the arguments from the command line

    name_of_attempt = args[0]                                   # String used to denominate the attempt.
    name_of_attempt = f"{str(date.today())}_{name_of_attempt}"  # Add the date to the name of the attempt
    fil_spat_size = 32                                          # Number of feature maps for the Spatial Feature Extractor model
    fil_dir_size = 32                                           # Number of feature maps for the Direct_CNN model
    lr = args[1]                                                # Learning rate
    n_layers = args[2]                                          # Number of layers in the network
    fil_encoder = 32                                            # Number of feature maps of encoder and decoder
    loss_scale = args[3]                                        # Loss scale factor
    kernel_size = args[4]                                       # Kernel size

    # Training and validation data for dataset
    train_filename = f"./data/{args[5]}.h5"
    val_filename = f"./data/{args[6]}.h5"

    # Extract the patch size from the dts name
    dts_name = args[5]
    dts_name = dts_name.split("_")
    patch_str = [elm for elm in dts_name if "ps" in elm][0]

    # Training and test set generators
    fl_scale = True           # If True the normalization is performed
    fl_scale_perpixel = True  # If True the normalization is done pixel per pixel, otherwise each patch is normalized by the mean 20 MHz value
    fl_2freq = False          # If set, the training is done on only 2 frequencies (in this case 20 and 50 MHz)
    P = int(patch_str[2:])    # Patch size
    dim_b = 1024              # Batch dimension
    dim_t = 2000              # Number of bins in the transient dimension


    # Additional string used to highlight if the approach was trained on two frequencies
    if fl_2freq:
        str_freqs = "_2freq"
        freqs = np.array((20e06, 50e06), dtype=np.float32)
        dim_encoding = 8  # Dimension in the encoding domain
    else:
        str_freqs = ""
        freqs = np.array((20e06, 50e06, 60e06), dtype=np.float32)
        dim_encoding = 12  # Dimension in the encoding domain

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
    net = PredictiveModel.PredictiveModel(name=name_of_attempt, dim_b=dim_b, lr=lr, n_layers=n_layers, freqs=freqs, P=P, saves_path='./saves',
                                          dim_t=dim_t, fil_size=fil_dir_size, fil_denoise_size=fil_spat_size,
                                          dim_encoding=dim_encoding, fil_encoder=fil_encoder, loss_scale_factor=loss_scale, kernel_size=kernel_size)
    # Summaries of the various networks
    #net.SpatialNet.summary()
    net.DirectCNN.summary()

    # Path of the weight in case we want to start from a pretrained network
    pretrain_filenamed = None
    pretrain_filenamev = None

    # Training loop
    net.training_loop(train_w_loader=train_loader,
                      test_w_loader=val_loader,
                      final_epochs=50000,
                      print_freq=1,
                      save_freq=25,
                      pretrain_filenamed=pretrain_filenamed,
                      pretrain_filenamev=pretrain_filenamev)
