import numpy as np
from pathlib import Path
import getopt
from datetime import date
import sys
sys.path.append("./src/")
sys.path.append("./data/")
sys.path.append("../utils/")
import DataLoader as DataLoader
import PredictiveModel_hidden as PredictiveModel


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

    arg_name = "train"  # Argument containing the name of the training attempt
    arg_help = "{0} -n <name>".format(argv[0])  # Help string

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(argv[1:], "hn:", ["help", "name="])
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-n", "--name"):
            arg_name = arg  # Set the attempt name

    print("Attempt name: ", arg_name)
    print()

    return arg_name


if __name__ == '__main__':
    name_of_attempt = arg_parser(sys.argv)                      # String used to denominate the attempt.
    name_of_attempt = f"{str(date.today())}_{name_of_attempt}"  # Add the date to the name of the attempt
    fil_spat_size = 32                                          # Number of feature maps for the Spatial Feature Extractor model
    fil_dir_size = 32                                           # Number of feature maps for the Direct_CNN model
    fil_encoder = 32                                            # Number of feature maps of encoder and decoder

    # Training and test set generators
    fl_scale = True           # If True the normalization is performed
    fl_scale_perpixel = True  # If True the normalization is done pixel per pixel, otherwise each patch is normalized by the mean 20 MHz value
    fl_2freq = False          # If set, the training is done on only 2 frequencies (in this case 20 and 50 MHz)
    P = 3                     # Patch size
    dim_b = 1024              # Batch dimension
    dim_t = 2000              # Number of bins in the transient dimension
    lr = 1e-03                # Learning rate

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
    train_filename = str(Path(f"./data/train_n40200_s{str(P)}_nonorm{str_freqs}.h5"))
    val_filename = str(Path(f"./data/val_n13400_s{str(P)}_nonorm{str_freqs}.h5"))

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
    net = PredictiveModel.PredictiveModel(name=name_of_attempt, dim_b=dim_b, lr=lr, freqs=freqs, P=P, saves_path='./saves',
                                          dim_t=dim_t, fil_size=fil_dir_size, fil_denoise_size=fil_spat_size,
                                          dim_encoding=dim_encoding, fil_encoder=fil_encoder)
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
