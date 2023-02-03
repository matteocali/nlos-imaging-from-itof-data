import numpy as np
import os
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

    arg_name = "train"     # Argument containing the name of the training attempt
    arg_lr = 1e-04         # Argument containing the learning rate
    arg_train_dts = ""     # Argument containing the name of the training dataset
    arg_val_dts = ""       # Argument containing the name of the validation dataset
    arg_filter = 32        # Argument containing the number of the filter to be used
    arg_loss = "mae"       # Argument containing the loss function to be used
    arg_n_layer = None     # Argument containing the number of layers to be used
    arg_batch_size = 2048  # Argument containing the batch size
    arg_n_epochs = 10000   # Argument containing the number of epochs
    arg_dropout = None     # Argument containing the dropout rate
    arg_alpha_scale = 1.0  # Argument containing the alpha scale
    arg_pretrained = None  # Argument containing the path to the pretrained weights
    arg_help = "{0} -n <name> -r <lr> -t <train> -v <validation> -f <filter> -l <loss> -s <n_layers> -b <batch_size> " \
               "-e <n_epochs> -d <dropout> -a <alpha_loss_scale> -p <pretrained>".format(argv[0])  # Help string

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, _ = getopt.getopt(argv[1:], "hn:r:t:v:f:l:s:b:e:d:a:p:", ["help", "name=", "lr=", "train=", "validation=",
                                                                        "filter=", "loss=", "n_layers=", "batch_size=",
                                                                        "n_epochs=", "dropout=", "alpha_scale=",
                                                                        "pretrained="])
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-n", "--name"):
            arg_name = arg  # Set the attempt name
        elif opt in ("-r", "--lr"):
            arg_lr = float(arg)  # Set the learning rate
        elif opt in ("-t", "--train"):
            arg_train_dts = arg  # Set the training dataset
        elif opt in ("-v", "--validation"):
            arg_val_dts = arg  # Set the validation dataset
        elif opt in ("-f", "--filter"):
            arg_filter = int(arg)  # Set the number of filters
        elif opt in ("-l", "--loss"):
            arg_loss = arg  # Set the loss function
        elif opt in ("-s", "--n_layers"):
            arg_n_layer = int(arg)  # Set the number of layers
        elif opt in ("-b", "--batch_size"):
            arg_batch_size = int(arg)  # Set the batch size
        elif opt in ("-e", "--n_epochs"):
            arg_n_epochs = int(arg)  # Set the number of epochs
        elif opt in ("-d", "--dropout"):
            arg_dropout = float(arg)  # Set the dropout rate
        elif opt in ("-a", "--alpha_scale"):
            arg_alpha_scale = float(arg)
        elif opt in ("-p", "--pretrained"):
            arg_pretrained = arg

    print("Attempt name: ", arg_name)
    print("Learning rate: ", arg_lr)
    print("Filter size: ", arg_filter)
    print("Loss function: ", arg_loss)
    print("Number of layers: ", arg_n_layer)
    print("Batch size: ", arg_batch_size)
    print("Number of epochs: ", arg_n_epochs)
    print("Dropout rate: ", arg_dropout)
    print("Train dataset name: ", arg_train_dts)
    print("Validation dataset name: ", arg_val_dts)
    print("Alpha loss scale: ", arg_alpha_scale)
    print("Pretrained weights: ", arg_pretrained)
    print()

    return [arg_name, arg_lr, arg_train_dts, arg_val_dts, arg_filter, arg_loss, arg_n_layer, arg_batch_size,
            arg_n_epochs, arg_dropout, arg_alpha_scale, arg_pretrained]


if __name__ == '__main__':
    args = arg_parser(sys.argv)  # Get the arguments from the command line

    name_of_attempt = args[0]                                   # String used to denominate the attempt.
    name_of_attempt = f"{str(date.today())}_{name_of_attempt}"  # Add the date to the name of the attempt
    fil_dir_size = args[4]                                      # Number of feature maps for the Direct_CNN model
    lr = args[1]                                                # Learning rate
    loss_fn = args[5]                                           # Loss function
    n_single_layer = args[6]                                    # Number of single layer networks
    n_epochs = args[8]                                          # Number of epochs
    dropout_rate = args[9]                                      # Dropout rate
    alpha_scale = args[10]                                      # Alpha loss scale
    pretrained_weights = args[11]                               # Pretrained weights

    # Training and validation data for dataset
    train_filename = f"./data/{args[2]}.h5"
    val_filename = f"./data/{args[3]}.h5"

    # Extract the patch size from the dts name
    dts_name = args[2]
    dts_name = dts_name.split("_")
    patch_str = [elm for elm in dts_name if "ps" in elm][0]

    # Training and test set generators
    fl_scale = True           # If True the normalization is performed
    P = int(patch_str[2:])    # Patch size
    dim_b = args[7]           # Batch dimension
    dim_t = 2000              # Number of bins in the transient dimension

    # Frequencies used by the iToF sensor
    if train_filename[-10:-3] == "stdfreq":
        freqs = np.array((20e06, 50e06, 60e06), dtype=np.float32)
    elif train_filename[-12:-3] == "multifreq":
        freqs = np.array(range(int(20e06), int(420e06), int(20e06)), dtype=np.float32)
    else:
        freqs = np.array((20e06, 50e06, 60e06), dtype=np.float32)

    # Put the loaded data in the right format for the network.
    train_loader = DataLoader.DataLoader(filename=train_filename,
                                         freqs=freqs,
                                         dim_batch=dim_b,
                                         fl_scale=fl_scale,
                                         P=P)
    val_loader = DataLoader.DataLoader(filename=val_filename,
                                       freqs=freqs,
                                       dim_batch=dim_b,
                                       fl_scale=fl_scale,
                                       P=P)

    # Prepare the main model
    net = PredictiveModel.PredictiveModel(name=name_of_attempt, dim_b=dim_b, lr=lr, freqs=freqs, P=P,
                                          saves_path='./saves', dim_t=dim_t, fil_size=fil_dir_size,
                                          loss_name=loss_fn, single_layers=n_single_layer, dropout_rate=dropout_rate,
                                          alpha_loss_factor=alpha_scale)
    # Summaries of the various networks
    net.DirectCNN.summary()

    # Training loop
    net.training_loop(train_w_loader=train_loader, test_w_loader=val_loader, final_epochs=n_epochs, print_freq=1,
                      save_freq=25, pretrain_filenamev=pretrained_weights)
