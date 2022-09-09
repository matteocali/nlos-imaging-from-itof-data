import numpy as np
import sys
sys.path.append("./src/") 
sys.path.append("../utils/") 
sys.path.append("../itraining/src/")
from src.fun_test_img_transient import test_img
from src.fun_test_synth import test_synth
import fnmatch
import glob
import os
import getopt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

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


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line
    (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_name = "test"   # Argument containing the name of the training attempt
    arg_lr = 1e-04       # Argument containing the learning rate
    arg_help = "{0} -n <name> -r <lr>".format(argv[0])  # Help string

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(argv[1:], "hn:r:", ["help", "name=", "lr="])
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-n", "--name"):
            arg_name = arg  # Set the attempt name
        elif opt in ("-r", "--lr"):
            arg_lr = float(arg)  # Set the learning rate

    print("Attempt name: ", arg_name)
    print("Learning rate: ", arg_lr)
    print()

    return [arg_name, arg_lr]


if __name__ == '__main__':
    args = arg_parser(sys.argv) # Get the arguments

    attempt_name = args[0]                                                                              # name of the stored approach weights
    lr = args[1]                                                                                        # kernel size for the convolutional layers
    win_server_path = "Z:/decaligm"                                                                     # path to the server
    win_server_path_2 = "Y:/matteo"                                                                     # path to the server
    git_folder_path = "thesis-nlos-for-itof/5_Tools/dl_nlos_reconstruction_mirror_v2"                   # path to the git folder
    dataset_folder = "datasets/mirror"                                                                  # path to the dataset folder
    data_path_real = "../../Datasets/S3S4S5/*"                                                          # path to the real images
    data_path_synth = f"{win_server_path_2}/{dataset_folder}/mirror_dts"                                # Path of the synthetic test set (same patch size as training and validation)
    processed_dts_folder = f"{win_server_path}/{git_folder_path}/training/data/val_balanced_dts_fixed_cam_n33500_ps11_nonorm.h5"                        # Path of the processed test set (same patch size as training and validation)
    #test_file_csv = f"{win_server_path}/{git_folder_path}/dataset_creation/data_split/test_images.csv"  # path to the test file
    test_file_csv = f"{win_server_path}/{git_folder_path}/dataset_creation/data_split/val_images.csv"
    #weights_folder = f"../training/saves/{attempt_name}/checkpoints/"                                  # path to the weights
    weights_folder = f"{win_server_path}/{git_folder_path}/training/saves/{attempt_name}/checkpoints/"  # path to the weights
    dim_t = 2000                                                                                        # number of bins in the transient dimension
    P = 11                                                                                              # patch size
    flag_norm_perpixel = True                                                                           # normalization per pixel
    flag_scale = True                                                                                   # whether to apply scaling on the inputs
    flag_plot = False                                                                                   # whether to plot and save the results
    flag_epoch = False                                                                                  # whether to test on a specific epoch
    fl_test_img = True
    fil_direct = 32
    num_epoch = 40000                                                                                   # epoch to test on
    epoch_name_d = attempt_name + "_d_e" + str(num_epoch) + "_weights.h5"                               # name of the epoch to test on for the spatial net
    epoch_name_v = attempt_name + "_v_e" + str(num_epoch) + "_weights.h5"                               # name of the epoch to test on for the direct net

    # Check if the iToF data uses two or three frequencies and set their value accordingly
    str_freqs = ""
    if attempt_name[-5:] == "2freq":
        freqs = np.array((20e06, 50e06), dtype=np.float32)
        str_freqs = "_2freq"
    elif attempt_name[-9:] == "multifrew":
        freqs = np.array((20e06, 50e06, 80e06), dtype=np.float32)
        str_freqs = "_multifreq"
    else:
        freqs = np.array((20e06, 50e06, 60e06), dtype=np.float32)

    out_path = f"./out/{attempt_name}"  # path to the output folder

    # Create needed directories
    if not os.path.exists(out_path):
        os.mkdir(out_path)

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
    names.sort()

    if fl_test_img:
        data_path = data_path_synth
    else:
        data_path = data_path_real

    test_type = "patch"
    if test_type == "img":
        test_img(weight_names=weight_names,
                 data_path=data_path,
                 out_path=out_path,
                 test_files=test_file_csv,
                 P=P,
                 lr=lr,
                 freqs=freqs,
                 fl_scale=flag_scale,
                 fl_norm_perpixel=flag_norm_perpixel,
                 fil_dir=fil_direct,
                 dim_t=dim_t,
                 plot_results=True)  # test on transient images
    elif test_type == "patch":
        test_synth(fl_test_img=False,
                   processed_dts_path=processed_dts_folder,
                   test_files=test_file_csv,
                   dts_path=data_path,
                   weight_names=weight_names,
                   P=P,
                   freqs=freqs,
                   lr=lr,
                   fl_scale=flag_scale,
                   fil_dir=fil_direct,
                   dim_t=dim_t)
