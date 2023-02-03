import os
import sys
import h5py
import numpy as np
import glob
import csv
import time
import getopt
sys.path.append("./src")
sys.path.append("../utils")
from fun_acquire_pixels import acquire_pixels, acquire_pixels_test


"""
Script for preparing training, validation and test set based on transient data.
For more details of the data used see fun_acquire_pixels where most of the computations are performed

The data is saved in the "data" folder under the training directory

Flags and variables:
    -out_dir:                   path to the output directory (by default the "data" folder in the training directory)
    -data_dir:                  location of the transient dataset
    -flag_new_shuffle:          Set to True to get a new random split of the dataset images. Otherwise the csv files in ./data_split/ will be used
    -n_patches:                 Number of patches taken from each images
    -s:                         Patch size
    -max_imgs:                  Maximum number of images of each set. Can be used to make a quick run of the process. Otherwise set it to high value (higher than max dataset size)
    -train_slice:               Percentage of images belonging to training dataset expressed as a number between 0 and 1. Test and validation will take each half of what remains
    -fl_get_train:              If set to True we will acquire the training set
    -fl_get_val:                If set to True we will acquire the validation set
    -fl_get_test:               If set to True we will acquire the test set
"""

def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line
    (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_name = "dts"                    # Argument containing the name of the dataset
    arg_data_path = ""                  # Argument containing the path where the raw data are located
    arg_out_path = "../training/data/"  # Argument containing the path of the output folder
    arg_shuffle = True                  # Argument containing the flag for shuffling the dataset
    arg_n_patches = 500                 # Argument containing the number of patches to be extracted from each image
    arg_patch_size = 11                 # Argument containing the patch size
    arg_groups = 111                    # Argument containing the groups to be processed
    arg_freq_type = "std"               # Argument containing the frequency type to be processed
    arg_help = "{0} -n <name> -i <input> -o <output> -s <shuffle> -p <patch> -d <n_patches> -g <groups> " \
               "-f <frequencies_set>".format(argv[0])      # Help string

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(argv[1:], "hn:i:o:s:p:d:g:f:", ["help", "name=", "input=", "output=", "shuffle=",
                                                                   "patch=", "n_patches=", "groups=", "freq_type="])
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-n", "--name"):
            arg_name = arg  # Set the attempt name
        elif opt in ("-i", "--input"):
            arg_data_path = arg  # Set the path to the raw data
            if arg_data_path[-1] != "/":
                arg_data_path += "/"
        elif opt in ("-o", "--output"):
            arg_out_path = arg  # Set the output path
            if arg_out_path[-1] != "/":
                arg_out_path += "/"
        elif opt in ("-s", "--shuffle"):
            if arg == "True":
                arg_shuffle = True
            else:
                arg_shuffle = False
        elif opt in ("-p", "--patch"):
            arg_patch_size = int(arg)
        elif opt in ("-d", "--n_patches"):
            arg_n_patches = int(arg)
        elif opt in ("-g", "--groups"):
            arg_groups = arg
            if arg_groups != "100" and arg_groups != "010" and arg_groups != "001" and arg_groups != "111":
                arg_groups = "111"
        elif opt in ("-f", "--frequencies"):
            arg_freq_type = arg
            if arg_freq_type != "std" and arg_freq_type != "multi":
                arg_freq_type = "std"
    print("Attempt name: ", arg_name)
    print("Input folder: ", arg_data_path)
    print("Output folder: ", arg_out_path)
    print("Shuffle: ", arg_shuffle)
    print("Patch size: ", arg_patch_size)
    print("Number of patches: ", arg_n_patches)
    print("Groups: ", arg_groups)
    print("Frequencies: ", arg_freq_type)
    print()

    return [arg_name, arg_data_path, arg_out_path, arg_shuffle, arg_patch_size, arg_n_patches, arg_groups, arg_freq_type]


if __name__ == '__main__':
    start_time = time.time()
    args = arg_parser(sys.argv)

    # Get the arguments from the command line
    dataset_name = args[0]  # Name of the dataset
    data_dir = args[1]      # Path to the raw data
    out_dir = args[2]       # Path to the output folder

    if not os.path.exists(out_dir):  # If the output folder does not exist, create it
        os.mkdir(out_dir)

    # Flags and variables
    np.random.seed(2019283)                      # set the random seed
    n_patches = args[5]                          # number of pixels taken from each image
    s = args[4]                                  # size of each input patch
    add_str = f"np_{n_patches}"                  # string to add to the output file name
    add_str += f"_ps{s}"                         # part of the dataset name containing the patch size
    max_imgs = 1000                              # maximum number of images to be used (if grater than the actual number, all the dataset will be used)
    train_slice = 0.6                            # percentage of images belonging to training dataset
    val_slice = round((1 - train_slice) / 2, 1)  # percentage of images belonging to validation dataset
    freq_type = args[7]                          # type of frequencies to be used

    # Frequencies used by the iToF sensor
    if freq_type == "std":
        freqs = np.array((20e06, 50e06, 60e06), dtype=np.float32)
        add_str += "_stdfreq"
    elif freq_type == "multi":
        freqs = np.array(range(int(20e06), int(420e06), int(20e06)), dtype=np.float32)
        add_str += "_multifreq"
    else:
        freqs = np.array((20e06, 50e06, 60e06), dtype=np.float32)
        add_str += "_stdfreq"

    # Choose which datasets you want to build
    flag_new_shuffle = args[3]  # whether to shuffle again the images and create new training validation and test datasets
    fl_get_train = False  # build the training set
    fl_get_val = False    # build the validation set
    fl_get_test = False   # build the test set
    groups = args[6]            # which groups to be processed
    if groups[0] == "1":
        fl_get_train = True
    if groups[1] == "1":
        fl_get_val = True
    if groups[2] == "1":
        fl_get_test = True

    # Grab all the names of the images of the dataset, shuffle them and save them in a csv file

    if flag_new_shuffle:
        # Load all the images of the dataset
        path = data_dir + "*.h5"     # path to the transient dataset
        filenames = glob.glob(path)  # get all the names of the files

        # Shuffle the images
        num_img = len(filenames)                  # number of images in the dataset
        indexes = np.arange(num_img)              # create an array with the indexes of the images
        indexes = np.random.permutation(indexes)  # shuffle the indexes
        indexes = np.array(indexes)               # convert the indexes to an array
        filenames = np.array(filenames)           # convert the names to an array
        filenames = filenames[indexes]            # shuffle the names according to the indexes

        for i, file in enumerate(filenames):
            filenames[i] = file.replace("\\", "/")  # replace the backslash by the slash for the correct path

        with open("shuffled_images.csv", "w", newline="") as csvfile:  # save the names of the images in a csv file
            wr = csv.writer(csvfile)
            for filename in filenames:
                wr.writerow([filename])

        images_list = []
        with open("shuffled_images.csv", "r") as csvfile:  # load the names of the images from the csv file
            wr = csv.reader(csvfile)
            for row in wr:
                images_list.append(row)

        images_list = [item for sublist in images_list for item in sublist]  # flatten the list of lists

        # Split the images in training, validation and test sets
        num_img = len(images_list)                      # number of images in the dataset
        n_train = int(np.round(train_slice * num_img))  # number of images in the training set
        n_val = int(np.round(val_slice * num_img))      # number of images in the validation set
        n_test = num_img - n_train - n_val              # number of images in the test set
        train_files = images_list[:n_train]             # names of the images in the training set
        val_files = images_list[n_train:n_train+n_val]  # names of the images in the validation set
        test_files = images_list[n_train+n_val:]        # names of the images in the test set
        np.random.shuffle(test_files)                   # shuffle the test set
        np.random.shuffle(val_files)                    # shuffle the validation set
        np.random.shuffle(train_files)                  # shuffle the training set

        with open("./data_split/train_images.csv", "w", newline="") as csvfile:  # save the names of the images in the train_dataset in a csv file
            wr = csv.writer(csvfile)
            for filename in train_files:
                wr.writerow([os.path.basename(filename)])
        with open("./data_split/val_images.csv", "w", newline="") as csvfile:  # save the names of the images in the val_dataset in a csv file
            wr = csv.writer(csvfile)
            for filename in val_files:
                wr.writerow([os.path.basename(filename)])
        with open("./data_split/test_images.csv", "w", newline="") as csvfile:  # save the names of the images in the test_dataset in a csv file
            wr = csv.writer(csvfile)
            for filename in test_files:
                wr.writerow([os.path.basename(filename)])

    # Load the names of the images from the csv file, considering the split in training, validation and test sets
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

    N_train = len(train_files)  # number of images in the training set
    N_val = len(val_files)      # number of images in the validation set
    N_test = len(test_files)    # number of images in the test set

    print(f"Training set size: {N_train}")
    print(f"Validation set size: {N_val}")
    print(f"Test set size: {N_test}\n")

    # TRAINING IMAGES
    if fl_get_train:
        train_files = np.asarray(train_files)
        train_files = [data_dir + file for file in train_files]
        print("Training dataset:")
        v_real, gt_depth_real, gt_alpha_real = acquire_pixels(images=train_files,
                                                              num_pixels=n_patches,
                                                              max_img=max_imgs,
                                                              s=s,
                                                              freqs=freqs)

        num_elem = v_real.shape[0]
        file_train = f"{out_dir}train_{dataset_name}_n{str(num_elem)}{add_str}.h5"
        with h5py.File(file_train, 'w') as f:
            f.create_dataset("name", data=dataset_name)
            f.create_dataset("gt_depth", data=gt_depth_real)
            f.create_dataset("gt_alpha", data=gt_alpha_real)
            f.create_dataset("raw_itof", data=v_real)
            f.create_dataset("freqs", data=freqs)

    # VALIDATION IMAGES
    if fl_get_val:
        val_files = np.asarray(val_files)
        val_files = [data_dir + fil for fil in val_files]
        print("\nValidation dataset:")
        v_real, gt_depth_real, gt_alpha_real = acquire_pixels(images=val_files,
                                                              num_pixels=n_patches,
                                                              max_img=max_imgs,
                                                              s=s,
                                                              freqs=freqs)

        num_elem = v_real.shape[0]
        file_val = f"{out_dir}val_{dataset_name}_n{str(num_elem)}{add_str}.h5"
        with h5py.File(file_val, 'w') as f:
            f.create_dataset("name", data=dataset_name)
            f.create_dataset("gt_depth", data=gt_depth_real)
            f.create_dataset("gt_alpha", data=gt_alpha_real)
            f.create_dataset("raw_itof", data=v_real)
            f.create_dataset("freqs", data=freqs)

    # TEST IMAGES
    if fl_get_test:
        test_files = np.asarray(test_files)
        test_files = [data_dir + fil for fil in test_files]
        print("\nTest dataset:")
        gt_depth_real, gt_alpha_real, v_real, names = acquire_pixels_test(images=test_files,
                                                                          max_img=max_imgs,
                                                                          s=s,
                                                                          freqs=freqs)

        num_elem = v_real.shape[0]
        file_test = f"{out_dir}test_{dataset_name}_n{str(num_elem)}{add_str}.h5"
        names = [n.encode("ascii", "ignore") for n in names]
        with h5py.File(file_test, 'w') as f:
            f.create_dataset("name", data=dataset_name)
            f.create_dataset("names", data=names)
            f.create_dataset("gt_depth", data=gt_depth_real)
            f.create_dataset("gt_alpha", data=gt_alpha_real)
            f.create_dataset("raw_itof", data=v_real)
            f.create_dataset("freqs", data=freqs)

    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    hours, minutes = divmod(minutes, 60)
    print("\nTotal time to build the whole dataset is %d:%02d:%02d" % (hours, minutes, seconds))
