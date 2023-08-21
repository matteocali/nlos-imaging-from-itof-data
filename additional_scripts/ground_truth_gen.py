import getopt
import os
import sys
import numpy as np
import modules.dataset_utils as dts
from pathlib import Path
from time import time


"""
Preprocess the dataset to link together the data with the corresponding ground truth
"""


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line
    (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_in_gt = ""  # Argument containing the input directory (gt)
    arg_in_dat = ""  # Argument containing the input dataset
    arg_out_gt = os.getcwd()  # Argument containing the output directory (gt)
    arg_out_dat = os.getcwd()  # Argument containing the output directory (dataset)
    arg_out_final = os.getcwd()  # Argument containing the output directory (final)
    arg_type = ""  # Argument that define which type of ground truth will be generated
    arg_help = "{0} -g <ground> -i <input> -o <output> -d <dataset> -f <final> -t <type>".format(
        argv[0]
    )  # Help string

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(
            argv[1:],
            "hg:i:o:d:f:t:",
            ["help", "ground=", "input=", "output=", "dataset=", "final=", "type="],
        )
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # Print the help message
            sys.exit(2)
        elif opt in ("-g", "--ground"):
            arg_in_gt = Path(arg)  # Set the input directory (gt)
        elif opt in ("-i", "--input"):
            arg_in_dat = Path(arg)  # Set the input directory (dataset)
        elif opt in ("-o", "--output"):
            arg_out_gt = Path(arg)  # Set the output directory (gt)
        elif opt in ("-d", "--dataset"):
            arg_out_dat = Path(arg)  # Set the output directory (dataset)
        elif opt in ("-t", "--type"):
            arg_type = arg  # Set the task
        elif opt in ("-f", "--final"):
            arg_out_final = Path(arg)

    print("Input folder (gt): ", arg_in_gt)
    print("Input folder (dataset): ", arg_in_dat)
    print("Output folder (gt): ", arg_out_gt)
    print("Output folder (dataset): ", arg_out_dat)
    print("Output folder (final): ", arg_out_final)
    print("Type of ground truth: ", arg_type)
    print()

    return [arg_in_gt, arg_in_dat, arg_out_gt, arg_out_dat, arg_out_final, arg_type]


if __name__ == "__main__":
    (
        in_folder_gt,
        in_folder_dat,
        out_folder_gt,
        out_folder_dat,
        final_folder,
        type_gt,
    ) = arg_parser(sys.argv)

    print(f"TASK: {type_gt}")
    start = time()
    if type_gt == "mirror":
        if not out_folder_gt.exists():
            dts.gt_mirror.build_mirror_gt(
                gt_path=in_folder_gt, out_path=out_folder_gt, fov=60, exp_time=0.01
            )
    elif type_gt == "fermat":
        if not out_folder_gt.exists():
            dts.gt_fermat.build_fermat_gt(
                gt_path=in_folder_gt,
                out_path=out_folder_gt,
                exp_time=0.01,
                fov=60,
                img_size=[320, 240],
                grid_size=[32, 24],
            )
    else:
        print("Wrong type provided\nPossibilities are: mirror, fermat")
        sys.exit(2)

    if not out_folder_dat.exists():
        dts.utils.load_dataset(
            d_path=in_folder_dat,
            out_path=out_folder_dat,
            freqs=np.array((20e06, 50e06, 60e06), dtype=np.float32),
        )

    try:
        if type_gt == "mirror":
            dts.gt_mirror.fuse_dt_gt_mirror(
                d_path=out_folder_dat,
                gt_path=out_folder_gt,
                out_path=final_folder,
                def_obj_pos=[0.9, 1.0, 1.65],
            )
        elif type_gt == "fermat":
            dts.gt_fermat.fuse_dt_gt_fermat(
                d_path=out_folder_dat,
                gt_path=out_folder_gt,
                out_path=final_folder,
                img_size=[320, 240],
                grid_size=[32, 24],
            )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(2)

    end = time()
    minutes, seconds = divmod(end - start, 60)
    hours, minutes = divmod(minutes, 60)
    print(
        f"Task <build gt (type {type_gt})> concluded in in %d:%02d:%02d\n"
        % (hours, minutes, seconds)
    )
    sys.exit(0)
