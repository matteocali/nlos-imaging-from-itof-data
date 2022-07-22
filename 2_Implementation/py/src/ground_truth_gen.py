import getopt
import os
import sys
from pathlib import Path
from time import time

from modules.dataset_func import build_mirror_gt, load_dataset, fuse_dt_gt


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line
    (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_in_gt = ""  # Argument containing the input directory (gt)
    arg_in_dat = "" # Argument containing the input dataset
    arg_out_gt = os.getcwd()  # Argument containing the output directory (gt)
    arg_out_dat = os.getcwd()  # Argument containing the output directory (dataset)
    arg_out_final = os.getcwd()  # Argument containing the output directory (final)
    arg_type = ""  # Argument that define which type of ground truth will be generated
    arg_help = "{0} -g <ground> -i <input> -o <output> -d <dataset> -f <final> -t <type>".format(argv[0])  # Help string

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(argv[1:], "hg:i:o:d:f:t:", ["help", "ground=", "input=", "output=", "dataset=", "final=", "type="])
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


if __name__ == '__main__':
    in_folder_gt, in_folder_dat, out_folder_gt, out_folder_dat, final_folder, type_gt = arg_parser(sys.argv)

    if type_gt == "mirror":
        print(f"TASK: {type_gt}")
        start = time()

        if not out_folder_gt.exists():
            build_mirror_gt(gt_path=in_folder_gt, out_path=out_folder_gt, fov=60, exp_time=0.01)

        if not out_folder_dat.exists():
            load_dataset(d_path=in_folder_dat, out_path=out_folder_dat)

        try:
            fuse_dt_gt(gt_path=out_folder_gt, d_path=out_folder_dat, out_path=final_folder)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(2)

        end = time()
        print(f"Task <{type_gt}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    else:
        print("Wrong type provided\nPossibilities are: mirror")
