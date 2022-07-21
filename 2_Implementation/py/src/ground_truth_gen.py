import getopt
import os
import sys
from pathlib import Path
from time import time

from modules.dataset_func import load_gt_data


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line
    (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_in = ""  # Argument containing the input directory
    arg_out = os.getcwd()  # Argument containing the output directory
    arg_type = ""  # Argument that define which type of ground truth will be generated
    arg_help = "{0} -i <input> -o <output> -t <type>".format(argv[0])  # Help string

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(argv[1:], "hi:o:t:", ["help", "input=", "output=", "type="])
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # Print the help message
            sys.exit(2)
        elif opt in ("-i", "--input"):
            arg_in = Path(arg)  # Set the input directory
        elif opt in ("-o", "--output"):
            arg_out = Path(arg)  # Set the output directory
        elif opt in ("-t", "--type"):
            arg_type = arg  # Set the task

    print("Input folder: ", arg_in)
    print("Output folder: ", arg_out)
    print("Type of ground truth: ", arg_type)
    print()

    return [arg_in, arg_out, arg_type]


if __name__ == '__main__':
    in_folder, out_folder, type_gt = arg_parser(sys.argv)

    if type_gt == "mirror":
        print(f"TASK: {type_gt}")
        start = time()

        # Generate the ground truth for the mirror task
        load_gt_data(in_folder, out_folder)

        end = time()
        print(f"Task <{type_gt}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    else:
        print("Wrong type provided\nPossibilities are: mirror")
