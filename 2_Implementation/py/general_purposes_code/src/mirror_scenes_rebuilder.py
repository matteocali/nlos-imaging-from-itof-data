import getopt
import os
import sys
from pathlib import Path
from time import time

from modules.dataset_func import build_point_cloud
from modules.utilities import load_h5, add_extension


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line
    (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_in = os.getcwd()  # Argument containing the input directory
    arg_out = os.getcwd()  # Argument containing the output directory
    arg_help = "{0} -i <input> -o <output>".format(argv[0])  # Help string

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(argv[1:], "hi:o:", ["help", "input=", "output="])
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            arg_in = Path(arg)  # Set the input directory
        elif opt in ("-o", "--output"):
            arg_out = Path(arg)  # Set the output directory

    print("Input folder: ", arg_in)
    print("Output folder: ", arg_out)
    print()

    return [arg_in, arg_out]


if __name__ == '__main__':
    in_path, out_folder = arg_parser(sys.argv)

    task = "mirror data reconstruction"

    print(f"TASK: {task}")
    start = time()

    # Load the data
    data = load_h5(add_extension(in_path, ".h5"))

    obj = build_point_cloud(data=data['depth_map'],
                            alpha=data['alpha_map_gt'],
                            fov=60,
                            img_size=(320, 240),
                            out_path=out_folder,
                            f_mesh=False,
                            visualize=True)

    end = time()
    minutes, seconds = divmod(end - start, 60)
    hours, minutes = divmod(minutes, 60)
    print(f"Task <{task}> concluded in in %d:%02d:%02d\n" % (hours, minutes, seconds))
