import getopt
import os
import sys
import modules.dataset_utils as dts
import modules.utilities as utils
from pathlib import Path
from time import time


"""
Generate the point cloud of a given scene
"""


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line
    (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_in = os.getcwd()  # Argument containing the input directory
    arg_out = os.getcwd()  # Argument containing the output directory
    arg_gt = 1  # Argument defining if the depth map will be masked with the gt or not
    arg_help = "{0} -i <input> -o <output> -g <gt>".format(argv[0])  # Help string

    try:
        # Recover the passed options and arguments from the command line (if any)
        opts, args = getopt.getopt(
            argv[1:], "hi:o:g", ["help", "input=", "output=", "gt="]
        )
    except getopt.GetoptError:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            arg_in = Path(arg)  # Set the input directory
        elif opt in ("-o", "--output"):
            arg_out = Path(arg)  # Set the output directory
        elif opt in ("-g", "--gt"):
            arg_gt = int(arg)

    print("Input folder: ", arg_in)
    print("Output folder: ", arg_out)
    print("GT: ", arg_gt)
    print()

    return [arg_in, arg_out, arg_gt]


if __name__ == "__main__":
    in_path, out_folder, gt = arg_parser(sys.argv)

    task = "mirror data reconstruction"

    print(f"TASK: {task}")
    start = time()

    # Load the data
    data = utils.load_h5(utils.add_extension(in_path, ".h5"))

    if gt == 1:
        alpha = data["alpha_map_gt"]
    else:
        alpha = data["alpha_map"]

    obj = dts.utils.build_point_cloud(
        data=data["depth_map"],
        alpha=alpha,
        fov=60,
        img_size=(320, 240),
        out_path=out_folder,
        f_mesh=False,
        visualize=True,
    )

    end = time()
    minutes, seconds = divmod(end - start, 60)
    hours, minutes = divmod(minutes, 60)
    print(f"Task <{task}> concluded in in %d:%02d:%02d\n" % (hours, minutes, seconds))
