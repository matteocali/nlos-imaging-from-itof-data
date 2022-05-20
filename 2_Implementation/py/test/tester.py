import numpy as np
from modules import transient_handler as tr, utilities as ut
from pathlib import Path
import os
from os.path import exists
import getopt
import sys
import time


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_in = os.getcwd()  # Argument containing the input directory
    arg_out = ""  # Argument containing the output directory
    arg_task = ""  # Argument that define the function that will be used
    arg_help = "{0} -i <input> -m <image> -o <output>".format(argv[0])  # Help string

    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:t:", ["help", "input=", "output=", "task="])  # Recover the passed options and arguments from the command line (if any)
    except:
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
        elif opt in ("-t", "--task"):
            arg_task = arg  # Set the task

    print('Input path:', arg_in)
    print('Output path:', arg_out)
    print()

    return [arg_in, arg_out, arg_task]


if __name__ == '__main__':
    arg_in, arg_out, arg_task = arg_parser(sys.argv)  # Recover the input and output folder from the console args

    if arg_task == "tr_video":
        print(f"TASK: {arg_task}")
        start = time.time()

        ut.create_folder(arg_out, "np_transient.npy")  # Create the output folder if not already present
        images = tr.transient_loader(img_path=arg_in, np_path=arg_out / "np_transient.npy", store=(not exists(arg_out / "np_transient.npy")))  # Load the transient

        tr.transient_video(images, arg_out, normalize=True)  # Generate the video

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    elif arg_task == "total_img":
        images = tr.transient_loader(img_path=arg_in, np_path=arg_out / "np_transient.npy", store=(not exists(arg_out / "np_transient.npy")))  # Load the transient
        tr.total_img(images, arg_out / "total_image", normalize=False)
    elif arg_task == "glb_tr_video":
        print(f"TASK: {arg_task}")
        start = time.time()

        images = tr.transient_loader(img_path=arg_in, np_path=arg_out / "np_transient.npy", store=(not exists(arg_out / "np_transient.npy")))  # Load the transient
        glb_images = tr.rmv_first_reflection(images=images, file_path=arg_out / "glb_np_transient.npy", store=(not exists(arg_out / "glb_np_transient.npy")))
        tr.transient_video(np.copy(glb_images), arg_out, normalize=True)
        tr.total_img(glb_images, arg_out / "total_image", normalize=False)

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    elif arg_task == "spot_bitmap":
        ut.spot_bitmap_gen(arg_out / "spot_bitmap.png", [640, 480], 10)
