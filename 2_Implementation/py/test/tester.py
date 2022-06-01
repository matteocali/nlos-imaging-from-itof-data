from modules import utilities as ut
from pathlib import Path
import os
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
    arg_task = ""  # Argument that defines the function that will be used
    arg_img_size = None  # Argument that defines the img resolution
    arg_spot_size = None  # Argument that defines the size of the white spot in the bitmap
    arg_help = "{0} -i <input> -o <output> -t <task> -m <img_resolution> -s <spot_size>".format(argv[0])  # Help string

    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:t:m:s:", ["help", "input=", "output=", "task=", "img_resolution=", "spot_size="])  # Recover the passed options and arguments from the command line (if any)
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
        elif opt in ("-m", "--img_resolution"):
            img_size = str(arg)  # Read the img size
            img_size = img_size.split(",")
            arg_img_size = (int(img_size[0]), int(img_size[1]))  # Set the image size
        elif opt in ("-s", "--spot_size"):
            arg_spot_size = int(arg)  # Set the spot size

    print('Input path: ', arg_in)
    if arg_out != "":
        print('Output path: ', arg_out)
    if arg_img_size is not None:
        print('Image size: ', arg_img_size)
    if arg_spot_size is not None:
        print('Spot size: ', arg_spot_size)
    print()

    return [arg_in, arg_out, arg_task, arg_img_size, arg_spot_size]


if __name__ == '__main__':
    arg_in, arg_out, arg_task, arg_img_size, arg_spot_size = arg_parser(sys.argv)  # Recover the input and output folder from the console args

    if arg_task == "spot_bitmap":
        print(f"TASK: {arg_task}")
        start = time.time()

        ut.spot_bitmap_gen(file_path=arg_out / "spot_bitmap.png",
                           img_size=arg_img_size,
                           spot_size=arg_spot_size)

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    elif arg_task == "test":
        from modules import exr_handler as exr
        img = exr.load_exr(arg_in)
    else:
        print("Wrong task provided\nPossibilities are: spot_bitmap")
