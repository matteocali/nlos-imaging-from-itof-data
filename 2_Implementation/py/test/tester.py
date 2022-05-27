import numpy as np
from modules import transient_handler as tr, utilities as ut, mitsuba_tests as mt
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
    arg_img_size = None  # Argument that define the img resolution
    arg_spot_size = None  # Argument that define the size of the white spot in the bitmap
    arg_exp_time = None  # Argument that define the used exposure time
    arg_fov = None  # Argument that define the fov of the camera
    arg_help = "{0} -i <input> -o <output> -t <task> -r <img_resolution> -s <spot_size> -e <exp_time> -f <fov>".format(argv[0])  # Help string

    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:t:r:s:e:f:", ["help", "input=", "output=", "task=", "img_resolution=", "spot_size=", "exp_time=", "fov="])  # Recover the passed options and arguments from the command line (if any)
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
        elif opt in ("-r", "--img_resolution"):
            img_size = str(arg)  # Read the img size
            img_size = img_size.split(",")
            arg_img_size = (int(img_size[0]), int(img_size[1]))  # Set the image size
        elif opt in ("-s", "--spot_size"):
            arg_spot_size = int(arg)  # Set the spot size
        elif opt in ("-e", "--exp_time"):
            arg_exp_time = float(arg)  # Set the exposure time
        elif opt in ("-f", "--fov"):
            arg_fov = float(arg)  # Set the fov

    print('Input path: ', arg_in)
    if arg_out != "":
        print('Output path: ', arg_out)
    if arg_img_size is not None:
        print('Image size: ', arg_img_size)
    if arg_spot_size is not None:
        print('Spot size: ', arg_spot_size)
    if arg_exp_time is not None:
        print('Exposure time: ', arg_exp_time)
    if arg_fov is not None:
        print('Field of view: ', arg_fov)
    print()

    return [arg_in, arg_out, arg_task, arg_img_size, arg_spot_size, arg_exp_time, arg_fov]


if __name__ == '__main__':
    arg_in, arg_out, arg_task, arg_img_size, arg_spot_size, arg_exp_time, arg_fov = arg_parser(sys.argv)  # Recover the input and output folder from the console args

    if arg_task == "tr_video":
        print(f"TASK: {arg_task}")
        start = time.time()

        ut.create_folder(arg_out, "np_transient.npy")  # Create the output folder if not already present
        images = tr.transient_loader(img_path=arg_in,
                                     np_path=arg_out / "np_transient.npy",
                                     store=(not exists(arg_out / "np_transient.npy")))  # Load the transient

        tr.transient_video(images=images,
                           out_path=arg_out,
                           normalize=True)  # Generate the video

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    elif arg_task == "total_img":
        print(f"TASK: {arg_task}")
        start = time.time()

        images = tr.transient_loader(img_path=arg_in,
                                     np_path=arg_out / "np_transient.npy",
                                     store=(not exists(arg_out / "np_transient.npy")))  # Load the transient
        tr.total_img(images=images,
                     out_path=arg_out / "total_image",
                     normalization_factor=17290)

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    elif arg_task == "glb_tr_video":
        print(f"TASK: {arg_task}")
        start = time.time()

        images = tr.transient_loader(img_path=arg_in,
                                     np_path=arg_out / "np_transient.npy",
                                     store=(not exists(arg_out / "np_transient.npy")))  # Load the transient
        glb_images = tr.rmv_first_reflection(images=images,
                                             file_path=arg_out / "glb_np_transient.npy",
                                             store=(not exists(arg_out / "glb_np_transient.npy")))
        tr.transient_video(images=np.copy(glb_images),
                           out_path=arg_out,
                           normalize=True)
        tr.total_img(images=glb_images,
                     out_path=arg_out / "total_image",
                     normalization_factor=17290)

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    elif arg_task == "spot_bitmap":
        print(f"TASK: {arg_task}")
        start = time.time()

        ut.spot_bitmap_gen(file_path=arg_out / "spot_bitmap.png",
                           img_size=arg_img_size,
                           spot_size=arg_spot_size)

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    elif arg_task == "hists":
        print(f"TASK: {arg_task}")
        start = time.time()

        ut.create_folder(arg_out, "np_transient.npy")
        images = tr.transient_loader(img_path=arg_in,
                                     np_path=arg_out / "np_transient.npy",
                                     store=(not exists(arg_out / "np_transient.npy")))  # Load the transient
        tr.histo_plt(radiance=images[:, 80, 60, :],
                     exp_time=arg_exp_time,
                     file_path=arg_out / "transient_histograms.svg")

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    elif arg_task == "cross":
        print(f"TASK: {arg_task}")
        start = time.time()

        images = tr.transient_loader(img_path=arg_in,
                                     np_path=arg_out / "np_transient.npy",
                                     store=(not exists(arg_out / "np_transient.npy")))  # Load the transient
        tot_img = tr.total_img(images=images,
                               out_path=arg_out / "total_image",
                               normalization_factor=17290)
        mt.cross_section_tester(images=images[:, :, :, :-1],
                                tot_img=tot_img,
                                exp_time=arg_exp_time,
                                fov=arg_fov,
                                output_path=arg_out)

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
