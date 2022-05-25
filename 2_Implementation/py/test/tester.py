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
    arg_exp_time = None  # Argument that define the used exposure time
    arg_help = "{0} -i <input> -m <image> -o <output> -t <task> -e <exp_time>".format(argv[0])  # Help string

    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:t:e:", ["help", "input=", "output=", "task=", "exp_time="])  # Recover the passed options and arguments from the command line (if any)
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
        elif opt in ("-e", "--exp_time"):
            arg_exp_time = float(arg)  # Set the task

    print('Input path:', arg_in)
    print('Output path:', arg_out)
    if arg_exp_time is not None:
        print('Exposure time:', arg_exp_time)
    print()

    return [arg_in, arg_out, arg_task, arg_exp_time]


if __name__ == '__main__':
    arg_in, arg_out, arg_task, arg_exp_time = arg_parser(sys.argv)  # Recover the input and output folder from the console args

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
                     normalize=True)

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
                     normalize=False)

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    elif arg_task == "spot_bitmap":
        print(f"TASK: {arg_task}")
        start = time.time()

        ut.spot_bitmap_gen(file_path=arg_out / "spot_bitmap.png",
                           img_size=[640, 480],
                           spot_size=10)

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

        theta_row, theta_col, row_distance, col_distance = tr.theta_calculator(tr.extract_center_peak(images)[0][0], images[0].shape[1], images[0].shape[0], 0.01, 39.597755)

        tot_img = tr.total_img2(images=images,
                                out_path=arg_out / "total_image",
                                normalize=True)

        tr.save_plot(theta_row, theta_col, list(tot_img[int(tot_img.shape[1]/2), :, 0]), list(tot_img[:, int(tot_img.shape[1]/2), 0]), row_distance, col_distance, str(arg_out / "cross_section"))

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
