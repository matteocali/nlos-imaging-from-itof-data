import os
import getopt
import sys
import time
import numpy as np
import modules.transient_utils as tr
import modules.utilities as ut
from pathlib import Path
from os.path import exists


"""
Scripts to analyze transient data
"""


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_in = os.getcwd()  # Argument containing the input directory
    arg_out = ""  # Argument containing the output directory
    arg_task = ""  # Argument that defines the function that will be used
    arg_exp_time = None  # Argument that defines the used exposure time
    arg_samples = None  # Argument that defines the number of samples used
    arg_pixel = [120, 160]  # Argument that defines the pixel used
    arg_help = "{0} -i <input> -o <output> -t <task> -e <exp_time> -f <fov> -r <rgb>, -s <samples>, -p <pixel>".format(
        argv[0]
    )  # Help string

    try:
        opts, args = getopt.getopt(
            argv[1:],
            "hi:o:t:e:s:p:",
            [
                "help",
                "input=",
                "output=",
                "task=",
                "exp_time=",
                "fov=",
                "rgb=",
                "samples=",
                "pixel=",
            ],
        )  # Recover the passed options and arguments from the command line (if any)
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
            arg_exp_time = float(arg)  # Set the exposure time
        elif opt in ("-s", "--samples"):
            arg_samples = int(arg)  # Set the normalization factor
        elif opt in ("-p", "--pixel"):
            arg_pixel = [
                int(arg.split(",")[0]),
                int(arg.split(",")[1]),
            ]  # Set the pixel

    print("Input path: ", arg_in)
    if arg_out != "":
        print("Output path: ", arg_out)
    if arg_exp_time is not None:
        print("Exposure time: ", arg_exp_time)
    if arg_samples is not None:
        print("Number of samples: ", arg_samples)
    if arg_pixel is not None:
        print("Pixel to consider:", arg_pixel)
    print()

    return [arg_in, arg_out, arg_task, arg_exp_time, arg_samples, arg_pixel]


if __name__ == "__main__":
    arg_in, arg_out, arg_task, arg_exp_time, arg_samples, arg_pixel = arg_parser(
        sys.argv
    )  # Recover the input and output folder from the console args

    if arg_task == "tr_video":
        print(f"TASK: {arg_task}")
        start = time.time()

        ut.create_folder(
            arg_out, "all"
        )  # Create the output folder if not already present
        images = tr.loader.transient_loader(
            img_path=arg_in,
            np_path=arg_out / "np_transient.npy",
            store=(not exists(arg_out / "np_transient.npy")),
        )  # Load the transient

        tr.tools.transient_video(
            images=images, out_path=arg_out, normalize=True
        )  # Generate the video

        end = time.time()
        print(
            f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2))
        )
    elif arg_task == "total_img":
        print(f"TASK: {arg_task}")
        start = time.time()

        images = tr.loader.transient_loader(
            img_path=arg_in,
            np_path=arg_out / "np_transient.npy",
            store=(not exists(arg_out / "np_transient.npy")),
        )  # Load the transient
        t = tr.tools.total_img(
            images=images, out_path=arg_out / "total_image", n_samples=arg_samples
        )

        end = time.time()
        print(
            f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2))
        )
    elif arg_task == "glb_tr_video":
        print(f"TASK: {arg_task}")
        start = time.time()

        images = tr.loader.transient_loader(
            img_path=arg_in,
            np_path=arg_out / "np_transient.npy",
            store=(not exists(arg_out / "np_transient.npy")),
        )  # Load the transient
        glb_images = tr.tools.rmv_first_reflection_img(
            images=images,
            file_path=arg_out / "glb_np_transient.npy",
            store=(not exists(arg_out / "glb_np_transient.npy")),
        )
        tr.tools.transient_video(
            images=np.copy(glb_images), out_path=arg_out, normalize=True
        )
        tr.tools.total_img(
            images=glb_images, out_path=arg_out / "total_image", n_samples=arg_samples
        )

        end = time.time()
        print(
            f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2))
        )
    elif arg_task == "hists":
        print(f"TASK: {arg_task}")
        start = time.time()

        ut.create_folder(arg_out, "all")
        images = tr.loader.transient_loader(
            img_path=arg_in,
            np_path=arg_out / "np_transient.npy",
            store=(not exists(arg_out / "np_transient.npy")),
        )  # Load the transient
        tr.plots.histo_plt(
            radiance=images[:, arg_pixel[1], arg_pixel[0], :],  # righe:colonne
            exp_time=arg_exp_time,
            interval=None,
            stem=False,
            file_path=arg_out / "transient_histograms.svg",
        )

        end = time.time()
        minutes, seconds = divmod(end - start, 60)
        hours, minutes = divmod(minutes, 60)
        print(
            f"Task <{arg_task}> concluded in in %d:%02d:%02d\n"
            % (hours, minutes, seconds)
        )
    elif arg_task == "hists_glb":
        print(f"TASK: {arg_task}")
        start = time.time()

        ut.create_folder(arg_out, "all")
        images = tr.loader.transient_loader(
            img_path=arg_in,
            np_path=arg_out / "np_transient.npy",
            store=(not exists(arg_out / "np_transient.npy")),
        )  # Load the transient
        glb_images = tr.tools.rmv_first_reflection_img(
            images=images,
            file_path=arg_out / "glb_np_transient.npy",
            store=(not exists(arg_out / "glb_np_transient.npy")),
        )
        tr.plots.histo_plt(
            radiance=glb_images[:, arg_pixel[1], arg_pixel[0], :],  # righe:colonne
            exp_time=arg_exp_time,
            interval=None,
            stem=False,
            file_path=arg_out / "transient_histograms.svg",
        )

        end = time.time()
        minutes, seconds = divmod(end - start, 60)
        hours, minutes = divmod(minutes, 60)
        print(
            f"Task <{arg_task}> concluded in in %d:%02d:%02d\n"
            % (hours, minutes, seconds)
        )
    else:
        print(
            "Wrong task provided\nPossibilities are: tr_video, total_img, glb_tr_video, hists, hists_glb"
        )
