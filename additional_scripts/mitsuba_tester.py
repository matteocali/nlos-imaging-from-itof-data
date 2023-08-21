import getopt
import sys
import time
import os
import modules.transient_utils as tr
import modules.mitsuba_utils as mt
from pathlib import Path
from os.path import exists


"""
Script used to test the mitsuba renderer
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
    arg_fov = None  # Argument that defines the fov of the camera
    arg_rgb = None  # Argument that defines the path where the rgb render is located
    arg_samples = None  # Argument that defines the number of samples used
    # For a 320x240px image with 10k samples, 1k beans and 0.01 exp --> norm_factor = 17290
    # For a 320x240px image with 5k samples, 1k beans and 0.01 exp --> norm_factor = 8645.888
    # For a 320x240px image with 10k samples, 1.5k beans and 0.01 exp --> norm_factor = 17291.414
    # For a 320x240px image with 10k samples, 1k beans and 0.02 exp --> norm_factor = 17290.703
    # --> norm_factor = n_samples * 1.7291
    arg_diff_limits = None  # Argument that defines the min and max interval of the difference colorbar
    arg_ratio_limits = (
        None  # Argument that defines the min and max interval of the ratio colorbar
    )
    arg_help = "{0} -i <input> -o <output> -t <task> -e <exp_time> -f <fov> -r <rgb>, -s <samples>, -d <diff>, -l <ratio>".format(
        argv[0]
    )  # Help string

    try:
        opts, args = getopt.getopt(
            argv[1:],
            "hi:o:t:e:f:r:s:d:l:",
            [
                "help",
                "input=",
                "output=",
                "task=",
                "exp_time=",
                "fov=",
                "rgb=",
                "samples=",
                "diff=",
                "ratio=",
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
        elif opt in ("-f", "--fov"):
            arg_fov = float(arg)  # Set the fov
        elif opt in ("-r", "--rgb"):
            arg_rgb = Path(arg)  # Set the rgb image path location
        elif opt in ("-s", "--samples"):
            arg_samples = int(arg)  # Set the normalization factor
        elif opt in ("-d", "--diff"):
            limits = str(arg)  # Read the img size
            limits = limits.split(",")
            arg_diff_limits = (float(limits[0]), float(limits[1]))  # Set the image size
        elif opt in ("-l", "--ratio"):
            limits = str(arg)  # Read the img size
            limits = limits.split(",")
            arg_ratio_limits = (
                float(limits[0]),
                float(limits[1]),
            )  # Set the image size

    print("Input path: ", arg_in)
    if arg_out != "":
        print("Output path: ", arg_out)
    if arg_exp_time is not None:
        print("Exposure time: ", arg_exp_time)
    if arg_fov is not None:
        print("Field of view: ", arg_fov)
    if arg_rgb is not None:
        print("RGB render path: ", arg_rgb)
    if arg_samples is not None:
        print("Number of samples: ", arg_samples)
    if arg_diff_limits is not None:
        print("Difference colorbar interval: ", arg_diff_limits)
    if arg_ratio_limits is not None:
        print("Ratio colorbar interval: ", arg_ratio_limits)
    print()

    return [
        arg_in,
        arg_out,
        arg_task,
        arg_exp_time,
        arg_fov,
        arg_rgb,
        arg_samples,
        arg_diff_limits,
        arg_ratio_limits,
    ]


if __name__ == "__main__":
    (
        arg_in,
        arg_out,
        arg_task,
        arg_exp_time,
        arg_fov,
        arg_rgb,
        arg_samples,
        arg_diff_limits,
        arg_ratio_limits,
    ) = arg_parser(
        sys.argv
    )  # Recover the input and output folder from the console args

    if arg_task == "cross":
        print(f"TASK: {arg_task}\n")
        start = time.time()

        images = tr.loader.transient_loader(
            img_path=arg_in,
            np_path=arg_out / "np_transient.npy",
            store=(not exists(arg_out / "np_transient.npy")),
        )  # Load the transient
        tot_img = tr.tools.total_img(
            images=images, out_path=None, n_samples=arg_samples
        )
        mt.tools.cross_section_tester(
            images=images[:, :, :, :-1],
            tot_img=tot_img,
            exp_time=arg_exp_time,
            fov=arg_fov,
            output_path=arg_out,
        )

        end = time.time()
        minutes, seconds = divmod(end - start, 60)
        hours, minutes = divmod(minutes, 60)
        print(
            f"Task <{arg_task}> concluded in in %d:%02d:%02d\n"
            % (hours, minutes, seconds)
        )
    elif arg_task == "distance_plot":
        print(f"TASK: {arg_task}\n")
        start = time.time()

        mt.plots.distance_plot(in_path=arg_in, out_name=arg_out)

        end = time.time()
        minutes, seconds = divmod(end - start, 60)
        hours, minutes = divmod(minutes, 60)
        print(
            f"Task <{arg_task}> concluded in in %d:%02d:%02d\n"
            % (hours, minutes, seconds)
        )
    elif arg_task == "mm_distance_plot":
        print(f"TASK: {arg_task}\n")
        start = time.time()

        mt.plots.mm_distance_plot(
            in_path=arg_in, step=1, max_value=30, out_name=arg_out
        )

        end = time.time()
        minutes, seconds = divmod(end - start, 60)
        hours, minutes = divmod(minutes, 60)
        print(
            f"Task <{arg_task}> concluded in in %d:%02d:%02d\n"
            % (hours, minutes, seconds)
        )
    elif arg_task == "tot_img_test":
        print(f"TASK: {arg_task}\n")
        start = time.time()

        images = tr.loader.transient_loader(
            img_path=arg_in,
            np_path=arg_out / "np_transient.npy",
            store=(not exists(arg_out / "np_transient.npy")),
        )  # Load the transient
        tot_img = tr.tools.total_img(
            images=images, out_path=arg_out / "total_image", n_samples=arg_samples
        )
        mt.tools.tot_img_tester(
            rgb_img_path=arg_rgb,
            total_img=tot_img,
            out_path=arg_out,
            diff_limits=arg_diff_limits,
            ratio_limits=arg_ratio_limits,
        )

        end = time.time()
        minutes, seconds = divmod(end - start, 60)
        hours, minutes = divmod(minutes, 60)
        print(
            f"Task <{arg_task}> concluded in in %d:%02d:%02d\n"
            % (hours, minutes, seconds)
        )
    elif arg_task == "norm_factor":
        print(f"TASK: {arg_task}\n")
        start = time.time()

        images = tr.loader.transient_loader(
            img_path=arg_in,
            np_path=arg_out / "np_transient.npy",
            store=(not exists(arg_out / "np_transient.npy")),
        )  # Load the transient
        tot_img = tr.tools.total_img(images=images)
        mt.utils.compute_norm_factor(tot_img=tot_img, o_img_path=arg_rgb)

        end = time.time()
        minutes, seconds = divmod(end - start, 60)
        hours, minutes = divmod(minutes, 60)
        print(
            f"Task <{arg_task}> concluded in in %d:%02d:%02d\n"
            % (hours, minutes, seconds)
        )
    elif arg_task == "plot_norm_factor":
        print(f"TASK: {arg_task}\n")
        start = time.time()

        mt.plots.plot_norm_factor(
            folder_path=arg_in, rgb_path=arg_rgb, out_path=arg_out
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
            "Wrong task provided\nPossibilities are: cross, distance_plot, mm_distance_plot, tot_img_test, norm_factor, plot_norm_factor"
        )
