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
    arg_task = ""  # Argument that defines the function that will be used
    arg_exp_time = None  # Argument that defines the used exposure time
    arg_samples = None  # Argument that defines the number of samples used
    arg_help = "{0} -i <input> -o <output> -t <task> -e <exp_time> -f <fov> -r <rgb>, -s <samples>".format(argv[0])  # Help string

    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:t:e:s:", ["help", "input=", "output=", "task=", "exp_time=", "fov=", "rgb=", "samples="])  # Recover the passed options and arguments from the command line (if any)
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

    print('Input path: ', arg_in)
    if arg_out != "":
        print('Output path: ', arg_out)
    if arg_exp_time is not None:
        print('Exposure time: ', arg_exp_time)
    if arg_samples is not None:
        print('Number of samples: ', arg_samples)
    print()

    return [arg_in, arg_out, arg_task, arg_exp_time, arg_samples]


if __name__ == '__main__':
    arg_in, arg_out, arg_task, arg_exp_time, arg_samples = arg_parser(sys.argv)  # Recover the input and output folder from the console args

    if arg_task == "tr_video":
        print(f"TASK: {arg_task}")
        start = time.time()

        ut.create_folder(arg_out, "all")  # Create the output folder if not already present
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
        t = tr.total_img(images=images,
                         out_path=arg_out / "total_image",
                         n_samples=arg_samples)

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    elif arg_task == "glb_tr_video":
        print(f"TASK: {arg_task}")
        start = time.time()

        images = tr.transient_loader(img_path=arg_in,
                                     np_path=arg_out / "np_transient.npy",
                                     store=(not exists(arg_out / "np_transient.npy")))  # Load the transient
        glb_images = tr.rmv_first_reflection_img(images=images,
                                                 file_path=arg_out / "glb_np_transient.npy",
                                                 store=(not exists(arg_out / "glb_np_transient.npy")))
        tr.transient_video(images=np.copy(glb_images),
                           out_path=arg_out,
                           normalize=True)
        tr.total_img(images=glb_images,
                     out_path=arg_out / "total_image",
                     n_samples=arg_samples)

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    elif arg_task == "hists":
        print(f"TASK: {arg_task}")
        start = time.time()

        ut.create_folder(arg_out, "all")
        images = tr.transient_loader(img_path=arg_in,
                                     np_path=arg_out / "np_transient.npy",
                                     store=(not exists(arg_out / "np_transient.npy")))  # Load the transient
        tr.histo_plt(radiance=images[:, 91, 182, :],  # righe:colonne
                     exp_time=arg_exp_time,
                     interval=None,
                     stem=False,
                     file_path=arg_out / "transient_histograms.svg")

        end = time.time()
        minutes, seconds = divmod(end - start, 60)
        hours, minutes = divmod(minutes, 60)
        print(f"Task <{arg_task}> concluded in in %d:%02d:%02d\n" % (hours, minutes, seconds))
    elif arg_task == "hists_glb":
        print(f"TASK: {arg_task}")
        start = time.time()

        ut.create_folder(arg_out, "all")
        images = tr.transient_loader(img_path=arg_in,
                                     np_path=arg_out / "np_transient.npy",
                                     store=(not exists(arg_out / "np_transient.npy")))  # Load the transient
        glb_images = tr.rmv_first_reflection_img(images=images,
                                                 file_path=arg_out / "glb_np_transient.npy",
                                                 store=(not exists(arg_out / "glb_np_transient.npy")))
        tr.histo_plt(radiance=glb_images[:, 119, 161, :],  # righe:colonne
                     exp_time=arg_exp_time,
                     interval=None,
                     stem=False,#True,
                     file_path=arg_out / "transient_histograms.svg")

        end = time.time()
        minutes, seconds = divmod(end - start, 60)
        hours, minutes = divmod(minutes, 60)
        print(f"Task <{arg_task}> concluded in in %d:%02d:%02d\n" % (hours, minutes, seconds))
    elif arg_task == "test":
        print(f"TASK: {arg_task}")
        start = time.time()

        dirs = ut.read_folders(arg_in, reorder=True)

        for i, dir_path in enumerate(dirs):
            folder_name = dir_path.split("\\")[-1]
            out_folder = arg_out / f"TEST_{folder_name}"

            print(f"Working on folder {(i + 1)}/{len(dirs)}:")
            print("Build the global transient video and the global total image:")
            images = tr.transient_loader(img_path=Path(dir_path),
                                         np_path=out_folder / "np_transient.npy",
                                         store=(not exists(arg_out / "np_transient.npy")))  # Load the transient
            glb_images = tr.rmv_first_reflection_img(images=np.copy(images),
                                                     file_path=out_folder / "glb_np_transient.npy",
                                                     store=(not exists(out_folder / "glb_np_transient.npy")))
            tr.transient_video(images=np.copy(glb_images),
                               out_path=out_folder,
                               normalize=True,
                               name="transient_glb")
            tr.total_img(images=glb_images,
                         out_path=out_folder / "total_image_glb",
                         n_samples=arg_samples)
            print("Build the transient video and the total image:")
            tr.transient_video(images=images,
                               out_path=out_folder,
                               normalize=True,
                               name="transient")
            tr.total_img(images=images,
                         out_path=out_folder / "total_image",
                         n_samples=arg_samples)
            print("Build the global histogram of the central pixel:")
            tr.histo_plt(radiance=glb_images[:, 119, 161, :],  # righe:colonne
                         exp_time=arg_exp_time,
                         interval=None,
                         stem=False,
                         file_path=out_folder / "transient_histograms_glb.svg")
            print("Build the histogram of the central pixel:")
            tr.histo_plt(radiance=images[:, 119, 161, :],  # righe:colonne
                         exp_time=arg_exp_time,
                         interval=None,
                         stem=False,
                         file_path=out_folder / "transient_histograms.svg")
            print()

        end = time.time()
        minutes, seconds = divmod(end - start, 60)
        hours, minutes = divmod(minutes, 60)
        print(f"Task <{arg_task}> concluded in in %d:%02d:%02d\n" % (hours, minutes, seconds))
    else:
        print("Wrong task provided\nPossibilities are: tr_video, total_img, glb_tr_video, hists, hists_glb")
