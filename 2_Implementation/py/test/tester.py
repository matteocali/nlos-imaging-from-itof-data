import modules.fermat_tools
from modules import utilities as ut, transient_handler as tr
from pathlib import Path
import os
from os.path import exists
import getopt
import sys
import time
import numpy as np
from tqdm import trange


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
            spot_size = str(arg)  # Read the img size
            spot_size = spot_size.split(",")
            arg_spot_size = (int(spot_size[0]), int(spot_size[1]))  # Set the spot size

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

        ut.spot_bitmap_gen(file_path=arg_out / "bitmaps",
                           img_size=arg_img_size,
                           spot_size=None,#arg_spot_size,
                           exact=False,
                           pattern=(32, 16),
                           split=True)

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    elif arg_task == "h5":
        print(f"TASK: {arg_task}")
        start = time.time()

        ut.create_folder(arg_out, "all")
        images = tr.transient_loader(img_path=arg_in,
                                     np_path=arg_out / "np_transient.npy",
                                     store=(not exists(arg_out / "np_transient.npy")))  # Load the transient
        ut.save_h5(data=images[:, :, :, 1], file_path=arg_out / "h5_data")
        #h5_file = ut.load_h5(arg_in)

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    elif arg_task == "itof2dtof":
        print(f"TASK: {arg_task}")
        start = time.time()

        ut.create_folder(arg_out, "all")
        images = tr.transient_loader(img_path=arg_in,
                                     np_path=arg_out / "np_transient.npy",
                                     store=(not exists(arg_out / "np_transient.npy")))  # Load the transient
        freqs = np.array((20e06, 50e06, 60e06), dtype=np.float32)
        phi = tr.phi(freqs=freqs,
                     exp_time=0.01,
                     dim_t=images.shape[0])

        tr.plot_phi(phi_matrix=phi, freq_values=freqs, file_path=arg_out / "phi", exp_time=0.01)

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    elif arg_task == "np2mat":
        print(f"TASK: {arg_task}")
        start = time.time()

        ut.create_folder(arg_out, "all")
        transient = tr.grid_transient_loader(transient_path=arg_in,
                                             np_path=arg_out / "np_transient.npy",
                                             store=(not exists(arg_out / "np_transient.npy")))  # Load the transient

        for i in range(transient.shape[0]):
            transient[i, :, :] = tr.clean_transient_tail(transient=transient[i, :, :], n_samples=20)

        #transient = tr.rmv_sparse_fermat_transient(transients=transient, channel=1, threshold=5, remove_data=False)

        '''
        for i in trange(transient.shape[0]):
            tr.histo_plt(radiance=tr.rmv_first_reflection_transient(transient=transient[i, :, :], file_path=arg_out / "hists" / "np" / f"np_transient_{i}.npy", store=(not exists(arg_out / "hists" / "np" / f"np_transient_{i}.npy"))),
                         exp_time=0.01,
                         interval=None,  # [15.7, 19.6],
                         stem=True,
                         file_path=arg_out / "hists" / f"transient_histograms{i}.svg")
        '''
        modules.fermat_tools.np2mat(data=transient,
                                    file_path=arg_out / "cube",
                                    data_grid_size=[32, 16],
                                    img_shape=[80, 60],
                                    fov=60,
                                    exp_time=0.01)

        end = time.time()
        print(f"Task <{arg_task}> concluded in in %.2f sec\n" % (round((end - start), 2)))
    else:
        print("Wrong task provided\nPossibilities are: spot_bitmap, h5, itof2dtof, np2mat")
