# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import getopt
import os
from pathlib import Path
import glob
import time
import warnings
from natsort import natsorted

import numpy
from tqdm import tqdm

import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_input = ""  # Arguments containing the input directory
    arg_output = ""  # Arguments containing the output directory
    arg_help = "{0} -i <input> -o <output>".format(argv[0])  # Help string

    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:", ["help", "input=", "output="])  # Recover the passed options and arguments from the command line (if any)
    except:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # Print the help message
            sys.exit(2)
        elif opt in ("-i", "--input"):
            arg_input = Path(arg)  # Set the input directory
        elif opt in ("-o", "--output"):
            arg_output = Path(arg)  # Set the output directory

    if arg_output == "":  # if no output folder is provided define the default one
        arg_output = Path("output")
        if not os.path.exists(arg_output):
            os.makedirs(arg_output)

    print('Input folder:', arg_input)
    print('Output folder:', arg_output)
    print()

    return [arg_input, arg_output]


def to_unit8(channels, size):
    """
    Function that convert the input array from float16 to unit8 (code from: https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values)
    :param channels: list of the three channels (as a mono-dimensional np array)
    :param size: size of each image
    :return: a list containing the three channels converted in unit8 and reshaped to match the image dimensions
    """

    (R, G, B) = [data.astype(np.float16) / max(data) for data in channels]  # Normalize the data to 0 - 1
    (R, G, B) = [255 * data for data in [R, G, B]]  # Now .astype(np.float16) scale by 255
    (R, G, B) = [data.astype(np.uint8).reshape(size[1], -1) for data in [R, G, B]]  # Convert the data in unit8 and reshape
    return [R, G, B]


def save_png(channels, size, name):
    """
    Function to save a png file starting from the three channels of an .exr file
    :param channels: R, G, B channels of an .exr file in float16
    :param size: size of each image
    :param name: name of the saved image
    """

    (R, G, B) = to_unit8(channels, size)  # Convert the three channel to unit8
    img_rgb = cv2.merge([B, G, R])  # Merge the three channel in a single image
    cv2.imwrite(f"{name}.png", img_rgb)  # save the obtained image


def reshape_frame(files, size, pt):
    """
    Function that load al the exr file in the input folder and reshape it in order to have three matrices, one for each channel containing all the temporal value
    :param files: list off all the file path to analyze
    :param size: size of the images
    :param pt: pixel type
    :return: list containing the reshaped frames for each channel
    """

    print(f"Reshaping {len(files)} frames:")
    start = time.time()

    # Define an empty matrix of size image_height x image_width x temporal_samples for each channel
    frame_A = np.empty([size[1], size[1], size[0]])
    frame_R = np.empty([size[1], size[1], size[0]])
    frame_G = np.empty([size[1], size[1], size[0]])
    frame_B = np.empty([size[1], size[1], size[0]])

    global_max = 0
    global_min = float("inf")

    for index, file in enumerate(tqdm(files)):  # For each provided file in the input folder
        img = OpenEXR.InputFile(file)  # Open each file

        (A, R, G, B) = [np.frombuffer(img.channel(Chan, pt), dtype=np.float16) for Chan in ("A", "R", "G", "B")]  # Extract the four channel from each image
        (A, R, G, B) = [data.reshape(size[1], -1).astype(np.float32) for data in [A, R, G, B]]  # reshape each vector to match the image size

        # Perform the reshaping saving the results in frame_i for i in A, R,G , B
        for i in range(size[0]):
            frame_A[index, :, i] = A[:, i]
            frame_R[index, :, i] = R[:, i]
            frame_G[index, :, i] = G[:, i]
            frame_B[index, :, i] = B[:, i]

        warnings.filterwarnings('ignore')
        local_min = min([np.nanmin(R), np.nanmin(G), np.nanmin(B)])
        local_max = max([np.nanmax(R), np.nanmax(G), np.nanmax(B)])

        if local_min < global_min: global_min = local_min
        if local_max > global_max: global_max = local_max

    time.sleep(0.05)
    end = time.time()
    print("Reshaping concluded in %.2f sec\n" % (round((end - start), 2)))

    return [frame_A, frame_R, frame_G, frame_B, global_min, global_max]


def channel_norm(channels, min_val, max_val):
    norm_channels = [channels[0]]
    for data in tqdm(channels[1:]):
        norm_channels.append((data - min_val) / (max_val - min_val))
        #norm_channels.append((data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data)))  # Normalize the value of each channel in the range [0, 1]
                                                                                              # Code from: https://www.stackvidhya.com/how-to-normalize-data-between-0-and-1-range/
    return norm_channels


def img_matrix(channels, min_val, max_val):

    print("Generating the image files:")
    start = time.time()

    print("Normalize channels value in the range [0, 1]")
    (frame_A_norm, frame_R_norm, frame_G_norm, frame_B_norm) = channel_norm(channels, min_val, max_val)
    time.sleep(0.05)

    print(f"Build the {np.shape(frame_A_norm)[2]} image matrices:")
    images = []
    for i in tqdm(range(np.shape(frame_A_norm)[2])):
        img = numpy.empty([np.shape(frame_A_norm)[0], np.shape(frame_A_norm)[1], len(channels)])
        img[:, :, 0] = frame_R_norm[:, :, i]
        img[:, :, 1] = frame_G_norm[:, :, i]
        img[:, :, 2] = frame_B_norm[:, :, i]
        img[:, :, 3] = frame_A_norm[:, :, i]
        #img[np.isnan(img)] = 0  # nan removal
        #cv2.imwrite(f"test_{i}.png", img*255)
        images.append(img)

    end = time.time()
    print("Images created successfully in %.2f sec\n" % (round((end - start), 2)))

    return images


def img_sum(images):
    return np.asarray(images).sum(axis=0)


def generate_video(images, output, time_samples):
    """
    code from: https://stackoverflow.com/questions/34975972/how-can-i-make-a-video-from-array-of-images-in-matplotlib
    :param img:
    """

    print("Generating the final video:")
    start = time.time()

    frames = []  # for storing the generated images
    fig = plt.figure()
    #norm = plt.Normalize(0, 1)                        #
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # code from: https://stackoverflow.com/questions/62778248/how-to-add-a-colorbar-for-iteratively-added-patches
    #plt.set_cmap("viridis")
    for i in tqdm(range(time_samples)):
        frames.append([plt.imshow(images[i], animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    #plt.show(block=True)
    ani.save(output / "transient.avi")

    end = time.time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))


if __name__ == '__main__':
    arg_input, arg_output = arg_parser(sys.argv)  # recover the input and output folder from the console args

    files = [file_name for file_name in glob.glob(str(arg_input) + "\*.exr")]  # Load the path of all the files in the inpud folder with extension .exr
                                                                               # code from: https://www.delftstack.com/howto/python/python-open-all-files-in-directory/
    files = natsorted(files, key=lambda y: y.lower())  # Sort alphanumeric in ascending order
                                                       # code from: https://studysection.com/blog/how-to-sort-a-list-in-alphanumeric-order-python/

    dw = OpenEXR.InputFile(files[0]).header()['dataWindow']    # Extract the data window dimension from the header of the exr file
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)  # Define the actual size of the image
    pt = Imath.PixelType(Imath.PixelType.HALF)                 # Define the type of the pixel (HALF = float16, FLOAT = float32)
                                                               # code from: https://excamera.com/articles/26/doc/intro.html

    (frame_A, frame_R, frame_G, frame_B, min_val, max_val) = reshape_frame(files, size, pt)
    #(frame_A, frame_R, frame_G, frame_B) = reshape_frame(files, size, pt)

    images = img_matrix([frame_A, frame_R, frame_G, frame_B], min_val, max_val)
    '''
    fig1 = plt.figure(1)
    plt.matshow(images[200])
    plt.colorbar()
    plt.show(block=False)
    '''

    total_img = img_sum(images)

    '''
    def no_nan(images):
        images_no_nan = []
        for img in images:
            img[np.isnan(img)] = 0
        return images
    
    images2 = no_nan(images)
    images2 = img_sum(images)
    images3 = (images2 - np.nanmin(images2)) / (np.nanmax(images2) - np.nanmin(images2))
    plt.imshow(images3)
    plt.show()
    '''

    generate_video(images, arg_output, size[0])