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
    arg_video = True  # Argument defining if it is required to render the transient video
    arg_video_type = "cv2"  # Argument defining if it is required to render the transient video
    arg_alpha = False  # Argument defining if the video will use or not the alpha channel
    arg_help = "{0} -i <input> -o <output>".format(argv[0])  # Help string

    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:v:t:a:", ["help", "input=", "output=", "video=", "type", "alpha"])  # Recover the passed options and arguments from the command line (if any)
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
        elif opt in ("-v", "--video"):
            arg_output = Path(arg)  # Set if we need the video or not
        elif opt in ("-t", "--type"):
            arg_output = Path(arg)  # Set the video type
        elif opt in ("-a", "--alpha"):
            arg_output = Path(arg)  # Set if we need the alpha channel or not

    if arg_output == "":  # if no output folder is provided define the default one
        arg_output = Path("output")
        if not os.path.exists(arg_output):
            os.makedirs(arg_output)

    print('Input folder:', arg_input)
    print('Output folder:', arg_output)
    print()

    return [arg_input, arg_output, arg_video, arg_video_type, arg_alpha]


def reed_files(path, extension, reorder=True):
    """
    Function to load all the files in a folder and if needed reoder them using the numbers in the name
    :param reorder: flag to toggle the reorder process
    :param path: folder path
    :param extension: extension of the files to load
    :return: list of file paths
    """
    files = [file_name for file_name in glob.glob(str(path) + "\*." + extension)]  # Load the path of all the files in the input folder with extension .exr
                                                                                   # code from: https://www.delftstack.com/howto/python/python-open-all-files-in-directory/
    if reorder:
        files = natsorted(files, key=lambda y: y.lower())  # Sort alphanumeric in ascending order
                                                           # code from: https://studysection.com/blog/how-to-sort-a-list-in-alphanumeric-order-python/
    return files


def reshape_frame(files):
    """
    Function that load al the exr file in the input folder and reshape it in order to have three matrices, one for each channel containing all the temporal value
    :param files: list off all the file path to analyze
    :return: list containing the reshaped frames for each channel
    """

    print(f"Reshaping {len(files)} frames:")
    start = time.time()  # Compute the execution time

    dw = OpenEXR.InputFile(files[0]).header()['dataWindow']    # Extract the data window dimension from the header of the exr file
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)  # Define the actual size of the image
    pt = Imath.PixelType(Imath.PixelType.HALF)                 # Define the type of the pixel (HALF = float16, FLOAT = float32)
                                                               # code from: https://excamera.com/articles/26/doc/intro.html

    # Define an empty matrix of size image_height x image_width x temporal_samples for each channel
    frame_A = np.empty([size[1], size[1], size[0]])
    frame_R = np.empty([size[1], size[1], size[0]])
    frame_G = np.empty([size[1], size[1], size[0]])
    frame_B = np.empty([size[1], size[1], size[0]])

    # Set initial value to the global min e global max (updated later)
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

        warnings.filterwarnings('ignore')  # Remove warning about the presence of matrix completely empty (full of nan)
        # Compute the local minimum and maximum (the one of the open fil). Ignore the Alpha matrix
        local_min = min([np.nanmin(R), np.nanmin(G), np.nanmin(B)])
        local_max = max([np.nanmax(R), np.nanmax(G), np.nanmax(B)])

        # Update the value of the global minimum and maximum
        if local_min < global_min: global_min = local_min
        if local_max > global_max: global_max = local_max

    time.sleep(0.05)  # Wait a bit to allow a proper visualization in the console
    end = time.time()
    print("Reshaping concluded in %.2f sec\n" % (round((end - start), 2)))

    return [frame_A, frame_R, frame_G, frame_B, global_min, global_max]


def channel_norm(channels, min_val=None, max_val=None):
    """
    Function to normalize the values of the channels in the range [0, 1]
    :param channels: list of 3 or 4 channels [A, R, G, B]
    :param min_val: global minimum value
    :param max_val: global maximum value
    :return: list of 3 or 4 normalized channels
    """

    # Check the number of input channels, if the Alpha one is provided do not normalize it
    if len(channels) == 4:
        norm_channels = [channels[0]]
        iteration = channels[1:]
    else:
        iteration = channels

    print("Normalize channels value in the range [0, 1]")
    # Normalize the channels
    for data in tqdm(iteration):
        if not (min_val == None) and not (max_val == None):
            norm_channels.append((data - min_val) / (max_val - min_val))  # Normalize the value of each channel in the range [0, 1] using the global minimum and maximum instead of the local ones
        else:
            norm_channels.append((data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data)))  # Normalize the value of each channel in the range [0, 1]
                                                                                                  # Code from: https://www.stackvidhya.com/how-to-normalize-data-between-0-and-1-range/
    return norm_channels


def save_png(img, path):
    """
    Function to save an image as a png
    :param img: image to save
    :param path: path and name
    """
    img = (255 * img).astype(np.uint8)  # Rescale the input value from [0, 1] to [0, 255] and convert them to unit8
    cv2.imwrite(str(path), img)  # Save the image


def img_matrix(channels, output=None, min_val=None, max_val=None):
    """
    Function that from the single channel matrices generate a proper image matrix fusing them
    :param output: output folder path
    :param channels: list of the 4 channels
    :param min_val: global minimum value (used for the normalization)
    :param max_val: global maximum value (used for the normalization)
    :return: list of image matrix [R, G, B, A]
    """

    print("Generating the image files:")
    start = time.time()  # Compute the execution time

    if not (min_val == None) and not (max_val == None):
        (frame_A_norm, frame_R_norm, frame_G_norm, frame_B_norm) = channel_norm(channels, min_val, max_val)  # Normalize the channels values in the range [0, 1]
    else:
        (frame_A_norm, frame_R_norm, frame_G_norm, frame_B_norm) = channel_norm(channels)
    time.sleep(0.05)  # Wait a bit to allow a proper visualization in the console

    # If needed create or empty the transient_images output folder
    if not output == None:
        out_path = output / "transient_images"
        if os.path.exists(out_path):  # If the folder is already present remove all its child files (code from: https://pynative.com/python-delete-files-and-directories/#h-delete-all-files-from-a-directory)
            for file_name in os.listdir(out_path):
                file = out_path / file_name  # Construct full file path
                if os.path.isfile(file):  # If the file is a file remove it
                    os.remove(file)
        else:  # Create the required folder if not already present
            os.makedirs(out_path)

    print(f"Build the {np.shape(frame_A_norm)[2]} image matrices:")
    images = []  # Empty list that will contain all the images
    # Fuse the channels together to obtain a proper [A, R, G, B] image
    for i in tqdm(range(np.shape(frame_A_norm)[2])):
        img = numpy.empty([np.shape(frame_A_norm)[0], np.shape(frame_A_norm)[1], len(channels)])  # Create an empty numpy array of the correct shape

        img[:, :, 0] = frame_R_norm[:, :, i]
        img[:, :, 1] = frame_G_norm[:, :, i]
        img[:, :, 2] = frame_B_norm[:, :, i]
        img[:, :, 3] = frame_A_norm[:, :, i]

        img[np.isnan(frame_A_norm[:, :, i])] = 0  # Remove all the nan value following the Alpha matrix
        images.append(img)

        # If needed save each image as a png
        if not output == None:
            save_png(img, out_path / f"img_{i}.png")

    end = time.time()
    print("Images created successfully in %.2f sec\n" % (round((end - start), 2)))

    return images


def total_img(images, output, alpha):
    """
    Function to build the image obtained by sum all the temporal instant of the transient
    :param images: list of all the images
    :param output: output path
    :param alpha: use the alpha channel or not
    """
    print("Generate the total image = sum over all the time instants")

    summed_images = np.asarray(images).sum(axis=0)  # Sum all the produced images over the time dimension
    summed_images = (summed_images - np.nanmin(summed_images)) / (np.nanmax(summed_images) - np.nanmin(summed_images))  # Normalize the values

    if not alpha: summed_images = summed_images[:,:,:-1]  # Check if it is required the alpha channel

    # Saving the images
    plt.imsave(output / "total_image_plt.png", summed_images)
    save_png(summed_images, output / "total_image.png")

    print("Process concluded\n")


def generate_transient_video(images, output, alpha=True, video_type="cv2"):
    """
    Function to generate a video of the transient images
    :param alpha: render with or without the alphamap
    :param cv2_video: if False render the pyplot video otherwise the one obtained by th png files
    :param images: list of images that will compose the video
    :param output: output folder path
    """

    print("Generating the final video:")
    start = time.time()  # Compute the execution time

    if not video_type == "cv2":
        # code from: https://stackoverflow.com/questions/34975972/how-can-i-make-a-video-from-array-of-images-in-matplotlib
        frames = []  # For storing the generated images
        fig = plt.figure()  # Create the figure

        for img in tqdm(images):
            if alpha:
                frames.append([plt.imshow(img, animated=True)])  # Create each frame
            else:
                frames.append([plt.imshow(img[:, :, :-1], animated=True)])  # Create each frame without the alphamap

        ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)  # Create the animation
        ani.save(output / "transient.avi")
    else:
        # code from: https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
        img_files = reed_files(output / "transient_images", "png")  # Load all the png transient files path

        img_list = []
        for file in img_files:
            img_list.append(cv2.imread(file))  # Load all the png transient files

        out = cv2.VideoWriter(str(output / "transient_cv2.avi"), cv2.VideoWriter_fourcc(*"mp4v"), 30, img_list[0].shape[0:2])  # Create the cv2 video

        for img in tqdm(img_list):
            if alpha:
                out.write(img)  # Populate the video
            else:
                out.write(img[:,:,:3])  # Populate the video without the alpha channel

        cv2.destroyAllWindows()
        out.release()

    end = time.time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))


if __name__ == '__main__':
    arg_input, arg_output, arg_video, arg_video_type, arg_alpha = arg_parser(sys.argv)  # Recover the input and output folder from the console args

    files = reed_files(str(arg_input), "exr")  # Load the path of all the files in the input folder with extension .exr

    (frame_A, frame_R, frame_G, frame_B, min_val, max_val) = reshape_frame(files)  # Reshape the frame in a standard layout

    images = img_matrix([frame_A, frame_R, frame_G, frame_B], arg_output, min_val, max_val)  # Create the image files

    total_img(images, arg_output, alpha=arg_alpha)

    if arg_video: generate_transient_video(images, arg_output, alpha=arg_alpha, video_type=arg_video_type)  # Generate the video