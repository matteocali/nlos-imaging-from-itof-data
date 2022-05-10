import sys
import getopt
import os
from pathlib import Path
import glob
import time
from natsort import natsorted

import numpy
from tqdm import tqdm

import imageio
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

    arg_input = os.getcwd()  # Argument containing the input directory
    arg_output = ""  # Argument containing the output directory
    arg_video = True  # Argument defining if it is required to render the transient video
    arg_out_type = "cv2"  # Argument defining the type of the output (both for images and video)
    arg_alpha = False  # Argument defining if the video will use or not the alpha channel
    arg_help = "{0} -i <input> -o <output> -p <ptype> -v <video> (default = True) -t <type> (default = cv2) -a <alpha> (default = False)".format(argv[0])  # Help string

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
            arg_video = bool(arg)  # Set if we need the video or not
        elif opt in ("-t", "--type"):
            arg_out_type = arg  # Set the output type (cv2, plt, both)
        elif opt in ("-a", "--alpha"):
            arg_alpha = bool(arg)  # Set if we need the alpha channel or not

    if arg_output == "":  # if no output folder is provided define the default one
        arg_output = Path("output")
        if not os.path.exists(arg_output):
            os.makedirs(arg_output)

    print('Input folder:', arg_input)
    print('Output folder:', arg_output)
    print()

    return [arg_input, arg_output, arg_video, arg_out_type, arg_alpha]


def create_folder(path):
    """
    Function to create a new folder if not already present or empty it
    :param path: path of the folder to create
    """
    if os.path.exists(path):  # If the folder is already present remove all its child files (code from: https://pynative.com/python-delete-files-and-directories/#h-delete-all-files-from-a-directory)
        for file_name in os.listdir(path):
            file = path / file_name  # Construct full file path
            if os.path.isfile(file):  # If the file is a file remove it
                os.remove(file)
    else:  # Create the required folder if not already present
        os.makedirs(path)


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
    pt = OpenEXR.InputFile(files[0]).header()['channels']['R'].type  # Recover the pixel type from the header (HALF = float16, FLOAT = float32)

    # Check if the pt is HALF (pt.v == 1) or FLOAT (pt.v == 2)
    if pt.v == Imath.PixelType.HALF:
        np_pt = np.float16  # Define the correspondent value in numpy (float16 or float32)
    elif pt.v == Imath.PixelType.FLOAT:
        np_pt = np.float32

    # Define an empty matrix of size image_height x image_width x temporal_samples for each channel
    frame_A = np.empty([len(files), size[1], size[0]], dtype=np.float32)
    frame_R = np.empty([len(files), size[1], size[0]], dtype=np.float32)
    frame_G = np.empty([len(files), size[1], size[0]], dtype=np.float32)
    frame_B = np.empty([len(files), size[1], size[0]], dtype=np.float32)

    for index, file in enumerate(tqdm(files)):  # For each provided file in the input folder
        img = OpenEXR.InputFile(file)  # Open each file

        (A, R, G, B) = [np.frombuffer(img.channel(Chan, pt), dtype=np_pt) for Chan in ("A", "R", "G", "B")]  # Extract the four channel from each image
        if np_pt == np.float16:
            (A, R, G, B) = [data.reshape(size[1], -1).astype(np.float32) for data in [A, R, G, B]]  # Reshape each vector to match the image size
        else:
            (A, R, G, B) = [data.reshape(size[1], -1) for data in [A, R, G, B]]  # Reshape each vector to match the image size

        # Perform the reshaping saving the results in frame_i for i in A, R,G , B
        for i in range(size[0]):
            frame_A[index, :, i] = A[:, i]
            frame_R[index, :, i] = R[:, i]
            frame_G[index, :, i] = G[:, i]
            frame_B[index, :, i] = B[:, i]

    time.sleep(0.05)  # Wait a bit to allow a proper visualization in the console
    end = time.time()
    print("Reshaping concluded in %.2f sec\n" % (round((end - start), 2)))

    return [frame_A, frame_R, frame_G, frame_B]


def save_png(img, path, alpha):
    """
    Function to save an image as a png
    :param alpha: define if the output will use or not the alpha channel
    :param img: image to save
    :param path: path and name
    """
    img = (255 * img).astype(np.uint8)  # Rescale the input value from [0, 1] to [0, 255] and convert them to unit8
    if alpha:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))  # Save the image
    else:
        cv2.imwrite(str(path), cv2.cvtColor(img[:,:,:-1], cv2.COLOR_RGB2BGR))  # Save the image


def save_plt(img, path, alpha):
    """
    Function to save an image as a matplotlib png
    :param alpha: define if the output will use or not the alpha channel
    :param img: image to save
    :param path: path and name
    """
    if not alpha:
        plt.imsave(path, img[:, :, :-1])
    else:
        plt.imsave(path, img)


def img_matrix(channels, output, out_type, alpha):
    """
    Function that from the single channel matrices generate a proper image matrix fusing them
    :param out_type: define the desired output type (cv2, plt, both)
    :param alpha: define if the outputted images will use or not the alpha channel
    :param output: output folder path
    :param channels: list of the 4 channels
    :return: list of image matrix [R, G, B, A]
    """

    print("Generating the image files:")
    start = time.time()  # Compute the execution time
    # If needed create or empty the transient_images output folder
    if not output == None:
        out_path = output / "transient_images"
        create_folder(out_path)
        if out_type == "both":
            out_path_cv2 = out_path / "cv2"
            create_folder(out_path_cv2)
            out_path_plt = out_path / "plt"
            create_folder(out_path_plt)

    print(f"Build the {np.shape(frame_A)[2]} image matrices:")
    images = []  # Empty list that will contain all the images
    # Fuse the channels together to obtain a proper [A, R, G, B] image
    for i in tqdm(range(np.shape(frame_A)[2])):
        img = numpy.empty([np.shape(frame_A)[0], np.shape(frame_A)[1], len(channels)], dtype=np.float32)  # Create an empty numpy array of the correct shape

        img[:, :, 0] = frame_R[:, :, i]
        img[:, :, 1] = frame_G[:, :, i]
        img[:, :, 2] = frame_B[:, :, i]
        img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))  # Normalize each image in [0, 1] ignoring the aplha channel
        img[:, :, 3] = frame_A[:, :, i]

        img[np.isnan(frame_A[:, :, i])] = 0  # Remove all the nan value following the Alpha matrix

        images.append(img)

        # If needed save each image as a png
        if not output == None:
            if out_type == "cv2":
                save_png(img, out_path / f"img_{i}.png", alpha)
            elif out_type == "plt":
                save_plt(img, out_path / f"img_{i}.png", alpha)
            elif out_type == "both":
                save_png(img, out_path_cv2 / f"img_cv2_{i}.png", alpha)
                save_plt(img, out_path_plt / f"img_plt_{i}.png", alpha)

    end = time.time()
    print("Images created successfully in %.2f sec\n" % (round((end - start), 2)))

    return images


def total_img(images, output, out_type, alpha):
    """
    Function to build the image obtained by sum all the temporal instant of the transient
    :param images: list of all the images
    :param output: output path
    """
    print("Generate the total image = sum over all the time instants")
    start = time.time()  # Compute the execution time

    summed_images = np.nansum(np.asarray(images)[:, :, :, :-1], axis=0)  # Sum all the produced images over the time dimension ignoring the alpha channel

    # Generate a mask matrix that will contain the number of active beans in each pixel (needed to normalize the image)
    mask = np.zeros([images[0].shape[0], images[0].shape[1]])
    for img in images:
        tmp = np.nansum(img, axis=2)
        mask[tmp.nonzero()] += 1
    mask = np.stack((mask, mask, mask), axis=2)  # make the mask a three layer matrix

    total_image = np.divide(summed_images, mask).astype(np.float32)

    imageio.plugins.freeimage.download()  # Download (if needed the required plugin in order to export .exr file)
    imageio.imwrite(output + ".exr", total_image)  # Save yhe image

    end = time.time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))


def plt_transient_video(images, path, alpha):
    """
    Function that generate a video of the transient and save it in the matplotlib format
    :param images: list of all the transient images
    :param path: path where to save the video
    :param alpha: define if it has to use the alpha channel or not
    """
    # code from: https://stackoverflow.com/questions/34975972/how-can-i-make-a-video-from-array-of-images-in-matplotlib
    frames = []  # For storing the generated images
    fig = plt.figure()  # Create the figure

    for img in tqdm(images):
        if alpha:
            frames.append([plt.imshow(img, animated=True)])  # Create each frame
        else:
            frames.append([plt.imshow(img[:, :, :-1], animated=True)])  # Create each frame without the alphamap

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)  # Create the animation
    ani.save(path)


def cv2_transient_video(in_path, out_path, alpha):
    """
    Function that generate a video of the transient and save it in the cv2 format
    :param in_path: path where all the transient images are located
    :param out_path: path where to save the video
    :param alpha: define if it has to use the alpha channel or not
    """
    # code from: https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
    img_files = reed_files(in_path, "png")  # Load all the png transient files path

    img_list = []
    for file in img_files:
        img_list.append(cv2.imread(file))  # Load all the png transient files

    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (img_list[0].shape[1], img_list[0].shape[0]))  # Create the cv2 video

    for img in tqdm(img_list):
        if alpha:
            out.write(img)  # Populate the video
        else:
            out.write(img[:, :, :3])  # Populate the video without the alpha channel

    cv2.destroyAllWindows()
    out.release()


def generate_transient_video(images, output, out_type, alpha):
    """
    Function to generate a video of the transient images
    :param alpha: render with or without the alphamap
    :param cv2_video: if False render the pyplot video otherwise the one obtained by th png files
    :param images: list of images that will compose the video
    :param out_type: define the desired output type (cv2, plt, both)
    :param output: output folder path
    """

    print("Generating the final video:")
    start = time.time()  # Compute the execution time

    if out_type == "plt": plt_transient_video(images, output / "transient.avi", alpha)
    elif out_type == "cv2": cv2_transient_video(output / "transient_images", output / "transient.avi", alpha)
    elif out_type == "both":
        print("Matplotlib version:")
        plt_transient_video(images, output / "transient_plt.avi", alpha)
        print("Opencv version:")
        cv2_transient_video(output / "transient_images" / "cv2", output / "transient_cv2.avi", alpha)

    end = time.time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))


if __name__ == '__main__':
    arg_input, arg_output, arg_video, arg_out_type, arg_alpha = arg_parser(sys.argv)  # Recover the input and output folder from the console args

    files = reed_files(str(arg_input), "exr")  # Load the path of all the files in the input folder with extension .exr

    (frame_A, frame_R, frame_G, frame_B, min_val, max_val) = reshape_frame(files)  # Reshape the frame in a standard layout

    images = img_matrix([frame_A, frame_R, frame_G, frame_B], arg_output, arg_out_type, arg_alpha)  # Create the image files

    total_img(images, arg_output, arg_out_type, alpha=arg_alpha)

    if arg_video: generate_transient_video(images, arg_output, alpha=arg_alpha, out_type=arg_out_type)  # Generate the video