import sys
import getopt
import os
from pathlib import Path
import glob
import time
import warnings
from natsort import natsorted
from tqdm import tqdm
import OpenEXR
import imageio
import Imath
import numpy as np
from matplotlib import pyplot as plt, cm, colors


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_input = os.getcwd()  # Argument containing the input directory
    arg_img = ""  # Argument containing the path of the target image
    arg_output = "total_image"  # Argument containing the output directory
    arg_help = "{0} -i <input> -m <image> -o <output>".format(argv[0])  # Help string

    try:
        opts, args = getopt.getopt(argv[1:], "hi:m:o:", ["help", "input=", "image=", "output="])  # Recover the passed options and arguments from the command line (if any)
    except:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # Print the help message
            sys.exit(2)
        elif opt in ("-i", "--input"):
            arg_input = Path(arg)  # Set the input directory
        elif opt in ("-m", "--image"):
            arg_img = Path(arg)  # Set the path of the target image
        elif opt in ("-o", "--output"):
            arg_output = arg  # Set the output directory

    print('Input folder:', arg_input)
    print('Input image:', arg_img)
    print('Output file name:', arg_output)
    print()

    return [arg_input, arg_img, arg_output]


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


def load_exr(path):
    """
    Function that load an exr image
    :param path: path of the image
    :return: a numpy matrix containing al the channels ([A, R, G, B] or [R, G, B])
    """
    img = OpenEXR.InputFile(path)  # Open the file
    n_channels = len(img.header()['channels'])  # Extract the number of channels of the image
    dw = img.header()['dataWindow']  # Extract the data window dimension from the header of the exr file
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)  # Define the actual size of the image
    pt = OpenEXR.InputFile(files[0]).header()['channels']['R'].type  # Recover the pixel type from the header

    # Check if the pt is HALF (pt.v == 1) or FLOAT (pt.v == 2)
    if pt.v == Imath.PixelType.HALF:
        np_pt = np.float16  # Define the correspondent value in numpy (float16 or float32)
    elif pt.v == Imath.PixelType.FLOAT:
        np_pt = np.float32

    if n_channels == 3:
        (R, G, B) = [np.frombuffer(img.channel(Chan, pt), dtype=np_pt) for Chan in ("R", "G", "B")]  # Extract the four channel from each image
        if np_pt == np.float16:
            (R, G, B) = [data.reshape(size[1], -1).astype(np.float32) for data in [R, G, B]]  # Reshape each vector to match the image size
        else:
            (R, G, B) = [data.reshape(size[1], -1) for data in [R, G, B]]  # Reshape each vector to match the image size

        return np.stack((R, G, B), axis=2)
    elif n_channels == 4:
        (A, R, G, B) = [np.frombuffer(img.channel(Chan, pt), dtype=np_pt) for Chan in ("A", "R", "G", "B")]  # Extract the four channel from each image
        if np_pt == np.float16:
            (A, R, G, B) = [data.reshape(size[1], -1).astype(np.float32) for data in [A, R, G, B]]  # Reshape each vector to match the image size
        else:
            (A, R, G, B) = [data.reshape(size[1], -1) for data in [A, R, G, B]]  # Reshape each vector to match the image size

        return np.stack((A, R, G, B), axis=2)


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

    # Define an empty matrix of size image_height x image_width x temporal_samples for each channel
    frame_A = np.empty([len(files), size[1], size[0]], dtype=np.float32)
    frame_R = np.empty([len(files), size[1], size[0]], dtype=np.float32)
    frame_G = np.empty([len(files), size[1], size[0]], dtype=np.float32)
    frame_B = np.empty([len(files), size[1], size[0]], dtype=np.float32)

    for index, file in enumerate(tqdm(files)):  # For each provided file in the input folder
        img = load_exr(file)

        # Perform the reshaping saving the results in frame_i for i in A, R,G , B
        for i in range(size[0]):
            frame_A[index, :, i] = img[:, i, 0]
            frame_R[index, :, i] = img[:, i, 1]
            frame_G[index, :, i] = img[:, i, 2]
            frame_B[index, :, i] = img[:, i, 3]

        warnings.filterwarnings('ignore')  # Remove warning about the presence of matrix completely empty (full of nan)

    time.sleep(0.05)  # Wait a bit to allow a proper visualization in the console
    end = time.time()
    print("Reshaping concluded in %.2f sec\n" % (round((end - start), 2)))

    return [frame_A, frame_R, frame_G, frame_B]


def img_matrix(channels):
    """
    Function that from the single channel matrices generate a proper image matrix fusing them
    :param channels: list of the 4 channels
    :return: list of image matrix [R, G, B, A]
    """

    print("Generating the image files:")
    start = time.time()  # Compute the execution time

    print(f"Build the {np.shape(frame_A)[2]} image matrices:")
    time.sleep(0.02)
    images = np.empty([np.shape(frame_A)[2], np.shape(frame_A)[0], np.shape(frame_A)[1], len(channels)], dtype=np.float32)  # Empty array that will contain all the images
    # Fuse the channels together to obtain a proper [A, R, G, B] image
    for i in tqdm(range(np.shape(frame_A)[2])):
        images[i, :, :, 0] = frame_R[:, :, i]
        images[i, :, :, 1] = frame_G[:, :, i]
        images[i, :, :, 2] = frame_B[:, :, i]
        images[i, :, :, 3] = frame_A[:, :, i]

        images[i, :, :, :][np.isnan(frame_A[:, :, i])] = 0  # Remove all the nan value following the Alpha matrix

    np.save("np_images.npy", np.asarray(images))  # Save the loaded images as a numpy array

    end = time.time()
    print("Images created successfully in %.2f sec\n" % (round((end - start), 2)))

    return images


def total_img(images, output):
    """
    Function to build the image obtained by sum all the temporal instant of the transient
    :param images: np.array containing of all the images
    :param output: output path
    :return: total image as a numpy matrix
    """
    print("Generate the total image = sum over all the time instants")
    start = time.time()  # Compute the execution time

    summed_images = np.nansum(images[:, :, :, :-1], axis=0)  # Sum all the produced images over the time dimension ignoring the alpha channel

    # Generate a mask matrix that will contain the number of active beans in each pixel (needed to normalize the image)
    mask = np.zeros([images[0].shape[0], images[0].shape[1]], dtype=np.float32)
    for img in images:
        tmp = np.nansum(img, axis=2)
        mask[tmp.nonzero()] += 1
    mask[np.where(mask == 0)] = 1  # Remove eventual 0 values
    mask = np.stack((mask, mask, mask), axis=2)  # make the mask a three layer matrix

    total_image = np.divide(summed_images, mask).astype(np.float32)

    imageio.plugins.freeimage.download()  # Download (if needed the required plugin in order to export .exr file)
    imageio.imwrite(output + ".exr", total_image)  # Save yhe image

    end = time.time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))

    return total_image


def compute_mse(x, y):
    """
    Compute the MSE error between two images
    The 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
    The lower the error, the more "similar" the two images are
    (code from: https://pyimagesearch.com/2014/09/15/python-compare-two-images/)
    :param x: image 1
    :param y: image 2 (must have same dimensions of image 1)
    :return The MSE value rounded at the fourth value after comma
    """

    err = np.sum((x.astype("float") - y.astype("float")) ** 2)  # Convert the images to floating point
                                                                # Take the difference between the images by subtracting the pixel intensities
                                                                # Square these difference and sum them up
    err /= float(x.shape[0] * x.shape[1])  # Handles the mean of the MSE

    return round(err, 4)


def img_comparison(o_img, t_img):
    """
    Function to plot the comparison between the real image and the one obtained by summing the transient over the temporal direction (+ compute the MSE)
    :param o_img: original image [R, G, B]
    :param t_img: transient image [R, G, B]
    """
    print("Compare the original images with the one obtained summing all the transient ones")
    print(f"The MSE is {compute_mse(o_img, t_img)}")

    # Extract the minimum and maximum displayed value to normalize the colors
    min_val = min([np.min(o_img), np.min(t_img)])
    max_val = max([np.max(o_img), np.max(t_img)])

    # Plot each channel of both the image, together with the colorbar
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs[0, 0].matshow(o_img[:, :, 0], cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs[0, 0].set_title("Red channel of the original image")
    axs[1, 0].matshow(o_img[:, :, 1], cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs[1, 0].set_title("Green channel of the original image")
    axs[2, 0].matshow(o_img[:, :, 2], cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs[2, 0].set_title("Blu channel of the original image")
    axs[0, 1].matshow(t_img[:, :, 0], cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs[0, 1].set_title("Red channel of the transient image")
    axs[1, 1].matshow(t_img[:, :, 1], cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs[1, 1].set_title("Green channel of the transient image")
    axs[2, 1].matshow(t_img[:, :, 2], cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs[2, 1].set_title("Blu channel of the transient image")
    fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=min_val, vmax=max_val), cmap=cm.get_cmap('jet')), ax=axs, label=r"Radiance [$W/(m^{2}·sr)$]")
    plt.savefig("channel_comparison.svg")
    fig.show()


    # Compute the differences between the original and transient image, channel by channel
    R_diff = abs(t_img[:, :, 0] - o_img[:, :, 0])
    G_diff = abs(t_img[:, :, 1] - o_img[:, :, 1])
    B_diff = abs(t_img[:, :, 2] - o_img[:, :, 2])

    # Extract the minimum and maximum displayed value to normalize the colors
    min_val = min([np.min(R_diff), np.min(G_diff), np.min(B_diff)])
    max_val = max([np.max(R_diff), np.max(G_diff), np.max(B_diff)])

    # Plot the difference between the two images, channel by channel
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))
    axs2[0].matshow(R_diff, cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs2[0].set_title("Difference on the red channel")
    axs2[1].matshow(G_diff, cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs2[1].set_title("Difference on the green channel")
    axs2[2].matshow(B_diff, cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs2[2].set_title("Difference on the blu channel")
    fig2.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=min_val, vmax=max_val), cmap=cm.get_cmap('jet')), ax=axs2, orientation="horizontal")
    plt.savefig("channel_differences.svg")
    fig2.show()


    o_img[np.where(o_img == 0)] = 1  # Remove eventual 0 values

    # Compute the ratio between the original and transient image, channel by channel
    R_div = t_img[:, :, 0] - o_img[:, :, 0]
    G_div = t_img[:, :, 1] - o_img[:, :, 1]
    B_div = t_img[:, :, 2] - o_img[:, :, 2]

    # Extract the minimum and maximum displayed value to normalize the colors
    min_val = min([np.min(R_div), np.min(G_div), np.min(B_div)])
    max_val = max([np.max(R_div), np.max(G_div), np.max(B_div)])

    # Plot the ratio between the two images, channel by channel
    fig3, axs3 = plt.subplots(1, 3, figsize=(18, 6))
    axs3[0].matshow(R_div, cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs3[0].set_title("Ratio on the red channel (original/transient)")
    axs3[1].matshow(G_div, cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs3[1].set_title("Ratio on the green channel (original/transient)")
    axs3[2].matshow(B_div, cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs3[2].set_title("Ratio on the blu channel (original/transient)")
    fig3.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=min_val, vmax=max_val), cmap=cm.get_cmap('jet')), ax=axs3, orientation="horizontal")
    plt.savefig("channel_ratio.svg")
    fig3.show()

    print("Press enter to end ...")
    input()  # Wait for a keystroke to close the windows


if __name__ == '__main__':
    arg_input, arg_img, arg_output = arg_parser(sys.argv)  # Recover the input and output folder from the console args

    files = reed_files(str(arg_input), "exr")  # Load the path of all the files in the input folder with extension .exr

    if not os.path.exists(Path("np_images.npy")):
        (frame_A, frame_R, frame_G, frame_B) = reshape_frame(files)  # Reshape the frame in a standard layout

        images = img_matrix([frame_A, frame_R, frame_G, frame_B])  # Create the image files
    else:  # If already exists a npy file containing all the transient images load it instead of processing everything again
        images = np.load("np_images.npy")

    total_image = total_img(images, arg_output)  # Create and save the total image

    original_img = load_exr(str(arg_img))  # Load the original image
    original_img[np.isnan(original_img[:, :, 0])] = 0  # Remove the nan value
    original_img = original_img[:, :, 1 :]  # Remove the alpha channel

    img_comparison(original_img, total_image)  # Compare the original render with the one obtained by summing up all the transient images