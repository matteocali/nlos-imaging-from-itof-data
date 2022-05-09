import sys
import getopt
import os
from pathlib import Path
import glob
import time
from statistics import mean

from natsort import natsorted
from tqdm import tqdm
import OpenEXR
import Imath
import numpy as np
import math
import matplotlib.pyplot as plt


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_input = os.getcwd()  # Argument containing the input directory
    arg_output = "cross_section_plot"  # Argument containing the output directory
    arg_exp = ""  # Argument containing the exposure_time
    arg_fov = ""  # Argument containing the fov camera
    arg_help = "{0} -i <input> -o <output>, -e <exposure>, -f <fov>".format(argv[0])  # Help string

    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:e:f:", ["help", "input=", "output=", "exposure=", "fov="])  # Recover the passed options and arguments from the command line (if any)
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
            arg_output = arg  # Set the output name
        elif opt in ("-e", "--exposure"):
            arg_exp = float(arg)  # Set the output name
        elif opt in ("-f", "--fov"):
            arg_fov = float(arg)  # Set the output name

    print('Input folder:', arg_input)
    print('Output file name:', arg_output)
    print('Exposure time:', arg_exp)
    print('FOV:', arg_fov)
    print()

    return [arg_input, arg_output, arg_exp, arg_fov]


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
    pt = OpenEXR.InputFile(files[0]).header()['channels']['R'].type  # Recover the pixel type from the header

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


def extract_cross_peak(channels):
    """
    Function that extract the position and the value of the first peak in the middle row and column in each channel alpha excluded
    :param channels: list of all the channels
    :return: return a list of 3 lists the firs containing the peak position in the center of the matrix for each channel the second one the value of each peak in the middle row, the fird one the value of each peak in the middle column
    """

    try:
        max_index_R, max_index_G, max_index_B = [np.nanargmax(channel, axis=2) for channel in channels]  # Find the index of the maximum value in the third dimension
    except ValueError:  # Manage the all NaN situation
        channels_no_nan = []
        for channel in channels:
            temp = channel
            temp[np.isnan(channel)] = 0
            channels_no_nan.append(temp)
        max_index_R, max_index_G, max_index_B = [np.nanargmax(channel, axis=2) for channel in channels_no_nan]

    peak_row_pos = [data[int(data.shape[0]/2), :] for data in [max_index_R, max_index_G, max_index_B]]  # Extract the peak position in the middle row
    peak_col_pos = [data[:, int(data.shape[1]/2)] for data in [max_index_R, max_index_G, max_index_B]]  # Extract the peak position in the middle column
    peak_row_values = [[channel[int(channel.shape[0]/2), i, peak_pos] for i, peak_pos in enumerate(peak_row_pos[index])] for index, channel in enumerate(channels)]  # Extract the value of the middle row in the peak position
    peak_col_values = [[channel[i, int(channel.shape[1]/2), peak_pos] for i, peak_pos in enumerate(peak_col_pos[index])] for index, channel in enumerate(channels)]  # Extract the value of the middle column in the peak position

    return [peak_row_pos, peak_col_pos, peak_row_values, peak_col_values]


def compute_distance(peak_pos, exposure_time):
    """
    Function that take the position of the peak and the exposure time and compute the measured distance
    :param peak_pos: position of the peak in the transient
    :param exposure_time: size of each time bean
    :return:
    """
    return round(((peak_pos * exposure_time) / 2), 4)  # General equation to compute the distance given the peak_position (p) and the exposure_time (e): ((p + 1) + e) / 2
                                                       # We have to add the correction term, - e/2, to compensate for the rounding in the been size: ((p + 1) + e) / 2 - e/2
                                                       # -> ((p + 1) + e) / 2 - e/2 = (pe + e)/2 - e/2 = pe/2 + e/2 - e/2 = pe/2


def compute_plane_distances_increment(p_distance, h_len, fov):
    """
    Function to compute the length in meter of each pixel
    :param p_distance: distance from the plane and the camera
    :param h_len: number of pixel of half the main row
    :param fov: field of view of the camera
    :return: a float value representing the encoded length of a pixel in meter
    """
    h_fov = fov / 2  # Compute the theta angle between the optical axes and the last pixel of the main row
    h_plane_length = p_distance * math.tan(math.radians(h_fov))  # Compute the length of half the visible plane in meters
    return h_plane_length / h_len  # Compute the size of a pixel dividing the length of half the plane for the number of contained pixel (in the main row)


def compute_plane_distance(p_increment, len):
    """
    Function to build the incremental length vector
    :param p_increment: length of a pixel
    :param len: total number of pixel in the row/column
    :return: a list containing, for each pixel, the distance in meters from the center of the plane
    """

    if len % 2 == 0:  # Check if the total number of pixel is even or not
        h_plane_distances = [p_increment]  # If it is even set the first elemenet of the right/top half of the vector as the length of one pixel
    else:
        h_plane_distances = [0]  # If it is not, set the first value of the right/top half of the vector as 0

    for i in range(1, int(math.floor(len/2)), 1):
        h_plane_distances.append(h_plane_distances[i - 1] + p_increment)  # Populate the rest of the vector, adding one by one to the previous value the length of a pixel

    if len % 2 == 0:  # populate the left/top part of the vector reflecting the other part (changing the sign), cheking if the length of the vector is even or not
        plane_distances = [- elm for elm in h_plane_distances[::-1]]
    else:
        plane_distances = [- elm for elm in h_plane_distances[1:][::-1]]

    for elm in h_plane_distances: plane_distances.append(elm)  # Merge the two half of the vector

    return plane_distances


def theta_calculator(peak_pos, peak_row_values, peak_col_values, e_time, fov):
    """
    Function to compute the theta value of each pixel
    :param peak_pos:
    :param peak_row_values:
    :param peak_col_values:
    :param e_time:
    :param fov:
    :return: two vectors containing the theta value of the main row and column and the two distance vector for the main row and column
    """
    principal_distance = compute_distance(peak_pos, e_time)  # Compute the distance from the plane and the camera

    p_increment = compute_plane_distances_increment(principal_distance, len(peak_row_values[0])/2, fov)  # Compute the length of a pixel
    plane_row_distance = compute_plane_distance(p_increment, len(peak_row_values[0]))  # Compute the incremental length vector for the main row
    plane_col_distance = compute_plane_distance(p_increment, len(peak_col_values[0]))  # Compute the incremental length vector for the main column

    theta_row = [math.degrees(math.atan(float(tmp) / principal_distance)) for tmp in plane_row_distance]  # Compute the theta vector for the main row
    theta_col = [math.degrees(math.atan(float(tmp) / principal_distance)) for tmp in plane_col_distance]  # Compute the theta vector for the main column

    return [theta_row, theta_col, plane_row_distance, plane_col_distance]


def plot_generator(x, real, model, ticks_interval, name, ext):
    """
    Function to save a single plot
    :param ticks_interval: number of skipped ticks between each other
    :param x: x axes values
    :param real: measured values to put on the y axes
    :param model: expected values (ground truth) to put on the y axes
    :param name: name of the plot
    :param ext: extension of the plot
    """

    # Create the x axes for the plot
    if len(x) % 2 == 0:
        r_x = [i for i in range(1, int(len(x) / 2) + 1, 1)]
        l_x = [- elm for elm in r_x[::-1]]
        x_label_full = [*l_x, *r_x]  # Define the all the ticks label to match the pixel position instead of the distance from the center
                                     # Code from: https://www.geeksforgeeks.org/python-ways-to-concatenate-two-lists/
        x_label = []  # List that will contain only the label of the desired ticks
        indexes = []  # List that will contain the index where the correspondent ticks are located
        for i, elm in enumerate(x_label_full):
            if i % ticks_interval == 0 and i < int(len(x) / 2):
                x_label.append(elm)
                indexes.append(i)
            if i == int(len(x) / 2):
                x_label.append(0)
            if i % ticks_interval == 0 and i > int(len(x) / 2):
                x_label.append(x_label_full[i - 1])
                indexes.append(i - 1)
        x_label.append(x_label_full[-1])
        indexes.append(len(x) - 1)

        ticks = [x[i] for i in indexes]  # List of only the desired ticks
        ticks.insert(int(len(ticks) / 2), 0)  # Add to the ticks list the one in 0
    else:
        r_x = [i for i in range(math.floor(len(x) / 2))]
        l_x = [- elm for elm in r_x[::-1]]
        l_x.append(0)
        x_label_full = [*l_x, *r_x]  # Define the all the ticks label to match the pixel position instead of the distance from the center
                                     # Code from: https://www.geeksforgeeks.org/python-ways-to-concatenate-two-lists/

        x_label = []  # List that will contain only the label of the desired ticks
        indexes = []  # List that will contain the index where the correspondent ticks are located
        for i, elm in enumerate(x_label_full):
            if i % ticks_interval == 0 and i < int(len(x) / 2):
                x_label.append(elm)
                indexes.append(i)
            if i == int(len(x) / 2) + 1:
                x_label.append(elm)
                indexes.append(i)
            if i % ticks_interval == 0 and i > int(len(x) / 2):
                x_label.append(x_label_full[i - 1])
                indexes.append(i - 1)
        x_label.append(x_label_full[-1])
        indexes.append(len(x) - 1)

        ticks = [x[i] for i in indexes]  # List of only the desired ticks

    plt.scatter(x, real, 6, label="Measured distances", color="tab:orange")  # Plot the measured data as dots
    plt.plot(x, model, '--', label="Ideal decaying of the intensities value", color="tab:blue")  # Plot the ground truth value as a dashed line
    plt.locator_params(axis='x', nbins=8)
    plt.xticks(ticks, x_label)
    plt.xlabel("Pixel position")  # Define the label on the x axes
    plt.ylabel(r"Radiance value on he red channel [$W/(m^{2}·sr)$]")  # Define the label on the y axes
    plt.grid()  # Add the grid to the plot
    plt.legend()  # Add the legend to the plot
    plt.savefig(name + ext)  # Save the generated plot
    plt.close()  # Close the currently open plot


def save_plot(theta_r, theta_c, r_values, c_values, row_distance, col_distance, output, ext = ".svg"):
    """
    Function to generate and save the two plots (one for the principal row and one for the principal column)
    :param theta_r: theta value on the main row
    :param theta_c: theta value on the main column
    :param r_values: measured values on the main row
    :param c_values: measured values on the main column
    :param row_distance: incremental length of the main row (pixel by pixel)
    :param col_distance: incremental length of the main column (pixel by pixel)
    :param output: name of the output file
    :param ext: extension of the saved file
    """

    print("Generating the two plots:")
    start = time.time()  # Compute the execution time

    y_r = [pow(math.cos(math.radians(theta)), 3) * mean(r_values[int(len(r_values)/2) - 3 : int(len(r_values)/2) + 3]) for theta in theta_r]  # Define the ground truth model for the main row as the cos^3(theta) * the max value of the radiance (outliers excluded)
    y_c = [pow(math.cos(math.radians(theta)), 3) * mean(c_values[int(len(c_values)/2) - 3 : int(len(c_values)/2) + 3]) for theta in theta_c]  # Define the ground truth model for the main column

    plot_generator(row_distance, r_values, y_r, 80, output + "_row", ext)  # Plot the main row plot
    plot_generator(col_distance, c_values, y_c, 60, output + "_col", ext)  # Plot the main column plot

    end = time.time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))


if __name__ == '__main__':
    arg_input, arg_output, arg_exp, arg_fov = arg_parser(sys.argv)  # Recover the input and output folder from the console args

    files = reed_files(str(arg_input), "exr")  # Load the path of all the files in the input folder with extension .exr

    (frame_A, frame_R, frame_G, frame_B) = reshape_frame(files)  # Reshape the frame in a standard layout

    peak_row_pos, peak_col_pos, peak_row_values, peak_col_values = extract_cross_peak([frame_R, frame_G, frame_B])
    
    theta_row, theta_col, row_distance, col_distance = theta_calculator(peak_row_pos[0][int(len(peak_row_pos[0])/2)], peak_row_values, peak_col_values, arg_exp, arg_fov)  # Compute the theta values and the incremental distance vector for the main row and column

    save_plot(theta_row, theta_col, peak_row_values[0], peak_col_values[0], row_distance, col_distance, arg_output)  # generate and save the two plots