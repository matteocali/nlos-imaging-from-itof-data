import numpy as np
from .. import transient_utils as tr, exr_handler as exr
from math import tan, radians, floor, degrees, atan


def compute_plane_distances_increment(p_distance, h_len, fov):
    """
    Function to compute the length in meter of each pixel
    :param p_distance: distance from the plane and the camera
    :param h_len: number of pixel of half the main row
    :param fov: field of view of the camera
    :return: a float value representing the encoded length of a pixel in meter
    """
    h_fov = (
        fov / 2
    )  # Compute the theta angle between the optical axes and the last pixel of the main row
    h_plane_length = p_distance * tan(
        radians(h_fov)
    )  # Compute the length of half the visible plane in meters
    return (
        h_plane_length / h_len
    )  # Compute the size of a pixel dividing the length of half the plane for the number of contained pixel (in the main row)


def compute_plane_distance(p_increment, length):
    """
    Function to build the incremental length vector
    :param p_increment: length of a pixel
    :param length: total number of pixel in the row/column
    :return: a list containing, for each pixel, the distance in meters from the center of the plane
    """

    if length % 2 == 0:  # Check if the total number of pixel is even or not
        h_plane_distances = [
            p_increment
        ]  # If it is even set the first elemenet of the right/top half of the vector as the length of one pixel
    else:
        h_plane_distances = [
            0
        ]  # If it is not, set the first value of the right/top half of the vector as 0

    for i in range(1, int(floor(length / 2)), 1):
        h_plane_distances.append(
            h_plane_distances[i - 1] + p_increment
        )  # Populate the rest of the vector, adding one by one to the previous value the length of a pixel

    if (
        length % 2 == 0
    ):  # populate the left/top part of the vector reflecting the other part (changing the sign), cheking if the length of the vector is even or not
        plane_distances = [-elm for elm in h_plane_distances[::-1]]
    else:
        plane_distances = [-elm for elm in h_plane_distances[1:][::-1]]

    for elm in h_plane_distances:
        plane_distances.append(elm)  # Merge the two half of the vector

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
    principal_distance = tr.tools.compute_radial_distance(
        peak_pos, e_time
    )  # Compute the distance from the plane and the camera

    p_increment = compute_plane_distances_increment(
        principal_distance, peak_row_values / 2, fov
    )  # Compute the length of a pixel
    plane_row_distance = compute_plane_distance(
        p_increment, peak_row_values
    )  # Compute the incremental length vector for the main row
    plane_col_distance = compute_plane_distance(
        p_increment, peak_col_values
    )  # Compute the incremental length vector for the main column

    theta_row = [
        degrees(atan(float(tmp) / principal_distance)) for tmp in plane_row_distance
    ]  # Compute the theta vector for the main row
    theta_col = [
        degrees(atan(float(tmp) / principal_distance)) for tmp in plane_col_distance
    ]  # Compute the theta vector for the main column

    return [theta_row, theta_col, plane_row_distance, plane_col_distance]


def extract_data_from_file(files):
    """
    Function to extract the distance and radiance value from the txt files
    :param files: list of all the paths of all the txt file that has to be analyzed
    :return: list containing all the distances and another list containing all the radiance values
    """

    distances = []  # Define the empty list that will contain the distances values
    radiance = []  # Define the empty list that will contain the radiance values
    for file in files:  # For each file
        with open(file, "r") as reader:  # Open the file
            content = reader.readlines()  # Read the content
            distances.append(
                float(content[1].split(":")[1][1:])
            )  # From the second line extract the distance value parsing the string
            radiance.append(
                float(content[2].split(":")[1][1:])
            )  # From the third line extract the radiance value parsing the string

    return [distances, radiance]


def compute_norm_factor(tot_img, o_img_path, out_file=None):
    """
    Function that returns the normalization to use in a specific setup
    :param tot_img: total image (numpy array)
    :param o_img_path: path of the rgb render
    :param out_file: path, name and extension of the output file (if nothing does not save it)
    :return: the normalization factor value
    """
    original_img = exr.load_exr(str(o_img_path))  # Load the original image
    original_img[np.isnan(original_img[:, :, 0])] = 0  # Remove the nan value
    original_img = original_img[:, :, 1:]  # Remove the alpha channel

    # Compute the ratio between the total image and the rgb one channel by channel
    r_div = tot_img[:, :, 0] / original_img[:, :, 0]
    g_div = tot_img[:, :, 1] / original_img[:, :, 1]
    b_div = tot_img[:, :, 2] / original_img[:, :, 2]

    # compute the np.mean value of the ratio channel by channel
    mean_r = np.mean(r_div)
    mean_g = np.mean(g_div)
    mean_b = np.mean(b_div)

    norm_factor = round(
        float(np.mean([mean_r, mean_g, mean_b])), 3
    )  # Compute the norm factor as the overall mena

    print(
        "The normalization factor is: %.3f \n" % norm_factor
    )  # Print the overall np.mean as the normalization factor

    # Save the results to a file
    if out_file is not None:
        with open(out_file, "w") as f:
            f.write(
                "The normalization factor is: %.3f \n"
                % round(np.mean([mean_r, mean_g, mean_b]), 3)
            )

    return norm_factor
