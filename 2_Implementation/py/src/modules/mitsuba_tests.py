from statistics import mean
from time import time
from matplotlib import pyplot as plt
from modules import transient_handler as tr
from math import tan, radians, floor, degrees, atan, cos


def compute_plane_distances_increment(p_distance, h_len, fov):
    """
    Function to compute the length in meter of each pixel
    :param p_distance: distance from the plane and the camera
    :param h_len: number of pixel of half the main row
    :param fov: field of view of the camera
    :return: a float value representing the encoded length of a pixel in meter
    """
    h_fov = fov / 2  # Compute the theta angle between the optical axes and the last pixel of the main row
    h_plane_length = p_distance * tan(radians(h_fov))  # Compute the length of half the visible plane in meters
    return h_plane_length / h_len  # Compute the size of a pixel dividing the length of half the plane for the number of contained pixel (in the main row)


def compute_plane_distance(p_increment, length):
    """
    Function to build the incremental length vector
    :param p_increment: length of a pixel
    :param length: total number of pixel in the row/column
    :return: a list containing, for each pixel, the distance in meters from the center of the plane
    """

    if length % 2 == 0:  # Check if the total number of pixel is even or not
        h_plane_distances = [p_increment]  # If it is even set the first elemenet of the right/top half of the vector as the length of one pixel
    else:
        h_plane_distances = [0]  # If it is not, set the first value of the right/top half of the vector as 0

    for i in range(1, int(floor(length / 2)), 1):
        h_plane_distances.append(h_plane_distances[i - 1] + p_increment)  # Populate the rest of the vector, adding one by one to the previous value the length of a pixel

    if length % 2 == 0:  # populate the left/top part of the vector reflecting the other part (changing the sign), cheking if the length of the vector is even or not
        plane_distances = [- elm for elm in h_plane_distances[::-1]]
    else:
        plane_distances = [- elm for elm in h_plane_distances[1:][::-1]]

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
    principal_distance = tr.compute_center_distance(peak_pos, e_time)  # Compute the distance from the plane and the camera

    p_increment = compute_plane_distances_increment(principal_distance, peak_row_values/2, fov)  # Compute the length of a pixel
    plane_row_distance = compute_plane_distance(p_increment, peak_row_values)  # Compute the incremental length vector for the main row
    plane_col_distance = compute_plane_distance(p_increment, peak_col_values)  # Compute the incremental length vector for the main column

    theta_row = [degrees(atan(float(tmp) / principal_distance)) for tmp in plane_row_distance]  # Compute the theta vector for the main row
    theta_col = [degrees(atan(float(tmp) / principal_distance)) for tmp in plane_col_distance]  # Compute the theta vector for the main column

    return [theta_row, theta_col, plane_row_distance, plane_col_distance]


def plot_generator(x, real, model, ticks_interval, name, ext, color):
    """
    Function to save a single plot
    :param color: color of the dots in the graph
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
        r_x = [i for i in range(floor(len(x) / 2))]
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

    plt.scatter(x, real, 6, label="Measured distances", color=color)  # Plot the measured data as dots
    plt.plot(x, model, '--', label="Ideal decaying of the intensities value", color="tab:blue")  # Plot the ground truth value as a dashed line
    plt.locator_params(axis='x', nbins=8)
    plt.xticks(ticks, x_label)
    plt.xlabel("Pixel position")  # Define the label on the x axes
    plt.ylabel(r"Radiance value on the red channel [$W/(m^{2}·sr)$]")  # Define the label on the y axes
    plt.grid()  # Add the grid to the plot
    plt.legend()  # Add the legend to the plot
    plt.savefig(name + ext)  # Save the generated plot
    plt.close()  # Close the currently open plot


def save_plot(theta_r, theta_c, row_distance, col_distance, r_values, c_values, output, ext=".svg"):
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
    start = time()  # Compute the execution time

    y_r = [pow(cos(radians(theta)), 3) * mean(r_values[int(len(r_values)/2) - 3:int(len(r_values)/2) + 3]) for theta in theta_r]  # Define the ground truth model for the main row as the cos^3(theta) * the max value of the radiance (outliers excluded)
    y_c = [pow(cos(radians(theta)), 3) * mean(c_values[int(len(c_values)/2) - 3:int(len(c_values)/2) + 3]) for theta in theta_c]  # Define the ground truth model for the main column

    plot_generator(row_distance, r_values, y_r, 80, output + "_row", ext, "tab:orange")  # Plot the main row plot
    plot_generator(col_distance, c_values, y_c, 60, output + "_col", ext, "tab:green")  # Plot the main column plot

    end = time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))


def cross_section_tester(images, tot_img, exp_time, fov, output_path):
    """
    Function that generate the graphs of the cross-section analysis on the principal row and column
    :param images: np array containing the images of the transient [n_beans, n_row, n_col, 3]
    :param tot_img: image corresponding to the sum of all the transient images over the temporal dimension
    :param exp_time: exposure time used during the rendering
    :param fov: field of view of the camera
    :param output_path: path of the output folder
    """
    theta_row, theta_col, row_distance, col_distance = theta_calculator(peak_pos=tr.extract_center_peak(images)[0][0],
                                                                        peak_row_values=images[0].shape[1],
                                                                        peak_col_values=images[0].shape[0],
                                                                        e_time=exp_time,
                                                                        fov=fov)

    save_plot(theta_r=theta_row,
              theta_c=theta_col,
              row_distance=row_distance,
              col_distance=col_distance,
              r_values=list(tot_img[int(tot_img.shape[1] / 2), :, 0]),
              c_values=list(tot_img[:, int(tot_img.shape[1] / 2), 0]),
              output=str(output_path / "cross_section"))