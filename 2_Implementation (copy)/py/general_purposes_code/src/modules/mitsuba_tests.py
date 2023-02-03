from statistics import mean
from time import time
from matplotlib import pyplot as plt, cm, colors
from modules import transient_handler as tr, utilities as ut, exr_handler as exr
from math import tan, radians, floor, degrees, atan, cos
from numpy import linspace, min, max, mean, where, isnan, format_float_scientific
from os.path import exists


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
    principal_distance = tr.compute_radial_distance(peak_pos, e_time)  # Compute the distance from the plane and the camera

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


def save_cross_section_plot(theta_r, theta_c, row_distance, col_distance, r_values, c_values, output, ext=".svg"):
    """
    Function to generate and save the two cross-section plots (one for the principal row and one for the principal column)
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

    save_cross_section_plot(theta_r=theta_row,
                            theta_c=theta_col,
                            row_distance=row_distance,
                            col_distance=col_distance,
                            r_values=list(tot_img[int(tot_img.shape[1] / 2), :, 0]),
                            c_values=list(tot_img[:, int(tot_img.shape[1] / 2), 0]),
                            output=str(output_path / "cross_section"))


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
            distances.append(float(content[1].split(":")[1][1:]))  # From the second line extract the distance value parsing the string
            radiance.append(float(content[2].split(":")[1][1:]))  # From the third line extract the radiance value parsing the string

    return [distances, radiance]


def save_distance_plot(dist, radiance, output, ext=".svg", annotation=False):
    """
    Function to generate and save the distance plot
    :param dist: vector containing al the measured distance values
    :param radiance: vector containing al the measured radiance values
    :param output: name of the output file
    :param ext: extension of the saved file
    :param annotation: boolean value to define if the plot will have annotation on the data
    """

    x, y = ut.generate_quadratic_model(min(dist), max(dist), max(radiance), 1000)  # Build the ground truth quadratic model

    plt.plot(x, y, '--', label="Ideal decaying of the intensities value")  # Plot the ground truth value as a dashed line
    plt.plot(dist, radiance, 'o', label="Measured distances")  # Plot the measured data as dots
    plt.xlabel("Distances value [m]")  # Define the label on the x axis
    plt.ylabel(r"Radiance value on the red channel [$W/(m^{2}·sr)$]")  # Define the label on the y axis
    if annotation:
        for i, j in zip(dist, radiance):
            plt.annotate(str(round(j, 3)), xy=(i, j))  # Add the annotation to each measured data point
    plt.grid()  # Add the grid to the plot
    plt.legend()  # Add the legend to the plot
    plt.savefig(output+ext)  # Save the generated plot
    plt.show()  # Show the plot


def save_millimiter_plot(dist, step, max_value, output, ext=".svg"):
    """
    Function to generate and save the plot
    :param max_value: maximum value visualized on the x axes
    :param step: number of skipped ticks between two visualized ones
    :param dist: vector containing al the measured distance values
    :param output: name of the output file
    :param ext: extension of the saved file
    """

    lin = linspace(step, max_value, 1000)  # Build the ground truth linear model

    x = [x for x in range(step, max_value + step, step)]  # x axes of the measured data

    plt.figure(figsize=(10, 8))
    plt.plot(lin, lin, '--', label="Ideal distances values")  # Plot the ground truth value as a dashed line
    plt.step(x, [x * 1000 for x in dist], where="post", linewidth=2, label="Measured distances")  # Plot the measured data as steps
    plt.xticks(list(range(step, 32, 2)))  # Define the values displayed on the x axes
    plt.yticks(list(range(step, 36, 1)))  # Define the values displayed on the y axes
    plt.xlabel("Distances value [mm]")  # Define the label on the x axes
    plt.ylabel("Distances value [mm]")  # Define the label on the y axes
    plt.grid()  # Add the grid to the plot
    plt.legend()  # Add the legend to the plot
    plt.savefig(output + ext)  # Save the generated plot
    plt.show()  # Show the plot


def distance_plot(in_path, out_name):
    """
    Function to generate and save the decaying distance plot
    :param in_path: input path containing all the txt file of the measured distances
    :param out_name: path and name (no extension) of the output file
    """
    files = ut.read_files(str(in_path), "txt")  # Load the path of all the files in the input folder with extension .exr
    dist, radiance = extract_data_from_file(files)  # Extract the distance and radiance values from the files
    save_distance_plot(dist, radiance, str(out_name))  # Generate and save the standard plot


def mm_distance_plot(in_path, step, max_value, out_name):
    """
    Function to generate and save the plot regarding the millimiter distance -> evaluate the quantization
    :param in_path: input path containing all the txt file of the measured distances
    :param step: number of skipped ticks between two visualized ones
    :param max_value: maximum value visualized on the x axes
    :param out_name: path and name (no extension) of the output file
    """
    files = ut.read_files(str(in_path), "txt")  # Load the path of all the files in the input folder with extension .exr
    dist, radiance = extract_data_from_file(files)  # Extract the distance and radiance values from the files
    save_millimiter_plot(dist, step, max_value, str(out_name))  # Generate and save the millimiter plot


def img_comparison(o_img, t_img, out_path, diff_limits=None, ratio_limits=None):
    """
    Function to plot the comparison between the real image and the one obtained by summing the transient over the temporal direction (+ compute the MSE)
    :param out_path: folder path where to save the graphs
    :param diff_limits: fixed min and max value of the diff colorbar (tuple)
    :param ratio_limits: fixed min and max value of the ratio colorbar (tuple)
    :param o_img: original image [R, G, B]
    :param t_img: transient image [R, G, B]
    """
    print("Compare the original images with the one obtained summing all the transient ones")
    print(f"The MSE is {ut.compute_mse(o_img, t_img)}\n")

    # Extract the minimum and maximum displayed value to normalize the colors
    min_val = min([min(o_img), min(t_img)])
    max_val = max([max(o_img), max(t_img)])

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
    plt.savefig(str(out_path / "channel_comparison.svg"))

    # Compute the differences between the original and transient image, channel by channel
    r_diff = abs(t_img[:, :, 0] - o_img[:, :, 0])
    g_diff = abs(t_img[:, :, 1] - o_img[:, :, 1])
    b_diff = abs(t_img[:, :, 2] - o_img[:, :, 2])

    # Extract the minimum and maximum displayed value to normalize the colors
    if diff_limits is None:
        min_val = min([min(r_diff), min(g_diff), min(b_diff)])
        max_val = max([max(r_diff), max(g_diff), max(b_diff)])
    else:
        min_val = diff_limits[0]
        max_val = diff_limits[1]

    # Plot the difference between the two images, channel by channel
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))
    axs2[0].matshow(r_diff, cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs2[0].set_title("Difference on the red channel")
    axs2[1].matshow(g_diff, cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs2[1].set_title("Difference on the green channel")
    axs2[2].matshow(b_diff, cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs2[2].set_title("Difference on the blu channel")
    fig2.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=min_val, vmax=max_val), cmap=cm.get_cmap('jet')), ax=axs2, orientation="horizontal")
    plt.savefig(str(out_path / "channel_differences.svg"))

    o_img[where(o_img == 0)] = 1  # Remove eventual 0 values

    # Compute the ratio between the original and transient image, channel by channel
    r_div = t_img[:, :, 0] / o_img[:, :, 0]
    g_div = t_img[:, :, 1] / o_img[:, :, 1]
    b_div = t_img[:, :, 2] / o_img[:, :, 2]

    # Extract the minimum and maximum displayed value to normalize the colors
    if ratio_limits is None:
        min_val = min([min(r_div), min(g_div), min(b_div)])
        max_val = max([max(r_div), max(g_div), max(b_div)])
    else:
        min_val = ratio_limits[0]
        max_val = ratio_limits[1]

    # Plot the ratio between the two images, channel by channel
    fig3, axs3 = plt.subplots(1, 3, figsize=(18, 6))
    axs3[0].matshow(r_div, cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs3[0].set_title("Ratio on the red channel (original/transient)")
    axs3[1].matshow(g_div, cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs3[1].set_title("Ratio on the green channel (original/transient)")
    axs3[2].matshow(b_div, cmap=cm.get_cmap("jet"), norm=colors.Normalize(vmin=min_val, vmax=max_val))
    axs3[2].set_title("Ratio on the blu channel (original/transient)")
    fig3.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=min_val, vmax=max_val), cmap=cm.get_cmap('jet')), ax=axs3, orientation="horizontal")
    plt.savefig(str(out_path / "channel_ratio.svg"))


def tot_img_tester(rgb_img_path, total_img, out_path, diff_limits=None, ratio_limits=None):
    """
    Function that compare the total image with the standard rgb render
    :param out_path: folder path where to save the graphs
    :param ratio_limits: fixed min and max value of the ratio colorbar (tuple)
    :param diff_limits: fixed min and max value of the diff colorbar (tuple)
    :param rgb_img_path: path of the standard RGB image
    :param total_img: total image
    """
    original_img = exr.load_exr(str(rgb_img_path))  # Load the original image
    original_img[isnan(original_img[:, :, 0])] = 0  # Remove the nan value
    original_img = original_img[:, :, 1:]  # Remove the alpha channel

    img_comparison(original_img, total_img, out_path, diff_limits, ratio_limits)  # Compare the original render with the one obtained by summing up all the transient images


def compute_norm_factor(tot_img, o_img_path, out_file=None):
    """
    Function that returns the normalization to use in a specific setup
    :param tot_img: total image (numpy array)
    :param o_img_path: path of the rgb render
    :param out_file: path, name and extension of the output file (if nothing does not save it)
    :return: the normalization factor value
    """
    original_img = exr.load_exr(str(o_img_path))  # Load the original image
    original_img[isnan(original_img[:, :, 0])] = 0  # Remove the nan value
    original_img = original_img[:, :, 1:]  # Remove the alpha channel

    # Compute the ratio between the total image and the rgb one channel by channel
    r_div = tot_img[:, :, 0] / original_img[:, :, 0]
    g_div = tot_img[:, :, 1] / original_img[:, :, 1]
    b_div = tot_img[:, :, 2] / original_img[:, :, 2]

    # compute the mean value of the ratio channel by channel
    mean_r = mean(r_div)
    mean_g = mean(g_div)
    mean_b = mean(b_div)

    norm_factor = round(float(mean([mean_r, mean_g, mean_b])), 3)  # Compute the norm factor as the overall mena

    print("The normalization factor is: %.3f \n" % norm_factor)  # Print the overall mean as the normalization factor

    # Save the results to a file
    if out_file is not None:
        with open(out_file, "w") as f:
            f.write("The normalization factor is: %.3f \n" % round(mean([mean_r, mean_g, mean_b]), 3))

    return norm_factor


def plot_norm_factor(folder_path, rgb_path, out_path):
    """
    Function that compute the normalization factor of different transient setups and plot the results
    :param folder_path: path of the folder containing all the transient information
    :param rgb_path: path to the folder containing all the rgb images
    :param out_path: path of the output directory
    """

    folders = ut.read_folders(folder_path)  # Load all the subdirectory of the folder containing the various transient
    rgb_images = ut.read_files(file_path=rgb_path, extension="exr")  # Load all the rgb renders

    norm_factors = []  # Initialize the list that will contain all the norm factors
    samples = []

    for index, transient in enumerate(folders):
        samples.append(transient.split("\\")[-1].split("_")[5])  # Extract from the name of the file the number of used samples

        images = tr.transient_loader(img_path=transient,
                                     np_path=out_path / ("np_transient_" + samples[index] + "_samples.npy"),
                                     store=(not exists(out_path / ("np_transient_" + samples[index] + "_samples.npy"))))  # Load the transient
        tot_img = tr.total_img(images=images)  # Generate the total img
        norm_factors.append(compute_norm_factor(tot_img=tot_img, o_img_path=rgb_images[index]))  # Compute the normalization factor and append to the list of all the normalization factors

        if samples[index][-1] == "k":
            samples[index] = samples[index][:-1] + "000"  # If the samples is identified using k as thousand replace it with "000"
        samples[index] = int(samples[index])  # Convert the samples value from string to int

    expected = [n_samples * 1.7291 for n_samples in samples]  # Compute the expected value of the normalization factor

    plt.plot(samples, expected, '--', label="Expected normalization factor:\n" + r"$<n\_samples> \cdot 1.7291$")  # Plot the expected values of the normalization factor
    for i, j in zip(samples, expected):  # Add the annotations on the point
        if j != expected[-1]:
            plt.annotate(str(format_float_scientific(j, precision=1, exp_digits=1, trim="-")), xy=(i + 3000, j))
        else:
            plt.annotate(str(format_float_scientific(j, precision=1, exp_digits=1, trim="-")), xy=(i - 10000, j))
    plt.plot(samples, norm_factors, 'o', label="Computed normalization factor")  # Plot the computed values of the normalization factor
    plt.xticks(range(min(samples), max(samples) + 10000, 15000), labels=[format_float_scientific(val, precision=1, exp_digits=1, trim="-") for val in range(min(samples), max(samples) + 10000, 15000)])  # Modify the xtiks to match the samples number
    plt.yticks(range(20000, 160000, 20000), labels=[format_float_scientific(val, precision=1, exp_digits=1, trim="-") for val in range(20000, 160000, 20000)])  # Modify the ytiks to match the expected values
    plt.xlabel("Number of samples")  # Add a label on the x axis
    plt.ylabel("Normalization factor")  # Add a label on the y axis
    plt.grid()  # Add the grid to the plot
    plt.legend()  # Add the legend to the plot
    plt.savefig(str(out_path / "norm_factor_plot.svg"))  # Save the plot as a .svg file
    plt.close()
