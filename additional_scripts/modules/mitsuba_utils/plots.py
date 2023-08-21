import numpy as np
from time import time
from matplotlib import pyplot as plt
from math import radians, floor, cos
from os.path import exists
from .utils import extract_data_from_file, compute_norm_factor
from .. import transient_utils as tr, utilities as ut


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
        l_x = [-elm for elm in r_x[::-1]]
        x_label_full = [
            *l_x,
            *r_x,
        ]  # Define the all the ticks label to match the pixel position instead of the distance from the center
        # Code from: https://www.geeksforgeeks.org/python-ways-to-concatenate-two-lists/
        x_label = []  # List that will contain only the label of the desired ticks
        indexes = (
            []
        )  # List that will contain the index where the correspondent ticks are located
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
        l_x = [-elm for elm in r_x[::-1]]
        l_x.append(0)
        x_label_full = [
            *l_x,
            *r_x,
        ]  # Define the all the ticks label to match the pixel position instead of the distance from the center
        # Code from: https://www.geeksforgeeks.org/python-ways-to-concatenate-two-lists/

        x_label = []  # List that will contain only the label of the desired ticks
        indexes = (
            []
        )  # List that will contain the index where the correspondent ticks are located
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

    plt.scatter(
        x, real, 6, label="Measured distances", color=color
    )  # Plot the measured data as dots
    plt.plot(
        x,
        model,
        "--",
        label="Ideal decaying of the intensities value",
        color="tab:blue",
    )  # Plot the ground truth value as a dashed line
    plt.locator_params(axis="x", nbins=8)
    plt.xticks(ticks, x_label)
    plt.xlabel("Pixel position")  # Define the label on the x axes
    plt.ylabel(
        r"Radiance value on the red channel [$W/(m^{2}·sr)$]"
    )  # Define the label on the y axes
    plt.grid()  # Add the grid to the plot
    plt.legend()  # Add the legend to the plot
    plt.savefig(name + ext)  # Save the generated plot
    plt.close()  # Close the currently open plot


def save_cross_section_plot(
    theta_r, theta_c, row_distance, col_distance, r_values, c_values, output, ext=".svg"
):
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

    y_r = [
        pow(cos(radians(theta)), 3)
        * np.mean(r_values[int(len(r_values) / 2) - 3 : int(len(r_values) / 2) + 3])
        for theta in theta_r
    ]  # Define the ground truth model for the main row as the cos^3(theta) * the np.max value of the radiance (outliers excluded)
    y_c = [
        pow(cos(radians(theta)), 3)
        * np.mean(c_values[int(len(c_values) / 2) - 3 : int(len(c_values) / 2) + 3])
        for theta in theta_c
    ]  # Define the ground truth model for the main column

    plot_generator(
        row_distance, r_values, y_r, 80, output + "_row", ext, "tab:orange"
    )  # Plot the main row plot
    plot_generator(
        col_distance, c_values, y_c, 60, output + "_col", ext, "tab:green"
    )  # Plot the main column plot

    end = time()
    print("Process concluded in %.2f sec\n" % (round((end - start), 2)))


def save_distance_plot(dist, radiance, output, ext=".svg", annotation=False):
    """
    Function to generate and save the distance plot
    :param dist: vector containing al the measured distance values
    :param radiance: vector containing al the measured radiance values
    :param output: name of the output file
    :param ext: extension of the saved file
    :param annotation: boolean value to define if the plot will have annotation on the data
    """

    x, y = ut.generate_quadratic_model(
        np.min(dist), np.max(dist), np.max(radiance), 1000
    )  # Build the ground truth quadratic model

    plt.plot(
        x, y, "--", label="Ideal decaying of the intensities value"
    )  # Plot the ground truth value as a dashed line
    plt.plot(
        dist, radiance, "o", label="Measured distances"
    )  # Plot the measured data as dots
    plt.xlabel("Distances value [m]")  # Define the label on the x axis
    plt.ylabel(
        r"Radiance value on the red channel [$W/(m^{2}·sr)$]"
    )  # Define the label on the y axis
    if annotation:
        for i, j in zip(dist, radiance):
            plt.annotate(
                str(round(j, 3)), xy=(i, j)
            )  # Add the annotation to each measured data point
    plt.grid()  # Add the grid to the plot
    plt.legend()  # Add the legend to the plot
    plt.savefig(output + ext)  # Save the generated plot
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

    lin = np.linspace(step, max_value, 1000)  # Build the ground truth linear model

    x = [x for x in range(step, max_value + step, step)]  # x axes of the measured data

    plt.figure(figsize=(10, 8))
    plt.plot(
        lin, lin, "--", label="Ideal distances values"
    )  # Plot the ground truth value as a dashed line
    plt.step(
        x,
        [x * 1000 for x in dist],
        where="post",
        linewidth=2,
        label="Measured distances",
    )  # Plot the measured data as steps
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
    files = ut.read_files(
        str(in_path), "txt"
    )  # Load the path of all the files in the input folder with extension .exr
    dist, radiance = extract_data_from_file(
        files
    )  # Extract the distance and radiance values from the files
    save_distance_plot(
        dist, radiance, str(out_name)
    )  # Generate and save the standard plot


def mm_distance_plot(in_path, step, max_value, out_name):
    """
    Function to generate and save the plot regarding the millimiter distance -> evaluate the quantization
    :param in_path: input path containing all the txt file of the measured distances
    :param step: number of skipped ticks between two visualized ones
    :param max_value: maximum value visualized on the x axes
    :param out_name: path and name (no extension) of the output file
    """
    files = ut.read_files(
        str(in_path), "txt"
    )  # Load the path of all the files in the input folder with extension .exr
    dist, radiance = extract_data_from_file(
        files
    )  # Extract the distance and radiance values from the files
    save_millimiter_plot(
        dist, step, max_value, str(out_name)
    )  # Generate and save the millimiter plot


def plot_norm_factor(folder_path, rgb_path, out_path):
    """
    Function that compute the normalization factor of different transient setups and plot the results
    :param folder_path: path of the folder containing all the transient information
    :param rgb_path: path to the folder containing all the rgb images
    :param out_path: path of the output directory
    """

    folders = ut.read_folders(
        folder_path
    )  # Load all the subdirectory of the folder containing the various transient
    rgb_images = ut.read_files(
        file_path=rgb_path, extension="exr"
    )  # Load all the rgb renders

    norm_factors = []  # Initialize the list that will contain all the norm factors
    samples = []

    for index, transient in enumerate(folders):
        samples.append(
            transient.split("\\")[-1].split("_")[5]
        )  # Extract from the name of the file the number of used samples

        images = tr.loader.transient_loader(
            img_path=transient,
            np_path=out_path / ("np_transient_" + samples[index] + "_samples.npy"),
            store=(
                not exists(
                    out_path / ("np_transient_" + samples[index] + "_samples.npy")
                )
            ),
        )  # Load the transient
        tot_img = tr.tools.total_img(images=images)  # Generate the total img
        norm_factors.append(
            compute_norm_factor(tot_img=tot_img, o_img_path=rgb_images[index])
        )  # Compute the normalization factor and append to the list of all the normalization factors

        if samples[index][-1] == "k":
            samples[index] = (
                samples[index][:-1] + "000"
            )  # If the samples is identified using k as thousand replace it with "000"
        samples[index] = int(
            samples[index]
        )  # Convert the samples value from string to int

    expected = [
        n_samples * 1.7291 for n_samples in samples
    ]  # Compute the expected value of the normalization factor

    plt.plot(
        samples,
        expected,
        "--",
        label="Expected normalization factor:\n" + r"$<n\_samples> \cdot 1.7291$",
    )  # Plot the expected values of the normalization factor
    for i, j in zip(samples, expected):  # Add the annotations on the point
        if j != expected[-1]:
            plt.annotate(
                str(np.format_float_scientific(j, precision=1, exp_digits=1, trim="-")),
                xy=(i + 3000, j),
            )
        else:
            plt.annotate(
                str(np.format_float_scientific(j, precision=1, exp_digits=1, trim="-")),
                xy=(i - 10000, j),
            )
    plt.plot(
        samples, norm_factors, "o", label="Computed normalization factor"
    )  # Plot the computed values of the normalization factor
    plt.xticks(
        range(np.min(samples), np.max(samples) + 10000, 15000),
        labels=[
            np.format_float_scientific(val, precision=1, exp_digits=1, trim="-")
            for val in range(np.min(samples), np.max(samples) + 10000, 15000)
        ],
    )  # Modify the xtiks to match the samples number
    plt.yticks(
        range(20000, 160000, 20000),
        labels=[
            np.format_float_scientific(val, precision=1, exp_digits=1, trim="-")
            for val in range(20000, 160000, 20000)
        ],
    )  # Modify the ytiks to match the expected values
    plt.xlabel("Number of samples")  # Add a label on the x axis
    plt.ylabel("Normalization factor")  # Add a label on the y axis
    plt.grid()  # Add the grid to the plot
    plt.legend()  # Add the legend to the plot
    plt.savefig(str(out_path / "norm_factor_plot.svg"))  # Save the plot as a .svg file
    plt.close()
