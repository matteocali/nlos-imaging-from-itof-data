import sys
import getopt
import os
from pathlib import Path
import glob
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_input = os.getcwd()  # Argument containing the input directory
    arg_output = "distance_plot"  # Argument containing the output file name
    arg_mm = False  # Argument to select if we are considering the millimeter case or not
    arg_step = 1  # Argument containing the increasing step in the millimiter test in mm
    arg_max = 30  # Argument containing the maximum distance in the millimiter test in mm
    arg_help = "{0} -i <input> -o <output> -m <millimiter> (default = False) -s <step> (default = 1) -v <max_value> (default = 30)".format(argv[0])  # Help string

    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:m:s:v:", ["help", "input=", "output=", "millimiter=", "step=", "value="])  # Recover the passed options and arguments from the command line (if any)
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
            arg_output = arg  # Set the output file name
        elif opt in ("-m", "--millimiter"):
            arg_mm = bool(arg)  # Set the millimiter flag
        elif opt in ("-s", "--step"):
            arg_step = int(arg)  # Set the step value
        elif opt in ("-v", "--value"):
            arg_max = int(arg)  # Set the step value

    print('Input folder:', arg_input)
    print('Output folder:', arg_output)
    print()

    return [arg_input, arg_output, arg_mm, arg_step, arg_max]


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


def extract_data_from_file(files):
    """
    Function to extract the distance and radiance value from the txt files
    :param files: list of all the paths of all the txt file that has to be analyzed
    :return: list containing all the distances and another list containing all the radiance values
    """

    distances = []  # Define the empty list that will contain the distances values
    radiances = []  # Define the empty list that will contain the radiances values
    for file in files:  # For each file
        with open(file, "r") as reader:  # Open the file
            content = reader.readlines()  # Read the content
            distances.append(float(content[1].split(":")[1][1:]))  # From the second line extract the distance value parsing the string
            radiances.append(float(content[2].split(":")[1][1:]))  # From the third line extract the radiance value parsing the string

    return [distances, radiances]


def generate_quadratic_model(min_x, max_x, max_y, precision):
    """
    Function that define the ground truth quadratic model
    :param min_x: minimum measured value of distances
    :param max_x: maximum measured value of distances
    :param max_y: maximum measured value of radiance
    :param precision: number of samples inside the linear space
    :return: x and y value of the quadratic model
    """
    x = np.linspace(min_x, max_x, precision)  # Define the x vector as a linear space
    #y = 25.471 / pow(x, 2)
    scaling = max_y * pow(min_x, 2)  # Define the scale factor to center the quadratic model on the measured data
    y = scaling / pow(x, 2)  # Build the quadratic model
    return [x, y]


def save_plot(dist, radiance, output, ext = ".svg", annotation = False):
    """
    Function to generate and save the plot
    :param dist: vector containing al the measured distance values
    :param radiance: vector containing al the measured radiance values
    :param output: name of the output file
    :param ext: extension of the saved file
    :param annotation: boolean value to define if the plot will have annotation on the data
    """

    x, y = generate_quadratic_model(min(dist), max(dist), max(radiance), 1000)  # Build the ground truth quadratic model

    plt.plot(x, y, '--', label="Ideal decaying of the intensities value")  # Plot the ground truth value as a dashed line
    plt.plot(dist, radiance, 'o', label="Measured distances")  # Plot the measured data as dots
    plt.xlabel("Distances value [m]")  # Define the label on the x axis
    plt.ylabel(r"Radiance value on he red channel [$W/(m^{2}·sr)$]")  # Define the label on the y axis
    if annotation:
        for i, j in zip(dist, radiance):
            plt.annotate(str(round(j, 3)), xy=(i, j))  # Add the annotation to each measured data point
    plt.grid()  # Add the grid to the plot
    plt.legend()  # Add the legend to the plot
    plt.savefig(output+ext)  # Save the generated plot
    plt.show()  # Show the plot


def save_millimiter_plot(dist, step, max_value, output, ext = ".svg"):
    """
    Function to generate and save the plot
    :param dist: vector containing al the measured distance values
    :param output: name of the output file
    :param ext: extension of the saved file
    """

    lin = np.linspace(step, max_value, 1000)  # Build the ground truth linear model

    x = [x for x in range(step, max_value + step, step)]  # x axes of the measured data

    plt.figure(figsize=(10, 8))
    plt.plot(lin, lin, '--', label="Ideal distances values")  # Plot the ground truth value as a dashed line
    plt.step(x, [x * 1000 for x in dist], linewidth=2, label="Measured distances")  # Plot the measured data as steps
    plt.xticks(list(range(step, 32, 2)))  # Define the values displayed on the x axes
    plt.yticks(list(range(step, 36, 1)))  # Define the values displayed on the y axes
    plt.xlabel("Distances value [mm]")  # Define the label on the x axes
    plt.ylabel("Distances value [mm]")  # Define the label on the y axes
    plt.grid()  # Add the grid to the plot
    plt.legend()  # Add the legend to the plot
    plt.savefig(output + ext)  # Save the generated plot
    plt.show()  # Show the plot


if __name__ == '__main__':
    arg_input, arg_output, arg_mm, arg_step, arg_max = arg_parser(sys.argv)  # Recover the input and output folder from the console args

    files = reed_files(str(arg_input), "txt")  # Load the path of all the files in the input folder with extension .exr

    dist, radiance = extract_data_from_file(files)  # Extract the distance and radiance values from the files

    if not arg_mm:
        save_plot(dist, radiance, arg_output)  # Generate and save the standard plot
    else:
        save_millimiter_plot(dist, arg_step, arg_max, arg_output)  # Generate and save the millimiter plot

    print("Process concluded")