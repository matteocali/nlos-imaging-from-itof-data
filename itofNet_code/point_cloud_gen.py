import os
import getopt
import sys
import numpy as np
import pickle
import open3d as o3d
from tqdm import trange
from matplotlib import pyplot as plt
from pathlib import Path
from modules.utils.point_clouds import point_cloud_gen


def arg_parser(argv):
    """
    Function used to parse the input argument given through the command line (code form https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/)
    :param argv: system arguments
    :return: list containing the input and output path
    """

    arg_in = os.getcwd()  # Argument containing the input directory
    arg_out = ""  # Argument containing the output directory
    arg_help = "{0} -i <input> -o <output>".format(argv[0])  # Help string

    try:
        opts, args = getopt.getopt(
            argv[1:], "hi:o:", ["help", "input=", "output="]
        )  # Recover the passed options and arguments from the command line (if any)
    except:
        print(arg_help)  # If the user provide a wrong options print the help string
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # Print the help message
            sys.exit(2)
        elif opt in ("-i", "--input"):
            arg_in = Path(arg)  # Set the input directory
        elif opt in ("-o", "--output"):
            arg_out = Path(arg)  # Set the output directory

    print("Input path: ", arg_in)
    print("Output path: ", arg_out)
    print()

    return [arg_in, arg_out]


if __name__ == "__main__":
    arg_in, arg_out = arg_parser(
        sys.argv
    )  # Recover the input and output folder from the console args

    # Load the pickle data
    data_path = Path(arg_in) / "results.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    depth = data["pred"]["depth"]
    gt_depth = data["gt"]["depth"]

    # Defien the output folder
    out_folder = Path(arg_out)
    plot_folder = out_folder / "plots"
    pc_folder = out_folder / "point_cloud"
    depth_folder = plot_folder / "depthmap"
    plot_folder.mkdir(parents=True, exist_ok=True)
    depth_folder.mkdir(parents=True, exist_ok=True)
    pc_folder.mkdir(parents=True, exist_ok=True)

    # Create the point cloud
    for i in trange(depth.shape[0], desc="Generating point cloud"):
        # Generate the point cloud fro the predicted depth map and the ground truth depth map
        pc_pred = point_cloud_gen(depthmap=depth[i, ...])
        pc_gt = point_cloud_gen(depthmap=gt_depth[i, ...])

        # Save the point cloud
        o3d.io.write_point_cloud(
            str(pc_folder / f"point_cloud_{i+1}_pred.ply"), pc_pred
        )
        o3d.io.write_point_cloud(str(pc_folder / f"point_cloud_{i+1}_gt.ply"), pc_gt)

        # Convert the pc to numpy
        pc_pred = np.asarray(pc_pred.points)
        pc_gt = np.asarray(pc_gt.points)

        # Visualize the point cloud and the mesh in matplotlib
        titles = ["Prediction", "Ground Truth", "Prediction & Ground truth"]
        data = [pc_pred, pc_gt]
        z_min, z_max = min(np.min(pc_gt[:, 2]), np.min(pc_pred[:, 2])), max(
            np.max(pc_gt[:, 2]), np.max(pc_pred[:, 2])
        )

        plt.rcParams["font.family"] = "serif"

        fig = plt.figure(figsize=(19, 6))
        axes = []
        for j in trange(len(titles), leave=False, desc="Buildimng the plot"):
            ax = fig.add_subplot(1, 3, j + 1, projection="3d")
            ax.view_init(elev=10.0, azim=45)

            if j < len(titles) - 1:
                ax.scatter(
                    data[j][:, 0],
                    data[j][:, 1],
                    data[j][:, 2],
                    cmap="viridis",
                    c=data[j][:, 2],
                    linewidth=0.5,
                    edgecolors="black",
                    s=10,
                )
            else:
                ax.scatter(
                    data[0][:, 0],
                    data[0][:, 1],
                    data[0][:, 2],
                    linewidth=0.5,
                    edgecolors="black",
                    s=10,
                )
                ax.scatter(
                    data[1][:, 0],
                    data[1][:, 1],
                    data[1][:, 2],
                    linewidth=0.5,
                    edgecolors="black",
                    s=10,
                    marker="^",
                )

                bapad = plt.rcParams["legend.borderaxespad"]
                fontsize = plt.rcParams["font.size"]
                axline = plt.rcParams[
                    "axes.linewidth"
                ]  # need this, otherwise the result will be off by a few pixels
                pad_points = (
                    bapad * fontsize + axline
                )  # padding is defined in relative to font size
                pad_inches = pad_points / 72.0  # convert from points to inches
                pad_pixels = (
                    pad_inches * fig.dpi
                )  # convert from inches to pixels using the figure's dpi

                # Find how may pixels there are on the x-axis
                x_pixels = ax.transAxes.transform((1, 0)) - ax.transAxes.transform(
                    (0, 0)
                )
                # Compute the ratio between the pixel offset and the total amount of pixels
                pad_xaxis = pad_pixels / x_pixels[0]

                # Set the legend
                ax.legend(["Prediction", "Ground truth"], loc=(39.5 * pad_xaxis, 0.3))

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_zlim(z_min, z_max)
            ax.set_title(titles[j], pad=0, fontsize=20)
            axes.append(ax)

        plt.tight_layout()
        plt.savefig(plot_folder / f"point_cloud_{i+1}_{45}.pdf")
        plt.close
        del fig, axes

        # Plot the depth map and save it
        fig = plt.figure(figsize=(8, 8))
        plt.matshow(gt_depth[i, ...].T, cmap="viridis")
        plt.savefig(depth_folder / f"depthmap_{i+1}.png")
        plt.close()
