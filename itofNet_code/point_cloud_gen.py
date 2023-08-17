import os
import getopt
import sys
import numpy as np
import pickle
import open3d as o3d
from tqdm import trange
from matplotlib import pyplot as plt
from pathlib import Path
from cv2 import fisheye, CV_16SC2, remap, INTER_LINEAR, undistort, initUndistortRectifyMap


def k_matrix_calculator(h_fov: float, img_shape: list) -> np.ndarray:
    """
    Function that compute the k matrix of a camera (matrix of the intrinsic parameters)
    :param h_fov: horizontal FOV (filed of view) of the camera (in degrees)
    :param img_shape: image size in pixel [n_pixel_row, n_pixel_col]
    :return: k matrix
    """

    v_fov = 2 * np.degrees(np.arctan((img_shape[1] / img_shape[0]) * np.tan(np.radians(h_fov / 2))))
    f_x = (img_shape[0] / 2) / np.tan(np.radians(h_fov / 2))
    f_y = (img_shape[1] / 2) / np.tan(np.radians(v_fov / 2))
    if img_shape[0] % 2 == 0:
        x = (img_shape[0] - 1) / 2
    else:
        x = img_shape[0] / 2
    if img_shape[1] % 2 == 0:
        y = (img_shape[1] - 1) / 2
    else:
        y = img_shape[1]/2

    return np.array([[f_x, 0, x],
                     [0, f_y, y],
                     [0, 0, 1]], dtype=np.float32)


def undistort_depthmap(dph, dm, k_ideal, k_real, d_real):
    """
    Undistort depth map using calibration output parameters
    :param dph: depthmap (1 channel image)
    :param dm: string with the name of the camera model (FISHEYE, RADIAL, RATIONAL)
    :param k_ideal: Camera matrix
    :param k_real: Camera matrix
    :param d_real: Distortion coefficient
    :return depthmap: undistorted depthmap with 3 dimension (x-axis coordinates, y-axis coordinates, z-coordinates)
    :return mask_valid_positive: validity mask (1=valid points, 0= oor or invalid dots)
    :return radial_dir: cosine(angle between optical axis and the pixel direction)
    """

    depth = dph.copy()
    mask_valid = 1.0*(depth < 30000)
    depth[depth > 30000] = 0
    shape_depth = (depth.shape[1], depth.shape[0])

    if dm == 'FISHEYE':
        [map1, map2] = fisheye.initUndistortRectifyMap(k_real, d_real, np.eye(3), k_ideal, shape_depth, CV_16SC2)
        depth = remap(depth, map1, map2, INTER_LINEAR)
        mask_valid = remap(mask_valid, map1, map2, INTER_LINEAR)

    elif dm == 'STANDARD':
        depth = undistort(depth, k_real, d_real, None, k_ideal)
        mask_valid = undistort(mask_valid, k_real, d_real, None, k_ideal)

    else:
        [map1, map2] = initUndistortRectifyMap(k_real, d_real, np.eye(3), k_ideal, shape_depth, CV_16SC2)
        depth = remap(depth, map1, map2, INTER_LINEAR)
        mask_valid = remap(mask_valid, map1, map2, INTER_LINEAR)

    mask_valid_positive = mask_valid > 0
    depth[mask_valid_positive] = np.divide(depth[mask_valid_positive], mask_valid[mask_valid_positive])

    z_matrix = depth
    x_matrix = (np.tile(np.arange(z_matrix.shape[1]), (z_matrix.shape[0], 1))).astype(dtype=float)    # [[0,1,2,3,...],[0,1,2,3,..],...]
    y_matrix = np.tile(np.arange(z_matrix.shape[0]), (z_matrix.shape[1], 1)).T.astype(dtype=float)  # [....,[1,1,1,1, ...][0,0,0,0,...]]

    x_undist_matrix = np.zeros_like(x_matrix, dtype=float)
    y_undist_matrix = np.zeros_like(y_matrix, dtype=float)
    z_undist_matrix = np.zeros_like(z_matrix, dtype=float)

    k_1 = np.linalg.inv(k_ideal)

    radial_dir = np.zeros([x_matrix.shape[0], x_matrix.shape[1], 3])
    for x in range(x_matrix.shape[0]):
        for y in range(x_matrix.shape[1]):
            prod = np.dot(k_1, np.asarray([x_matrix[x, y], y_matrix[x, y], 1]))
            prod = prod/np.linalg.norm(prod)

            x_undist_matrix[x, y] = z_matrix[x, y]*prod[0]
            y_undist_matrix[x, y] = z_matrix[x, y]*prod[1]
            z_undist_matrix[x, y] = z_matrix[x, y]*prod[2]
            radial_dir[x, y, :] = prod

    depthmap = np.stack([x_undist_matrix, y_undist_matrix, z_undist_matrix], axis=2)

    return depthmap, mask_valid_positive, radial_dir


def point_cloud_gen(depthmap: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Function used to pgenerate the pointcloud starting from the depthmap\n
    Params:
        - depthmap (np.ndarray): The depth map\n
    Return:
        - depthmap (np.ndarray): The preprocessed depth map
    """
    
    ## Extract the point coordinate wrt the camera
    k_matrix = k_matrix_calculator(60, [depthmap.shape[0], depthmap.shape[1]])  # Calculate the K matrix

    # Undistort the depth map and find the x, y, z coordinates of the points in the camera coordinates system
    pc = undistort_depthmap(dph=np.copy(depthmap),
                            dm="STANDARD",
                            k_ideal=k_matrix,
                            k_real=k_matrix,
                            d_real=np.array([[0, 0, 0, 0, 0]], dtype=np.float32))[0]
    
    n_points = np.count_nonzero(pc[:, :, 0])                     # Count the number of points that actually corresponds to an object
    coordinates = np.zeros([n_points, 3])                        # Create a matrix to store the coordinates of the points
    coordinates[:, 0] = pc[:, :, 0][np.where(pc[:, :, 0] != 0)]  # Store the x coordinates of the points
    coordinates[:, 1] = pc[:, :, 1][np.where(pc[:, :, 1] != 0)]  # Store the y coordinates of the points
    coordinates[:, 2] = pc[:, :, 2][np.where(pc[:, :, 2] != 0)]  # Store the z coordinates of the points

    ## Generate the point cloud
    pcd = o3d.geometry.PointCloud()                       # Create a new point cloud
    pcd.points = o3d.utility.Vector3dVector(coordinates)  # Set the points
    pcd.estimate_normals(fast_normal_computation=True)    # Estimate the normals
    pcd = pcd.normalize_normals()                         # Normalize the normals

    ## Remove the outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd


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
        opts, args = getopt.getopt(argv[1:], "hi:o:", ["help", "input=", "output="])  # Recover the passed options and arguments from the command line (if any)
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

    print('Input path: ', arg_in)
    print('Output path: ', arg_out)
    print()

    return [arg_in, arg_out]


if __name__ == '__main__':
    arg_in, arg_out = arg_parser(sys.argv)  # Recover the input and output folder from the console args

    # Load the numpy data
    data_path = Path(arg_in) / 'results.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    depth = data['pred']['depth']
    gt_depth = data['gt']['depth']

    # Defien the output folder
    out_folder = Path(arg_out)
    plot_folder = out_folder / 'plots'
    pc_folder = out_folder / 'point_cloud'
    depth_folder = plot_folder / 'depthmap'
    plot_folder.mkdir(parents=True, exist_ok=True)
    depth_folder.mkdir(parents=True, exist_ok=True)
    pc_folder.mkdir(parents=True, exist_ok=True)

    # Create the point cloud
    for i in trange(depth.shape[0], desc="Generating point cloud"):
        # Generate the point cloud fro the predicted depth map and the ground truth depth map
        pc_pred = point_cloud_gen(depthmap=depth[i, ...])
        pc_gt = point_cloud_gen(depthmap=gt_depth[i, ...])

        # Save the point cloud
        o3d.io.write_point_cloud(str(pc_folder / f"point_cloud_{i+1}_pred.ply"), pc_pred)
        o3d.io.write_point_cloud(str(pc_folder / f"point_cloud_{i+1}_gt.ply"), pc_gt)

        # Convert the pc to numpy
        pc_pred = np.asarray(pc_pred.points)
        pc_gt = np.asarray(pc_gt.points)
        
        # Visualize the point cloud and the mesh in matplotlib
        titles = ['Prediction', 'Ground Truth', 'Prediction & Ground truth']
        data = [pc_pred, pc_gt]
        z_min, z_max = min(np.min(pc_gt[:, 2]), np.min(pc_pred[:, 2])), max(np.max(pc_gt[:, 2]), np.max(pc_pred[:, 2]))

        plt.rcParams['font.family'] = 'serif'

        fig = plt.figure(figsize=(19, 6))
        axes = []
        for j in trange(len(titles), leave=False, desc='Buildimng the plot'):
            ax = fig.add_subplot(1, 3, j + 1, projection='3d')
            ax.view_init(elev=10., azim=45)

            if j < len(titles) - 1:
                ax.scatter(data[j][:, 0], data[j][:, 1], data[j][:, 2], cmap='viridis', c=data[j][:, 2], linewidth=0.5, edgecolors='black', s=10)
            else:
                ax.scatter(data[0][:, 0], data[0][:, 1], data[0][:, 2], linewidth=0.5, edgecolors='black', s=10)
                ax.scatter(data[1][:, 0], data[1][:, 1], data[1][:, 2], linewidth=0.5, edgecolors='black', s=10, marker='^')

                bapad = plt.rcParams['legend.borderaxespad']
                fontsize = plt.rcParams['font.size']
                axline = plt.rcParams['axes.linewidth']  # need this, otherwise the result will be off by a few pixels
                pad_points = bapad*fontsize + axline  # padding is defined in relative to font size
                pad_inches = pad_points/72.0  # convert from points to inches
                pad_pixels = pad_inches*fig.dpi  # convert from inches to pixels using the figure's dpi

                # Find how may pixels there are on the x-axis
                x_pixels = ax.transAxes.transform((1,0)) - ax.transAxes.transform((0,0))
                # Compute the ratio between the pixel offset and the total amount of pixels 
                pad_xaxis = pad_pixels/x_pixels[0]

                # Set the legend
                ax.legend(['Prediction', 'Ground truth'], loc=(39.5*pad_xaxis,0.3))

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_zlim(z_min, z_max)
            ax.set_title(titles[j], pad=0, fontsize=20)
            axes.append(ax)

        plt.tight_layout()
        plt.savefig(plot_folder / f"point_cloud_{i+1}_{45}.pdf")
        plt.close
        del fig, axes

        # Plot the depth map and save it 
        fig = plt.figure(figsize=(8, 8))
        plt.matshow(gt_depth[i, ...].T, cmap='viridis')
        plt.savefig(depth_folder / f"depthmap_{i+1}.png")
        plt.close()
