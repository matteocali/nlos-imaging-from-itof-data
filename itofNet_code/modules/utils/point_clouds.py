import numpy as np
import open3d as o3d
from cv2 import (
    fisheye,
    CV_16SC2,
    remap,
    INTER_LINEAR,
    undistort,
    initUndistortRectifyMap,
)


def k_matrix_calculator(h_fov: float, img_shape: list) -> np.ndarray:
    """
    Function that compute the k matrix of a camera (matrix of the intrinsic parameters)
    :param h_fov: horizontal FOV (filed of view) of the camera (in degrees)
    :param img_shape: image size in pixel [n_pixel_row, n_pixel_col]
    :return: k matrix
    """

    v_fov = 2 * np.degrees(
        np.arctan((img_shape[1] / img_shape[0]) * np.tan(np.radians(h_fov / 2)))
    )
    f_x = (img_shape[0] / 2) / np.tan(np.radians(h_fov / 2))
    f_y = (img_shape[1] / 2) / np.tan(np.radians(v_fov / 2))
    if img_shape[0] % 2 == 0:
        x = (img_shape[0] - 1) / 2
    else:
        x = img_shape[0] / 2
    if img_shape[1] % 2 == 0:
        y = (img_shape[1] - 1) / 2
    else:
        y = img_shape[1] / 2

    return np.array([[f_x, 0, x], [0, f_y, y], [0, 0, 1]], dtype=np.float32)


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
    mask_valid = 1.0 * (depth < 30000)
    depth[depth > 30000] = 0
    shape_depth = (depth.shape[1], depth.shape[0])

    if dm == "FISHEYE":
        [map1, map2] = fisheye.initUndistortRectifyMap(
            k_real, d_real, np.eye(3), k_ideal, shape_depth, CV_16SC2
        )
        depth = remap(depth, map1, map2, INTER_LINEAR)
        mask_valid = remap(mask_valid, map1, map2, INTER_LINEAR)

    elif dm == "STANDARD":
        depth = undistort(depth, k_real, d_real, None, k_ideal)
        mask_valid = undistort(mask_valid, k_real, d_real, None, k_ideal)

    else:
        [map1, map2] = initUndistortRectifyMap(
            k_real, d_real, np.eye(3), k_ideal, shape_depth, CV_16SC2
        )
        depth = remap(depth, map1, map2, INTER_LINEAR)
        mask_valid = remap(mask_valid, map1, map2, INTER_LINEAR)

    mask_valid_positive = mask_valid > 0
    depth[mask_valid_positive] = np.divide(
        depth[mask_valid_positive], mask_valid[mask_valid_positive]
    )

    z_matrix = depth
    x_matrix = (np.tile(np.arange(z_matrix.shape[1]), (z_matrix.shape[0], 1))).astype(
        dtype=float
    )  # [[0,1,2,3,...],[0,1,2,3,..],...]
    y_matrix = np.tile(np.arange(z_matrix.shape[0]), (z_matrix.shape[1], 1)).T.astype(
        dtype=float
    )  # [....,[1,1,1,1, ...][0,0,0,0,...]]

    x_undist_matrix = np.zeros_like(x_matrix, dtype=float)
    y_undist_matrix = np.zeros_like(y_matrix, dtype=float)
    z_undist_matrix = np.zeros_like(z_matrix, dtype=float)

    k_1 = np.linalg.inv(k_ideal)

    radial_dir = np.zeros([x_matrix.shape[0], x_matrix.shape[1], 3])
    for x in range(x_matrix.shape[0]):
        for y in range(x_matrix.shape[1]):
            prod = np.dot(k_1, np.asarray([x_matrix[x, y], y_matrix[x, y], 1]))
            prod = prod / np.linalg.norm(prod)

            x_undist_matrix[x, y] = z_matrix[x, y] * prod[0]
            y_undist_matrix[x, y] = z_matrix[x, y] * prod[1]
            z_undist_matrix[x, y] = z_matrix[x, y] * prod[2]
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
    k_matrix = k_matrix_calculator(
        60, [depthmap.shape[0], depthmap.shape[1]]
    )  # Calculate the K matrix

    # Undistort the depth map and find the x, y, z coordinates of the points in the camera coordinates system
    pc = undistort_depthmap(
        dph=np.copy(depthmap),
        dm="STANDARD",
        k_ideal=k_matrix,
        k_real=k_matrix,
        d_real=np.array([[0, 0, 0, 0, 0]], dtype=np.float32),
    )[0]

    n_points = np.count_nonzero(
        pc[:, :, 0]
    )  # Count the number of points that actually corresponds to an object
    coordinates = np.zeros(
        [n_points, 3]
    )  # Create a matrix to store the coordinates of the points
    coordinates[:, 0] = pc[:, :, 0][
        np.where(pc[:, :, 0] != 0)
    ]  # Store the x coordinates of the points
    coordinates[:, 1] = pc[:, :, 1][
        np.where(pc[:, :, 1] != 0)
    ]  # Store the y coordinates of the points
    coordinates[:, 2] = pc[:, :, 2][
        np.where(pc[:, :, 2] != 0)
    ]  # Store the z coordinates of the points

    ## Generate the point cloud
    pcd = o3d.geometry.PointCloud()  # Create a new point cloud
    pcd.points = o3d.utility.Vector3dVector(coordinates)  # Set the points
    pcd.estimate_normals(fast_normal_computation=True)  # Estimate the normals
    pcd = pcd.normalize_normals()  # Normalize the normals

    ## Remove the outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd
