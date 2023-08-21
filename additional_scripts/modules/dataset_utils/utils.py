import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
from scipy.signal import find_peaks
from ..utilities import load_list, save_list, load_h5, save_h5, read_files, k_matrix_calculator, read_folders
from ..fermat_utils.utils import undistort_depthmap, compute_bin_center
from ..transient_utils.loader import transient_loader
from ..transient_utils.utils import phi, amp_phi_compute


def gen_filter(exp_coeff: float, sigma: float) -> np.ndarray:
    """
    This function generates a Difference of Gaussian filter with exponential falloff
    :param exp_coeff: exponential falloff
    :param sigma: standard deviation of the gaussian kernel
    :return: DoG filter with an exponential falloff
    """

    t = np.array([i / 10 for i in range(-50, 51, 1)], dtype=np.float32)
    t = t.reshape(t.size, 1)
    ind = np.where(t == 0)[0][0]
    delta = np.zeros(t.shape)
    delta[ind] = 1
    deltas = np.zeros(t.shape)
    deltas[ind + 1] = 1

    pd = delta + deltas * (-np.exp(-exp_coeff * 0.001))
    g = np.exp(-np.square(t) / sigma**2)
    return np.convolve(pd[:, 0], g[:, 0], "same")


def compute_discont(tr: np.ndarray, exp_time: float) -> np.ndarray:
    """
    This function detects path length discontinuities in each transient
    This function computes the discontinuity map of the given transient
    :param tr: n*m matrix, n: #transient, m: #temporal bins
    :param exp_time: exposure time
    :return: n*k matrix, storing discontinuities in transients, n: #transient, k: #discontinuities per transient
    """

    # PARAMETERS
    num_of_discont = 1  # Number of discontinuity to search for
    exp_coeff = [0.3]  # Model the exponential falloff of the SPAD signal
    sigma_blur = [1]  # Difference of Gaussian, standard deviation
    whether_sort_disconts = (
        True  # Sort the discontinuity by their distance from the center of the image
    )

    # COMPUTE THE DISCONTINUITY
    n_samples = tr.shape[0]
    temp_bin_center = compute_bin_center(exp_time, tr.shape[1])
    num_of_bin_center = 1
    if num_of_bin_center == 1:
        x = np.array(temp_bin_center, dtype=np.float32)
    else:
        assert num_of_bin_center == n_samples

    all_disconts = np.empty((n_samples, num_of_discont))

    tr_vec = tr.reshape(tr.shape[0], tr.shape[1])

    for i in range(n_samples):
        if num_of_bin_center > 1:
            x = temp_bin_center[i, :]
        y = tr_vec[i]
        if np.nanmax(y) != 0:
            y = y / np.nanmax(y)

        # Convolve the transient with the DoG f, and keep the np.maximum f response
        dgy = np.full(y.shape, -np.Inf)
        for exp_val in exp_coeff:
            for sigma in sigma_blur:
                f = gen_filter(exp_val, sigma)
                dgy_one = np.maximum(
                    np.convolve(y, f, "same"), np.convolve(y, np.flip(f), "same")
                )
                dgy = np.maximum(dgy, dgy_one)
        if np.nanmax(dgy) != 0:
            dgy = dgy / np.nanmax(dgy)

        # Discontinuities correspond to larger f responses
        # noinspection PyUnboundLocalVariable
        locs_peak, peak_info = find_peaks(
            dgy, prominence=0
        )  # FIX THE LOCS INDEX SHOULD BE A VALUE OF X
        if locs_peak.size != 0:
            # noinspection PyUnboundLocalVariable
            locs_peak = x[locs_peak]
        p = peak_info["prominences"]
        if p.size != 0 and np.all(p == p[0]):
            ind_p = np.array([i for i in range(0, p.size)], dtype=int)
        else:
            ind_p = np.argsort(p)[::-1]  # Sort the prominence in descending order

        if ind_p.size == 0:
            ind_p = 0
        if locs_peak.size == 0:
            locs_peak = [np.nan]
        else:
            locs_peak = locs_peak[ind_p]

        disconts = np.full([1, num_of_discont], np.nan)
        if np.size(locs_peak) >= num_of_discont:
            disconts = locs_peak[:num_of_discont]
        else:
            disconts[: np.size(locs_peak)] = locs_peak

        if whether_sort_disconts:
            disconts = np.sort(disconts)

        # store discont
        all_disconts[i, :] = disconts

    return all_disconts


def build_point_cloud(
    data: np.ndarray,
    out_path: Path,
    fov: int,
    img_size: tuple[int, int],
    alpha: np.ndarray = None,
    f_mesh: bool = True,
    visualize: bool = False,
) -> (o3d.geometry.PointCloud, o3d.geometry.TriangleMesh):
    """
    Build a point cloud (and mesh) from a depth map
    :param data: depth map
    :param out_path: folder where to save the point cloud and the mesh
    :param fov: field of view of the camera
    :param img_size: size of the image in pixel (width, height)
    :param alpha: alpha map
    :param f_mesh: if True, build a mesh
    :param visualize: flag to visualize the point cloud and the mesh
    :return: point cloud and mesh
    """

    if alpha is not None:  # If the alpha map is provided
        data = data * alpha  # Apply the alpha map

    k_matrix = k_matrix_calculator(fov, list(img_size)[::-1])  # Calculate the K matrix

    pc = undistort_depthmap(
        dph=np.copy(data),
        dm="RADIAL",
        k_ideal=k_matrix,
        k_real=k_matrix,
        d_real=np.array([[0, 0, 0, 0, 0]], dtype=np.float32),
    )[
        0
    ]  # Find the x, y, z coordinates of the points in the camera coordinates system

    n_points = np.count_nonzero(
        pc[:, :, 0]
    )  # Count the number of points that actually corresponds to an object
    t = np.zeros([n_points, 3])  # Create a matrix to store the coordinates of the points
    t[:, 0] = pc[:, :, 0][
        np.where(pc[:, :, 0] != 0)
    ]  # Store the x coordinates of the points
    t[:, 1] = pc[:, :, 1][
        np.where(pc[:, :, 1] != 0)
    ]  # Store the y coordinates of the points
    t[:, 2] = pc[:, :, 2][
        np.where(pc[:, :, 2] != 0)
    ]  # Store the z coordinates of the points

    pcd = o3d.geometry.PointCloud()  # Create a new point cloud
    pcd.points = o3d.utility.Vector3dVector(t)  # Set the points
    pcd.estimate_normals(fast_normal_computation=True)  # Estimate the normals

    objs = [pcd]

    if f_mesh:  # If the mesh is requested
        radii = [0.005, 0.01, 0.02, 0.04]  # Create a list of radii
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd=pcd, radii=o3d.utility.DoubleVector(radii)
        )  # Create the mesh
        objs.append(rec_mesh)

    if visualize:  # If the visualization is requested
        o3d.visualization.draw_geometries(
            geometry_list=objs,
            window_name="Point cloud and mesh visualization",
            point_show_normal=True,
            mesh_show_back_face=True,
            mesh_show_wireframe=True,
        )  # Visualize the point cloud and the mesh

    o3d.io.write_point_cloud(
        str(out_path / "point_cloud.ply"), pcd
    )  # Save the point cloud
    if f_mesh:
        o3d.io.write_triangle_mesh(
            str(out_path / "mesh.ply"), rec_mesh
        )  # Save the mesh

    return objs


def ampl_ratio_hists(dts_path: Path, out_path: Path) -> None:
    """
    Build the histogram of the amplification ratio
    :param dts_path: path of the dataset
    :param out_path: path of the output folder
    :return: None
    """

    if not (
        out_path.parent / "ampl_ratios"
    ).exists():  # If the output list doesn't exist
        dts_files = read_files(dts_path, "h5")  # Get the list of files in dts_path

        if len(dts_files) == 0:  # If there are no files in dts_path
            raise ValueError("The dataset folder is np.empty")  # Raise an error

        ampl_ratios = []  # Create a list to store the amplification ratio

        for dts_file in tqdm(
            dts_files, desc="Computing amplification ratio"
        ):  # For each file in dts_path
            dts = load_h5(dts_file)  # Load the file
            amp = dts["amp_itof"]  # Get the amplitude data
            amp_ratio = np.swapaxes(amp[..., 0] / amp[..., 1], 0, 1)
            ampl_ratios.append(
                np.nanmax(amp_ratio) - np.nanmin(amp_ratio)
            )  # Compute the amplification ratio and store it

        save_list(
            ampl_ratios, out_path.parent / "ampl_ratios"
        )  # Save the amplification ratio
    else:
        ampl_ratios = load_list(out_path.parent / "ampl_ratios")

    plt.hist(ampl_ratios)
    plt.title("Amplitude ratio hists")
    plt.savefig(out_path)
    plt.close()


def load_dataset(d_path: Path, out_path: Path, freqs: np.ndarray = None) -> None:
    """
    Load the dataset and save it in the out_path folder
    :param d_path: folder containing the dataset (raw output of mitsuba)
    :param out_path: folder where the dataset will be saved after processing
    :param freqs: frequencies used by the iToF sensor
    """

    if not out_path.exists():  # Create out_path if it doesn't exist
        out_path.mkdir(parents=True)

    batches_folder = read_folders(d_path)  # Get the list of batches
    data_path = []
    for batch_folder in batches_folder:
        data_folder = read_folders(batch_folder)  # Get the list of data in each batch
        data_path = (
            data_path + data_folder
        )  # Put together (in the same list) all the file present in all the batches

    for file_path in tqdm(data_path, desc="Loading dataset"):  # For each file
        file_name = str(Path(file_path).name)  # Get the file name
        tr = transient_loader(file_path)[
            :, :, :, 1
        ]  # Load the data and put them in standard form (only green channel is considered)

        if (
            freqs is not None
        ):  # If the frequencies are provided, compute the iToF amplitude and phase
            phi_data = phi(
                freqs
            )  # Compute the phi function, required to compute the iToF style output
            tr = np.swapaxes(
                np.moveaxis(tr, 0, -1), 0, 1
            )  # Reshape the tr data to match the layout that will be used in the following
            tr_phi = np.matmul(tr, phi_data.T)  # Compute the iToF transient data
            amp, phs = amp_phi_compute(
                tr_phi
            )  # Compute the amplitude and phase of the iToF transient data

            save_h5(
                file_path=out_path / file_name,
                data={
                    "data": tr,
                    "tr_itof": tr_phi,
                    "amp_itof": amp,
                    "phase_itof": phs,
                },
                fermat=False,
            )  # Save the data in the out_path folder as a h5 file
        else:
            save_h5(
                file_path=out_path / file_name, data={"data": tr}, fermat=True
            )  # Save the data in the out_path folder as a h5 file
