import numpy as np
import torch
import h5py
import scipy.constants as const
from pathlib import Path
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import *
from utils import CustomTransforms
from utils import utils
from utils.utils import depth_radial2cartesian


# Constants
OBJECTS = ["cube", "cone", "cylinder", "sphere"]
DISTANCES = [0.7, 1]
FREQUENCIES = range(int(10e6), int(70e6), int(10e6))
ACCEPTED_FREQS = int(20e6)
ACCEPTED_DTYPES = "depth"


def plane_fitting(depth):
    """
    Fits a plane to the given points.
    """

    import numpy as np

    X = np.repeat(np.expand_dims(np.array((range(depth.shape[0]))), 0), depth.shape[1], axis=0)
    Y = np.repeat(np.expand_dims(np.array((range(depth.shape[1]))), 1), depth.shape[0], axis=1)

    focal = 0.5 * 320 / np.tan(0.5 * 60 * np.pi / 180)

    Z = depth.copy()
    Z = np.flip(np.flip(Z.T, axis=0), axis=1)
    Z = depth_radial2cartesian(Z, focal)  # type: ignore

    x = X.reshape(X.size)
    y = Y.reshape(Y.size)
    z = Z.reshape(Z.size)  # type: ignore

    # Fit a plane to the points
    A = np.array([x, y, np.ones(len(x))]).T
    a, b, c = np.linalg.lstsq(A, z, rcond=None)[0]

    # Calculate the normal vector
    n = np.array([a, b, -1])
    n = n / np.linalg.norm(n)

    # Calculate the distance from the origin
    d = np.dot(n, np.array([0, 0, 0]))

    return X, Y, Z, a, b, c, d


def plot_fitted_plane(X, Y, Z, a, b, c, d):
    """
    Plots the given points and the fitted plane.
    :param X: x coordinates of the points
    :param Y: y coordinates of the points
    :param Z: z coordinates of the points
    :param a: a parameter of the plane equation
    :param b: b parameter of the plane equation
    :param c: c parameter of the plane equation
    :param d: d parameter of the plane equation
    :return: None
    """

    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Plot the points
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z)

    # Plot the plane
    xx, yy = np.meshgrid(range(320), range(240))
    zz = (a * xx + b * yy + d) / c
    ax.plot_surface(xx, yy, zz, alpha=0.2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()


def depth2itof(depths: list[np.ndarray], freqs: list[int], ampls: list[float]) -> np.ndarray:
    """
    Function used to convert the depth map to the correspondent itof depth map\n
    Param:
        - depth (list[np.ndarray]): The depth map
        - freqs (list[int]): The frequencies of the data (Hz)
        - ampl (list[float]): The amplitude of the data\n
    Return:
        - itof (np.ndarray): The itof data
    """

    # Compute the conversion value
    conv_values = [(4 * const.pi * freq) / const.c for freq in freqs]
    # Computhe the shift value
    phis = [depth * conv_value for depth, conv_value in zip(depths, conv_values)]

    # Compute the iToF data
    itof = np.empty((len(freqs) * 2, depths[0].shape[0], depths[0].shape[1]), dtype=np.float32)
    for i, ampl, phi in zip(range(len(ampls)), ampls, phis):
        itof[i] = ampl * np.cos(phi)               # Real part
        itof[i + len(freqs)] = ampl * np.sin(phi)  # Imaginary part

    return itof


def load(path: Path) -> np.ndarray:
    """
    Load the data from the h5 file.\n
    Param:
        - path (Path): The path of the h5 file\n
    Return:
        - data (dict): The data loaded from the h5 file
    """

    data = None

    with h5py.File(path, "r") as f:
        for key in f.keys():
            elm = f[key]
            data = np.array(elm)
    
    return data  # type: ignore


if __name__ == "__main__":
    # Load the real empty scene depth at 20MHz
    # Define the input path
    mat_files_path = Path(
        "/media/matteocali/shared_ssd/NLoS imaging using iToF/real_dts/20230622_real_NLoS_scenes/"
    )
    mat_files = sorted(list(mat_files_path.glob("*.mat")))

    for mat_file in mat_files:  # Fill the dictionaries
        if mat_file.stem.startswith("scene_wall"):
            freq, type = mat_file.stem.split("_")[3:]
            freq = int(freq[3:] + "000000")
            if freq == ACCEPTED_FREQS and type == ACCEPTED_DTYPES:
                empty_scenes_path = mat_file

    real_depth = load(empty_scenes_path)  # type: ignore
    real_amplitude = load(empty_scenes_path.parent / "scene_wall_only_MHz20_amplitude.mat")  # type: ignore
    # Add the 239th col as a copy of the 238th col
    real_depth = np.hstack((real_depth, np.expand_dims(real_depth[:, 238], axis=1)))
    real_amplitude = np.hstack((real_amplitude, np.expand_dims(real_amplitude[:, 238], axis=1)))

    # Fit a plane to the points
    _, _, _, a, b, c, d = plane_fitting(real_depth)

    # Compute the depth of the extracted plane
    xx, yy = np.meshgrid(range(320), range(240))
    zz = (a * xx + b * yy + d) / c

    # Compute the absolute difference between the extracted plane and the real depth
    abs_diff = abs(zz - real_depth.T).mean()
    print(f"Absolute mean difference real depth: {round(abs_diff, 4)}")


    # Load the synthetic empty scene depth at 20MHz
    synth_empty_scene = torch.load("neural_network_v2_code/datasets/empty_scene/processed_data/processed_test_dts.pt")
    
    # Extarct the itof data at 20MHz
    itof_data = synth_empty_scene[0]['itof_data']
    itof_data[0, ...] = itof_data[1, ...]
    itof_data[1, ...] = itof_data[4, ...]
    itof_data = itof_data[:2, ...]

    # Define the gaussian noise std
    estimated_std = abs_diff / 2.25
    guessed_std = 0.03

    # Apply the gaussian noise
    noisy_itof = CustomTransforms.AddGaussianNoise(0, estimated_std)(itof_data)
    noisy_itof_guess = CustomTransforms.AddGaussianNoise(0, guessed_std)(itof_data)

    # Compute the depth of the extracted plane and of the original one
    depth = utils.itof2depth(itof_data, 20e6)
    noisy_depth = utils.itof2depth(noisy_itof, 20e6)

    # Compute the absolute difference between the noisy and non noisy planes
    diff = abs(noisy_depth - depth).mean().item()
    print(f"Absolute mean difference synthetic depth: {round(diff, 4)}")

    print(f"STD used for the gaussian noise: {round(estimated_std, 6)}")

    # Compute the real_itof_data
    real_itof_data = depth2itof([real_depth], [ACCEPTED_FREQS], [real_amplitude])  # type: ignore
    real_itof_data = np.flip(np.flip(real_itof_data, axis=0), axis=1)

    ampl_real_20 = np.sqrt(real_itof_data[0, ...]**2 + real_itof_data[1, ...]**2)

    # Plot the noisy and non noisy itof_data
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    img0 = ax[0, 0].imshow(itof_data[0, ...].numpy().T, cmap='jet')
    ax[0, 0].set_title("Original itof data")
    divider = make_axes_locatable(ax[0, 0])                  # Defien the colorbar axis
    cax = divider.append_axes("right", size="5%", pad=0.05)  # Set the colorbar location
    fig.colorbar(img0, cax=cax)                              # Plot the colorbar
    img1 = ax[0, 1].imshow(noisy_itof[0, ...].numpy().T, cmap='jet')
    ax[0, 1].set_title(f"Noisy itof data (STD used = {round(estimated_std, 6)})")
    divider = make_axes_locatable(ax[0, 1])                  # Defien the colorbar axis
    cax = divider.append_axes("right", size="5%", pad=0.05)  # Set the colorbar location
    fig.colorbar(img1, cax=cax)                              # Plot the colorbar
    img2 = ax[1, 0].imshow(noisy_itof_guess[0, ...].numpy().T, cmap='jet')
    ax[1, 0].set_title(f"Noisy itof data (STD used = {round(guessed_std, 6)})")
    divider = make_axes_locatable(ax[1, 0])                  # Defien the colorbar axis
    cax = divider.append_axes("right", size="5%", pad=0.05)  # Set the colorbar location
    fig.colorbar(img2, cax=cax)                              # Plot the colorbar
    img3 = ax[1, 1].imshow((real_itof_data[0, ...]/ampl_real_20).T, cmap='jet')
    img3.set_clim(0.5, 1)
    ax[1, 1].set_title("Real itof data")
    divider = make_axes_locatable(ax[1, 1])                     # Defien the colorbar axis
    cax = divider.append_axes("right", size="5%", pad=0.05)  # Set the colorbar location
    fig.colorbar(img3, cax=cax)                              # Plot the colorbar
    plt.tight_layout()
    plt.savefig("neural_network_v2_code/extras/noise evaluation.png")
    plt.show()
