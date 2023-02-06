import numpy as np
import scipy.constants as const
from pathlib import Path
from matplotlib import pyplot as plt


def format_time(s_time: float, f_time: float):
    """
    Function used to format the time in a human readable format
        param:
            - s_time: start time
            - f_time: finish time
        return:
            - string containing the time in a human readable format
    """

    minutes, seconds = divmod(f_time - s_time, 60)
    hours, minutes = divmod(minutes, 60)
    return "%d:%02d:%02d" % (hours, minutes, seconds)

def phi_func(freqs, dim_t=2000, exp_time=0.01):
    """
    Function used to generate the phi matrix
        param:
            - freqs: frequencies
            - dim_t: number of time steps
            - exp_time: exposure time
        return:
            - phi matrix
    """

    min_t = 0
    max_t = 2 * exp_time / const.c * dim_t
    step_t = (max_t - min_t) / dim_t
    times = np.arange(dim_t) * step_t
    phi_arg = 2 * const.pi * np.matmul(freqs.reshape(-1, 1), times.reshape(1, -1))
    phi = np.concatenate([np.cos(phi_arg), np.sin(phi_arg)], axis=0)
    return phi


def save_np_as_img(data: np.ndarray, path: Path):
    """
    Function used to save each element of the numpy array as an image
        param:
            - data: numpy array
            - path: path where to save the images
    """

    # Create the folder where to save the images
    path.mkdir(parents=True, exist_ok=True)

    # For each element of the numpy array
    for i in range(data.shape[0]):
        # Save the image
        plt.imsave(str(path / f"{i}.png"), data[i, :, :], cmap="jet")