import numpy as np
import scipy.constants as const
import torch


def hfov2focal(hdim: int, hfov: float) -> float:
    """
    Function used to convert the horizontal field of view to the focal length
        param:
            - hdim: horizontal dimension of the image (pixels)
            - hfov: horizontal field of view (degrees)
        return:
            - focal length
    """

    return 0.5 * hdim / np.tan(0.5 * hfov * np.pi / 180)


def depth_cartesian2radial(
    depth: torch.Tensor or np.ndarray, focal: float
) -> torch.Tensor or np.ndarray:
    """
    Function used to convert the depth map from cartesian to radial coordinates
        param:
            - depth: depth map in cartesian coordinates
            - focal: focal length of the camera
        return:
            - depth map in radial coordinates
    """

    if isinstance(depth, np.ndarray):
        env = np
    else:
        env = torch

    res_v = depth.shape[0]
    res_h = depth.shape[1]

    axis_v = env.linspace(-res_v / 2 + 1 / 2, res_v / 2 - 1 / 2, res_v)
    axis_h = env.linspace(-res_h / 2 + 1 / 2, res_h / 2 - 1 / 2, res_h)

    conversion_matrix = env.zeros((res_v, res_h))
    for i in range(res_v):
        for j in range(res_h):
            conversion_matrix[i, j] = 1 / env.sqrt(1 + (axis_v[i] / focal) ** 2 + (axis_h[j] / focal) ** 2)  # type: ignore

    return depth / conversion_matrix


def depth_radial2cartesian(
    depth: torch.Tensor or np.ndarray, focal: float
) -> torch.Tensor or np.ndarray:
    """
    Function used to convert the depth map from radial to cartesian coordinates
        param:
            - depth: depth map in radial coordinates
            - focal: focal length of the camera
        return:
            - depth map in cartesian coordinates
    """

    if isinstance(depth, np.ndarray):
        env = np
    else:
        env = torch

    res_v = depth.shape[0]
    res_h = depth.shape[1]
    axis_v = env.linspace(-res_v / 2 + 1 / 2, res_v / 2 - 1 / 2, res_v)
    axis_h = env.linspace(-res_h / 2 + 1 / 2, res_h / 2 - 1 / 2, res_h)

    conversion_matrix = env.zeros((res_v, res_h))
    for i in range(res_v):
        for j in range(res_h):
            conversion_matrix[i, j] = env.sqrt(1 + (axis_v[i] / focal) ** 2 + (axis_h[j] / focal) ** 2)  # type: ignore

    return depth * conversion_matrix


def itof2depth(
    itof: torch.Tensor | np.ndarray, freqs: tuple | float | int
) -> torch.Tensor | np.ndarray:
    """
    Function used to convert the itof depth map to the correspondent radial depth map
        param:
            - itof: itof depth map
            - freqs: frequencies of the itof sensor (Hz)
        return:
            - radial depth map
    """

    # Select the correct data type
    if isinstance(itof, np.ndarray):
        env = np
        arr = np.array
    else:
        env = torch
        arr = torch.Tensor

    # Perform a check on freqs tu ensure that it is a tuple
    freqs = (
        tuple([freqs])
        if (isinstance(freqs, float) or isinstance(freqs, int))
        else freqs
    )

    # Check if there is the batch dimension
    if len(itof.shape) == 3 and isinstance(itof, torch.Tensor):
        itof = itof.unsqueeze(0)
    elif len(itof.shape) == 3 and isinstance(itof, np.ndarray):
        itof = itof[np.newaxis, ...]

    n_freqs = (
        1 if isinstance(freqs, float) or isinstance(freqs, int) else len(freqs)
    )  # Number of frequencies used by the iToF sensor

    if n_freqs != itof.shape[1] // 2:
        raise ValueError(
            "The number of frequencies is not equal to the number of channels in the itof map"
        )

    # Compute the phase shift value (for each frequency)
    phi = env.arctan2(itof[:, n_freqs:, ...], itof[:, :n_freqs, ...]).squeeze(0)  # type: ignore

    # Compute the conversion value (for each frequency)
    conv_value = const.c / (4 * const.pi * arr(freqs))
    # If necessary change the device of the conversion value
    if isinstance(itof, torch.Tensor):
        conv_value = conv_value.to(itof.device)  # type: ignore

    # Compute the radialdepth map
    depth = phi * conv_value

    # Set nan values to 0
    depth = env.nan_to_num(depth, nan=0, posinf=1e10, neginf=-1e10)  # type: ignore

    # Remove unnecessary dimensions
    if isinstance(depth, torch.Tensor) and len(depth.shape) == 4:
        depth = depth.squeeze(1)

    return depth  # type: ignore


def depth2itof(
    depth: torch.Tensor or np.ndarray, freq: float, ampl: float = 1.0
) -> torch.Tensor or np.ndarray:
    """
    Function used to convert the depth map to the correspondent itof depth map
        param:
            - depth: radial depth map
            - freq: frequency of the itof sensor (Hz)
            - ampl: amplitude of the data
        return:
            - itof data at the given frequence (Hz) as the real and immaginary part of the correspondent phasor
    """

    # Select the correct data type
    if isinstance(depth, np.ndarray):
        env = np
    else:
        env = torch

    # Compute the conversion value
    conv_value = (4 * const.pi * freq) / const.c
    # Computhe the shift value
    phi = depth * conv_value

    # Compute the real and imaginary part of the phasor
    real_phi = ampl * env.cos(phi)
    im_phi = ampl * env.sin(phi)

    # Compute the iToF data
    itof = env.stack((real_phi, im_phi), 0)  # type: ignore

    return itof  # type: ignore



