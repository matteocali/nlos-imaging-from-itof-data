import numpy as np
import scipy.constants as const


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
