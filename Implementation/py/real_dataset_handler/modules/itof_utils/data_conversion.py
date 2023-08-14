import torch
import numpy as np
import scipy.constants as const


def itof2depth(itof: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Function used to convert the itof depth map to the correspondent radial depth map
        param:
            - itof: itof depth map
            - freqs: frequencies of the itof sensor (Hz)
        return:
            - radial depth map
    """

    n_freqs = freqs.shape[0]  # Number of frequencies used by the iToF sensor

    if n_freqs != itof.shape[0] // 2:
        raise ValueError("The number of frequencies is not equal to the number of channels in the itof map")

    # Compute the phase shift value (for each frequency)
    phi = np.arctan2(itof[n_freqs:, ...], itof[:n_freqs, ...])

    # Compute the conversion value (for each frequency)
    conv_value =  const.c / (4 * const.pi * freqs)

    # Compute the radialdepth map
    depths = np.empty(phi.shape, dtype=np.float32)
    for i in range(conv_value.shape[0]):
        depths[i, ...] = phi[i, ...] * conv_value[i]

    # Set nan values to 0
    depths = np.nan_to_num(depths, nan=0, posinf=1e10, neginf=-1e10)
    
    return depths


def depth2itof(depth: np.ndarray, freqs: np.ndarray, ampl: np.ndarray) -> np.ndarray:
    """
    Function used to convert the depth map to the correspondent itof depth map
        param:
            - depth: radial depth map
            - freqs: frequency of the itof sensor (Hz)
            - ampl: amplitude of the data
        return:
            - itof data at the given frequence (Hz) as the real and immaginary part of the correspondent phasor
    """

    # Compute the conversion value
    conv_value = (4 * const.pi * freqs) / const.c
    # Computhe the shift value
    phi = np.empty(depth.shape, dtype=np.float32)
    
    for i in range(conv_value.shape[0]):
        phi[i, ...] = depth[i, ...] * conv_value[i]

    # Compute the real and imaginary part of the phasor
    real_phi = ampl * np.cos(phi)
    im_phi = ampl * np.sin(phi)

    # Compute the iToF data
    itof = np.empty((real_phi.shape[0] * 2, real_phi.shape[1], real_phi.shape[2]), dtype=np.float32)
    itof[:real_phi.shape[0], ...] = real_phi
    itof[real_phi.shape[0]:, ...] = im_phi

    return itof  # type: ignore
