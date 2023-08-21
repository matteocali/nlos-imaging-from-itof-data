import numpy as np
from tqdm import trange
from .. import utilities as ut
from math import pi
from .tools import extract_peak, rmv_first_reflection_transient, rmv_glb


def phi(freqs: np.ndarray, exp_time: float = 0.01, dim_t: int = 2000) -> np.ndarray:
    """
    Function to convert dToF output (transient) into iToF measurements (phi values for the different frequencies used)
    :param freqs: target frequency values to be used
    :param exp_time: exposure time (time bin size * c)
    :param dim_t: total number of temporal bins
    :return: matrix phy containing all the phi measurements
    """

    c = 3e8
    # min_t = 0.1 / c
    min_t = 0
    max_t = 2 * exp_time / c * dim_t
    step_t = (max_t - min_t) / dim_t
    times = np.arange(dim_t) * step_t
    phi_arg = 2 * pi * np.matmul(freqs.reshape(-1, 1), times.reshape(1, -1))
    return np.concatenate([np.cos(phi_arg), np.sin(phi_arg)], axis=0)


def amp_phi_compute(v_in):
    n_fr = int(v_in.shape[-1] / 2)
    # Compute useful additional fields
    amp_in = np.sqrt(v_in[:, :, :n_fr] ** 2 + v_in[:, :, n_fr:] ** 2)
    phi_in = np.arctan2(v_in[:, :, :n_fr], v_in[:, :, n_fr:])
    return amp_in, phi_in


def active_beans_percentage(transient: np.ndarray) -> float:
    """
    Function that compute the percentage of active beans given a transient vector (single channel)
    :param transient: single channel transient vector
    :return: the percentage value
    """

    non_zero_beans = np.where(transient != 0)[0]  # Find all the non-zero bins
    return (
        len(non_zero_beans) / len(transient) * 100
    )  # Divide the number of active bins (len(non_zero_beans)) by the total number of bins (len(transient)) and multiply by 100 (in order to obtain a percentage)


def direct_global_ratio(transient: np.ndarray) -> list:
    """
    Function to compute the global direct ratio
    :param transient: single transient vector
    :return: a list containing the ratio between the global and the direct component for each channel
    """

    mono = len(transient.shape) == 1  # Check if the images are Mono or RGBA

    _, p_value = extract_peak(transient)  # Compute the direct component location
    glb = rmv_first_reflection_transient(
        transient, verbose=False
    )  # Extract the global component from the transient data
    glb_sum = np.sum(glb, axis=0)  # Sum oll the global component

    if not mono:
        ratio = (
            []
        )  # Define an empty list that will contain the ratio value for each channel
        for c_index in range(glb_sum.shape[0]):  # For each channel
            ratio.append(
                glb_sum[c_index] / p_value[c_index]
            )  # Append to the list the ratio value
    else:
        ratio = [glb_sum / p_value]  # Compute the ratio value

    return ratio


def clear_tr_ratio(transient: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Function to remove all the imprecise transient based on an Otsu thresholding
    :param transient: transient data
    :param method: method used to compute the threshold
    :return: transient data with spurious one set to 0 (only the global is set to zero)
    """

    tr = np.copy(
        transient
    )  # Copy the transient information in order to not override the given one

    ratio = np.zeros(
        tr.shape[0]
    )  # Create the np.ndarray that will contain the ratio data as an array full of np.zeros
    for i in range(tr.shape[0]):
        ratio[i] = direct_global_ratio(tr[i, :, 1])[
            0
        ]  # Populate the ratio np.ndarray computing the ratio value for each transient

    h_data = np.histogram(
        ratio, 100
    )  # Compute the histogram values of the provided data using 100 bins

    if method == "otsu":
        t_value, threshold = ut.otsu_hist_threshold(
            h_data
        )  # Compute the threshold using the Otsu's method
    elif method == "balanced":
        t_value, threshold = ut.balanced_hist_thresholding(
            h_data
        )  # Compute the threshold using the balanced method

    # noinspection PyUnboundLocalVariable
    indexes = np.where(ratio < t_value)[
        0
    ]  # Find all the transient that does not satisfy the requirements of the threshold

    for index in indexes:  # For each index of interest
        glb = rmv_first_reflection_transient(
            tr[index, :, 1], verbose=False
        )  # Extract the global component
        try:
            start = np.where(glb != 0)[0][
                0
            ]  # Define the starting location of the global component
            tr[index, start:, :] = np.zeros(
                [tr[index, start:, 1].shape[0], tr.shape[2]]
            )  # Set the global component to 0
        except IndexError:
            pass  # If the global component is empty, do nothing
    return tr


def clear_tr_energy(transient: np.ndarray, threshold: int) -> np.ndarray:
    """
    Function to remove all the imprecise transient based on an energy thresholding
    :param transient: Transient data af the image (as a long list)
    :param threshold: Threshold value (percentage of the maximum energy)
    :return: cleaned transient data
    """

    tr = np.copy(
        transient
    )  # Copy the transient information in order to not override the given one
    glb = np.zeros(
        transient.shape
    )  # Create the np.ndarray that will contain the global component as an array full of np.zeros
    glb_sum = np.zeros(
        transient.shape[0]
    )  # Create the np.ndarray that will contain the sum of the global component as an array full of np.zeros
    max_glb_value = 0  # Create the variable that will contain the maximum value of the global component

    for i in trange(transient.shape[0], desc="cleaning transient"):
        glb[i, :, :] = rmv_first_reflection_transient(
            transient[i, :, :], verbose=False
        )  # Extract the global component
        glb_sum[i] = np.mean(
            np.sum(glb[i, :, :], axis=1)
        )  # Compute the sum of the global component
        max_glb_value = max(
            max_glb_value, np.max(glb_sum[i])
        )  # Find the maximum value of the global component

    for i in range(transient.shape[0]):
        if glb_sum[i] < max_glb_value * (
            threshold / 100
        ):  # If the sum of the global component is less than 10% of the maximum value
            tr[i, :, :] = rmv_glb(tr[i, :, :])  # Set the global component to 0

    return tr
