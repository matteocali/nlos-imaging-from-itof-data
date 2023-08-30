import numpy as np
import scipy.constants as const


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
    times = np.arange(dim_t) * step_t  # type: ignore
    phi_arg = 2 * const.pi * np.matmul(freqs.reshape(-1, 1), times.reshape(1, -1))
    phi = np.concatenate([np.cos(phi_arg), np.sin(phi_arg)], axis=0)
    return phi
