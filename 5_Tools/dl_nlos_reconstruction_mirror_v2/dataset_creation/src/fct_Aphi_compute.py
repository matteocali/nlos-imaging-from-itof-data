import numpy as np


def Aphi_compute(v_in):

    n_fr = int(v_in.shape[-1]/2)
    p = v_in.shape[1]
    i_mid = int((p - 1) / 2)

    # Compute useful additional fields
    a_in = np.sqrt(v_in[:, i_mid, i_mid, :n_fr] ** 2 + v_in[:, i_mid, i_mid, n_fr:] ** 2)
    phi_in = np.arctan2(v_in[:, i_mid, i_mid, :n_fr], v_in[:, i_mid, i_mid, n_fr:])

    return a_in, phi_in




