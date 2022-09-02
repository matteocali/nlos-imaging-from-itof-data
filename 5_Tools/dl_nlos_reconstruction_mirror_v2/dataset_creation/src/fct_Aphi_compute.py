import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
import utils



def Aphi_compute(v_in,v_g,v_d):

    n_fr = int(v_in.shape[-1]/2)
    P = v_in.shape[1]
    i_mid = int((P-1)/2)
    # Compute useful additional fields
    A_in = np.sqrt(v_in[:, i_mid, i_mid, :n_fr]**2 + v_in[:, i_mid, i_mid, n_fr:]**2)
    phi_in = np.arctan2(v_in[:, i_mid, i_mid, :n_fr], v_in[:, i_mid, i_mid, n_fr:])
    A_g = np.sqrt(v_g[:, i_mid, i_mid, :n_fr]**2 + v_g[:, i_mid, i_mid, n_fr:]**2)
    phi_g = np.arctan2(v_g[:, i_mid, i_mid, :n_fr], v_g[:, i_mid, i_mid, n_fr:])
    A_d = np.sqrt(v_d[:, i_mid, i_mid, :n_fr]**2 + v_d[:, i_mid, i_mid, n_fr:]**2)
    phi_d = np.arctan2(v_d[:, i_mid, i_mid, :n_fr], v_d[:, i_mid, i_mid, n_fr:])

    return A_in, phi_in, A_g, phi_g, A_d, phi_d




