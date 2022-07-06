import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy


"""
This function is used to scale vin appropriately.
We convolve the input image with an averaging square kernel of side P.
The original input is going to be scaled per pixel by this average
"""

def scale_vin(v_in,P):
    n_f = int(v_in.shape[-1]/2)   # Number of frequencies
    dim_x = v_in.shape[1] 
    dim_y = v_in.shape[2]

    v_in = np.squeeze(v_in)
    ampl = np.sqrt(v_in[...,0]**2 + v_in[...,n_f]**2)
    w = np.full((P,P),1/P**2)   # Weights for the mean computation
    w_out = scipy.ndimage.filters.convolve(ampl,w)  # Computation of the mean for each patch
    w_out = w_out[...,np.newaxis]
    v_out = v_in/w_out          # Scaling

    v_out = v_out[np.newaxis,...]

    return v_out


