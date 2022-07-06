import numpy as np
import sys
import matplotlib.pyplot as plt



"""
Function that splits input transient information into its direct and global parts. 
Returns two arrays corresponding to the direct and global parts of the transient. The sum of the two gives the input
"""



def split_transient_direct_global(tr):


    dim_t = tr.shape[-1]

    # Discrete derivative
    der = tr[...,1:] - tr[...,:-1]

    # Position of the minimum of the derivative
    split_ind = np.argmin(der,axis=-1)
    split_ind = split_ind[...,np.newaxis]

    # Keep going to the right until we find a discete derivative higher or equal to 0
    while True:
        add_ind = np.take_along_axis(der,split_ind+1,axis=-1)
        add_ind = np.where(add_ind<0,1,0)
        split_ind += add_ind
        if np.all(add_ind==0):
            break

    # shift the position of one element to the left
    split_ind += 1

            
    xd = np.zeros(tr.shape)
    xg = np.zeros(tr.shape)

    

    for ix in range(tr.shape[0]):
        for iy in range(tr.shape[1]):
            ind = int(split_ind[ix,iy])
            xd[ix,iy,:ind] = tr[ix,iy,:ind]
            xg[ix,iy,ind:] = tr[ix,iy,ind:]

    # Get the direct to form a single peak
    peak_val = np.sum(xd,axis=-1,keepdims=True)
    weights = np.where(peak_val==0,0,xd/peak_val)

    indexes = np.arange(dim_t)
    ind_val = (weights @ indexes).astype(np.int32)
    
    # Place the max value in the correct position
    new_xd = np.zeros(tr.shape)
    new_xd[np.arange(tr.shape[0])[:,None,None],np.arange(tr.shape[1])[None,:,None],ind_val[...,None]] = peak_val


    return new_xd, xg






