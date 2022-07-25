import numpy as np
import h5py
import os
import sys
import scipy.io as sio
import matplotlib.pyplot as plt
sys.path.append("../utils/")
import utils

def compute_v(A,phi):     #Given the amplitude and phase information we can compute the input iToF measurements
    n_fr = A.shape[-1]
    v_in = np.zeros((A.shape[0],A.shape[1],A.shape[2],2*n_fr),dtype=np.float32)
    v_in[...,:n_fr] = A*np.cos(phi)
    v_in[...,n_fr:] = A*np.sin(phi)
    return v_in

def compute_phi(depth,freq):
    phi = depth*4*np.pi*freq/utils.c()
    phi = phi%(2*np.pi)
    return phi

def create_dataset(depth_names,ampl_names,gt_names,freqs,out_name,P,n_for_image=250):
    n_fr = freqs.shape[0]  # number of frequencies
    n_img = len(gt_names)
    s = P # Patch size

    n_fr = freqs.shape[0]
    fl_3freq = (n_fr==3)
    depth_names.sort()
    ampl_names.sort()
    gt_names.sort()
    dim_x = 240
    dim_y = 320

    a_stack = np.zeros((n_img*n_for_image,s,s,n_fr),dtype=np.float32)
    d_stack = np.zeros((n_img*n_for_image,s,s,n_fr),dtype=np.float32)
    gt_stack = np.zeros((n_img*n_for_image,s,s),dtype=np.float32)
    phi_stack = np.zeros((n_img*n_for_image,s,s,n_fr),dtype=np.float32)
    phigt_stack = np.zeros((n_img*n_for_image,s,s,n_fr),dtype=np.float32)

    counter = 0
    for i in range(n_img):
        # Load amplitudes, depths with MPI and the ground truth depths
        a20 = sio.loadmat(ampl_names[i*3])
        a50 = sio.loadmat(ampl_names[i*3+1])
        if fl_3freq:
            a60 = sio.loadmat(ampl_names[i*3+2])
        d20 = sio.loadmat(depth_names[i*3])
        d50 = sio.loadmat(depth_names[i*3+1])
        if fl_3freq:
            d60 = sio.loadmat(depth_names[i*3+2])
        gt = sio.loadmat(gt_names[i])
        print(os.path.basename(gt_names[i]))
        a20 = a20["amplitude"]
        a50 = a50["amplitude"]
        if fl_3freq:
            a60 = a60["amplitude"]

        # Standardization at 20 MHz
        #a20/= np.mean(a20)
        #a50/= np.mean(a20)
        #a60/= np.mean(a20)

        d20 = d20["depth_PU"]
        d50 = d50["depth_PU"]
        if fl_3freq:
            d60 = d60["depth_PU"]
        gt = gt["depth_GT_radial"]

        # Check if the patch contains any invalid pixel
        j = 0
        while j < n_for_image:
            ix = np.random.randint(dim_x-s)
            iy = np.random.randint(dim_y-s)
            a20p = a20[ix:ix+s,iy:iy+s]
            a50p = a50[ix:ix+s,iy:iy+s]
            if fl_3freq:
                a60p = a60[ix:ix+s,iy:iy+s]
            d20p = d20[ix:ix+s,iy:iy+s]
            d50p = d50[ix:ix+s,iy:iy+s]
            if fl_3freq:
                d60p = d60[ix:ix+s,iy:iy+s]
            gtp = gt[ix:ix+s,iy:iy+s]
            if np.any(d20p*d50p*a20p*a50p*gtp==0):
                counter += 1
                continue
            a_stack[i*n_for_image+j,:,:,0] = a20p
            a_stack[i*n_for_image+j,:,:,1] = a50p
            if fl_3freq:
                a_stack[i*n_for_image+j,:,:,2] = a60p
            d_stack[i*n_for_image+j,:,:,0] = d20p
            d_stack[i*n_for_image+j,:,:,1] = d50p
            if fl_3freq:
                d_stack[i*n_for_image+j,:,:,2] = d60p
            gt_stack[i*n_for_image+j,:,:] = gtp

            j+=1
    for i in range(n_fr):
        phi_stack[...,i] = compute_phi(d_stack[...,i],freqs[i])
        phigt_stack[...,i] = compute_phi(gt_stack,freqs[i])
    v_stack = compute_v(a_stack,phi_stack) 
    vgt_stack = compute_v(a_stack,phigt_stack) 

    out_name += str(vgt_stack.shape[0]) + "_s" + str(s) + ".h5"

    datasetname=2
    with h5py.File(out_name,"w") as f:
        f.create_dataset(name="name",data=datasetname)
        f.create_dataset(name="depth_ground_truth",data=gt_stack)
        f.create_dataset(name="depth_mpi",data=d_stack)
        f.create_dataset(name="raw_itof",data=v_stack)
        f.create_dataset(name="gt_raw_itof",data=vgt_stack)
        f.create_dataset(name="amplitude_raw",data=a_stack)
        f.create_dataset(name="phase_raw",data=phi_stack)
        f.create_dataset(name="phase_direct",data=phigt_stack)
    

    


        





