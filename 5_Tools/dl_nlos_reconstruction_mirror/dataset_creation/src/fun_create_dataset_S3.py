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

def create_dataset(data_path,freqs,out_name):
    n_fr = freqs.shape[0]  # number of frequencies
    dataset_name = 3
    dict_out = {}
    with h5py.File(data_path,"r") as f:
        dict_out["name"] = dataset_name
        dict_out["raw_itof"] = f["v_in"][:]
        dict_out["amplitude_raw"] = f["ampl_f"][:]
        depth_gt = f["depth_gt"][:]
        depth_mpi = f["depth_f"][:]
        phi_mpi = np.zeros(f["ampl_f"].shape,dtype=np.float32)
        phid = np.zeros(f["ampl_f"].shape,dtype=np.float32)
        for i in range(n_fr):
            phid[...,i] = compute_phi(depth_gt,freqs[i])
            phi_mpi[...,i] = compute_phi(depth_mpi[...,i],freqs[i])
        dict_out["phase_raw"] = phi_mpi
        
        
        dict_out["depth_ground_truth"] = f["depth_gt"][:]
        dict_out["phase_direct"] = phid
    out_name += "full_imgs.h5"
    with h5py.File(out_name,"w") as f:
        for key in dict_out.keys():
            f.create_dataset(name=key,data=dict_out[key])
    sys.exit()
        
