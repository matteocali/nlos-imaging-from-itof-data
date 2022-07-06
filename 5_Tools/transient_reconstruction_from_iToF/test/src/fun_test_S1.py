import numpy as np
import os
import tensorflow as tf
import h5py
import math
import matplotlib
import scipy
import glob
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("../training/src/")
sys.path.append("../utils/") # Adds higher directory to python modules path
import DataLoader
import utils
import depth_estimation
font = {'size'   : 6}
import PredictiveModel_hidden as PredictiveModel
matplotlib.rc('font', **font)
import scipy.io as sio
from fun_scale_vin import scale_vin


def compute_v(A,phi):     #Given the amplitude and phase information we can compute the input iToF measurements
    n_fr = A.shape[-1]
    v_in = np.zeros((A.shape[0],A.shape[1],2*n_fr),dtype=np.float32)
    v_in[...,:n_fr] = A*np.cos(phi)
    v_in[...,n_fr:] = A*np.sin(phi)
    return v_in

# Compute the phase displacement from the depth 
def compute_phi(depth,freq):
    phi = depth*4*np.pi*freq/utils.c()
    phi = phi%(2*np.pi)
    return phi


def testS1(data_path,weights_path,attempt_name,Sname,P,freqs,fl_norm_perpixel=False,fl_plot=False,fil_direct=8,fil_denoise=32):
    ff = freqs.shape[0] # Number of frequencies
    fl_3freq = (ff==3)  # Whether we are using 3 frequencies (or 2)
    ampl_names = glob.glob(data_path+"*amplitude.mat")
    depth_names = glob.glob(data_path+"*depth.mat")
    gt_names = glob.glob(data_path+"ground_truth/*depth.mat")
    ampl_names.sort()
    depth_names.sort()
    gt_names.sort()
    n_img = len(gt_names)   #number of images

    temp = sio.loadmat(ampl_names[0])
    dim_x = temp["amplitude"].shape[0]
    dim_y = temp["amplitude"].shape[1]

    v_ins = np.zeros((n_img,dim_x,dim_y,2*ff),dtype=np.float32)
    depth_gts = np.zeros((n_img,dim_x,dim_y),dtype=np.float32)

    # Input data loading and preprocessing
    d20s = np.zeros((n_img,dim_x,dim_y),dtype=np.float32)
    d50s = np.zeros((n_img,dim_x,dim_y),dtype=np.float32)
    d60s = np.zeros((n_img,dim_x,dim_y),dtype=np.float32)
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
        A = np.zeros((dim_x,dim_y,ff),dtype=np.float32)
        A[...,0] = a20["amplitude"]
        A[...,1] = a50["amplitude"]
        if fl_3freq:
            A[...,2] = a60["amplitude"]
        d20 = d20["depth_PU"]
        d50 = d50["depth_PU"]
        if fl_3freq:
            d60 = d60["depth_PU"]
        d20s[i] = d20
        d50s[i] = d50
        if fl_3freq:
            d60s[i] = d60
        phi = np.zeros((dim_x,dim_y,ff),dtype=np.float32)
        phi[...,0] = compute_phi(d20,20e06)
        phi[...,1] = compute_phi(d50,50e06)
        if fl_3freq:
            phi[...,2] = compute_phi(d60,60e06)
        vin = compute_v(A,phi)
        v_ins[i] = vin
        depth_gts[i] = gt["depth_GT_radial"]
        depths_mpi = np.stack((d20s,d50s,d60s),axis=-1)
    
    # Fix values if they are out of range
    depths_mpi = np.where(depths_mpi>7.5,7.5,depths_mpi)
    # index following the central element of the patch
    s_pad=int((P-1)/2)

    # Indexes corresponding to frequencies
    fns = np.arange(ff)
    num_tested = 0

    # Variables to keep track of the errors images per image
    avg1f = np.zeros(ff)
    avgpr = np.zeros(ff)
    avgprf = np.zeros(ff)
    avgmin=0
    avgmean=0
    avgmax=0

    for k in range(n_img):
        img_name = os.path.basename(gt_names[k][:-4])
        print(img_name)
        
        depth_gt = depth_gts[k]
        v_in = v_ins[k]

        fmod_str = list(map(str,(freqs/10e05).astype(np.int32)))
        for i in range(ff):
            fmod_str[i] += "MHz"
        
        errs_pr = np.zeros(ff)
        errs_prf = np.zeros(ff)
        errs_1f = np.zeros(ff)
        ampl = np.sqrt(v_in[...,:ff]**2+v_in[...,ff:]**2)
        v_in = v_in[np.newaxis,:,:,:] 

        # Normalize the input either per pixel or with a windowed average
        if fl_norm_perpixel:
            max_phasor_ampl = ampl[...,0]
            max_phasor_ampl = max_phasor_ampl[np.newaxis,:,:,np.newaxis]
            v_in/=max_phasor_ampl
        else: 
            v_in = scale_vin(v_in,P)

        net = PredictiveModel.PredictiveModel(name='DeepBVE_nf_std',
                                                   dim_b = 1,
                                                   P = P, 
                                                   freqs = freqs,
                                                   saves_path='./saves',
                                                   fil_size=fil_direct,
                                                   fil_denoise_size=fil_denoise)
        # Load weights
        if P != 3:
            net.SpatialNet.load_weights(weights_path[0])
        net.DirectCNN.load_weights(weights_path[1])


        fl_denoise = not(net.P==net.out_win)   #If the two values are different, then the denoising network has been used
        # Make prediction
        v_input = tf.convert_to_tensor(np.pad(v_in,pad_width=[[0,0],[s_pad,s_pad],[s_pad,s_pad],[0,0]],mode="edge"),dtype="float32")
        if fl_denoise:
            v_in_v = net.SpatialNet(v_input)
            off = int((v_in_v.shape[1] - dim_x)/2)
        else: 
            v_in_v = v_input
        
        [v_out_g, v_out_d,  phid] = net.DirectCNN(v_in_v)

        depth_prs = phid*utils.c()/(4*utils.pi()*freqs)
        depth_prs = np.squeeze(depth_prs)

        for fn in fns:
            if fn == 0:
               #depth20 = depth_gt
               depth20 = depth_prs[...,0]
               #depth20 = np.minimum(depths_mpi[k,...,0],depth_prs[...,0])
               #depth20 = np.minimum(depths_mpi[k,...,2],depth20)
               #depth20 = depths_mpi[k,...,0]
               #depth20 = np.minimum(depth_prs[...,0],depth20)

            fmod=freqs[fn]
            v_out_fmod = np.stack([v_out_d[:,:,:,fn], v_out_d[:,:,:,fn+ff]], axis=-1)
            depth_pr = depth_prs[...,fn]
            if  fn == 1 or fn ==2:
                amb_range = utils.amb_range(fmod)
                n_amb = np.round((depth20-depth_pr)/amb_range)
                depth_pr = depth_pr+ n_amb*amb_range
                depth_pr = np.where(depth_pr<0,depth_pr+amb_range,depth_pr)
            # Error computation for the predicted depth
            error_pr = np.squeeze(depth_pr - depth_gt)
            mae_pr = np.mean(np.abs(error_pr))
            avg_mae_pr = np.mean(mae_pr)
        
            depth_prf = np.ndarray(depth_pr.shape, dtype=np.float32)
            depth_prf = cv2.bilateralFilter(depth_pr, -1, sigmaColor=0.8, sigmaSpace=3)  #sigmaColor was 0.05
        
            depth_pr = np.squeeze(depth_pr)
            print("MAE ", np.mean(np.abs(depth_gt-depth_pr))*100, " cm")
            print("MAE bil ", np.mean(np.abs(depth_gt-depth_prf))*100, " cm")

            # Error computation for the predicted depth
            error_prf = np.squeeze(depth_prf - depth_gt)


            mae_prf = np.mean(np.abs(error_prf)) 
            mae_prf = np.mean(np.abs(error_prf)) 
            avg_mae_prf = np.mean(mae_prf)
        
            # Get phasor vector for fmod=20MHz
            v_fmod = np.stack([v_in[:,:,:,fn], v_in[:,:,:,fn+ff]], axis=3)

            # Compute ambiguity range for fmod
            amb_range = utils.c() / (2*fmod)
        
            # Depth computation using naive approach (using only fmod=60MHz)
            depth_1f = depths_mpi[k,...,fn]
         
            # Error computation for the predicted depth
            error_1f = np.squeeze(depth_1f - depth_gt)
            mae_1f = np.mean(np.abs(error_1f)) 
            avg_mae_1f = np.mean(mae_1f)
            avg_mae_pr = np.round(avg_mae_pr*100,3)
            avg_mae_prf = np.round(avg_mae_prf*100,3)
            avg_mae_1f = np.round(avg_mae_1f*100,3)          
            errs_pr[fn] = avg_mae_pr
            errs_prf[fn] = avg_mae_prf
            errs_1f[fn] = avg_mae_1f
            M = 0.2


            err_pr = np.squeeze(error_pr)
            err_prf = np.squeeze(error_prf)
            err_1f = np.squeeze(error_1f)

            # Cropping as done by agresti
            err_pr = err_pr[:,32:-32]
            err_prf = err_prf[:,32:-32]
            err_1f = err_1f[:,32:-32]

            img_pr =  np.mean(np.abs(err_pr))
            img_prf =  np.mean(np.abs(err_prf))
            img_1f =  np.mean(np.abs(err_1f))
            img_pr = np.round(img_pr*100,2)
            img_prf = np.round(img_prf*100,2)
            img_1f = np.round(img_1f*100,2)
            avg1f[fn]+=img_1f
            avgpr[fn]+=img_pr
            avgprf[fn]+=img_prf
            
            # Error computation for min, mean and max metrics between the predictionsat different frequencies
            if fn == 0:
                depth_min = np.copy(depth_prf)
                depth_mean = np.copy(depth_prf)
                depth_max = np.copy(depth_prf)
            if fn==1 or fn==2:
                depth_min = np.where(depth_prf<depth_min,depth_prf,depth_min)
                depth_max = np.where(depth_prf>depth_max,depth_prf,depth_max)
                depth_mean += depth_prf
            #In case we only have 2 frequencies
            if fn==1 and ff==2:
                fn=2
            if fn == 2:
                depth_mean/=ff
                err_min = np.squeeze(depth_min-depth_gt)
                err_minv = np.mean(np.abs(err_min))
                err_mean = np.squeeze(depth_mean-depth_gt)
                err_meanv = np.mean(np.abs(err_mean))
                err_max = np.squeeze(depth_max-depth_gt)
                err_maxv = np.mean(np.abs(err_max))
                avgmin += err_minv
                avgmean += err_meanv
                avgmax += err_maxv
                print("MIN ", err_minv*100, "cm")
                print("MEAN ", err_meanv*100, "cm")
                print("MAX ", err_maxv*100, "cm")

            # Plot results
            if fl_plot:
                plt.subplot(1,3,1)
                plt.imshow(err_1f, cmap="jet")
                plt.title("Direct MAE: " +  str(img_1f) + " [cm]", fontsize=6)
                cbar = plt.colorbar(fraction=0.046, pad=0.04)
                plt.clim(-M,M)
                cbar.set_label('Depth [m]', labelpad=10, fontsize=6)
                plt.subplot(1,3,2)
                plt.imshow(err_pr, cmap="jet")
                plt.title("Pred MAE: " +  str(img_pr) + " [cm]", fontsize=6)
                cbar = plt.colorbar(fraction=0.046, pad=0.04)
                plt.clim(-M,M)
                cbar.set_label('Depth [m]', labelpad=10, fontsize=6)
                plt.subplot(1,3,3)
                plt.imshow(err_prf, cmap="jet")
                plt.title("Bilat MAE: " +  str(img_prf) + " [cm]", fontsize=6)
                cbar = plt.colorbar(fraction=0.046, pad=0.04)
                plt.clim(-M,M)
                cbar.set_label('Depth [m]', labelpad=10, fontsize=6)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig("./out/"+attempt_name+"/" + Sname + "/" +img_name+"_"+ fmod_str[fn]+"_"+str(k)+".png" ,dpi=1200)
                plt.close()
        
         
        
            error_pr = error_pr[depth_gt>0]
            error_prf = error_prf[depth_gt>0]
            error_1f = error_1f[depth_gt>0]
            pr_me = str(np.round(100*np.mean(error_pr),2))
            prf_me = str(np.round(100*np.mean(error_prf),2))
            of_me = str(np.round(100*np.mean(error_1f),2))
        

        num_tested += 1

    avg1f/=n_img
    avgpr/=n_img
    avgprf/=n_img
    avgmin/=n_img
    avgmean/=n_img
    avgmax/=n_img
    # Show results
    print("Dataset " + Sname +  " errors at 20 MHz:" )
    print("1freq: " + str(avg1f[0]) + " cm")
    print("predicted: " + str(avgpr[0]) + " cm")
    print("predicted + bilat: " + str(avgprf[0]) + " cm")
    print("Dataset " + Sname +  " errors at 50 MHz:" )
    print("1freq: " + str(avg1f[1]) + " cm ")
    print("predicted: " + str(avgpr[1]) + " cm")
    print("predicted + bilat: " + str(avgprf[1]) + " cm")
    if fl_3freq:
        print("Dataset " + Sname +  " errors at 60 MHz:" )
        print("1freq: " + str(avg1f[2]) + " cm")
        print("predicted: " + str(avgpr[2]) + " cm")
        print("predicted + bilat: " + str(avgprf[2]) + " cm")

    print("MIN depth: " + str(avgmin*100) + " cm")
    print("MEAN depth: " + str(avgmean*100) + " cm")
    print("MAX depth: " + str(avgmax*100) + " cm")





