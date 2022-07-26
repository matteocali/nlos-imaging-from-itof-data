import os, sys
sys.path.append("../training/src/")
import numpy as np
import tensorflow as tf
import h5py
import math
import matplotlib
import scipy
import glob
import matplotlib.pyplot as plt
import cv2
import DataLoader
import utils
import depth_estimation
font = {'size'   : 6}
import PredictiveModel_hidden as PredictiveModel
#import PredictiveModel_hidden_2 as PredictiveModel
matplotlib.rc('font', **font)
from fun_scale_vin import scale_vin


"""
The function here implemented has been used to test the model performance on the S3, S4 and S5 real datasets.

Inputs:
    -name:              Type=string,  
                        Indicates the location of the test dataset    
    -weights_path:      Type=string
                        Location of the trained weights
    -attempt_name:      Type=string
                        Name of the trained model 
    -Sname:             Type=string
                        Name of the dataset (one between S3, S4 and S5)
    -P:                 Type=int
                        Patch size
    -freqs:             Type=Array of float
                        Frequencies used to train the method
    -fl_scale_perpixel  Type=bool
                        Whether to scale the input by the 20 MHz component per pixel or using a kernel
    -fl_plot            Type=bool
                        Whether to plot and save the results
"""


def test_real(name,weights_path,attempt_name,Sname,P,freqs,fl_scale_perpixel=False,fl_plot=False,num_fil_direct=8,num_fil_denoise=32):


    fl_img_PhDRT = True
    ff = freqs.shape[0]  # of frequencies
    fl_3freq = (ff==3)   # Check if we trained with three frequencies (the alternative is two)
    fl_denoise = not(P==3)      # Check if the Spatial Feature Extractor has been used

    # Load data for testing
    with h5py.File(name,"r") as f:
        depth_gts = f["depth_gt"][:]
        v_ins = f["v_in"][:]
    # If we trained with two frequencies, load just those two
    if not fl_3freq:
        v_ins = v_ins[...,[0,1,3,4]]
    # Initialize parameters
    n_img = depth_gts.shape[0] # Number of test images
    dim_x = depth_gts.shape[1]
    dim_y = depth_gts.shape[2]
    s_pad=int((P-1)/2)         # Index of the middle pixel
    fns = np.arange(ff)        # Indexes used to follow the employed frequencies
    num_tested = 0
    img_name = os.path.basename(name)[:-3]

    # The following matrices are preallocated to keep track of the errors pixel per pixel on each dataset image
    errors = np.zeros((n_img,dim_x,dim_y,2*ff),dtype = np.float32)
    errors_denoise = np.zeros((n_img,dim_x,dim_y,2*ff),dtype = np.float32)
    errors_1f = np.zeros((n_img,dim_x,dim_y,2*ff),dtype = np.float32)
    errors_bl = np.zeros((n_img,dim_x,dim_y,ff),dtype = np.float32)
    
    # Variables used to keep track of the average error per image
    avg1f = np.zeros(ff)
    avgpr = np.zeros(ff)
    avgprf = np.zeros(ff)
    avgmin = 0
    avgmean = 0
    avgmax = 0

    # Network definition and weights loading
    net = PredictiveModel.PredictiveModel(name='DeepBVE_nf_std', dim_b=1, freqs=freqs, P=P, saves_path='./saves',
                                          fil_size=num_fil_direct, fil_denoise_size=num_fil_denoise)

    if fl_denoise:
        net.SpatialNet.load_weights(weights_path[0])
    net.decoder.load_weights(weights_path[1])
    net.encoder.load_weights(weights_path[2])
    net.predv_encoding.load_weights(weights_path[3])
    net.DirectCNN.load_weights(weights_path[4])

    # Input processing (scaling by the 20 Mhz amplitude)
    for i in range(n_img):
        v_in = v_ins[i]
        ampl = np.sqrt(v_in[...,:ff]**2+v_in[...,ff:]**2)
        max_phasor_ampl = ampl[...,0]
        max_phasor_ampl = max_phasor_ampl[...,np.newaxis]

        v_in = np.divide(v_in, max_phasor_ampl)
        #else:
       #     v_in = scale_vin(v_in,P)
        v_ins[i] = v_in

    v_input = tf.convert_to_tensor(np.pad(v_ins,pad_width=[[0,0],[s_pad,s_pad],[s_pad,s_pad],[0,0]],mode="edge"),dtype="float32")

    # Inference
    if fl_denoise:
        v_mid = net.SpatialNet(v_input)
    else:
        v_mid = v_input
    [v_out_g, v_out_d, v_free] = net.DirectCNN(v_mid)
    phid = np.arctan2(v_out_d[:,:,:,ff:],v_out_d[:,:,:,:ff])
    phid = np.where(np.isnan(phid),0.,phid)   # Needed for the first epochs to correct the output in case of a 0/0
    phid = phid%(2*math.pi)
        
    # Depth computation 
    depth_predictions = phid*utils.c()/(4*utils.pi()*freqs)

    if fl_img_PhDRT:
        v_free = np.squeeze(v_free)
        dim_encoding = 4*ff
        v_out_encoding = np.concatenate((v_free,v_out_g),axis=-1)
        v_out_encoding = np.reshape(v_out_encoding,(n_img,-1,dim_encoding)) # If the middle network is not used

        x_out_g = np.squeeze(net.decoder(v_out_encoding))
        x_out_g = np.reshape(x_out_g,(n_img,dim_x,dim_y,-1))
        #shape_diff = 2000 - x_out_g.shape[-1]
        #missing = np.zeros((x_out_g.shape[0],x_out_g.shape[1],shape_diff),dtype=np.float32)
        #x_out_g = np.concatenate((missing,x_out_g),axis=-1)
        #print(x_out_g.shape)
        [Ad,depthpr] = utils.x_direct_from_itof_direct(v_out_d) 
        x_out_d = np.zeros(x_out_g.shape)
        time_ind = (depthpr*2000/5).astype(np.int32)
        time_ind = time_ind[...,np.newaxis]
        Ad = Ad[...,np.newaxis]
        np.put_along_axis(x_out_d,time_ind,Ad,-1)
        x_out = x_out_d + x_out_g

        ind_save = [2,4,5]
        for ind in ind_save:
            out_name = "./out_PhDRT/S4_" + str(ind) + ".h5"
            with h5py.File(out_name,"w") as f:
                f.create_dataset(name = "transient",data=x_out[ind])
        sys.exit()


    # cycle with phase unwrapping, image plots and error computation
    for k in range(n_img):
        
        depth_prs = depth_predictions[k]   # Focus on a single image
        depth_gt = depth_gts[k]
        v_in = v_ins[k]
        depth_valid = np.where(depth_gt<0.3,0.,1.)   # Identification of the image pixels which are valid
        
        # Useful strings for saving the plots
        fmod_str = list(map(str,(freqs/10e05).astype(np.int32)))
        for i in range(ff):
            fmod_str[i] += "MHz"
        
        
        errs_pr = np.zeros(ff)
        errs_prf = np.zeros(ff)
        errs_1f = np.zeros(ff)

        # Cycle on the three modulation frequencies
        for fn in fns:
            if fn == 0:
                depth20  = depth_prs[...,0]   # Used for phase unwrapping
            fmod=freqs[fn]

            # Depth computation with the naive approach (single frequency)
            v_fmod = np.stack([v_in[...,fn], v_in[...,fn+ff]], axis=-1)
            depth_1f = depth_estimation.freq1_depth_estimation(v_fmod, fmod=fmod)
            # Compute ambiguity range for fmod
            amb_range = utils.c() / (2*fmod)
            # Compensate depth offset from ground truth
            n_amb = np.round((depth20-depth_1f)/amb_range)
            depth_1f += n_amb*amb_range


            v_out_fmod = np.stack([v_out_d[:,:,:,fn], v_out_d[:,:,:,fn+ff]], axis=-1)
            depth_pr = depth_prs[...,fn]
            # Phase unwrapping (Not really needed in this case as all images are very close range)
            if  fn == 1 or fn == 2:
                amb_range = utils.amb_range(fmod)
                n_amb = np.round((depth20-depth_pr)/amb_range)

            # Error computation for the predicted depth
            error_pr = np.squeeze(depth_pr - depth_gt)
            mae_pr = np.sum(depth_valid*np.abs(error_pr)) / np.sum(depth_valid)
            avg_mae_pr = np.mean(mae_pr)
            depth_prf = np.ndarray(depth_pr.shape, dtype=np.float32)
            
            # Bilateral filtering
            depth_prf = cv2.bilateralFilter(np.float32(depth_pr), -1, sigmaColor=0.8, sigmaSpace=3)  

            print("MAE ", np.sum(np.abs(depth_gt-depth_pr)*depth_valid)/np.sum(depth_valid)*100, " cm")
            print("MAE bil ", np.sum(np.abs(depth_gt-depth_prf)*depth_valid)/np.sum(depth_valid)*100, " cm")
            # Error computation for the predicted depth
            error_prf = np.squeeze(depth_prf - depth_gt)


            errors[num_tested,:,:,fn] = np.abs(error_pr)
            errors_bl[num_tested,:,:,fn] = np.abs(error_prf)

            mae_prf = np.sum(depth_valid*np.abs(error_prf)) / np.sum(depth_valid)
            avg_mae_prf = np.mean(mae_prf)
        
            # Error computation for single frequency
            error_1f = np.squeeze(depth_1f - depth_gt)
            errors_1f[num_tested,:,:,fn] = np.abs(error_1f)
            errors[num_tested,:,:,ff+fn] = np.abs(error_1f)
            mae_1f = np.sum(depth_valid*np.abs(error_1f)) / np.sum(depth_valid)
            avg_mae_1f = np.mean(mae_1f)
            avg_mae_pr = np.round(avg_mae_pr*100,3)
            avg_mae_prf = np.round(avg_mae_prf*100,3)
            avg_mae_1f = np.round(avg_mae_1f*100,3)          
            errs_pr[fn] = avg_mae_pr
            errs_prf[fn] = avg_mae_prf
            errs_1f[fn] = avg_mae_1f
            M = 0.2


            depth_vali = depth_valid
            err_pr = np.squeeze(error_pr)*depth_valid  
            err_prf = np.squeeze(error_prf)*depth_valid
            err_1f = np.squeeze(error_1f)*depth_valid
            img_pr =  np.sum(np.abs(err_pr)) / np.sum(depth_vali)
            img_prf =  np.sum(np.abs(err_prf)) / np.sum(depth_vali)
            img_1f =  np.sum(np.abs(err_1f)) / np.sum(depth_vali)
            img_pr = np.round(img_pr*100,2)
            img_prf = np.round(img_prf*100,2)
            img_1f = np.round(img_1f*100,2)

            # Compute pixelwise minimum, mean and maximum between the predictions at different frequencies
            if fn == 0:
                depth_min = np.copy(depth_prf)
                depth_mean = np.copy(depth_prf)
                depth_max = np.copy(depth_prf)
            if fn==1 or fn==2:
                depth_min = np.where(depth_prf<depth_min,depth_prf,depth_min)
                depth_max = np.where(depth_prf>depth_max,depth_prf,depth_max)
                depth_mean += depth_prf
            # In case of 2 frequencies
            if fn == 2 or (fn==1 and ff==2):
                depth_mean/=ff
                err_min = np.squeeze(depth_min-depth_gt)
                err_minv = np.sum(np.abs(err_min)*depth_vali)/np.sum(depth_vali)
                err_mean = np.squeeze(depth_mean-depth_gt)
                err_meanv = np.sum(np.abs(err_mean)*depth_vali)/np.sum(depth_vali)
                err_max = np.squeeze(depth_max-depth_gt)
                err_maxv = np.sum(np.abs(err_max)*depth_vali)/np.sum(depth_vali)
                avgmin += err_minv
                avgmean += err_meanv
                avgmax += err_maxv
                print("MIN ", err_minv*100, "cm")
                print("MEAN ", err_meanv*100, "cm")
                print("MAX ", err_maxv*100, "cm")

            avg1f[fn]+=img_1f
            avgpr[fn]+=img_pr
            avgprf[fn]+=img_prf

            # Plot the results
            if fl_plot:
                err_1f = np.where(depth_valid==0,-5,err_1f)
                err_pr = np.where(depth_valid==0,-5,err_pr)
                err_prf = np.where(depth_valid==0,-5,err_prf)
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
                plt.savefig("./out/"+attempt_name+"/" + Sname + "/" +img_name+"_"+ fmod_str[fn]+"_"+str(k) ,dpi=1200)
                plt.close()
        
                if fn==2:
                    err_min = np.where(depth_valid==0,-5,err_min)
                    err_mean = np.where(depth_valid==0,-5,err_mean)
                    err_max = np.where(depth_valid==0,-5,err_max)
                    err_minv = np.round(100*err_minv,2)
                    err_meanv = np.round(100*err_meanv,2)
                    err_maxv = np.round(100*err_maxv,2)
                    plt.figure()
                    plt.subplot(2,2,1)
                    plt.imshow(err_1f, cmap="jet")
                    plt.title("Direct MAE: " +  str(img_1f) + " [cm]", fontsize=6)
                    cbar = plt.colorbar(fraction=0.046, pad=0.04)
                    plt.clim(-M,M)
                    cbar.set_label('Depth [m]', labelpad=10, fontsize=6)
                    plt.subplot(2,2,2)
                    plt.imshow(err_min, cmap="jet")
                    plt.title("Min operation MAE: " +  str(err_minv) + " [cm]", fontsize=6)
                    cbar = plt.colorbar(fraction=0.046, pad=0.04)
                    plt.clim(-M,M)
                    cbar.set_label('Depth [m]', labelpad=10, fontsize=6)
                    plt.subplot(2,2,3)
                    plt.imshow(err_mean, cmap="jet")
                    plt.title("Mean operation MAE: " +  str(err_meanv) + " [cm]", fontsize=6)
                    cbar = plt.colorbar(fraction=0.046, pad=0.04)
                    plt.clim(-M,M)
                    cbar.set_label('Depth [m]', labelpad=10, fontsize=6)
                    plt.subplot(2,2,4)
                    plt.imshow(err_max, cmap="jet")
                    plt.title("Max operation MAE: " +  str(err_maxv) + " [cm]", fontsize=6)
                    cbar = plt.colorbar(fraction=0.046, pad=0.04)
                    plt.clim(-M,M)
                    cbar.set_label('Depth [m]', labelpad=10, fontsize=6)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.savefig("./out/"+attempt_name+"/" + Sname + "/" +img_name+"_operations_"+str(k) ,dpi=1200)
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
    errors = errors[:num_tested,...,1]*1000
    errors_1f = errors_1f[:num_tested,...,1]*1000
    errors_bl = errors_bl[:num_tested,...,1]*1000
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





