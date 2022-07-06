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
import GenerativeModel
import utils
import depth_estimation
font = {'size'   : 6}
import PredictiveModel_hidden as PredictiveModel
#import PredictiveModel_hidden_2 as PredictiveModel
matplotlib.rc('font', **font)
import random
import csv
from fun_metrics_computation_transient import metrics_computation_transient

# Function for transient data prediction on single pixels
# It also presents the comparison with the pixels from iToF2dToF


def test_synth(name,weights_path,attempt_name,Sname,P,freqs,fl_test_img=False,fl_scale=False,fil_dir=8,fil_den=32,fil_auto=32,fl_newhidden=True, dim_t=2000):

    # index pointing at the middle element of the patch
    mind = int((P-1)/2)

    ff = freqs.shape[0]
    dim_encoding = ff*4
    with h5py.File(name,"r") as f:
        v_in = f["raw_itof"][:]
        v_in_d = f["direct_itof"][:]
        v_in_g = f["global_itof"][:]
        xd = f["transient"][:]
        xg = f["transient_global"][:]
    
    dim_dataset = v_in.shape[0]
    dim_x = v_in.shape[1]
    dim_y = v_in.shape[2]
    dim_t_data = xg.shape[-1]
    
    if dim_t != dim_t_data:
        xd_new = np.zeros((dim_dataset,dim_t),dtype=np.float32)
        xg_new = np.zeros((dim_dataset,dim_t),dtype=np.float32)
        xd_new[...,:dim_t_data] = xd
        xg_new[...,:dim_t_data] = xg
        xd = xd_new
        xg = xg_new

    # SCALING
    if fl_scale:
        v_a = np.sqrt( v_in[...,0]**2 + v_in[...,ff]**2 ) 
        v_a = np.mean(v_a,axis=(1,2))
        v_a_max = v_a[:,np.newaxis,np.newaxis,np.newaxis]
        scale = np.copy(np.squeeze(v_a_max))
            
        # Scale all factors
        v_in /= v_a_max
        v_in_d /= v_a_max
        v_in_g /= v_a_max
        v_a_max = np.squeeze(v_a_max)
        v_a_max = v_a_max[...,np.newaxis]
        xd /= v_a_max
        xg /= v_a_max     




 
    net = PredictiveModel.PredictiveModel(name='DeepBVE_nf_std',
                                               dim_b = dim_dataset,
                                               P = P, 
                                               freqs = freqs,
                                               saves_path='./saves',
                                               fil_size=fil_dir,
                                               dim_t=dim_t,
                                               dim_encoding=dim_encoding,
                                               fil_denoise_size=fil_den,
                                               fil_encoder=fil_auto)


    if P != 3:
        net.SpatialNet.load_weights(weights_path[0])
    net.decoder.load_weights(weights_path[1])
    net.encoder.load_weights(weights_path[2])
    net.predv_encoding.load_weights(weights_path[3])
    net.DirectCNN.load_weights(weights_path[4])


    fl_denoise = not(net.P==net.out_win)   #If the two values are different, then the denoising network has been used
        
    # Ground truth global itof component
    v_in_g = v_in_g[:,mind,mind,:]


    # Make prediction
    #v_input = tf.convert_to_tensor(v_in,dtype="float32")
    v_input = np.copy(v_in)
    if fl_denoise:
        v_in_v = net.SpatialNet(v_input)
    else: 
        v_in_v = v_input

    if fl_newhidden: 
        [v_out_g, v_out_d, v_free] = net.DirectCNN(v_in_v)
        if v_out_g.shape[-1] != v_out_d.shape[-1]:
            v_out_g = v_in[:,mind,mind,:]-np.squeeze(v_out_d)
        phid = np.arctan2(v_out_d[:,:,:,ff:],v_out_d[:,:,:,:ff])
        phid = np.where(np.isnan(phid),0.,phid)   # Needed for the first epochs to correct the output in case of a 0/0
        phid = phid%(2*math.pi)
        v_out_d = np.squeeze(v_out_d)
        v_out_g = np.squeeze(v_out_g)
        v_free = np.squeeze(v_free)

        # Network prediction
        v_gd = np.concatenate((v_free,v_out_g),axis=-1)
        x_CNN = net.decoder(v_gd)
        x_CNN = np.squeeze(x_CNN)
        
        # Autoencoder prediction
        v_enc = net.encoder(xg)
        v_enc = np.squeeze(v_enc)
        v_enc = np.concatenate((v_enc,v_in_g),axis=-1)
        x_auto = net.decoder(v_enc)
        x_auto = np.squeeze(x_auto)

    else:
        [v_out_g, v_out_d,  phid] = net.DirectCNN(v_in_v)

        v_out_d = np.squeeze(v_out_d)
        v_out_g = np.squeeze(v_out_g)


    
        # Network prediction
        v_gd = np.concatenate((v_out_d,v_out_g),axis=-1)
        x_CNN = net.decoder(v_gd)
        x_CNN = np.squeeze(x_CNN)

        # Autoencoder prediction
        v_enc = net.encoder(xg)
        v_enc = np.squeeze(v_enc)
        x_auto = net.decoder(v_enc)
        x_auto = np.squeeze(x_auto)
    
    # Fix shape of output to match input by adding zeroes at the beginning
    shape_diff = xg.shape[-1] - x_CNN.shape[-1]
    missing = np.zeros((x_CNN.shape[0],shape_diff),dtype=np.float32)
    x_auto = np.concatenate((missing,x_auto),axis=-1)
    x_CNN = np.concatenate((missing,x_CNN),axis=-1)
    

    #metrics_computation_transient(xd, xg, x_auto, v_out_d, v_in[:,mind,mind,:])
    ####
    print("PATCHES: SCALED")
    print("\n")

    metrics_computation_transient(xd, xg, x_CNN, v_out_d, v_in[:,mind,mind,:],freqs=freqs)

    # Bring everything back to the original scale
    scale = scale[:,np.newaxis]
    xd *= scale
    xg *= scale
    x_out_g1= x_CNN*scale
    v_out_d *= scale
    scale = scale[...,np.newaxis,np.newaxis]
    v_in *= scale


    print("PATCHES: UNSCALED")
    print("\n")

    metrics_computation_transient(xd, xg, x_out_g1, v_out_d, v_in[:,mind,mind,:])


    # Perform the inference on full images
    if fl_test_img:
        # Load a particular set of images (trainin/validation/test)
        files = []
        with open("../dataset_creation/data_split/test_images.csv", "r") as csvfile:
            wr = csv.reader(csvfile)
            for row in wr:
                files.append(os.path.basename(row[0]))
        img_path = "../../dataset_rec/"
        img_names = []
        for fil in files:
            img_names.append(img_path+fil)
        num_img = len(img_names)
        #img_names = glob.glob(img_path+"*.h5")
        #random.shuffle(img_names)
        s_pad = int((P-1)/2)
        MAE_tot = 0
        MAE_base = 0
        mean_v = 0
        mean_vd = 0
        query_name = "twalls_TOF_11_rec.h5"



        for el,name in enumerate(img_names):
            print(os.path.basename(name))
            #if os.path.basename(name) != query_name:
            #    continue
            with h5py.File(name,"r") as f:
                for key in f.keys():
                    image = f[key][:]
            #image = image[30:61,30:61]

            #image = image[200:301,0:101,:]
            dim_x = image.shape[0]
            dim_y = image.shape[1]
            dim_t = image.shape[2]
            phi = np.transpose(utils.phi(freqs,dim_t))
           



            # Compute the IQ components
            ind_maxima = np.argmax(image,axis=-1)
            val_maxima = np.zeros(ind_maxima.shape,dtype=np.float32)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    val_maxima[i,j] = np.sum(image[i,j,:ind_maxima[i,j]+5])
                    image[i,j,:ind_maxima[i,j]+5] = 0
                    image[i,j,ind_maxima[i,j]] = val_maxima[i,j]

            v_im = image@phi


            # Split the direct and global components

            im_d = np.zeros(image.shape,dtype=np.float32)
            im_g = np.zeros(image.shape,dtype=np.float32)
            

            for i in range(dim_x):
                for j in range(dim_y):
                    im_d[i,j,:np.argmax(image[i,j,:])+5] = image[i,j,:np.argmax(image[i,j,:])+5]
                    im_g[i,j,np.argmax(image[i,j,:])+5:] = image[i,j,np.argmax(image[i,j,:])+5:]

            # Compute the v vectors corresponding to direct and global components

            v_im_d = im_d@phi
            v_im_g = im_g@phi
            mean_v += np.mean(np.abs(v_im))
            mean_vd += np.mean(np.abs(v_im_d))

            # Compute phase and amplitude metrics of all v vectors

            A_im = np.sqrt(v_im[...,:3]**2 + v_im[...,3:]**2)
            A_im_g = np.sqrt(v_im_g[...,:3]**2 + v_im_g[...,3:]**2)
            A_im_d = np.sqrt(v_im_d[...,:3]**2 + v_im_d[...,3:]**2)

            phi_im = np.arctan2(v_im[...,3:],v_im[...,:3])
            phi_im_g = np.arctan2(v_im_g[...,3:],v_im_g[...,:3])
            phi_im_d = np.arctan2(v_im_d[...,3:],v_im_d[...,:3])

            # Pad the input before giving it to the network
            v_im_input = v_im[np.newaxis,...]
            v_im_input = tf.convert_to_tensor(np.pad(v_im_input,pad_width=[[0,0],[s_pad,s_pad],[s_pad,s_pad],[0,0]],mode="edge"),dtype="float32")

            # Inference
            if fl_denoise:
                v_im_input = net.SpatialNet(v_im_input)
            

            [v_im_g_out, v_im_d_out,  phi_im_d_out] = net.DirectCNN(v_im_input)
            phi_im_d_out = np.squeeze(phi_im_d_out.numpy())
            A_im_d_out = np.sqrt(v_im_d_out[...,:3]**2+v_im_d_out[...,3:]**2)
            A_im_d_out = np.squeeze(A_im_d_out)

            ####
            for i in range(3):
                plt.figure()
                plt.title("ampl")
                plt.imshow(A_im_d[...,i]-A_im_d_out[...,i],cmap="jet")
                plt.colorbar()
                plt.figure()
                plt.title("phase")
                plt.imshow(phi_im_d[...,i]-phi_im_d_out[...,i],cmap="jet")
                plt.colorbar()
                plt.figure()
                plt.title("vdiff")
                plt.imshow(np.squeeze(v_im_d[...,i]-v_im_d_out[...,i]),cmap="jet")
                plt.colorbar()
            plt.show()
            
            ####

            # Computation of a few error metrics on the ouput values
            v_im_d_out = np.squeeze(v_im_d_out)
            print("Shapes")
            print(v_im.shape,v_im_d.shape,v_im_d_out.shape)
            print("MAE between prediction and ground truth: ", np.mean(np.abs(v_im_d_out-v_im_d)))
            print("MAE between input and ground truth: ", np.mean(np.abs(v_im-v_im_d)))
            MAE_tot += np.mean(np.abs(v_im_d_out-v_im_d))
            MAE_base += np.mean(np.abs(v_im-v_im_d))
        MAE_tot/=num_img
        MAE_base/=num_img
        mean_v/=num_img
        mean_vd/=num_img
        print("Num images ",num_img) 
        print("Network MAE: ", MAE_tot)
        print("Baseline MAE: ", MAE_base)
        print("Mean absolute v: ", mean_v)
        print("Mean absolute vd: ", mean_vd)


