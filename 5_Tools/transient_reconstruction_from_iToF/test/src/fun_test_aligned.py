import numpy as np
import os
import tensorflow as tf
import h5py
import math
import matplotlib
import scipy
import scipy.signal
import glob
import matplotlib.pyplot as plt
import time
import pandas as pd
import cv2
import sys
sys.path.append("../training/src/")
sys.path.append("../utils/") # Adds higher directory to python modules path
import DataLoader
import GenerativeModel
import utils
import depth_estimation
font = {'size'   : 6}
#import PredictiveModel_itof2dtof as PredictiveModel
import PredictiveModel_hidden as PredictiveModel
#import PredictiveModel_hidden_2 as PredictiveModel
import PredictiveModel_old
from fun_metrics_computation_transient_images import metrics_computation_transient_images
#import PredictiveModel_2freq_big as PredictiveModel
matplotlib.rc('font', **font)



def phi_remapping(v,d_min=0,d_max=4):
    # Function used to map the phasor to a desired range
    fn = v.shape[-1]
    ampl = np.sqrt(v[...,:fn]**2 + v[...,fn:]**2)
    phi = np.arctan2(v[...,fn:],v[...,:fn])
    phi = (phi+2*utils.pi())%(2*utils.pi())
    phi = phi/7.5*d_max
    phi = (phi-utils.pi())%(2*utils.pi()) - utils.pi()
    v_new = np.copy(v)
    v_new[...,:fn] = ampl*np.cos(phi)
    v_new[...,fn:] = ampl*np.sin(phi)
    
    return v_new


def test_aligned(name,weights_path,attempt_name,Sname,P,freqs,fl_scale,fl_norm_perpixel,fil_dir,fil_den,fil_auto,fl_test_old=False,old_weights=None,Pold=3,fl_newhidden=True,dim_t=2000):

    fl_matte = True          # Whether we are using the simulated transient with matte or plastic
    fl_real_input = True     # Whether we are using real data or synthetic as input
    ff = freqs.shape[0]
    path = "../../dataset_rec/"
    if fl_matte:
        test_files = "../../Datasets/syn_real_alignment/real_synth_aligned.h5"
    else:
        test_files = "../../Datasets/syn_real_alignment/real_synth_aligned_plastic.h5"
    data_dict = {}
    st = time.time()
    with h5py.File(test_files, "r") as f:
        print(f.keys())
        for key in f.keys():
            data_dict[key] = f[key][:]
    en = time.time()


    dim_dataset=1


    # If we are testing also the old network, load the weights and needed data to test that one too
    if fl_test_old:
        net_old = PredictiveModel_old.PredictiveModel(name='DeepBVE_nf_std',
                                               dim_b = dim_dataset,
                                               P = Pold, 
                                               freqs = freqs,
                                               saves_path='./saves')
        gen_old = GenerativeModel.GenerativeModel(dim_b = dim_dataset,
                                                    dim_x=1,
                                                    dim_y=1,
                                                    dim_t=2000)

        if old_weights is None:
            print("Using old network, but no weights were provided")
            sys.exit()
        if P != 3:
            net_old.SpatialNet.load_weights(old_weights[0])
        net_old.DirectCNN.load_weights(old_weights[1])
        net_old.TransientNet.load_weights(old_weights[2])

    net = PredictiveModel.PredictiveModel(name='DeepBVE_nf_std',
                                               dim_b = dim_dataset,
                                               P = P, 
                                               freqs = freqs,
                                               saves_path='./saves',
                                               dim_t=dim_t,
                                               fil_size=fil_dir,
                                               fil_denoise_size=fil_den,
                                               fil_encoder=fil_auto)

    if P != 3:
        net.SpatialNet.load_weights(weights_path[0])
    net.decoder.load_weights(weights_path[1])
    net.encoder.load_weights(weights_path[2])
    net.predv_encoding.load_weights(weights_path[3])
    net.DirectCNN.load_weights(weights_path[4])

    avg_s = []
    avg_sm = []
    avg_b = []
    avg_mae = []
    avg_emd = []
    avg_pdf = []
    avg_cdf = []
    fracs = []
    all_names = []
    avg_s_old = []
    avg_sm_old = []
    avg_b_old = []
    avg_mae_old = []
    avg_emd_old = []
    avg_pdf_old = []
    avg_cdf_old = []
    fracs_old = []
    all_names_old = []

    #  Direct metrics
    dir_amp_b = []
    dir_amp_p = []
    dir_pos_b = []
    dir_pos_p = []

    num_imgs = data_dict["transient"].shape[0]
    print("loading time: ", en-st, " [s]")
    for xx in range(num_imgs):
        st1 = time.time() 
        maxv=2 
        if xx < maxv:
            continue
        valid_mask = data_dict["mask"][xx]

        tr = data_dict["transient"][xx]
        x_d = data_dict["transient_direct"][xx]
        x_g = data_dict["transient_global"][xx]
        v_in = data_dict["raw_itof_syn"][xx]
        v_in_real = data_dict["raw_itof"][xx]
        v_d_gt = data_dict["direct_itof"][xx]
        v_g_gt = data_dict["global_itof"][xx]
        dim_t_data = tr.shape[-1]
    

        phi = np.transpose(utils.phi(freqs,dim_t))
        #tr = np.swapaxes(tr,0,1)
        print("Mean magnitude" , np.mean(np.sum(tr,axis=-1)))
        dep = np.argmax(tr,axis=-1)/2000*2.5



        s_pad = int((P-1)/2) 
        (dim_x,dim_y,dim_t) = tr.shape
        if fl_scale:
            # Compute the scaling kernel for the image pixels
            ampl = np.sqrt(v_in[...,:3]**2+v_in[...,3:]**2)
            norm_fact = np.ones((P,P))/P**2
            ampl20 = np.squeeze(ampl[...,0])
            norm_fact = scipy.signal.convolve2d(ampl20,norm_fact,mode="same")
            if fl_norm_perpixel:
                norm_fact = ampl[...,0]
            
            v_in = v_in[np.newaxis,:,:,:] 
            norm_fact = norm_fact[np.newaxis,...,np.newaxis]



            

            # Give the correct number of dimensions to each matrix and then scale them 
            norm_fact = np.where(norm_fact==0.,1.,norm_fact)
            v_in /= norm_fact
            v_d_gt = v_d_gt[np.newaxis,...]
            v_d_gt /= norm_fact[0,...]
            v_g_gt = v_g_gt[np.newaxis,...]
            v_g_gt /= norm_fact[0,...]
            norm_fact = np.squeeze(norm_fact)
            norm_fact = norm_fact[...,np.newaxis]
            x_d/=norm_fact
            x_g/=norm_fact
            tr/=norm_fact

            #Compute the scale for the real images
            ampl = np.sqrt(v_in_real[...,:3]**2+v_in_real[...,3:]**2)
            norm_fact_real = np.ones((P,P))/P**2
            ampl20 = np.squeeze(ampl[...,0])
            norm_fact_real = scipy.signal.convolve2d(ampl20,norm_fact_real,mode="same")
            if fl_norm_perpixel:
                norm_fact_real = ampl[...,0]
            
            v_in_real = v_in_real[np.newaxis,:,:,:] 
            norm_fact_real = norm_fact_real[np.newaxis,...,np.newaxis]
            v_in_real/=norm_fact_real
        else:
            v_in = v_in[np.newaxis,:,:,:] 

        v_diff = (v_in-v_in_real)*valid_mask[...,None]
        norm_fact = np.where(norm_fact==1,0.,norm_fact)
       
        a_real = np.sqrt(v_in_real[...,0]**2+v_in_real[...,3]**2)
        a_real6 = np.sqrt(v_in_real[...,2]**2+v_in_real[...,5]**2)
        a_in = np.sqrt(v_in[...,0]**2+v_in[...,3]**2)
        a_in6 = np.sqrt(v_in[...,2]**2+v_in[...,5]**2)
        ind = 120

        plt.figure()
        plt.imshow(np.squeeze(v_in[...,0]),cmap="jet")
        plt.colorbar()
        plt.figure()
        plt.imshow(np.squeeze(v_in_real[...,0]),cmap="jet")
        plt.colorbar()
        plt.figure()
        plt.imshow(np.squeeze(v_in[...,3]),cmap="jet")
        plt.colorbar()
        plt.figure()
        plt.imshow(np.squeeze(v_in_real[...,3]),cmap="jet")
        plt.colorbar()
        plt.figure()
        plt.title("Diff 20 MHz")
        plt.imshow(np.squeeze(v_diff[...,0]),cmap="jet")
        plt.colorbar()
        plt.figure()
        plt.imshow(np.squeeze(v_diff[...,1]),cmap="jet")
        plt.figure()
        plt.imshow(np.squeeze(v_diff[...,2]),cmap="jet")
        plt.figure()
        plt.imshow(np.squeeze(v_diff[...,3]),cmap="jet")
        plt.figure()
        plt.imshow(np.squeeze(v_diff[...,4]),cmap="jet")
        plt.figure()
        plt.imshow(np.squeeze(v_diff[...,5]),cmap="jet")
        plt.show()

        if fl_real_input:
            v_in = v_in_real
        

        fl_denoise = not(net.P==net.out_win)   #If the two values are different, then the denoising network has been used
        # Make prediction
        #v_input = tf.convert_to_tensor(np.pad(v_in,pad_width=[[0,0],[s_pad,s_pad],[s_pad,s_pad],[0,0]],mode="edge"),dtype="float32")
        v_input = np.pad(v_in,pad_width=[[0,0],[s_pad,s_pad],[s_pad,s_pad],[0,0]],mode="edge")
        if fl_denoise:
            v_in_v = net.SpatialNet(v_input)
        else: 
            v_in_v = v_input
     
        if fl_newhidden:
            [v_out_g, v_out_d,  v_free] = net.DirectCNN(v_in_v)
            if v_out_g.shape[-1] != v_out_d.shape[-1]:
                v_out_g = v_in - v_out_d
            phid = np.arctan2(v_out_d[:,:,:,ff:],v_out_d[:,:,:,:ff])
            phid = np.where(np.isnan(phid),0.,phid)   # Needed for the first epochs to correct the output in case of a 0/0
            phid = phid%(2*math.pi)
            v_out_d = np.squeeze(v_out_d)
            v_out_g = np.squeeze(v_out_g)
            val = np.mean(np.abs(v_out_g-np.squeeze(v_g_gt)))/2
            #plt.figure()
            #plt.imshow(np.squeeze(v_in_v[...,0]),cmap="jet")
            #plt.figure()
            #plt.imshow(np.abs(v_out_g-np.squeeze(v_g_gt))[...,0],cmap="jet")
            #plt.colorbar()
            #plt.clim(-val,val)
            #plt.figure()
            #plt.imshow(np.abs(v_out_g-np.squeeze(v_g_gt))[...,1],cmap="jet")
            #plt.colorbar()
            #plt.clim(-val,val)
            #plt.figure()
            #plt.imshow(np.abs(v_out_g-np.squeeze(v_g_gt))[...,2],cmap="jet")
            #plt.colorbar()
            #plt.clim(-val,val)
            #plt.figure()
            #plt.imshow(np.abs(v_out_g-np.squeeze(v_g_gt))[...,3],cmap="jet")
            #plt.colorbar()
            #plt.clim(-val,val)
            #plt.figure()
            #plt.imshow(np.abs(v_out_g-np.squeeze(v_g_gt))[...,4],cmap="jet")
            #plt.colorbar()
            #plt.clim(-val,val)
            #plt.figure()
            #plt.imshow(np.abs(v_out_g-np.squeeze(v_g_gt))[...,5],cmap="jet")
            #plt.colorbar()
            #plt.clim(-val,val)
            #plt.show()
            v_free = np.squeeze(v_free)
            v_out_encoding = np.concatenate((v_free,v_out_g),axis=-1)
        else:
            [v_out_g, v_out_d,  phid] = net.DirectCNN(v_in_v)
            v_out_d = np.squeeze(v_out_d)
            v_out_g = np.squeeze(v_out_g)
            v_out_encoding = np.concatenate((v_out_d,v_out_g),axis=-1) # If the middle network is not used
        v_out_encoding = np.reshape(v_out_encoding,(-1,12)) # If the middle network is not used
        x_out_g = np.squeeze(net.decoder(v_out_encoding))
        x_out_g = np.reshape(x_out_g,(dim_x,dim_y,-1))


        shape_diff = x_g.shape[-1] - x_out_g.shape[-1]
        missing = np.zeros((x_out_g.shape[0],x_out_g.shape[1],shape_diff),dtype=np.float32)
        x_out_g = np.concatenate((missing,x_out_g),axis=-1)

        # Older network
        if fl_test_old:
            s_pad_old = int((Pold-1)/2) 
            v_input_old = np.pad(v_in,pad_width=[[0,0],[s_pad_old,s_pad_old],[s_pad_old,s_pad_old],[0,0]],mode="edge")
            if Pold != 3:
                v_in_v_old = net_old.SpatialNet(v_input_old)
            else: 
                v_in_v_old = v_input_old

            [v_out_g_old, v_out_d_old,  phid_old] = net_old.DirectCNN(v_in_v_old)
            z_old = net_old.TransientNet([v_out_g_old,v_out_d_old])
            # original network predicts the cumulative. We need to get back to the original space
            xg_old = gen_old(z_old)
            xg_old_cum = np.squeeze(xg_old)
            xg_old = xg_old_cum[...,1:] - xg_old_cum[...,:-1]
            all_z = np.zeros((xg_old_cum.shape[0],xg_old_cum.shape[1],1),dtype=np.float32)
            xg_old = np.concatenate((all_z,xg_old),axis=-1)

            
            print("TEST ON OLD NETWORK")
            err_so, err_som, err_bo, err_maeo, err_emdo, mean_pdfo, mean_cdfo, frac_starto = metrics_computation_transient_images(tr, x_g, xg_old, v_out_d_old, np.squeeze(v_in))
            if not np.isnan(err_so):
                avg_s_old.append(err_so)
            if not np.isnan(err_som):
                avg_sm_old.append(err_som)
            if not np.isnan(err_so):
                avg_b_old.append(err_bo)
            if not np.isnan(err_maeo):
                avg_mae_old.append(err_maeo)
            if not np.isnan(err_emdo):
                avg_emd_old.append(err_emdo)
            if not np.isnan(mean_pdfo):
                avg_pdf_old.append(mean_pdfo)
            if not np.isnan(mean_cdfo):
                avg_cdf_old.append(mean_cdfo)
            if not np.isnan(err_so):
                fracs_old.append(frac_starto)
            if not np.isnan(err_so):
                all_names_old.append(os.path.basename(name))


        v_in = tf.squeeze(v_in)

    

        
        err_s, err_sm, err_b, err_mae, err_emd, mean_pdf, mean_cdf, frac_start, dir_amp_base, dir_amp_pred, dir_pos_base, dir_pos_pred  = metrics_computation_transient_images(tr, x_g, x_out_g, v_out_d, v_in, mask=valid_mask, fl_matte=fl_matte)
        
        # AUTOENCODER PART
        fl_auto = False
        if fl_auto:
            v_enc = net.encoder(x_g.reshape(-1,2000))
            if fl_newhidden:
                v_g_gt = np.reshape(v_g_gt,(-1,2*ff))
                v_enc = np.squeeze(v_enc)
                print(v_enc.shape)
                print(v_g_gt.shape)
                v_enc = np.concatenate((v_enc,v_g_gt),axis=-1)
            x_dec = net.decoder(np.squeeze(v_enc))
            x_dec = np.reshape(x_dec,(dim_x,dim_y,-1))
            x_dec = np.concatenate((missing,x_dec),axis=-1)
            err_s, err_sm,  err_b, err_mae, err_emd, mean_pdf, mean_cdf, frac_start, dir_amp_base, dir_amp_pred, dir_pos_base, dir_pos_pred  = metrics_computation_transient_images(tr, x_g, x_dec, v_out_d, v_in)

        if not np.isnan(err_s):
            avg_s.append(err_s)
        if not np.isnan(err_sm):
            avg_sm.append(err_sm)
        if not np.isnan(err_s):
            avg_b.append(err_b)
        if not np.isnan(err_mae):
            avg_mae.append(err_mae)
        if not np.isnan(err_emd):
            avg_emd.append(err_emd)
        if not np.isnan(mean_pdf):
            avg_pdf.append(mean_pdf)
        if not np.isnan(mean_cdf):
            avg_cdf.append(mean_cdf)
        if not np.isnan(err_s):
            fracs.append(frac_start)
        if not np.isnan(err_s):
            all_names.append(os.path.basename(name))
        dir_amp_b.append(dir_amp_base)
        dir_amp_p.append(dir_amp_pred)
        dir_pos_b.append(dir_pos_base)
        dir_pos_p.append(dir_pos_pred)
        continue
    
        err = np.abs(v_d_gt-v_out_d)
        plt.figure()
        plt.scatter(np.abs(v_in).flatten(),err.flatten())
        plt.show()

        v_in_err = np.squeeze(np.abs(v_in-v_d_gt))
        v_in_err = np.mean(v_in_err,axis=-1)
        c = 0
        vd_err = np.squeeze(np.abs(v_out_d-v_d_gt))
        #vd_err = np.mean(vd_err,axis=-1)
        Adgt = np.sqrt(v_d_gt[...,:3]**2+v_d_gt[...,3:]**2)
        Adpr = np.sqrt(v_out_d[...,:3]**2+v_out_d[...,3:]**2)
        Ad_err = np.squeeze(np.abs(Adpr-Adgt))
        Ad_err = np.mean(Ad_err,axis=-1)
        print("vd error ", np.mean(vd_err))
        print("vin error ", np.mean(np.abs(v_in_err)))
        plt.figure()
        plt.title("ground truth")
        plt.imshow(v_d_gt[...,c],cmap="jet")
        plt.colorbar()
        plt.figure()
        plt.title("prediction")
        plt.imshow(v_out_d[...,c],cmap="jet")
        plt.colorbar()
        plt.figure()
        plt.title("Vd prediction error")
        plt.imshow(vd_err[...,0],cmap="jet")
        plt.colorbar()
        plt.clim(0,1)
        plt.show()

        plt.figure()
        plt.title("Amplitude prediction error")
        plt.imshow(Ad_err,cmap="jet")
        plt.colorbar()
        plt.clim(-0.2,0.2)


        A = np.sqrt(v_out_d[...,:3]**2+v_out_d[...,3:]**2)
        A0 = np.squeeze(A[...,0]) 
        x_pr = np.squeeze(gen(z_out))
        x_pr_cum = np.copy(x_pr)
        x_pr = x_pr[...,1:]-x_pr[...,:-1]

        t = np.arange(0,5,0.0025)
        t1 = np.arange(0,5-0.0025,0.0025)

        x_gt = tr
        x_gt_nod = np.copy(x_gt)
        peak_v = np.zeros((x_gt.shape[0],x_gt.shape[1]),dtype=np.float32)
        x_pr_d = np.squeeze(np.copy(x_pr))
        Add,indd = utils.x_direct_from_itof_direct(np.squeeze(v_out_d))
        Agg = np.max(x_pr_cum,axis=-1)
        indd = (indd*400).astype(np.int32)
        for i in range(x_gt_nod.shape[0]):
            for j in range(x_gt_nod.shape[1]):
                x_gt_nod[i,j,:np.argmax(x_gt_nod[i,j])+5] = 0
                peak_v[i,j] = np.sum(x_gt[i,j,:np.argmax(x_gt[i,j])+5])
                x_pr_d[i,j,indd[i,j]] = Add[i,j]

        plt.figure()
        plt.title("Total light difference")
        plt.imshow(np.sum(x_pr_d,axis=-1)-np.sum(tr,axis=-1),cmap="jet")
        plt.colorbar()
            
        x_gtc_nod = np.cumsum(x_gt_nod,axis=-1)
        print(np.mean(np.max(x_gtc_nod,axis=-1)))
        print(np.mean(np.max(x_pr_cum,axis=-1)))
        ratio_gd = np.max(x_gtc_nod,axis=-1)/peak_v
        plt.figure()
        plt.title("true global magnitude")
        plt.hist(np.max(x_gtc_nod,axis=-1).flatten(),bins=100)
        plt.savefig("./out_back/hist_gt.png")
        plt.figure()
        plt.title("predicted global magnitude")
        plt.hist(np.max(x_pr_cum,axis=-1).flatten(),bins=100)
        plt.savefig("./out_back/hist_pred.png")
    




    
        err_img = np.mean(np.abs(x_gtc_nod-x_pr_cum),axis=-1)
        fl_save_back = False
        if fl_save_back:
            num_save = 20
            for i in range(num_save):
                t = np.arange(0,2000,1)/400
                t1 = np.arange(0,1999,1)/400

                indy = np.random.randint(dim_x)
                indx = np.random.randint(dim_y)
                #plt.title("" )
                temp_gt_d = np.copy(x_gt[indx,indy,:])
                temp_gt_g = np.copy(temp_gt_d)
                temp_gt_d[np.argmax(temp_gt_d)+5:]=0
                temp_gt_g[:np.argmax(temp_gt_g)+5]=0
                temp_gt_g = np.where(temp_gt_g==0,np.nan,temp_gt_g)
            
                temp_pr_d = np.copy(x_pr_d[indx,indy,:])
                temp_pr_g = np.copy(temp_pr_d)
                temp_pr_d[np.argmax(temp_pr_d)+5:]=0
                temp_pr_g[:np.argmax(temp_pr_g)+5]=0


                plt.figure()
                plt.plot(t,temp_gt_d,"b")
                plt.scatter(t,temp_gt_g,s=5,c="b")
                plt.plot(t1,temp_pr_d,"red")
                plt.scatter(t1,temp_pr_g,s=5,c="red")
                plt.yscale("log")
                plt.legend(["ground truth","SDT network prediction"])
                #plt.show()
                plt.savefig("./out_back/global_pred_"+str(indy)+"_"+str(indx)+".png",bbox_inches='tight')
                plt.close()


        ratio_pred = np.max(x_pr_cum,axis=-1)/(1-np.max(x_pr_cum,axis=-1))
        ratio_pred2 = np.max(x_pr_cum,axis=-1)/A0
        plt.figure()
        plt.title("Predicted G-D ratio")
        plt.imshow(ratio_pred,cmap="jet")
        plt.colorbar()
        plt.clim(-0.6,0.6)
        plt.savefig("./out_back/ratio_pred.png",bbox_inches='tight')

        plt.figure()
        plt.title("Predicted G-D ratio with direct")
        plt.imshow(ratio_pred,cmap="jet")
        plt.colorbar()
        plt.clim(-0.6,0.6)
        plt.savefig("./out_back/ratio_pred_direct.png",bbox_inches='tight')


        plt.figure()
        plt.title("ratio")
        plt.imshow(ratio_gd,cmap="jet")
        plt.colorbar()
        plt.clim(-0.6,0.6)
        plt.savefig("./out_back/ratio_image.png",bbox_inches='tight')
        plt.figure()
        plt.title("error")
        plt.imshow(err_img,cmap="jet")
        plt.colorbar()
        plt.clim(-0.2,0.2)
        plt.savefig("./out_back/error_image.png",bbox_inches='tight')

        plt.figure()
        plt.title("error ratio pred")
        plt.imshow(err_img/ratio_pred,cmap="jet")
        plt.colorbar()
        plt.clim(-0.2,0.2)
        plt.figure()
        plt.title("error ratio gt")
        plt.imshow(err_img/ratio_gd,cmap="jet")
        plt.colorbar()
        plt.clim(-0.2,0.2)
        plt.show()



        # VIDEO
        #with h5py.File("synth_gt" + str(ind) +".h5","w") as f:
        #    f.create_dataset(name="transient_pred",data=tr)
        #    sys.exit()
        #fr = x_pr[...,0]
        #im = plt.imshow(fr)
        #for i in range(dim_t-2):
        #    frame = x_pr[...,i]
        #    im.set_data(frame)
        #    plt.pause(0.02)
        #plt.show()
    fracs_old = np.asarray(fracs_old)
    fracs_old/=np.sum(fracs_old)
    mean_s_old = np.sum(100*np.asarray(avg_s_old)*np.asarray(fracs_old))
    mean_sm_old = np.sum(100*np.asarray(avg_sm_old)*np.asarray(fracs_old))
    mean_b_old = np.sum(100*np.asarray(avg_b_old)*np.asarray(fracs_old))
    print("OLD NETWORK")
    print("   ")
    print("Error on the start is of ",mean_s_old, " cm")
    print("Error on the start (based on max) is of ",mean_sm_old, " cm")
    print("Error on the baricenter is of ", mean_b_old, " cm")
    print("MAE error is of ", np.mean(avg_mae_old))
    print("EMD error is of ", np.mean(avg_emd_old))
    print("The mean PDF is of ", np.mean(avg_pdf_old))
    print("The mean CDF is of ", np.mean(avg_cdf_old))
    print("start errors are ", np.round(avg_s_old,2)*100)
    print("fractions are ", fracs_old)
    print("names are ", all_names_old)

    plt.figure()
    plt.title("OLD histogram of start error")
    plt.hist(avg_s,bins=10)
    plt.figure()
    plt.title("OLD histogram of baricenter error")
    plt.hist(avg_b,bins=10)
    plt.figure()
    plt.title("OLD histogram of MAE")
    plt.hist(avg_mae,bins=10)
    plt.figure()
    plt.title("OLD histogram of EMD")
    plt.hist(avg_emd,bins=10)
    plt.figure()
    plt.title("OLD histogram of average pdf")
    plt.hist(avg_pdf,bins=10)
    plt.figure()
    plt.title("OLD histogram of average cdf")
    fracs = np.asarray(fracs)
    fracs/=np.sum(fracs)
    mean_s = np.sum(100*np.asarray(avg_s)*np.asarray(fracs))
    mean_sm = np.sum(100*np.asarray(avg_sm)*np.asarray(fracs))
    mean_b = np.sum(100*np.asarray(avg_b)*np.asarray(fracs))
    print("NEW NETWORK")
    print("   ")
    print("DIRECT COMPONENT METRICS")
    print("Baseline error on the direct position (60 MHz)", 100*np.round(np.mean(dir_pos_b),4), " cm")
    print("Network error on the direct position ", 100*np.round(np.mean(dir_pos_p),4) , " cm")
    print("Baseline error on the direct amplitude ", np.mean(dir_amp_b))
    print("Network error on the direct ampltiude ", np.mean(dir_amp_p))

    print("   ")
    print("GLOBAL COMPONENT METRICS")
    print("Error on the start is of ",mean_s, " cm")
    print("Error on the start (based on max) is of ",mean_sm, " cm")
    print("Error on the baricenter is of ", mean_b, " cm")
    print("MAE error is of ", np.mean(avg_mae))
    print("EMD error is of ", np.mean(avg_emd))
    print("The mean PDF is of ", np.mean(avg_pdf))
    print("The mean CDF is of ", np.mean(avg_cdf))
    print("start errors are ", np.round(avg_s,2)*100)
    print("fractions are ", fracs)
    print("names are ", all_names)

    plt.figure()
    plt.title("histogram of start error")
    plt.hist(avg_s,bins=10)
    plt.figure()
    plt.title("histogram of baricenter error")
    plt.hist(avg_b,bins=10)
    plt.figure()
    plt.title("histogram of MAE")
    plt.hist(avg_mae,bins=10)
    plt.figure()
    plt.title("histogram of EMD")
    plt.hist(avg_emd,bins=10)
    plt.figure()
    plt.title("histogram of average pdf")
    plt.hist(avg_pdf,bins=10)
    plt.figure()
    plt.title("histogram of average cdf")
    plt.hist(avg_cdf,bins=10)
    plt.show()