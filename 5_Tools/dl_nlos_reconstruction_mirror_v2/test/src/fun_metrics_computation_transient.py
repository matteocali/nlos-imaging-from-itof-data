import numpy as np
import matplotlib.pyplot as plt
import os,sys 
sys.path.append("../utils/")
import utils
import depth_estimation


"""
This function computes various metrics monitoring the accuracy of a model for what concerns its prediction of direct and global components 

"""




def metrics_computation_transient(trans, trans_nod, pred_trans_nod, pred_vd, v, freqs=np.array((20e06,50e06,60e06),dtype=np.float32)):
    dim_b = trans.shape[0]
    dim_t = trans.shape[1]
    phi = utils.phi(freqs,dim_t)
    nf = freqs.shape[0]
    max_d = 0.5*utils.max_t(dim_t)*utils.c()

    # 0) Clean the vector
    pred_trans_nod = np.squeeze(pred_trans_nod)
    ind_pmax = np.argmax(pred_trans_nod,axis=-1)
    #pred_trans_nod = pred_trans_nod.numpy()
    st_val = pred_trans_nod[:,0]
    st_val = st_val[:,np.newaxis]
    pred_trans_nod = pred_trans_nod - st_val
    pred_trans_nod = np.where(pred_trans_nod<0,0,pred_trans_nod)
    for i in range(ind_pmax.shape[0]):
        if ind_pmax[i]-100>0:
            temp_vec = pred_trans_nod[i,ind_pmax[i]-100:ind_pmax[i]]
            min_ind = np.argmin(temp_vec)
            #pred_trans_nod[i,:ind_pmax[i]-min_ind] = 0

    # 1) Metrics computation for the direct component
    print("1) Metrics regarding the direct component")

    trans_d = trans - trans_nod          # Compute the transient composed of the direct component alone
    gt_ind = np.argmax(trans,axis=-1)    # Find the position of the maximum
    gt_pos = gt_ind/dim_t*max_d  # Convert in meters

    # Find the value of the magnitude by summing all bins of the direct
    gt_mag = np.sum(trans_d,axis=-1)

    # Compute the predicted estimation as an average between the predictions at the various frequencies
    # N.B. the minimum operation instead improves the performance of the baseline and worsens the one of the network
    pred_base = np.zeros((nf,gt_pos.shape[0]),dtype=np.float32)
    pred_pos = np.zeros((nf,gt_pos.shape[0]),dtype=np.float32)
    for i in range(nf):
        pred_base[i,:] = depth_estimation.freq1_depth_estimation(v[:,[i,i+nf]],freqs[i])
        pred_pos[i,:] = depth_estimation.freq1_depth_estimation(pred_vd[:,[i,i+nf]],freqs[i])
        if i>0:    # Take care of phase unwrapping for higher frequencies
            amb_range = utils.amb_range(freqs[i])
            pred_base[i] = np.where(pred_base[0]-pred_base[i]>amb_range/2,pred_base[i]+amb_range,pred_base[i])
            pred_pos[i] = np.where(pred_pos[0]-pred_pos[i]>amb_range/2,pred_pos[i]+amb_range,pred_pos[i])

    #base_mean =  np.mean(pred_base,axis=0)
    base_mean =  pred_base[-1]
    pred_mean =  np.mean(pred_pos,axis=0)

    MAE_depth_base = np.mean(np.abs(gt_pos-base_mean))
    MAE_depth_direct = np.mean(np.abs(gt_pos-pred_mean))

    for i in range(nf):
        print("freq" , freqs[i])
        print("base",np.mean(np.abs(pred_base[i]-gt_pos))*100)
        print("net",np.mean(np.abs(pred_pos[i]-gt_pos))*100)
    print("DEPTH")
    print("Baseline MAE on the depth of the direct component consists of {} cm".format(MAE_depth_base*100))
    print("MAE on the depth of the direct component consists of {} cm".format(MAE_depth_direct*100))


    # Computation of the error on the magnitude of the direct component
    base_ampl = np.sqrt(v[...,:nf]**2 + v[...,nf:]**2)
    pred_ampl = np.sqrt(pred_vd[...,:nf]**2 + pred_vd[...,nf:]**2)
    


    ###
    print(gt_mag.shape)
    print(base_ampl.shape)
    print(pred_ampl.shape)
    for i in range(nf):
        print("freq" , freqs[i])
        print("base",np.mean(np.abs(base_ampl[...,i]-gt_mag)))
        print("net",np.mean(np.abs(pred_ampl[...,i]-gt_mag)))

    ###
    base_ampl =  np.mean(base_ampl,axis=-1)
    pred_ampl =  np.mean(pred_ampl,axis=-1)

    err_base = gt_mag-base_ampl
    err_pred = gt_mag-pred_ampl

    MAE_ampl_base = np.mean(np.abs(err_base))
    MAE_ampl_direct = np.mean(np.abs(err_pred))

    print("MAGNITUDE")
    print("Baseline MAE on the magnitude of the direct component consists of {} ".format(MAE_ampl_base))
    print("MAE on the network prediction of the magnitude of the direct component consists of {} ".format(MAE_ampl_direct))


    
    ###################################################

    # Metrics computation regarding the global component
    print("2) Metrics regarding the global component")


    # Compute the cumulatives
    trans_nod_cum = np.cumsum(trans_nod,axis=-1)
    pred_trans_nod_cum = np.cumsum(pred_trans_nod,axis=-1)
    #pred_trans_nod_cum = pred_trans_nod

    # MAE between the distributions
    MAE = np.mean(np.abs(trans_nod-pred_trans_nod))

    # EMD (MAE between the cumulatives)
    EMD = np.mean(np.abs(trans_nod_cum-pred_trans_nod_cum))
    EMD_sign = np.mean((trans_nod_cum-pred_trans_nod_cum))
    mae_temp = np.mean(np.abs(trans_nod-pred_trans_nod),axis=-1)

    print("The mean value of the gt distribution is: ", np.mean(trans_nod))
    print("The MAE between the distribution is: ", MAE)
    print("The mean value of the gt cumulative is: ", np.mean(trans_nod_cum))
    print("The MAE between the cumulatives is: ", EMD)
    print("The ME between the cumulatives is: ", EMD_sign)


    


    # Error on the starting point of the cumulative
    gt_start = np.argmax(np.where(trans_nod_cum>0,1,0),axis=-1)/dim_t*max_d
    #pred_start = np.argmax(np.where(pred_trans_nod_cum>0,1,0),axis=-1)/dim_t*max_d
    pred_start = np.argmax(pred_trans_nod,axis=-1)/dim_t*max_d

    # Compute the error keeping out the elements with all 0 ground truth
    err_start = np.abs(gt_start-pred_start)
    #err_start = gt_start-pred_start
    mask_start = np.where((np.max(trans_nod_cum,axis=-1)>0) & (np.max(pred_trans_nod_cum,axis=-1)>0))
    err_start_copy = np.copy(err_start)
    err_start = err_start[mask_start[0]]
    print("The error on the start of the global consists of ", np.mean(err_start)*100, "[cm]")

    # Error on the baricenter of the predicted global
    pos = np.arange(dim_t)
    pos = pos[:,np.newaxis]
    weights_gt = np.sum(trans_nod,axis=-1)
    weights_gt = np.where(weights_gt==0,1,weights_gt)
    weights_gt = weights_gt[:,np.newaxis]
    bar_pos = (trans_nod/weights_gt)@pos/dim_t*max_d
    weights = np.sum(pred_trans_nod,axis=-1)
    weights = np.where(weights==0,1,weights)
    weights = weights[:,np.newaxis]
    pred_bar_pos = (pred_trans_nod/weights)@pos/dim_t*max_d   # Baricenter computation making the global a distribution

    err_bar = np.abs(bar_pos-pred_bar_pos)
    #err_bar = (bar_pos-pred_bar_pos).numpy()
    err_bar_copy = np.copy(err_bar)
    err_bar = err_bar[mask_start[0]]
    

    print("The error on the baricenter of the global is of ",  np.mean(err_bar)*100,  "[cm]" )

    num_show = 40
    num =np.random.randint(err_start_copy.shape[0],size=num_show)
    for i in num:
        plt.figure()
        plt.plot(trans_nod[i,:],"g")
        plt.plot(pred_trans_nod[i,:],"b")
        plt.axis("off")
        #plt.figure()
        #plt.title("start "+ str(np.round(err_start_copy[i]*100,2))+ "cm  baricenter "+ str(np.round(err_bar_copy[i][0]*100,2) ) + " cm" + " MAE " + str(mae_temp[i]))
        #plt.plot(trans_nod[i,:],"g")
        #plt.axis("off")
        #plt.figure()
        #plt.plot(pred_trans_nod[i,:],"g")
        #plt.axis("off")
    plt.show()

    #sys.exit()