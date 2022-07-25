import numpy as np
import sys
"""
The file implements some different loss functions over transient data.
Each loss function takes in input two matrices of the same size and computes the desired loss between the two.
For each of the loss functions also a normalized version is provided:
    the loss per pixel is computed by dividing for the pixel mean. The average is computed over the last dimension of the first image. Both images are normalized for the same values.
"""


def loss_mse(A,B,mask=None):
    if np.any(mask==None):
        mask = np.zeros((A.shape[:-1]),dtype=np.int32)
    MSE = np.mean((A-B)**2,axis=-1)
    MSE = np.ma.masked_array(MSE,mask=mask)
    MSE = np.mean(MSE)
    return MSE
    
def loss_mae(A,B,mask=None):
    if np.any(mask==None):
        mask = np.zeros((A.shape[:-1]),dtype=np.int32)
    MAE = np.mean(np.abs(A-B),axis=-1)    
    MAE = np.ma.masked_array(MAE,mask=mask)
    MAE = np.mean(MAE)      
    return MAE
    
def loss_EMD(A,B,mask=None):
    if np.any(mask==None):
        mask = np.zeros((A.shape[:-1]),dtype=np.int32)
    sum_A = np.sum(A,axis=-1)
    
    cum_A = np.cumsum(A,axis=-1)
    cum_B = np.cumsum(B,axis=-1)
    mask1 = np.where(sum_A==0,1,0)    
    EMD = np.abs(cum_A-cum_B)/sum_A[...,np.newaxis]
    EMD = np.mean(EMD,axis=-1)
    EMD = np.ma.masked_array(EMD,mask=mask1)

    EMD = np.mean(EMD)
    return EMD
    
    
def loss_wEMD(A,B,mask=None):
    if np.any(mask==None):
        mask = np.zeros((A.shape[:-1]),dtype=np.int32)
    sum_A = np.sum(A,axis=-1)
    mask1 = np.where(sum_A==0,1,0)    

    cum_A = np.cumsum(A,axis=-1)
    cum_B = np.cumsum(B,axis=-1)
    w = np.zeros((cum_A.shape),dtype= np.float32)
    win = 100
    for i in range(cum_A.shape[-1]):
        min_ind = np.max(i-win,0)
        rwin = i-min_ind
        w[...,i] =  np.sum(np.abs(cum_A[...,min_ind:i]-cum_B[...,min_ind:i]))/rwin
    wEMD = np.abs(cum_A-cum_B)*w/sum_A[...,np.newaxis]
    wEMD = np.mean(wEMD,axis=-1)
    wEMD = np.ma.masked_array(wEMD,mask=mask1)
    wEMD = np.mean(wEMD)
    return wEMD

