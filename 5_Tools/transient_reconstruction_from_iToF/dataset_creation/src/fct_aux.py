import numpy as np
import matplotlib.pyplot as plt


def find_ind_peaks_full(cumv,wsize):
 
    # The function finds the index of the first and second peaks of the backscattering vector for each of the 
    # pixels in the image. The search is based on an approximation of the second derivative of the empirical cumulative distribution.
    
    #Inputs:
    # -empirical cumulative distribution for each pixel (WITHOUT THE DIRECT COMPONENT): size = WxHxB
    # -window size: Size of the window used to compute the second derivative (the total window size is 2xwsize)
    
    #Outputs:
    # -Second derivative of the empirical cumulative distribution cumputed pixel by pixel: size = WxHxB
    # -The indexes of the first and second noise peaks of the backscattering vectors. The second one is returned as ind2=2000 if the second peak does not exist (too small w.r.t. the rest)
    der2 = np.zeros((cumv.shape),dtype = np.float32)
    # Find first peak
    for i in range(wsize+1,cumv.shape[2]-wsize):
        upper = np.sum(np.abs(cumv[:,:,i+1:i+1+wsize]-cumv[:,:,i:i+wsize]),axis=-1)
        lower = np.sum(np.abs(cumv[:,:,i-wsize:i]-cumv[:,:,i-1-wsize:i-1]),axis=-1)
        
        der2[:,:,i] = upper[:,:]-lower[:,:]
    temp = np.where(cumv>0,1,0)
    ind1 = np.argmax(temp,axis=-1)
    for i in range(cumv.shape[0]):
        for j in range(cumv.shape[1]):
            if der2[i,j,ind1[i,j]] < 0.33*np.max(der2[i,j,:]):
                ind1[i,j] = np.argmax(der2[i,j,:],axis=-1)
    #ind1 = np.argmax(der2,axis=-1)
    tmp = np.copy(der2)
    # Clean all information regarding v1
    #tmp[:,:,:ind1+wsize] = 0
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            tmp[i,j,:ind1[i,j]+wsize] = 0
    ind2 = np.argmax(tmp,axis=-1)
    
    # Check if the second peak is a peak or not
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            tmp[i,j,np.argmax(tmp[i,j])-wsize:np.argmax(tmp[i,j])+wsize] = 0
    #tmp[:,:,np.argmax(tmp)-wsize:np.argmax(tmp)+wsize] = 0
    ind3 = np.argmax(tmp,axis=-1)
    for i in range(ind2.shape[0]):
        for j in range(ind2.shape[1]):
            if der2[i,j,ind3[i,j]] > 0.75*der2[i,j,ind2[i,j]]:   # TOO SLOW
                ind2[i,j] = 2000
    #ind2[der2[ind3]>0.75*der2[ind2]] = 2000
    
    return der2,ind1,ind2

def find_ind_peaks_vec(cumv,wsize):
 
    # The function finds the index of the first and second peaks of the backscattering vector for each of the 
    # pixels in the image. The search is based on an approximation of the second derivative of the empirical cumulative distribution.
    
    #Inputs:
    # -empirical cumulative distribution for each pixel (WITHOUT THE DIRECT COMPONENT): size = WxHxB
    # -window size: Size of the window used to compute the second derivative (the total window size is 2xwsize)
    
    #Outputs:
    # -Second derivative of the empirical cumulative distribution cumputed pixel by pixel: size = WxHxB
    # -The indexes of the first and second noise peaks of the backscattering vectors. The second one is returned as ind2=2000 if the second peak does not exist (too small w.r.t. the rest)

    der2 = np.zeros((cumv.shape),dtype = np.float32)
    
    # Find first peak
    for i in range(wsize+1,cumv.shape[-1]-wsize):
        upper = np.sum(np.abs(cumv[:,i+1:i+1+wsize]-cumv[:,i:i+wsize]),axis=-1)
        lower = np.sum(np.abs(cumv[:,i-wsize:i]-cumv[:,i-1-wsize:i-1]),axis=-1)
        
        der2[:,i] = upper[:]-lower[:]
    temp = np.where(cumv>0,1,0)
    ind1 = np.argmax(temp,axis=-1)
    for i in range(cumv.shape[0]):
        if der2[i,ind1[i]] < 0.33*np.max(der2[i,:]):
            ind1[i] = np.argmax(der2[i,:],axis=-1)
    tmp = np.copy(der2)
    
    # Clean all information regarding v1
    for i in range(tmp.shape[0]):
        tmp[i,:ind1[i]+wsize] = 0
    ind2 = np.argmax(tmp,axis=-1)
    
    # Check if the second peak is a peak or not
    for i in range(tmp.shape[0]):
        tmp[i,np.argmax(tmp[i])-wsize:np.argmax(tmp[i])+wsize] = 0
        
    ind3 = np.argmax(tmp,axis=-1)
    for i in range(ind2.shape[0]):
        if der2[i,ind3[i]] > 0.75*der2[i,ind2[i]]:  
                ind2[i] = 2000
    
    return der2,ind1,ind2
    
    
def find_ind_peaks(cumv,wsize):
 
    # The function finds the index of the first and second peaks of the backscattering vector for a single pixel.
    #The search is based on an approximation of the second derivative of the empirical cumulative distribution.
    
    #Inputs:
    # -empirical cumulative distribution of the pixel (WITHOUT THE DIRECT COMPONENT): size = B
    # -window size: Size of the window used to compute the second derivative (the total window size is 2xwsize)
    
    #Outputs:
    # -Second derivative of the empirical cumulative distribution cumputed pixel by pixel: size = B
    # -The indexes of the first and second noise peaks of the backscattering vector. The second one is returned as ind2=2000 if the second peak does not exist (too small w.r.t. the rest)
 
    der2 = np.zeros((cumv.shape),dtype = np.float32)
    # Find first peak
    for i in range(wsize+1,cumv.shape[0]-wsize):
        upper = np.sum(np.abs(cumv[i+1:i+1+wsize]-cumv[i:i+wsize]))
        lower = np.sum(np.abs(cumv[i-wsize:i]-cumv[i-1-wsize:i-1]))
        der2[i] = upper-lower
    
    temp = np.where(cumv>0,1,0)
    ind1 = np.argmax(temp)
    if der2[ind1] < 0.33*np.max(der2):
        ind1 = np.argmax(der2)
    tmp = np.copy(der2)
    # Clean all information regarding v1
    tmp[:ind1+wsize] = 0
    ind2 = np.argmax(tmp)
    
    # Check if the second peak is a peak or not
    tmp[np.argmax(tmp)-wsize:np.argmax(tmp)+wsize] = 0
    ind3 = np.argmax(tmp)
    if der2[ind3] > 0.75*der2[ind2]:
        ind2 = 2000
    return der2,ind1,ind2
