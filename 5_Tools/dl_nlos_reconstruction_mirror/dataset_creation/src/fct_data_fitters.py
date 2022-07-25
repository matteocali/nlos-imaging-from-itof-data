import numpy as np
from class_fit import fitData, fitCum
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys
# The followinf function fits the incoming backscattering vector fitting one curve at a time
def split_fitter(a1_,a2_,b1_,b2_,depth,back,indpeak2):
    
    # Fitting first peak
    data_fit = fitData()
    data_fit.b1 = b1_
    bou = ([0,0.8,a1_],[3,2.6,np.inf])
    popt1, popc1 = curve_fit(data_fit.weibull_mod,depth[:indpeak2-1],back[:indpeak2-1],p0=[1,1,a1_],method="lm")
    fit_first = data_fit.weibull_mod(depth,popt1[0],popt1[1],popt1[2])
    
    # Fitting second peak
    if indpeak2 != 2000:
        data_fit = fitData()
        data_fit.b1 = b2_
        back1 = back - fit_first
        back1[:indpeak2-1] = 0
        #meanv1 = meanv - fit_first
        #meanv1[:indpeak2-1] = 0
        bou = ([0,0.8,a2_],[3,2.6,np.inf])
        popt1, popc1 = curve_fit(data_fit.weibull_mod,depth[indpeak2-1:],back1[indpeak2-1:],p0=[1,1,a2_],method="lm")
        fit_second = data_fit.weibull_mod(depth,popt1[0],popt1[1],popt1[2])
    else:
        fit_second = np.zeros((fit_first.shape),dtype = np.float32)
    return fit_first + fit_second




# The following fitter tries to fit directly the input data
def data_fitter(a1_,b1_,b2_,depth,meanv,maxval):
    
    data_fit = fitData()
    data_fit.b1 = b1_
    data_fit.b2 = b2_
    data_fit.maxval = maxval
    bou = ([0.01,0.01,0.8,0.8,a1_],[np.inf,np.inf,2.6,2.6,np.inf])
    popt, popc = curve_fit(data_fit.weibull_mix,depth,meanv,p0=[0.3,1,1,1.2,a1_],method="lm")
    fit_data = data_fit.weibull_mix(depth,popt[0],popt[1],popt[2],popt[3],popt[4])
    
    return fit_data

def data_fitter_lam(a1_,a2_,b1_,b2_,depth,meanv,maxval):
    
    data_fit = fitData()
    data_fit.b1 = b1_
    data_fit.b2 = b2_
    data_fit.maxval = maxval
    #bou = ([0.8,0.8,a1_,a1_],[2.6,2.6,np.inf,np.inf])
    if a2_==0:
        a2min = 0
        a2max = 0.01
    else:
        a2min = 0
        a2max = np.inf
    bou = ([0.8,1,a1_,-np.inf],[2.6,2.6,np.inf,np.inf])
    popt, popc = curve_fit(data_fit.weibull_mix_mod_lam,depth,meanv,p0=[1,1.1,a1_,a2_],bounds=bou)
    #popt, popc = curve_fit(data_fit.weibull_mix_mod_lam,depth,meanv,p0=[1,1.2,a1_,a2_],method="lm")
    fit_data = data_fit.weibull_mix_mod_lam(depth,popt[0],popt[1],popt[2],popt[3])
    
    return fit_data
    
    
def cumul_fitter(a1_,b1_,b2_,depth,cumv):

    cum_fit = fitCum()
    cum_fit.b1 = b1_
    cum_fit.b2 = b2_
    cum_fit.maxval = cumv[-1]
    data_fit = fitData()
    data_fit.b1 = b1_
    data_fit.b2 = b2_
    data_fit.maxval = cumv[-1]
    if a1_ >= cum_fit.maxval:
        a1_ = cum_fit.maxval-0.01*cum_fit.maxval

    bou = ([0.01,0.01,0.6,0.6,a1_],[0.8,0.8,1.3,1.3,cum_fit.maxval])
    a1_start = (cum_fit.maxval+a1_)/2
    if a1_==cum_fit.maxval:
        popt, popc = curve_fit(cum_fit.weibull_mix_cum,depth,cumv,p0=[0.4,0.4,1,1,a1_start])
    else:
        popt, popc = curve_fit(cum_fit.weibull_mix_cum,depth,cumv,p0=[0.4,0.4,1,1,a1_start],bounds = bou)
    fit_cum = cum_fit.weibull_mix_cum(depth,popt[0],popt[1],popt[2],popt[3],popt[4])
    fit_data = data_fit.weibull_mix(depth,popt[0],popt[1],popt[2],popt[3],popt[4])
    ind =  int(b2_*400)
    
    fit_data[ind] = fit_data[ind+2]
    fit_data[ind+1] = fit_data[ind+2]
    params = np.zeros((6),dtype = np.float32)
    params[:5] = popt[:]
    if b2_ <= 0:
        params[5] = 0
    else:
        params[5] = cum_fit.maxval-popt[4]
    return fit_cum,fit_data,params
    
    