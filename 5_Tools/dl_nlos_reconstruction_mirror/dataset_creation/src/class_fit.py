import numpy as np


class fitData():
    def __init__(self):
        pass
     
    def weibull_mix_mod(self,x,lam1,lam2,k1,k2,a1):
        mix = np.zeros((x.shape),dtype=np.float32)
        b1 = self.b1
        b2 = self.b2
        maxval = self.maxval
        a2 = maxval-a1
        if (b1>=b2):
            mix[x-b1<=0] = 0
            mix[x-b1>0] = a1*((x[x>b1]-b1)/lam1)**(k1-1)*np.exp(-(x[x>b1]-b1)/lam1)
            
        else:
            mix[x>=b2] = a1*((x[x>=b2]-b1)/lam1)**(k1-1)*np.exp(-(x[x>=b2]-b1)/lam1) + a2*((x[x>=b2]-b2)/lam2)**(k2-1)*np.exp(-(x[x>=b2]-b2)/lam2)
            mix[(x>b1)&(x<b2)] = a1*((x[(x>b1)&(x<b2)]-b1)/lam1)**(k1-1)*np.exp(-(x[(x>b1)&(x<b2)]-b1)/lam1) 

        return mix
        
    def weibull_mix_mod_lam(self,x,k1,k2,a1,a2):
        mix = np.zeros((x.shape),dtype=np.float32)
        b1 = self.b1
        b2 = self.b2
        maxval = self.maxval
        #a2 = (maxval-a1/k1)*k2
        if (b1>=b2):
            mix[x-b1<=0] = 0
            mix[x-b1>0] = a1*((x[x>b1]-b1))**(k1-1)*np.exp(-(x[x>b1]-b1)**k1)
            
        else:
            with np.errstate(divide='raise'):
                try:
                    mix[x>=b2] = a1*((x[x>=b2]-b1))**(k1-1)*np.exp(-(x[x>=b2]-b1)**k1) + a2*((x[x>=b2]-b2))**(k2-1)*np.exp(-(x[x>=b2]-b2)**k2)
                except FloatingPointError:
                    print(a1,a2,b1,b2,k1,k2)
                    print((x[x>=b2]-b2))
                    print(((x[x>=b2]-b2))**(k2-1))
            mix[(x>b1)&(x<b2)] = a1*((x[(x>b1)&(x<b2)]-b1))**(k1-1)*np.exp(-(x[(x>b1)&(x<b2)]-b1)**k1) 
        return mix
        
    def weibull_mix(self,x,lam1,lam2,k1,k2,a1):
        mix = np.zeros((x.shape),dtype=np.float32)
        b1 = self.b1
        b2 = self.b2
        maxval = self.maxval
        a2 = maxval-a1
        if (b1>=b2):
            mix[x-b1<=0] = 0
            mix[x-b1>0] = a1*k1/lam1*((x[x>b1]-b1)/lam1)**(k1-1)*np.exp(-((x[x>b1]-b1)/lam1)**k1)
           
        else:
            mix[x>b2] = a1*k1/lam1*((x[x>b2]-b1)/lam1)**(k1-1)*np.exp(-((x[x>b2]-b1)/lam1)**k1) + a2*k2/lam2*((x[x>b2]-b2)/lam2)**(k2-1)*np.exp(-((x[x>b2]-b2)/lam2)**k2)
            mix[(x>b1)&(x<=b2)] = a1*k1/lam1*((x[(x>b1)&(x<=b2)]-b1)/lam1)**(k1-1)*np.exp(-((x[(x>b1)&(x<=b2)]-b1)/lam1)**k1) 

        return mix
    
    def rayleigh_mix(self,x,sig1,sig2,a1,a2,b1,b2):
        
        mix = np.zeros((x.shape),dtype=np.float32)
        b1 = self.b1
        if (b1>=b2):
            mix[x-b1<=0] = 0
            mix[x-b1>0] = a1*((x[x-b1>0]-b1)/sig1**2*(np.exp(-(x[x-b1>0]-b1)**2/(2*sig1**2))))
        else:
            mix[x-b1<=0] = a2*((x[x-b1<=0]-b2)/sig2**2*(np.exp(-(x[x-b1<=0]-b2)**2/(2*sig2**2)))) 
            mix[x-b1>0] = a1*((x[x-b1>0]-b1)/sig1**2*(np.exp(-(x[x-b1>0]-b1)**2/(2*sig1**2)))) + a2*((x[x-b1>0]-b2)/sig2**2*(np.exp(-(x[x-b1>0]-b2)**2/(2*sig2**2)))) 
        
        return mix
        
    def weibull(self,x,lam,k,a):
        mix = np.zeros((x.shape),dtype = np.float32)
        b = self.b1
        mix[x<=b] = 0
        mix[x>b] = a*k/lam*((x[x>b]-b)/lam)**(k-1)*np.exp(-((x[x>b]-b)/lam)**k)    
        return mix
        
    def weibull_mod(self,x,lam,k,a):
        mix = np.zeros((x.shape),dtype = np.float32)
        b = self.b1
        mix[x<=b] = 0
        with np.errstate(divide='raise'):
            try:
                #mix[x>b] = a*((x[x>b]-b)/lam)**(k-1)*np.exp(-((x[x>b]-b)/lam))    
                mix[x>b] = a*(np.abs((x[x>b]-b)/lam))**(k-1)*np.exp(-(np.abs((x[x>b]-b)/lam)))    
            except FloatingPointError:
                print(a,b,lam,k)
                print((x[x>b]-b)/lam)
        return mix
        
    def rayleigh(self,x,sig,a,b):
        mix = np.zeros((x.shape),dtype=np.float32)
        mix[x-b<=0] = 0
        mix[x-b>0] = a*((x[x-b>0]-b)/sig**2*(np.exp(-(x[x-b>0]-b)**2/(2*sig**2))))
        return mix



class fitCum():
    def __init__(self):
        pass
    
    def weibull_mix_cum(self,x,lam1,lam2,k1,k2,a1):
        mix = np.zeros((x.shape),dtype=np.float32)
        b1 = self.b1
        b2 = self.b2
        maxval = self.maxval
        a2 = maxval-a1
        if (b1>=b2):
            mix[x<=b1] = 0
            mix[x>b1] = a1-a1*np.exp(-((x[x>b1]-b1)/lam1)**k1)
        else:
            mix[x<b1] = 0
            mix[x>=b2] = a1-a1*np.exp(-((x[x>=b2]-b1)/lam1)**k1) + a2-a2*np.exp(-((x[x>=b2]-b2)/lam2)**k2)
            mix[(x>b1)&(x<b2)] = a1-a1*np.exp(-((x[(x>b1)&(x<b2)]-b1)/lam1)**k1)
        return mix
        
    def weibull_log_cum(self,x,lam1,lam2,k1,k2,a1):
        mix = np.zeros((x.shape),dtype=np.float32)
        b1 = self.b1
        b2 = self.b2
        maxval = self.maxval
        a2 = maxval-a1
        if (b1>=b2):
            mix[x<=b1] = 0
            mix[x>b1] = a1-a1*np.exp(-((x[x>b1]-b1)/lam1)**k1)
        else:
            mix[x>=b2] = a1-a1*np.exp(-((x[x>=b2]-b1)/lam1)**k1) + a2-a2*np.exp(-((x[x>=b2]-b2)/lam2)**k2)
            mix[(x>b1)&(x<b2)] = a1-a1*np.exp(-((x[(x>b1)&(x<b2)]-b1)/lam1)**k1)
        return mix
    
    
    
    
    
    
    
    
    
    
    
    