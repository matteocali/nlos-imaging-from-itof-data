import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import sys

    
    
# Plot depth map
def plot_depth(depth_map, min_d=None, max_d=None, ntiks=5, title='Depth map'):
    
    # Compute depth range dynamically
    if min_d is None:
        min_d = 0
    if max_d is None:
        max_d = np.max(depth_map)
    
    plt.imshow(depth_map, cmap='jet')
    plt.clim(min_d, max_d)
    plt.colorbar(ticks=np.linspace(min_d, max_d, ntiks), format='%.1f [m]')
    plt.title(title)
    plt.show()



# Get speed of light [m/s]
def c():
    return 299792458.0


# Get depth and magnitude of the direct component given the predicted itof direct measurements and the corresponding modulation frequencies. 
def x_direct_from_itof_direct(v_d,freqs=np.array((20e06,50e06,60e06),dtype=np.float32)):
    nf = int(v_d.shape[-1]/2)  # number of frequencies
    A = np.sqrt(v_d[...,:nf]**2 + v_d[...,nf:]**2)
    A = np.mean(A,axis=-1)  # average the three predictions
    phi = np.arctan2(v_d[...,nf:],v_d[...,:nf])
    depth = (phi+2*math.pi)%(2*math.pi)
    depth = c()*depth/(freqs*4*math.pi)
    depth = np.min(depth,axis=-1) # minimum operation to choose between the different predictions. Useful in case of MPI


    return A,depth


def amb_range(freq):
    amb_range = c()/(2*freq)
    return amb_range

# Get pi
def pi():
    return math.pi

def step_t():
    return 2*0.00249827/c()

def max_t(dim_t=2000):
    return step_t()*dim_t
    
def max_d(dim_t=2000):
    return 0.5*max_t(dim_t)*c()

def phi(freqs, dim_t=2000, exp_time=0.01):
    pi = math.pi
    # 0.00249827
    min_t = 0
    max_t = 2 * exp_time / c() * dim_t
    step_t = (max_t - min_t) / dim_t
    times = np.arange(dim_t) * step_t
    phi_arg = 2 * pi * np.matmul(freqs.reshape(-1, 1), times.reshape(1, -1))
    phi = np.concatenate([np.cos(phi_arg), np.sin(phi_arg)], axis=0)
    return phi

def phi_10f(dim_t=2000):
    pi = math.pi
    freqs = np.array((10e06,20e06,30e06,40e06,50e06,60e06,70e06,80e06,90e06, 100e06),dtype = 'float32')
    min_t = 0
    max_t = 2*0.00249827/c()*dim_t
    step_t = (max_t-min_t) / dim_t
    times = np.arange(dim_t) * step_t
    phi_arg = 2 * pi * np.matmul(freqs.reshape(-1,1),times.reshape(1,-1))
    phi = np.concatenate([np.cos(phi_arg), np.sin(phi_arg)],axis=0)
    return phi

# Reconstruct depth map from transient matrix
# In case of no noise, the depth corresponds to the first non-zero index of the backscattering vector x.
# In the general case, we add some noise tolerance:
#           d = argmax_i { i : x_i>=M*max_i(x_i) }
# the default value is M=0.01


def tr2depth(x,tol=0.00001,dim_t=2000):
   t_depth = 0.5*c()*step_t()
   depth = t_depth*np.float32((x>tol).argmax(axis=-1))
   return depth


def plot_backscattering(x):
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(x)
    ax.set_yscale('log')
    plt.title('Backscattering vector (log scale)')
    ax.set_ylabel('Intensity')
    ax.set_xlabel('Time')
    plt.show()
    return
