import numpy as np
import scipy.optimize as opt
import utils
import warnings
import time

def freq1_depth_estimation(v, fmod):
    """
    Inputs:
        v: (n_batch,dim_x,dim_y,2)   -> v[b,u,v,0] = Real[v_fmod], v[b,u,v,1] = Imag[v_fmod]
        fmod: Modulation frequency used

    Returns:
        estimated depth map
    """
    
    # Compute phase of the received phasors
    phi = np.arctan2(v[...,1], v[...,0])
    phi = (phi+2*utils.pi()) % (2*utils.pi())  # Map in [0,2pi]
    
    # Compute the corresponding depth
    depth = (phi*utils.c()) / (4*utils.pi()*fmod)
    return depth
    
    


def freq3_sra_backscattering_estimation(v, freqs_list, epsilon=0.1, min_t=0, max_t=utils.max_t(), dim_t=2000):
    """
    Inputs:
        v: (n_batch,dim_x,dim_y,6)   -> v[b,u,v,0]=Real[freqs_list[0]], v[b,u,v,1]=Real[freqs_list[1]], v[b,u,v,2]=Real[freqs_list[2]], 
                                            v[b,u,v,3]=Imag[freqs_list[0]], v[b,u,v,4]=Imag[freqs_list[1]], v[b,u,v,5]=Imag[freqs_list[2]]
        freqs_list: List of the three modulation frequencies used

    Returns:
        estimated backscattering vector
    """
    
    
    # Get input dimensions
    dim_x = v.shape[0]
    dim_y = v.shape[1]
    
    # Construct matrix phi
    step_t = (max_t-min_t) / dim_t
    freqs = np.array(freqs_list, dtype='float32')
    times = np.arange(dim_t, dtype='float32') * step_t
    phi_arg = 2 * utils.pi() * np.matmul(np.reshape(freqs, (-1,1)), np.reshape(times, (1,-1)))
    phi = np.concatenate([np.cos(phi_arg), np.sin(phi_arg)], axis=0)
    
    # Build matrix Q_2M
    MM = 6
    Q = np.ndarray((2**MM,MM), dtype='float32')
    for i in range(2**MM):
        s = '{:0>6}'.format(bin(i)[2:])
        for j in range(MM):
            Q[i,j] = s[j]
    Q = 2*Q - 1
    

    # SRA algorithm
    v_norm = np.sum(np.abs(v), axis=-1)
    c = np.ones((dim_t), dtype='float32')
    A = np.matmul(Q, phi)
    lb = 0
    ub = None
    x = np.ndarray((dim_x,dim_y,dim_t), dtype='float32')
    succ = np.ndarray((dim_x,dim_y), dtype=bool)
    #warnings.filterwarnings("ignore")
 
    for i in range(dim_x):
        start = time.time()
        for j in range(dim_y):
            
            # Get current variable
            v_curr = np.reshape(v[i,j], (-1,1))
            b_curr = np.matmul(Q, v_curr) + epsilon * v_norm[i,j]
            
            # Solve linear programming problem
            #res = opt.linprog(c=c, A_ub=A, b_ub=b_curr, A_eq=None, b_eq=None, bounds=(lb, ub), method='interior-point', options={'sym_pos': False })
            res = opt.linprog(c=c, A_ub=A, b_ub=b_curr, A_eq=None, b_eq=None, bounds=(lb, ub), method='interior-point')
            succ[i,j] = res.success
            x[i,j] = res.x
        print(i)
        ending = time.time()
        print("line time is ", ending-start)
    nings.filterwarnings("default")
        
    return x, succ
