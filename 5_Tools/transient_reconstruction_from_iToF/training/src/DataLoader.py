import numpy as np
import tensorflow as tf
import h5py
import sys
sys.path.append("../utils")
import utils
import matplotlib.pyplot as plt
import math


class DataLoader:
    """ The DataLoader class loads a dataset stored on h5file, at each invocation of method
        DataLoader.next_batch() the next batch is returned
        
        The structure of the h5file f must be the following:
            f['data_shape']   : Data shapes (dim_data, dim_x_out, dim_y_out, P_data, dim_t)
            f['v_in']         : Input phasors vector (without noise)
            f['x_gt']         : Ground truth backscattering vector
            f['depth_gt']     : Ground truth depth
            f['mpi_gt']       : Mask indicating for each pixel the presence of MPI or not
            f['freqs']        : Modulation frequencies used to generate v_in
            f['phi']          : Phasor matrix phi (v_in = phi * x_gt)
            f['z_gt']         : Ground truth in the z domain        
            f['std2']         : Standard deviation of the second component of the backscattering vector

        Note: The value P represents the number of additional pixels present in the input's borders 
        (wrt the output size) along each spatial dimension. If the output size is WxH, the
        corresponding input size is (W+2P)x(H+2P). The output and the input patches are centered,
        therefore the output pixel of coordinate (0,0) corresponds to  the input pixel of coordinate (P,P).
                     __________________
                     |        |         |
                     |        P         |
                     |       _|__       |
                     |<-P-> |    |      |
                     |      |____|      |
                     |                  |
                     |                  |
                     |__________________|
    """
    
    
    
    def __init__(self, filename, freqs, dim_batch=256,  N_batches=None, P=None,fl_scale=True, fl_scale_perpixel=True):
        """ Initialize the data loader class
            
        Inputs:
            filename:           dtype='string'
                                Filename of the h5file containing the dataset
            
            freqs:              dtype='float32', shape=(# of frequencies,)
                                Frequencies used for training. Usually 20, 50 and 60 MHz.
                        
            dim_batch:          dtype='int', shape=(1,)
                                Dimension of each batch to load
            
            N_batches:          dtype='int', shape=(1,)
                                Number of batches to load, if N_batches=None, load all batches from the h5file
                                Note: The dimension of the returned batches is always equals to dim_batch, 
                                eventually discard the last batch of the dataset if |last batch|<dim_batch
            
            P:                  dtype='int', shape=(1,)
                                Custom P value for the output patches, if P=None use P = P_data
                                Note: The custom P value for the output patches must be <= than the P value of the original dataset (P_data)

            fl_scale_perpixel:  dtype='bool', shape=(1,)
                                How to perform the scaling. If set to 'True', the scaling is done by the amplitude at 20MHz pixel per pixel, 
                                otherwise the mean amplitude at 20 MHz is used for each patch. 

            fl_log_scale:       dtype='bool', shape=(1,)
                                Whether to apply a logarithmic scaling to the input patches according to their 20 amplitude mean
            
        """
        
        self.filename = filename
        self.b = 0
        self.f = h5py.File(self.filename, "r")
        self.fl_scale = fl_scale                   # whether to perform any scaling on the inputs
        self.fl_scale_perpixel = fl_scale_perpixel # whether to normalize the v vectors pixel per pixel or for the mean inside the window
        self.fl_load_transient = True             # Flag for loading transient data. Automatically set to 1 if we have a key called 'transient'
        # Load characteristics of dataset
        temp_v_in = self.f["raw_itof"]
        data_shape = temp_v_in.shape
        self.freqs = freqs
        self.n_fr = freqs.shape[0]  # number of frequencies
        
        
        # Get dimensions of the portion of dataset to load
        self.dim_batch = dim_batch
        if N_batches is None:
            self.N_batches = int(np.floor(data_shape[0] / self.dim_batch))
        else:
            self.N_batches = N_batches
        self.P = P

        
        

        # Initialize the iterator (and close file)
        self.init_iter()
    
    
    
    def init_iter(self):
        """ Set batch iterator at the beginning of the h5file and close the file
        """
        self.b = 0
        self.f.close()


    def add_shot_noise(self,dict_data):

        shot_scale = self.shot_scale
        freqs = np.array((20e06,50e06,60e06),dtype=np.float32)
        
        tr = dict_data["transient"]
        v = dict_data["raw_itof"]

        if v.shape[:-1] != tr.shape[:-1]:
            print("Impossible to add shot noise. The input transient does not match the input iToF shape!")
            sys.exit()
        dim_t = tr.shape[-1]
        imid = int((v.shape[1]-1)/2)
        



        pi = math.pi
        min_t = 0
        max_t = 2*0.00249827/utils.c()*dim_t
        step_t = 2*0.00249827/utils.c()
        times = np.arange(dim_t) * step_t

        # Compute the raw measurements with 4 different internal phase delays (0,90,180 and 270 degrees)
        phi_arg0 =  2 * pi * tf.matmul(freqs.reshape(-1,1),times.reshape(1,-1))
        phi_arg90 =  2 * pi * tf.matmul(freqs.reshape(-1,1),times.reshape(1,-1)) +  pi/2
        phi_arg180 =  2 * pi * tf.matmul(freqs.reshape(-1,1),times.reshape(1,-1)) + pi
        phi_arg270 =  2 * pi * tf.matmul(freqs.reshape(-1,1),times.reshape(1,-1)) + 3/2*pi
        phi0 = tf.cos(phi_arg0)
        phi90 = tf.cos(phi_arg90)
        phi180 = tf.cos(phi_arg180)
        phi270 = tf.cos(phi_arg270)
        phi0 = 0.5 + 0.5*(phi0)
        phi90 = 0.5 + 0.5*(phi90)
        phi180 = 0.5 + 0.5*(phi180)
        phi270 = 0.5 + 0.5*(phi270)
        phi0 = tf.transpose(phi0)
        phi90 = tf.transpose(phi90)
        phi180 = tf.transpose(phi180)
        phi270 = tf.transpose(phi270)
        raw_0 = tr@phi0
        raw_90 = tr@phi90
        raw_180 = tr@phi180
        raw_270 = tr@phi270

        # scaling factor to put the max to 0
        m0 = tf.math.reduce_max(raw_0)
        m90 = tf.math.reduce_max(raw_90)
        m180 = tf.math.reduce_max(raw_180)
        m270 = tf.math.reduce_max(raw_270)
        s_fact = tf.math.reduce_max([m0,m90,m180,m270])

        raw_0 = raw_0/s_fact*shot_scale
        raw_90 = raw_90/s_fact*shot_scale
        raw_180 = raw_180/s_fact*shot_scale
        raw_270 = raw_270/s_fact*shot_scale


        raw_0 = tf.random.normal(raw_0.shape,raw_0,tf.math.sqrt(raw_0)) 
        raw_90 = tf.random.normal(raw_90.shape,raw_90,tf.math.sqrt(raw_90)) 
        raw_180 = tf.random.normal(raw_180.shape,raw_180,tf.math.sqrt(raw_180)) 
        raw_270 = tf.random.normal(raw_270.shape,raw_270,tf.math.sqrt(raw_270)) 

    
        raw_0 = raw_0/shot_scale*s_fact
        raw_90 = raw_90/shot_scale*s_fact
        raw_180 = raw_180/shot_scale*s_fact
        raw_270 = raw_270/shot_scale*s_fact


        # Recompute the corresponding phasor values
        v_1 = raw_0-raw_180
        v_2 = raw_270-raw_90
        
        v_shot = tf.concat([v_1,v_2],axis=-1)
        tr = tr[:,imid,imid,:]
        v_noshot = v
        dict_data["raw_itof"] = v_shot
        dict_data["raw_itof_noshot"] = v_noshot
        dict_data["transient"] = tr
        dict_data["transient_global"] = dict_data["transient_global"][:,imid,imid,:]

        return dict_data

    
    
    def next_batch(self):
        """ Returns a dictionary containing the next batch, 
            if the dataset has reached its end returns None and resets 

        Returns:
            dict_data:  Dictionary containing all the needed training information linked to the dataset.
                        Note: The data contained inside the dictionary depends on the dataset. Some datasets have transient information, others don't
                        The transient data can is recognised if the input dataset has an entry called 'transient'

                        KEYS:
                        1) General
                            -'raw_itof': raw itof measurements
                            -'direct_itof': itof measurements corresponding to the direct component
                            -'global_itof': itof measurements corresponding to the global component
                            -'phase'
                            -'v_scale': scaling factors used to scale the input data ('raw_itof,'direct_itof' and 'global_itof'). 
                                        Either one value per pixel or one per patch, according to the value of 'fl_scale_perpixel'
                            -'phase_raw': phase measurements corresponding to the raw itof measurements (input phase with MPI and possibly other noise sources added)
                            -'phase_direct': phase measurements corresponding to the direct component (ground truth used for MPI denoising)
                            -'amplitude_raw': amplitude corresponding to the raw itof measurements
                            


                        2) Only for transient datasets
                            -'phase_global': phase measurements corresponding to the global component (phase corresponding to the noise. Not used)
                            -'amplitude_direct': amplitude corresponding to the direct itof measurements (Not used)
                            -'amplitude_global': amplitude corresponding to the global itof measurements (Not used)
                            -'transient': transient information. In order to keep in check the memory consumption, it is only available for the central pixel
                            -'transient_global': transient information without the peak of the direct component. In order to keep in check the memory consumption, it is only available for the central pixel


                        NOTE: Not all keys appear in the code below. The reason is that some of them do not need to be processed and just go through the for cycle. 
                              Moreover, not all of them may be needed for training (as the amplitude ones for example)
                              They are all built during the data loading procedure.


        """
        # Open h5file
        if self.b==0:
            self.f = h5py.File(self.filename, 'r')
            

        # Iterate over all the batches
        if self.b < self.N_batches:     

            if "transient" in self.f.keys():
                self.fl_load_transient = True
            # Compute sample indices corresponding to the current batch
            i_init = self.b * self.dim_batch
            i_end = (self.b+1) * self.dim_batch
            # Load next batch


            dict_data = {}
            for key in self.f.keys():
                if key == "name":
                    dict_data["name"] = self.f["name"][()]
                else:
                    dict_data[key] = tf.convert_to_tensor(self.f[key][i_init:i_end,...],dtype="float32")



            # Compute the scaling factor (based on the amplitude at 20 MHz). Per pixel or per patch
            if self.fl_scale:
                v_in = dict_data["raw_itof"]
                v_a = tf.math.sqrt(tf.math.square( v_in[...,0] ) + tf.math.square( v_in[...,self.n_fr] )) 
                if self.fl_scale_perpixel:
                    v_a_max = tf.expand_dims(v_a,axis=-1)
                else:
                    v_a = tf.math.reduce_mean(v_a,axis=[1,2])
                    v_a_max = tf.expand_dims(v_a,axis=-1)
                    v_a_max = tf.expand_dims(v_a_max,axis=-1)
                    v_a_max = tf.expand_dims(v_a_max,axis=-1)
                    v_a_max = tf.where(v_a_max==0,1.,v_a_max)
            
                # Scale all factors
                dict_data["raw_itof"]/=v_a_max
                if self.fl_load_transient:
                    dict_data["direct_itof"]/=v_a_max
                    dict_data["global_itof"]/=v_a_max
                    dict_data["v_scale"] = v_a_max     #Save the scaling factor
                    # Needed for tackling both scaling cases. Recall that the transient is only for the central pixel, not the whole patch 
                    if v_a_max.shape[1]>1:
                        mid = int((self.P-1)/2)
                        v_a_max = v_a_max[:,mid,mid]
                    v_a_max = tf.squeeze(v_a_max)
                    v_a_max = tf.expand_dims(v_a_max,axis=-1)
                    dict_data["transient"]/=v_a_max
                    dict_data["transient_global"]/=v_a_max
                    


            else:
                dict_data["v_scale"]=None
           
            self.b+=1
            # Return next batch
            return dict_data   #return also the coefficients of normalization so that I can use them later
        
        else:   # If the file is finished...
            self.init_iter()
            return  None
