import numpy as np
import tensorflow as tf
import h5py
import sys
sys.path.append("../utils")


class DataLoader:
    """ The DataLoader class loads a dataset stored on h5 file, at each invocation of method
        DataLoader.next_batch() the next batch is returned
        
        The structure of the h5 file f must be the following:
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
                     ____________________
                     |         |        |
                     |         P        |
                     |       __|__      |
                     |<-P-> |    |      |
                     |      |____|      |
                     |                  |
                     |                  |
                     |__________________|
    """

    def __init__(self, filename, freqs, dim_batch=256, N_batches=None, P=None, fl_scale=True):
        """
        Initialize the data loader class
        Inputs:
            filename:           dtype='string'
                                Filename of the h5 file containing the dataset

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

            fl_log_scale:       dtype='bool', shape=(1,)
                                Whether to apply a logarithmic scaling to the input patches according to their 20 amplitude mean
            
        """

        # Flags and initial parameters
        self.filename = filename
        self.b = 0
        self.f = h5py.File(self.filename, "r")
        self.fl_scale = fl_scale  # whether to perform any scaling on the inputs

        # Load characteristics of dataset
        temp_v_in = self.f["raw_itof"]
        data_shape = temp_v_in.shape
        self.freqs = freqs
        self.n_fr = freqs.shape[0]  # number of frequencies

        # Get the dimensions of the portion of dataset to load
        self.dim_batch = dim_batch
        if N_batches is None:
            self.N_batches = int(np.floor(data_shape[0] / self.dim_batch))
        else:
            self.N_batches = N_batches
        self.P = P

        # Initialize the iterator (and close file)
        self.init_iter()
    
    
    def init_iter(self):
        """
        Set batch iterator at the beginning of the h5 file and close the file
        """

        self.b = 0
        self.f.close()


    def next_batch(self):
        """
        Returns a dictionary containing the next batch, if the dataset has reached its end returns None and resets

        Returns:
            dict_data:  Dictionary containing all the needed training information linked to the dataset.
                        Note: The data contained inside the dictionary depends on the dataset. Some datasets have transient information, others don't
                        The transient data is recognised if the input dataset has an entry called 'transient'

                        KEYS:
                        1) General
                            -'raw_itof': raw itof measurements
                            -'direct_itof': itof measurements corresponding to the direct component
                            -'global_itof': itof measurements corresponding to the global component
                            -'v_scale': scaling factors used to scale the input data ('raw_itof,'direct_itof' and 'global_itof'). 
                                        Either one value per pixel or one per patch, according to the value of 'fl_scale_perpixel'
                            -'phase_raw': phase measurements corresponding to the raw itof measurements (input phase with MPI and possibly other noise sources added)
                            -'phase_direct': phase measurements corresponding to the direct component (ground truth used for MPI denoising)
                            -'amplitude_raw': amplitude corresponding to the raw itof measurements
                            -'gt': ground truth depth

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

        # Open h5 file
        if self.b == 0:
            self.f = h5py.File(self.filename, 'r')

        # Iterate over all the batches
        if self.b < self.N_batches:
            # Compute sample indices corresponding to the current batch
            i_init = self.b * self.dim_batch
            i_end = (self.b+1) * self.dim_batch
            non_zeros_pos = np.empty((1))

            # Load next batch
            dict_data = {}
            for key in self.f.keys():
                if key == "name":
                    dict_data["name"] = self.f["name"][()]
                else:
                    dict_data[key] = tf.convert_to_tensor(self.f[key][i_init:i_end, ...], dtype="float32")

            # If necessary remove all the wrong patches (i.e. with no valid pixels = 0)
            if tf.where(dict_data["raw_itof"] == 0).shape[0] > 0:
                zero_pos = np.where(dict_data["raw_itof"] == 0)
                zero_pos = np.squeeze(zero_pos[0])
                zero_pos = np.unique(zero_pos)
                index_to_gather = np.ones((dict_data["raw_itof"].shape[0]), dtype=bool)
                index_to_gather[zero_pos] = False
                index_to_gather = np.where(index_to_gather)[0]
                dict_data["raw_itof"] = tf.gather(dict_data["raw_itof"], index_to_gather)
                dict_data["gt_alpha"] = tf.gather(dict_data["gt_alpha"], index_to_gather)
                dict_data["gt_depth"] = tf.gather(dict_data["gt_depth"], index_to_gather)

            # Compute the scaling factor (based on the amplitude at 20 MHz). Per pixel or per patch
            if self.fl_scale:
                v_in = dict_data["raw_itof"]
                v_a = tf.math.sqrt(tf.math.square(v_in[..., 0]) + tf.math.square(v_in[..., self.n_fr]))
                v_a = tf.expand_dims(v_a, axis=-1)
                # Scale the iToF raw data
                dict_data["raw_itof"] /= v_a

            self.b += 1

            # Return next batch
            return dict_data  # return also the coefficients of normalization so that I can use them later
        
        else:   # If the file is finished...
            self.init_iter()
            return None
