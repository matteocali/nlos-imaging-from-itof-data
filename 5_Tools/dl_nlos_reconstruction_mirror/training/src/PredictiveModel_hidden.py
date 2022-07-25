import tensorflow as tf
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import math
import tensorflow.keras.layers as layers
import os
import sys
sys.path.append("../../utils/")
sys.path.append("../")
import utils
#import GenerativeModel
import Autoencoder_sameconv as Autoencoder_Interp
import time
from tensorflow.keras import optimizers


class PredictiveModel:
    
    # Initialize the network
    def __init__(self, name, dim_b, freqs, P, saves_path, use_S1=True, use_walls=True, use_transient=True, dim_t=2000, fil_size=8,fil_denoise_size=32,fil_z_size=32, dim_encoding = 12,fil_encoder=32):
        """ Initialize the Predictive model class
            
        Inputs:
            name:               dtype='string'
                                Name of the predictive model
            
            dim_b:              dtype='int', shape=(1,)
                                Batch dimension

            freqs:              dtype='float32', shape=(# of frequencies,)
                                Modulation frequencies of the input itof data
            
            saves_path:         dtype='string'
                                Path to the directory where to save checkpoints and logs

            use_S1:             dtype='bool'
                                Whether to train with the S1 dataset 

            use_walls:          dtype='bool'
                                Whether to train on transi data 

            use_transient:      dtype='bool'
                                Whether to train on transient data or not (loading transient data makes the training slower)

            dim_t:              dtype='int'
                                Number of bins in the transient dimension

            fil_denoise_size:   dtype='int'
                                Number of feature maps for the Spatial Feature Extractor

            fil_size:           dtype='int'
                                Number of feature maps for the Direct CNN

            fil_z_size:         dtype='int'
                                Number of feature maps for the Transient Reconstruction Module

        """
         
        # Initializing the flags and other input parameters
        self.fl_old_transient  = False             # Used to work with the older version of the network
        self.use_S1 = use_S1
        self.use_walls = use_walls
        self.use_transient = use_transient

        self.name = name
        self.dim_b = dim_b
        self.dim_t = dim_t
        self.dim_encoding = dim_encoding
        self.fil_encoder = fil_encoder
        self.P = P
        self.ex = int((self.P-1)/2) # Index keeping track of the middle of the patch
        self.fn = freqs.shape[0]  # Number of frequencies
        self.fn2 = self.fn*2      # Number of raw measurements (twice the number of frequencies)
        self.fl_2freq = (self.fn==2) # Whether we are training with 2 frequencies or not
        self.num_fil_denoise = fil_denoise_size # Number of filters for each of the convolutional layers of the Spatial Feature Extractor
        self.fil_pred = fil_size   # Number of filter for the Direct CNN
        self.fil_z_size = fil_z_size   # Number of filter for the transient reconstruction network

        #Defining all parameters needed by the denoising network
        self.in_shape = (P,P,self.fn2)    # Shape of the input (batch size excluded)
        self.out_win = 3                          # Side of the window provided in output of the Spatial Feature Extractor
        self.padz = self.P-self.out_win+2         # Padding needed 
        self.fl_denoise = not (P==3)              # Whether to use the Spatial Feature Extractor
        self.k_size = 3                           # Kenel size for each layer of the denoiser network
        self.f_skip = True                        # Whether to use a skip connection or not

        self.freqs = tf.convert_to_tensor(freqs , dtype = "float32"  )  
        
        # Create saves directory if it does not exist
        if not os.path.exists(saves_path):
            os.makedirs(saves_path)
            
        # Create current network directory, if it does not exist
        self.net_path = os.path.join(saves_path, self.name)
        if not os.path.exists(self.net_path):
            os.makedirs(self.net_path)
            
        # Create log directory, if it does not exist
        self.log_path = os.path.join(self.net_path, 'logs')
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.log_path_train = os.path.join(self.log_path, 'train')
        if not os.path.exists(self.log_path_train):
            os.makedirs(self.log_path_train)
        self.log_path_validation = os.path.join(self.log_path, 'validation')
        if not os.path.exists(self.log_path_validation):
            os.makedirs(self.log_path_validation)

        # Create checkpoints path, if it does not exist
        self.checkpoint_path = os.path.join(self.net_path, 'checkpoints')
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
            
        # Define autoencoder model with transposed convolutions
        self.Autoencoder = Autoencoder_Interp.Autoencoder_Interp(self.dim_b,self.dim_t,self.dim_encoding,self.fil_encoder)
        self.encoder = self.Autoencoder.encoder()
        self.decoder = self.Autoencoder.interpConv()

        self.predv_encoding = self.def_Encoding()      # Small network sending the output of the global direct subdivision to the input of the decoder


        # Define autoencoder with interpolation
        #self.Autoencoder = Autoencoder.Autoencoder(self.dim_b,self.dim_t,self.fil_encoder)
        #self.encoder = self.Autoencoder.encoder()
        #self.decoder = self.Autoencoder.decoder()

        # Define generative model
        #self.gen_model = GenerativeModel.GenerativeModel(self.dim_b, self.P-self.padz, self.P-self.padz, self.dim_t)
        
        # Define predictive models
        in_shape=(self.P,self.P,self.fn*2)
        self.SpatialNet = self.def_SpatialNet()
        self.DirectCNN = self.def_DirectCNN()
        self.TransientNet = self.def_TransientNet()
        
        
        # Define loss function and metrics
        self.loss_fn = self.def_loss
        
        # Define optimizer
        self.lr = 1e-05
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        
        # Track and save the best performing model over the testset at each epoach
        self.best_loss_test = -1
        self.best_epoach = -1
        self.c = 299792458.  # speed of light
    
    
    # Load weights
    def load_weights(self, weight_filenamed=None,weight_filenamev=None,weight_filenamez=None):
        if weight_filenamed is not None:
            self.SpatialNet.load_weights(weight_filenamed, by_name=True)
        if weight_filenamev is not None:
            self.DirectCNN.load_weights(weight_filenamev, by_name=True)
        if weight_filenamez is not None:
            self.TransientNet.load_weights(weight_filenamez, by_name=True)
    
    
    
    # Save weights
    def save_weights(self, suffix):
        weight_filenamed = self.name + '_d_' + suffix + '.h5'
        weight_filenamev = self.name + '_v_' + suffix + '.h5'
        weight_filename_enc = self.name + '_enc_' + suffix + '.h5'
        weight_filename_dec = self.name + '_dec_' + suffix + '.h5'
        weight_filename_predv_enc = self.name + '_predv_enc_' + suffix + '.h5'
        self.SpatialNet.save_weights(os.path.join(self.checkpoint_path, weight_filenamed))
        self.DirectCNN.save_weights(os.path.join(self.checkpoint_path, weight_filenamev))
        self.encoder.save_weights(os.path.join(self.checkpoint_path, weight_filename_enc))
        self.decoder.save_weights(os.path.join(self.checkpoint_path, weight_filename_dec))
        self.predv_encoding.save_weights(os.path.join(self.checkpoint_path, weight_filename_predv_enc))
        if self.fl_old_transient:
            weight_filenamez = self.name + '_z_' + suffix + '.h5'
            self.TransientNet.save_weights(os.path.join(self.checkpoint_path, weight_filenamez))
   
    # Compute the amplitude of the phasors given the raw measurements
    def A_compute(self,v):
        A = tf.math.sqrt(tf.math.square(v[...,:self.fn])+tf.math.square(v[...,self.fn:]))
        return A


    # Returns the ambiguity range values at the three different frequencies
    def ambiguity_compute(self):
        return self.c/(2*self.freqs)

    
    def def_SpatialNet(self):

        # Get the input parameters
        in_shape = self.in_shape
        out_win = self.out_win
        b_size = self.dim_b
        num_fil = self.num_fil_denoise
        k_size = self.k_size
        f_skip = self.f_skip

        # Compute other useful values given the input parameters
        in_win = self.P
        if not self.fl_denoise:     # Just needed to avoid errors when the model is not used (if the two...
            in_win = in_win+4     # ... windows are equal, then it is skipped in training
            in_shape = list(in_shape)
            in_shape[0] = in_shape[0]+4
            in_shape[1] = in_shape[1]+4
            in_shape = tuple(in_shape)
        pad = int((in_win-out_win)/2)

        n_layers = int((in_win-out_win)/(self.k_size-1))-2   # Number of layers needed to change the shape from input to output
        v_in = tf.keras.Input(shape=(None,None,self.fn2),batch_size=b_size,name="v_in_shot",dtype="float32")
        v_in_wxw = tf.strided_slice(v_in,begin=[0,pad,pad,0],end=[-1,-pad,-pad,-1],end_mask=1001)  # Needed for skip connnection 

        # Convolutional layers
        out = layers.Conv2D(filters=num_fil,
                             kernel_size=k_size,
                             activation="relu",
                             name="layer_in")(v_in)
        for i in range(n_layers):
            lname = "conv" + str(i)
            out = layers.Conv2D(filters=num_fil,
                                 kernel_size=k_size,
                                 activation="relu",
                                 name=lname)(out)


        # Last layer must have the same number of filters of the input
        out=layers.Conv2D(filters=in_shape[-1],
                           kernel_size=k_size,
                           activation=None,
                           name="layer_out")(out)

        # Skip connection
        if f_skip:
            out+=v_in_wxw
            
    
        model = tf.keras.Model(inputs=v_in, outputs=out)
        return model



    def def_DirectCNN(self):
        # Define the input placeholder
        v_in = tf.keras.Input(shape=(None, None, self.fn2), batch_size=None, 
                              dtype='float32', name='v_in')
        ind_ = int((self.out_win-1)/2)  # Index keeping track of the middle value
        v_in_1x1 = tf.strided_slice(v_in,begin=[0,ind_,ind_,0],end=[-1,-ind_,-ind_,-1],end_mask=1001)  

        # Two branches, one processing a 3x3 patch around the central pixel, and the other focusing only on the central pixel itself
        out_1x1 = layers.Conv2D(filters=self.fil_pred,
                                kernel_size=1,                # pixel-level (1x1) features extraction
                                strides=1,
                                padding="valid",
                                data_format='channels_last',
                                activation='relu',
                                use_bias=True,
                                trainable = True,
                                name='c_1x1')(v_in_1x1)
        out_3x3 = layers.Conv2D(filters=self.fil_pred,
                                kernel_size=self.out_win,                # local-level (3x3) features extraction
                                strides=1,
                                padding="valid",
                                data_format='channels_last',
                                activation='relu',
                                use_bias=True,
                                trainable = True,
                                name='c_3x3')(v_in)
        c_out = tf.concat([out_1x1, out_3x3], axis=-1, name='cat1') # Features concatenation
        #c_out = out_1x1
        
        #Convolutional layers leading to the prediction of the direct itof measurements
        c_out = layers.Conv2D(filters=self.fil_pred,
                           kernel_size=1,
                           strides=1,
                           padding="valid",
                           data_format='channels_last',
                           activation='relu',
                           use_bias=True,
                           trainable = True,
                           name='cd1')(c_out)

        v_res = layers.Conv2D(filters=2*self.fn2, # =12 va modificato 2*depth_map*alphamap
                           kernel_size=1,
                           strides=1,
                           padding="valid",
                           data_format='channels_last',
                           activation=None,
                           use_bias=True,
                           trainable = True,
                           name='cd2')(c_out)

        v_res_constr = tf.slice(v_res,begin=[0,0,0,0],size=[-1,-1,-1,self.fn2])  ' sei elementi ciascuno ne voglio solo uno a testo'
        v_res_free = tf.slice(v_res,begin=[0,0,0,self.fn2],size=[-1,-1,-1,self.fn2])

        v_out_d = v_in_1x1 + v_res_constr  # questo non va bene perche' non uso i residui, uso dirertaam4nte 'out
        v_out_g = v_in_1x1 - v_out_d  # non serve piu'

        # Phase corresponding to the predicted direct itof measurements
        phi = tf.math.atan2(v_out_d[:,:,:,self.fn:],v_out_d[:,:,:,:self.fn])
        phi = tf.where(tf.math.is_nan(phi),0.,phi)   # Needed for the first epochs to correct the output in case of a 0/0
        phi = phi%(2*math.pi) # tolgo tutte le phi

        model_pred = tf.keras.Model(inputs=v_in, outputs=[v_out_g,v_out_d,v_res_free], name=self.name)
        return model_pred


    def def_TransientNet(self):

        pad = int(self.P-self.out_win)+2
        v_in_g = tf.keras.Input(shape=(self.P-pad, self.P-pad, self.fn2), batch_size=self.dim_b, 
                              dtype='float32', name='v_in_g')
        v_in_d = tf.keras.Input(shape=(self.P-pad, self.P-pad, self.fn2), batch_size=self.dim_b, 
                              dtype='float32', name='v_in_d')
        v_both = tf.concat([v_in_g,v_in_d],axis=-1)  # Stack of the two inputs


        # Computation of the offset. The global cannot be predicted before the direct
        phid = tf.math.atan2(v_in_d[...,self.fn:],v_in_d[...,:self.fn])
        phid = (phid+2*math.pi)%(2*math.pi)
        posd = phid*utils.c()/(self.freqs*4*math.pi)
        posd = posd/utils.max_d()  # put between 0 and 1

        # Branch specialized in lam prediction
        lam_out = layers.Conv2D(filters=self.fil_z_size,
                              kernel_size=1,
                              strides=1,
                              padding="valid",
                              data_format='channels_last',
                              activation='relu',
                              use_bias=True,
                              trainable = True,
                              name='lam1')(v_in_g)
        
        
        lam_out = layers.Conv2D(filters=1,
                                 kernel_size=1,
                                 strides=1,
                                 padding="valid",
                                 data_format='channels_last',
                                 activation='relu',
                                 use_bias=False,
                                 trainable = True,
                                 name='lam2')(lam_out)
           
        # Branch specialized in k prediction
        k_out = layers.Conv2D(filters=self.fil_z_size,
                              kernel_size=1,
                              strides=1,
                              padding="valid",
                              data_format='channels_last',
                              activation='relu',
                              use_bias=True,
                              trainable = True,
                              name='k1')(v_in_g)
        
        
        k_out = layers.Conv2D(filters=1,
                                 kernel_size=1,
                                 strides=1,
                                 padding="valid",
                                 data_format='channels_last',
                                 activation='relu',
                                 use_bias=False,
                                 trainable = True,
                                 name='k2')(k_out)

        # Branch specialized in amplitude (a values) prediction
        a_out = layers.Conv2D(filters=self.fil_z_size,
                               kernel_size=1,
                              strides=1,
                              padding="valid",
                              data_format='channels_last',
                              activation='relu',
                              use_bias=True,
                              trainable = True,
                              name='a1')(v_both)
        
        
        a_out = layers.Conv2D(filters=1,
                                 kernel_size=1,
                                 strides=1,
                                 data_format='channels_last',
                                 activation='relu',
                                 use_bias=True,
                                 trainable = True,
                                 name='a2')(a_out)

        # Branch specialized in time (b values) position prediction
        b_out = layers.Conv2D(filters=self.fil_z_size,
                                 kernel_size=1,
                                 strides=1,
                                 padding="valid",
                                 data_format='channels_last',
                                 activation='relu',
                                 use_bias=True,
                                 trainable = True,
                              name='b1')(v_in_g)
        
        
        b_out = layers.Conv2D(filters=1,
                                 kernel_size=1,
                                 strides=1,
                                 padding="valid",
                                 data_format='channels_last',
                                 activation='relu',
                                 use_bias=True,
                                 trainable = True,
                                 name='b2')(b_out)

        b_out = b_out + posd  # b_out is the offset w.r.t. the position of the direct component
        
        lam_out = lam_out + 0.1   # The +0.1 ensures that the Weibull function does not go into dangerous range (given a random initialization, some values could as well be nan)
        k_out = k_out + 0.1   

        # Concatenation of the predicted values to form the latent variable z
        z_out = tf.keras.layers.concatenate([tf.slice(lam_out, begin=[0,0,0,0], size=[-1,-1,-1,1]),
                                             tf.slice(k_out, begin=[0,0,0,0], size=[-1,-1,-1,1]),
                                             tf.slice(a_out, begin=[0,0,0,0], size=[-1,-1,-1,1]),
                                             tf.slice(b_out, begin=[0,0,0,0], size=[-1,-1,-1,1])]
                                                      , name='z')
        


        # Define predictive model
        model_pred = tf.keras.Model(inputs=[v_in_g,v_in_d], outputs=z_out, name=self.name)

        return model_pred

    def def_Encoding(self):
        v_in = tf.keras.Input(shape=(2*self.fn2), batch_size=None, 
                              dtype='float32')

        v_dense = layers.Dense(units = 64, activation = "relu")(v_in)
        
        v_out = layers.Dense(units = self.fn2)(v_dense)


        model_pred = tf.keras.Model(inputs=v_in, outputs=v_out, name=self.name)


        return model_pred
    
    # Define custom loss function
    def def_loss(self, data_dict,  epoch):
        """NOTE:
           - time ranges are all between 0 and 5
           - all transient vectors are normalized w.r.t. the max of the cumulative distribution of the noise
           - all v values are normalized to 1, thus the need for the mutliplicative costants
        """
        # Choose what kind of loss and networks to use according to the dataset we are using. 
        # 'name'=1 corresponds to the transient dataset, 'name=2' to the S1 dataset and 'name'=3 to the S3 real dataset
        if data_dict["name"]  == 1:
            loss, loss_list = self.loss_walls(data_dict,epoch) # Uso solo questa loss
        elif data_dict["name"] == 2:
            loss,loss_list = self.loss_S1(data_dict,epoch)
        elif data_dict["name"] == 3:
            loss,loss_list = self.loss_S3(data_dict,epoch)
        return loss, loss_list


    # Loss computed on the transient dataset
    def loss_walls(self, data_dict, epoch):
        # Load the needed data
        v_in = data_dict["raw_itof"]
        v_d = data_dict["direct_itof"]
        v_g = data_dict["global_itof"]
        # VA CARICATO IL PIXEL CENTRALE DELLA DEPTHMAP
        scale = data_dict["v_scale"]
        i_mid = int((self.P-1)/2) # index keeping track of the middle position


        

        # Compute the phase ground truth from the direct itof measurements (could also be loaded of course)
        phase_gt = tf.math.atan2(v_d[...,self.fn:],v_d[...,:self.fn])
        phase_gt = phase_gt%(2*math.pi)
        # Compute transport light model matrix
        phi = tf.cast(tf.transpose(utils.phi(np.array((20e06,50e06,60e06),dtype=np.float32),2000)),"float32")
        if self.use_transient:
            x_g = data_dict["transient_global"]
            x = data_dict["transient"]
            sum_xg = np.sum(x_g,axis=-1)
            sum_x = np.sum(x,axis=-1)
            ratio_gd = sum_xg/(sum_x-sum_xg)  # Ratio between global and direct components. Used to give weights to the EMD function
            #x_g = tf.math.cumsum(x_g,axis=-1) # Compute the cumulative
            self.dim_t = x_g.shape[-1]        # Make sure the number of bins is correct
        # Use the Spatial Feature Extractor to produce an intermediate result
        if self.fl_denoise:
            v_nf = self.SpatialNet(v_in) 
        else:
            v_nf = v_in
        # Process the output with the Direct CNN
        v_out_g,v_out_d,v_free = self.DirectCNN(v_nf) # nel caso della depth non mi serve v_free
        # If desired, compute also the transient prediction
        if self.use_transient: # potrebbe essere false
            v_in_1x1 = v_in[:,self.ex,self.ex,:]
            v_in_1x1 = tf.expand_dims(v_in_1x1,1)
            v_in_1x1 = tf.expand_dims(v_in_1x1,1)
            x_g = tf.expand_dims(x_g,axis=-1)

            # Encoder part of Autoencoder
            v_encoding = self.encoder(x_g)

            # Add to the encoding the vg ground truth to complete the hidden space
            v_encoding = tf.squeeze(v_encoding)
            v_encoding = tf.concat([v_encoding,v_g[:,self.ex,self.ex,:]],axis=1)

            # Decoder part of Autoencoder
            x_auto = self.decoder(v_encoding)
            x_auto = tf.squeeze(x_auto)

            x_g = tf.squeeze(x_g)
            
            # If needed, add some zeroes in front of the output to get the shapes to matchwith the input
            shape_diff = x_g.shape[-1] - x_auto.shape[-1]
            missing = tf.zeros([x_auto.shape[0],shape_diff],dtype=tf.float32)
            x_auto = tf.concat([missing,x_auto],axis=-1)
            
            # Get the hidden space vector as predicted by the DirectCNN (concatenation of free parameters and v_out_g)
            v_out_g = tf.squeeze(v_out_g)
            v_out_d = tf.squeeze(v_out_d)
            v_free = tf.squeeze(v_free)
            v_out_encoding = tf.concat([v_free,v_out_g],axis=1)

            # Output of DirectCNN through Autoencoder
            x_CNN = self.decoder(tf.squeeze(v_out_encoding))
            x_CNN = tf.squeeze(x_CNN)

            # Fix the output shape as before, by adding zeroes if needed
            x_CNN = tf.concat([missing,x_CNN],axis=-1)

            # Loss not needed for this implementation
            loss_rec = 0

            # Consistency loss between output of the encoder and of the DirectCNN
            v_encoding = tf.squeeze(v_encoding)
            loss_consistency = tf.math.reduce_mean(tf.math.abs(v_encoding-tf.squeeze(v_out_encoding)))

            # Constraint on the total variance of the vector to avoid spikes on the autoencoder output.
            # This is performed on both configurations: Encoder-Decoder and DirectCNN-Decoder
            loss_tvm1 = tf.math.reduce_mean(tf.math.abs(x_auto[1:]-x_auto[:-1]))
            loss_tvm2 = tf.math.reduce_mean(tf.math.abs(x_CNN[1:]-x_CNN[:-1]))
            loss_tvm = loss_tvm2

            # Compute loss encoding 
            v_g = v_g[:,self.ex,self.ex,:]
            loss_encoding = tf.math.reduce_mean(tf.math.abs(v_encoding-v_out_encoding))

        else: 
            loss_emd = 0
            loss_emd1 = 0
            loss_emd2 = 0
            loss_consistency = 0
            loss_tvm = 0

        # Compute the loss between true and predicted direct measurements (Loss in the iToF domain)
        # It is computed on the output of the DirectCNN network
        v_d = v_d[:,i_mid,i_mid,:]
        loss_vd = tf.math.abs(tf.squeeze(v_out_d)-v_d)
        loss_vd = tf.math.reduce_mean(loss_vd)
        
        # MAE losses on the prediction of the Encoder-Decoder and DirectCNN-Decoder configurations.
        # The MAE is computed both on the pdf and the cdf
        # The MAE loss on the pdf here computed is not used in the current implementation. The actual pdf loss is defined below.

        # Le 4 los successive vanno tolte se non uso la seconda rete mettere un if con else che le pone a zero
        loss_mae_auto = tf.math.reduce_mean(tf.math.abs(x_auto-x_g))
        loss_mae_CNN = tf.math.reduce_mean(tf.math.abs(x_CNN-x_g))
        loss_cum_auto = tf.math.reduce_mean(tf.math.abs(tf.math.cumsum(x_auto,axis=-1)-tf.math.cumsum(x_g,axis=-1)))
        loss_cum_CNN = tf.math.reduce_mean(tf.math.abs(tf.math.cumsum(x_CNN,axis=-1)-tf.math.cumsum(x_g,axis=-1)))

        # Complessive loss where the pdf loss (scaled by its mean) and the pdf loss are put together.
        loss_emd1 = self.MAE_loss_2(x_g,x_auto) + 10*loss_cum_auto
        loss_emd2 = self.MAE_loss_2(x_g,x_CNN) + 10*loss_cum_CNN 

        ######
        # Definition of the complessive loss of our network architecture
        # The first if can be useful if we want a different initial loss for our network
        #####
        if epoch<-1:  # posso toglierlo
            loss_value = loss_emd1 + loss_emd2/1000 + loss_consistency 
        else:
            loss_value = loss_emd1 + loss_emd2 + 1000*loss_vd + 1000*loss_consistency

        loss_emd = tf.math.reduce_mean(loss_emd1) + loss_emd2

        # Keep track of the losses
        losses = [loss_vd,loss_rec,loss_emd,loss_emd1,loss_emd2,loss_consistency,loss_tvm]
        loss_list = [losses]
        return loss_value, loss_list	

    # RMSE loss forcing the mean of the vectors to be 1. Either the mean of each vector or of all the batch
    def RMSE_loss(self, x, x_pr, fl_scale_each=False):
        loss = tf.math.abs(x - x_pr)
        if fl_scale_each:
            scaling = tf.math.reduce_mean(x,axis=-1,keepdims=True)
            scaling = tf.where(scaling==0,1,scaling)
        else:
            scaling = tf.math.reduce_mean(x)
            if scaling == 0:
                scaling = 1
        loss/=scaling
        loss = loss**2
        loss_z = tf.where(x==0,loss,0)
        loss_z = tf.math.reduce_sum(loss_z)/tf.math.reduce_sum(tf.where(x==0,1.,0.))
        loss_nz = tf.where(x==0,0,loss)
        loss_nz = tf.math.reduce_sum(loss_nz)/tf.math.reduce_sum(tf.where(x==0,0.,1.))
        return loss_z, loss_nz

    def MAE_loss(self, x, x_pr, fl_scale_each=False):
        loss = tf.math.abs(x - x_pr)
        if fl_scale_each:
            scaling = tf.math.reduce_mean(x,axis=-1,keepdims=True)
            scaling = tf.where(scaling==0,1,scaling)
        else:
            scaling = tf.math.reduce_mean(x)
            if scaling == 0:
                scaling = 1
        loss/=scaling
        loss_z = tf.where(x==0,loss,0)
        loss_z = tf.math.reduce_sum(loss_z)/tf.math.reduce_sum(tf.where(x==0,1.,0.))
        loss_nz = tf.where(x==0,0,loss)
        loss_nz = tf.math.reduce_sum(loss_nz)/tf.math.reduce_sum(tf.where(x==0,0.,1.))
        return loss_z, loss_nz

    def RMSE_loss_2(self, x, x_pr, fl_scale_each=False):
        loss = tf.math.abs(x - x_pr)
        if fl_scale_each:
            scaling = tf.math.reduce_mean(x,axis=-1,keepdims=True)
            scaling = tf.where(scaling==0,1,scaling)
        else:
            scaling = tf.math.reduce_mean(x)
            if scaling == 0:
                scaling = 1
        loss/=scaling
        loss = loss**2
        loss = tf.sqrt(tf.math.reduce_mean(loss))
        return loss

    def MAE_loss_2(self, x, x_pr, fl_scale_each=False):
        loss = tf.math.abs(x - x_pr)
        if fl_scale_each:
            scaling = tf.math.reduce_mean(x,axis=-1,keepdims=True)
            scaling = tf.where(scaling==0,1,scaling)
        else:
            scaling = tf.math.reduce_mean(x)
            if scaling == 0:
                scaling = 1
        loss/=scaling
        loss = tf.math.reduce_mean(loss)
        return loss

    
    # Compute weighted Earth mover's distance between two vectors
    def EMDc_loss(self, x_cum, x_cum_pr, weights=None):
        
        # Compute difference between cumulative CDFs
        diff_cdf = x_cum-x_cum_pr         # SLOWEST OPERATION
        diff_cdf = tf.math.abs(diff_cdf)

        if weights is not None:
            diff_cdf = tf.math.reduce_mean(diff_cdf,axis=-1)
            diff_cdf *= weights
        # Compute density weighted Earth mover's distance
        emd = tf.math.reduce_mean(diff_cdf)
        return emd

    # Loss function defined for the S1 dataset
    def loss_S1(self, data_dict, epoch):

        # Load the needed data
        v_in = data_dict["raw_itof"]
        phase_gt = data_dict["phase_direct"]

        # Predict the intermediate result with the Spatial Feature Extractor
        if self.fl_denoise:
            v_out_nf = tf.squeeze(self.SpatialNet(v_in))
        else:
            v_out_nf = v_in
        init_ind = int(self.ex-(self.out_win-1)/2)
        # Process the intermediate result with the Direct CNN
        v_out_g,v_out_d,phi_pr = self.DirectCNN(v_out_nf)
        # Compute loss in the phase domain
        loss_phi = tf.math.abs(phase_gt[:,self.ex,self.ex,:]-tf.squeeze(phi_pr))
        loss20 = loss_phi[...,0]
        loss50 = loss_phi[...,1]
        loss50 = tf.where(loss50>math.pi,loss50-2*math.pi,loss50)
        loss50 = loss50%(math.pi)
        if self.fl_2freq:
            loss_phi = (loss50+loss20)/2
        else:
            loss60 = loss_phi[...,2]
            loss60 = tf.where(loss60>math.pi,loss60-2*math.pi,loss60)
            loss60 = loss60%(math.pi)
            loss_phi = (loss60+loss50+loss20)/3
        loss_phi = tf.math.reduce_mean(tf.math.abs(loss_phi))

        # Total loss
        loss_value =   loss_phi 
        phi_loss = [loss_phi]
        loss_list = [phi_loss]
        return loss_value, loss_list	
    
    def loss_S3(self, data_dict, epoch):

        # Load the needed data
        v_in = data_dict["raw_itof"]
        phase_gt = data_dict["phase_direct"]
        depth_gt = data_dict["depth_ground_truth"]   # Needed to keep track only of the valid pixels

        depth_valid = tf.where(depth_gt>0.3,1.,0.)
        depth_valid = tf.expand_dims(depth_valid,axis=-1)  # compute the loss only on the valid pixels
        depth_valid = depth_valid[:,self.ex:-self.ex,self.ex:-self.ex,:]

        # Apply the Spatial Feature Extractor to the input
        if self.fl_denoise:
            v_out_nf = tf.squeeze(self.SpatialNet(v_in))
        else:
            v_out_nf = v_in
        init_ind = int(self.ex-(self.out_win-1)/2)
        # Apply the Direct CNN to the intermediate result
        v_out_g,v_out_d,phi_pr = self.DirectCNN(v_out_nf)
        # Compute loss in the phase domain
        if self.fl_denoise:
            loss_phi = tf.math.abs(phase_gt[:,self.ex:-self.ex,self.ex:-self.ex,:]-tf.squeeze(phi_pr))
        else:
            loss_phi = tf.math.abs(phase_gt[:,self.ex:-self.ex,self.ex:-self.ex,:]-tf.squeeze(phi_pr))
        loss20 = loss_phi[...,0]
        loss50 = loss_phi[...,1]
        loss50 = tf.where(loss50>math.pi,loss50-2*math.pi,loss50)
        loss50 = loss50%(math.pi)
        if self.fl_2freq:
            loss_phi = (loss50+loss20)/2
        else:
            loss60 = loss_phi[...,2]
            loss60 = tf.where(loss60>math.pi,loss60-2*math.pi,loss60)
            loss60 = loss60%(math.pi)
            loss_phi = (loss60+loss50+loss20)/3
        loss_phi = tf.math.reduce_sum(tf.math.abs(loss_phi)*tf.squeeze(depth_valid))/tf.math.reduce_sum(depth_valid)

        # Total loss
        loss_value =   loss_phi 
        phi_loss = [loss_phi]
        loss_list = [phi_loss]
        return loss_value, loss_list	
    





    
    
    
    # Compute loss function and useful metrics over a given max number of batches
    def loss_perbatches(self, loader, N_batches=5,epoch=0):
        loader.init_iter()
        b = 0
        while b<N_batches:
            
            # Get one batch (restart from the beginning if the end is reached)
            data_dict = loader.next_batch()
            if data_dict is None:
                loader.init_iter()
                continue
            loss_batch, loss_list_batch = self.loss_fn(data_dict, epoch)
            # Update loss and metrics value
            if b<=0:
                loss = loss_batch
                loss_list = loss_list_batch
            else:
                loss = loss + loss_batch
                for i in range(len(loss_list)):
                    loss_list[i] = [a+b for a,b in zip(loss_list[i], loss_list_batch[i])]
            b += 1
        loss = loss / b
        for i in range(len(loss_list)):
            loss_list[i] = [a/b for a in loss_list[i]]
        
        return loss, loss_list
    

    
    
    
    
    
    # Training loop
    def training_loop(self, train_S1_loader=0, test_S1_loader=0, train_w_loader=0, test_w_loader=0, val_S3_loader=0, final_epochs=2000, init_epoch=0, print_freq=5, save_freq=5, pretrain_filenamed=None,pretrain_filenamev=None,pretrain_filenamez=None,use_S1=True,use_walls=True):

        
        if train_S1_loader is None:
            train_S1_loader = 0
            val_S1_loader = 0
        if train_w_loader is None:
            train_w_loader = 0
            test_w_loader = 0
        # Compute the initial loss and metrics
        if (not self.use_S1) and (not self.use_walls):
            print("ERROR: AT least one dataset must be used")
            sys.exit()
        if self.use_S1:
            loss_trainS1, loss_list_trainS1 = self.loss_perbatches(train_S1_loader, N_batches=1)
            loss_testS1, loss_list_testS1 = self.loss_perbatches(test_S1_loader,  N_batches=test_S1_loader.N_batches)
        else:
            loss_trainS1 = 0
            loss_testS1 = 0
        if self.use_walls:
            loss_trainw, loss_list_trainw = self.loss_perbatches(train_w_loader, N_batches=1)
            loss_testw, loss_list_testw = self.loss_perbatches(test_w_loader,  N_batches=test_w_loader.N_batches)
        else:
            loss_trainw = 0
            loss_testw = 0
        if not self.use_transient:
            loss_testS3, loss_list_testS3 = self.loss_perbatches(val_S3_loader, N_batches=test_S3_loader.N_batches)
        
        
        # Create log file and record initial loss and metrics
        summary_tr = tf.summary.create_file_writer(self.log_path_train)
        with summary_tr.as_default():
            if self.use_S1:
                tf.summary.scalar('loss_S1', loss_trainS1, step=init_epoch)
                tf.summary.scalar('err_phi_S1', loss_list_trainS1[0][0], step=init_epoch)
            if self.use_walls:
                tf.summary.scalar('loss_walls', loss_trainw, step=init_epoch)
                tf.summary.scalar('err_vd', loss_list_trainw[0][0], step=init_epoch)
                tf.summary.scalar('err_rec', loss_list_trainw[0][1], step=init_epoch)
                tf.summary.scalar('err_transient', loss_list_trainw[0][2], step=init_epoch)
                tf.summary.scalar('err_transient_autoencoder', loss_list_trainw[0][3], step=init_epoch)
                tf.summary.scalar('err_transient_vg_decoder', loss_list_trainw[0][4], step=init_epoch)
                tf.summary.scalar('err_consistency', loss_list_trainw[0][5], step=init_epoch)
                tf.summary.scalar('err_tvm', loss_list_trainw[0][6], step=init_epoch)

        summary_val = tf.summary.create_file_writer(self.log_path_validation)
        with summary_val.as_default():
            if self.use_S1:
                tf.summary.scalar('loss_S1', loss_testS1, step=init_epoch)
                tf.summary.scalar('err_phi_S1', loss_list_testS1[0][0], step=init_epoch)
            if self.use_walls:
                tf.summary.scalar('loss_walls', loss_testw, step=init_epoch)
                tf.summary.scalar('err_vd', loss_list_testw[0][0], step=init_epoch)
                tf.summary.scalar('err_rec', loss_list_testw[0][1], step=init_epoch)
                tf.summary.scalar('err_transient', loss_list_testw[0][2], step=init_epoch)
                tf.summary.scalar('err_transient_autoencoder', loss_list_testw[0][3], step=init_epoch)
                tf.summary.scalar('err_transient_vg_decoder', loss_list_testw[0][4], step=init_epoch)
                tf.summary.scalar('err_consistency', loss_list_testw[0][5], step=init_epoch)
                tf.summary.scalar('err_tvm', loss_list_testw[0][6], step=init_epoch)
            if not self.use_transient:
                tf.summary.scalar('loss_S3', loss_testS3, step=init_epoch)
                tf.summary.scalar('err_phi_S3', loss_list_testS3[0][0], step=init_epoch)
        loss_train = loss_trainS1 + loss_trainw
        if not self.use_transient:
            loss_test = loss_testw
        else:
            loss_test = loss_list_testw[0][4] # If we are using the transient, we want to optimize on the transient prediction, not S3
        print("Epoch = %d,\t train_loss = %f,\t test_loss = %f" % (init_epoch, loss_train, loss_test))
        # Save first model as best performing model over the test set
        self.best_loss_test = loss_test
        self.best_epoach = init_epoch
        self.save_weights(suffix='e'+str(init_epoch)+'_best_weights')
       
        # Use some pretrained weights if provided
        if (pretrain_filenamed is not None) or (pretrain_filenamev is not None) or (pretrain_filenamez is not None):
            self.load_weights(pretrain_filenamed,pretrain_filenamev,pretrain_filenamez)
        
        init_time = time.time()
        for epoch in range(init_epoch, final_epochs):
            # START_TRAINING over all the batches in the training dataset
            train_S1_loader.init_iter()
            train_w_loader.init_iter()
            loss_train = 0
            loss_trainw = 0
            loss_trainS1 = 0
            tr_count=0  # Use to keep count of the number of loops
            while True:
                
                # Get one batch (stop if the end is reached)
                if self.use_S1:
                    data_dictS1 = train_S1_loader.next_batch()  
                else:
                    data_dictS1=0
                if self.use_walls:
                    data_dictw = train_w_loader.next_batch() 
                else:
                    data_dictw = 0

                if data_dictS1 is None:
                    break
                if data_dictw is None:
                    break

                if self.use_S1:
                    with tf.GradientTape() as denoise_tape, tf.GradientTape() as predv_tape1:
                        lossS1, loss_listS1 = self.loss_fn(data_dictS1,  epoch)

                if self.use_walls:
                    with tf.GradientTape() as denoise_tape2, tf.GradientTape() as predv_tape2, tf.GradientTape() as predenc_tape, tf.GradientTape() as preddec_tape, tf.GradientTape() as predv_enc_tape:
                        lossw, loss_listw = self.loss_fn(data_dictw,  epoch)
                if self.use_S1:
                    if tr_count == 0:
                        loss_list_trainS1 = loss_listS1
                    else:
                        for i in range(len(loss_list_trainS1)):
                            loss_list_trainS1[i] = [a+b for a,b in zip(loss_list_trainS1[i], loss_listS1[i])]
                    loss_trainS1 += lossS1
                
                if self.use_walls:
                    if tr_count == 0:
                        loss_list_trainw = loss_listw
                    else:
                        for i in range(len(loss_list_trainw)):
                            loss_list_trainw[i] = [a+b for a,b in zip(loss_list_trainw[i], loss_listw[i])]
                    loss_trainw += lossw

                # Computes gradient of the loss function wrt the training variables...
                # ...and then pply one batch optimization (apply gradients to training variables)
                if self.use_S1:
                    if self.fl_denoise:
                        grads_d = denoise_tape.gradient(lossS1, self.SpatialNet.trainable_variables)
                        self.optimizer.apply_gradients(grads_and_vars=zip(grads_d, self.SpatialNet.trainable_variables))
                    grads_v = predv_tape1.gradient(lossS1, self.DirectCNN.trainable_variables)
                    self.optimizer.apply_gradients(grads_and_vars=zip(grads_v, self.DirectCNN.trainable_variables))
                if self.use_walls:
                    if self.fl_denoise:
                        grads_d2 = denoise_tape2.gradient(lossw, self.SpatialNet.trainable_variables)
                        self.optimizer.apply_gradients(grads_and_vars=zip(grads_d2, self.SpatialNet.trainable_variables))
                    grads_v = predv_tape2.gradient(lossw, self.DirectCNN.trainable_variables)
                    self.optimizer.apply_gradients(grads_and_vars=zip(grads_v, self.DirectCNN.trainable_variables))
                    if self.use_transient:
                        if self.fl_old_transient:
                            grads_z = predz_tape.gradient(lossw, self.TransientNet.trainable_variables)
                            self.optimizer.apply_gradients(grads_and_vars=zip(grads_z, self.TransientNet.trainable_variables))
                        else:
                            grads_enc = predenc_tape.gradient(lossw, self.encoder.trainable_variables)
                            self.optimizer.apply_gradients(grads_and_vars=zip(grads_enc, self.encoder.trainable_variables))
                            grads_dec = preddec_tape.gradient(lossw, self.decoder.trainable_variables)
                            self.optimizer.apply_gradients(grads_and_vars=zip(grads_dec, self.decoder.trainable_variables))
                            grads_predv_enc = predv_enc_tape.gradient(lossw, self.predv_encoding.trainable_variables)
                            if grads_predv_enc[0] is not None:
                                self.optimizer.apply_gradients(grads_and_vars=zip(grads_predv_enc, self.predv_encoding.trainable_variables))

                tr_count += 1
            # END_TRAINING over all the batches in the training set
            if self.use_S1:
                loss_trainS1/= tr_count
                for i in range(len(loss_list_trainS1)):
                    loss_list_trainS1[i] = [a/tr_count for a in loss_list_trainS1[i]]
            if self.use_walls:
                loss_trainw/= tr_count
                for i in range(len(loss_list_trainw)):
                    loss_list_trainw[i] = [a/tr_count for a in loss_list_trainw[i]]
            
            # Compute loss and metrics for the current epoach
            if self.use_S1:
                loss_testS1, loss_list_testS1 = self.loss_perbatches(test_S1_loader,  N_batches=test_S1_loader.N_batches,epoch=epoch)
            if self.use_walls:
                loss_testw, loss_list_testw = self.loss_perbatches(test_w_loader,  N_batches=test_w_loader.N_batches,epoch=epoch)
            if not self.use_transient:
                loss_testS3, loss_list_testS3 = self.loss_perbatches(val_S3_loader,  N_batches=val_S3_loader.N_batches,epoch=epoch)
            
            # Update log
            with summary_tr.as_default(): 
                if self.use_S1:
                    tf.summary.scalar('loss_S1', loss_trainS1, step=epoch+1)
                    tf.summary.scalar('err_phi_S1', loss_list_trainS1[0][0], step=epoch+1)
                if self.use_walls:
                    tf.summary.scalar('loss_walls', loss_trainw, step=epoch+1)
                    tf.summary.scalar('err_vd', loss_list_trainw[0][0], step=epoch+1)
                    tf.summary.scalar('err_rec', loss_list_trainw[0][1], step=epoch+1)
                    tf.summary.scalar('err_transient', loss_list_trainw[0][2], step=epoch+1)
                    tf.summary.scalar('err_transient_autoencoder', loss_list_trainw[0][3], step=epoch+1)
                    tf.summary.scalar('err_transient_vg_decoder', loss_list_trainw[0][4], step=epoch+1)
                    tf.summary.scalar('err_consistency', loss_list_trainw[0][5], step=epoch+1)
                    tf.summary.scalar('err_tvm', loss_list_trainw[0][6], step=epoch+1)
            with summary_val.as_default(): 
                if self.use_S1:
                    tf.summary.scalar('loss_S1', loss_testS1, step=epoch+1)
                    tf.summary.scalar('err_phi_S1', loss_list_testS1[0][0], step=epoch+1)
                if self.use_walls:
                    tf.summary.scalar('loss_walls', loss_testw, step=epoch+1)
                    tf.summary.scalar('err_vd', loss_list_testw[0][0], step=epoch+1)
                    tf.summary.scalar('err_rec', loss_list_testw[0][1], step=epoch+1)
                    tf.summary.scalar('err_transient', loss_list_testw[0][2], step=epoch+1)
                    tf.summary.scalar('err_transient_autoencoder', loss_list_testw[0][3], step=epoch+1)
                    tf.summary.scalar('err_transient_vg_decoder', loss_list_testw[0][4], step=epoch+1)
                    tf.summary.scalar('err_consistency', loss_list_testw[0][5], step=epoch+1)
                    tf.summary.scalar('err_tvm', loss_list_testw[0][6], step=epoch+1)
                if not self.use_transient:
                    tf.summary.scalar('loss_S3', loss_testS3, step=epoch+1)
                    tf.summary.scalar('err_phi_S3', loss_list_testS3[0][0], step=epoch+1)
            loss_train = loss_trainS1 + loss_trainw
            #loss_train = loss_testS3
            if not self.use_transient:
                loss_test = loss_testw
            else:
                loss_test = loss_list_testw[0][4]
            # Track and save best performing model over the testset
            if loss_test<self.best_loss_test:             
                
                # Save new best model
                old_weight_filenamed = self.name + '_d_e'+str(self.best_epoach)+'_best_weights.h5'
                old_weight_filenamev = self.name + '_v_e'+str(self.best_epoach)+'_best_weights.h5'
                if self.fl_old_transient:
                    old_weight_filenamez = self.name + '_z_e'+str(self.best_epoach)+'_best_weights.h5'
                old_weight_filename_enc = self.name + '_enc_e'+str(self.best_epoach)+'_best_weights.h5'
                old_weight_filename_dec = self.name + '_dec_e'+str(self.best_epoach)+'_best_weights.h5'
                old_weight_filename_predv_enc = self.name + '_predv_enc_e'+str(self.best_epoach)+'_best_weights.h5'
        
                self.best_loss_test = loss_test
                self.best_epoach = epoch+1
                self.save_weights(suffix='e'+str(self.best_epoach)+'_best_weights')
                
                # Remove old best model
                os.remove(os.path.join(self.checkpoint_path, old_weight_filenamed))
                os.remove(os.path.join(self.checkpoint_path, old_weight_filenamev))
                os.remove(os.path.join(self.checkpoint_path, old_weight_filename_enc))
                os.remove(os.path.join(self.checkpoint_path, old_weight_filename_dec))
                os.remove(os.path.join(self.checkpoint_path, old_weight_filename_predv_enc))
                if self.fl_old_transient:
                    os.remove(os.path.join(self.checkpoint_path, old_weight_filenamez))
            
            # Print loss
            if (epoch+1)%print_freq==0:

                end_time = time.time()
                tot_time = end_time-init_time
                print("Epoch = %d,\t train_loss = %f,\t test_loss = %f,\t  epoch time [s] = %f" % (epoch+1, loss_train, loss_test,tot_time))
                init_time = end_time
            
            # Save weights
            if (epoch+1)%save_freq==0:
                self.save_weights(suffix='e'+str(epoch+1)+'_weights')
