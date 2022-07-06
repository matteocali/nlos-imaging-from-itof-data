import tensorflow as tf
import sys
import tensorflow.keras.layers as layers

# Class defining the autoencoder structure of our network.
# Decoder and Encoder are composed of multiple same size convolutions and pooling/upsampling layers
# All computations are unidimensional

class Autoencoder_Interp:
    
    def __init__(self, dim_b,  dim_BV, dim_hidden=12, num_filters=32):
        
        self.dim_b = dim_b
        self.dim_BV = dim_BV
        self.dim_hidden = dim_hidden
        self.num_filters = num_filters

        # Get the correct shapes for the transpose convolutions
        self.k_size_dec = 3
        self.k_size_enc = 3   
        
    def encoder(self):
            
            # Define the input placeholder
            x_gt = tf.keras.Input(shape=(self.dim_BV,1), batch_size=self.dim_b, 
                                  dtype='float32', name='x_gt')
    
            c_out = tf.keras.layers.AveragePooling1D(pool_size=10)(x_gt)

            c_out = layers.Conv1D(filters=self.num_filters,
                               activation="relu",
                               padding="same",
                               kernel_size=self.k_size_enc,
                               strides=1)(c_out)

            c_out = layers.Conv1D(filters=self.num_filters,
                               activation="relu",
                               padding="same",
                               kernel_size=self.k_size_enc,
                               strides=1)(c_out)

            c_out = tf.keras.layers.AveragePooling1D(pool_size=2)(c_out)

            c_out = layers.Conv1D(filters=self.num_filters,
                               activation="relu",
                               padding="same",
                               kernel_size=self.k_size_enc,
                               strides=1)(c_out)

            c_out = layers.Conv1D(filters=self.num_filters,
                               activation="relu",
                               padding="same",
                               kernel_size=self.k_size_enc,
                               strides=1)(c_out)
            

            c_out = tf.keras.layers.AveragePooling1D(pool_size=2)(c_out)

            c_out = layers.Conv1D(filters=self.num_filters,
                               activation="relu",
                               padding="same",
                               kernel_size=self.k_size_enc,
                               strides=1)(c_out)

            c_out = layers.Conv1D(filters=self.num_filters,
                               activation="relu",
                               padding="same",
                               kernel_size=self.k_size_enc,
                               strides=1)(c_out)

            c_out = tf.keras.layers.AveragePooling1D(pool_size=2)(c_out)

            c_out = layers.Conv1D(filters=self.num_filters,
                               activation="relu",
                               padding="same",
                               kernel_size=self.k_size_enc,
                               strides=1)(c_out)

            c_out = layers.Conv1D(filters=self.num_filters,
                               activation="relu",
                               padding="same",
                               kernel_size=self.k_size_enc,
                               strides=1)(c_out)



            hidden = layers.Conv1D(filters=int(self.dim_hidden/2),
                               activation=None,
                               kernel_size=25,
                               strides=1)(c_out)




            # Define Encoder model
            model_encoder = tf.keras.Model(inputs=x_gt, outputs=hidden)
            model_encoder.summary()
            return model_encoder
            
    def interpConv(self):
            # Define the input placeholder
            hidden = tf.keras.Input(shape=(self.dim_hidden), batch_size=None, 
                                  dtype='float32', name='hidden')


            c_out = layers.Dense(units=25*self.num_filters,
                               activation="relu")(hidden)

            c_out = tf.reshape(c_out,[-1,25,self.num_filters])

            c_out = layers.Conv1D(filters=self.num_filters,
                               activation="relu",
                               padding="same",
                               kernel_size=self.k_size_dec,
                               strides=1)(c_out)

            c_out = layers.Conv1D(filters=self.num_filters,
                               activation="relu",
                               padding="same",
                               kernel_size=self.k_size_dec,
                               strides=1)(c_out)

            c_out = tf.keras.layers.UpSampling1D(size=2)(c_out)

            c_out = layers.Conv1D(filters=self.num_filters,
                               activation="relu",
                               padding="same",
                               kernel_size=self.k_size_dec,
                               strides=1)(c_out)
            
            c_out = layers.Conv1D(filters=self.num_filters,
                               activation="relu",
                               padding="same",
                               kernel_size=self.k_size_dec,
                               strides=1)(c_out)

            c_out = tf.keras.layers.UpSampling1D(size=2)(c_out)

            c_out = layers.Conv1D(filters=self.num_filters,
                               activation="relu",
                               padding="same",
                               kernel_size=self.k_size_dec,
                               strides=1)(c_out)

            c_out = layers.Conv1D(filters=self.num_filters,
                               activation="relu",
                               padding="same",
                               kernel_size=self.k_size_dec,
                               strides=1)(c_out)

            c_out = tf.keras.layers.UpSampling1D(size=2)(c_out)

            c_out = layers.Conv1D(filters=self.num_filters,
                               activation="relu",
                               padding="same",
                               kernel_size=self.k_size_dec,
                               strides=1)(c_out)

            c_out = layers.Conv1D(filters=1,
                               activation="elu",
                               padding="causal",
                               kernel_size=self.k_size_dec,
                               use_bias="True",
                               strides=1)(c_out)

            

            c_out = tf.keras.layers.UpSampling1D(size=10)(c_out)
            
            c_out = tf.squeeze(c_out)
    
            # Define Decoder model
            model_decoder = tf.keras.Model(inputs=hidden, outputs=c_out)
            model_decoder.summary()
            return model_decoder 
