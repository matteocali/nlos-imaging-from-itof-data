import tensorflow as tf
import tensorflow.keras.layers as layers
import sys


class GenerativeModel(layers.Layer):
    """ The GenerativeModel class implements a non-trainable keras layer which
        acts as analytic generative model.
        
        Given in input a latent variable z=[a,b,lam,k] produces in output the associated 
        cumulative backscattering vector cum_x=[x_0, ...x_N] where:
                    cum_x_n = a*(1-exp(((x_n-b)/lam)**k)) 
    """
    
    
    
    def __init__(self, dim_b, dim_x, dim_y, dim_t):
        """ Initialize the generative model
            Note: __init__: where you can do all input-independent initialization
            
        Inputs:
            dim_b:           dtype='int', shape=(1,)
                             Batch dimension
            
            dim_x:           dtype='int', shape=(1,)
                             Width in the spatial dimension
            
            dim_y:           dtype='int', shape=(1,)
                             Height in the spatial dimension
            
            dim_t:           dtype='int', shape=(1,)
                             Time dimension, i.e. numbers of discretization time steps for the backscattering vector
        """
        
        self.dim_b = dim_b
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_t = dim_t
        
        # Set ranges for latent parameter (position)
        self.pos_min = 0
        self.pos_max = dim_t-1
        
        # Generate time indices, and reshape in the form (dim_b,dim_x,dim_y,dim_t)
        self.time_indices = tf.range(0, self.dim_t, dtype='float32')
        self.time_indices = tf.reshape(self.time_indices, shape=(1,1,1,-1))
        self.time_indices = tf.tile(self.time_indices, multiples=[self.dim_b, self.dim_x, self.dim_y, 1])
        
        # Call super class initializer
        super(GenerativeModel, self).__init__()
    
    
    def build(self, input_shape):
        """ Note: build: where you know the shapes of the input tensors and can do the rest of the initialization
        """
        
        # Call super class initializer
        super(GenerativeModel, self).build(input_shape)
    
    
    
    def compute_output_shape(input_shape):
        """ Note: Compute output shape in order to do shape inference without actually executing the computation
        """
        
        # Return output shape
        return (self.dim_b, self.dim_x, self.dim_y, self.dim_t)
    
    
    
    def call(self, z):
        """ Generate transient vector from its latent representation
            Note: call: where you do the forward computation
            
        Inputs:
            z:               dtype='float32', shape=(dim_b,dim_x,dim_y,4)
                             Latent representation of the output backscattering vector ( z[b,u,v,:] = [A1,T1,A2,T2] )

        Returns:
            x:               dtype='float32', shape=(dim_b,dim_x,dim_y,dim_t)
                             Output backscattering vector ( x[b,u,v,x_0,...,x_N] with x_n = gauss(A1,T1,0) + gauss(A2,T2,0) )
        """
        
        # Get normalized amplitude and position
        lam = tf.slice(z, begin=[0,0,0,0], size=[-1,-1,-1,1])
        k = tf.slice(z, begin=[0,0,0,1], size=[-1,-1,-1,1])
        a = tf.slice(z, begin=[0,0,0,2], size=[-1,-1,-1,1])
        b = tf.slice(z, begin=[0,0,0,3], size=[-1,-1,-1,1])
        b = self.pos_min/self.dim_t*5 + (self.pos_max-self.pos_min)/self.dim_t*5 * b  # Put b in the correct range [0,5] (maximum depth in the dataset is 5 meters)
        
        # Compute cumulative weibull without direct component
        x = self.weibull_cum(a, b, lam, k)
        return  x
    
    def weibull_cum(self, a, b, lam, k): 	
        """ Compute weibull cumulative distribution given its parameters
        """    
        weibull_cum = tf.math.divide(self.time_indices,self.dim_t)
        weibull_cum = tf.math.multiply(weibull_cum,5)
        weibull_cum = tf.math.subtract(weibull_cum,b)
        weibull_cum = tf.where(weibull_cum>0,tf.math.divide(weibull_cum,lam),0)
        weibull_cum = tf.where(weibull_cum>0,-tf.math.pow(weibull_cum,k),0)
        weibull_cum = tf.where(weibull_cum<=0,tf.math.exp(weibull_cum),0)
        weibull_cum = tf.where(weibull_cum>0,tf.math.subtract(1,weibull_cum),0)
        weibull_cum = tf.where(weibull_cum>0,tf.math.multiply(a,weibull_cum),0)
        return weibull_cum
    

