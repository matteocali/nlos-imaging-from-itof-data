import tensorflow as tf
import tensorflow.keras.layers as layers
import os
import sys
import time
from tensorflow.keras.utils import plot_model

sys.path.append("../../utils/")
sys.path.append("../")
import Autoencoder_sameconv as Autoencoder_Interp


class PredictiveModel:
    def __init__(self, name, dim_b, lr, freqs, P, saves_path, dim_t=2000,
                 fil_size=8):
        """
        Initialize the Predictive model class
        Inputs:
            name:               dtype='string'
                                Name of the predictive model
            
            dim_b:              dtype='int', shape=(1,)
                                Batch dimension

            freqs:              dtype='float32', shape=(# of frequencies,)
                                Modulation frequencies of the input itof data
            
            saves_path:         dtype='string'
                                Path to the directory where to save checkpoints and logs

            dim_t:              dtype='int'
                                Number of bins in the transient dimension

            fil_size:           dtype='int'
                                Number of feature maps for the Direct CNN
        """

        # Initializing the flags and other input parameters
        self.name = name                                           # name of the predictive model
        self.dim_b = dim_b                                         # batch dimension
        self.dim_t = dim_t                                         # number of bins in the transient dimension
        self.P = P                                                 # patch size
        self.ex = int((self.P - 1) / 2)                            # index keeping track of the middle of the patch
        self.fn = freqs.shape[0]                                   # number of frequencies
        self.fn2 = self.fn * 2                                     # number of raw measurements (twice the number of frequencies)
        self.fl_2freq = (self.fn == 2)                             # whether we are training with 2 frequencies or not
        self.fil_pred = fil_size                                   # number of filter for the Direct CNN
        self.freqs = tf.convert_to_tensor(freqs, dtype="float32")  # modulation frequencies

        # Defining all parameters needed by the denoising network
        self.in_shape = (P, P, self.fn2)       # Shape of the input (batch size excluded)
        self.out_win = 3                       # Side of the window provided in output of the Spatial Feature Extractor
        self.padz = self.P - self.out_win + 2  # Padding needed
        self.fl_denoise = False                # Whether to use the Spatial Feature Extractor
        self.k_size = 3                        # Kernel size for each layer of the denoiser network
        self.f_skip = True                     # Whether to use a skip connection or not

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
        self.log_path_img = os.path.join(self.log_path, 'img')
        if not os.path.exists(self.log_path_img):
            os.makedirs(self.log_path_img)

        # Create checkpoints path, if it does not exist
        self.checkpoint_path = os.path.join(self.net_path, 'checkpoints')
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        # Define predictive models
        # self.SpatialNet = self.def_SpatialNet()
        self.DirectCNN = self.def_DirectCNN()

        # Define loss function and metrics
        self.loss_fn = self.def_loss

        # Define optimizer
        self.lr = lr
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

        # Track and save the best performing model over the test set at each epoch
        self.best_loss_test = -1
        self.best_epoch = -1
        self.c = 299792458.  # speed of light

    def load_weights(self, weight_filename_v=None):
        """
        Load weights from a file
        """

        if weight_filename_v is not None:
            self.DirectCNN.load_weights(weight_filename_v, by_name=True)

    def save_weights(self, suffix):
        """
        Save weights to a file
        """

        weight_filename_v = self.name + '_v_' + suffix + '.h5'
        self.DirectCNN.save_weights(os.path.join(self.checkpoint_path, weight_filename_v))

    def A_compute(self, v):
        """
        Compute the amplitude of the phasors given the raw measurements
        """

        A = tf.math.sqrt(tf.math.square(v[..., :self.fn]) + tf.math.square(v[..., self.fn:]))
        return A

    def ambiguity_compute(self):
        """
        Compute the ambiguity range values at the three different frequencies
        """

        return self.c / (2 * self.freqs)

    def def_DirectCNN(self):
        """
        Define the DirectCNN model
        """

        # Define the input placeholder
        v_in = tf.keras.Input(shape=(None, None, self.fn2), batch_size=None, dtype='float32', name='v_in')

        c_out = layers.Conv2D(filters=self.fil_pred,
                              kernel_size=3,
                              strides=1,
                              padding='valid',
                              data_format='channels_last',
                              activation='relu',
                              use_bias=True,
                              trainable=True,
                              name="conv_1")(v_in)
        # Convolutional layers leading to the prediction of the depth data
        for i in range(3):
            l_name = f"conv_{str(i + 2)}"
            c_out = layers.Conv2D(filters=self.fil_pred,
                                  kernel_size=3,
                                  strides=1,
                                  padding='valid',
                                  data_format='channels_last',
                                  activation='relu',
                                  use_bias=True,
                                  trainable=True,
                                  name=l_name)(c_out)
        final_out = layers.Conv2D(filters=2,
                                  kernel_size=3,
                                  strides=1,
                                  padding='valid',
                                  data_format='channels_last',
                                  activation=None,
                                  use_bias=True,
                                  trainable=True,
                                  name=f'conv_5')(c_out)

        # Separate the two output: depth_map and alpha_map
        depth_map = tf.slice(final_out, begin=[0, 0, 0, 0], size=[-1, -1, -1, 1], name='depth_map')
        alpha_map = tf.slice(final_out, begin=[0, 0, 0, 1], size=[-1, -1, -1, 1], name='alpha_map')

        model_pred = tf.keras.Model(inputs=v_in, outputs=[depth_map, alpha_map], name=self.name)
        plot_model(model_pred, os.path.join(self.net_path, "CNN_model.png"), show_shapes=True)
        return model_pred

    def def_loss(self, data_dict):
        """
        Define custom loss function
        NOTE:
           - time ranges are all between 0 and 5
           - all transient vectors are normalized w.r.t. the max of the cumulative distribution of the noise
           - all v values are normalized to 1, thus the need for the mutliplicative costants
        """

        # Choose what kind of loss and networks to use according to the dataset we are using.
        loss, loss_list = self.loss_data(data_dict)
        return loss, loss_list

    # Loss computed on the transient dataset
    def loss_data(self, data_dict):
        # Load the needed data
        v_in = data_dict["raw_itof"]
        gt_depth = data_dict["gt_depth"]
        gt_alpha = data_dict["gt_alpha"]

        # Extract just the single pixel from the gt
        i_mid = int((self.P - 1) / 2)  # index keeping track of the middle position
        gt_depth = tf.slice(gt_depth, begin=[0, i_mid, i_mid], size=[-1, 1, 1])
        gt_alpha = tf.slice(gt_alpha, begin=[0, i_mid, i_mid], size=[-1, 1, 1])

        # Process the output with the Direct CNN
        pred_depth_map, pred_alpha_map = self.DirectCNN(v_in)
        pred_depth_map = tf.squeeze(pred_depth_map, axis=-1)
        pred_alpha_map = tf.squeeze(pred_alpha_map, axis=-1)

        # Compute the masked data
        pred_msk_depth = pred_depth_map * gt_alpha

        # Compute the loss
        loss_depth = tf.math.reduce_sum(tf.math.abs(gt_depth - pred_msk_depth)) / tf.math.reduce_sum(gt_alpha)  # MAE loss on the masked depth
        loss_alpha = tf.math.abs(gt_alpha - pred_alpha_map)
        loss_alpha = tf.math.reduce_mean(loss_alpha)
        final_loss = loss_depth + loss_alpha

        # Keep track of the losses
        loss_list = [[loss_depth, loss_alpha]]
        return final_loss, loss_list

    # Compute loss function and useful metrics over a given max number of batches
    def loss_perbatches(self, loader, n_batches=5):
        loader.init_iter()
        b = 0
        while b < n_batches:
            # Get one batch (restart from the beginning if the end is reached)
            data_dict = loader.next_batch()
            if data_dict is None:
                loader.init_iter()
                continue
            loss_batch, loss_list_batch = self.loss_fn(data_dict)
            # Update loss and metrics value
            if b <= 0:
                loss = loss_batch
                loss_list = loss_list_batch
            else:
                loss = loss + loss_batch
                for i in range(len(loss_list)):
                    loss_list[i] = [a + b for a, b in zip(loss_list[i], loss_list_batch[i])]
            b += 1
        loss = loss / b
        for i in range(len(loss_list)):
            loss_list[i] = [a / b for a in loss_list[i]]

        return loss, loss_list

    # Training loop
    def training_loop(self, train_w_loader=0, test_w_loader=0, final_epochs=2000, init_epoch=0, print_freq=5,
                      save_freq=5, pretrain_filenamev=None):

        if train_w_loader is None:
            train_w_loader = 0
            test_w_loader = 0

        # Compute the initial loss and metrics
        loss_trainw, loss_list_trainw = self.loss_perbatches(train_w_loader, n_batches=1)
        loss_testw, loss_list_testw = self.loss_perbatches(test_w_loader, n_batches=test_w_loader.N_batches)

        # Create log file and record initial loss and metrics
        summary_tr = tf.summary.create_file_writer(self.log_path_train)
        with summary_tr.as_default():
            tf.summary.scalar('loss_data', loss_trainw, step=init_epoch)
            tf.summary.scalar('loss_depth', loss_list_trainw[0][0], step=init_epoch)
            tf.summary.scalar('loss_alpha', loss_list_trainw[0][1], step=init_epoch)

        summary_val = tf.summary.create_file_writer(self.log_path_validation)
        with summary_val.as_default():
            tf.summary.scalar('loss_data', loss_trainw, step=init_epoch)
            tf.summary.scalar('loss_depth', loss_list_trainw[0][0], step=init_epoch)
            tf.summary.scalar('loss_alpha', loss_list_trainw[0][1], step=init_epoch)

        loss_train = loss_trainw
        loss_test = loss_testw
        print("Epoch = %d,\t train_loss = %f,\t test_loss = %f" % (init_epoch, loss_train, loss_test))
        # Save first model as best performing model over the test set
        self.best_loss_test = loss_test
        self.best_epoch = init_epoch
        self.save_weights(suffix='e' + str(init_epoch) + '_best_weights')

        # Use some pretrained weights if provided
        if pretrain_filenamev is not None:
            self.load_weights(pretrain_filenamev)

        init_time = time.time()
        for epoch in range(init_epoch, final_epochs):
            # START_TRAINING over all the batches in the training dataset
            train_w_loader.init_iter()
            loss_trainw = 0
            tr_count = 0  # use to keep count of the number of loops
            while True:
                # Get one batch (stop if the end is reached)
                data_dictw = train_w_loader.next_batch()
                if data_dictw is None:
                    break

                with tf.GradientTape() as denoise_tape2, tf.GradientTape() as predv_tape2:
                        lossw, loss_listw = self.loss_fn(data_dictw)

                if tr_count == 0:
                        loss_list_trainw = loss_listw
                else:
                    for i in range(len(loss_list_trainw)):
                        loss_list_trainw[i] = [a + b for a, b in zip(loss_list_trainw[i], loss_listw[i])]
                loss_trainw += lossw

                # Computes gradient of the loss function wrt the training variables
                # and then apply one batch optimization (apply gradients to training variables)
                grads_v = predv_tape2.gradient(lossw, self.DirectCNN.trainable_variables)
                self.optimizer.apply_gradients(grads_and_vars=zip(grads_v, self.DirectCNN.trainable_variables))

                tr_count += 1
            # END_TRAINING over all the batches in the training set
            loss_trainw /= tr_count
            for i in range(len(loss_list_trainw)):
                loss_list_trainw[i] = [a / tr_count for a in loss_list_trainw[i]]

            # Compute loss and metrics for the current epoch
            loss_testw, loss_list_testw = self.loss_perbatches(test_w_loader, n_batches=test_w_loader.N_batches)

            # Update log
            with summary_tr.as_default():
                tf.summary.scalar('loss_data', loss_trainw, step=epoch + 1)
                tf.summary.scalar('loss_depth', loss_list_trainw[0][0], step=epoch + 1)
                tf.summary.scalar('loss_alpha', loss_list_trainw[0][1], step=epoch + 1)
            with summary_val.as_default():
                tf.summary.scalar('loss_data', loss_testw, step=epoch + 1)
                tf.summary.scalar('loss_depth', loss_list_testw[0][0], step=epoch + 1)
                tf.summary.scalar('loss_alpha', loss_list_testw[0][1], step=epoch + 1)

            loss_train = loss_trainw
            loss_test = loss_testw

            # Track and save best performing model over the test set
            if loss_test < self.best_loss_test:
                # Save new best model
                old_weight_filename_d = self.name + '_d_e' + str(self.best_epoch) + '_best_weights.h5'
                old_weight_filename_v = self.name + '_v_e' + str(self.best_epoch) + '_best_weights.h5'

                self.best_loss_test = loss_test
                self.best_epoch = epoch + 1
                self.save_weights(suffix='e' + str(self.best_epoch) + '_best_weights')

                # Remove old best model
                os.remove(os.path.join(self.checkpoint_path, old_weight_filename_v))
            # Print loss
            if (epoch + 1) % print_freq == 0:
                end_time = time.time()
                tot_time = end_time - init_time
                print("Epoch = %d,\t train_loss = %f,\t test_loss = %f,\t  epoch time [s] = %f" % (epoch + 1, loss_train, loss_test, tot_time))
                init_time = end_time

            # Save weights
            if (epoch + 1) % save_freq == 0:
                self.save_weights(suffix='e' + str(epoch + 1) + '_weights')
