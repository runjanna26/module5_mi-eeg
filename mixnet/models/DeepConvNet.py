import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from mixnet import models
from mixnet.loss import *

class DeepConvNet(models.base.BaseModel):
    
    def __init__(self, 
                 optimizer,
                 input_shape=(1,400,20),
                 num_class=2, 
                 loss=[SparseCategoricalCrossentropy()],
                 loss_names=['crossentropy'],
                 loss_weights=[1.0],
                 model_name='DeepConvNet',
                 data_format='channels_last',
                 **kwargs):
        super().__init__(num_class, loss, loss_names, loss_weights, optimizer, data_format, **kwargs)
        self.input_shape = input_shape
        self.num_class = num_class
        self.model_name = model_name
        self._config(**kwargs)
        
    def _config(self, **kwargs):
        self.kernLength = 50
        self.norm_rate = 0.5
        self.dropout_rate = 0.5
        if self.data_format == 'channels_first':
            self.Chans = self.input_shape[1]
            self.Samples = self.input_shape[2]
        else:
            self.Chans = self.input_shape[0]
            self.Samples = self.input_shape[1]
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
            
    def build(self, print_summary=True, load_weights=False):
        '''
        build a model and return as tf.keras.models.Model
        --------
        Keras implementation of the Deep Convolutional Network as described in
        Schirrmeister et. al. (2017), Human Brain Mapping.

        This implementation assumes the input is a 2-second EEG signal sampled at
        128Hz, as opposed to signals sampled at 250Hz as described in the original
        paper. We also perform temporal convolutions of length (1, 5) as opposed
        to (1, 10) due to this sampling rate difference.

        Note that we use the max_norm constraint on all convolutional layers, as
        well as the classification layer. We also change the defaults for the
        BatchNormalization layer. We used this based on a personal communication
        with the original authors.

                          ours        original paper
        pool_size        1, 2        1, 3
        strides          1, 2        1, 3
        conv filters     1, 5        1, 10

        Note that this implementation has not been verified by the original
        authors.
        '''
        
        input_main   = layers.Input(self.input_shape)
        block1       = layers.Conv2D(25, (1, 5),
                              input_shape=(self.input_shape),
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
        block1       = layers.Conv2D(25, (self.Chans, 1),
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1       = layers.BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
        block1       = layers.Activation('elu')(block1)
        block1       = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
        block1       = layers.Dropout(self.dropout_rate)(block1)

        block2       = layers.Conv2D(50, (1, 5),
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block2       = layers.BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
        block2       = layers.Activation('elu')(block2)
        block2       = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
        block2       = layers.Dropout(self.dropout_rate)(block2)

        block3       = layers.Conv2D(100, (1, 5),
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
        block3       = layers.BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
        block3       = layers.Activation('elu')(block3)
        block3       = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
        block3       = layers.Dropout(self.dropout_rate)(block3)

        block4       = layers.Conv2D(200, (1, 5),
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
        block4       = layers.BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
        block4       = layers.Activation('elu')(block4)
        block4       = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
        block4       = layers.Dropout(self.dropout_rate)(block4)

        flatten      = layers.Flatten()(block4)

        dense        = layers.Dense(self.num_class, kernel_constraint = max_norm(self.norm_rate))(flatten)
        softmax      = layers.Activation('softmax')(dense)
        
        model = Model(inputs=input_main, outputs=softmax, name=self.model_name)
        
        if print_summary:
            model.summary()
        if load_weights:
            print('loading weights from', self.weights_dir)
            model.load_weights(self.weights_dir)
        return model
    
    @tf.function
    def train_step(self, x, y, loss_weights):
        with tf.GradientTape() as tape:
            y_logis = self.model(x, training=True)
            crossentropy_loss = self.loss.crossentropy(y, y_logis)
            losses = [crossentropy_loss]
            train_loss = tf.reduce_sum(crossentropy_loss)
            self.train_acc_metric.update_state(y, y_logis)
        grads = tape.gradient(train_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        logs = dict({'train_loss': train_loss})
        logs.update(dict(zip(['train_'+loss_name+'_loss' for loss_name in self.loss_names], losses)))
        logs.update(dict({'train_acc': self.train_acc_metric.result()}))
        return logs, y_logis
    
    @tf.function
    def val_step(self, x, y, loss_weights):
        y_logis = self.model(x, training=False)
        crossentropy_loss = self.loss.crossentropy(y, y_logis)
        losses = [crossentropy_loss]
        val_loss = tf.reduce_sum(crossentropy_loss)
        self.val_acc_metric.update_state(y, y_logis)
        logs = dict({'val_loss': val_loss})
        logs.update(dict(zip(['val_'+loss_name+'_loss' for loss_name in self.loss_names], losses)))
        logs.update(dict({'val_acc': self.val_acc_metric.result()}))
        return logs, y_logis
    
    @tf.function
    def test_step(self, x, y, loss_weights):
        y_logis = self.model(x, training=False)
        crossentropy_loss = self.loss.crossentropy(y, y_logis)
        losses = [crossentropy_loss]
        test_loss = tf.reduce_sum(crossentropy_loss)
        self.test_acc_metric.update_state(y, y_logis)
        logs = dict({'test_loss': test_loss})
        logs.update(dict(zip(['test_'+loss_name+'_loss' for loss_name in self.loss_names], losses)))
        logs.update(dict({'test_acc': self.test_acc_metric.result()}))
        return logs, y_logis

    @tf.function
    def pred_step(self, x):
        y_logis = self.model(x, training=False)
        return y_logis