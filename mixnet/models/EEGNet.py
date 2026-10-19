import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from mixnet import models
from mixnet.loss import *

class EEGNet(models.base.BaseModel):
    
    def __init__(self, 
                 optimizer,
                 input_shape=(1,400,20),
                 num_class=2, 
                 loss=[SparseCategoricalCrossentropy()],
                 loss_names=['crossentropy'],
                 loss_weights=[1.0],
                 model_name='EEGNet',
                 data_format='channels_first',
                 **kwargs):
        super().__init__(num_class, loss, loss_names, loss_weights, optimizer, data_format, **kwargs)
        self.input_shape = input_shape
        self.num_class = num_class
        self.model_name = model_name
        self._config(**kwargs)
        
    def _config(self, **kwargs):
        self.kernLength = 200 
        self.F1 = 8
        self.D = 2
        self.F2 = int(self.F1*self.D)
        self.norm_rate = 0.25
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
        print("This model use the kernel length of: ", self.kernLength)
        '''
        build a model and return as tf.keras.models.Model
        '''
        input1       = layers.Input(shape=self.input_shape)

        ##################################################################
        block1       = layers.Conv2D(self.F1, (1, self.kernLength), padding='same',
                              input_shape=self.input_shape,
                              use_bias=False)(input1)
        block1       = layers.BatchNormalization()(block1)
        block1       = layers.DepthwiseConv2D((self.Chans, 1), use_bias=False,
                                       depth_multiplier=self.D,
                                       depthwise_constraint=max_norm(1.))(block1)
        block1       = layers.BatchNormalization()(block1)
        block1       = layers.Activation('elu')(block1)
        block1       = layers.AveragePooling2D((1, 4))(block1)
        block1       = layers.Dropout(self.dropout_rate)(block1)

        block2       = layers.SeparableConv2D(self.F2, (1, self.kernLength//4),
                                       use_bias=False, padding='same')(block1)
        block2       = layers.BatchNormalization()(block2)
        block2       = layers.Activation('elu')(block2)
        block2       = layers.AveragePooling2D((1, 8))(block2)
        block2       = layers.Dropout(self.dropout_rate)(block2)

        flatten      = layers.Flatten(name='flatten')(block2)

        dense        = layers.Dense(self.num_class, name='dense',
                             kernel_constraint=max_norm(self.norm_rate))(flatten)
        softmax      = layers.Activation('softmax', name='softmax')(dense)
        
        model = Model(inputs=input1, outputs=softmax, name=self.model_name)
        
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