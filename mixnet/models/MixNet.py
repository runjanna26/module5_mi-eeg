import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from mixnet import models
from mixnet.loss import *

class MixNet(models.base.BaseModel):
    
    def __init__(self, 
                 optimizer,
                 input_shape=(1,400,20),
                 latent_dim=None,
                 num_class=2, 
                 loss=[MeanSquaredError(), triplet_loss(), SparseCategoricalCrossentropy()],
                 loss_names=['mse', 'triplet', 'crossentropy'],
                 loss_weights=[1.0, 1.0, 1.0],
                 model_name='MixNet',
                 data_format='channels_last',
                 **kwargs):
        super().__init__(num_class, loss, loss_names, loss_weights, optimizer, data_format, **kwargs)
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.num_class = num_class
        self.model_name = model_name
        self._config(**kwargs)
        
    def _config(self, **kwargs):
        self.D, self.T, self.C = self.input_shape
        self.LATENT_DIM = self.latent_dim if self.latent_dim is not None else self.C if self.num_class==2 else 64
        self.sfreq = 100
        self.P1 = (1,self.T//self.sfreq)
        self.P2 = (1,4) # MI Task
        self.F1 = self.C
        self.F2 = self.C//2
        self.FLAT = self.T//self.P1[1]//self.P2[1]
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
            
    def build(self, print_summary=True, load_weights=False):
        '''
        build a model and return as tf.keras.models.Model
        '''
        'encoder'
        encoder_input  = layers.Input(self.input_shape, name='en_input')
        en_conv        = layers.Conv2D(self.F1, (1, 64), activation='elu', padding='same', 
                                kernel_constraint=max_norm(2., axis=(0, 1, 2)), name='en_conv1')(encoder_input)
        en_conv        = layers.BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1, name='en_bn1')(en_conv)
        en_conv        = layers.AveragePooling2D(pool_size=self.P1, name='en_avg1')(en_conv)  
        en_conv        = layers.Conv2D(self.F2, (1, 32), activation='elu', padding='same', 
                                kernel_constraint=max_norm(2., axis=(0, 1, 2)), name='en_conv2')(en_conv)
        en_conv        = layers.BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1, name='en_bn2')(en_conv)
        en_conv        = layers.AveragePooling2D(pool_size=self.P2, name='en_avg2')(en_conv)
        en_conv        = layers.Flatten(name='en_flat')(en_conv)
        z              = layers.Dense(self.LATENT_DIM, kernel_constraint=max_norm(0.5), name='z')(en_conv)
        encoder        = Model(inputs=encoder_input, outputs=z , name='encoder')
        if print_summary:
            encoder.summary()
        
        'decoder'
        decoder_input  = layers.Input(shape=(self.LATENT_DIM,), name='de_input')
        de_conv        = layers.Dense(1*self.FLAT*self.F2, activation='elu', 
                               kernel_constraint=max_norm(0.5), name='de_dense')(decoder_input)
        de_conv        = layers.Reshape((1, self.FLAT, self.F2), name='de_reshape')(de_conv)
        de_conv        = layers.Conv2DTranspose(filters=self.F2, kernel_size=(1, 64), 
                                         activation='elu', padding='same', strides=self.P2, 
                                         kernel_constraint=max_norm(2., axis=(0, 1, 2)), name='de_deconv1')(de_conv)
        decoder_output = layers.Conv2DTranspose(filters=self.F1, kernel_size=(1, 32), 
                                         activation='elu', padding='same', strides=self.P1, 
                                         kernel_constraint=max_norm(2., axis=(0, 1, 2)), name='de_deconv2')(de_conv)
        decoder        = Model(inputs=decoder_input, outputs=decoder_output, name='decoder')
        if print_summary:
            decoder.summary()

        'Build the computation graph for training'
        z              = encoder(encoder_input)
        xr             = decoder(z)
        y              = layers.Dense(self.num_class, activation='softmax', kernel_constraint=max_norm(0.5), name='classifier')(z)
        model          = Model(inputs=encoder_input, outputs=[xr, z, y], name=self.model_name)
        if print_summary:
            model.summary()
        if load_weights:
            print('loading weights from', self.weights_dir)
            model.load_weights(self.weights_dir)
        return model
    
    @tf.function
    def train_step(self, x, y, loss_weights):
        with tf.GradientTape() as tape:
            xr, z, y_logis = self.model(x, training=True)
            mse_loss = self.loss.mse(x, xr) 
            trp_loss = self.loss.triplet(y, z)
            crossentropy_loss = self.loss.crossentropy(y, y_logis)
            losses = [mse_loss, trp_loss, crossentropy_loss]
            train_loss = tf.reduce_sum(loss_weights*losses)
            self.train_acc_metric.update_state(y, y_logis)
        grads = tape.gradient(train_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        logs = dict({'train_loss': train_loss})
        logs.update(dict(zip(['train_'+loss_name+'_loss' for loss_name in self.loss_names], losses)))
        logs.update(dict({'train_acc': self.train_acc_metric.result()}))
        return logs, (xr, z, y_logis)
    
    @tf.function
    def val_step(self, x, y, loss_weights):
        xr, z, y_logis = self.model(x, training=False)
        mse_loss = self.loss.mse(x, xr)
        trp_loss = self.loss.triplet(y, z)
        crossentropy_loss = self.loss.crossentropy(y, y_logis)
        losses = [mse_loss, trp_loss, crossentropy_loss]
        val_loss = tf.reduce_sum(loss_weights*losses)
        self.val_acc_metric.update_state(y, y_logis)
        logs = dict({'val_loss': val_loss})
        logs.update(dict(zip(['val_'+loss_name+'_loss' for loss_name in self.loss_names], losses)))
        logs.update(dict({'val_acc': self.val_acc_metric.result()}))
        return logs, (xr, z, y_logis)
    
    @tf.function
    def test_step(self, x, y, loss_weights):
        xr, z, y_logis = self.model(x, training=False)
        mse_loss = self.loss.mse(x, xr)
        trp_loss = self.loss.triplet(y, z)
        crossentropy_loss = self.loss.crossentropy(y, y_logis)
        losses = [mse_loss, trp_loss, crossentropy_loss]
        test_loss = tf.reduce_sum(loss_weights*losses)
        self.test_acc_metric.update_state(y, y_logis)
        logs = dict({'test_loss': test_loss})
        logs.update(dict(zip(['test_'+loss_name+'_loss' for loss_name in self.loss_names], losses)))
        logs.update(dict({'test_acc': self.test_acc_metric.result()}))
        return logs, (xr, z, y_logis)
    
    @tf.function
    def pred_step(self, x):
        xr, z, y_logis = self.model(x, training=False)
        return (xr, z, y_logis)