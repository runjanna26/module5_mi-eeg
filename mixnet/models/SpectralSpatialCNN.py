import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from mixnet import models
from mixnet.loss import *

class SpectralSpatialCNN(models.base.BaseModel):
    def __init__(self, 
                 optimizer,
                 input_shape=(28,28,1),
                 num_class=2, 
                 loss=[SparseCategoricalCrossentropy()],
                 loss_names=['crossentropy'],
                 loss_weights=[1.0],
                 model_name='SpectralSpatialCNN',
                 data_format='channels_last',
                 **kwargs):
        super().__init__(num_class, loss, loss_names, loss_weights, optimizer, data_format, **kwargs)
        self.input_shape = input_shape
        self.num_class = num_class
        self.model_name = model_name
        self._config(**kwargs)
        
    def _config(self, **kwargs):    
        self.n_subbands = 20
        self.dropout_rate = 0.5 
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
                
    def __2dcnn_backbone(self, model_input): 
        # 2DCNN backbone
        model = layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu', padding="same")(model_input) 
        model = layers.Conv2D(filters=14, kernel_size=(3, 3), activation='relu', padding="same")(model)
        model = layers.Conv2D(filters=18, kernel_size=(3, 3), activation='relu', padding="same")(model)
        model = layers.Flatten()(model)
        model = layers.Dense(256)(model)
        return model

    def build(self, print_summary=True, load_weights=False): 
        '''
        build a model and return as tf.keras.models.Model
        '''
        input_data        = [layers.Input(shape=self.input_shape) for i in range(self.n_subbands)]
        model_subband     = [self.__2dcnn_backbone(input_data[i]) for i in range(self.n_subbands)] 
        out_model_subband = layers.Concatenate()([model_subband[i] for i in range(self.n_subbands)])
        fc                = layers.Dense(1024)(out_model_subband)
        fc                = layers.Dropout(self.dropout_rate)(fc)
        fc                = layers.Dense(self.num_class)(fc)
        softmax           = layers.Activation('softmax', name='softmax')(fc)
        
        model             = Model(inputs=input_data, outputs=softmax, name=self.model_name)
        
        if print_summary:
            model.summary()
        if load_weights:
            print('loading weights from', self.weights_dir)
            model.load_weights(self.weights_dir)
        return model
    
    @tf.function
    def train_step(self, x, y, loss_weights):
        with tf.GradientTape() as tape:
            y_logis = self.model([x[:,i,:,:,:] for i in range(self.n_subbands)], training=True)
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
        y_logis = self.model([x[:,i,:,:,:] for i in range(self.n_subbands)], training=False)
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
        y_logis = self.model([x[:,i,:,:,:] for i in range(self.n_subbands)], training=False)
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
        y_logis = self.model([x[:,i,:,:,:] for i in range(self.n_subbands)], training=False)
        return y_logis
   