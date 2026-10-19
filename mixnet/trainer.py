import tensorflow as tf
from mixnet import callbacks 
from mixnet.loss import SparseCategoricalCrossentropy
from mixnet.utils import compute_class_weight, dotdict
from mixnet.gradients import GradientBlending
import math
import os
import copy
import numpy as np
from abc import abstractmethod

class Trainer(object):
    
    def __init__(self,
                 loss, 
                 loss_names,
                 loss_weights, 
                 optimizer,
                 data_format,
                 metrics='accuracy',
                 adaptive_gradient=False, 
                 policy=None,
                 warmup_epoch=5,
                 epochs=200,
                 min_epochs=0,
                 batch_size=100,
                 lr=1e-2,
                 min_lr=1e-3,
                 factor=0.5,
                 patience=5,
                 es_patience=20,
                 monitor='val_loss',
                 mode='min',
                 save_best_only=True,
                 save_weight_only=True,
                 shuffle=True,
                 seed=1234,
                 verbose=1,
                 log_path='logs',
                 prefix_log='model',
                 **kwargs):    
        self.loss              = loss
        self.loss_names        = loss_names
        self.loss_weights      = loss_weights
        self.optimizer         = optimizer
        self.metrics           = metrics
        self.adaptive_gradient = adaptive_gradient
        self.policy            = policy
        self.warmup_epoch      = warmup_epoch
        self.data_format = data_format
        self.epochs = epochs
        self.min_epochs = min_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.es_patience = es_patience
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weight_only = save_weight_only
        self.shuffle = shuffle
        self.seed = seed
        self.verbose = verbose
        self.log_path = log_path
        self.prefix_log = prefix_log
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
    #     self.__config__()
    
    # def __config__(self):
        self.optimizer.lr          = self.lr
        self.loss, self.loss_names = self._config_loss(self.loss, self.loss_names)
        self.loss_weights          = self.cast_to_list(self.loss_weights)
        self.n_losses              = len(self.loss)
        self.adaptive_masked       = self.__masked_adaptive_gradient(self.adaptive_gradient)
        self.n_adaptive            = sum(self.adaptive_masked)
        self.init_weights          = tf.constant(self.__init_loss_weights(self.loss_weights), dtype=tf.float32)
        self.loss_weights          = tf.constant(self.__init_loss_weights(self.loss_weights), dtype=tf.float32)
        self.train_acc_metric      = tf.keras.metrics.SparseCategoricalAccuracy() if self.metrics == 'accuracy' else self.metrics
        self.val_acc_metric        = tf.keras.metrics.SparseCategoricalAccuracy() if self.metrics == 'accuracy' else self.metrics
        self.test_acc_metric       = tf.keras.metrics.SparseCategoricalAccuracy() if self.metrics == 'accuracy' else self.metrics
        self.pred_acc_metric       = tf.keras.metrics.SparseCategoricalAccuracy() if self.metrics == 'accuracy' else self.metrics
        self.gradient              = GradientBlending(n=self.n_losses,
                                                      init_weights=self.__init_loss_weights(self.loss_weights),
                                                      adaptive_masked=self.adaptive_masked)
        self.min_epochs            = self.min_epochs if isinstance(self.min_epochs, int) \
                                     else int(self.min_epochs*self.epochs) if isinstance(self.min_epochs, float) \
                                     else 0 
        self.best_loss_weights     = None
        self.best_epoch            = 0
        self.weights_dir           = os.path.join(self.log_path, self.prefix_log+'_out_weights.h5')
        self.csv_dir               = os.path.join(self.log_path, self.prefix_log+'_out_log.csv')
        self.time_log              = os.path.join(self.log_path, self.prefix_log+'_time_log.csv')
            
        self.csv_logger             = tf.keras.callbacks.CSVLogger(self.csv_dir)
        self.checkpointer           = tf.keras.callbacks.ModelCheckpoint(monitor=self.monitor, filepath=self.weights_dir, 
                                        verbose=self.verbose, save_best_only=self.save_best_only, 
                                        save_weight_only=self.save_weight_only)
        self.time_callback          = callbacks.TimeHistory(self.time_log)
        self.reduce_lr              = callbacks.ReduceLROnPlateau(monitor=self.monitor, patience=self.patience, 
                                        factor=self.factor, mode=self.mode, verbose=self.verbose, 
                                        min_lr=self.min_lr,lr=self.optimizer.lr) 
        tf.keras.backend.set_image_data_format(self.data_format)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        '''seed all'''
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
    
    def compile(self, model):
        self.model = model
        
    def load_weights(self, weight_dir):
        print('.... loading weights from', self.weights_dir)
        self.model.load_weights(weight_dir)
        
    def __masked_adaptive_gradient(self, adaptive_gradient):
        '''masked a boolen for each loss 
        Ex. adaptive_gradient = True --> return [True, True, True]
        Ex. adaptive_gradient = False or None --> return [False, False, False]
        Ex. adaptive_gradient = [True, False, False] --> return [True, False, False]
        '''
        return [False]*self.n_losses if not adaptive_gradient \
            else [True]*self.n_losses if adaptive_gradient==True \
            else adaptive_gradient
        
    def __init_loss_weights(self, loss_weights):
        '''init loss_weights [1./n, 1./n, ..., ] or [1., 1, ...., ] if loss_weights is None'''
        return [1./self.n_adaptive if self.adaptive_masked[i] 
                else 1. for i in range(self.n_losses)] if loss_weights is None else loss_weights
    
    def _config_loss(self, loss, loss_names):
        '''
        config loss an loss_names
        returns:
            loss: as dotdict
            loss_names: as list of str
        '''
        def __inier_setup__(loss_obj):
            try:
                name = loss_obj.__name__
            except:
                try:
                    name = loss_obj.name
                except:
                    name = loss_obj
            if name in ['mse', 'MSE' ,'MeanSquaredError', 'mean_squared_error', 'inner_mse']:
                name = 'mse'
            if name in ['clf', 'SparseCategoricalCrossentropy' ,'sparse_categorical_crossentropy', 'inner_scce']:
                name = 'crossentropy'
            if name in ['trp', 'triplet', 'triplet_loss', 'inner_triplet']:
                name = 'triplet'
            return name
        
        if not isinstance(loss, dotdict):    
            loss = self.cast_to_list(loss)
            loss_names =  [__inier_setup__(obj) for obj in loss] if loss_names is None else loss_names
            loss = dotdict(dict(zip(loss_names, loss)))
        
        return loss, loss_names
    
    @staticmethod 
    def cast_to_list(obj):
        return obj if isinstance(obj, list) else [obj] # cast to list
    
    @staticmethod   
    def mean_dict(list_of_dict):
        '''
        average list of a dict
        return:
            dict
        '''
        result = {}
        for key in list_of_dict[0].keys():
            tmp = []
            for d in list_of_dict:
                try:
                    tmp.append(tf.reduce_mean(d[key]))
                except:
                    tmp.append(d[key])
            result[key] =  tf.reduce_mean(tmp)
        return result

    @staticmethod
    def print_report(dict_logs):
        print(''.join(' - {}: {:.4f}'.format(k, v) for k,v in dict_logs.items()))
    
    @tf.function
    @abstractmethod
    def train_step(self, x, y, loss_weights=None):
        '''
        custom train_step function
        '''
        pass
    
    @tf.function
    @abstractmethod
    def val_step(self, x, y, loss_weights=None):
        '''
        custom val_step function
        '''
        pass
    
    @tf.function
    @abstractmethod
    def test_step(self, x, y, loss_weights=None):
        '''
        custom test_step function
        '''
        pass
    
    @tf.function
    @abstractmethod
    def pred_step(self, x):
        '''
        custom pred_step function
        '''
        pass
  
    
    def training(self, x, y, validation_data):
        '''
        training loop 
        '''
        if self.min_epochs > self.epochs:
            raise Exception('ValueError: Found min_epochs > epochs')
        
        x_val, y_val     = validation_data
        x                = tuple(x) if type(x) == list else x
        y                = tuple(y) if type(y) == list else y
        x_val            = tuple(x_val) if type(x_val) == list else x_val
        y_val            = tuple(y_val) if type(y_val) == list else y_val
        

        try: # tf >= v2.3.*
            self.callbacks   = tf.keras.callbacks.CallbackList([self.checkpointer,self.csv_logger,self.reduce_lr,self.time_callback],
                                                                model=self.model,
                                                                add_progbar=self.verbose!=0,
                                                                verbose=self.verbose,
                                                                epochs=self.epochs)
        except: # tf <= v2.2.*
            self.callbacks   = tf.python.keras.callbacks.CallbackList([self.checkpointer,self.csv_logger,self.reduce_lr,self.time_callback],
                                                                model=self.model,
                                                                add_progbar=self.verbose!=0,
                                                                verbose=self.verbose,
                                                                epochs=self.epochs)
        
        if self.class_balancing: # compute_class_weight if class_balancing is True
            class_weight  = compute_class_weight(y)
            print("This iteration is taking into account of class weight: ", class_weight)
            try:
                self.loss.crossentropy = SparseCategoricalCrossentropy(class_weight=class_weight)
            except:
                self.loss[self.loss_names.index('crossentropy')] = SparseCategoricalCrossentropy(class_weight=class_weight)
        print("This iteration is taking into account of batch size: ", self.batch_size)
        self.n_batch   = math.ceil(len(x)/self.batch_size)
        val_batch_size = len(x_val)//self.n_batch

        if self.shuffle:
            train_dataset = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(len(x), seed=self.seed).batch(self.batch_size)
            val_dataset   = tf.data.Dataset.from_tensor_slices((x_val,y_val)).shuffle(len(x_val), seed=self.seed).batch(val_batch_size)
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((x,y)).batch(self.batch_size)
            val_dataset   = tf.data.Dataset.from_tensor_slices((x_val,y_val)).batch(val_batch_size)
    
        if self.adaptive_gradient:
            self.gradient.build(policy=self.policy, batch_size=self.batch_size, valid_batch_size=val_batch_size)
        
        self.callbacks.on_train_begin()
        loss_weights = self.init_weights
        global_step  = 0
        for epoch in range(self.epochs):

            self.callbacks.on_epoch_begin(epoch)
            ########################### training ###########################
            train_logs = []
            val_logs = []
            for step, ((x_batch_train, y_batch_train), (x_batch_val, y_batch_val)) in enumerate(zip(train_dataset, val_dataset)):
                self.callbacks.on_train_batch_begin(step)
                tmp_logs, _ = self.train_step(x_batch_train, y_batch_train, loss_weights)
                train_logs.append(tmp_logs)
                self.callbacks.on_train_batch_end(step, tmp_logs)
                
                tmp_val_logs, _ = self.val_step(x_batch_val, y_batch_val, loss_weights)
                
                if self.adaptive_gradient:
                    train_losses = [tmp_logs['train_'+name+'_loss'].numpy() for name in self.loss_names]
                    val_losses   = [tmp_val_logs['val_'+name+'_loss'].numpy() for name in self.loss_names]
                    self.gradient.add_point_train_loss(*train_losses)
                    self.gradient.add_point_valid_loss(*val_losses)
                    ###################### compute adaptive weight ######################
                    
                    # if (global_step % self.blending_step)==0 and (epoch >= self.warmup_epoch):
                    if epoch >= self.warmup_epoch:
                        new_loss_weights = self.gradient.compute_adaptive_weight(to_tensor=True)
                        loss_weights     = copy.copy(new_loss_weights)      
                tmp_val_logs.update(dict({'w_'+name: w for name, w in zip(self.loss_names, loss_weights)}))   
                val_logs.append(tmp_val_logs)
                global_step+=1
                 

            #################
            train_mean_logs = self.mean_dict(train_logs)
            epoch_logs      = copy.copy(train_mean_logs)
            val_mean_logs   = self.mean_dict(val_logs)
            epoch_logs.update(val_mean_logs)
            self.callbacks.on_epoch_end(epoch, epoch_logs)
            self.train_acc_metric.reset_states()
            self.val_acc_metric.reset_states()
            
            ########################### checkpoint ##########################
            # init save
            if epoch == 0:
                self.model.save_weights(self.checkpointer.filepath)
            # update learning_rate
            self.optimizer.lr = self.reduce_lr.lr
            # early stopping
            if self.checkpointer.best < float(val_mean_logs[self.monitor]):
                    count_no_improve += 1
                    if count_no_improve >= self.es_patience and epoch>=self.min_epochs:
                        self.loss_weights = loss_weights
                        print('Epoch {:05d}: early stopping'.format(epoch+1))
                        print('Best epoch {:05d}: Best loss_weights: {}'.format(best_epoch+1, self.best_loss_weights.numpy()))
                        self.callbacks.on_train_end()
                        break
            else:
                count_no_improve       = 0
                best_epoch             = epoch
                self.best_loss_weights = copy.copy(loss_weights)
        
        ############################ All Epochs Done ########################## 
        self.loss_weights = loss_weights
        self.best_loss_weights = copy.copy(loss_weights)
        self.callbacks.on_train_end()
    
    def testing(self, x, y):
        x            = tuple(x) if type(x) == list else x
        y            = tuple(y) if type(y) == list else y
        test_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(len(x))

        for x_batch_test, y_batch_test in test_dataset: # 1 batch
            test_logs, y_pred = self.test_step(x_batch_test, y_batch_test, self.loss_weights)
        self.test_acc_metric.reset_states()
        self.print_report(test_logs)
        for key in test_logs.keys():
            test_logs[key] = test_logs[key].numpy()
        return test_logs, tuple(yp.numpy() for yp in y_pred)

    def prediction(self, x):
        x            = tuple(x) if type(x) == list else x
        test_dataset = tf.data.Dataset.from_tensor_slices(x).batch(len(x))

        for x_batch_test in test_dataset: # 1 batch
            y_pred = self.pred_step(x_batch_test)
        self.pred_acc_metric.reset_states()
        return tuple(yp.numpy() for yp in y_pred)