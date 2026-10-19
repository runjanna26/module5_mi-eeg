import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import classification_report, f1_score, confusion_matrix, f1_score, recall_score, accuracy_score
from mixnet.trainer import Trainer
from abc import abstractmethod

class BaseModel(Trainer):
    
    def __init__(self, num_class, loss, loss_names, loss_weights, optimizer, data_format, **kwargs): 
        super().__init__(loss, loss_names, loss_weights, optimizer, data_format, **kwargs)
        self.num_class = num_class
        self.f1_average = 'binary' if self.num_class == 2 else 'macro'
        self.print_summary = True
        
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
    
    @abstractmethod
    def _config(self):
        '''
        config params for the model
        '''
        pass
    
    @abstractmethod
    def build(self, print_summary=True, load_weights=False):
        '''
        build a model and return as tf.keras.models.Model
        '''
        pass
        
    def fit(self, X_train, y_train, X_val, y_val):

        # if X_train.ndim != 4:
        #     raise Exception('ValueError: `X_train` is incompatible: expected ndim=4, found ndim='+str(X_train.ndim))
        # elif X_val.ndim != 4:
        #     raise Exception('ValueError: `X_val` is incompatible: expected ndim=4, found ndim='+str(X_val.ndim))
        
        model = self.build(print_summary=self.print_summary)
        super().compile(model=model)
        super().training(x=X_train, y=y_train, validation_data=(X_val, y_val))
        
    def evaluate(self, X_test, y_test):
        # if X_test.ndim != 4:
        #     raise Exception('ValueError: `X_test` is incompatible: expected ndim=4, found ndim='+str(X_test.ndim))
        
        model = self.build(print_summary=self.print_summary, load_weights=True)
        super().compile(model=model) 
        start = time.time()
        evaluation, test_pred = super().testing(x=X_test, y=y_test)
        end = time.time()
        if model.name.startswith('MIN2Net'):
            try:
                y_pred_decoder, zs, y_pred_clf = test_pred[0], test_pred[1:-1], test_pred[-1]
                zs = zs[0] if len(zs)==1 else np.array(zs)
                y_pred_argm = np.argmax(y_pred_clf, axis=1)
                Y = {'y_true': y_test, 'y_pred': y_pred_argm, 'y_pred_clf':y_pred_clf, 'latent': zs, 'y_pred_decoder': y_pred_decoder}
            except:
                y_pred_decoder, y_pred_clf = test_pred[0], test_pred[1]
                y_pred_argm = np.argmax(y_pred_clf, axis=1)
                Y = {'y_true': y_test, 'y_pred': y_pred_argm, 'y_pred_clf':y_pred_clf, 'y_pred_decoder': y_pred_decoder}
                
        elif model.name.startswith('MixNet'):
            y_pred_decoder, zs, y_pred_clf = test_pred[0], test_pred[1:-1], test_pred[-1]
            zs = zs[0] if len(zs)==1 else np.array(zs)
            y_pred_argm = np.argmax(y_pred_clf, axis=1)
            Y = {'y_true': y_test, 'y_pred': y_pred_argm, 'y_pred_clf':y_pred_clf, 'latent': zs, 'y_pred_decoder': y_pred_decoder}  
        else:
            y_pred_argm = np.argmax(test_pred, axis=1)
            Y = {'y_true': y_test, 'y_pred': y_pred_argm, 'y_pred_clf':test_pred}
        
        mem_usage = tf.config.experimental.get_memory_info('GPU:0')['current'] 
        print("Checking average current GPU memory usage", mem_usage)    
        print("F1-score is computed based on {}".format(self.f1_average))
        f1 = f1_score(y_test, y_pred_argm, average=self.f1_average)
        print(classification_report(y_test, y_pred_argm))
        evaluation.update({'f1-score': f1, 'prediction_time': end-start, 'memory_usage': mem_usage})
        evaluation.update(dict(zip(['w_'+name+'_loss' for name in self.loss_names], self.best_loss_weights.numpy())))   
        return Y, evaluation


    def predict(self, X_test, y_test):
        # if X_test.ndim != 4:
        #     raise Exception('ValueError: `X_test` is incompatible: expected ndim=4, found ndim='+str(X_test.ndim))
        
        model = self.build(print_summary=self.print_summary, load_weights=True)
        super().compile(model=model)
        start = time.time()
        test_pred = super().prediction(x=X_test)
        end = time.time()
        if model.name.startswith('MIN2Net'):
            try:
                y_pred_decoder, zs, y_pred_clf = test_pred[0], test_pred[1:-1], test_pred[-1]
                zs = zs[0] if len(zs)==1 else np.array(zs)
                y_pred_argm = np.argmax(y_pred_clf, axis=1)
                Y = {'y_true': y_test, 'y_pred': y_pred_argm, 'y_pred_clf':y_pred_clf, 'latent': zs, 'y_pred_decoder': y_pred_decoder}
            except:
                y_pred_decoder, y_pred_clf = test_pred[0], test_pred[1]
                y_pred_argm = np.argmax(y_pred_clf, axis=1)
                Y = {'y_true': y_test, 'y_pred': y_pred_argm, 'y_pred_clf':y_pred_clf, 'y_pred_decoder': y_pred_decoder}
                    
        elif model.name.startswith('MixNet'):
            y_pred_decoder, zs, y_pred_clf = test_pred[0], test_pred[1:-1], test_pred[-1]
            zs = zs[0] if len(zs)==1 else np.array(zs)
            y_pred_argm = np.argmax(y_pred_clf, axis=1)
            Y = {'y_true': y_test, 'y_pred': y_pred_argm, 'y_pred_clf':y_pred_clf, 'latent': zs, 'y_pred_decoder': y_pred_decoder}    
        else:
            y_pred_argm = np.argmax(test_pred, axis=1)
            Y = {'y_true': y_test, 'y_pred': y_pred_argm, 'y_pred_clf':test_pred}
            
        return Y
    
    def sen_spec(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        total = tn + fp + fn + tp 
        accuracy = (tn+tp)/total
        recall = recall_score(y_true, y_pred, pos_label = 1, average="binary")
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        f1_score_binary = 2*tp/(2*tp+fp+fn)
        f1_score_macro = f1_score(y_true, y_pred, average='macro')
        f1_score_weighted = f1_score(y_true, y_pred, average='weighted')
        print("Verifying sensitivity {} and recall {}".format(sensitivity, recall))
        return sensitivity, specificity, f1_score_binary, f1_score_macro, f1_score_weighted
    