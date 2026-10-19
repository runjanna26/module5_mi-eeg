from mixnet.models import *
from mixnet.loss import *
from mixnet.utils import dotdict
from tensorflow.keras.optimizers import Adam
from os import path
import glob

def get_params(dataset, train_type, num_class, margin, num_chs=None, adaptive_gradient=False, policy=None, loss_weights=None, log_dir='logs', **kwargs):
    
    model_name  = 'MIN2Net_original'
    n_subjects   = 9  if dataset == 'BCIC2a' else \
                   9  if dataset == 'BCIC2b' else \
                   12 if dataset == 'BNCI2015_001' else \
                   14 if dataset == 'SMR_BCI' else \
                   14 if dataset == 'HighGamma' else \
                   54 if dataset == 'OpenBMI' else 0
                   
    time_points = 400
    # Note that, the last dimension is the number of channels in each dataset
    if dataset == 'BCIC2b':
        input_shape  = (1, time_points, 3) 
    elif dataset == 'BNCI2015_001':
        input_shape  = (1, time_points, 13)
    elif dataset == 'SMR_BCI':
        input_shape  = (1, time_points, 15)  
    else:
        input_shape  = (1, time_points, 20) # The input_shape for 'BCIC2a', 'HighGamma', and 'OpenBMI' datasets    
        
    latent_dim   = input_shape[2] if num_class == 2 else 64 # n_channels or 64 (in case of three-class classification task)
    loss_weights = [1., 1., 1.] if loss_weights == None else loss_weights

    
    log_path = '{}/{}/{}_{}_classes_{}_{}_margin_{}'.format(log_dir, 
                                                    model_name, 
                                                    train_type, 
                                                    num_class, 
                                                    dataset, 
                                                    policy if adaptive_gradient==True else str(loss_weights), 
                                                    margin)
    if train_type == 'subject_dependent':
        factor = 0.5
        es_patience = 20
        lr = 0.001
        min_lr = 0.0001
        batch_size = 32 if dataset == 'HighGamma' or dataset == 'BCIC2b' or dataset == 'BNCI2015_001' else 10 # 10 for other datasets
        patience = 5
        epochs = 200
        min_epochs = 0
        
    elif train_type == 'subject_independent':
        factor = 0.5
        es_patience = 20
        lr = 0.001
        min_lr = 0.0001
        batch_size = 100 #100
        patience = 5
        epochs = 200
        min_epochs = 0
        
    params = dotdict({
                'model': MIN2Net_original,
                'model_params': dotdict({
                        'model_name': model_name, 
                        'input_shape': input_shape,
                        'latent_dim': latent_dim,
                        'class_balancing': True,
                        'f1_average': 'macro',
                        'num_class': num_class, 
                        'loss': [MeanSquaredError(), triplet_loss(margin=margin), SparseCategoricalCrossentropy()],
                        'loss_names': ['mse', 'triplet', 'crossentropy'],
                        'loss_weights': loss_weights,
                        'adaptive_gradient': adaptive_gradient,
                        'policy': policy,
                        'warmup_epoch': 5,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'optimizer': Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                        'lr': lr,
                        'min_lr': min_lr,
                        'factor': factor,
                        'patience': patience,
                        'es_patience': es_patience,
                        'min_epochs': min_epochs,
                        'verbose': 1, 
                        'log_path': log_path,
                        'data_format': 'channels_last'
                }),

                'data_params': dotdict({
                        'dataset': dataset,
                        'train_type': train_type,
                        'data_format': 'NTCD',
                        'data_type':'time_domain',
                        'dataset_path': 'datasets',
                        'n_subjects': n_subjects,
                        'n_folds': 5
                }), 
                
                'log_path': log_path
            }) 
    
    return params
  