from mixnet.models import *
from mixnet.loss import *
from mixnet.utils import dotdict
from tensorflow.keras.optimizers import Adam
from os import path

def get_params(dataset, train_type, data_type, num_class, margin, n_component, warmup, latent_dim=None, adaptive_gradient=True, policy=None, loss_weights=None, log_dir='logs', **kwargs):
    
    model_name  = 'MixNet'
    n_subjects   = 9  if dataset == 'BCIC2a' else \
                   9  if dataset == 'BCIC2b' else \
                   12 if dataset == 'BNCI2015_001' else \
                   14 if dataset == 'SMR_BCI' else \
                   14 if dataset == 'HighGamma' else \
                   54 if dataset == 'OpenBMI' else 0
                   
    time_points = 400
    freq_bands = [[4, 8], [8, 12], [12, 16], 
            [16, 20], [20, 24], [24, 28], 
            [28, 32], [32, 36], [36, 40]]
    n_freq_bands = len(freq_bands)
    
    if data_type == 'spectral_spatial_signals' :
    	pass
    else:
    	raise ValueError('The data type is not correct for MixNet')
    
    input_shape = (1, time_points, int(n_component*n_freq_bands)) # The input_shape for all datasets
    
    latent_dim   = input_shape[2] if latent_dim == None else latent_dim
    log_path = '{}/{}/{}_{}_classes_{}_{}_csp_components_{}_margin_{}_latent_dim_{}_warmup_{}'.format(log_dir, 
                                            model_name,           
                                            train_type, 
                                            num_class, 
                                            dataset, 
                                            policy if adaptive_gradient==True else str(loss_weights), 
                                            n_component,
                                            margin,
                                            latent_dim,
                                            warmup)
    
    loss_weights = [1.0, 1.0, 1.0] if adaptive_gradient == True else loss_weights
    
    if train_type == 'subject_dependent':
        factor = 0.5
        es_patience = 20 
        lr = 0.001
        min_lr = 0.0001
        batch_size = 32  
        patience = 5
        epochs = 200
        min_epochs = 0
    elif train_type == 'subject_independent':
        factor = 0.5
        es_patience = 20  
        lr = 0.001
        min_lr = 0.0001
        batch_size = 100 
        patience = 5
        epochs = 200
        min_epochs = 0
        
    params = dotdict({
                'model': MixNet,
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
                        'warmup_epoch': warmup,
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
                        'data_type': data_type,
                        'dataset_path': 'datasets',
                        'n_subjects': n_subjects,
                        'n_folds': 5,
                        'load_path': 'datasets/{}/{}/{}_class'.format(dataset, data_type, num_class)
                }), 
                
                'log_path': log_path
            }) 
    
    return params
  