import importlib
import os

def get_params(model_name='MyTest', **kwargs):
    '''return model parameters
    **kwargs = {'dataset': dataset, 
                 'train_type': train_type, 
                 'classes': classes,
                 'loss_weights': loss_weights,
                 'adaptive_gradient': adaptive_gradient,
                 'policy': policy,
                 'log_dir': log_dir}
    '''
    path =  os.path.join(os.path.dirname(__file__), model_name+'.py')
    print('... loading config params from:', path)
    spec = importlib.util.spec_from_file_location('configs.'+model_name, path)
    Module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Module)
    return Module.get_params(**kwargs)