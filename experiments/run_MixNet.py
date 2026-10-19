import tensorflow as tf
import numpy as np
import os
import argparse
from mixnet.utils import write_log, DataLoader, str2bool
from configs import exp_config
'''
In case of evaluating classification performance on a particular dataset, please run the script with its optimal setting as examples:

Subject-dependent manner:
1. python run_MixNet.py --model_name 'MixNet' --dataset 'HighGamma' --train_type 'subject_dependent' --data_type 'spectral_spatial_signals' --adaptive_gradient True --policy 'HistoricalTangentSlope' --log_dir 'logs' --num_class 2 --GPU 0 --margin 1.0 --n_component 6 --warmup 5 
2. python run_MixNet.py --model_name 'MixNet' --dataset 'BCIC2a' --train_type 'subject_dependent' --data_type 'spectral_spatial_signals' --adaptive_gradient True --policy 'HistoricalTangentSlope' --log_dir 'logs' --num_class 2 --GPU 0 --margin 1.0 --n_component 2 --warmup 7
3. python run_MixNet.py --model_name 'MixNet' --dataset 'BCIC2b' --train_type 'subject_dependent' --data_type 'spectral_spatial_signals' --adaptive_gradient True --policy 'HistoricalTangentSlope' --log_dir 'logs' --num_class 2 --GPU 0 --margin 0.1 --latent_dim 4 --n_component 2 --warmup 7
4. python run_MixNet.py --model_name 'MixNet' --dataset 'BNCI2015_001' --train_type 'subject_dependent' --data_type 'spectral_spatial_signals' --adaptive_gradient True --policy 'HistoricalTangentSlope' --log_dir 'logs' --num_class 2 --GPU 0 --margin 10.0 --n_component 4 --warmup 2
5. python run_MixNet.py --model_name 'MixNet' --dataset 'SMR_BCI' --train_type 'subject_dependent' --data_type 'spectral_spatial_signals' --adaptive_gradient True --policy 'HistoricalTangentSlope' --log_dir 'logs' --num_class 2 --GPU 0 --margin 0.5 --n_component 2 --warmup 2
6. python run_MixNet.py --model_name 'MixNet' --dataset 'OpenBMI' --train_type 'subject_dependent' --data_type 'spectral_spatial_signals' --adaptive_gradient True --policy 'HistoricalTangentSlope' --log_dir 'logs' --num_class 2 --GPU 0 --margin 5.0 --n_component 4 --warmup 5

Subject-independent manner:
1. python run_MixNet.py --model_name 'MixNet' --dataset 'HighGamma' --train_type 'subject_independent' --data_type 'spectral_spatial_signals' --adaptive_gradient True --policy 'HistoricalTangentSlope' --log_dir 'logs' --num_class 2 --GPU 0 --margin 1.0 --n_component 10 --warmup 5 
2. python run_MixNet.py --model_name 'MixNet' --dataset 'BCIC2a' --train_type 'subject_independent' --data_type 'spectral_spatial_signals' --adaptive_gradient True --policy 'HistoricalTangentSlope' --log_dir 'logs' --num_class 2 --GPU 0 --margin 0.1 --n_component 4 --latent_dim 128 --warmup 5
3. python run_MixNet.py --model_name 'MixNet' --dataset 'BCIC2b' --train_type 'subject_independent' --data_type 'spectral_spatial_signals' --adaptive_gradient True --policy 'HistoricalTangentSlope' --log_dir 'logs' --num_class 2 --GPU 0 --margin 5.0 --latent_dim 18 --n_component 2 --warmup 3
4. python run_MixNet.py --model_name 'MixNet' --dataset 'BNCI2015_001' --train_type 'subject_independent' --data_type 'spectral_spatial_signals' --adaptive_gradient True --policy 'HistoricalTangentSlope' --log_dir 'logs' --num_class 2 --GPU 0 --margin 100.0 --latent_dim 128 --n_component 2 --warmup 5
5. python run_MixNet.py --model_name 'MixNet' --dataset 'SMR_BCI' --train_type 'subject_independent' --data_type 'spectral_spatial_signals' --adaptive_gradient True --policy 'HistoricalTangentSlope' --log_dir 'logs' --num_class 2 --GPU 0 --margin 5.0 --n_component 6 --warmup 5
6. python run_MixNet.py --model_name 'MixNet' --dataset 'OpenBMI' --train_type 'subject_independent' --data_type 'spectral_spatial_signals' --adaptive_gradient True --policy 'HistoricalTangentSlope' --log_dir 'logs' --num_class 2 --GPU 0 --margin 100.0 --n_component 2 --warmup 5

'''
                
def main(subject):
    # create an object of DataLoader
    loader = DataLoader(subject=subject, n_component=args.n_component, num_class=args.num_class, **config.data_params)

    results = []
    for fold in range(1, config.data_params.n_folds+1):

        prefix_log = 'S{:03d}_fold{:02d}'.format(subject, fold)
        model = config.model(prefix_log=prefix_log, **config.model_params)
        
        # load dataset
        X_train, y_train = loader.load_train_set(fold=fold)
        X_val, y_val = loader.load_val_set(fold=fold)
        X_test, y_test = loader.load_test_set(fold=fold)
        print("Check type of MI classes: ", np.unique(y_train))
        model.fit(X_train, y_train, X_val, y_val)
        Y, evaluation = model.evaluate(X_test, y_test)

        # logging
        csv_file = config.log_path+'/S{:03d}_all_results.csv'.format(subject)
        if fold==1:
            write_log(csv_file, data=evaluation.keys(), mode='w')
        write_log(csv_file, data=evaluation.values(), mode='a')
        results.append(Y)
        tf.keras.backend.clear_session()
        
    # writing results
    np.save(config.log_path+'/S{:03d}_prediction_results.npy'.format(subject), results)
    print('------------------------- S{:03d} Done--------------------------'.format(subject))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MixNet', help='model name')
    parser.add_argument('--dataset', type=str, default='HighGamma', help='dataset name: ex. [BCIC2a/BCIC2b/BNCI2015_001/SMR_BCI/HighGamma/OpenBMI]')
    parser.add_argument('--train_type', type=str, default='subject_dependent', help='Train type: ex. subject_dependent, subject_independent')
    parser.add_argument('--data_type', type=str, default='spectral_spatial_signals', help='Train type: ex. spectral_spatial_signals')
    parser.add_argument('--num_class', type=int, default=2, help='number of classes')
    parser.add_argument('--latent_dim', type=int, default=None, help='number of latent_dims')
    parser.add_argument('--loss_weights', nargs='+', default=None, type=float, help='loss_weights (beta): ex. [beta1,beta2,beta3,...]')
    parser.add_argument('--adaptive_gradient', type=str2bool, default=True, help='adaptive loss weights')
    parser.add_argument('--policy', type=str, default='HistoricalTangentSlope', help='adaptive gradient policy for MixNet')
    parser.add_argument('--log_dir', type=str, default='logs', help='path to save logs')
    parser.add_argument('--subjects', nargs='+', default=None, type=int, help='list of range test subject, None=all subject')
    parser.add_argument('--GPU', type=str, default='0', help='GPU ID')
    parser.add_argument('--margin', type=float, default=1.0, help='margin (alpha)')
    parser.add_argument('--n_component', type=int, default=None, help='number of CSP components used')
    parser.add_argument('--warmup', type=int, default=5, help='warm-up period')
    args = parser.parse_args()
    
    # Specify GPU used
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    print(tf.config.experimental.list_physical_devices('GPU'))

    exp_setup = {'dataset': args.dataset,
                 'train_type': args.train_type, 
                 'data_type': args.data_type,
                 'num_class': args.num_class,
                 'margin': args.margin,
                 'n_component': args.n_component,
                 'warmup':args.warmup,
                 'latent_dim': args.latent_dim,
                 'loss_weights': args.loss_weights,
                 'adaptive_gradient': args.adaptive_gradient,
                 'policy': args.policy,
                 'log_dir': args.log_dir}
    config = exp_config.get_params(args.model_name, **exp_setup)
    print(config)                                
    for directory in [config.log_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    if args.subjects == None: #loop to train all subjects
        for subject in range(1, config.data_params.n_subjects+1):
            main(subject)
    elif len(args.subjects)==2:
        for subject in range(args.subjects[0], args.subjects[1]+1):
            main(subject)
    else:
        for subject in args.subjects:
            main(subject)
