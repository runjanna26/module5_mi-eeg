import numpy as np
import os
from mixnet.utils import write_log, DataLoader
import argparse
from configs.FBCSP_SVM_config import config
from mixnet.models import SVM
from mixnet.utils import write_log, DataLoader, str2bool

'''
In case of evaluating classification performance on a particular dataset, please run the script as an example: 

python run_FBCSP_SVM.py --model_name 'FBCSP_SVM' --dataset 'BCIC2a' --train_type 'subject_dependent' --data_type 'fbcsp' --num_class 2  --num_chs 20 

'''

def main(subject):
    # create object of DataLoader
    loader = DataLoader(dataset=args.dataset, train_type=args.train_type, 
                        subject=subject, data_type=args.data_type, dataset_path=dataset_path, n_class=args.num_class)

    results = []
    for fold in range(1, n_folds+1):
        model_name = args.model_name + '_'+'S{:03d}_fold{:02d}'.format(subject, fold)
        svm = SVM(log_path=log_path, 
                model_name=model_name,
                num_class=args.num_class, 
                tuned_parameters=tuned_parameters)
                  
        # load dataset
        # X = (#trial, #feature)
        # Y = (#trial)
        X_train, y_train = loader.load_train_set(fold=fold)
        X_val, y_val = loader.load_val_set(fold=fold)
        X_test, y_test = loader.load_test_set(fold=fold)
        print('Check the import dimension of Tr {} Val {} and Te {}'.format(X_train.shape, X_val.shape, X_test.shape))
        print('Check the number of classes: Tr {} Val {} and Te {}'.format(np.unique(y_train), np.unique(y_val), np.unique(y_test)))
        
        # train and test using SVM
        svm.fit(X_train, y_train, X_val, y_val)
        Y, evaluation = svm.evaluate(X_test, y_test)
        csv_file = log_path+'/S{:03d}_all_results.csv'.format(subject)

        # logging
        if fold==1:
            write_log(csv_file, data=evaluation.keys(), mode='w')
        write_log(csv_file, data=evaluation.values(), mode='a')
        results.append(Y)
        
    # writing file
    np.save(log_path+'/S{:03d}_prediction_results.npy'.format(subject), results)
    print('------------------------- S{:03d} Done--------------------------'.format(subject))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='FBCSP_SVM', help='model name')
    parser.add_argument('--dataset', type=str, default='HighGamma', help='dataset name: ex. [BCIC2a/BCIC2b/BNCI2015_001/SMR_BCI/HighGamma/OpenBMI]')
    parser.add_argument('--train_type', type=str, default='subject_dependent', help='Train type: ex. subject_dependent, subject_independent')
    parser.add_argument('--data_type', type=str, default='fbcsp', help='Train type: ex. fbcsp')
    parser.add_argument('--num_class', type=int, default=2, help='number of classes')
    parser.add_argument('--num_chs', type=int, default=20, help='number of classes')
    parser.add_argument('--log_dir', type=str, default='logs', help='path to save logs')
    parser.add_argument('--subjects', nargs='+', default=None, type=int, help='list of range test subject, None=all subject')
    args = parser.parse_args()

    # load config params from config.py
    CONSTANT = config['FBCSP_SVM']
    n_folds = CONSTANT['n_folds']
    dataset_path = CONSTANT['dataset_path']
    tuned_parameters = CONSTANT['tuned_parameters']
    
    # The below log_path use 
    log_path = '{}/{}/{}_{}_classes_{}'.format(args.log_dir, 
                                                    args.model_name, 
                                                     args.train_type, 
                                                    str(args.num_class), 
                                                    args.dataset)
    for directory in [log_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    if args.subjects == None: #loop to train all subjects
        for subject in range(1, CONSTANT[args.dataset]['n_subjects']+1):
            main(subject)
    elif len(args.subjects)==2:
        for subject in range(args.subjects[0], args.subjects[1]+1):
            main(subject)
    else:
        for subject in args.subjects:
            main(subject)
