import numpy as np
import csv
import wget
import os
import sklearn
import scipy.signal
import argparse
import mne
import moabb
from moabb.paradigms import MotorImagery

# lib path
PATH = os.path.dirname(os.path.realpath(__file__))

class dotdict(dict):
    '''dot.notation access to dictionary attributes'''

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def load_raw(dataset):
    # folder_name = str(PATH)+'/datasets'
    folder_name = 'datasets'
    if dataset == 'OpenBMI':
        try:
            num_subjects = 54
            sessions = [1, 2]
            save_path = folder_name + '/' + dataset + '/raw'
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            for session in sessions:
                for person in range(1, num_subjects+1):
                    file_name = '/sess{:02d}_subj{:02d}_EEG_MI.mat'.format(session,person)
                    if os.path.exists(save_path+file_name):
                        os.remove(save_path+file_name) # if exist, remove file
                    print('\n===Download is being processed on session: {} subject: {}==='.format(session, person))
                    url = 'ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100542/session{}/s{}{}'.format(session, person, file_name)
                    print('save to: '+save_path+file_name)
                    wget.download(url,  save_path+file_name)
            print('\nDone!')
        except:
            raise Exception('Path Error: file does not exist, please directly download at http://gigadb.org/dataset/100542')
    elif dataset == 'BCIC2a':
        try:
            num_subjects = 9
            sessions = ['T', 'E']
            save_path = folder_name + '/' + dataset + '/raw'
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

            for session in sessions:
                for person in range(1, num_subjects+1):
                    file_name = '/A{:02d}{}.mat'.format(person, session)
                    if os.path.exists(save_path+file_name):
                        os.remove(save_path+file_name) # if exist, remove file
                    print('\n===Download is being processed on session: {} subject: {}==='.format(session, person))
                    url = 'https://lampx.tugraz.at/~bci/database/001-2014'+file_name
                    print('save to: '+save_path+file_name)
                    wget.download(url, save_path+file_name)
            print('\nDone!')
        except:
            raise Exception('Path Error: file does not exist, please directly download at http://bnci-horizon-2020.eu/database/data-sets')
    elif dataset == 'BCIC2b':
        try:
            num_subjects = 9
            sessions = ['T', 'E']
            save_path = folder_name + '/' + dataset + '/raw'
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

            for session in sessions:
                for person in range(1, num_subjects+1):
                    file_name = '/B{:02d}{}.mat'.format(person, session)
                    if os.path.exists(save_path+file_name):
                        os.remove(save_path+file_name) # if exist, remove file
                    print('\n===Download is being processed on session: {} subject: {}==='.format(session, person))
                    url = 'https://lampx.tugraz.at/~bci/database/004-2014'+file_name
                    print('save to: '+save_path+file_name)
                    wget.download(url, save_path+file_name)
            print('\nDone!')
        except:
            raise Exception('Path Error: file does not exist, please directly download at http://bnci-horizon-2020.eu/database/data-sets')
    elif dataset == 'SMR_BCI':
        try:
            num_subjects = 14
            sessions = ['T', 'E']
            save_path = folder_name + '/' + dataset + '/raw'
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            for session in sessions:
                for person in range(1, num_subjects+1):
                    file_name = '/S{:02d}{}.mat'.format(person, session)
                    if os.path.exists(save_path+file_name):
                        os.remove(save_path+file_name) # if exist, remove file
                    print('\n===Download is being processed on session: {} subject: {}==='.format(session, person))
                    url = 'https://lampx.tugraz.at/~bci/database/002-2014'+file_name
                    print('save to: '+save_path+file_name)
                    wget.download(url,  save_path+file_name)
            print('\nDone!')
        except:
            raise Exception('Path Error: file does not exist, please directly download at http://bnci-horizon-2020.eu/database/data-sets')
    elif dataset == 'BNCI2015_001': 
        try:
            save_path = folder_name + '/' + dataset + '/raw'
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            
            # There are MI-EEG signals from 12 subjects
            num_subjects = 12
            # Execute loops to download the MI-EEG dataset through the moabb library
            for person in range(1, num_subjects+1):
                print('\n===Download is being processed on subject: {} ==='.format(person))
                dataset = moabb.datasets.BNCI2015_001()
                paradigm = MotorImagery(n_classes=2)
                X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[person])
                
                # Define session 0A as a training set and session 0B as an evaluation set 
                train_idx = np.where(meta['session'].values == '0A')[0]
                test_idx = np.where(meta['session'].values == '1B')[0]
                X_train = X[train_idx]
                y_train = labels[train_idx]
                X_test = X[test_idx]
                y_test = labels[test_idx]
                print("Training data dimension {} and its label {}".format(X_train.shape, y_train.shape))
                print("Testing data dimension {} and its label {}".format(X_test.shape, y_test.shape))
                np.savez(save_path+'/S{:02d}.npz'.format(person),
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test, 
                        y_test=y_test)
            print('\nDone!')
        except:
            raise Exception('Path Error: file does not exist, please directly download at https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2015_001.html#moabb.datasets.BNCI2015_001')
    elif dataset == 'HighGamma': 
        try:
            save_path = folder_name + '/' + dataset + '/raw'
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            
            # There are MI-EEG signals from 14 subjects
            num_subjects = 14
            # Execute loops to download the MI-EEG dataset through the MOABB library
            for person in range(1, num_subjects+1):
                print('\n===Download is being processed on subject: {} ==='.format(person))
                dataset = moabb.datasets.Schirrmeister2017()
                paradigm = MotorImagery(n_classes=4)
                X, y, meta = paradigm.get_data(dataset=dataset, subjects=[person])
                tr_id = list(meta[meta['run']=='0train'].index)
                te_id = list(meta[meta['run']=='1test'].index)
                X_train, X_test = X[tr_id], X[te_id]
                y_train, y_test = y[tr_id], y[te_id]
                print("Training data dimension {} and its label {}".format(X_train.shape, y_train.shape))
                print("Testing data dimension {} and its label {}".format(X_test.shape, y_test.shape))
                np.savez(save_path+'/S{:02d}.npz'.format(person),
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test, 
                        y_test=y_test)
            print('\nDone!')
        except:
            raise Exception('Path Error: file does not exist, please directly download at https://moabb.neurotechx.com/docs/generated/moabb.datasets.Schirrmeister2017.html#moabb.datasets.Schirrmeister2017')
            
class DataLoader:
    def __init__(self, dataset, train_type=None, data_type=None, num_class=2, subject=None, data_format=None, dataset_path='/datasets', n_component=None, **kwargs):

        self.dataset = dataset #Dataset name: 'OpenBMI', 'SMR_BCI', 'BCIC2a'
        self.train_type = train_type # 'subject_dependent', 'subject_independent'
        self.data_type = data_type # 'fbcsp', 'spectral_spatial', 'time_domain'
        self.dataset_path = dataset_path
        self.subject = subject # id, start at 1
        self.data_format = data_format # 'channels_first', 'channels_last'
        self.fold = None # fold, start at 1
        self.prefix_name = 'S'
        self.num_class = num_class
        self.n_component = n_component
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

        if self.n_component is not None:
            self.path = self.dataset_path+'/'+self.dataset+'/'+self.data_type+'/'+str(self.num_class)+'_class/'+str(self.n_component)+'_csp_components/'+self.train_type
        else:
            self.path = self.dataset_path+'/'+self.dataset+'/'+self.data_type+'/'+str(self.num_class)+'_class/'+self.train_type
        print('The datset used is: ', self.path)
    def _change_data_format(self, X):
        if self.data_format == 'NCTD':
            # (#n_trial, #channels, #time, #depth)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        elif self.data_format == 'NDCT':
        	# (#n_trial, #depth, #channels, #time)
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        elif self.data_format == 'NTCD':
            # (#n_trial, #time, #channels, #depth)
            if X.ndim == 3:
                X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
                X = np.swapaxes(X, 1, 3)
            elif X.ndim == 4:
                X = np.swapaxes(X, 2, 3)
        elif self.data_format == 'NSHWD':
            # (#n_trial, #Freqs, #height, #width, #depth)
            X = zero_padding(X)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1)
        elif self.data_format == None:
            pass
        else:
            raise Exception('Value Error: data_format requires None, \'NCTD\', \'NDCT\', \'NTCD\' or \'NSHWD\', found data_format={}'.format(self.data_format))
        print('change data_format to \'{}\', new dimention is {}'.format(self.data_format, X.shape))
        return X

    def load_train_set(self, fold, **kwargs):
        self.fold = fold
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
    
        # load 
        X, y =  np.array([]),  np.array([])
        try:
            self.file_x = self.path+'/X_train_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            self.file_y = self.path+'/y_train_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            X = self._change_data_format(np.load(self.file_x, allow_pickle=True))
            y = np.load(self.file_y, allow_pickle=True)
        except:
            raise Exception('Path Error: file does not exist, please check this path {}, and {}'.format(self.file_x, self.file_y))
        return X, y

    def load_val_set(self, fold, **kwargs):
        self.fold = fold
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

        # load 
        X, y =  np.array([]),  np.array([])
        try:
            self.file_x = self.path+'/X_val_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            self.file_y = self.path+'/y_val_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            X = self._change_data_format(np.load(self.file_x, allow_pickle=True))
            y = np.load(self.file_y, allow_pickle=True)
        except:
            raise Exception('Path Error: file does not exist, please check this path {}, and {}'.format(self.file_x, self.file_y))
        return X, y
    
    def load_test_set(self, fold, **kwargs):
        self.fold = fold
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

        # load 
        X, y =  np.array([]),  np.array([])
        try:
            self.file_x = self.path+'/X_test_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            self.file_y = self.path+'/y_test_{}{:03d}_fold{:03d}.npy'.format(self.prefix_name, self.subject, self.fold)
            X = self._change_data_format(np.load(self.file_x, allow_pickle=True))
            y = np.load(self.file_y, allow_pickle=True)
        except:
            raise Exception('Path Error: file does not exist, please check this path {}, and {}'.format(self.file_x, self.file_y))
        return X, y

def compute_class_weight(y_train):
    """compute class balancing

    Args:
        y_train (list, ndarray): [description]

    Returns:
        (dict): class weight balancing
    """
    return dict(zip(np.unique(y_train), 
                    sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(y_train),
                                                      y=y_train))) 
        
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def write_log(filepath='test.log', data=[], mode='w'):
    '''
    filepath: path to save
    data: list of data
    mode: a = update data to file, w = write a new file
    '''
    try:
        with open(filepath, mode) as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(data)
    except IOError:
        raise Exception('I/O error')

def zero_padding(data, pad_size=4):
    if len(data.shape) != 4:
        raise Exception('Dimension is not match!, must have 4 dims')
    new_shape = int(data.shape[2]+(2*pad_size))
    data_pad = np.zeros((data.shape[0], data.shape[1], new_shape, new_shape))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_pad[i,j,:,:] = np.pad(data[i,j,:,:], [pad_size, pad_size], mode='constant')
    print(data_pad.shape)
    return data_pad 

def butter_bandpass_filter(data, lowcut, highcut, sfreq, order):
    nyq = 0.5 * sfreq
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    y = scipy.signal.filtfilt(b, a, data)
    return y

def resampling(data, sfreq, data_len):
    if len(data.shape) != 3:
        raise Exception('Dimesion error', "--> please use three-dimensional input")
    new_smp_point = int(data_len*sfreq)
    data_resampled = np.zeros((data.shape[0], data.shape[1], new_smp_point))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_resampled[i,j,:] = scipy.signal.resample(data[i,j,:], new_smp_point)
    return data_resampled


def psd_welch(data, smp_freq):
    if len(data.shape) != 3:
        raise Exception("Dimension Error, must have 3 dimension")
    n_samples,n_chs,n_points = data.shape
    data_psd = np.zeros((n_samples,n_chs,89))
    for i in range(n_samples):
        for j in range(n_chs):
            freq, power_den = scipy.signal.welch(data[i,j], smp_freq, nperseg=n_points)
            index = np.where((freq>=8) & (freq<=30))[0].tolist()
            # print("the length of---", len(index))
            data_psd[i,j] = power_den[index]
    return data_psd
