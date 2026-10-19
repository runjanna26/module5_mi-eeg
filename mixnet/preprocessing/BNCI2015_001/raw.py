import numpy as np
import scipy.io as sio
from mixnet.utils import resampling
from mixnet.preprocessing.config import CONSTANT
CONSTANT = CONSTANT['BNCI2015_001']
# n_chs = CONSTANT['n_chs']
# n_trials = CONSTANT['n_trials']
window_len = CONSTANT['trial_len']*CONSTANT['orig_smp_freq']
orig_chs = CONSTANT['orig_chs']
trial_len = CONSTANT['trial_len'] 
orig_smp_freq = CONSTANT['orig_smp_freq']

def read_raw(PATH, subject):
    file_name = PATH+'/S{:02d}.npz'.format(subject)
    data = np.load(file_name)
    X_train, y_tr = data['X_train'], data['y_train']
    X_test, y_te = data['X_test'], data['y_test']

    # Categorical encoding for text labels where class 0 is right_hand and class 1 is feet
    y_train = [0 if i == 'right_hand' else 1 for i in y_tr]
    y_test = [0 if j == 'right_hand' else 1 for j in y_te]
    return X_train, y_train, X_test, y_test

def chanel_selection(sel_chs): 
    chs_id = []
    for name_ch in sel_chs:
        ch_id = np.where(np.array(orig_chs) == name_ch)[0][0]
        chs_id.append(ch_id)
        print('chosen_channel:', name_ch, '---', 'Index_is:', ch_id)
    return chs_id
        
def load_crop_data(PATH, subject, start, stop, new_smp_freq, num_class, id_chosen_chs):
    X_tr, y_train, X_te, y_test = read_raw(PATH, subject)
    if new_smp_freq < orig_smp_freq:
        start_time = int(start*new_smp_freq) # 1s
        stop_time = int(stop*new_smp_freq) # 5s
        X_tr = resampling(X_tr, new_smp_freq, trial_len)
        X_te = resampling(X_te, new_smp_freq, trial_len)
    else:
        start_time = int(start*orig_smp_freq) # 1s
        stop_time = int(stop*orig_smp_freq) # 5s
    X_train = X_tr[:,:,start_time:stop_time]
    X_test = X_te[:,:,start_time:stop_time]
    # Selecting only interested channels
    X_train = np.take(X_train, id_chosen_chs, axis=1)
    X_test = np.take(X_test, id_chosen_chs, axis=1)
    print("Verify dimension training {} and testing {}".format(X_train.shape, X_test.shape)) 
    return X_train, y_train, X_test, y_test