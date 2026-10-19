import numpy as np
from mixnet.utils import resampling
from mixnet.preprocessing.config import CONSTANT
CONSTANT = CONSTANT['HighGamma']
# n_chs = CONSTANT['n_chs']
window_len = CONSTANT['trial_len']*CONSTANT['orig_smp_freq']
orig_chs = CONSTANT['orig_chs']
trial_len = CONSTANT['trial_len'] 
orig_smp_freq = CONSTANT['orig_smp_freq']

def read_raw(PATH, subject, num_class):
    file_name = PATH+'/S{:02d}.npz'.format(subject)
    data = np.load(file_name)
    X_tr, y_tr = data['X_train'], __convert_class(data['y_train'])
    X_te, y_te = data['X_test'], __convert_class(data['y_test'])
    if num_class == 2:
        tr_id = list(np.where((y_tr == 0) | (y_tr == 1))[0])
        te_id = list(np.where((y_te == 0) | (y_te == 1))[0])
        X_tr, y_tr = X_tr[tr_id], y_tr[tr_id]
        X_te, y_te = X_te[te_id], y_te[te_id]
    print("y_train", np.unique(y_tr))
    print("y_test", np.unique(y_te))
    return X_tr, y_tr, X_te, y_te

def __convert_class(labels_list):
    new_labels = []
    for lab in labels_list:
        if lab == 'left_hand':
            new_labels.append(0)
        elif lab == 'right_hand':
            new_labels.append(1)
        elif lab == 'feet':
            new_labels.append(2)
        else:
            new_labels.append(3) # class rest
    return np.array(new_labels)

def chanel_selection(sel_chs): 
    chs_id = []
    for name_ch in sel_chs:
        ch_id = np.where(np.array(orig_chs) == name_ch)[0][0]
        chs_id.append(ch_id)
        print('chosen_channel:', name_ch, '---', 'Index_is:', ch_id)
    return chs_id
        
def load_crop_data(PATH, subject, new_smp_freq, num_class, id_chosen_chs):
    X_tr, y_train, X_te, y_test = read_raw(PATH, subject, num_class)
    X_train = np.take(X_tr, id_chosen_chs, axis=1)
    X_test = np.take(X_te, id_chosen_chs, axis=1)
    if new_smp_freq < orig_smp_freq:
        print("===== Downsampling is being processed =====")
        print("===== Downsampling from {} Hz to {} Hz =====".format(orig_smp_freq, new_smp_freq))
        X_train = resampling(X_train, new_smp_freq, trial_len)
        X_test = resampling(X_test, new_smp_freq, trial_len)
    print("Verify dimension training {} and testing {} data".format(X_train.shape, X_test.shape)) 
    return X_train, y_train, X_test, y_test 


