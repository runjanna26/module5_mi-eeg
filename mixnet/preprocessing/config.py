from mixnet.utils import PATH

CONSTANT = {
    'BCIC2a': {
        'raw_path': 'datasets/BCIC2a/raw', # raw data path 'raw_path': 'datasets/BCIC2a'
        'n_subjs': 9,
        'n_trials': 144,
        'n_trials_per_class': 72,
        'n_chs': 20,
        'orig_smp_freq': 250,                  # Original sampling frequency (Hz)
        'trial_len': 7,                        # 7s
        'MI': {
            'start': 2,                        # start at time = 2 s
            'stop': 6,                         # stop at time = 6 s
            'len': 4,                          # 4s
        },
        'orig_chs': ['FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                    'P1', 'Pz', 'P2'],
        'sel_chs': ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 
                    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                    'P1', 'Pz', 'P2'], 
        'three_chs': ['C3', 'Cz', 'C4']  
        
    },
    
    'BCIC2b': {
        'raw_path': 'datasets/BCIC2b/raw', # raw data path 'raw_path': 'datasets/BCIC2b'
        'n_subjs': 9,
        'n_chs': 3,
        'orig_smp_freq': 250,                  # Original sampling frequency (Hz)
        'trial_len': 8,                        # 8s
        'MI': {
            'start': 3,                        # start at time = 3 s
            'stop': 7,                         # stop at time = 8 s
            'len': 4,                          # 4s
        },
        'orig_chs': ['C3', 'Cz', 'C4'],
        'sel_chs': ['C3',  'Cz', 'C4'],
        'three_chs': ['C3',  'Cz', 'C4']
    },
    
    'BNCI2015_001': {
        'raw_path': 'datasets/BNCI2015_001/raw', # raw data path
        'n_subjs': 12,
        'n_trials_tr': 200,
        'n_trials_te': 200, 
        'n_trials_per_class': 100,
        'n_chs': 13,
        'orig_smp_freq': 512,                   # Original sampling frequency  (Hz)
        'trial_len': 5,                         # 5s
        'MI': {
            'start': 1,                         # start at time = 1 s
            'stop': 5,                          # stop at time = 5 s
            'len': 4,                           # 4s
        },
        'orig_chs': ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4'],
        'sel_chs': ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4'],
        'three_chs': ['C3', 'Cz', 'C4'] 
    },
    
    'SMR_BCI': {
        'raw_path': 'datasets/SMR_BCI/raw', # raw data path
        'n_subjs': 14,
        'n_trials_tr': 100,
        'n_trials_te': 60, 
        'n_chs': 15,
        'orig_smp_freq': 512,                   # Original sampling frequency  (Hz)
        'trial_len': 8,                         # 7s
        'MI': {
            'start': 4,                         # start at time = 4 s
            'stop': 8,                          # stop at time = 8 s
            'len': 4,                           # 4s
        },
        'orig_chs': ['FCC3',                   'FCCz',                 'FCC4',
                    'C5h', 'C3', 'C3h',       'C1h', 'Cz', 'C2h',       'C4h', 'C4', 'C6h',
                           'CCP3',                   'CCPz',                 'CCP4'],
        'sel_chs': [       'FCC3',                   'FCCz',                 'FCC4', 
                    'C5h', 'C3', 'C3h',       'C1h', 'Cz', 'C2h',       'C4h','C4', 'C6h', 
                           'CCP3',                   'CCPz',                 'CCP4'],
        'three_chs': ['C3', 'Cz', 'C4'] 
    },
    
    'OpenBMI': {
        'raw_path': 'datasets/OpenBMI/raw', # raw data path
        'n_subjs': 54,
        'n_trials_2_class': 100,
        'n_trials_3_class': 150, 
        'n_chs': 62,
        'orig_smp_freq': 1000,                  # Original sampling frequency  (Hz)
        'trial_len': 8,                         # 8s (cut-off)
        'MI': {
            'start': 0,                         # start at time = 0 s
            'stop': 4,                          # stop at time = 0 s
            'len': 4,                           # 4s
        },
        'orig_chs': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 
                    'C3','Cz','C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 
                    'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1',
                    'C2', 'C6', 'CP3','CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 
                    'TP7', 'TPP9h', 'FT10','FTT10h','TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 
                    'AF7', 'AF3', 'AF4', 'AF8', 'PO3','PO4'],
        'sel_chs': ['FC5', 'FC3', 'FC1', 'FC2', 'FC4','FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 
                    'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],
        'three_chs': ['C3', 'Cz', 'C4'] 
    },
    
    'HighGamma': {
        'raw_path': 'datasets/HighGamma/raw', # raw data path
        'n_subjs': 14,
        'n_chs': 129,
        'orig_smp_freq': 500,                  # Original sampling frequency  (Hz)
        'trial_len': 4,                         # 8s (cut-off)
        'MI': {
            'start': 0,                         # start at time = 0 s
            'stop': 4,                          # stop at time = 0 s
            'len': 4,                           # 4s
        },
        'orig_chs': ['Fp1', 'Fp2', 'Fpz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
       'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5',
       'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1',
       'Oz', 'O2', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',
       'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4',
       'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8',
       'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'TPP9h', 'TPP10h',
       'PO9', 'PO10', 'P9', 'P10', 'AFF1', 'AFz', 'AFF2', 'FFC5h',
       'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h',
       'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h', 'CPP3h', 'CPP4h',
       'CPP6h', 'PPO1', 'PPO2', 'I1', 'Iz', 'I2', 'AFp3h', 'AFp4h',
       'AFF5h', 'AFF6h', 'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h', 'FTT9h',
       'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h', 'FTT10h', 'TTP7h', 'CCP1h',
       'CCP2h', 'TTP8h', 'TPP7h', 'CPP1h', 'CPP2h', 'TPP8h', 'PPO9h',
       'PPO5h', 'PPO6h', 'PPO10h', 'POO9h', 'POO3h', 'POO4h', 'POO10h',
       'OI1h', 'OI2h', 'STI 014'],
        'sel_chs': ['FC5', 'FC3', 'FC1', 'FC2', 'FC4','FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 
                    'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
    }

}