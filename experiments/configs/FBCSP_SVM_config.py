config = {
    'FBCSP_SVM': {
            'BCIC2a': {
                    'n_subjects': 9
            },
            'BCIC2b': {
                    'n_subjects': 9
            },
            'BNCI2015_001': {
                    'n_subjects': 12
            },
            'SMR_BCI': {
                    'n_subjects': 14
            },
            'HighGamma': {
                    'n_subjects': 14
            },
            'OpenBMI': {
                    'n_subjects': 54
            },
            'dataset_path': 'datasets',
            'n_folds': 5,
            'tuned_parameters' : [{
                        'kernel': ['rbf'],
                        'gamma': [1e-2, 1e-3],
                        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                        },
                        {
                        'kernel': ['sigmoid'],
                        'gamma': [1e-2, 1e-3],
                        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                        },
                        {
                        'kernel': ['linear'],
                        'gamma': [1e-2, 1e-3],
                        'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]
                        }]

    }

}
