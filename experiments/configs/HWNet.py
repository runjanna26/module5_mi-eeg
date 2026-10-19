"""
=============================================================================
 EXPERIMENT CONFIG FOR HWNet  (BCIC2a, subject-dependent)
=============================================================================

`experiments/run_HWNet.py` loads this file through
`configs/exp_config.get_params()` and uses the returned dotdict to build the
model, the DataLoader and the log directory.

You will come back to this file constantly -- it is where you tune the
training model without touching model code, because every key of
`model_params` is forwarded to `HWNet.__init__` and re-applied as
`self.<key>` at the end of `HWNet._config()`.

-----------------------------------------------------------------------------
 THE ONE THING THAT BREAKS FIRST: SHAPES
-----------------------------------------------------------------------------
mixnet.utils.DataLoader is the class that loads your data and passes it 
to your model. 

Every array your Part 1 pipeline saved is passed through
`mixnet.utils.DataLoader._change_data_format()` on its way to your model --
`load_train_set()`, `load_val_set()` and `load_test_set()` all call it. It is
about twenty lines; open it and read it alongside this section.

The letters are axis names: N = trials, C = channels, T = time, D = depth
(the singleton axis a Conv2D needs), S = sub-bands/freqs, H/W = height/width
of a 2-D spatial map.

The below table shows the five valid values for `data_format` and what they do to 
the shape of your data. The result is what your model will see, so make sure it 
matches the input shape you defined in `HWNet.__init__()`.

  data_format | what the method does                      | result
  ------------+-------------------------------------------+--------------
  'NCTD'      | reshape(n, C, T, 1)                       | (n, C, T, 1)
  'NDCT'      | reshape(n, 1, C, T)                       | (n, 1, C, T)
  'NTCD'      | reshape(n, C, T, 1), then swapaxes(1, 3)  | (n, 1, T, C)
  'NSHWD'     | reshape(n, S, H, W, 1)                    | (n, S, H, W, 1)
   None       | nothing at all                            | (n, C, T)

Anything other than these five values raises:
    Value Error: data_format requires None, 'NCTD', 'NDCT', 'NTCD' or 'NSHWD'
    
Note: You can add your custom data_format to the list above, but you must also implement it in
`mixnet.utils.DataLoader._change_data_format()`
-----------------------------------------------------------------------------
"""

import os

from tensorflow.keras.optimizers import Adam

from mixnet.loss import *
from mixnet.models import *
from mixnet.utils import dotdict

# Input sets written by prep_HW.py (name -> number of signals). The data have
# shape (n_trials, 400, n_signals); data_format=None passes them through.
FRONT_ENDS = {'fbn18': 18,     # final: filter bank + CSP + per-trial norm + unit scale
              'fb18': 18,      # without per-trial normalisation
              'fbn18u': 18,    # without unit scaling
              'fbrn18': 18,    # CSP with fixed shrinkage 0.05
              'fblw18': 18,    # CSP with Ledoit-Wolf shrinkage
              'fbsc18': 18}    # stationary CSP

# Final model. Reservoir settings and ridge strength from cmaes_search.py
# (validation folds only, results/cmaes_best_nodirect.json).
FINAL_HP = dict(prep='fbn18', arch='esn', units=54, spectral_radius=0.3410,
                input_scale=0.01236, leak=0.7524, connectivity=0.3151, reservoir_seed=0,
                washout=50, skip=False, pool='logvar', standardise=True, ridge_lambda=64.49)

# Other runs reported in report.pdf. Select one with the HW_VARIANT environment
# variable; it is written to its own log directory (suffix = variant name):
#     HW_VARIANT=rn_ablate_reservoir python run_HWNet.py --log_dir logs_tuning
VARIANTS = {
    'rn_final': {},
    # one change at a time from the final model
    'rn_ablate_trial_norm': dict(prep='fb18'),
    'rn_ablate_unit_scale': dict(prep='fbn18u'),
    'rn_ablate_reservoir': dict(arch='no_reservoir'),
    'rn_ablate_logvar_pool': dict(pool='mean'),
    'rn_ablate_standardise': dict(standardise=False),
    'rn_add_direct_inputs': dict(skip=True),
    # the final model with two other random reservoirs
    'rn_final_seed1': dict(reservoir_seed=1),
    'rn_final_seed2': dict(reservoir_seed=2),
}
# CSP variants, with and without the reservoir
for _name, _prep in (('fixed', 'fbrn18'), ('lw', 'fblw18'), ('scsp', 'fbsc18')):
    VARIANTS['rn_csp_' + _name] = dict(prep=_prep)
    VARIANTS['rn_csp_{}_nores'.format(_name)] = dict(prep=_prep, arch='no_reservoir')
FINAL = 'rn_final'


def get_params(dataset, train_type, data_type, num_class, loss_weights=None,
               log_dir='logs', **kwargs):
    """Return a dotdict of parameters for `run_HWNet.py`."""
    model_name = 'HWNet'
    if dataset != 'BCIC2a':
        raise ValueError('The HWNet config is written for BCIC2a only')
    n_subjects = 9

    variant = os.environ.get('HW_VARIANT') or FINAL
    hp = dict(FINAL_HP, **VARIANTS[variant])
    print('HWNet variant: {} -> {}'.format(variant, hp))

    loss_weights = [1.] if loss_weights is None else loss_weights
    log_path = '{}/{}/{}_{}_classes_{}'.format(log_dir, model_name, train_type,
                                               str(num_class), dataset)
    if os.environ.get('HW_VARIANT'):
        log_path += '_' + variant
    # The final input set is datasets/BCIC2a/HW_prep (plain prep_HW.py);
    # the others are in HW_prep_<name>.
    if hp['prep'] != FINAL_HP['prep']:
        data_type = data_type + '_' + hp['prep']

    params = dotdict({
        'model': HWNet,
        'model_params': dotdict({
            'model_name': model_name,
            'input_shape': (400, FRONT_ENDS[hp['prep']]),
            'arch': hp['arch'],
            'units': hp['units'],
            'spectral_radius': hp['spectral_radius'],
            'leak': hp['leak'],
            'input_scale': hp['input_scale'],
            'connectivity': hp['connectivity'],
            'reservoir_seed': hp['reservoir_seed'],
            'washout': hp['washout'],
            'skip': hp['skip'],
            'pool': hp['pool'],
            'standardise': hp['standardise'],
            'ridge_lambda': hp['ridge_lambda'],
            'f1_average': 'macro',
            'num_class': num_class,
            'loss': [SparseCategoricalCrossentropy()],
            'loss_names': ['crossentropy'],
            'loss_weights': loss_weights,
            # The readout is solved in closed form (HWNet.fit), so the gradient
            # training settings of the framework are not used.
            'optimizer': Adam(),
            'verbose': 1,
            'log_path': log_path,
            'data_format': 'channels_last'
        }),

        'data_params': dotdict({
            'dataset': dataset,
            'train_type': train_type,
            'data_format': None,
            'data_type': data_type,
            'num_class': num_class,
            'dataset_path': 'datasets',
            'n_subjects': n_subjects,
            'n_folds': 5,
            'load_path': 'datasets/{}/{}/{}_class/'.format(dataset, data_type, num_class)
        }),

        'log_path': log_path
    })

    return params
