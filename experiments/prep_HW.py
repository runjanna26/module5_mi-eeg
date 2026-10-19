"""
=============================================================================
 HOMEWORK -- RUN YOUR PREPROCESSING PIPELINE  (Part 1)
=============================================================================

Reads the raw BCIC2a .mat files and writes the reservoir input signals to
    datasets/BCIC2a/HW_prep/2_class/subject_dependent/

Usage
    python prep_HW.py                    # the final front end (FINAL below)
    HW_PREP=fb18 python prep_HW.py       # another input set (ablations, CSP variants),
                                         # written to datasets/BCIC2a/HW_prep_fb18/...

Prerequisite
    python download_datasets.py --dataset 'BCIC2a'
"""
import os

import mixnet.preprocessing as prep

FILTER_BANK = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32),
               (32, 36), (36, 40)]

# Input sets used in the report. Each gives signals of shape (400, 18).
FRONT_ENDS = {
    'fbn18': dict(bands=FILTER_BANK, n_pairs=1, trial_norm='band'),      # final
    'fb18': dict(bands=FILTER_BANK, n_pairs=1),                          # no trial norm
    'fbn18u': dict(bands=FILTER_BANK, n_pairs=1, trial_norm='band',
                   unit_scale=False),                                    # no unit scaling
    'fbrn18': dict(bands=FILTER_BANK, n_pairs=1, shrinkage=0.05, trial_norm='band'),
    'fblw18': dict(bands=FILTER_BANK, n_pairs=1, csp='lw', trial_norm='band'),
    'fbsc18': dict(bands=FILTER_BANK, n_pairs=1, csp='scsp', alpha=0.03, trial_norm='band'),
}
FINAL = 'fbn18'

name = os.environ.get('HW_PREP')
if name:
    prep.BCIC2a.HW_prep.DATA_TYPE = 'HW_prep_' + name
print('Front end:', name or FINAL, FRONT_ENDS[name or FINAL])

prep.BCIC2a.HW_prep.subject_dependent_setting(
    k_folds=5,
    pick_smp_freq=100,
    save_path='datasets',
    num_class=2,
    **FRONT_ENDS[name or FINAL])
