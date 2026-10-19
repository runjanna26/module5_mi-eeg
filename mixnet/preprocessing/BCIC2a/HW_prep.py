"""
=============================================================================
 HOMEWORK PART 1 -- BUILD YOUR OWN PREPROCESSING PIPELINE  (BCIC2a)
=============================================================================

Your job in this file is to turn *raw* BCI Competition IV-2a MI-EEG trials
into the feature tensors that your model in `mixnet/models/HWNet.py`
will be trained on.

WHAT IS ALREADY DONE FOR YOU (do not change unless you know why):
  * reading + cropping the raw .mat files              -> `_load_bcic2a()`
  * the 5-fold subject-dependent cross-validation loop -> `subject_dependent_setting()`
  * saving the .npy files with the exact file names    -> `_save_data_with_valset()`
    that `mixnet.utils.DataLoader` expects

WHAT YOU MUST IMPLEMENT (the two `TODO` functions below):
  1. `fit_transform()` -- learn any data-driven parameters on the TRAINING
     fold only, and return the transformed training features.
  2. `transform()`     -- apply the *already learned* parameters to the
     validation / test folds.

-----------------------------------------------------------------------------
THE GOLDEN RULE OF THIS ASSIGNMENT
-----------------------------------------------------------------------------
 Anything that *LEARNS* from data (CSP filters, PCA, a mean/std scaler, a
 whitening matrix, a channel-selection criterion, ...) must be fitted on the
 TRAINING fold ONLY, then merely *applied* to validation and test data.
 Fitting on all the data first and splitting afterwards is data leakage: it
 silently inflates your accuracy and makes the benchmark against MixNet
 meaningless. Marks are deducted for leakage.

 Stateless operations (band-pass filtering, cropping, resampling, log of the
 variance, ...) are safe to apply to every split independently.
-----------------------------------------------------------------------------

Data you receive in `fit_transform` / `transform`
    X : np.ndarray, shape (n_trials, n_channels, n_samples)
        n_channels = len(sel_chs)              (20 by default, see config.py)
        n_samples  = MI_len * pick_smp_freq    (4 s * 100 Hz = 400 by default)
    y : np.ndarray, shape (n_trials,)  with values in {0, 1}
        0 = left hand, 1 = right hand

Data you must return
    X_out : np.ndarray, shape (n_trials, ...anything you like...)
        Keep the trial axis first. Whatever trailing shape you choose decides
        the `input_shape` of your model: `DataLoader` reshapes it according to
        `data_format`, and the result minus the trial axis is what your
        network receives. The driver below prints the shape after every fold
        -- note it down, you need it in `experiments/configs/HWNet.py`. See
        the data_format table in that file.
"""

import os

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedKFold

from mixnet.preprocessing.BCIC2a import raw
from mixnet.preprocessing.config import CONSTANT

CONSTANT = CONSTANT['BCIC2a']
raw_path = CONSTANT['raw_path']
n_subjs = CONSTANT['n_subjs']
orig_smp_freq = CONSTANT['orig_smp_freq']
MI_len = CONSTANT['MI']['len']

# Name of the folder your features are written to. It must match the
# `--data_type` argument of run_HWNet.py. Keep it as is.
DATA_TYPE = 'HW_prep'

# Defaults of the pipeline; every one can be overridden from prep_HW.py.
BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32),
         (32, 36), (36, 40)]          # same filter bank as FBCSP / MixNet
N_PAIRS = 1                           # CSP pairs per band (1 pair = 2 filters); 0 = no CSP
FILTER_ORDER = 4
EPS = 1e-8


def _bandpass(X, lo, hi, fs, order=FILTER_ORDER):
    """Zero-phase Butterworth band-pass along time. Stateless: the coefficients
    depend only on (lo, hi, fs), and each trial is filtered on its own."""
    sos = butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, X, axis=-1)


def _trial_covs(Xb, y, c):
    """Trace-normalised covariance of every trial of class c: (n_c, ch, ch)."""
    out = []
    for i in np.where(y == c)[0]:
        cov = Xb[i] @ Xb[i].T
        out.append(cov / np.trace(cov))
    return np.stack(out)


def _lw_shrinkage(covs, mean):
    """Ledoit-Wolf shrinkage intensity of the mean of per-trial covariances.

    The mean is an estimate built from n noisy matrices; the shrinkage target is
    mu * I with mu = trace(mean) / ch. Intensity = (variance of the estimate) /
    (squared distance of the estimate from the target), clipped to [0, 1]
    (Ledoit & Wolf, 2004). Estimated from the data it is given, so call it with
    training-fold trials only.
    """
    n, ch = covs.shape[0], covs.shape[1]
    mu = np.trace(mean) / ch
    d2 = np.sum((mean - mu * np.eye(ch)) ** 2)
    b2 = np.sum((covs - mean) ** 2) / n ** 2
    return float(np.clip(b2 / max(d2, 1e-30), 0.0, 1.0))


def _matrix_abs(A):
    """|A| = V |D| V^T for a symmetric matrix."""
    vals, vecs = np.linalg.eigh(A)
    return (vecs * np.abs(vals)) @ vecs.T


def _fit_csp(Xb, y, n_pairs, shrinkage=0.0, csp='csp', alpha=0.0):
    """CSP filters of one band. LEARNED -- call with training-fold trials only.

    Xb: (n, ch, t) band-passed trials, y: (n,) labels in {0, 1}.
    Returns (2 * n_pairs, ch): the filters whose output variance is smallest
    and largest for class 0 relative to both classes together.

    csp='csp'   plain CSP; `shrinkage` pulls each class covariance toward a
                scaled identity by a fixed fraction.
    csp='lw'    the same, with the fraction estimated per class by Ledoit-Wolf.
    csp='scsp'  stationary CSP (Samek et al., 2012): the denominator gets a
                penalty that grows with how much a filter's power varies from
                trial to trial. `alpha` is the penalty's weight relative to the
                total covariance.
    """
    n_ch = Xb.shape[1]
    covs, per_trial = [], []
    for c in (0, 1):
        tc = _trial_covs(Xb, y, c)
        C = tc.mean(axis=0)
        k = _lw_shrinkage(tc, C) if csp == 'lw' else shrinkage
        # With ~58 trials per class and 20 channels the class covariance is
        # noisy, and CSP overfits to that noise; shrink toward a scaled identity.
        covs.append((1.0 - k) * C + k * np.trace(C) / n_ch * np.eye(n_ch))
        per_trial.append((tc, C))
    total = covs[0] + covs[1]
    if csp == 'scsp' and alpha > 0:
        pen = sum(np.mean([_matrix_abs(ci - C) for ci in tc], axis=0) for tc, C in per_trial)
        total = total + alpha * pen * np.trace(total) / np.trace(pen)
    elif csp not in ('csp', 'lw', 'scsp'):
        raise ValueError('Unknown csp: {}'.format(csp))
    # Generalised eigenproblem C0 w = lambda (C0 + C1 [+ penalty]) w.
    # The ridge is scaled to the matrix so it only matters if `total` is singular.
    ridge = EPS * np.trace(total) / n_ch
    vals, vecs = eigh(covs[0], total + ridge * np.eye(n_ch))
    order = np.argsort(vals)
    pick = np.r_[order[:n_pairs], order[-n_pairs:]]
    return vecs[:, pick].T


def _signals(X, filters, bands, fs, trial_norm='none'):
    """Band-pass every band and, if CSP filters are given, project on them.
    Learns nothing: `filters` are supplied by the caller.

    trial_norm='band' divides the signals of each band by the root-mean-square
    amplitude of that band in the same trial. It uses one trial at a time, so it
    is stateless. It removes the overall power of the band, which changes from
    day to day, and keeps the balance between the CSP filters, which is what
    separates the classes.
    Returns (n, n_signals, t) with the bands stacked along the signal axis."""
    out = []
    for b, (lo, hi) in enumerate(bands):
        Xb = _bandpass(X, lo, hi, fs)
        Z = Xb if filters is None else np.einsum('kc,nct->nkt', filters[b], Xb)
        if trial_norm == 'band':
            Z = Z / (np.sqrt((Z ** 2).mean(axis=(1, 2), keepdims=True)) + EPS)
        elif trial_norm != 'none':
            raise ValueError('Unknown trial_norm: {}'.format(trial_norm))
        out.append(Z)
    return np.concatenate(out, axis=1)


# =============================================================================
#  TODO 1 / 2 -- fit on the training fold and transform it
#
#  NOTE: you are allowed to add any extra arguments to `fit_transform()` and `transform()`
#  but you must NOT change the signature of `subject_dependent_setting()`.
# =============================================================================
def fit_transform(X, y, pick_smp_freq, bands=None, n_pairs=N_PAIRS, shrinkage=0.0,
                  trial_norm='none', unit_scale=True, csp='csp', alpha=0.0, **kwargs):
    """Learn the pipeline parameters on the TRAINING fold and transform it.

    Called once per (subject, fold) with the training split only.

    Args:
        X (np.ndarray): (n_trials, n_channels, n_samples) raw cropped signals.
        y (np.ndarray): (n_trials,) labels in {0, 1}. Use them if your
            pipeline is supervised (e.g. CSP); ignore them otherwise.
        pick_smp_freq (int): sampling rate of `X` in Hz (100 by default).
        **kwargs: whatever extra hyper-parameters you forward from
            `experiments/prep_HW.py` (band definitions, n_components, ...).

    Returns:
        X_out (np.ndarray): (n_trials, ...) transformed training features.
        state (dict): everything `transform()` needs to reproduce exactly the
            same transformation on unseen data. Must be picklable/plain Python.
            Return an empty dict {} if your pipeline is fully stateless.
            
    Useful helpers already in the repo:
        from mixnet.utils import butter_bandpass_filter, resampling, psd_welch
        from mixnet.preprocessing import FBCSP, SpectralSpatialMapping
        
        NOTE: You can also add your own helper functions in this file or in
        `mixnet/preprocessing/BCIC2a/HW_prep_helpers.py` if you want to keep this file clean.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    fs = pick_smp_freq
    bands = [tuple(b) for b in (BANDS if bands is None else bands)]

    # Learned step 1 (optional): CSP filters per band, from the training fold.
    filters = None
    if n_pairs > 0:
        filters = [_fit_csp(_bandpass(X, lo, hi, fs), y, n_pairs, shrinkage, csp, alpha)
                   for lo, hi in bands]
    sig = _signals(X, filters, bands, fs, trial_norm)        # (n, n_signals, t)

    # Learned step 2: one scale per signal (its standard deviation over all
    # training-fold trials and samples), so the reservoir input has unit scale.
    scale = sig.std(axis=(0, 2), keepdims=True) + EPS
    if not unit_scale:                                       # ablation only
        scale = np.ones_like(scale)

    state = {'filters': filters, 'scale': scale, 'bands': bands, 'fs': fs,
             'trial_norm': trial_norm}
    # time first: (n, t, n_signals), the layout a recurrent layer expects
    return np.transpose(sig / scale, (0, 2, 1)).astype(np.float32), state


# =============================================================================
#  TODO 2 / 2 -- apply the learned pipeline to validation / test folds
#
#  NOTE: you are allowed to add any extra arguments to `fit_transform()` and `transform()`
#  but you must NOT change the signature of `subject_dependent_setting()`.
# =============================================================================
def transform(X, state, pick_smp_freq, **kwargs):
    """Apply the pipeline learned by `fit_transform()` to unseen trials.

    Called twice per (subject, fold): once for the validation split and once
    for the test split. 
    NOTE: It must NOT re-fit anything -- only read from `state`.

    Args:
        X (np.ndarray): (n_trials, n_channels, n_samples) raw cropped signals.
        state (dict): the object returned by `fit_transform()`.
        pick_smp_freq (int): sampling rate of `X` in Hz.
        **kwargs: same extra hyper-parameters as `fit_transform()`.

    Returns:
        X_out (np.ndarray): (n_trials, ...) features with the SAME trailing
            shape as the output of `fit_transform()`.
    """
    if pick_smp_freq != state['fs']:
        raise ValueError('transform() got pick_smp_freq={} but the pipeline was '
                         'fitted at {} Hz'.format(pick_smp_freq, state['fs']))
    X = np.asarray(X, dtype=np.float64)
    # Everything data-driven comes from `state`; nothing is re-estimated here.
    sig = _signals(X, state['filters'], state['bands'], state['fs'], state['trial_norm'])
    return np.transpose(sig / state['scale'], (0, 2, 1)).astype(np.float32)


# =============================================================================
#  Provided for you -- the subject-dependent 5-fold CV driver
#  
#  NOTE: you are allowed to add any extra arguments to `fit_transform()`, 
# `transform()`, and `subject_dependent_setting()`
#  but you must NOT change the signature of `subject_dependent_setting()`.
# =============================================================================
def subject_dependent_setting(k_folds, pick_smp_freq, save_path, num_class=2,
                              sel_chs=None, subjects=None, **kwargs):
    """Run the whole pipeline for every subject in the subject-dependent setting.

    Subject-dependent means: for each of the 9 subjects we train, validate and
    test on that subject alone. Session T is split into train/validation with
    stratified k-fold CV; session E is the untouched test set of every fold.

    Args:
        k_folds (int): number of CV folds (5 to match the MixNet baseline).
        pick_smp_freq (int): target sampling rate in Hz (100 in the baseline).
        save_path (str): dataset root, e.g. 'datasets'.
        num_class (int): 2 = left hand vs right hand.
        sel_chs (list[str] | None): channel names; None = all 20 from config.py.
        subjects (list[int] | None): 1-based subject ids; None = all 9.
        **kwargs: forwarded verbatim to `fit_transform()` / `transform()`.
    """
    sel_chs = CONSTANT['sel_chs'] if sel_chs is None else sel_chs
    save_path = os.path.join(
        save_path, 'BCIC2a', DATA_TYPE, '{}_class'.format(num_class),
        'subject_dependent')
    os.makedirs(save_path, exist_ok=True)

    id_chosen_chs = raw.chanel_selection(sel_chs)
    subjects = range(1, n_subjs + 1) if subjects is None else subjects

    for subject in subjects:
        X_tr, y_tr, X_te, y_te = _load_bcic2a(
            subject, pick_smp_freq, num_class, id_chosen_chs)

        if X_tr.ndim != 3:
            raise Exception('Dimension Error, must have 3 dimensions, '
                            'found {}'.format(X_tr.shape))

        skf = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)
        for fold, (train_index, val_index) in enumerate(skf.split(X_tr, y_tr)):
            print('SUBJECT:', subject, 'FOLD:', fold + 1,
                  'TRAIN:', len(train_index), 'VALIDATION:', len(val_index))

            X_tr_cv, X_val_cv = X_tr[train_index], X_tr[val_index]
            y_tr_cv, y_val_cv = y_tr[train_index], y_tr[val_index]

            # ---- your pipeline: fit on train only, then apply everywhere ----
            X_tr_out, state = fit_transform(
                X_tr_cv, y_tr_cv, pick_smp_freq=pick_smp_freq, **kwargs)
            X_val_out = transform(
                X_val_cv, state, pick_smp_freq=pick_smp_freq, **kwargs)
            X_te_out = transform(
                X_te, state, pick_smp_freq=pick_smp_freq, **kwargs)
            # -----------------------------------------------------------------

            _check_shapes(X_tr_out, X_val_out, X_te_out,
                          y_tr_cv, y_val_cv, y_te)
            print('Feature shapes -- train {}, val {}, test {}'.format(
                X_tr_out.shape, X_val_out.shape, X_te_out.shape))

            save_name = 'S{:03d}_fold{:03d}'.format(subject, fold + 1)
            _save_data_with_valset(save_path, save_name,
                                   X_tr_out, y_tr_cv,
                                   X_val_out, y_val_cv,
                                   X_te_out, y_te)
            print('The preprocessing of subject {} from fold {} is DONE!!!'
                  .format(subject, fold + 1))


def _load_bcic2a(subject, pick_smp_freq, num_class, id_chosen_chs):
    """Read session T (train) and session E (test), crop to the MI window."""
    start = CONSTANT['MI']['start']   # 2 s after the cue
    stop = CONSTANT['MI']['stop']     # 6 s after the cue
    return raw.load_crop_data(PATH=raw_path, subject=subject,
                              start=start, stop=stop,
                              new_smp_freq=pick_smp_freq,
                              num_class=num_class,
                              id_chosen_chs=id_chosen_chs)


def _check_shapes(X_tr, X_val, X_te, y_tr, y_val, y_te):
    """Fail early and loudly on the mistakes that cost the most debugging time."""
    for name, X, y in [('train', X_tr, y_tr), ('val', X_val, y_val),
                       ('test', X_te, y_te)]:
        X = np.asarray(X)
        if X.shape[0] != len(y):
            raise Exception(
                'Trial-axis mismatch on the {} split: X has {} trials but y '
                'has {}. Your transform must keep the trial axis first and '
                'must not drop or reorder trials.'.format(
                    name, X.shape[0], len(y)))
        if not np.all(np.isfinite(X)):
            raise Exception(
                'The {} split contains NaN or Inf. This usually means an '
                'unstable filter (check the order and the band edges against '
                'the Nyquist frequency) or a division by a zero variance.'
                .format(name))
    if X_tr.shape[1:] != X_val.shape[1:] or X_tr.shape[1:] != X_te.shape[1:]:
        raise Exception(
            'Feature shapes differ across splits: train {}, val {}, test {}. '
            'transform() must produce the same trailing shape as '
            'fit_transform().'.format(
                X_tr.shape[1:], X_val.shape[1:], X_te.shape[1:]))


def _save_data_with_valset(save_path, NAME, X_train, y_train,
                           X_val, y_val, X_test, y_test):
    """Write the six .npy files under the names `DataLoader` looks for."""
    np.save(save_path + '/X_train_' + NAME + '.npy', X_train)
    np.save(save_path + '/X_val_' + NAME + '.npy', X_val)
    np.save(save_path + '/X_test_' + NAME + '.npy', X_test)
    np.save(save_path + '/y_train_' + NAME + '.npy', y_train)
    np.save(save_path + '/y_val_' + NAME + '.npy', y_val)
    np.save(save_path + '/y_test_' + NAME + '.npy', y_test)
    print('save DONE')
