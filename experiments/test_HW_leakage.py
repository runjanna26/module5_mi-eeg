"""
Leakage tests for mixnet/preprocessing/BCIC2a/HW_prep.py.

Usage (from experiments/, after download_datasets.py)
    python test_HW_leakage.py            # or: pytest test_HW_leakage.py

The tests run the real 5-fold driver `subject_dependent_setting()` on one
subject, with the file writer replaced by an in-memory recorder, and check that
nothing learned by the pipeline depends on session E or on the validation fold.
"""
import copy

import numpy as np

from mixnet.preprocessing.BCIC2a import HW_prep

SUBJECT = 1
K_FOLDS = 5
FS = 100


def _run_driver(corrupt_test=False, corrupt_val=False, **prep):
    """Run the provided CV driver and record, per fold, the fitted state and the
    features it would have saved. Optionally replace session E, or the
    validation trials, with noise before the pipeline sees them."""
    rng = np.random.RandomState(0)
    record = {'states': [], 'train': [], 'val': [], 'test': []}
    orig_load, orig_save = HW_prep._load_bcic2a, HW_prep._save_data_with_valset
    orig_fit, orig_transform = HW_prep.fit_transform, HW_prep.transform

    def load(*args, **kwargs):
        X_tr, y_tr, X_te, y_te = orig_load(*args, **kwargs)
        if corrupt_test:
            X_te = rng.randn(*X_te.shape) * X_te.std() * 5.0
        return X_tr, y_tr, X_te, y_te

    def fit(X, y, **kwargs):
        X_out, state = orig_fit(X, y, **kwargs)
        record['states'].append(copy.deepcopy(state))
        return X_out, state

    calls = {'n': 0}

    def transform(X, state, **kwargs):
        # The driver calls transform() for the validation split first, then test.
        is_val = calls['n'] % 2 == 0
        calls['n'] += 1
        if corrupt_val and is_val:
            X = rng.randn(*X.shape) * X.std() * 5.0
        return orig_transform(X, state, **kwargs)

    def save(save_path, name, X_train, y_train, X_val, y_val, X_test, y_test):
        record['train'].append(X_train)
        record['val'].append(X_val)
        record['test'].append(X_test)

    HW_prep._load_bcic2a, HW_prep._save_data_with_valset = load, save
    HW_prep.fit_transform, HW_prep.transform = fit, transform
    try:
        HW_prep.subject_dependent_setting(
            k_folds=K_FOLDS, pick_smp_freq=FS, save_path='/tmp/_hw_leak_unused',
            num_class=2, subjects=[SUBJECT], **prep)
    finally:
        HW_prep._load_bcic2a, HW_prep._save_data_with_valset = orig_load, orig_save
        HW_prep.fit_transform, HW_prep.transform = orig_fit, orig_transform
    return record


def _assert_states_equal(a, b):
    assert len(a) == len(b) == K_FOLDS
    for sa, sb in zip(a, b):
        assert sa.keys() == sb.keys()
        for fa, fb in zip(sa['filters'], sb['filters']):
            np.testing.assert_array_equal(fa, fb)
        np.testing.assert_array_equal(sa['scale'], sb['scale'])
        for key in ('bands', 'fs'):
            assert sa[key] == sb[key]


_CLEAN = None


def _clean():
    global _CLEAN
    if _CLEAN is None:
        _CLEAN = _run_driver()
    return _CLEAN


def test_state_does_not_depend_on_session_E():
    clean, corrupt = _clean(), _run_driver(corrupt_test=True)
    _assert_states_equal(clean['states'], corrupt['states'])
    for f in range(K_FOLDS):
        np.testing.assert_array_equal(clean['train'][f], corrupt['train'][f])
        np.testing.assert_array_equal(clean['val'][f], corrupt['val'][f])
        # sanity: the corruption really reached the pipeline
        assert not np.allclose(clean['test'][f], corrupt['test'][f])


def test_state_does_not_depend_on_validation_fold():
    clean, corrupt = _clean(), _run_driver(corrupt_val=True)
    _assert_states_equal(clean['states'], corrupt['states'])
    for f in range(K_FOLDS):
        np.testing.assert_array_equal(clean['train'][f], corrupt['train'][f])
        np.testing.assert_array_equal(clean['test'][f], corrupt['test'][f])
        assert not np.allclose(clean['val'][f], corrupt['val'][f])


def test_transform_is_per_trial_and_read_only():
    """transform() must use no statistic of the batch it is given (otherwise the
    test session would normalise itself) and must not modify `state`."""
    rng = np.random.RandomState(1)
    X_fit = rng.randn(40, 20, 400)
    y_fit = np.repeat([0, 1], 20)
    X_new = rng.randn(12, 20, 400) * 3.0 + 1.0
    _, state = HW_prep.fit_transform(X_fit, y_fit, pick_smp_freq=FS)
    before = copy.deepcopy(state)

    full = HW_prep.transform(X_new, state, pick_smp_freq=FS)
    single = np.concatenate(
        [HW_prep.transform(X_new[i:i + 1], state, pick_smp_freq=FS)
         for i in range(len(X_new))])
    np.testing.assert_allclose(full, single, rtol=1e-5, atol=1e-6)
    _assert_states_equal([before] * K_FOLDS, [state] * K_FOLDS)


def test_train_signals_have_unit_scale_and_shape_is_as_configured():
    clean = _clean()
    for f in range(K_FOLDS):
        assert clean['train'][f].shape[1:] == (400, 18)
        assert clean['val'][f].shape[1:] == (400, 18)
        assert clean['test'][f].shape == (144, 400, 18)
        np.testing.assert_allclose(clean['train'][f].std(axis=(0, 1)), 1, atol=1e-3)


CSP_VARIANTS = [dict(csp='lw', trial_norm='band'),
                dict(csp='scsp', alpha=0.03, trial_norm='band')]


def test_csp_variants_do_not_depend_on_session_E_or_validation_fold():
    """Ledoit-Wolf shrinkage and the stationary penalty are estimated from data,
    so they get the same noise-replacement test as the default pipeline."""
    for prep in CSP_VARIANTS:
        clean = _run_driver(**prep)
        for kw in (dict(corrupt_test=True), dict(corrupt_val=True)):
            corrupt = _run_driver(**kw, **prep)
            _assert_states_equal(clean['states'], corrupt['states'])
            for f in range(K_FOLDS):
                np.testing.assert_array_equal(clean['train'][f], corrupt['train'][f])


if __name__ == '__main__':
    tests = [test_state_does_not_depend_on_session_E,
             test_state_does_not_depend_on_validation_fold,
             test_transform_is_per_trial_and_read_only,
             test_train_signals_have_unit_scale_and_shape_is_as_configured,
             test_csp_variants_do_not_depend_on_session_E_or_validation_fold]
    for test in tests:
        test()
        print('PASS', test.__name__)
    print('All {} leakage tests passed.'.format(len(tests)))
