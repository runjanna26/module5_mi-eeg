"""
CMA-ES search of the echo state network's hyperparameters, on VALIDATION folds only.

The reservoir is fixed and the readout is closed-form ridge regression, so one
candidate setting can be scored without TensorFlow: for every subject and fold
of session T, the reservoir is run in NumPy on X_train and X_val (X_test is never
loaded), the readout is fitted on the training fold, and the mean squared error
on the validation fold is recorded. The objective is that error averaged over
9 subjects x 5 folds x 3 random reservoirs, so the search cannot settle on one
lucky reservoir draw.

Searched (CMA-ES works in [0, 1]^6, mapped as below):
    units            2 ** [3, 6]       -> 8 ... 64, rounded to an even number
    spectral_radius  [0.0, 1.2]
    input_scale      10 ** [-2, 0.5]
    leak             [0.05, 1.0]
    connectivity     [0.05, 1.0]
    ridge_lambda     10 ** [-3, 3]

Usage (from experiments/, after prep_HW.py)
    python cmaes_search.py
Writes results/cmaes_history[_nodirect].csv and results/cmaes_best[_nodirect].json.
Set DIRECT_INPUTS=1 to let the readout also see the pooled inputs (the earlier design).
"""
import itertools
import json
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import cma
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mixnet.models.HWNet import reservoir_weights

DIRECT = os.environ.get('DIRECT_INPUTS', '0') == '1'   # readout also sees the pooled inputs
TAG = '' if DIRECT else '_nodirect'
ROOT = 'datasets/BCIC2a/HW_prep/2_class/subject_dependent/{}_S{:03d}_fold{:03d}.npy'
WASHOUT = 50
SEEDS = (0, 1, 2)
SIGMA0, POPSIZE, MAX_GEN = 0.25, 12, 25
_CACHE = {}


def decode(z):
    z = np.clip(z, 0.0, 1.0)
    units = int(2 * round(2 ** (3 + 3 * z[0]) / 2))
    return dict(units=units, spectral_radius=1.2 * z[1], input_scale=10 ** (-2 + 2.5 * z[2]),
                leak=0.05 + 0.95 * z[3], connectivity=0.05 + 0.95 * z[4],
                ridge_lambda=10 ** (-3 + 6 * z[5]))


def folds():
    if not _CACHE:
        for s, f in itertools.product(range(1, 10), range(1, 6)):
            _CACHE[(s, f)] = tuple(np.load(ROOT.format(k, s, f)) for k in
                                   ('X_train', 'y_train', 'X_val', 'y_val'))
    return _CACHE


def pooled(X, w_in, w, leak):
    """Log-variance over time of reservoir states and inputs (direct inputs)."""
    x = np.zeros((X.shape[0], w.shape[0]))
    states = np.empty((X.shape[0], X.shape[1], w.shape[0]))
    for t in range(X.shape[1]):
        x = (1 - leak) * x + leak * np.tanh(X[:, t, :] @ w_in + x @ w)
        states[:, t] = x
    f_res = np.log(states[:, WASHOUT:].var(axis=1) + 1e-8)
    if not DIRECT:
        return f_res
    f_in = np.log(X[:, WASHOUT:].var(axis=1) + 1e-8)
    return np.c_[f_res, f_in]


def ridge_val_mse(f_tr, y_tr, f_va, y_va, lam):
    mu, sd = f_tr.mean(0), f_tr.std(0) + 1e-8
    a = np.c_[(f_tr - mu) / sd, np.ones(len(f_tr))]
    t = 2.0 * np.eye(2)[y_tr.astype(int)] - 1.0
    pen = lam * np.eye(a.shape[1])
    pen[-1, -1] = 0.0
    w = np.linalg.solve(a.T @ a + pen, a.T @ t)
    s = np.c_[(f_va - mu) / sd, np.ones(len(f_va))] @ w
    t_va = 2.0 * np.eye(2)[y_va.astype(int)] - 1.0
    return ((s - t_va) ** 2).mean(), (s.argmax(1) == y_va).mean()


def objective(z):
    hp = decode(np.asarray(z))
    mses, accs = [], []
    for seed in SEEDS:
        for (s, f), (Xtr, ytr, Xva, yva) in folds().items():
            w_in, w = reservoir_weights(Xtr.shape[-1], hp['units'], hp['spectral_radius'], hp['input_scale'], hp['connectivity'], seed)
            mse, acc = ridge_val_mse(pooled(Xtr, w_in, w, hp['leak']), ytr,
                                     pooled(Xva, w_in, w, hp['leak']), yva, hp['ridge_lambda'])
            mses.append(mse)
            accs.append(acc)
    return float(np.mean(mses)), float(np.mean(accs))


if __name__ == '__main__':
    # start from the hand-picked setting used so far
    start = [(np.log2(16) - 3) / 3, 0.9 / 1.2, (np.log10(0.5) + 2) / 2.5, 1.0, (0.2 - 0.05) / 0.95,
             0.5]
    es = cma.CMAEvolutionStrategy(start, SIGMA0, {'bounds': [0, 1], 'popsize': POPSIZE,
                                                   'seed': 1, 'maxiter': MAX_GEN,
                                                   'verbose': -9})
    history = []
    with Parallel(n_jobs=POPSIZE) as pool:
        while not es.stop():
            cands = es.ask()
            scores = pool(delayed(objective)(c) for c in cands)
            es.tell(cands, [m for m, _ in scores])
            for c, (m, a) in zip(cands, scores):
                history.append(dict(generation=es.countiter, val_mse=m, val_acc=a, **decode(c)))
            best = min(history, key=lambda r: r['val_mse'])
            print('gen {:2d}  best val_mse {:.4f}  val_acc {:.3f}  {}'.format(
                es.countiter, best['val_mse'], best['val_acc'],
                {k: round(v, 3) for k, v in best.items() if k not in
                 ('generation', 'val_mse', 'val_acc')}), flush=True)
    pd.DataFrame(history).to_csv('results/cmaes_history{}.csv'.format(TAG), index=False)
    best['direct_inputs'] = DIRECT
    json.dump(best, open('results/cmaes_best{}.json'.format(TAG), 'w'), indent=1)
    print('best:', best)
