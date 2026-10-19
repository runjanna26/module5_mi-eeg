"""
Plot the output of every module of the proposed pipeline for real trials.

One subject (A03), cross-validation fold 1 of session T. Everything that is
learned (CSP filters, unit scale, readout) is fitted on the TRAINING fold, and
the plotted example trials come from the VALIDATION fold of that split. Session
E is not used.

Usage (from experiments/, after cmaes_search.py)
    python plot_module_outputs.py
Writes results/figures/fig7_module_outputs.png (slides) and
results/figures_report/fig7_module_outputs.png (report).
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

from mixnet.models.HWNet import reservoir_weights
from mixnet.preprocessing.BCIC2a import HW_prep, raw
from mixnet.preprocessing.config import CONSTANT

SUBJECT, FOLD, FS, WASHOUT = 3, 1, 100, 50
BAND = 1                      # 8-12 Hz, the mu band
CHS = CONSTANT['BCIC2a']['sel_chs']
C3, C4 = CHS.index('C3'), CHS.index('C4')
FILTER_BANK = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36),
               (36, 40)]
PREP = dict(bands=FILTER_BANK, n_pairs=1, trial_norm='band')      # final preprocessing

C_A, C_B = '#2a78d6', '#6250d6'
C_LEFT, C_RIGHT = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'


def run_reservoir(S, hp, seed=0):
    w_in, w = reservoir_weights(S.shape[-1], hp['units'], hp['spectral_radius'], hp['input_scale'], hp['connectivity'], seed)
    x = np.zeros((S.shape[0], hp['units']))
    states = np.empty((S.shape[0], S.shape[1], hp['units']))
    for t in range(S.shape[1]):
        x = (1 - hp['leak']) * x + hp['leak'] * np.tanh(S[:, t, :] @ w_in + x @ w)
        states[:, t] = x
    return states


def pooled(S, states):
    """Log-variance over time of every reservoir state (no direct inputs)."""
    return np.log(states[:, WASHOUT:].var(axis=1) + 1e-8)


def main():
    hp = json.load(open('results/cmaes_best_nodirect.json'))
    chs = raw.chanel_selection(CHS)
    X, y, _X_te, _y_te = HW_prep._load_bcic2a(SUBJECT, FS, 2, chs)   # session E unused
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    tr, va = list(skf.split(X, y))[FOLD - 1]

    # learned on the training fold only
    S_tr, state = HW_prep.fit_transform(X[tr], y[tr], pick_smp_freq=FS, **PREP)
    S_va = HW_prep.transform(X[va], state, pick_smp_freq=FS, **PREP)
    R_tr, R_va = run_reservoir(S_tr, hp), run_reservoir(S_va, hp)
    F_tr, F_va = pooled(S_tr, R_tr), pooled(S_va, R_va)
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-8
    a = np.c_[(F_tr - mu) / sd, np.ones(len(F_tr))]
    pen = hp['ridge_lambda'] * np.eye(a.shape[1])
    pen[-1, -1] = 0.0
    w = np.linalg.solve(a.T @ a + pen, a.T @ (2.0 * np.eye(2)[y[tr].astype(int)] - 1.0))
    Z_va = (F_va - mu) / sd
    score = np.c_[Z_va, np.ones(len(Z_va))] @ w
    margin = score[:, 1] - score[:, 0]          # > 0: right hand

    # one correctly classified validation trial per class, the most typical one
    yv = y[va].astype(int)
    pick = []
    for c in (0, 1):
        idx = np.where((yv == c) & ((margin > 0) == bool(c)))[0]
        pick.append(idx[np.argsort(np.abs(margin[idx] - np.median(margin[idx])))[0]])

    # intermediate signals of the two example trials
    Xv = X[va]
    band = HW_prep._bandpass(Xv[pick], *FILTER_BANK[BAND], FS)
    csp = np.einsum('kc,nct->nkt', state['filters'][BAND], band)
    norm = S_va[pick][:, :, 2 * BAND:2 * BAND + 2]        # after trial norm + unit scale
    t = np.arange(X.shape[-1]) / FS

    for report in (False, True):
        out_dir = 'results/figures_report' if report else 'results/figures'
        os.makedirs(out_dir, exist_ok=True)
        plt.rcParams.update({'font.family': 'sans-serif',
                             'font.sans-serif': ['Liberation Sans', 'DejaVu Sans'],
                             'font.size': 11, 'axes.spines.top': False,
                             'axes.spines.right': False, 'axes.edgecolor': '#c3c2b7',
                             'axes.labelcolor': INK2, 'xtick.color': INK2,
                             'ytick.color': INK2, 'axes.grid': False})
        fig = plt.figure(figsize=(12.8, 10.6) if report else (12.8, 13.0))
        gs = fig.add_gridspec(7, 2, height_ratios=[1, 1, 1, 1, 1.25, 1.0, 1.0],
                              hspace=0.75, wspace=0.24, left=0.215, right=0.99,
                              top=0.95 if report else 0.9, bottom=0.05)
        if not report:
            fig.text(0.01, 0.975, 'What each module outputs', fontsize=22, fontweight='bold',
                     va='top')
            fig.text(0.01, 0.94, 'Subject A03, two validation trials of fold 1. Everything '
                     'learned was fitted on the training fold.', fontsize=13, color=INK2,
                     va='top')
        row_titles = ['1  Raw EEG\n(C3, C4)', '2  Filter bank\n8–12 Hz band',
                      '3  CSP filters\n(8–12 Hz)', '4  Trial norm +\nunit scale',
                      '5  Reservoir\n(units × time)', '6  Log-variance\npooling']
        for col, (k, title) in enumerate(zip(pick, ['Left-hand imagery', 'Right-hand imagery'])):
            for row in range(6):
                ax = fig.add_subplot(gs[row, col])
                if row == 0:
                    ax.set_title(title, fontsize=13, fontweight='bold', color=INK)
                if row in (0, 1, 2, 3):
                    if row == 0:
                        a1, a2, l1, l2 = Xv[k, C3], Xv[k, C4], 'C3', 'C4'
                    elif row == 1:
                        a1, a2, l1, l2 = band[col, C3], band[col, C4], 'C3', 'C4'
                    elif row == 2:
                        a1, a2, l1, l2 = csp[col, 0], csp[col, 1], 'filter 1', 'filter 2'
                    else:
                        a1, a2, l1, l2 = norm[col, :, 0], norm[col, :, 1], 'filter 1', \
                            'filter 2'
                    off = 3.0 * max(np.std(a1), np.std(a2))
                    ax.plot(t, a1 + off, color=C_A, lw=0.8)
                    ax.plot(t, a2 - off, color=C_B, lw=0.8)
                    ax.set_yticks([off, -off])
                    ax.set_yticklabels(['{} · var {:.3g}'.format(l1, np.var(a1)),
                                        '{} · var {:.3g}'.format(l2, np.var(a2))],
                                       fontsize=8.5)
                    ax.set_xlim(0, 4)
                    ax.tick_params(axis='x', labelsize=9)
                    if row == 3:
                        ax.set_xlabel('time in the imagery window (s)', fontsize=9)
                    else:
                        ax.set_xticklabels([])
                elif row == 4:
                    states = R_va[k].T
                    lim = np.abs(states[:, WASHOUT:]).max()
                    ax.imshow(states, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim,
                              extent=[0, 4, states.shape[0], 0])
                    ax.text(1.0, 1.04, 'colour range ±{:.3f}'.format(lim), transform=ax.transAxes,
                            ha='right', va='bottom', fontsize=8.5, color=INK2)
                    ax.axvspan(0, WASHOUT / FS, color='white', alpha=0.6)
                    ax.text(WASHOUT / FS / 2, states.shape[0] / 2, 'washout', rotation=90,
                            ha='center', va='center', fontsize=8, color=INK2)
                    ax.set_yticks([0, states.shape[0] - 1])
                    ax.set_yticklabels(['unit 1', 'unit {}'.format(states.shape[0])],
                                       fontsize=8.5)
                    ax.set_xlabel('time (s)', fontsize=9)
                    ax.tick_params(axis='x', labelsize=9)
                else:
                    z = Z_va[k]
                    ax.bar(np.arange(len(z)), z, 0.8, color=C_B)
                    ax.axhline(0, color=INK2, lw=0.8)
                    ax.set_ylim(-3.5, 3.5)
                    ax.set_xlim(-1, len(z))
                    ax.set_xticks([0, len(z) - 1])
                    ax.set_xticklabels(['unit 1', 'unit {}'.format(len(z))], fontsize=8.5)
                    ax.set_yticks([-3, 0, 3])
                    ax.tick_params(axis='y', labelsize=8.5)
                    ax.set_ylabel('standardised', fontsize=8.5)
                if col == 0:
                    pos = ax.get_position()
                    fig.text(0.005, pos.y0 + pos.height / 2, row_titles[row], ha='left',
                             va='center', fontsize=10.5, fontweight='bold', color=INK)
        ax = fig.add_subplot(gs[6, :])
        rng = np.random.RandomState(0)
        for c, colour, lab in ((0, C_LEFT, 'left-hand trials'), (1, C_RIGHT, 'right-hand trials')):
            m = margin[yv == c]
            ax.scatter(m, c + rng.uniform(-0.15, 0.15, len(m)), s=28, color=colour, label=lab,
                       edgecolor='white', linewidth=0.6, zorder=3)
        ax.scatter(margin[pick], [0, 1], s=120, facecolor='none', edgecolor=INK, linewidth=1.4,
                   zorder=4, label='the two trials above')
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['true left', 'true right'], fontsize=9)
        ax.set_xlabel('readout output: right-hand score − left-hand score '
                      '(> 0 → predicted right)', fontsize=9)
        ax.tick_params(axis='x', labelsize=9)
        ax.set_ylim(-0.6, 1.6)
        ax.legend(loc='upper left', fontsize=8.5, frameon=False, ncol=3)
        pos = ax.get_position()
        fig.text(0.005, pos.y0 + pos.height / 2, '7  Ridge readout\n(validation fold)',
                 ha='left', va='center', fontsize=10.5, fontweight='bold', color=INK)
        path = os.path.join(out_dir, 'fig7_module_outputs.png')
        fig.savefig(path, dpi=200 if report else 150, facecolor='white')
        plt.close(fig)
        print('Written:', path)
    correct = int(((margin > 0) == yv).sum())
    print('validation accuracy of this fold: {:.3f}'.format(correct / len(yv)))
    json.dump({'n_val': int(len(yv)), 'n_correct': correct,
               'var_left': [float(np.var(csp[0, 0])), float(np.var(csp[0, 1]))],
               'var_right': [float(np.var(csp[1, 0])), float(np.var(csp[1, 1]))],
               'state_max_abs': float(np.abs(R_va[:, WASHOUT:]).max()),
               'state_rms': float(np.sqrt((R_va[:, WASHOUT:] ** 2).mean()))},
              open('results/module_outputs_info.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
