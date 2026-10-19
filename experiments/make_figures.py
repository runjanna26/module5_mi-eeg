"""Draw the result figures from the saved logs, in two versions.

Usage (from experiments/, after benchmark.py and the tuning runs)
    python make_figures.py
Writes
    results/figures/         16:9 PNGs with title and subtitle, for slides
    results/figures_report/  the same charts without titles, sized for report.pdf
"""
import glob
import json
import os
import re

import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIXNET_DIR = glob.glob('logs/MixNet/*')[0]
HWNET_DIR = 'logs/HWNet/subject_dependent_2_classes_BCIC2a'
TUNE = 'logs_tuning/HWNet/subject_dependent_2_classes_BCIC2a_'
FBCSP_DIR = 'logs_compare/FBCSP_SVM/subject_dependent_2_classes_BCIC2a'
BAND_ATT_DIR = '../archive/band_attention/experiments/logs_HWNet/subject_dependent_2_classes_BCIC2a'
FINAL = 'rn_final'
CMA = json.load(open('results/cmaes_best_nodirect.json'))
N_UNITS = int(CMA['units'])
N_TRAIN = 2 * N_UNITS + 2                 # readout without direct inputs
N_FIXED = 18 * N_UNITS + N_UNITS ** 2
# colours: fixed per model in every figure
C_MIX, C_HW, C_HW_LIGHT, C_CTRL, C_ATT = '#2a78d6', '#eb6834', '#f3b79d', '#898781', '#c98500'
C_FB = '#1baf7a'
C_RAND, C_RAND_FILL = '#6250d6', '#e6e2fa'
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
SUBJ = ['A{:02d}'.format(i) for i in range(1, 10)]


def test_acc(run_dir):
    return np.array([pd.read_csv(f)['test_acc'].mean()
                     for f in sorted(glob.glob(run_dir + '/S*_all_results.csv'))])


def val_acc(run_dir):
    """Validation accuracy at the checkpointed epoch, mean over folds per subject."""
    out = {}
    for f in sorted(glob.glob(run_dir + '/S*_fold*_out_log.csv')):
        subject = int(re.search(r'S(\d+)_', os.path.basename(f)).group(1))
        log = pd.read_csv(f)
        out.setdefault(subject, []).append(log['val_acc'][log['val_loss'].idxmin()])
    return np.array([np.mean(out[k]) for k in sorted(out)])


ABLATIONS = [  # (label, run directory suffix): final configuration with one change
    ('Without per-trial band normalisation', 'rn_ablate_trial_norm'),
    ('Without unit scaling', 'rn_ablate_unit_scale'),
    ('Without the reservoir', 'rn_ablate_reservoir'),
    ('Mean instead of log-variance pooling', 'rn_ablate_logvar_pool'),
    ('Without standardisation', 'rn_ablate_standardise'),
    ('With direct inputs added', 'rn_add_direct_inputs')]


def fbcsp_acc():
    return np.array([pd.read_csv(f)['accuracy'].mean()
                     for f in sorted(glob.glob(FBCSP_DIR + '/S*_all_results.csv'))])


def n_params(variant):
    """Trained parameters of a run of the final family (rn_*)."""
    if variant.endswith('_nores') or variant.endswith('ablate_reservoir'):
        return 2 * 18 + 2                       # readout on the 18 pooled inputs
    if variant.endswith('direct_inputs'):
        return 2 * (N_UNITS + 18) + 2
    return N_TRAIN


def pct(v):
    return '{:.1f}'.format(100 * v)


def draw(report):
    """report=False: slide figures. report=True: untitled figures for the PDF."""
    out_dir = 'results/figures_report' if report else 'results/figures'
    surface = 'white' if report else SURFACE
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'DejaVu Sans'],
        'font.size': 15, 'figure.facecolor': surface, 'axes.facecolor': surface,
        'savefig.facecolor': surface, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
        'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
        'axes.axisbelow': True, 'xtick.major.size': 0, 'ytick.major.size': 0})

    def new_fig(title, subtitle, report_size, left=0.08, right=0.97, bottom=0.11):
        if report:
            return plt.subplots(figsize=report_size)
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        fig.subplots_adjust(left=left, right=right, top=0.82, bottom=bottom)
        fig.text(0.08, 0.94, title, fontsize=23, fontweight='bold', va='top')
        fig.text(0.08, 0.875, subtitle, fontsize=15, color=INK2, va='top')
        return fig, ax

    def save(fig, name, tight=True):
        os.makedirs(out_dir, exist_ok=True)
        if report and tight:
            fig.tight_layout(pad=0.4)
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=200 if report else 150)
        plt.close(fig)
        print('Written:', path)

    mix, hw = test_acc(MIXNET_DIR), test_acc(HWNET_DIR)

    # ---- 0. method diagram: MixNet baseline above, HWNet (ESN) below ------------
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = new_fig('The two pipelines: MixNet baseline and the echo state network',
                      'Orange = trained on the training fold. Violet = random, never trained. '
                      'Grey = fixed operation. "new" = module not in MixNet.',
                      (12.8, 5.9), left=0.02, right=0.98, bottom=0.03)
    ax.set_xlim(0, 100)
    ax.set_ylim(1.5, 53)
    ax.axis('off')
    bw, bh, gap, x0 = 10.7, 9.0, 2.0, 0.2
    learned = dict(fc='#fbe3d8', ec=C_HW)
    fixed = dict(fc='#f0efec', ec=AXIS)
    random_ = dict(fc=C_RAND_FILL, ec=C_RAND)
    cx = [x0 + bw / 2 + i * (bw + gap) for i in range(8)]

    def box(xc, yc, head, lines, style):
        ax.add_patch(FancyBboxPatch((xc - bw / 2, yc - bh / 2), bw, bh, lw=1.5,
                                    boxstyle='round,pad=0,rounding_size=0.9', **style))
        n = len(lines)
        ax.text(xc, yc + 1.25 * n, head, ha='center', va='center', fontsize=11.5,
                fontweight='bold', color=INK)
        for k, line in enumerate(lines):
            ax.text(xc, yc + 1.25 * n - 2.5 * (k + 1), line, ha='center', va='center',
                    fontsize=10, color=INK2)

    def arrow(p0, p1, rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=13, lw=1.5,
                                     color=INK2, shrinkA=0, shrinkB=0,
                                     connectionstyle='arc3,rad={}'.format(rad)))

    def row(yc, boxes):
        for i, (head, lines, style) in enumerate(boxes):
            box(cx[i], yc, head, lines, style)
            if i:
                arrow((cx[i - 1] + bw / 2, yc), (cx[i] - bw / 2, yc))

    shared = [('EEG trial', ['20 channels', '× 400 samples'], fixed),
              ('Filter bank', ['9 band-passes', '4–40 Hz'], fixed),
              ('CSP per band', ['2 spatial filters', 'per band'], learned),
              ('Signals', ['18 time series', '× 400 samples'], fixed)]
    ya, yaux, yb = 43.0, 30.0, 13.5
    ax.text(x0, 50.6, '(a) MixNet baseline — 44,831 parameters in training, 30,161 at test time',
            fontsize=12.5, fontweight='bold', color=INK, va='center')
    row(ya, shared + [('Conv block 1', ['18 filters, 1 × 64', 'pool to 100'], learned),
                      ('Conv block 2', ['9 filters, 1 × 32', 'pool to 25'], learned),
                      ('Latent vector', ['Dense(18)'], learned),
                      ('Classifier', ['Dense(2), softmax', 'cross-entropy'], learned)])
    box(cx[5], yaux, 'Decoder', ['rebuilds the 18', 'signals, MSE loss'], learned)
    box(cx[7], yaux, 'Triplet loss', ['same class close,', 'other class far'], fixed)
    xl = cx[6]
    ax.plot([xl, xl], [ya - bh / 2, yaux], color=INK2, lw=1.5, solid_capstyle='round')
    arrow((xl, yaux), (cx[5] + bw / 2, yaux))
    arrow((xl, yaux), (cx[7] - bw / 2, yaux))
    ax.text(cx[4] + bw / 2, yaux, 'training only: three losses,\nweighted by adaptive\n'
            'gradient blending', ha='right', va='center', fontsize=10.5, color=INK2,
            linespacing=1.25)

    ax.text(x0, 21.1, '(b) HWNet, proposed echo state network — {} trained parameters, '
            '{:,} fixed random weights'.format(N_TRAIN, N_FIXED), fontsize=12.5,
            fontweight='bold', color=INK, va='center')
    row(yb, shared[:2] + [('CSP per band', ['2 spatial filters', 'per band'], learned),
                          ('Normalise', ['per trial and band,', 'then unit scale'], learned),
                          ('Reservoir', ['{} leaky units'.format(N_UNITS), 'random, fixed'], random_),
                          ('Log-variance', ['over time, of each', 'reservoir state'], fixed),
                          ('Standardise', ['training-fold', 'mean and std'], learned),
                          ('Ridge readout', ['closed form,', 'λ from CMA-ES'], learned)])
    for i in (3, 4, 5, 6, 7):        # modules that are new compared with MixNet
        ax.text(cx[i] + bw / 2 - 0.4, yb + bh / 2 + 0.3, 'new', ha='right', va='bottom',
                fontsize=9.5, fontweight='bold', color=C_HW)
    save(fig, 'fig0_method_diagram.png')

    # ---- 1. per-subject accuracy -----------------------------------------------
    fb = fbcsp_acc()
    fig, ax = new_fig('Session E accuracy per subject',
                      'Mean of 5 folds. MixNet: 44,831 parameters. HWNet (echo state network): '
                      '{} trained parameters. FBCSP+SVM: repository baseline.'.format(N_TRAIN), (12.8, 4.3))
    x = np.arange(10, dtype=float) * 1.15
    x[-1] += 0.5
    w = 0.3
    series = [(np.r_[mix, mix.mean()], C_MIX, 'MixNet'),
              (np.r_[fb, fb.mean()], C_FB, 'FBCSP + SVM'),
              (np.r_[hw, hw.mean()], C_HW, 'HWNet (ESN)' if report else 'HWNet (ESN, mine)')]
    for k, (vals, colour, label) in enumerate(series):
        off = (k - 1) * (w + 0.02)
        ax.bar(x + off, 100 * vals, w, color=colour, label=label)
        for xi, v in zip(x, vals):
            ax.text(xi + off, 100 * v + 1.5, '{:.0f}'.format(100 * v), ha='center',
                    fontsize=10.5, color=INK2)
    ax.axhline(50, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.set_xlim(x[0] - 1.1, x[-1] + 0.7)
    ax.text(x[0] - 1.05, 52, 'chance', color=MUTED, fontsize=12, ha='left')
    ax.set_xticks(x)
    ax.set_xticklabels(SUBJ + ['Mean'])
    ax.get_xticklabels()[-1].set_fontweight('bold')
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel('Test accuracy (%)')
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=14)
    save(fig, 'fig1_accuracy_per_subject.png')

    # ---- 2. per-subject difference ---------------------------------------------
    diff = 100 * (hw - mix)
    order = np.argsort(diff)[::-1]
    n_pos = int((diff > 0).sum())
    fig, ax = new_fig('HWNet minus MixNet, per subject',
                      'Accuracy difference in points. Bar colour = the model that is ahead.',
                      (6.4, 4.6))
    ax.bar(range(9), diff[order], 0.62, color=[C_HW if d > 0 else C_MIX for d in diff[order]])
    span = max(abs(diff).max(), 1.0)
    for i, d in enumerate(diff[order]):
        ax.text(i, d + (0.03 if d > 0 else -0.03) * span, '{:+.1f}'.format(d).replace('-', '−'),
                ha='center', va='bottom' if d > 0 else 'top',
                fontsize=12 if report else 13.5, color=INK)
    ax.axhline(0, color=INK2, lw=1.2)
    ax.set_xticks(range(9))
    ax.set_xticklabels([SUBJ[i] for i in order], fontsize=12.5 if report else 15)
    lo, hi = min(diff.min(), 0) - 0.22 * span, max(diff.max(), 0) + 0.38 * span
    ax.set_ylim(lo, hi)
    ax.set_ylabel('HWNet − MixNet (points)' if report else 'Accuracy difference (points)')
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.spines['bottom'].set_visible(False)
    if n_pos:
        ax.text((n_pos - 1) / 2, hi - 0.09 * span, 'HWNet ahead', color=INK2, fontsize=13,
                ha='center')
    if n_pos < 9:
        ax.text((n_pos + 8) / 2, hi - 0.09 * span, 'MixNet ahead', color=INK2, fontsize=13,
                ha='center')
    save(fig, 'fig2_difference_per_subject.png')

    # ---- 3. validation vs test -------------------------------------------------
    mix_v, hw_v = val_acc(MIXNET_DIR).mean(), val_acc(HWNET_DIR).mean()
    fig, ax = new_fig('From validation to the test session',
                      'Mean accuracy over 9 subjects. Validation: same day as training '
                      '(session T). Test: another day (session E).', (6.4, 4.6))
    hw_top_val, hw_top_test = hw_v >= mix_v, hw.mean() >= mix.mean()
    for v, t, c, name, top_v, top_t in [(mix_v, mix.mean(), C_MIX, 'MixNet', not hw_top_val,
                                         not hw_top_test),
                                        (hw_v, hw.mean(), C_HW, 'HWNet', hw_top_val, hw_top_test)]:
        ax.plot([0, 1], [100 * v, 100 * t], color=c, lw=3, marker='o', ms=11,
                markeredgecolor=surface, markeredgewidth=2)
        drop = '{:+.1f}'.format(100 * (t - v)).replace('-', '−')
        ax.text(-0.07, 100 * v + (0.5 if top_v else -0.5), '{}\n{}'.format(name, pct(v)),
                ha='right', va='bottom' if top_v else 'top', fontsize=13.5, color=INK,
                linespacing=1.15)
        ax.text(1.07, 100 * t + (0.3 if top_t else -0.3), '{}\n({})'.format(pct(t), drop),
                ha='left', va='bottom' if top_t else 'top', fontsize=13.5, color=INK,
                linespacing=1.15)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(70, 92)
    ax.set_yticks(range(70, 91, 5))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Validation folds', 'Test session'], fontsize=13.5 if report else 15)
    ax.set_ylabel('Accuracy (%)')
    ax.yaxis.grid(True, color=GRID, lw=1)
    save(fig, 'fig3_validation_vs_test.png')

    # ---- 4. leave-one-out ablation of every new module --------------------------
    rows = [('Final: all modules', HWNET_DIR)] + [(lab, TUNE + v) for lab, v in ABLATIONS]
    val = [100 * val_acc(d).mean() for _, d in rows]
    tst = [100 * test_acc(d).mean() for _, d in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2) if report else (12.8, 7.2),
                             sharey=True, gridspec_kw=dict(wspace=0.08))
    if not report:
        fig.subplots_adjust(left=0.30, right=0.97, top=0.78, bottom=0.11)
        fig.text(0.02, 0.94, 'What each new module contributes', fontsize=23,
                 fontweight='bold', va='top')
        fig.text(0.02, 0.875, 'The final model with one module removed at a time. Test '
                 'accuracy read after the final model was fixed.', fontsize=15, color=INK2,
                 va='top')
    y = np.arange(len(rows))[::-1]
    for ax, vals, name, lim in [(axes[0], val, 'Validation accuracy (%)', (50, 92)),
                                (axes[1], tst, 'Test accuracy (%), session E', (50, 82))]:
        ax.barh(y, vals, 0.62, color=[C_HW] + [C_HW_LIGHT] * (len(rows) - 1))
        for yi, v in zip(y, vals):
            if v - lim[0] < 8:      # short bar: put the value outside, right of the bar
                ax.text(v + 0.5, yi, '{:.1f}'.format(v), va='center', ha='left', fontsize=12.5,
                        color=INK)
            else:
                ax.text(v - 0.25, yi, '{:.1f}'.format(v), va='center', ha='right',
                        fontsize=12.5, color=INK)
        ax.axvline(vals[0], color=INK2, lw=1.2, ls=(0, (2, 2)))
        ax.set_xlim(*lim)
        ax.set_xlabel(name + ', axis starts at 50', fontsize=13)
        ax.xaxis.grid(True, color=GRID, lw=1)
        ax.spines['bottom'].set_visible(False)
    axes[1].axvline(100 * mix.mean(), color=C_MIX, lw=2, ls=(0, (4, 3)))
    axes[1].text(100 * mix.mean() - 0.2, y[-1] - 0.75, 'MixNet', color=C_MIX, fontsize=12,
                 ha='right', va='center')
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([n for n, _ in rows], fontsize=13)
    axes[0].get_yticklabels()[0].set_fontweight('bold')
    axes[0].set_ylim(-1.2, len(rows) - 0.4)
    if report:
        fig.subplots_adjust(left=0.27, right=0.99, top=0.98, bottom=0.14)
    save(fig, 'fig4_module_ablation.png', tight=False)

    # ---- 5. accuracy against trained parameters (post hoc) ----------------------
    fig, ax = new_fig('Accuracy against number of trained parameters',
                      'Mean session E accuracy of every model that was trained. Read after '
                      'selection; log scale.', (12.8, 4.8))
    pts = []
    for d in sorted(glob.glob(TUNE + 'rn_*')):
        v = d[len(TUNE):]
        if 'seed' in v or v == FINAL or ('ablate_' in v and v != 'rn_ablate_reservoir') \
                or v.endswith('direct_inputs'):
            continue
        kind = 'control' if n_params(v) == 38 else 'esn'
        pts.append((n_params(v), 100 * test_acc(d).mean(), kind, v))
    for kind, colour, marker, label in [('esn', C_HW, 'o', 'Echo state network variants'),
                                        ('control', C_CTRL, 's', 'No-reservoir controls')]:
        sel = [p for p in pts if p[2] == kind]
        ax.scatter([p[0] for p in sel], [p[1] for p in sel], s=70, color=colour, marker=marker,
                   edgecolor=surface, linewidth=1.5, label=label, zorder=3,
                   alpha=0.55 if kind == 'esn' else 1.0)
    band = 100 * test_acc(BAND_ATT_DIR).mean()
    ax.scatter([N_TRAIN], [100 * hw.mean()], s=190, color=C_HW, edgecolor=INK, linewidth=1.6,
               zorder=4, label='Final ESN (selected on validation)')
    ax.scatter([140], [band], s=130, color=C_ATT, marker='D', edgecolor=surface,
               linewidth=1.5, zorder=3, label='Band-attention net (earlier)')
    ax.scatter([44831], [100 * mix.mean()], s=150, color=C_MIX, marker='^', edgecolor=surface,
               linewidth=1.5, zorder=3, label='MixNet')
    ax.annotate('Final ESN (plain CSP)\n{} params, {}'.format(N_TRAIN, pct(hw.mean())),
                (N_TRAIN, 100 * hw.mean()), xytext=(150, -34), textcoords='offset points',
                ha='left', va='top', fontsize=12.5,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    ax.annotate('MixNet\n44,831 params, {}'.format(pct(mix.mean())), (44831, 100 * mix.mean()),
                xytext=(-12, -4), textcoords='offset points', ha='right', va='top', fontsize=12.5)
    ax.annotate('Band attention\n140 params, {:.1f}'.format(band), (140, band), xytext=(12, -2),
                textcoords='offset points', ha='left', va='top', fontsize=12.5)
    for v, lab in (('rn_csp_fixed', 'fixed shrinkage'), ('rn_csp_lw', 'Ledoit-Wolf'),
                   ('rn_csp_scsp', 'stationary CSP'), ('rn_csp_fixed_nores', 'fixed, no res.'),
                   ('rn_csp_lw_nores', 'Ledoit-Wolf, no res.'),
                   ('rn_csp_scsp_nores', 'stationary, no res.'),
                   ('rn_ablate_reservoir', 'plain CSP, no res.')):
        d = TUNE + v
        if glob.glob(d + '/S*_all_results.csv'):
            ax.annotate(lab, (n_params(v), 100 * test_acc(d).mean()), xytext=(8, 0),
                        textcoords='offset points', va='center', fontsize=10.5, color=INK2)
    ax.set_xscale('log')
    ax.set_xlim(9, 90000)
    lo = min(p[1] for p in pts)
    ax.set_ylim(np.floor(lo) - 1, 82)
    ax.set_xlabel('Trained parameters (log scale)')
    ax.set_ylabel('Test accuracy (%)')
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.legend(loc='lower right', frameon=False, fontsize=12)
    save(fig, 'fig5_accuracy_vs_parameters.png')

    # ---- 8. CMA-ES search (validation only) ------------------------------------
    hist = pd.read_csv('results/cmaes_history_nodirect.csv')
    best = hist.loc[hist['val_mse'].idxmin()]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4) if report else (12.8, 7.2),
                             gridspec_kw=dict(wspace=0.25))
    if not report:
        fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.12)
        fig.text(0.02, 0.94, 'CMA-ES search of the reservoir settings', fontsize=23,
                 fontweight='bold', va='top')
        fig.text(0.02, 0.875, 'Validation folds only: 300 settings, each scored over 9 subjects '
                 '× 5 folds × 3 random reservoirs.', fontsize=15, color=INK2, va='top')
    ax = axes[0]
    ax.scatter(hist['generation'], hist['val_mse'], s=18, color=C_HW_LIGHT, zorder=2,
               label='one setting')
    run_best = hist.groupby('generation')['val_mse'].min().cummin()
    ax.plot(run_best.index, run_best.values, color=C_HW, lw=2.5, zorder=3,
            label='best so far')
    # objective of the hand-picked starting setting (16 units, radius 0.9, input 0.5,
    # no leak, connectivity 0.2, lambda 1), computed with cmaes_search.objective
    import cmaes_search
    z0 = [(np.log2(16) - 3) / 3, 0.75, (np.log10(0.5) + 2) / 2.5, 1.0, 0.15 / 0.95, 0.5]
    START_MSE = cmaes_search.objective(z0)[0]
    json.dump({'start_mse': START_MSE}, open('results/cmaes_start.json', 'w'))
    ax.axhline(START_MSE, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(25, START_MSE + 0.003, 'hand-picked start', ha='right', va='bottom', fontsize=11,
            color=INK2)
    ax.set_xlabel('CMA-ES generation')
    ax.set_ylabel('Validation MSE (lower is better)')
    ax.set_ylim(hist['val_mse'].min() - 0.01, 0.70)
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.legend(frameon=False, fontsize=12, loc='upper center')
    ax = axes[1]
    ax.scatter(2 * hist['units'] + 2, 100 * hist['val_acc'], s=20, color=C_HW_LIGHT,
               zorder=2)
    ax.scatter([2 * best['units'] + 2], [100 * best['val_acc']], s=160,
               facecolor='none', edgecolor=INK, linewidth=1.6, zorder=3)
    ax.annotate('selected: {} units,\n{} trained parameters'.format(
        int(best['units']), int(2 * best['units'] + 2)),
        (2 * best['units'] + 2, 100 * best['val_acc']), xytext=(-14, 10),
        textcoords='offset points', ha='right', fontsize=12)
    ax.set_xlabel('Trained parameters of the readout')
    ax.set_ylabel('Validation accuracy (%)')
    ax.set_ylim(70, 84)
    ax.yaxis.grid(True, color=GRID, lw=1)
    if report:
        fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.14)
    save(fig, 'fig8_cmaes_search.png', tight=False)


if __name__ == '__main__':
    draw(report=False)
    draw(report=True)
