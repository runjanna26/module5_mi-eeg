"""
Confusion matrices on session E for MixNet, FBCSP+SVM and HWNet.

Predictions are read from the files the run scripts already saved
(S*_prediction_results.npy: y_true and y_pred of every fold). Each model has 9
subjects x 5 folds x 144 session E trials = 6,480 predictions, pooled here.
Nothing is trained and nothing is fitted.

Usage (from experiments/, after the final HWNet run)
    python plot_confusion.py
Writes results/figures[_report]/fig9_confusion_models.png and
fig10_confusion_subjects.png, and results/confusion_summary.json.
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

RUNS = {'MixNet': glob.glob('logs/MixNet/*')[0],
        'FBCSP + SVM': 'logs_compare/FBCSP_SVM/subject_dependent_2_classes_BCIC2a',
        'HWNet (ESN)': 'logs/HWNet/subject_dependent_2_classes_BCIC2a'}
CLASSES = ['left hand', 'right hand']
SEQ = LinearSegmentedColormap.from_list('seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5',
                                                '#1c5cab', '#0d366b'])
INK, INK2 = '#0b0b0b', '#52514e'


def counts(y_true, y_pred):
    m = np.zeros((2, 2), dtype=int)
    for t, p in zip(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)):
        m[t, p] += 1
    return m


def load(run_dir):
    """{subject: 2x2 count matrix summed over the 5 folds}."""
    out = {}
    for f in sorted(glob.glob(run_dir + '/S*_prediction_results.npy')):
        subject = int(os.path.basename(f)[1:4])
        folds = np.load(f, allow_pickle=True)
        out[subject] = sum(counts(d['y_true'], d['y_pred']) for d in folds)
    return out


def draw_matrix(ax, m, title, size=1.0, fontsize=13):
    row = m / m.sum(axis=1, keepdims=True)
    ax.imshow(row, cmap=SEQ, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, '{:.1f} %\n({:,})'.format(100 * row[i, j], m[i, j]), ha='center',
                    va='center', fontsize=fontsize * size, linespacing=1.3,
                    color='white' if row[i, j] > 0.55 else INK)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES, fontsize=11 * size)
    ax.set_yticklabels(CLASSES, fontsize=11 * size, rotation=90, va='center')
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which='minor', color='white', lw=3)
    ax.tick_params(which='both', length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=13 * size, fontweight='bold', color=INK, pad=8)


def main():
    data = {name: load(d) for name, d in RUNS.items()}
    summary = {}
    for name, per in data.items():
        total = sum(per.values())
        summary[name] = {'matrix': total.tolist(),
                         'recall_left': float(total[0, 0] / total[0].sum()),
                         'recall_right': float(total[1, 1] / total[1].sum()),
                         'accuracy': float(np.trace(total) / total.sum()),
                         'predicted_right_share': float(total[:, 1].sum() / total.sum()),
                         'n_subjects': len(per),
                         'subject_recall_left': [float(m[0, 0] / m[0].sum()) for _, m in sorted(per.items())],
                         'subject_recall_right': [float(m[1, 1] / m[1].sum()) for _, m in sorted(per.items())]}
    json.dump(summary, open('results/confusion_summary.json', 'w'), indent=1)

    for report in (False, True):
        out_dir = 'results/figures_report' if report else 'results/figures'
        os.makedirs(out_dir, exist_ok=True)
        plt.rcParams.update({'font.family': 'sans-serif',
                             'font.sans-serif': ['Liberation Sans', 'DejaVu Sans'],
                             'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2})

        # 1 -- the three models side by side
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6 if report else 6.0))
        for ax, (name, per) in zip(axes, data.items()):
            total = sum(per.values())
            draw_matrix(ax, total, '{}  ·  accuracy {:.1f} %'.format(
                name, 100 * np.trace(total) / total.sum()))
            ax.set_xlabel('predicted class', fontsize=11)
        axes[0].set_ylabel('true class', fontsize=11)
        if report:
            fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.14, wspace=0.28)
        else:
            fig.subplots_adjust(left=0.07, right=0.99, top=0.72, bottom=0.14, wspace=0.28)
            fig.text(0.02, 0.95, 'Confusion matrices on the test session', fontsize=22,
                     fontweight='bold', va='top')
            fig.text(0.02, 0.88, 'Rows sum to 100 %. All 9 subjects × 5 folds × 144 trials per '
                     'model.', fontsize=14, color=INK2, va='top')
        path = os.path.join(out_dir, 'fig9_confusion_models.png')
        fig.savefig(path, dpi=200 if report else 150, facecolor='white')
        plt.close(fig)
        print('Written:', path)

        # 2 -- HWNet per subject
        per = data['HWNet (ESN)']
        fig, axes = plt.subplots(3, 3, figsize=(8.6, 8.4) if report else (10.0, 10.4))
        for ax, (s, m) in zip(axes.ravel(), sorted(per.items())):
            draw_matrix(ax, m, 'A{:02d}  ·  {:.0f} %'.format(s, 100 * np.trace(m) / m.sum()),
                        size=0.78, fontsize=11.5)
        if report:
            fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.04, wspace=0.3,
                                hspace=0.42)
        else:
            fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.04, wspace=0.3,
                                hspace=0.42)
            fig.text(0.02, 0.975, 'HWNet confusion matrix per subject', fontsize=22,
                     fontweight='bold', va='top')
            fig.text(0.02, 0.93, 'Rows sum to 100 %; 5 folds × 144 test trials per subject.',
                     fontsize=14, color=INK2, va='top')
        path = os.path.join(out_dir, 'fig10_confusion_subjects.png')
        fig.savefig(path, dpi=200 if report else 150, facecolor='white')
        plt.close(fig)
        print('Written:', path)
    for name, s in summary.items():
        print('{:<12} acc {:.3f}  recall left {:.3f}  recall right {:.3f}  predicts right {:.3f}'
              .format(name, s['accuracy'], s['recall_left'], s['recall_right'],
                      s['predicted_right_share']))


if __name__ == '__main__':
    main()
