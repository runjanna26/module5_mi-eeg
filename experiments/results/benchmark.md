Collecting results from 2 run(s):
  [ok]   HWNet: 9 subjects x up to 5 folds  (logs/HWNet/subject_dependent_2_classes_BCIC2a)
  [ok]   MixNet: 9 subjects x up to 5 folds  (logs/MixNet/subject_dependent_2_classes_BCIC2a_HistoricalTangentSlope_csp_components_2_margin_1.0_latent_dim_18_warmup_7)

## Per-subject results (mean over folds)

| subject | HWNet [test_acc] | MixNet [test_acc] | HWNet [f1-score] | MixNet [f1-score] |
|---------|------------------|-------------------|------------------|-------------------|
| 1       | 0.8792           | 0.8875            | 0.8783           | 0.8861            |
| 2       | 0.5472           | 0.5528            | 0.5343           | 0.5354            |
| 3       | 0.9083           | 0.9306            | 0.9074           | 0.9304            |
| 4       | 0.6250           | 0.6972            | 0.6060           | 0.6867            |
| 5       | 0.7514           | 0.6319            | 0.7485           | 0.5930            |
| 6       | 0.6458           | 0.6042            | 0.6388           | 0.5880            |
| 7       | 0.9153           | 0.8847            | 0.9151           | 0.8828            |
| 8       | 0.9222           | 0.9278            | 0.9222           | 0.9276            |
| 9       | 0.8500           | 0.8667            | 0.8482           | 0.8656            |

## Summary across subjects (mean +/- std)

| run    | n_subjects | test_acc_mean | test_acc_std | f1-score_mean | f1-score_std |
|--------|------------|---------------|--------------|---------------|--------------|
| HWNet  | 9          | 0.7827        | 0.1443       | 0.7776        | 0.1502       |
| MixNet | 9          | 0.7759        | 0.1524       | 0.7662        | 0.1629       |

Written: results/benchmark_per_subject.csv and results/benchmark_summary.csv

### Paired comparison on test_acc (9 subjects)
HWNet                       : 0.7827
MixNet                      : 0.7759
difference (cand - base)    : +0.0068
candidate wins on           : 3/9 subjects
paired t-test               : t = 0.3829, p = 0.7118
Wilcoxon signed-rank        : W = 22.0000, p = 1.0000

### Paired comparison on f1-score (9 subjects)
HWNet                       : 0.7776
MixNet                      : 0.7662
difference (cand - base)    : +0.0115
candidate wins on           : 3/9 subjects
paired t-test               : t = 0.5279, p = 0.6119
Wilcoxon signed-rank        : W = 22.0000, p = 1.0000
