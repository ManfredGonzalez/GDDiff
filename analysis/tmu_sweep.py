"""Privacy-utility curve over the fusion step t_mu (operating point lambda) of Mean-GDDiff (reviewer R2).

Collects the sweep variants lam<lambda> (and 'full' = lambda 0.5) of ablation_metrics.py / ablation_utility.py,
all run on the same 300 images with identical settings and seed, and adds the fusion step
t_mu = 10 * (T - 2 - idx), idx = int(T * (1 - (lambda + 0.1))), as computed by the sampler (T = 100 steps).
Gauss- and Lpl-GDDiff (no average-face branch) are shown as horizontal references.

Usage (env: gaze):  python analysis/tmu_sweep.py   -> results_ablation/tmu_sweep.csv, tmu_sweep.pdf
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_metrics import OUT, VARIANTS  # noqa: E402

T = 100
UTIL = {'gaze_error_deg': 'Gaze dev. (deg)', 'pose_error_deg': 'Pose dev. (deg)',
        'expression_changed': 'Expression change rate', 'age_diff_years': 'Age dev. (years)',
        'gender_flipped': 'Gender flip rate'}


def t_mu(lam):
    per = 1 - (lam + 0.1)  # as in diffusion.py, then get_percentile_index(time_pairs, 100 * per)
    idx = min(int(T * ((100 * per) / 100)), T - 1)
    return 10 * (T - 2 - idx)


def main():
    q = pd.read_csv(os.path.join(OUT, 'ablation_summary.csv')).set_index('variant')
    u = pd.read_csv(os.path.join(OUT, 'utility/ablation_utility_summary.csv'))
    u = u.groupby('metric')[[c for c in u if c.startswith('mean_')]].mean().T  # two estimators averaged
    u.index = u.index.str.replace('mean_', '', regex=False)
    df = q[['auc', 'f1_eer', 'mse', 'gacc_rafd', 'self_cos']].join(u[list(UTIL)])
    sweep = {v: float(VARIANTS[v][2]) for v in VARIANTS if v.startswith('lam')}
    sweep['full'] = 0.5
    s = df.loc[[v for v in sweep if v in df.index]].copy()
    s.insert(0, 'lambda', [sweep[v] for v in s.index])
    s.insert(1, 't_mu', [t_mu(sweep[v]) for v in s.index])
    s = s.sort_values('lambda')
    s.to_csv(os.path.join(OUT, 'tmu_sweep.csv'))
    pd.set_option('display.width', 200)
    print(s.round(4).to_string())
    print('\nreferences (no average-face branch)')
    print(df.loc[['gauss', 'lapl']].round(4).to_string())

    cols = {'auc': 'Verification AUC', 'mse': 'MSE', 'gacc_rafd': 'gAcc (RaFD)', **UTIL}
    plt.rcParams['font.size'] = 12  # matplotlib default 10, +2 pt
    fig = plt.figure(figsize=(11.5, 5.2))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.07], wspace=0.42, hspace=0.38,
                          left=0.06, right=0.94, bottom=0.13, top=0.92)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]
    for ax, (c, label) in zip(axes, cols.items()):
        ax.plot(s['lambda'], s[c], 'o-', color='C0', ms=4, label='Mean-GDDiff')
        for ref, style in [('gauss', '--'), ('lapl', ':')]:
            ax.axhline(df.loc[ref, c], ls=style, color='gray', lw=1, label=f'{ref.capitalize()}-GDDiff')
        ax.set_title(label, fontsize=12)
        ax.set_xticks(s['lambda'][::2])
        ax.grid(alpha=0.3)
    for ax in axes[4:]:
        ax.set_xlabel(r'$\lambda$')
    axes[0].legend(fontsize=9)
    # single conversion bar for the whole figure: user parameter lambda (left) -> fusion step t_mu (right)
    bar = fig.add_subplot(gs[:, 4])
    bar.set_ylim(s['lambda'].max() + 0.05, s['lambda'].min() - 0.05)
    bar.set_xticks([])
    bar.set_yticks(s['lambda'])
    bar.set_yticklabels([f'{v:.1f}' for v in s['lambda']], fontsize=10)
    bar.set_title(r'$\lambda$' + '   ' + r'$t_\mu$', fontsize=12)
    right = bar.twinx()
    right.set_ylim(bar.get_ylim())
    right.set_yticks(s['lambda'])
    right.set_yticklabels([str(t) for t in s['t_mu']], fontsize=10)
    for v in s['lambda']:
        bar.axhline(v, color='gray', lw=0.6)
    fig.savefig(os.path.join(OUT, 'tmu_sweep.pdf'))
    fig.savefig(os.path.join(OUT, 'tmu_sweep.png'), dpi=150)
    print('saved', os.path.join(OUT, 'tmu_sweep.pdf'))


if __name__ == '__main__':
    main()
