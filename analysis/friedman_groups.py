"""Separate Friedman tests for privacy, image quality and data utility (editor request),
plus an equal-weight overall ranking, for the 13 methods of the paper.

Blocks (each block ranks all methods; rank 1 = best, ties get average ranks):
  privacy   AUC of the 4 verifiers (AdaFace, TransFace, SwinFace, ArcFace) x 4 datasets = 16 blocks,
            read from Table tab:iv_multiverifier in main.tex (lower AUC = more private)
  quality   FD (Table tab:iqdu_all_datasets in main.tex) and MSE (full-precision per-image means from
            the toolkit's mse_<ds>_<method>.csv) x 4 datasets = 8 blocks (lower = better)
  utility   gaze, head pose, expression, age, gender x 4 datasets = 20 blocks; deviation of the
            de-identified prediction from the original prediction (lower = better). Each attribute has
            two estimators; their ranks are averaged and re-ranked so an attribute counts once per dataset.
Overall: mean of the three group mean ranks (equal weight per group).
Critical difference: Bonferroni-Dunn, CD = q_a * sqrt(k(k+1) / (6N)), q_a = z_{1 - a/(2(k-1))}.

Usage (env: gaze):  python analysis/friedman_groups.py <main.tex> [--out analysis/results_friedman]
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm, rankdata

TK = os.path.expanduser('~/Desktop/master_research/deid-toolkit/root_dir/results')
DATASETS = ['rafd', 'xm2vts_aligned', 'LFW_unified_orig', 'CALFW']  # column order of the paper tables
METHODS = {  # paper name -> toolkit technique label
    'Mean-GDDiff (0.1)': '0.1_mean_GauDeiDiff', 'Mean-GDDiff (0.2)': '0.2_mean_GauDeiDiff',
    'Mean-GDDiff (0.5)': '0.5_mean_GauDeiDiff', 'DeepPrivacy': 'deepprivacy2', 'FAMS': 'face_anon_diff',
    'Gauss-GDDiff': 'gauss_GauDeiDiff', 'CLEANIR': 'cleanir', 'Lpl-GDDiff': 'lpl_GauDeiDiff',
    'CPP-DeID (0.9)': '0.9_cpp-deid', 'FADM': 'fadm', 'LDFA': 'ldfa', 'CPP-DeID (0.5)': '0.5_cpp-deid',
    'AMT-GAN': 'AMT-GAN'}
UTILITY = {  # attribute -> list of (summary csv, model, column)
    'gaze': [('gaze/gaze_summary.csv', 'gaze3d', 'angular_error_mean'), ('gaze/gaze_summary.csv', 'gazetr', 'angular_error_mean')],
    'pose': [('pose/pose_summary.csv', 'semiuhpe', 'geodesic_error_mean'), ('pose/pose_summary.csv', '6drepnet360', 'geodesic_error_mean')],
    'expression': [('expression/expression_summary.csv', 'qcs', 'change_rate'), ('expression/expression_summary.csv', 'fmae', 'change_rate')],
    'age': [('age_gender/age_gender_summary.csv', 'mivolo', 'age_abs_diff_mean'), ('age_gender/age_gender_summary.csv', 'hycnxtvit', 'age_abs_diff_mean')],
    'gender': [('age_gender/age_gender_summary.csv', 'mivolo', 'gender_flip_rate'), ('age_gender/age_gender_summary.csv', 'fairface', 'gender_flip_rate')],
}


def table_rows(tex, label):
    """Data rows (name, numbers) of the tabular following \\label{label}, with \\cellcolor and \\textbf removed."""
    block = tex[tex.index('\\label{%s}' % label):]
    block = block[:block.index('\\end{tabular}')]
    rows = []
    for line in block.splitlines():
        line = re.sub(r'\\cellcolor\{[^}]*\}|\\textbf\{([^}]*)\}', lambda m: m.group(1) or '', line)
        cells = [c.strip().rstrip('\\').strip() for c in line.split('&')]
        if len(cells) > 3 and cells[0] in METHODS:
            rows.append((cells[0], cells[1:]))
    return rows


def privacy_blocks(tex):
    blocks = {}
    for name, cells in table_rows(tex, 'tab:iv_multiverifier'):
        if cells[0] != 'AUC':
            continue
        vals = [float(c) for c in cells[1:17]]
        for j, (ds, ver) in enumerate([(d, v) for d in DATASETS for v in ['AdaFace', 'TransFace', 'SwinFace', 'ArcFace']]):
            blocks.setdefault(f'{ver} AUC | {ds}', {})[name] = vals[j]
    return blocks


def quality_blocks(tex):
    blocks = {}
    fd_cols = {'rafd': 0, 'xm2vts_aligned': 3, 'LFW_unified_orig': 5, 'CALFW': 7}  # FD, MSE, (gACC on RaFD)
    files = {f.lower(): f for f in os.listdir(TK)}
    for name, cells in table_rows(tex, 'tab:iqdu_all_datasets'):
        for ds, j in fd_cols.items():
            blocks.setdefault(f'FD | {ds}', {})[name] = float(cells[j])
            f = files.get(f'mse_{ds}_{METHODS[name]}.csv'.lower())
            mse = pd.read_csv(os.path.join(TK, f)).mse.mean()
            printed = float(cells[j + 1])
            assert abs(round(mse, 2) - printed) <= 0.011, (name, ds, mse, printed)
            blocks.setdefault(f'MSE | {ds}', {})[name] = mse
    return blocks


def utility_blocks():
    blocks = {}
    cache = {}
    for attr, sources in UTILITY.items():
        for ds in DATASETS:
            ranks = []
            for csv, model, col in sources:
                df = cache.setdefault(csv, pd.read_csv(os.path.join(TK, csv)))
                d = df[(df.model == model) & (df.dataset == ds)].set_index('technique')[col]
                ranks.append(rankdata([d[METHODS[m]] for m in METHODS]))
            mean_rank = np.mean(ranks, axis=0)
            blocks[f'{attr} | {ds}'] = dict(zip(METHODS, mean_rank))  # re-ranked in friedman()
    return blocks


def friedman(blocks, alpha=0.05):
    names = list(METHODS)
    R = np.array([rankdata([blocks[b][m] for m in names]) for b in blocks])  # N x k, 1 = best (lowest)
    N, k = R.shape
    stat, p = friedmanchisquare(*R.T)
    W = stat / (N * (k - 1))
    cd = norm.ppf(1 - alpha / (2 * (k - 1))) * np.sqrt(k * (k + 1) / (6 * N))
    return pd.Series(R.mean(0), index=names), dict(N=N, k=k, chi2=stat, p=p, W=W, CD=cd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('main_tex')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_friedman'))
    args = ap.parse_args()
    tex = open(args.main_tex).read()
    groups = {'privacy': privacy_blocks(tex), 'quality': quality_blocks(tex), 'utility': utility_blocks()}
    os.makedirs(args.out, exist_ok=True)
    table, stats = {}, {}
    for g, blocks in groups.items():
        assert all(len(b) == len(METHODS) for b in blocks.values()), g
        table[g], stats[g] = friedman(blocks)
        pd.DataFrame(blocks).T.to_csv(os.path.join(args.out, f'blocks_{g}.csv'))
    df = pd.DataFrame(table)
    df['overall'] = df[['privacy', 'quality', 'utility']].mean(1)
    df = df.sort_values('overall')
    df.round(2).to_csv(os.path.join(args.out, 'mean_ranks.csv'))
    pd.DataFrame(stats).T.to_csv(os.path.join(args.out, 'friedman_stats.csv'))
    pd.set_option('display.width', 200)
    print('Mean ranks (1 = best; 13 methods)')
    print(df.round(2).to_string())
    print('\nFriedman statistics per group')
    for g, s in stats.items():
        print(f'  {g:8s} N={s["N"]:2d} blocks  chi2({s["k"] - 1})={s["chi2"]:.2f}  p={s["p"]:.2e}  '
              f'Kendall W={s["W"]:.3f}  CD(Bonferroni-Dunn, a=0.05)={s["CD"]:.3f}')
    print('\ngroup ranks (position 1-13):')
    print(df[['privacy', 'quality', 'utility', 'overall']].rank().astype(int).to_string())


if __name__ == '__main__':
    main()
