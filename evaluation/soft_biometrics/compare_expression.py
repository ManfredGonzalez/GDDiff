"""Compare expression predictions of each de-identified baseline against the originals.

For every (model, dataset, technique) each de-identified image is matched to its
original by filename. Reported:
    change_rate   % of images whose predicted expression differs from the original's
                  (lower = expression better preserved; no ground-truth labels needed)
    js_div        mean Jensen-Shannon divergence (base 2, in [0, 1]) between the two
                  predicted class distributions

Outputs (in root_dir/results/expression/):
    expression_per_image_<model>.csv, expression_summary.csv, expression_table_<model>.csv

Usage (env: gaze or toolkit):
    python gaze_estimation/compare_expression.py
"""
import glob
import os

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.expanduser('~/Desktop/master_research/deid-toolkit/root_dir/results/expression')
MODELS = ['qcs', 'fmae']
DATASETS = ['CALFW', 'LFW_unified_orig', 'rafd', 'xm2vts_aligned']


def js_divergence(p, q, eps=1e-12):
    p, q = p + eps, q + eps
    m = (p + q) / 2
    kl = lambda a, b: (a * np.log2(a / b)).sum(1)
    return (kl(p, m) + kl(q, m)) / 2


def main():
    summary = []
    for model in MODELS:
        per_image = []
        for ds in DATASETS:
            orig_csv = os.path.join(RESULTS_DIR, model, f'{model}_{ds}_Original.csv')
            if not os.path.exists(orig_csv):
                print(f'[missing] {orig_csv}')
                continue
            orig = pd.read_csv(orig_csv)
            pcols = [c for c in orig.columns if c.startswith('p_')]
            prefix = os.path.join(RESULTS_DIR, model, f'{model}_{ds}_')
            for csv in sorted(glob.glob(prefix + '*.csv')):
                tech = csv[len(prefix):-len('.csv')]
                if tech == 'Original':
                    continue
                m = orig.merge(pd.read_csv(csv), on='img', suffixes=('_o', '_d'))
                changed = (m['pred_o'] != m['pred_d']).values
                js = js_divergence(m[[c + '_o' for c in pcols]].values, m[[c + '_d' for c in pcols]].values)
                per_image.append(pd.DataFrame({'model': model, 'dataset': ds, 'technique': tech, 'img': m['img'],
                                               'pred_original': m['pred_o'], 'pred_deid': m['pred_d'],
                                               'changed': changed, 'js_div': js}))
                summary.append({'model': model, 'dataset': ds, 'technique': tech,
                                'n_original': len(orig), 'n_matched': len(m),
                                'change_rate': changed.mean() * 100, 'js_div_mean': js.mean()})
        if per_image:
            pd.concat(per_image, ignore_index=True).to_csv(
                os.path.join(RESULTS_DIR, f'expression_per_image_{model}.csv'), index=False)

    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(RESULTS_DIR, 'expression_summary.csv'), index=False)
    pd.set_option('display.width', 200)
    for model in MODELS:
        s = summary[summary.model == model]
        if s.empty:
            continue
        table = s.pivot(index='technique', columns='dataset', values='change_rate')[DATASETS]
        table['Average'] = table.mean(1)
        table = table.sort_values('Average').round(1)
        table.to_csv(os.path.join(RESULTS_DIR, f'expression_table_{model}.csv'))
        print(f'\n=== {model}: expression change rate (% of images whose predicted expression changed) ===')
        print(table.to_string())


if __name__ == '__main__':
    main()
