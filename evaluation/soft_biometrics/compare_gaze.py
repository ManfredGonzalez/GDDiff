"""Compare gaze predictions of each de-identified baseline against the originals.

For every (model, dataset, technique) the de-identified image is matched to its
original by filename and the angular error (deg) between the two predicted 3D
gaze vectors is computed, plus the absolute pitch / yaw differences.

Outputs (in root_dir/results/gaze/):
    gaze_per_image_<model>.csv   one row per matched image
    gaze_summary.csv             one row per (model, dataset, technique)
    gaze_table_<model>.csv       technique x dataset table of mean angular error

Usage (env: gaze or toolkit):
    python gaze_estimation/compare_gaze.py
"""
import glob
import os

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.expanduser('~/Desktop/master_research/deid-toolkit/root_dir/results/gaze')
MODELS = ['gazetr', 'gaze3d']
DATASETS = ['CALFW', 'LFW_unified_orig', 'rafd', 'xm2vts_aligned']


def angular_error(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(1), -1.0, 1.0)))


def main():
    vec = ['gaze_x', 'gaze_y', 'gaze_z']
    summary, per_image_all = [], []
    for model in MODELS:
        per_image = []
        for ds in DATASETS:
            orig_csv = os.path.join(RESULTS_DIR, model, f'{model}_{ds}_Original.csv')
            if not os.path.exists(orig_csv):
                print(f'[missing] {orig_csv}')
                continue
            orig = pd.read_csv(orig_csv)
            prefix = os.path.join(RESULTS_DIR, model, f'{model}_{ds}_')
            for csv in sorted(glob.glob(prefix + '*.csv')):
                tech = csv[len(prefix):-len('.csv')]
                if tech == 'Original' or '_limit' in tech:
                    continue
                deid = pd.read_csv(csv)
                m = orig.merge(deid, on='img', suffixes=('_o', '_d'))
                err = angular_error(m[[c + '_o' for c in vec]].values, m[[c + '_d' for c in vec]].values)
                dpitch = (m['pitch_deg_o'] - m['pitch_deg_d']).abs()
                dyaw = (m['yaw_deg_o'] - m['yaw_deg_d']).abs()
                per_image.append(pd.DataFrame({'model': model, 'dataset': ds, 'technique': tech,
                                               'img': m['img'], 'angular_error_deg': err,
                                               'abs_pitch_diff_deg': dpitch, 'abs_yaw_diff_deg': dyaw}))
                summary.append({'model': model, 'dataset': ds, 'technique': tech,
                                'n_original': len(orig), 'n_matched': len(m),
                                'angular_error_mean': err.mean(), 'angular_error_std': err.std(),
                                'angular_error_median': np.median(err),
                                'pitch_mae': dpitch.mean(), 'yaw_mae': dyaw.mean()})
        if per_image:
            per_image = pd.concat(per_image, ignore_index=True)
            per_image.to_csv(os.path.join(RESULTS_DIR, f'gaze_per_image_{model}.csv'), index=False)
            per_image_all.append(per_image)

    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(RESULTS_DIR, 'gaze_summary.csv'), index=False)

    pd.set_option('display.width', 200)
    for model in MODELS:
        s = summary[summary.model == model]
        if s.empty:
            continue
        s = s.assign(cell=s.angular_error_mean.map('{:.2f}'.format) + ' ± ' + s.angular_error_std.map('{:.2f}'.format))
        table = s.pivot(index='technique', columns='dataset', values='cell')[[d for d in DATASETS if d in s.dataset.values]]
        means = s.pivot(index='technique', columns='dataset', values='angular_error_mean')
        table['Average'] = means.mean(1).map('{:.2f}'.format)
        table = table.loc[means.mean(1).sort_values().index]
        table.to_csv(os.path.join(RESULTS_DIR, f'gaze_table_{model}.csv'))
        print(f'\n=== {model}: angular error (deg) vs original, mean ± std (lower = gaze better preserved) ===')
        print(table.to_string())
        incomplete = s[s.n_matched < s.n_original]
        if len(incomplete):
            print('  images without a de-identified counterpart (excluded):')
            for _, r in incomplete.iterrows():
                print(f'    {r.dataset:17} {r.technique:22} {r.n_original - r.n_matched}')


if __name__ == '__main__':
    main()
