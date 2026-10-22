"""Compare head pose predictions of each de-identified baseline against the originals.

For every (model, dataset, technique) each de-identified image is matched to its
original by filename. The error is the geodesic distance between the two
predicted rotations, arccos((tr(R_o^T R_d) - 1) / 2) in degrees, i.e. the
smallest rotation turning one head pose into the other. Absolute yaw / pitch /
roll differences are reported too.

Outputs (in root_dir/results/pose/):
    pose_per_image_<model>.csv, pose_summary.csv, pose_table_<model>.csv

Usage (env: gaze or toolkit):
    python gaze_estimation/compare_pose.py
"""
import glob
import os

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.expanduser('~/Desktop/master_research/deid-toolkit/root_dir/results/pose')
MODELS = ['semiuhpe', '6drepnet360']
DATASETS = ['CALFW', 'LFW_unified_orig', 'rafd', 'xm2vts_aligned']
R_COLS = [f'r{i}{j}' for i in range(3) for j in range(3)]


def geodesic_deg(Ra, Rb):
    cos = (np.einsum('nij,nij->n', Ra, Rb) - 1.0) / 2.0  # tr(Ra^T Rb) = sum_ij Ra_ij Rb_ij
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def angle_diff(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


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
            prefix = os.path.join(RESULTS_DIR, model, f'{model}_{ds}_')
            for csv in sorted(glob.glob(prefix + '*.csv')):
                tech = csv[len(prefix):-len('.csv')]
                if tech == 'Original':
                    continue
                m = orig.merge(pd.read_csv(csv), on='img', suffixes=('_o', '_d'))
                Ro = m[[c + '_o' for c in R_COLS]].values.reshape(-1, 3, 3)
                Rd = m[[c + '_d' for c in R_COLS]].values.reshape(-1, 3, 3)
                err = geodesic_deg(Ro, Rd)
                d = {k: angle_diff(m[f'{k}_deg_o'], m[f'{k}_deg_d']) for k in ['yaw', 'pitch', 'roll']}
                per_image.append(pd.DataFrame({'model': model, 'dataset': ds, 'technique': tech, 'img': m['img'],
                                               'geodesic_error_deg': err, **{f'abs_{k}_diff_deg': v for k, v in d.items()}}))
                summary.append({'model': model, 'dataset': ds, 'technique': tech,
                                'n_original': len(orig), 'n_matched': len(m),
                                'geodesic_error_mean': err.mean(), 'geodesic_error_std': err.std(),
                                'geodesic_error_median': np.median(err),
                                **{f'{k}_mae': v.mean() for k, v in d.items()}})
        if per_image:
            pd.concat(per_image, ignore_index=True).to_csv(
                os.path.join(RESULTS_DIR, f'pose_per_image_{model}.csv'), index=False)

    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(RESULTS_DIR, 'pose_summary.csv'), index=False)
    pd.set_option('display.width', 200)
    for model in MODELS:
        s = summary[summary.model == model]
        if s.empty:
            continue
        cell = s.geodesic_error_mean.map('{:.2f}'.format) + ' ± ' + s.geodesic_error_std.map('{:.2f}'.format)
        table = s.assign(cell=cell).pivot(index='technique', columns='dataset', values='cell')[DATASETS]
        means = s.pivot(index='technique', columns='dataset', values='geodesic_error_mean')
        table['Average'] = means.mean(1).map('{:.2f}'.format)
        table = table.loc[means.mean(1).sort_values().index]
        table.to_csv(os.path.join(RESULTS_DIR, f'pose_table_{model}.csv'))
        print(f'\n=== {model}: head pose geodesic error (deg) vs original, mean ± std (lower = pose better preserved) ===')
        print(table.to_string())


if __name__ == '__main__':
    main()
