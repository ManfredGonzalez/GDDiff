"""Compare age / gender predictions of each de-identified baseline against the originals.

For every (model, dataset, technique) each de-identified image is matched to its
original by filename. Reported (lower = attribute better preserved):
    age_abs_diff_mean   mean |age_deid - age_original| in years   (MiVOLO, HyCnXtViT)
    gender_flip_rate    % of images whose predicted gender differs (MiVOLO, FairFace)

Outputs (in root_dir/results/age_gender/):
    age_gender_per_image_<model>.csv, age_gender_summary.csv,
    age_table_<model>.csv, gender_table_<model>.csv

Usage (env: gaze or toolkit):
    python gaze_estimation/compare_age_gender.py
"""
import glob
import os

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.expanduser('~/Desktop/master_research/deid-toolkit/root_dir/results/age_gender')
MODELS = ['mivolo', 'hycnxtvit', 'fairface']
DATASETS = ['CALFW', 'LFW_unified_orig', 'rafd', 'xm2vts_aligned']


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
                row = {'model': model, 'dataset': ds, 'technique': tech,
                       'n_original': len(orig), 'n_matched': len(m)}
                pi = pd.DataFrame({'model': model, 'dataset': ds, 'technique': tech, 'img': m['img']})
                if 'age_o' in m:
                    diff = (m['age_d'] - m['age_o']).values
                    pi['age_original'], pi['age_deid'] = m['age_o'], m['age_d']
                    row.update(age_abs_diff_mean=np.abs(diff).mean(), age_abs_diff_std=np.abs(diff).std(),
                               age_signed_diff_mean=diff.mean())
                if 'gender_o' in m:
                    flipped = (m['gender_o'] != m['gender_d']).values
                    pi['gender_original'], pi['gender_deid'] = m['gender_o'], m['gender_d']
                    row.update(gender_flip_rate=flipped.mean() * 100)
                per_image.append(pi)
                summary.append(row)
        if per_image:
            pd.concat(per_image, ignore_index=True).to_csv(
                os.path.join(RESULTS_DIR, f'age_gender_per_image_{model}.csv'), index=False)

    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(RESULTS_DIR, 'age_gender_summary.csv'), index=False)
    pd.set_option('display.width', 200)
    for col, task, unit in [('age_abs_diff_mean', 'age', 'mean |age difference| in years'),
                            ('gender_flip_rate', 'gender', '% of images whose predicted gender flipped')]:
        for model in MODELS:
            s = summary[(summary.model == model)]
            if col not in s or s[col].isna().all():
                continue
            table = s.pivot(index='technique', columns='dataset', values=col)[DATASETS]
            table['Average'] = table.mean(1)
            table = table.sort_values('Average').round(2)
            table.to_csv(os.path.join(RESULTS_DIR, f'{task}_table_{model}.csv'))
            print(f'\n=== {model}: {task} ({unit}) ===')
            print(table.to_string())


if __name__ == '__main__':
    main()
