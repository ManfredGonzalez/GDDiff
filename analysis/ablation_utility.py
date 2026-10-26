"""Soft-biometric utility of the average-face-branch ablation variants (reviewer R2.4).

Runs the estimators of the soft-biometric utility figures (deid-toolkit/gaze_estimation,
copied in evaluation/soft_biometrics/) with their own loaders, preprocessing and centre
crops on the 300 original subset images and on the outputs of each ablation variant
(lambda = 0.5, identical settings, seed and image order, so every image is paired):
    full      robust_base/0.5_mean      x0 and residual averaged, zero mask after t_mu
    from_G    abl300_from_G/0.5_mean    average-face branch spawned from the Gaussian branch only
    no_zmask  abl300_no_zmask/0.5_mean  averaged, Gaussian weight instead of the zero mask
    gauss     robust_base/0.5_gauss     Gauss-GDDiff (no average-face branch)
    lapl      robust_base/0.5_Lapl      Lpl-GDDiff (no average-face branch)

Deviation from the prediction on the original image, as in the utility figures:
gaze angular error (deg), head-pose geodesic error (deg), expression class changed (0/1),
absolute age difference (years), gender flipped (0/1).

  --part predict --model <name>   per-image predictions -> results_ablation/utility/pred_<name>.csv
        (env: gaze for every model except mivolo, which needs env mivolo)
  --part compare                  deviations, per-variant means and paired tests of 'full'
        against every other variant (Wilcoxon signed-rank; exact McNemar for 0/1 measures)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SB = os.path.expanduser('~/Desktop/master_research/deid-toolkit/gaze_estimation')
REF = os.path.join(REPO, 'exp/robustness_work/ref300')  # symlinks to the original aligned images
OUT = os.path.join(REPO, 'analysis/results_ablation/utility')
SETS = ['rafd', 'lfw']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_metrics import PAIRS, VARIANTS, out_dir  # noqa: E402
TASK = {'gazetr': 'gaze', 'gaze3d': 'gaze', 'semiuhpe': 'pose', '6drepnet360': 'pose', 'qcs': 'expression',
        'fmae': 'expression', 'mivolo': 'age_gender', 'hycnxtvit': 'age', 'fairface': 'gender'}


def folders():
    yield 'original', {s: os.path.join(REF, s) for s in SETS}
    for v in VARIANTS:
        if all(os.path.isdir(out_dir(v, s)) for s in SETS):
            yield v, {s: out_dir(v, s) for s in SETS}


def batched(net, predict, dataset, keys):
    import torch
    from torch.utils.data import DataLoader
    out = {k: [] for k in keys}
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=64, num_workers=6, shuffle=False):
            res = predict(net, batch.to('cuda'))
            for k, v in zip(keys, res if isinstance(res, tuple) else (res,)):
                out[k].append(np.asarray(v.cpu() if hasattr(v, 'cpu') else v))
    return {k: np.concatenate(v) for k, v in out.items()}


def part_predict(model):
    sys.path.insert(0, SB)
    from run_gaze import list_images
    task = TASK[model]
    if task == 'gaze':
        import run_gaze as m
        load, data_cls, predict = m.MODELS[model]
        make_ds = lambda folder, files: data_cls(folder, files)
        run = lambda net, ds: {'g': batched(net, predict, ds, ['g'])['g']}
    elif task == 'pose':
        import run_pose as m
        load, tf, predict = m.MODELS[model]
        make_ds = lambda folder, files: m.FolderData(folder, files, tf)
        run = lambda net, ds: {'R': batched(net, predict, ds, ['R', 'y', 'p', 'r'])['R'].reshape(-1, 9)}
    elif task == 'expression':
        import torch
        import run_expression as m
        load, make_tf, predict, classes = m.MODELS[model]
        tf = make_tf(m.CENTER_CROP[model])
        make_ds = lambda folder, files: m.FolderData(folder, files, tf)
        softmax_predict = lambda net, b: torch.softmax(predict(net, b), 1)
        run = lambda net, ds: {'prob': batched(net, softmax_predict, ds, ['prob'])['prob']}
    else:
        import run_age_gender as m
        load, make_tf, predict = m.MODELS[model]
        tf = make_tf(m.CENTER_CROP[model])
        make_ds = None
    net = load('cuda')
    rows = []
    for variant, by_set in folders():
        for s, folder in by_set.items():
            files = list_images(folder)
            names = [os.path.splitext(f)[0] for f in files]
            if make_ds is None:
                df = m.run_folder(net, predict, tf, folder, files)
                cols = {c: df[c].values for c in ('age', 'p_male') if c in df}
            else:
                res = run(net, make_ds(folder, files))
                cols = {}
                for k, arr in res.items():
                    arr = arr.reshape(len(files), -1)
                    for j in range(arr.shape[1]):
                        cols[f'{k}{j}'] = arr[:, j]
            rows.append(pd.DataFrame({'variant': variant, 'set': s, 'img': names, **cols}))
            print(model, variant, s, len(files), flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(OUT, f'pred_{model}.csv'), index=False)


def deviations(model, pred):
    ref = pred[pred.variant == 'original'].drop(columns='variant')
    task = TASK[model]
    out = []
    for v in pred.variant.unique():
        if v == 'original':
            continue
        d = pred[pred.variant == v].drop(columns='variant').merge(ref, on=['set', 'img'], suffixes=('', '_o'))
        res = {}
        if task == 'gaze':
            a = d[['g0', 'g1', 'g2']].values
            b = d[['g0_o', 'g1_o', 'g2_o']].values
            cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
            res['gaze_error_deg'] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        elif task == 'pose':
            Ra = d[[f'R{i}' for i in range(9)]].values.reshape(-1, 3, 3)
            Rb = d[[f'R{i}_o' for i in range(9)]].values.reshape(-1, 3, 3)
            res['pose_error_deg'] = np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n', Ra, Rb) - 1) / 2, -1, 1)))
        elif task == 'expression':
            pc = [c for c in d if c.startswith('prob') and not c.endswith('_o')]
            res['expression_changed'] = (d[pc].values.argmax(1) != d[[c + '_o' for c in pc]].values.argmax(1)).astype(float)
        if 'age' in d:
            res['age_diff_years'] = (d['age'] - d['age_o']).abs().values
        if 'p_male' in d:
            res['gender_flipped'] = ((d['p_male'] >= 0.5) != (d['p_male_o'] >= 0.5)).astype(float).values
        for metric, vals in res.items():
            out.append(pd.DataFrame({'model': model, 'metric': metric, 'variant': v, 'set': d['set'].values,
                                     'img': d['img'].values, 'deviation': vals}))
    return pd.concat(out, ignore_index=True)


def paired_test(x, y, binary):
    from scipy.stats import binomtest, wilcoxon
    if binary:  # exact McNemar on discordant pairs
        b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
        return binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    diff = x - y
    return wilcoxon(x, y).pvalue if np.any(diff != 0) else 1.0


def part_compare():
    dev = pd.concat([deviations(m, pd.read_csv(os.path.join(OUT, f'pred_{m}.csv')))
                     for m in TASK if os.path.exists(os.path.join(OUT, f'pred_{m}.csv'))], ignore_index=True)
    dev.to_csv(os.path.join(OUT, 'deviations_per_image.csv'), index=False)
    rows = []
    for (model, metric), g in dev.groupby(['model', 'metric']):
        piv = g.pivot_table(index=['set', 'img'], columns='variant', values='deviation')
        row = dict(model=model, metric=metric, n=len(piv))
        for v in VARIANTS:
            row[f'mean_{v}'] = piv[v].mean() if v in piv else np.nan
        for a, b in PAIRS:
            if a in piv and b in piv:
                ok = piv[[a, b]].dropna()
                row[f'p_{a}_vs_{b}'] = paired_test(ok[a].values, ok[b].values, metric.endswith(('changed', 'flipped')))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT, 'ablation_utility_summary.csv'), index=False)
    pd.set_option('display.width', 250)
    print('Mean deviation from the original prediction (lower = attribute better preserved); '
          'p: paired test of full vs. variant on the same images')
    print(summary.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['predict', 'compare'], required=True)
    ap.add_argument('--model', choices=list(TASK))
    args = ap.parse_args()
    part_predict(args.model) if args.part == 'predict' else part_compare()


if __name__ == '__main__':
    main()
