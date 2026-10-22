"""Cross-branch dependence of the Gaussian / LoG residuals at the fusion step t_mu.

Reviewer question: the variance-reduction reading of the averaged residual
eps_A = (eps_G + eps_L) / 2 depends on how correlated the two branches are.
This script measures it empirically from dumps written by the sampler when
GDDIFF_FUSION_LOG is set (functions/ddpg_scheme.py), one file per
(image, seed, lambda):  <image>_seed<s>_per<lambda>.npz  with
    et_hat  (2, 3, H, W)  residuals of the G (index 0) and L (index 1) branches at t_mu
    x0_t    (2, 3, H, W)  their x_{0|t} estimates
    face_bbox (x, y, w, h)

Randomness = the sampler's own stochasticity (independent initial latents and
injected noise per branch), varied through --seed. For every image and pixel,
across seeds:
    var_G, var_L, var_A, cov(G, L), rho = cov / sqrt(var_G var_L)
Pooled per region (inside / outside the face box):
    rho_pooled = sum cov / sqrt(sum var_G * sum var_L)
    Var(A)/Var(G), Var(A)/Var(L), Var(A)/max(Var(G), Var(L))
    separation = mean (mu_G - mu_L)^2 / mean((var_G + var_L) / 2)
        (> 0 means the branches target different expected residuals)

Usage:
    python analysis/branch_correlation.py <fusion_dir> [--out results.csv]
    where <fusion_dir> contains per<lambda>/ subfolders of .npz dumps.
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

NAME_RE = re.compile(r'(?P<img>.+)_seed(?P<seed>\d+)_per(?P<per>[0-9.]+)\.npz$')


def region_masks(bbox, shape):
    h, w = shape
    inside = np.zeros((h, w), bool)
    x, y, bw, bh = [int(v) for v in bbox]
    if bw > 0:
        inside[max(y, 0):y + bh, max(x, 0):x + bw] = True
    return {'inside_face': inside, 'outside_face': ~inside}


def image_stats(files, key):
    """Across-seed statistics for one image; returns {region: dict}."""
    data = [np.load(f) for f in files]
    e = np.stack([d[key].astype(np.float64) for d in data])  # (S, 2, C, H, W)
    g, l = e[:, 0], e[:, 1]
    a = (g + l) / 2
    S = e.shape[0]
    mu_g, mu_l = g.mean(0), l.mean(0)
    var_g, var_l, var_a = g.var(0, ddof=1), l.var(0, ddof=1), a.var(0, ddof=1)
    cov = ((g - mu_g) * (l - mu_l)).sum(0) / (S - 1)
    bbox = data[0]['face_bbox']
    out = {}
    for region, m in region_masks(bbox, var_g.shape[-2:]).items():
        if not m.any():
            continue
        sel = lambda x: x[:, m]  # (C, n_pixels)
        vg, vl, va, c = (sel(x).sum() for x in (var_g, var_l, var_a, cov))
        sep = (sel((mu_g - mu_l) ** 2)).mean() / max(sel((var_g + var_l) / 2).mean(), 1e-12)
        out[region] = dict(rho=c / np.sqrt(vg * vl), var_A_over_G=va / vg, var_A_over_L=va / vl,
                           var_A_over_max=va / max(vg, vl), var_G=vg / m.sum(), var_L=vl / m.sum(),
                           separation=sep, n_seeds=S)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fusion_dir')
    ap.add_argument('--out', default=None)
    ap.add_argument('--key', default='et_hat', choices=['et_hat', 'x0_t'])
    args = ap.parse_args()

    groups = defaultdict(list)
    for f in glob.glob(os.path.join(args.fusion_dir, 'per*', '*.npz')):
        m = NAME_RE.search(os.path.basename(f))
        groups[(m['per'], m['img'])].append(f)

    rows = []
    for (per, img), files in sorted(groups.items()):
        if len(files) < 3:
            continue
        dataset = 'RaFD' if img.startswith('Rafd') else 'LFW'
        for region, st in image_stats(sorted(files), args.key).items():
            rows.append(dict(lam=float(per), image=img, dataset=dataset, region=region, **st))
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)

    cols = ['rho', 'var_A_over_G', 'var_A_over_L', 'var_A_over_max', 'separation']
    pd.set_option('display.width', 220)
    print(f'quantity: {args.key}; images per lambda: {df.groupby("lam").image.nunique().to_dict()}; '
          f'seeds per image: {int(df.n_seeds.median())}')
    print('\nmedian [IQR] over images, by lambda and region')
    for (lam, region), d in df.groupby(['lam', 'region']):
        cells = '  '.join(f'{c}={d[c].median():.3f} [{d[c].quantile(.25):.3f}-{d[c].quantile(.75):.3f}]'
                          for c in cols)
        print(f'lambda={lam:<4} {region:13} {cells}')
    print('\nmedian by dataset and region (all lambdas)')
    print(df.groupby(['dataset', 'region'])[cols].median().round(3).to_string())
    print(f'\nshare of (image, lambda, region) cases with Var(A) <= max(Var(G), Var(L)): '
          f'{(df.var_A_over_max <= 1 + 1e-9).mean() * 100:.1f}%')


if __name__ == '__main__':
    main()
