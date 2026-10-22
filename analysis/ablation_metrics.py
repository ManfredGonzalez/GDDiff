"""Metrics for the average-face-branch ablation (reviewer R2.4) on the 300-image subset.

All variants were sampled with the paper's command at lambda = 0.5, the same seed and the
same image order, so they differ only in how the average-face branch A is formed:
    full      robust_base/0.5_mean        x0 and residual averaged over G and L, zero mask after t_mu
    from_G    abl300_from_G/0.5_mean      A spawned from the Gaussian branch only (no averaging)
    no_zmask  abl300_no_zmask/0.5_mean    averaged, but A keeps the Gaussian weight instead of Z
    gauss     robust_base/0.5_gauss       Gauss-GDDiff, i.e. no average-face branch
    lapl      robust_base/0.5_Lapl        Lpl-GDDiff, i.e. no average-face branch

  --part identity  (env: adaface, run from the deid-toolkit root)
      AUC and F1 at the EER threshold on the subset's evaluation pairs, AdaFace as in the toolkit.
  --part quality   (env: toolkit, run from the deid-toolkit root)
      MSE (toolkit mse.py), gAcc on the 150 RaFD images (toolkit restnet18_GD.py, filename labels),
      FD with dgm-eval / DINOv2 on the 300 images (biased at n=300: compare variants only).
  --part summary   prints the ablation table.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLKIT = os.path.expanduser('~/Desktop/master_research/deid-toolkit')
SAMPLES = os.path.join(REPO, 'exp/image_samples')
W = os.path.join(REPO, 'exp/robustness_work')
OUT = os.path.join(REPO, 'analysis/results_ablation')
SETS = {'rafd': 'rafd', 'lfw': 'LFW_unified_orig'}
VARIANTS = {'full': ('robust_base', 'mean'), 'from_G': ('abl300_from_G', 'mean'),
            'no_zmask': ('abl300_no_zmask', 'mean'), 'gauss': ('robust_base', 'gauss'),
            'lapl': ('robust_base', 'Lapl')}
DGM_EVAL = os.environ.get('DGM_EVAL_SRC', '')


def out_dir(variant, sub):
    run, branch = VARIANTS[variant]
    return os.path.join(SAMPLES, run, sub, f'0.5_{branch}')


def reference_dirs():
    """Folders of symlinks to the ORIGINAL aligned images of the subset (not the re-saved inputs)."""
    subset = pd.read_csv(os.path.join(W, 'robust_inputs/subset.csv'))
    dirs = {}
    for sub, ds in SETS.items():
        d = os.path.join(W, 'ref300', sub)
        os.makedirs(d, exist_ok=True)
        for n in subset[subset.set == sub].img:
            dst = os.path.join(d, n + '.jpg')
            if not os.path.exists(dst):
                os.symlink(os.path.join(TOOLKIT, 'root_dir/datasets/aligned', ds, n + '.jpg'), dst)
        dirs[sub] = d
    return dirs


def eer_f1(y, s):
    from sklearn.metrics import f1_score, roc_curve
    fpr, tpr, thr = roc_curve(y, s)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return f1_score(y, (s >= thr[i]).astype(int)), (fpr[i] + 1 - tpr[i]) / 2


def part_identity(args):
    from sklearn.metrics import roc_auc_score
    import torch
    from PIL import Image
    sys.path.insert(0, os.path.join(TOOLKIT, 'root_dir/evaluation/identity_verification/AdaFace'))
    os.chdir(TOOLKIT)
    from inference import load_pretrained_model, to_input
    model = load_pretrained_model('ir_50').cuda().eval()
    cache = {}

    def emb(path):
        if path not in cache:
            img = Image.open(path).convert('RGB').resize((112, 112), Image.Resampling.LANCZOS)
            with torch.no_grad():
                f, _ = model(to_input(img).cuda())
            cache[path] = f[0].cpu().numpy()
        return cache[path]

    subset = pd.read_csv(os.path.join(W, 'robust_inputs/subset.csv'))
    rows = []
    for variant in VARIANTS:
        ys, ss = [], []
        for sub, ds in SETS.items():
            names = set(subset[subset.set == sub].img)
            pairs = pd.read_csv(os.path.join(TOOLKIT, 'root_dir/results', f'adaface_optimized_{ds}_0.5_mean_GauDeiDiff.csv'))
            pairs['b'] = pairs.img_b.map(lambda f: os.path.splitext(f)[0])
            pairs = pairs[pairs.b.isin(names)]
            for a, b, g in zip(pairs.img, pairs.b, pairs.ground_truth):
                p = os.path.join(out_dir(variant, sub), b + '.jpg')
                if os.path.exists(p):
                    ss.append(float(emb(os.path.join(TOOLKIT, 'root_dir/datasets/aligned', ds, a)) @ emb(p)))
                    ys.append(int(g))
        y, s = np.array(ys), np.array(ss)
        f1, eer = eer_f1(y, s)
        rows.append(dict(variant=variant, n_pairs=len(y), auc=roc_auc_score(y, s), f1_eer=f1, eer=eer))
        print(rows[-1], flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'ablation_identity.csv'), index=False)


def part_quality(args):
    os.chdir(TOOLKIT)
    os.makedirs(OUT, exist_ok=True)
    refs = reference_dirs()
    py = sys.executable
    rows = []
    for variant in VARIANTS:
        row = dict(variant=variant)
        mse = []
        for sub in SETS:
            f = os.path.join(OUT, f'mse_{variant}_{sub}.csv')
            subprocess.run([py, 'root_dir/evaluations/mse.py', refs[sub], out_dir(variant, sub), '--save_path', f,
                            '--dataset_name', sub, '--technique_name', variant], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            mse.append(pd.read_csv(f).mse.values)
        row['mse'] = float(np.concatenate(mse).mean())
        f = os.path.join(OUT, f'gacc_{variant}_rafd.csv')
        subprocess.run([py, 'root_dir/evaluations/restnet18_GD.py', refs['rafd'], out_dir(variant, 'rafd'),
                        '--save_path', f, '--dataset_name', 'rafd', '--technique_name', variant], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        row['gacc_rafd'] = float(pd.read_csv(f).isMatch.mean())
        if DGM_EVAL:  # FD on the 300 images, real = original aligned images of the same subset
            merged_ref, merged_out = os.path.join(W, 'fd_ref300'), os.path.join(W, f'fd_{variant}')
            for d, src in [(merged_ref, refs), (merged_out, {s: out_dir(variant, s) for s in SETS})]:
                os.makedirs(d, exist_ok=True)
                for s, folder in src.items():
                    for fn in os.listdir(folder):
                        dst = os.path.join(d, f'{s}_{fn}')
                        if not os.path.exists(dst):
                            os.symlink(os.path.join(folder, fn), dst)
            env = dict(os.environ, PYTHONPATH=DGM_EVAL)
            res = subprocess.run([os.path.expanduser('~/anaconda3/envs/dgmeval/bin/python'), '-m', 'dgm_eval',
                                  merged_ref, merged_out, '--model', 'dinov2', '--metrics', 'fd'],
                                 capture_output=True, text=True, env=env, cwd=W)
            fd = [l for l in res.stdout.splitlines() if l.strip().startswith('fd:')]
            row['fd_300'] = float(fd[-1].split(':')[1]) if fd else np.nan
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'ablation_quality.csv'), index=False)


def part_summary(args):
    idn = pd.read_csv(os.path.join(OUT, 'ablation_identity.csv'))
    q = pd.read_csv(os.path.join(OUT, 'ablation_quality.csv'))
    df = idn.merge(q, on='variant')
    df.to_csv(os.path.join(OUT, 'ablation_summary.csv'), index=False)
    pd.set_option('display.width', 200)
    print(df.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['identity', 'quality', 'summary'], required=True)
    args = ap.parse_args()
    {'identity': part_identity, 'quality': part_quality, 'summary': part_summary}[args.part](args)


if __name__ == '__main__':
    main()
