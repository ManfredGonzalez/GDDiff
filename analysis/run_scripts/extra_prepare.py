"""Inputs for the extra revision experiments (timing of baselines, reconstruction attack, eyeglasses)."""
import os
import random
import shutil

import pandas as pd

W = os.path.expanduser('~/Desktop/master_research/DeiDDPG/exp/robustness_work')
S = os.path.expanduser('~/Desktop/master_research/DeiDDPG/exp/image_samples')
A = os.path.expanduser('~/Desktop/master_research/deid-toolkit/root_dir/datasets/aligned')
X = os.path.join(W, 'extra')


def fresh(d):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


# 1) timing: flat copies of the 20 benchmark images and of the first 2 (to subtract model loading)
bench = sorted(os.path.join(r, f) for r, _, fs in os.walk(os.path.join(W, 'bench20')) for f in fs if f.endswith('.jpg'))[:20]
for n, d in [(20, fresh(os.path.join(X, 'bench20_flat'))), (2, fresh(os.path.join(X, 'bench2_flat')))]:
    for p in bench[:n]:
        shutil.copy(p, d)
print('bench', len(bench))

# 2) reconstruction attack: de-identified outputs fed back as observations (symlinks, same names/sets)
src = {'atk_full': ('robust_base', '0.5_mean'), 'atk_lam0.1': ('sweep300_0.1', '0.1_mean'),
       'atk_gauss': ('robust_base', '0.5_gauss'), 'atk_lapl': ('robust_base', '0.5_Lapl')}
for v, (run, br) in src.items():
    for s in ['rafd', 'lfw']:
        d = fresh(os.path.join(X, 'attack_in', v, s))
        for f in os.listdir(os.path.join(S, run, s, br)):
            os.symlink(os.path.join(S, run, s, br, f), os.path.join(d, f))
for s in ['rafd', 'lfw']:  # control: plain restoration of the original inputs
    d = fresh(os.path.join(X, 'attack_in', 'atk_orig', s))
    for f in os.listdir(os.path.join(W, 'robust_inputs/clean', s)):
        os.symlink(os.path.join(W, 'robust_inputs/clean', s, f), os.path.join(d, f))

# 3) eyeglasses: 300 random GlassesGAN images per dataset (seed 0); originals without glasses share the name
rows = []
for s, g, o in [('rafd', 'rafd_glasses', 'rafd'), ('xm2vts', 'xm2vts_glasses_aligned', 'xm2vts_aligned')]:
    names = sorted(f for f in os.listdir(os.path.join(A, g)) if f.endswith('.jpg') and os.path.exists(os.path.join(A, o, f)))
    pick = sorted(random.Random(0).sample(names, 300))
    for kind, folder in [('glasses', g), ('plain', o)]:
        d = fresh(os.path.join(X, 'glasses_in', kind, s))
        for f in pick:
            shutil.copy(os.path.join(A, folder, f), d)
    rows += [dict(set=s, img=os.path.splitext(f)[0]) for f in pick]
pd.DataFrame(rows).to_csv(os.path.join(X, 'glasses_subset.csv'), index=False)
print('glasses', len(rows))
