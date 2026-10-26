"""Method overview figure: weighting field and clean-image estimate of each GDDiff branch along sampling.

Uses the per-step x0 estimates stored with GDDIFF_TRAJ_LOG (see intro_trajectories.py; same face) and the
sampler's own kernel functions with the detected face box and the linear variance schedule. The Gaussian and
LoG rows come from the lambda=0.1 run (identical to the other runs before its fusion step, and the source of the
Gaussian/LoG curves of the introduction figure); the average-face row from the lambda=0.5 run.

Usage (env: adaface, run from the deid-toolkit root):
    python <DeiDDPG>/analysis/method_overview.py --traj <dir> --orig <image> --outputs <image_samples>
        --bbox x y w h --out <pdf>
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'functions'))
sys.path.insert(0, 'root_dir/evaluation/identity_verification/AdaFace')
import ddpg_scheme  # noqa: E402
from inference import load_pretrained_model, to_input  # noqa: E402

COLUMNS = [800, 680, 580, 400, 200, 0]  # timesteps shown (580 = t_mu for lambda 0.5)
LAM = '0.5'
COLORS = {'G': '#f4a259', 'L': '#e15759', 'A': '#b07aa1'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj', required=True)
    ap.add_argument('--orig', required=True)
    ap.add_argument('--outputs', required=True)
    ap.add_argument('--bbox', type=int, nargs=4, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    name = os.path.splitext(os.path.basename(args.orig))[0]
    load = lambda lam: torch.load(glob.glob(os.path.join(args.traj, f'{name}_seed*_per{lam}.pt'))[0])
    tr_gl, tr_a = load('0.1'), load(LAM)
    to_pil = lambda x: Image.fromarray(((x.float().clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).numpy())
    model = load_pretrained_model('ir_50').cuda().eval()

    def emb(img):
        with torch.no_grad():
            f, _ = model(to_input(img.convert('RGB').resize((112, 112), Image.Resampling.LANCZOS)).cuda())
        v = f[0].cpu().numpy()
        return v / np.linalg.norm(v)

    e_orig = emb(Image.open(args.orig))
    x, y, w, h = args.bbox
    variances = np.linspace(0.01 * w * h, w * h, len(tr_gl))  # as in ddpg_diffusion
    k_first = next(k for k, (_, xs) in enumerate(tr_a) if xs.shape[0] == 3)
    t_mu = tr_a[k_first - 1][0]  # fusion step: G/L estimates averaged here, branch evolves from the next step

    def step_of(traj, t):
        return int(np.argmin([abs(s - t) for s, _ in traj]))

    def field(kind, t):
        k = step_of(tr_gl, t)
        if kind == 'G':
            return ddpg_scheme.create_gaussian_kernel((256, 256), args.bbox, variances[k])
        if kind == 'L':
            return ddpg_scheme.create_laplacian_kernel((256, 256), args.bbox, variances[k])
        return ddpg_scheme.create_zero_kernel((256, 256), args.bbox)

    def estimate(kind, t):
        if kind == 'A':
            if t > t_mu:
                return None
            xs = tr_a[step_of(tr_a, t)][1]
            return to_pil(xs[2] if xs.shape[0] == 3 else (xs[0].float() + xs[1].float()) / 2)
        return to_pil(tr_gl[step_of(tr_gl, t)][1][0 if kind == 'G' else 1])

    finals = {}
    for kind, (lam, br) in {'G': ('0.1', 'gauss'), 'L': ('0.1', 'Lapl'), 'A': (LAM, 'mean')}.items():
        f = glob.glob(os.path.join(args.outputs, f'intro_traj_{lam}', '*', f'{lam}_{br}', name + '.*'))[0]
        finals[kind] = Image.open(f).convert('RGB')

    plt.rcParams['font.size'] = 13
    n_c = len(COLUMNS)
    fig = plt.figure(figsize=(1.25 * (n_c + 2) + 1.5, 1.25 * 6 + 0.7))
    gs = fig.add_gridspec(6, n_c + 3, width_ratios=[1.15, 1.25] + [1] * n_c + [0.08], wspace=0.05, hspace=0.12,
                          left=0.02, right=0.94, top=0.93, bottom=0.03)
    rows = [('G', 'field', r'Gaussian field $G_t$'), ('G', 'img', r'$\hat{x}_0$, Gaussian branch'),
            ('L', 'field', r'LoG field $L_t$'), ('L', 'img', r'$\hat{x}_0$, LoG branch'),
            ('A', 'field', r'zero mask $Z$'), ('A', 'img', r'$\hat{x}_0$, average-face branch')]
    im = None
    for r, (kind, what, label) in enumerate(rows):
        lab = fig.add_subplot(gs[r, 1])
        lab.axis('off')
        lab.text(0.95, 0.5, label, ha='right', va='center', fontsize=12, transform=lab.transAxes)
        for c, t in enumerate(COLUMNS):
            ax = fig.add_subplot(gs[r, c + 2])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f'$t={t}$' + (r' $(t_\mu)$' if t == t_mu else ''), fontsize=13)
            if kind == 'A' and t > t_mu:  # branch does not exist yet
                ax.set_facecolor('#f2f2f2')
                if what == 'img':
                    ax.text(0.5, 0.5, 'not yet\nspawned', ha='center', va='center', fontsize=11, color='gray',
                            transform=ax.transAxes)
                for sp in ax.spines.values():
                    sp.set_color('#cccccc')
                continue
            if what == 'field':
                im = ax.imshow(field(kind, t), cmap='magma', vmin=0, vmax=1.15)
                ax.add_patch(Rectangle((x, y), w, h, fill=False, ec='white', lw=0.8, ls='--'))
            else:
                img = finals[kind] if t == 0 else estimate(kind, t)
                ax.imshow(img.resize((256, 256)))
                for sp in ax.spines.values():
                    sp.set_edgecolor(COLORS[kind])
                    sp.set_linewidth(2)
                if kind == 'A' and t == t_mu:
                    ax.text(0.03, 0.04, 'mean of $G$, $L$', transform=ax.transAxes, fontsize=11, color='white',
                            bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0))
                if t == 0:
                    ax.text(0.03, 0.04, f'cos {float(emb(img) @ e_orig):.2f}', transform=ax.transAxes, fontsize=11,
                            color='white', bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0))
    # input column
    ax = fig.add_subplot(gs[2:4, 0])
    ax.imshow(Image.open(args.orig).convert('RGB').resize((256, 256)))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec='lime', lw=1.2))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('input $y$ and\ndetected face box', fontsize=12)
    ax.text(0.5, -0.08, 'original: cos 1.00', transform=ax.transAxes, ha='center', va='top', fontsize=11)
    cax = fig.add_subplot(gs[0:1, n_c + 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=10)
    cb.set_label('guidance weight', fontsize=10)
    fig.text(0.6, 0.975, r'reverse diffusion $\longrightarrow$ (noise $t=1000$ to output $t=0$)',
             ha='center', fontsize=13)
    fig.savefig(args.out)
    fig.savefig(args.out.replace('.pdf', '.png'), dpi=150)
    print('t_mu', t_mu, 'saved', args.out)


if __name__ == '__main__':
    main()
