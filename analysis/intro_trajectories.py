"""Measured identity-space trajectories of GDDiff for the introduction figure.

GDDiff is run on one RaFD face at lambda = 0.1, 0.3, 0.5 with GDDIFF_TRAJ_LOG set, which stores the clean-image
estimate x0 of every branch at every reverse step. Each estimate is embedded with AdaFace (IR-50, whole image
resized to 112 px, as in the identity evaluation) and the embeddings are projected to 2-D with PCA fitted on all
trajectories and the original. The cosine similarity to the original identity is reported for the final outputs.

Usage (env: adaface, run from the deid-toolkit root):
    python <DeiDDPG>/analysis/intro_trajectories.py --traj <dir> --orig <image> --outputs <image_samples> --out <pdf>
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
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, 'root_dir/evaluation/identity_verification/AdaFace')
from inference import load_pretrained_model, to_input  # noqa: E402

LAMBDAS = ['0.1', '0.3', '0.5']
COLORS = {'G': '#f4a259', 'L': '#e15759', '0.1': '#59a14f', '0.3': '#4e79a7', '0.5': '#b07aa1'}
BRANCH_DIR = {'G': 'gauss', 'L': 'Lapl'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj', required=True)
    ap.add_argument('--orig', required=True)
    ap.add_argument('--outputs', required=True, help='exp/image_samples (runs intro_traj_<lambda>)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    model = load_pretrained_model('ir_50').cuda().eval()

    def emb(img):
        with torch.no_grad():
            f, _ = model(to_input(img.convert('RGB').resize((112, 112), Image.Resampling.LANCZOS)).cuda())
        return f[0].cpu().numpy()

    def to_pil(x):
        a = ((x.float().clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).numpy()
        return Image.fromarray(a)

    name = os.path.splitext(os.path.basename(args.orig))[0]
    e_orig = emb(Image.open(args.orig))
    paths = {}  # key -> (timesteps, embeddings)
    for lam in LAMBDAS:
        traj = torch.load(glob.glob(os.path.join(args.traj, f'{name}_seed*_per{lam}.pt'))[0])
        t = np.array([s for s, _ in traj])
        if lam == LAMBDAS[0]:  # G and L branches (batch 0 and 1); identical across runs up to the latest t_mu
            for b, key in [(0, 'G'), (1, 'L')]:
                paths[key] = (t, np.stack([emb(to_pil(x[b])) for _, x in traj]))
        steps = [k for k, (_, x) in enumerate(traj) if x.shape[0] == 3]  # average-face branch exists after t_mu
        start = steps[0] - 1  # spawned from the fused estimate at t_mu: start the path at the branch midpoint
        fused = (emb(to_pil(traj[start][1][0])) + emb(to_pil(traj[start][1][1]))) / 2
        paths[lam] = (np.concatenate([[t[start]], t[steps]]),
                      np.vstack([fused, np.stack([emb(to_pil(traj[k][1][2])) for k in steps])]))

    # final outputs as saved by the sampler, and their identity similarity to the original
    finals = {}
    for key, (lam, br) in {'G': ('0.1', 'gauss'), 'L': ('0.1', 'Lapl'), **{l: (l, 'mean') for l in LAMBDAS}}.items():
        f = glob.glob(os.path.join(args.outputs, f'intro_traj_{lam}', '*', f'{lam}_{br}', name + '.*'))[0]
        img = Image.open(f).convert('RGB')
        e = emb(img)
        finals[key] = (img, float(e @ e_orig / np.linalg.norm(e) / np.linalg.norm(e_orig)))

    allv = np.vstack([e_orig[None]] + [p for _, p in paths.values()])
    allv = allv / np.linalg.norm(allv, axis=1, keepdims=True)
    mu = allv.mean(0)
    _, _, vt = np.linalg.svd(allv - mu, full_matrices=False)
    proj = lambda v: ((v / np.linalg.norm(v, axis=-1, keepdims=True)) - mu) @ vt[:2].T

    fig = plt.figure(figsize=(9.2, 6.2))
    ax = fig.add_axes([0.03, 0.05, 0.66, 0.9])
    labels = {'L': r'Laplacian branch ($L_t$)', 'G': r'Gaussian branch ($G_t$)',
              **{l: rf'Average branch ($\lambda$={l}, $t_\mu$={paths[l][0][0]})' for l in LAMBDAS}}
    for key in ['L', 'G'] + LAMBDAS:
        xy = proj(paths[key][1])
        ax.plot(xy[:, 0], xy[:, 1], '-', color=COLORS[key], lw=2.2, alpha=0.9, label=labels[key])
        ax.scatter(*xy[-1], marker='X' if key in 'GL' else 'D', s=90, color=COLORS[key], edgecolor='k', zorder=5)
        if key in LAMBDAS:
            ax.scatter(*xy[0], marker='o', s=60, color=COLORS[key], edgecolor='k', zorder=5)
    x0 = proj(paths['G'][1][:1])[0]
    ax.scatter(*x0, s=120, color='k', zorder=6, label=r'start ($x_T$)')
    o = proj(e_orig[None])[0]
    ax.scatter(*o, marker='*', s=320, color='k', zorder=6, label='original identity')
    ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(Image.open(args.orig).convert('RGB').resize((84, 84))), zoom=1),
                                 o, xybox=(0, -60), boxcoords='offset points', frameon=False))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('identity space (AdaFace embedding, PCA 1)', fontsize=11)
    ax.set_ylabel('PCA 2', fontsize=11)
    ax.legend(fontsize=9, loc='best')

    # right: final outputs ordered by measured identity similarity (top = most suppressed)
    order = sorted(finals, key=lambda k: finals[k][1])
    n = len(order)
    for i, key in enumerate(order):
        a = fig.add_axes([0.73, 0.95 - (i + 1) * 0.9 / n + 0.01, 0.16, 0.9 / n - 0.02])
        a.imshow(finals[key][0].resize((160, 160)))
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor(COLORS[key])
            sp.set_linewidth(3)
        tag = {'G': 'Gauss', 'L': 'Lpl'}.get(key, rf'Mean $\lambda$={key}')
        a.set_ylabel(f'{tag}\ncos {finals[key][1]:.2f}', fontsize=9, rotation=0, ha='right', va='center', labelpad=4)
    fig.text(0.81, 0.965, 'identity suppressed', ha='center', fontsize=10)
    fig.text(0.81, 0.02, 'utility preserved', ha='center', fontsize=10)
    fig.savefig(args.out)
    fig.savefig(args.out.replace('.pdf', '.png'), dpi=150)

    # exact view: identity similarity of each branch's x0 estimate to the original along the reverse process
    cos = lambda v: v @ e_orig / np.linalg.norm(v, axis=-1) / np.linalg.norm(e_orig)
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(10, 6.2))
    ax = fig.add_axes([0.08, 0.1, 0.6, 0.84])
    for key in ['L', 'G'] + LAMBDAS:
        t, v = paths[key]
        ax.plot(t, cos(v), '-', color=COLORS[key], lw=2.4, label=labels[key])
        ax.scatter(t[-1], finals[key][1], marker='X' if key in 'GL' else 'D', s=90, color=COLORS[key], edgecolor='k', zorder=5)
        if key in LAMBDAS:
            ax.scatter(t[0], cos(v[:1])[0], marker='o', s=70, color=COLORS[key], edgecolor='k', zorder=5)
            ax.axvline(t[0], color=COLORS[key], ls=':', lw=1)
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.text(990, 0.97, 'original (cos 1)', va='top', fontsize=10)
    ax.set_xlim(1000, -10)
    ax.set_xlabel(r'diffusion timestep $t$ (reverse process $\rightarrow$)')
    ax.set_ylabel(r'identity similarity to original (AdaFace cos of $\hat{x}_0$)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(0, 0.62))
    h = 0.84 / (n + 1)  # n outputs + the original, below a small gap
    column = [(k, finals[k][0], COLORS[k]) for k in order] + [('orig', Image.open(args.orig).convert('RGB'), 'k')]
    for i, (key, img, color) in enumerate(column):
        y0 = 0.94 - (i + 1) * h - (0.02 if key == 'orig' else 0)
        a = fig.add_axes([0.70, y0 + 0.005, 0.14, h - 0.01])
        a.imshow(img.resize((160, 160)))
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor(color)
            sp.set_linewidth(3)
        tag = {'G': 'Gauss', 'L': 'Lpl', 'orig': 'Original'}.get(key, rf'Mean $\lambda$={key}')
        sim = 1.0 if key == 'orig' else finals[key][1]
        a.text(1.06, 0.5, f'{tag}\ncos {sim:.2f}', transform=a.transAxes, fontsize=10, va='center')
    fig.text(0.77, 0.965, 'identity suppressed', ha='center', va='center', fontsize=11)
    fig.text(0.77, 0.025, 'identity preserved', ha='center', va='center', fontsize=11)
    sim_out = args.out.replace('.pdf', '_similarity.pdf')
    fig.savefig(sim_out)
    fig.savefig(sim_out.replace('.pdf', '.png'), dpi=150)
    print({k: round(v[1], 3) for k, v in finals.items()})
    print('saved', args.out)


if __name__ == '__main__':
    main()
