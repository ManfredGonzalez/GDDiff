"""Candidate paper figures for the gaze-preservation (data utility) results.

Privacy axis: AdaFace verification ROC-AUC on the de-identified pairs
(root_dir/results/adaface_optimized_<ds>_<tech>.csv, column 'cossim'),
lower = identity better hidden. Utility axis: mean angular gaze error (deg)
between original and de-identified image (root_dir/results/gaze/gaze_summary.csv),
lower = gaze better preserved.

Usage (env: toolkit):
    python gaze_estimation/plot_gaze_figures.py        # gaze figures -> results/gaze/figures/
    python gaze_estimation/plot_gaze_figures.py pose   # head pose figure -> results/pose/figures/
    python gaze_estimation/plot_gaze_figures.py per_dataset [gaze|pose|age|gender ...]  # one figure per dataset
    python gaze_estimation/plot_gaze_figures.py grid  # all datasets in one figure (small multiples)
    python gaze_estimation/plot_gaze_figures.py compact  # 2x4 grid (Gaze3D + SemiUHPE) and 1x2 mean/range plot
    python gaze_estimation/plot_gaze_figures.py by_dataset  # dot plot + heatmap, gaze | pose, per dataset
    python gaze_estimation/plot_gaze_figures.py expression  # expression dot plot, QCS | FMAE
    python gaze_estimation/plot_gaze_figures.py age_gender  # age (MiVOLO | HyCnXtViT), gender (MiVOLO | FairFace)
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

ROOT = os.path.expanduser('~/Desktop/master_research/deid-toolkit/root_dir/results')
GAZE_DIR = os.path.join(ROOT, 'gaze')
OUT_DIR = os.path.join(GAZE_DIR, 'figures')

DATASETS = ['CALFW', 'LFW_unified_orig', 'rafd', 'xm2vts_aligned']
DATASET_NAMES = {'CALFW': 'CALFW', 'LFW_unified_orig': 'LFW', 'rafd': 'RaFD', 'xm2vts_aligned': 'XM2VTS'}
MODEL_NAMES = {'gaze3d': 'Gaze3D (Vuillecard & Odobez, 2025)', 'gazetr': 'GazeTR (Cheng & Lu, 2022)'}

PRETTY = {
    '0.0_mean_GauDeiDiff': r'Mean-GDDiff ($\lambda$=0.0)',
    '0.1_mean_GauDeiDiff': r'Mean-GDDiff ($\lambda$=0.1)',
    '0.2_mean_GauDeiDiff': r'Mean-GDDiff ($\lambda$=0.2)',
    '0.5_mean_GauDeiDiff': r'Mean-GDDiff ($\lambda$=0.5)',
    '0.9_cpp-deid': r'CPP-DeID ($\lambda$=0.9)',
    '0.5_cpp-deid': r'CPP-DeID ($\lambda$=0.5)',
    '0.2_cpp-deid': r'CPP-DeID ($\lambda$=0.2)',
    '0.0_cpp-deid': r'CPP-DeID ($\lambda$=0.0)',
    'gauss_GauDeiDiff': 'Gauss-GDDiff',
    'lpl_GauDeiDiff': 'Lpl-GDDiff',
    'Original': 'Original Faces',
    'AMT-GAN': 'AMT-GAN',
    'cleanir': 'CLEANIR',
    'deepprivacy2': 'DeepPrivacy',
    'face_anon_diff': 'FAMS',
    'fadm': 'FADM',
    'ldfa': 'LDFA',
}
# short labels for crowded scatter plots
SHORT = {
    '0.0_mean_GauDeiDiff': r'$\lambda$=0.0', '0.1_mean_GauDeiDiff': r'$\lambda$=0.1',
    '0.2_mean_GauDeiDiff': r'$\lambda$=0.2', '0.5_mean_GauDeiDiff': r'$\lambda$=0.5',
    '0.0_cpp-deid': r'$\lambda$=0.0', '0.2_cpp-deid': r'$\lambda$=0.2',
    '0.5_cpp-deid': r'$\lambda$=0.5', '0.9_cpp-deid': r'$\lambda$=0.9',
}

OURS_MEAN = ['0.0_mean_GauDeiDiff', '0.1_mean_GauDeiDiff', '0.2_mean_GauDeiDiff', '0.5_mean_GauDeiDiff']
OURS_OTHER = ['gauss_GauDeiDiff', 'lpl_GauDeiDiff']
OURS = OURS_MEAN + OURS_OTHER
CPP = ['0.0_cpp-deid', '0.2_cpp-deid', '0.5_cpp-deid', '0.9_cpp-deid']
BASELINES = ['AMT-GAN', 'cleanir', 'deepprivacy2', 'face_anon_diff', 'fadm', 'ldfa']

# categorical slots 1-3 of the reference palette (validated all-pairs, light mode);
# other baselines stay neutral so "ours" carries the emphasis.
C_OURS, C_CPP, C_BASE = '#2a78d6', '#eb6834', '#52514e'
INK, INK_2, INK_3, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
AGGRESSIVE_AUC = 0.85  # verification AUC below which a method is a strong de-identifier

mpl.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.edgecolor': INK_3, 'axes.linewidth': 0.6,
    'xtick.color': INK_2, 'ytick.color': INK_2, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'savefig.dpi': 400, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})


def family(t):
    return 'ours' if t in OURS else 'cpp' if t in CPP else 'base'


def load(summary_csv=None, ycol='angular_error_mean'):
    summary = pd.read_csv(summary_csv or os.path.join(GAZE_DIR, 'gaze_summary.csv'))
    auc = []
    for (ds, t), _ in summary.groupby(['dataset', 'technique']):
        d = pd.read_csv(os.path.join(ROOT, f'adaface_optimized_{ds}_{t}.csv'))
        y = d.ground_truth.astype(int)
        auc.append({'dataset': ds, 'technique': t, 'auc': roc_auc_score(y, d.cossim),
                    'auc_orig': roc_auc_score(y, d.cossim_originals)})
    auc = pd.DataFrame(auc)
    df = summary.merge(auc, on=['dataset', 'technique'])
    avg = (df.groupby(['model', 'technique'])[[ycol, 'auc']].mean().reset_index())
    orig_auc = auc.groupby('dataset').auc_orig.first()
    return df, avg, orig_auc


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, name + '.pdf'))
    fig.savefig(os.path.join(OUT_DIR, name + '.png'), dpi=200)
    plt.close(fig)
    print('saved', os.path.join(OUT_DIR, name + '.pdf'))


def style_grid(ax, axis='both'):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ #
# Candidate A: privacy-utility scatter (one model)
# ------------------------------------------------------------------ #
LABEL_OFFSETS = {  # (dx, dy) in points, horizontal alignment
    'AMT-GAN': (-5, 2, 'right'), 'fadm': (5, -2, 'left'), 'face_anon_diff': (5, -3, 'left'),
    'cleanir': (5, -3, 'left'), 'deepprivacy2': (5, 1, 'left'), 'ldfa': (5, 0, 'left'),
    'gauss_GauDeiDiff': (3, -8, 'left'), 'lpl_GauDeiDiff': (-5, -6, 'right'),
}
# labels moved into free space with a leader line, per gaze model (data coords)
LEADER_LABELS = {
    'gaze3d': {'gauss_GauDeiDiff': (0.868, 9.5, 'center'), 'lpl_GauDeiDiff': (0.948, 9.4, 'center')},
    'gazetr': {'gauss_GauDeiDiff': (0.858, 14.0, 'center'), 'lpl_GauDeiDiff': (0.94, 14.0, 'center'),
               'cleanir': (0.79, 14.9, 'center'), 'AMT-GAN': (0.955, 8.7, 'center')},
}
CPP_LABELS = {  # lambda=0.0 and 0.2 land on the same spot -> one shared label
    '0.9_cpp-deid': (r'$\lambda$=0.9', 0, -8, 'center'),
    '0.5_cpp-deid': (r'$\lambda$=0.5', 0, 7, 'center'),
    '0.2_cpp-deid': (r'$\lambda$=0.0, 0.2', 2, 7, 'center'),
}


GAZE_YLABEL = r'Gaze angular error ($^\circ$) $\leftarrow$ better utility'


def draw_scatter(ax, avg, model, orig_auc_mean, labels=True, legend=True, ycol='angular_error_mean',
                 ylabel=GAZE_YLABEL, leader_labels=None, lambda_offsets=None, cpp_labels=None):
    a = avg[avg.model == model].set_index('technique')
    style_grid(ax)
    ax.axvspan(0.6, AGGRESSIVE_AUC, color='#f1f0ec', zorder=0, lw=0)
    ax.text(0.605, 0.985, 'strong de-identification\n' + rf'(AUC < {AGGRESSIVE_AUC})', transform=ax.get_xaxis_transform(),
            va='top', ha='left', fontsize=6.5, color=INK_2, style='italic')

    # family trajectories along lambda
    for members, c in [(OURS_MEAN, C_OURS), (CPP, C_CPP)]:
        ax.plot(a.loc[members, 'auc'], a.loc[members, ycol], color=c, lw=1.0, alpha=0.6, zorder=2)

    ring = dict(edgecolors='white', linewidths=0.8, zorder=3)
    ax.scatter(a.loc[BASELINES, 'auc'], a.loc[BASELINES, ycol], marker='^', s=34, c=C_BASE, **ring)
    ax.scatter(a.loc[CPP, 'auc'], a.loc[CPP, ycol], marker='s', s=26, c=C_CPP, **ring)
    ax.scatter(a.loc[OURS_MEAN, 'auc'], a.loc[OURS_MEAN, ycol], marker='o', s=36, c=C_OURS, **ring)
    ax.scatter(a.loc[OURS_OTHER, 'auc'], a.loc[OURS_OTHER, ycol], marker='D', s=24, c=C_OURS, **ring)
    ax.scatter([orig_auc_mean], [0], marker='*', s=70, c=INK, edgecolors='white', linewidths=0.6, zorder=4, clip_on=False)
    ax.annotate('Original', (orig_auc_mean, 0), xytext=(-4, 4), textcoords='offset points',
                ha='right', fontsize=6.5, color=INK_2)

    if labels:
        leaders = (leader_labels if leader_labels is not None else LEADER_LABELS).get(model, {})
        for t, (x, yv, ha) in leaders.items():
            ours_lambda = t in OURS_MEAN
            ax.annotate(SHORT[t] if ours_lambda else PRETTY[t], (a.loc[t, 'auc'], a.loc[t, ycol]), xytext=(x, yv),
                        textcoords='data', ha=ha, va='center', fontsize=6 if ours_lambda else 6.5,
                        color=C_OURS if ours_lambda else INK_2,
                        arrowprops=dict(arrowstyle='-', color=INK_3, lw=0.5, shrinkA=1, shrinkB=3))
        for t in BASELINES + OURS_OTHER:
            if t in leaders:
                continue
            dx, dy, ha = LABEL_OFFSETS.get(t, (5, 0, 'left'))
            ax.annotate(PRETTY[t], (a.loc[t, 'auc'], a.loc[t, ycol]), xytext=(dx, dy),
                        textcoords='offset points', ha=ha, va='center', fontsize=6.5, color=INK_2)
        for t in OURS_MEAN:
            if t in leaders:
                continue
            ax.annotate(SHORT[t], (a.loc[t, 'auc'], a.loc[t, ycol]), xytext=(lambda_offsets or {}).get(t, (0, -8)),
                        textcoords='offset points', ha='center', va='center', fontsize=6, color=C_OURS)
        for t, (txt, dx, dy, ha) in (cpp_labels or CPP_LABELS).items():
            ax.annotate(txt, (a.loc[t, 'auc'], a.loc[t, ycol]), xytext=(dx, dy),
                        textcoords='offset points', ha=ha, va='center', fontsize=6, color=C_CPP)

    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'Verification AUC (AdaFace) $\leftarrow$ more private')
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(handles=legend_handles(), loc='lower left', frameon=True, framealpha=0.95, edgecolor=GRID, borderpad=0.5)


def legend_handles():
    return [
        Line2D([], [], marker='o', ls='-', lw=1, color=C_OURS, mec='white', ms=6, label=r'Mean-GDDiff (ours, $\lambda$ sweep)'),
        Line2D([], [], marker='D', ls='', color=C_OURS, mec='white', ms=4.5, label='Gauss / Lpl-GDDiff (ours)'),
        Line2D([], [], marker='s', ls='-', lw=1, color=C_CPP, mec='white', ms=5, label=r'CPP-DeID ($\lambda$ sweep)'),
        Line2D([], [], marker='^', ls='', color=C_BASE, mec='white', ms=6, label='Other baselines'),
        Line2D([], [], marker='*', ls='', color=INK, mec='white', ms=8, label='Original faces'),
    ]


def fig_scatter_single(avg, orig_auc):
    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    draw_scatter(ax, avg, 'gaze3d', orig_auc.mean())
    save(fig, 'A_privacy_vs_gaze_scatter_gaze3d')


# ------------------------------------------------------------------ #
# Candidate B: same scatter for both gaze estimators side by side
# ------------------------------------------------------------------ #
def fig_scatter_both(avg, orig_auc):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    for ax, model in zip(axes, ['gaze3d', 'gazetr']):
        draw_scatter(ax, avg, model, orig_auc.mean(), legend=(model == 'gaze3d'))
        ax.set_title(MODEL_NAMES[model], loc='left', color=INK)
    axes[1].set_ylabel('')
    fig.tight_layout(w_pad=1.5)
    save(fig, 'B_privacy_vs_gaze_scatter_both_models')


# ------------------------------------------------------------------ #
# Candidate C: ranked dot plot, per-dataset spread, both models
# ------------------------------------------------------------------ #
def fig_dotplot(df, avg):
    order = (avg[avg.model == 'gaze3d'].sort_values('angular_error_mean').technique.tolist())
    ds_markers = {'CALFW': 'o', 'LFW_unified_orig': 's', 'rafd': '^', 'xm2vts_aligned': 'D'}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6), sharey=True)
    y = np.arange(len(order))
    for ax, model in zip(axes, ['gaze3d', 'gazetr']):
        style_grid(ax, 'x')
        d = df[df.model == model]
        for i, t in enumerate(order):
            if t in OURS:
                ax.axhspan(i - 0.5, i + 0.5, color='#e6effa', lw=0, zorder=0)
            c = C_OURS if t in OURS else C_CPP if t in CPP else C_BASE
            vals = d[d.technique == t].set_index('dataset').angular_error_mean
            ax.plot([vals.min(), vals.max()], [i, i], color=c, lw=1.2, alpha=0.35, solid_capstyle='round', zorder=1)
            for ds, mk in ds_markers.items():
                ax.scatter(vals[ds], i, marker=mk, s=12, facecolors='white', edgecolors=c, linewidths=0.7, zorder=2)
            m = vals.mean()
            ax.scatter(m, i, marker='|', s=90, c=c, linewidths=2.0, zorder=3)
        ax.set_title(MODEL_NAMES[model], loc='left', color=INK)
        ax.set_xlabel(r'Gaze angular error vs. original ($^\circ$)')
        ax.set_xlim(0, None)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([PRETTY[t] for t in order])
    for lbl, t in zip(axes[0].get_yticklabels(), order):
        lbl.set_color(C_OURS if t in OURS else INK)
        if t in OURS:
            lbl.set_fontweight('bold')
    axes[0].invert_yaxis()
    axes[0].tick_params(axis='y', length=0)
    handles = [Line2D([], [], marker='|', ls='', color=INK_2, ms=9, mew=2, label='Mean over datasets')]
    handles += [Line2D([], [], marker=mk, ls='', mfc='white', mec=INK_2, ms=4, label=DATASET_NAMES[ds])
                for ds, mk in ds_markers.items()]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.55, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, 'C_gaze_error_dotplot')


# ------------------------------------------------------------------ #
# Candidate D: heatmap technique x dataset, rows grouped by privacy level
# ------------------------------------------------------------------ #
def fig_heatmap(df, avg):
    g = avg[avg.model == 'gaze3d'].set_index('technique')
    strong = g[g.auc < AGGRESSIVE_AUC].sort_values('angular_error_mean').index.tolist()
    weak = g[g.auc >= AGGRESSIVE_AUC].sort_values('angular_error_mean').index.tolist()
    order = strong + weak
    cmap = mpl.colors.LinearSegmentedColormap.from_list('seq_blue', ['#f4f8fd', '#a9c9f0', '#2a78d6', '#0d3a73'])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.7), sharey=True)
    for ax, model in zip(axes, ['gaze3d', 'gazetr']):
        d = df[df.model == model].pivot(index='technique', columns='dataset', values='angular_error_mean')
        d = d.loc[order, DATASETS]
        d['Mean'] = d.mean(1)
        vmax = np.ceil(d.values.max())
        im = ax.imshow(d.values, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
        for (i, j), v in np.ndenumerate(d.values):
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=6.3,
                    color='white' if v > 0.55 * vmax else INK, fontweight='bold' if j == 4 else 'normal')
        ax.set_xticks(range(5))
        ax.set_xticklabels([DATASET_NAMES[c] for c in DATASETS] + ['Mean'])
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axvline(3.5, color='white', lw=2.5)
        ax.axhline(len(strong) - 0.5, color=INK, lw=0.9)
        ax.set_title(MODEL_NAMES[model], color=INK, pad=16)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6, length=2)
        cb.set_label(r'Angular error ($^\circ$)', fontsize=7)
    axes[0].set_yticks(range(len(order)))
    labels = [f'{PRETTY[t]}  ({g.loc[t, "auc"]:.2f})' for t in order]
    axes[0].set_yticklabels(labels)
    for lbl, t in zip(axes[0].get_yticklabels(), order):
        lbl.set_color(C_OURS if t in OURS else INK)
        if t in OURS:
            lbl.set_fontweight('bold')
    fig.text(0.01, 0.01, 'Rows: method (mean verification AUC). Above the line: strong de-identification '
             rf'(AUC < {AGGRESSIVE_AUC}); below: weak. Ours in blue.', fontsize=6.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, 'D_gaze_error_heatmap')


# ------------------------------------------------------------------ #
# Head pose: same layout as candidate B (both pose estimators side by side)
# ------------------------------------------------------------------ #
POSE_DIR = os.path.join(ROOT, 'pose')
POSE_MODEL_NAMES = {'semiuhpe': 'SemiUHPE (Zhou et al., 2025)', '6drepnet360': '6DRepNet360 (Hempel et al., 2024)'}
POSE_YLABEL = r'Head pose error ($^\circ$) $\leftarrow$ better utility'
# the methods bunch up at 2-4 deg, so most labels sit in free space with a leader line (data coords)
EXPR_DIR = os.path.join(ROOT, 'expression')
AG_DIR = os.path.join(ROOT, 'age_gender')
AG_MODEL_NAMES = {'mivolo': 'MiVOLO', 'hycnxtvit': 'HyCnXtViT', 'fairface': 'FairFace'}
EXPR_MODEL_NAMES = {'qcs': 'QCS (Wang et al., 2025)', 'fmae': 'FMAE (Ning et al., 2025)'}
EXPR_YLABEL = r'Expression change rate (%) $\leftarrow$ better utility'

POSE_LEADER_LABELS = {
    'semiuhpe': {'cleanir': (0.745, 1.3, 'center'), 'face_anon_diff': (0.815, 0.8, 'center'),
                 'fadm': (0.885, 1.25, 'center'), 'AMT-GAN': (0.962, 2.75, 'center'),
                 'lpl_GauDeiDiff': (0.915, 4.65, 'center'), 'gauss_GauDeiDiff': (0.88, 5.3, 'center'),
                 'deepprivacy2': (0.835, 6.1, 'center'), '0.0_mean_GauDeiDiff': (0.86, 2.5, 'center')},
    '6drepnet360': {'face_anon_diff': (0.80, 0.9, 'center'), 'cleanir': (0.735, 1.9, 'center'),
                    'fadm': (0.882, 0.9, 'center'), 'AMT-GAN': (0.972, 3.0, 'center'),
                    'lpl_GauDeiDiff': (0.93, 7.6, 'center'), 'gauss_GauDeiDiff': (0.885, 8.2, 'center'),
                    'deepprivacy2': (0.83, 9.0, 'center')},
}
POSE_LAMBDA_OFFSETS = {'6drepnet360': {'0.1_mean_GauDeiDiff': (0, 8)}}  # clear CLEANIR's leader
POSE_CPP_LABELS = {
    '0.9_cpp-deid': (r'$\lambda$=0.9', 0, -8, 'center'),
    '0.5_cpp-deid': (r'$\lambda$=0.5', 0, 7, 'center'),
    '0.2_cpp-deid': (r'$\lambda$=0.0, 0.2', 0, 8, 'center'),
}


def fig_pose_scatter_both():
    _, avg, orig_auc = load(os.path.join(POSE_DIR, 'pose_summary.csv'), ycol='geodesic_error_mean')
    avg.to_csv(os.path.join(POSE_DIR, 'pose_privacy_summary.csv'), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
    for ax, model in zip(axes, ['semiuhpe', '6drepnet360']):
        draw_scatter(ax, avg, model, orig_auc.mean(), legend=False, ycol='geodesic_error_mean',
                     ylabel=POSE_YLABEL, leader_labels=POSE_LEADER_LABELS, cpp_labels=POSE_CPP_LABELS,
                     lambda_offsets=POSE_LAMBDA_OFFSETS.get(model))
        ax.set_ylim(0, avg[avg.model == model].geodesic_error_mean.max() * 1.25)  # headroom for the region note
        ax.set_title(POSE_MODEL_NAMES[model], loc='left', color=INK)
    axes[1].set_ylabel('')
    # shared legend under the panels: nothing in the clustered region gets covered
    fig.legend(handles=legend_handles(), loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.5, rect=(0, 0.07, 1, 1))
    os.makedirs(os.path.join(POSE_DIR, 'figures'), exist_ok=True)
    for ext, kw in [('pdf', {}), ('png', {'dpi': 200})]:
        fig.savefig(os.path.join(POSE_DIR, 'figures', f'B_privacy_vs_headpose_scatter_both_models.{ext}'), **kw)
    plt.close(fig)
    print('saved', os.path.join(POSE_DIR, 'figures', 'B_privacy_vs_headpose_scatter_both_models.pdf'))


# ------------------------------------------------------------------ #
# Per-dataset versions of candidate B (gaze and head pose), labels placed
# automatically with adjustText (pip install adjustText)
# ------------------------------------------------------------------ #
TASKS = {
    'gaze': dict(summary=os.path.join(GAZE_DIR, 'gaze_summary.csv'), ycol='angular_error_mean',
                 models=['gaze3d', 'gazetr'], names=MODEL_NAMES, ylabel=GAZE_YLABEL,
                 out=os.path.join(GAZE_DIR, 'figures', 'per_dataset'), stem='B_privacy_vs_gaze'),
    'pose': dict(summary=os.path.join(POSE_DIR, 'pose_summary.csv'), ycol='geodesic_error_mean',
                 models=['semiuhpe', '6drepnet360'], names=POSE_MODEL_NAMES, ylabel=POSE_YLABEL,
                 out=os.path.join(POSE_DIR, 'figures', 'per_dataset'), stem='B_privacy_vs_headpose'),
    'expression': dict(summary=os.path.join(EXPR_DIR, 'expression_summary.csv'), ycol='change_rate',
                       models=['qcs', 'fmae'], names=EXPR_MODEL_NAMES, ylabel=EXPR_YLABEL,
                       out=os.path.join(EXPR_DIR, 'figures', 'per_dataset'), stem='B_privacy_vs_expression'),
    'age': dict(summary=os.path.join(AG_DIR, 'age_gender_summary.csv'), ycol='age_abs_diff_mean',
                models=['mivolo', 'hycnxtvit'], names=AG_MODEL_NAMES,
                ylabel=r'Age difference (years) $\leftarrow$ better utility',
                out=os.path.join(AG_DIR, 'figures', 'per_dataset'), stem='B_privacy_vs_age'),
    'gender': dict(summary=os.path.join(AG_DIR, 'age_gender_summary.csv'), ycol='gender_flip_rate',
                   models=['mivolo', 'fairface'], names=AG_MODEL_NAMES,
                   ylabel=r'Gender flip rate (%) $\leftarrow$ better utility',
                   out=os.path.join(AG_DIR, 'figures', 'per_dataset'), stem='B_privacy_vs_gender'),
}
DOT_XLABELS = {'age': 'Absolute age difference vs. original (years)',
               'gender': 'Gender flip rate vs. original (%)',
               'gaze': r'Gaze angular error vs. original ($^\circ$)',
               'pose': r'Head pose error vs. original ($^\circ$)',
               'expression': 'Expression change rate vs. original (%)'}


def draw_scatter_auto(ax, d, orig_auc, ycol, ylabel, xlim):
    from adjustText import adjust_text
    a = d.set_index('technique').copy()
    # a single method far above all others squeezes the rest; draw it at the top edge instead
    vals = a[ycol].sort_values(ascending=False)
    capped = {}
    if vals.iloc[0] > 2 * vals.iloc[1]:
        ymax = vals.iloc[1] * 1.45
        for t in vals.index[vals > ymax]:
            capped[t] = a.loc[t, ycol]
            a.loc[t, ycol] = ymax * 0.88
    else:
        ymax = vals.iloc[0] * 1.3
    style_grid(ax)
    ax.axvspan(xlim[0], AGGRESSIVE_AUC, color='#f1f0ec', zorder=0, lw=0)
    ax.text(0.015, 0.02, 'strong de-identification\n' + rf'(AUC < {AGGRESSIVE_AUC})', transform=ax.transAxes,
            va='bottom', ha='left', fontsize=6.5, color=INK_2, style='italic')
    for members, c in [(OURS_MEAN, C_OURS), (CPP, C_CPP)]:
        ax.plot(a.loc[members, 'auc'], a.loc[members, ycol], color=c, lw=1.0, alpha=0.6, zorder=2)
    ring = dict(edgecolors='white', linewidths=0.8, zorder=3)
    ax.scatter(a.loc[BASELINES, 'auc'], a.loc[BASELINES, ycol], marker='^', s=34, c=C_BASE, **ring)
    ax.scatter(a.loc[CPP, 'auc'], a.loc[CPP, ycol], marker='s', s=26, c=C_CPP, **ring)
    ax.scatter(a.loc[OURS_MEAN, 'auc'], a.loc[OURS_MEAN, ycol], marker='o', s=36, c=C_OURS, **ring)
    ax.scatter(a.loc[OURS_OTHER, 'auc'], a.loc[OURS_OTHER, ycol], marker='D', s=24, c=C_OURS, **ring)
    ax.scatter([orig_auc], [0], marker='*', s=70, c=INK, edgecolors='white', linewidths=0.6, zorder=4, clip_on=False)
    # no 'Original' text here: the star is explained in the legend and often sits on top of AMT-GAN
    for t, true_v in capped.items():  # off-scale marker: arrow at the top edge with the real value
        ax.annotate('', (a.loc[t, 'auc'], ymax), xytext=(a.loc[t, 'auc'], a.loc[t, ycol]),
                    arrowprops=dict(arrowstyle='->', color=INK_3, lw=0.7), zorder=2)
        ax.annotate(rf'(off-scale: {true_v:.1f}$^\circ$)', (a.loc[t, 'auc'], a.loc[t, ycol]), xytext=(7, -1),
                    textcoords='offset points', ha='left', va='center', fontsize=6, color=INK_2, style='italic')

    ax.set_xlim(*xlim)
    ax.set_ylim(0, ymax)
    texts = []
    for t in BASELINES + OURS_OTHER:
        texts.append(ax.text(a.loc[t, 'auc'], a.loc[t, ycol], PRETTY[t], fontsize=6.5, color=INK_2, zorder=5))
    for t in OURS_MEAN:
        texts.append(ax.text(a.loc[t, 'auc'], a.loc[t, ycol], SHORT[t], fontsize=6, color=C_OURS, zorder=5))
    for t in CPP:
        texts.append(ax.text(a.loc[t, 'auc'], a.loc[t, ycol], SHORT[t], fontsize=6, color=C_CPP, zorder=5))
    # obstacles: every marker plus points sampled along the two lambda-sweep lines
    ox, oy = [a['auc'].values, [orig_auc]], [a[ycol].values, [0.0]]
    for members in (OURS_MEAN, CPP):
        xs, ys = a.loc[members, 'auc'].values, a.loc[members, ycol].values
        for i in range(len(members) - 1):
            ox.append(np.linspace(xs[i], xs[i + 1], 12)[1:-1])
            oy.append(np.linspace(ys[i], ys[i + 1], 12)[1:-1])
    adjust_text(texts, ax=ax, x=np.concatenate(ox), y=np.concatenate(oy), expand=(1.2, 1.5),
                force_text=(0.2, 0.4), force_static=(0.25, 0.5), max_move=(12, 12), iter_lim=2000,
                min_arrow_len=6, arrowprops=dict(arrowstyle='-', color=INK_3, lw=0.5))
    ax.set_xlabel(r'Verification AUC (AdaFace) $\leftarrow$ more private')
    ax.set_ylabel(ylabel)


def fig_per_dataset(task):
    cfg = TASKS[task]
    df, _, orig_auc = load(cfg['summary'], ycol=cfg['ycol'])
    lo = np.floor(min(df.auc.min(), AGGRESSIVE_AUC) * 20) / 20 - 0.02
    os.makedirs(cfg['out'], exist_ok=True)
    for ds in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
        for ax, model in zip(axes, cfg['models']):
            d = df[(df.model == model) & (df.dataset == ds)]
            draw_scatter_auto(ax, d, orig_auc[ds], cfg['ycol'], cfg['ylabel'], (lo, 1.0))
            ax.set_title(f"{cfg['names'][model]} | {DATASET_NAMES[ds]}", loc='left', color=INK)
        axes[1].set_ylabel('')
        fig.legend(handles=legend_handles(), loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(w_pad=1.5, rect=(0, 0.07, 1, 1))
        base = os.path.join(cfg['out'], f"{cfg['stem']}_{DATASET_NAMES[ds]}")
        fig.savefig(base + '.pdf')
        fig.savefig(base + '.png', dpi=200)
        plt.close(fig)
        print('saved', base + '.pdf')


# ------------------------------------------------------------------ #
# Small multiples: rows = estimators, columns = datasets, no text labels.
# Identity by marker shape (baselines) and colour family + shade (lambda sweeps).
# ------------------------------------------------------------------ #
BASELINE_MARKERS = {'AMT-GAN': '^', 'cleanir': 'v', 'deepprivacy2': 'P', 'face_anon_diff': 'X',
                    'fadm': 'h', 'ldfa': 'p'}
OURS_SHADES = ['#9cc3f0', '#5f9de6', '#2a78d6', '#164f99']   # lambda 0.0 -> 0.5, light -> dark
CPP_SHADES = ['#f5b596', '#f08e60', '#eb6834', '#b04619']    # lambda 0.0 -> 0.9


def draw_grid_panel(ax, d, orig_auc, ycol, ylim, cap):
    a = d.set_index('technique').copy()
    style_grid(ax)
    ax.axvspan(0.5, AGGRESSIVE_AUC, color='#f1f0ec', zorder=0, lw=0)
    y = a[ycol].clip(upper=cap)
    ring = dict(edgecolors='white', linewidths=0.6, zorder=3)
    for members, shades, mk, ms in [(OURS_MEAN, OURS_SHADES, 'o', 26), (CPP, CPP_SHADES, 's', 20)]:
        ax.plot(a.loc[members, 'auc'], y[members], color=shades[2], lw=0.9, alpha=0.55, zorder=2)
        ax.scatter(a.loc[members, 'auc'], y[members], c=shades, marker=mk, s=ms, **ring)
    ax.scatter(a.loc[OURS_OTHER, 'auc'], y[OURS_OTHER], marker='D', s=16, c=[OURS_SHADES[1], OURS_SHADES[3]], **ring)
    for t, mk in BASELINE_MARKERS.items():
        ax.scatter(a.loc[t, 'auc'], y[t], marker=mk, s=26, c=C_BASE, **ring)
    for t in a.index[a[ycol] > cap]:  # off-scale: arrow + true value
        ax.annotate('', (a.loc[t, 'auc'], ylim), xytext=(a.loc[t, 'auc'], cap),
                    arrowprops=dict(arrowstyle='->', color=INK_3, lw=0.6), zorder=2)
        ax.annotate(rf'{a.loc[t, ycol]:.0f}$^\circ$', (a.loc[t, 'auc'], cap), xytext=(4, -1), textcoords='offset points',
                    ha='left', va='center', fontsize=5.5, color=INK_2, style='italic')
    ax.scatter([orig_auc], [0], marker='*', s=50, c=INK, edgecolors='white', linewidths=0.5, zorder=4, clip_on=False)
    ax.set_ylim(0, ylim)


def grid_rows(task_names, models=None):
    rows = []
    for task in task_names:
        cfg = TASKS[task]
        df, _, orig_auc = load(cfg['summary'], ycol=cfg['ycol'])
        for model in (models or {}).get(task, cfg['models']):
            rows.append(dict(task=task, model=model, df=df[df.model == model], orig_auc=orig_auc,
                             ycol=cfg['ycol'], name=cfg['names'][model]))
    return rows


def grid_legend_handles():
    h = [Line2D([], [], marker='o', ls='-', lw=0.9, color=OURS_SHADES[2], mfc=OURS_SHADES[2], mec='white', ms=5,
                label=r'Mean-GDDiff (ours), $\lambda$ 0.0$\rightarrow$0.5 light$\rightarrow$dark'),
         Line2D([], [], marker='D', ls='', color=OURS_SHADES[1], mec='white', ms=4, label='Gauss-GDDiff (ours)'),
         Line2D([], [], marker='D', ls='', color=OURS_SHADES[3], mec='white', ms=4, label='Lpl-GDDiff (ours)'),
         Line2D([], [], marker='s', ls='-', lw=0.9, color=CPP_SHADES[2], mec='white', ms=4.5,
                label=r'CPP-DeID, $\lambda$ 0.0$\rightarrow$0.9 light$\rightarrow$dark')]
    h += [Line2D([], [], marker=mk, ls='', color=C_BASE, mec='white', ms=5.5, label=PRETTY[t])
          for t, mk in BASELINE_MARKERS.items()]
    h += [Line2D([], [], marker='*', ls='', color=INK, mec='white', ms=7, label='Original faces'),
          Line2D([], [], marker='s', ls='', color='#f1f0ec', ms=8, label=rf'Strong de-id. (AUC < {AGGRESSIVE_AUC})')]
    return h


def fig_grid(task_names, out_path, models=None):
    rows = grid_rows(task_names, models)
    all_auc = pd.concat([r['df'] for r in rows])
    lo = np.floor(min(all_auc.auc.min(), AGGRESSIVE_AUC) * 20) / 20 - 0.02
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(7.1, 1.75 * n + 0.9), sharex='col', squeeze=False)  # 7.1in = 2-col width
    for i, r in enumerate(rows):
        v = r['df'][r['ycol']].values
        q = np.quantile(v, 0.97)
        cap = q * 1.15 if v.max() > 1.5 * q else np.inf  # shared y per row; clip only real outliers
        ylim = (q * 1.15 if np.isfinite(cap) else v.max()) * 1.1
        cap = min(cap, ylim * 0.94)
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            draw_grid_panel(ax, r['df'][r['df'].dataset == ds], r['orig_auc'][ds], r['ycol'], ylim,
                            cap if np.isfinite(cap) else np.inf)
            ax.set_xlim(lo, 1.0)
            if j > 0:
                ax.tick_params(labelleft=False)
            if i == 0:
                ax.set_title(DATASET_NAMES[ds], color=INK, fontsize=8.5)
            if i == n - 1:
                ax.set_xlabel('Verification AUC', fontsize=7)
        unit = 'gaze error' if r['task'] == 'gaze' else 'head pose error'
        axes[i, 0].set_ylabel(f"{r['name'].split(' (')[0]}\n{unit} ($^\\circ$)", fontsize=7)
    fig.text(0.5, 0.0, r'$\leftarrow$ more private (x)     |     $\downarrow$ better utility (y)', ha='center',
             fontsize=6.5, color=INK_2)
    fig.legend(handles=grid_legend_handles(), loc='lower center', ncol=4, frameon=False, fontsize=6.3,
               bbox_to_anchor=(0.5, 0.015), columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(h_pad=0.6, w_pad=0.4, rect=(0, 0.62 / (1.75 * n + 0.9), 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path + '.pdf')
    fig.savefig(out_path + '.png', dpi=200)
    plt.close(fig)
    print('saved', out_path + '.pdf')


# ------------------------------------------------------------------ #
# Most compact: 1 x 2 (gaze | pose), one point per method = mean over the
# 4 datasets, vertical bar = min-max error across the datasets.
# ------------------------------------------------------------------ #
PRIMARY = {'gaze': 'gaze3d', 'pose': 'semiuhpe'}


def fig_compact_range(out_path):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    for ax, task in zip(axes, ['gaze', 'pose']):
        cfg = TASKS[task]
        df, _, orig_auc = load(cfg['summary'], ycol=cfg['ycol'])
        d = df[df.model == PRIMARY[task]]
        g = d.groupby('technique').agg(x=('auc', 'mean'), xmin=('auc', 'min'), xmax=('auc', 'max'),
                                       y=(cfg['ycol'], 'mean'), ymin=(cfg['ycol'], 'min'), ymax=(cfg['ycol'], 'max'))
        style_grid(ax)
        ax.axvspan(0.5, AGGRESSIVE_AUC, color='#f1f0ec', zorder=0, lw=0)
        colour = {t: OURS_SHADES[i] for i, t in enumerate(OURS_MEAN)}
        colour.update({t: CPP_SHADES[i] for i, t in enumerate(CPP)})
        colour.update({'gauss_GauDeiDiff': OURS_SHADES[1], 'lpl_GauDeiDiff': OURS_SHADES[3]})
        for t, r in g.iterrows():  # range bars first, faint, so the means sit on top
            c = colour.get(t, C_BASE)
            # only the utility spread: the AUC spread is identical in both panels and crowds the plot
            ax.plot([r.x, r.x], [r.ymin, r.ymax], color=c, lw=1.1, alpha=0.4, zorder=1, solid_capstyle='round')
        ring = dict(edgecolors='white', linewidths=0.7, zorder=3)
        for members, shades, mk, ms in [(OURS_MEAN, OURS_SHADES, 'o', 34), (CPP, CPP_SHADES, 's', 26)]:
            ax.plot(g.loc[members, 'x'], g.loc[members, 'y'], color=shades[2], lw=1.0, alpha=0.6, zorder=2)
            ax.scatter(g.loc[members, 'x'], g.loc[members, 'y'], c=shades, marker=mk, s=ms, **ring)
        ax.scatter(g.loc[OURS_OTHER, 'x'], g.loc[OURS_OTHER, 'y'], marker='D', s=22,
                   c=[OURS_SHADES[1], OURS_SHADES[3]], **ring)
        for t, mk in BASELINE_MARKERS.items():
            ax.scatter(g.loc[t, 'x'], g.loc[t, 'y'], marker=mk, s=34, c=C_BASE, **ring)
        ax.scatter([orig_auc.mean()], [0], marker='*', s=60, c=INK, edgecolors='white', linewidths=0.5,
                   zorder=4, clip_on=False)
        ax.set_xlim(0.55, 1.0)
        ax.set_ylim(0, None)
        ax.set_xlabel(r'Verification AUC (AdaFace) $\leftarrow$ more private')
        unit = 'Gaze angular error' if task == 'gaze' else 'Head pose error'
        ax.set_ylabel(unit + r' ($^\circ$) $\leftarrow$ better utility')
        ax.set_title(cfg['names'][PRIMARY[task]], loc='left', color=INK)
    fig.legend(handles=grid_legend_handles() + [Line2D([], [], color=C_BASE, lw=0.8, alpha=0.5,
               label='Error range over the 4 datasets')], loc='lower center', ncol=5, frameon=False, fontsize=6.3,
               bbox_to_anchor=(0.5, -0.01), columnspacing=1.1, handletextpad=0.4)
    fig.tight_layout(w_pad=1.5, rect=(0, 0.17, 1, 1))
    fig.savefig(out_path + '.pdf')
    fig.savefig(out_path + '.png', dpi=200)
    plt.close(fig)
    print('saved', out_path + '.pdf')


# ------------------------------------------------------------------ #
# Per-dataset dot plot and heatmap: rows = methods sorted by privacy
# (mean verification AUC, most private on top), gaze | pose side by side.
# ------------------------------------------------------------------ #
DS_MARKERS = {'CALFW': 'o', 'LFW_unified_orig': 's', 'rafd': '^', 'xm2vts_aligned': 'D'}


def privacy_sorted_tables(panels=(('gaze', 'gaze3d'), ('pose', 'semiuhpe'))):
    """Error tables (technique x dataset) per (task, model) panel, rows sorted by mean AUC."""
    out = {}
    for task, model in panels:
        cfg = TASKS[task]
        df, _, _ = load(cfg['summary'], ycol=cfg['ycol'])
        d = df[df.model == model]
        out[(task, model)] = d.pivot(index='technique', columns='dataset', values=cfg['ycol'])[DATASETS]
        auc = d.pivot(index='technique', columns='dataset', values='auc')[DATASETS]
    order = auc.mean(1).sort_values().index.tolist()  # AUC is the same table for every panel
    return out, auc.loc[order], order


def label_rows(ax, order, auc):
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f'{PRETTY[t]}  ({auc.loc[t].mean():.2f})' for t in order])
    for lbl, t in zip(ax.get_yticklabels(), order):
        lbl.set_color(C_OURS if t in OURS else INK)
        if t in OURS:
            lbl.set_fontweight('bold')
    ax.tick_params(axis='y', length=0)


DOTPLOT_FONT_BUMP = 2  # pt added to every text size in the dot plots


def fig_dotplot_by_dataset(out_path, panels=(('gaze', 'gaze3d'), ('pose', 'semiuhpe'))):
    bump = DOTPLOT_FONT_BUMP
    sizes = {k: mpl.rcParams[k] + bump for k in ['font.size', 'axes.labelsize', 'axes.titlesize',
                                                  'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize']}
    with mpl.rc_context(sizes):
        _dotplot(out_path, panels, bump)


def _dotplot(out_path, panels, bump):
    tables, auc, order = privacy_sorted_tables(panels)
    n_strong = int((auc.mean(1) < AGGRESSIVE_AUC).sum())
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.4), sharey=True)
    y = np.arange(len(order))
    for ax, (task, model) in zip(axes, panels):
        t_ = tables[(task, model)].loc[order]
        style_grid(ax, 'x')
        for i, t in enumerate(order):
            if t in OURS:
                ax.axhspan(i - 0.5, i + 0.5, color='#e6effa', lw=0, zorder=0)
            c = C_OURS if t in OURS else C_CPP if t in CPP else C_BASE
            vals = t_.loc[t]
            ax.plot([vals.min(), vals.max()], [i, i], color=c, lw=1.0, alpha=0.35, zorder=1)
            for ds, mk in DS_MARKERS.items():
                ax.scatter(vals[ds], i, marker=mk, s=22, facecolors='white', edgecolors=c, linewidths=0.9, zorder=2)
        ax.axhline(n_strong - 0.5, color=INK_2, lw=0.7, ls=(0, (3, 2)))
        ax.set_xlim(0, None)
        # note just under the divider, on the right unless the first row below reaches that far
        below = t_.loc[order[n_strong]]
        xmax = ax.get_xlim()[1]
        right = below.max() < 0.68 * xmax
        # otherwise on the left, just above the divider (rows there start further right)
        ax.text(0.99 if right else 0.01, (n_strong - 0.5) / len(order), rf'$\uparrow$ AUC < {AGGRESSIVE_AUC}',
                transform=ax.transAxes, ha='right' if right else 'left', va='top' if right else 'bottom',
                fontsize=6 + bump, color=INK_2, style='italic')
        ax.set_title(TASKS[task]['names'][model].split(' (')[0], loc='left', color=INK)  # no citation
        ax.set_xlabel(DOT_XLABELS[task])
    # transAxes y for the divider label must follow the inverted axis
    axes[0].invert_yaxis()
    for ax in axes:
        for txt in ax.texts:
            below_line = txt.get_va() == 'top'
            txt.set_y(1 - n_strong / len(order) + (-0.008 if below_line else 0.006))
    label_rows(axes[0], order, auc)
    handles = [Line2D([], [], marker=mk, ls='', mfc='white', mec=INK_2, ms=5.5, label=DATASET_NAMES[ds])
               for ds, mk in DS_MARKERS.items()]
    handles.append(Line2D([], [], color=INK_2, lw=1, alpha=0.4, label='range across datasets'))
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.06, 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path + '.pdf')
    fig.savefig(out_path + '.png', dpi=200)
    plt.close(fig)
    print('saved', out_path + '.pdf')


def fig_heatmap_by_dataset(out_path):
    tables, auc, order = privacy_sorted_tables()
    n_strong = int((auc.mean(1) < AGGRESSIVE_AUC).sum())
    cmap = mpl.colors.LinearSegmentedColormap.from_list('seq_blue', ['#f4f8fd', '#a9c9f0', '#2a78d6', '#0d3a73'])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.9), sharey=True)
    for ax, task in zip(axes, ['gaze', 'pose']):
        t_ = tables[(task, PRIMARY[task])].loc[order]
        vmax = np.quantile(t_.values, 0.98)  # an outlier alone should not wash out the scale
        im = ax.imshow(t_.values, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
        for (i, j), v in np.ndenumerate(t_.values):
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=6.2,
                    color='white' if v > 0.55 * vmax else INK)
        ax.set_xticks(range(4))
        ax.set_xticklabels([DATASET_NAMES[d] for d in DATASETS])
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(n_strong - 0.5, color=INK, lw=0.9)
        model = PRIMARY[task]
        ax.set_title(f"{TASKS[task]['names'][model].split(' (')[0]}: "
                     + ('gaze error' if task == 'gaze' else 'head pose error') + r' ($^\circ$)', color=INK, pad=14)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, extend='max')
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6, length=2)
    label_rows(axes[0], order, auc)
    fig.text(0.01, 0.01, 'Rows: method (mean verification AUC), most private first; line = AUC '
             rf'{AGGRESSIVE_AUC}. Ours in blue.', fontsize=6.3, color=INK_2)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.03, 1, 1))
    fig.savefig(out_path + '.pdf')
    fig.savefig(out_path + '.png', dpi=200)
    plt.close(fig)
    print('saved', out_path + '.pdf')


def main():
    import sys
    if 'by_dataset' in sys.argv[1:]:
        fig_dotplot_by_dataset(os.path.join(GAZE_DIR, 'figures', 'dotplot_gaze_and_headpose_by_dataset'))
        fig_heatmap_by_dataset(os.path.join(GAZE_DIR, 'figures', 'heatmap_gaze_and_headpose_by_dataset'))
        fig_dotplot_by_dataset(os.path.join(GAZE_DIR, 'figures', 'dotplot_gaze_by_dataset_both_models'),
                               panels=(('gaze', 'gaze3d'), ('gaze', 'gazetr')))
        fig_dotplot_by_dataset(os.path.join(POSE_DIR, 'figures', 'dotplot_headpose_by_dataset_both_models'),
                               panels=(('pose', 'semiuhpe'), ('pose', '6drepnet360')))
        return
    if 'age_gender' in sys.argv[1:]:
        fig_dotplot_by_dataset(os.path.join(AG_DIR, 'figures', 'dotplot_age_by_dataset_both_models'),
                               panels=(('age', 'mivolo'), ('age', 'hycnxtvit')))
        fig_dotplot_by_dataset(os.path.join(AG_DIR, 'figures', 'dotplot_gender_by_dataset_both_models'),
                               panels=(('gender', 'mivolo'), ('gender', 'fairface')))
        return
    if 'expression' in sys.argv[1:]:
        fig_dotplot_by_dataset(os.path.join(EXPR_DIR, 'figures', 'dotplot_expression_by_dataset_both_models'),
                               panels=(('expression', 'qcs'), ('expression', 'fmae')))
        return
    if 'compact' in sys.argv[1:]:
        fig_grid(['gaze', 'pose'], os.path.join(GAZE_DIR, 'figures', 'compact_grid_gaze3d_semiuhpe_all_datasets'),
                 models={'gaze': ['gaze3d'], 'pose': ['semiuhpe']})
        fig_compact_range(os.path.join(GAZE_DIR, 'figures', 'compact_mean_range_gaze_and_headpose'))
        return
    if 'grid' in sys.argv[1:]:
        fig_grid(['gaze'], os.path.join(GAZE_DIR, 'figures', 'grid_privacy_vs_gaze_all_datasets'))
        fig_grid(['pose'], os.path.join(POSE_DIR, 'figures', 'grid_privacy_vs_headpose_all_datasets'))
        fig_grid(['gaze', 'pose'], os.path.join(ROOT, 'gaze', 'figures', 'grid_privacy_vs_gaze_and_headpose_all_datasets'))
        return
    if 'per_dataset' in sys.argv[1:]:
        tasks = [t for t in ('gaze', 'pose', 'age', 'gender') if t in sys.argv[2:]] or ['gaze', 'pose']
        for t in tasks:
            fig_per_dataset(t)
        return
    if 'pose' in sys.argv[1:]:
        fig_pose_scatter_both()
        return
    df, avg, orig_auc = load()
    avg.to_csv(os.path.join(GAZE_DIR, 'gaze_privacy_summary.csv'), index=False)
    fig_scatter_single(avg, orig_auc)
    fig_scatter_both(avg, orig_auc)
    fig_dotplot(df, avg)
    fig_heatmap(df, avg)


if __name__ == '__main__':
    main()
