"""Failure-case gallery for the face-detection robustness study (reviewer R2.3).

Rows, selected automatically:
  * real detector failures on the full datasets (existing Mean-GDDiff lambda=0.5 outputs):
    the two detected boxes with the lowest IoU against MTCNN, and one fallback case;
  * from the robustness sweep: the images that retain the most identity (highest
    AdaFace cos(original, output)) under the largest box shift and under each occluder,
    next to the same image's unperturbed output.
Each row: input with the box used (red) and the MTCNN reference (green) | unperturbed
Mean-GDDiff output | perturbed / failure output, annotated with cos(original, output).

Usage (env: gddiff):
    python analysis/make_failure_gallery.py --inputs <robust_inputs> --out analysis/results_detection/failure_gallery.pdf
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'analysis'))
from robustness_metrics import CONDITIONS, SETS, output_path, perturb_bbox  # noqa: E402

TOOLKIT = os.path.expanduser('~/Desktop/master_research/deid-toolkit/root_dir')
RES = os.path.join(REPO, 'analysis/results_detection')
DS_NAME = {'rafd': 'RaFD', 'LFW_unified_orig': 'LFW', 'CALFW': 'CALFW', 'xm2vts_aligned': 'XM2VTS'}
COND_LABEL = {'shift040': 'box shift 0.40', 'occl_eyes': 'eye occluder', 'occl_lower': 'lower-face occluder',
              'occl_patch': 'random patch'}


def show(ax, path, title, boxes=()):
    ax.imshow(Image.open(path).convert('RGB').resize((256, 256)))
    for (x, y, w, h), color in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=1.6, ec=color))
    ax.set_title(title, fontsize=6.6, pad=2)
    ax.axis('off')


def existing_output(ds, img):
    folder = os.path.join(TOOLKIT, 'datasets/GauDeiDiff', ds, '0.5_mean')
    for ext in ('.jpg', '.png'):
        if os.path.exists(os.path.join(folder, img + ext)):
            return os.path.join(folder, img + ext)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--out', default=os.path.join(RES, 'failure_gallery.pdf'))
    args = ap.parse_args()
    det = pd.read_csv(os.path.join(RES, 'detection_study.csv'))
    per_img = pd.read_csv(os.path.join(RES, 'robustness_identity_per_image.csv'))
    per_img = per_img[per_img.branch == 'mean']
    box_of = lambda r: (r.used_x, r.used_y, r.used_w, r.used_h)
    ref_of = lambda r: (r.ref_x, r.ref_y, r.ref_w, r.ref_h)
    rows = []

    # real detector failures (existing outputs, all datasets)
    off = det[~det.fallback & det.mtcnn_found].sort_values('iou')
    picks = [r for _, r in off.iterrows() if existing_output(r.dataset, r.img)][:2]
    fb = [r for _, r in det[det.fallback & det.mtcnn_found].iterrows() if existing_output(r.dataset, r.img)][:1]
    for r in picks + fb:
        kind = 'no detection (fallback)' if r.fallback else f'misplaced box (IoU {r.iou:.2f})'
        rows.append(dict(input=os.path.join(TOOLKIT, 'datasets/aligned', r.dataset, r.img + '.jpg'),
                         input_title=f'{DS_NAME[r.dataset]}: {kind}',
                         boxes=[(box_of(r), 'red'), (ref_of(r), 'lime')],
                         left=None, right=existing_output(r.dataset, r.img), right_title='Mean-GDDiff output'))

    # sweep: worst case per perturbation
    for cond in ['shift040', 'occl_eyes', 'occl_lower', 'occl_patch']:
        c = per_img[per_img.condition == cond].sort_values('self_cos', ascending=False)
        if c.empty:
            continue
        w = c.iloc[0]
        base = per_img[(per_img.condition == 'base') & (per_img.img == w.img)].self_cos
        r = det[(det.dataset == SETS[w.set]) & (det.img == w.img)].iloc[0]
        inp, spec = CONDITIONS[cond]
        box = perturb_bbox(box_of(r), spec, w.img + '.jpg') if spec else box_of(r)
        rows.append(dict(input=os.path.join(args.inputs, inp, w.set, w.img + '.jpg'),
                         input_title=f'{DS_NAME[SETS[w.set]]}: {COND_LABEL[cond]}',
                         boxes=[(box, 'red'), (ref_of(r), 'lime')],
                         left=output_path('base', w.set, 'mean', w.img),
                         left_title=f'unperturbed (cos {base.iloc[0]:.2f})' if len(base) else 'unperturbed',
                         right=output_path(cond, w.set, 'mean', w.img),
                         right_title=f'perturbed (cos {w.self_cos:.2f})'))

    # two panels side by side: (a) real failures, input | output; (b) controlled, input | unperturbed | perturbed
    real = [r for r in rows if r['left'] is None]
    ctrl = [r for r in rows if r['left'] is not None]
    n_rows = max(len(real), len(ctrl))
    fig = plt.figure(figsize=(7.1, 1.55 * n_rows + 0.12))
    gs = fig.add_gridspec(n_rows, 6, width_ratios=[1, 1, 0.12, 1, 1, 1], wspace=0.06, hspace=0.28,
                          left=0.01, right=0.99, bottom=0.01, top=1 - 0.1 / (1.55 * n_rows + 0.12))
    for i, r in enumerate(real):
        show(fig.add_subplot(gs[i, 0]), r['input'], r['input_title'], r['boxes'])
        show(fig.add_subplot(gs[i, 1]), r['right'], 'Mean-GDDiff output')
    for i, r in enumerate(ctrl):
        show(fig.add_subplot(gs[i, 3]), r['input'], r['input_title'], r['boxes'])
        show(fig.add_subplot(gs[i, 4]), r['left'], r['left_title'])
        show(fig.add_subplot(gs[i, 5]), r['right'], r['right_title'])
    if len(real) < n_rows:  # free cell under panel (a): box legend
        ax = fig.add_subplot(gs[len(real):, 0:2])
        ax.axis('off')
        ax.legend(handles=[Rectangle((0, 0), 1, 1, fill=False, ec='red', lw=1.6, label='face box used by GDDiff'),
                           Rectangle((0, 0), 1, 1, fill=False, ec='lime', lw=1.6, label='MTCNN reference box')],
                  loc='center', frameon=False, fontsize=7.5)
    # panel descriptions are given in the caption (left: real failures, right: controlled perturbations)
    fig.savefig(args.out, dpi=300)
    fig.savefig(args.out.replace('.pdf', '.png'), dpi=150)
    print('saved', args.out, len(rows), 'rows')


if __name__ == '__main__':
    main()
