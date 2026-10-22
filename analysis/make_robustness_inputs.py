"""Inputs for the face-detection robustness study (reviewer R2.3).

1. Subset: 150 RaFD + 150 LFW originals, drawn at random (seed 0) among the images
   that appear as the de-identified side (img_b) of the toolkit's evaluation pairs,
   so verification AUC can be computed on their existing pairs.
2. Occluded copies of each, placed with the five MTCNN landmarks of the original:
     eyes   black bar over both eyes (sunglasses-like)
     lower  light-blue block from below the nose to the chin (mask-like)
     patch  uniform-noise square covering 30% of the face box at a random position
   The full GDDiff pipeline, including face detection, is then run on these.

Writes ImageFolder-style folders <out>/{clean,occl_eyes,occl_lower,occl_patch}/{rafd,lfw}/
and <out>/subset.csv.

Usage (env: gddiff):
    python analysis/make_robustness_inputs.py <out_dir>
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLKIT = os.path.expanduser('~/Desktop/master_research/deid-toolkit/root_dir')
SETS = {'rafd': 'rafd', 'lfw': 'LFW_unified_orig'}
N_PER_SET = 150


def load_mtcnn():
    fa_dir = os.path.join(REPO, 'AdaFace/face_alignment')
    sys.path.insert(0, fa_dir)
    from mtcnn_pytorch.src import detector
    cwd = os.getcwd()
    os.chdir(fa_dir)
    nets = {n: getattr(detector, n)().eval() for n in ('PNet', 'RNet', 'ONet')}
    os.chdir(cwd)
    for n, net in nets.items():
        setattr(detector, n, lambda net=net: net)
    return detector.detect_faces


def occlude(img, box, lm, kind, rng):
    a = np.array(img).copy()
    H, W = a.shape[:2]
    x0, y0, x1, y1 = box
    eyes = np.array([[lm[0], lm[5]], [lm[1], lm[6]]])
    nose, mouth = np.array([lm[2], lm[7]]), np.array([(lm[3] + lm[4]) / 2, (lm[8] + lm[9]) / 2])
    iod = np.linalg.norm(eyes[1] - eyes[0])
    if kind == 'eyes':
        c = eyes.mean(0)
        hw, hh = max(1.0 * iod, 0.4 * (x1 - x0)), 0.28 * max(iod, 0.4 * (x1 - x0))
        r = (c[0] - hw, c[1] - hh, c[0] + hw, c[1] + hh)
        color = (15, 15, 15)
    elif kind == 'lower':
        top = nose[1] + 0.25 * (mouth[1] - nose[1])
        r = (x0, top, x1, y1)
        color = (170, 205, 225)
    else:  # patch
        side = np.sqrt(0.3 * (x1 - x0) * (y1 - y0))
        px = rng.uniform(x0, max(x0, x1 - side))
        py = rng.uniform(y0, max(y0, y1 - side))
        r, color = (px, py, px + side, py + side), None
    rx0, ry0 = int(max(0, r[0])), int(max(0, r[1]))
    rx1, ry1 = int(min(W, r[2])), int(min(H, r[3]))
    if color is None:
        a[ry0:ry1, rx0:rx1] = rng.integers(0, 256, (ry1 - ry0, rx1 - rx0, 3), dtype=np.uint8)
    else:
        a[ry0:ry1, rx0:rx1] = color
    return Image.fromarray(a)


def main():
    out = sys.argv[1]
    rng = np.random.default_rng(0)
    detect = load_mtcnn()
    rows = []
    for sub, ds in SETS.items():
        pairs = pd.read_csv(os.path.join(TOOLKIT, 'results', f'adaface_optimized_{ds}_0.5_mean_GauDeiDiff.csv'))
        cands = sorted({os.path.splitext(f)[0] for f in pairs.img_b})
        chosen = rng.choice(cands, size=N_PER_SET, replace=False)
        for name in sorted(chosen):
            img = Image.open(os.path.join(TOOLKIT, 'datasets/aligned', ds, name + '.jpg')).convert('RGB')
            small = img.resize((256, 256), Image.BILINEAR)
            try:
                boxes, lms = detect(small)
            except ValueError:
                boxes = []
            if len(boxes) == 0:
                print('no MTCNN face, skipped', name)
                continue
            k = int(np.argmax(boxes[:, 4]))
            s = img.size[0] / 256
            box, lm = boxes[k, :4] * s, lms[k] * s
            for kind in ('clean', 'eyes', 'lower', 'patch'):
                folder = os.path.join(out, 'clean' if kind == 'clean' else f'occl_{kind}', sub)
                os.makedirs(folder, exist_ok=True)
                im = img if kind == 'clean' else occlude(img, box, lm, kind, rng)
                im.save(os.path.join(folder, name + '.jpg'), quality=95)
            rows.append(dict(set=sub, dataset=ds, img=name))
    pd.DataFrame(rows).to_csv(os.path.join(out, 'subset.csv'), index=False)
    print(pd.DataFrame(rows).groupby('set').size())


if __name__ == '__main__':
    main()
