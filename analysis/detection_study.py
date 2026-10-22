"""How well does the face detector used by GDDiff work on the evaluation datasets?

Replicates the pipeline exactly (guided_diffusion/diffusion.py): the aligned image
is resized to 256x256 (torchvision Resize, as in datasets/__init__.py), converted
to uint8 BGR, and passed to OpenCV's Haar cascade (Viola-Jones) with default
detectMultiScale parameters; the first detection faces[0] is used, and if there is
none, the centred fallback box covering 65% of the image (get_fallback_bbox).

Each box actually used is compared with a reference box from MTCNN (bundled with
AdaFace, more robust to pose and occlusion than a frontal cascade):
    centre_offset = ||c_used - c_ref|| / sqrt(w_ref * h_ref)
    scale_ratio   = sqrt(area_used / area_ref)
    iou           = IoU(box_used, box_ref)

Usage (env: gddiff):
    python analysis/detection_study.py --out analysis/results_detection/detection_study.csv
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIGNED = os.path.expanduser('~/Desktop/master_research/deid-toolkit/root_dir/datasets/aligned')
DATASETS = ['CALFW', 'LFW_unified_orig', 'rafd', 'xm2vts_aligned']

RESIZE = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])


def pipeline_input(path):
    """uint8 BGR 256x256 image exactly as diffusion.py feeds it to the detector."""
    x = RESIZE(Image.open(path).convert('RGB'))
    x = (x * 2 - 1 + 1) / 2  # data_transform followed by inverse_data_transform
    rgb = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def fallback_box(size=256, scale=0.65):
    s = int(size * scale)
    return (size // 2 - s // 2, size // 2 - s // 2, s, s)


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=None, help='images per dataset (debug)')
    args = ap.parse_args()

    cascade = cv2.CascadeClassifier(os.path.join(REPO, 'haarcascade_frontalface_default.xml'))
    # Original MTCNN procedure (mtcnn_pytorch/src/detector.py), not AdaFace's modified wrapper in
    # face_alignment/mtcnn.py, which falls back to the Haar cascade itself. detect_faces() rebuilds its
    # three networks on every call, so its constructors are replaced by factories returning nets built once.
    fa_dir = os.path.join(REPO, 'AdaFace/face_alignment')
    sys.path.insert(0, fa_dir)
    from mtcnn_pytorch.src import detector as mtcnn_detector
    cwd = os.getcwd()
    os.chdir(fa_dir)  # get_nets loads 'mtcnn_pytorch/src/weights/*.npy' relative to this folder
    nets = {name: getattr(mtcnn_detector, name)().eval() for name in ('PNet', 'RNet', 'ONet')}
    os.chdir(cwd)
    for name, net in nets.items():
        setattr(mtcnn_detector, name, lambda net=net: net)

    rows = []
    for ds in DATASETS:
        files = sorted(f for f in os.listdir(os.path.join(ALIGNED, ds)) if f.lower().endswith(('.jpg', '.png')))
        for f in tqdm(files[:args.limit], desc=ds):
            bgr = pipeline_input(os.path.join(ALIGNED, ds, f))
            faces = cascade.detectMultiScale(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            n = 0 if faces is None else len(faces)
            if n:
                used = tuple(int(v) for v in faces[0])
                largest = max(faces, key=lambda b: b[2] * b[3])
                first_is_largest = bool((faces[0] == largest).all())
            else:
                used, first_is_largest = fallback_box(), None

            try:
                boxes, _ = mtcnn_detector.detect_faces(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            except ValueError:  # np.vstack of an empty list: no stage-1 candidates at all
                boxes = []
            row = dict(dataset=ds, img=os.path.splitext(f)[0], n_haar=n, fallback=n == 0,
                       first_is_largest=first_is_largest, used_x=used[0], used_y=used[1], used_w=used[2],
                       used_h=used[3], mtcnn_found=len(boxes) > 0)
            if len(boxes):
                b = boxes[np.argmax(boxes[:, 4])]
                ref = (b[0], b[1], b[2] - b[0], b[3] - b[1])
                cu = np.array([used[0] + used[2] / 2, used[1] + used[3] / 2])
                cr = np.array([ref[0] + ref[2] / 2, ref[1] + ref[3] / 2])
                row.update(ref_x=ref[0], ref_y=ref[1], ref_w=ref[2], ref_h=ref[3], mtcnn_score=b[4],
                           centre_offset=np.linalg.norm(cu - cr) / np.sqrt(ref[2] * ref[3]),
                           scale_ratio=np.sqrt(used[2] * used[3] / (ref[2] * ref[3])), iou=iou(used, ref))
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option('display.width', 200)
    g = df.groupby('dataset')
    summary = pd.DataFrame({
        'n': g.size(),
        'haar_detected_%': g.fallback.apply(lambda s: (~s).mean() * 100),
        'multiple_detections_%': g.n_haar.apply(lambda s: (s > 1).mean() * 100),
        'first_not_largest_%': g.first_is_largest.apply(lambda s: (s == False).mean() * 100),  # noqa: E712
        'mtcnn_found_%': g.mtcnn_found.apply(lambda s: s.mean() * 100),
    })
    det = df[~df.fallback & df.mtcnn_found]
    for q in (50, 90, 99):
        summary[f'offset_p{q}'] = det.groupby('dataset').centre_offset.quantile(q / 100)
    summary['scale_median'] = det.groupby('dataset').scale_ratio.median()
    summary['iou_median'] = det.groupby('dataset').iou.median()
    summary['iou<0.5_%'] = det.groupby('dataset').iou.apply(lambda s: (s < 0.5).mean() * 100)
    print(summary.round(3).to_string())
    print('\nall datasets, Haar-detected boxes: centre-offset percentiles (fraction of face size):',
          {q: round(det.centre_offset.quantile(q / 100), 3) for q in (50, 90, 99)},
          '| scale-ratio p1/p50/p99:',
          [round(det.scale_ratio.quantile(q), 3) for q in (0.01, 0.5, 0.99)])
    fb = df[df.fallback & df.mtcnn_found]
    if len(fb):
        print('fallback boxes vs MTCNN: centre offset p50/p90 =',
              round(fb.centre_offset.median(), 3), round(fb.centre_offset.quantile(.9), 3),
              '| IoU median', round(fb.iou.median(), 3))


if __name__ == '__main__':
    main()
