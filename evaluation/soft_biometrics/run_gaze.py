"""Run gaze estimation (GazeTR or gaze3d) on the original aligned datasets and on
every de-identified baseline, saving one CSV per (dataset, technique).

Images are already aligned face crops, so the whole image is used as the face /
head crop for both models (no detector). Each CSV has one row per image with
the predicted unit gaze vector (gaze_x, gaze_y, gaze_z) in the camera frame,
plus pitch/yaw in degrees.

Usage (env: gaze):
    python gaze_estimation/run_gaze.py --model gazetr
    python gaze_estimation/run_gaze.py --model gaze3d
"""
import argparse
import os
import sys
from functools import partial

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MASTER = os.path.expanduser('~/Desktop/master_research')
DATASETS_ROOT = os.path.join(MASTER, 'deid-toolkit/root_dir/datasets')
RESULTS_DIR = os.path.join(MASTER, 'deid-toolkit/root_dir/results/gaze')

DATASETS = ['CALFW', 'LFW_unified_orig', 'rafd', 'xm2vts_aligned']
TECHNIQUES = ['AMT-GAN', 'cleanir', 'cpp-deid', 'deepprivacy2',
              'face_anon_diff', 'fadm', 'GauDeiDiff', 'ldfa']
IMG_EXTS = ('.jpg', '.jpeg', '.png')


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def collect_jobs():
    """(dataset, technique_label, folder) for originals and every baseline.
    Techniques with subfolders (e.g. cpp-deid/<ds>/0.5) are named like the
    toolkit does: '<subfolder>_<technique>' (e.g. '0.5_cpp-deid')."""
    jobs = []
    for ds in DATASETS:
        jobs.append((ds, 'Original', os.path.join(DATASETS_ROOT, 'aligned', ds)))
        for tech in TECHNIQUES:
            path = os.path.join(DATASETS_ROOT, tech, ds)
            if not os.path.isdir(path):
                print(f'[skip] missing folder {path}')
                continue
            subdirs = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
            if subdirs:
                for sub in subdirs:
                    # normalise case (GauDeiDiff/CALFW has 'Lpl' vs 'lpl' elsewhere)
                    jobs.append((ds, f'{sub.lower()}_{tech}', os.path.join(path, sub)))
            else:
                jobs.append((ds, tech, path))
    return jobs


def vec_to_pitch_yaw(v):
    """Unit gaze vector (camera frame, z towards scene) -> pitch, yaw in degrees.
    Uses the same convention as GazeTR's gtools.gazeto3d (inverse of it)."""
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    pitch = np.degrees(np.arcsin(-v[:, 1]))
    yaw = np.degrees(np.arctan2(-v[:, 0], -v[:, 2]))
    return pitch, yaw


# ------------------------------- GazeTR -------------------------------- #
# ETH-XGaze faces are tight crops, the aligned images include hair/background.
# A centred crop of 60% of the side was the best match on RaFD (frontal gaze,
# cameras at 45/90/135 deg -> predicted yaw about -39 / 0 / +39 deg).
GAZETR_CROP = 0.6


class GazeTRData(Dataset):
    # GazeTR (ETH-XGaze) expects BGR, 224x224, scaled to [0, 1], no normalisation
    def __init__(self, folder, files):
        self.folder, self.files = folder, files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(os.path.join(self.folder, self.files[i]))
        h, w = img.shape[:2]
        kh, kw = int(h * GAZETR_CROP), int(w * GAZETR_CROP)
        y0, x0 = (h - kh) // 2, (w - kw) // 2
        img = img[y0:y0 + kh, x0:x0 + kw]
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def load_gazetr(device):
    sys.path.insert(0, os.path.join(MASTER, 'GazeTR'))
    from model import Model
    net = Model()
    state = torch.load(os.path.join(MASTER, 'GazeTR/checkpoints/GazeTR-H-ETH.pt'), map_location='cpu')
    net.load_state_dict(state)
    return net.to(device).eval()


def predict_gazetr(net, batch):
    # output is (yaw, pitch) in radians, as consumed by gtools.gazeto3d
    yp = net({'face': batch}).cpu().numpy()
    yaw, pitch = yp[:, 0], yp[:, 1]
    vec = np.stack([-np.cos(pitch) * np.sin(yaw), -np.sin(pitch), -np.cos(pitch) * np.cos(yaw)], 1)
    return vec


# ------------------------------- gaze3d -------------------------------- #
class Gaze3DData(Dataset):
    # whole aligned image used as the head bbox, then the repo's own transform
    def __init__(self, folder, files):
        sys.path.insert(0, os.path.join(MASTER, 'gaze3d'))
        from torchvision.transforms import Compose
        from src.data.components.transforms import (BboxReshape, Concatenate, Crop,
                                                    Normalize, ToImage, ToTensor)
        # same as TRANSFORM in gaze3d/utils_demo.py (imported directly to avoid its video deps)
        self.transform = Compose([BboxReshape(square=True, ratio=-0.1), ToImage(), Crop(output_size=224),
                                  Concatenate(), ToTensor(),
                                  Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.folder, self.files = folder, files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = np.array(Image.open(os.path.join(self.folder, self.files[i])).convert('RGB'))
        h, w = img.shape[:2]
        sample = {'images': [img], 'head_bbox': [[0, 0, w, h]]}
        return self.transform(sample)['images']


def load_gaze3d(device):
    sys.path.insert(0, os.path.join(MASTER, 'gaze3d'))
    from src.models.gat_model import GaT, HeadDict, MLPHead, Swin3D
    net = GaT(encoder=Swin3D(pretrained=False),
              head_dict=HeadDict(names=['gaze'],
                                 modules=[partial(MLPHead, hidden_dim=256, num_layers=1, out_features=3)]))
    ckpt = torch.load(os.path.join(MASTER, 'gaze3d/checkpoints/gat_stwsge_gaze360_gf.ckpt'), map_location='cpu')
    net.load_state_dict(ckpt['state_dict'], strict=True)
    return net.to(device).eval()


def predict_gaze3d(net, batch):
    gaze = torch.nn.functional.normalize(net(batch)['gaze'], p=2, dim=2, eps=1e-8)
    return gaze[:, 0, :].cpu().numpy()  # single frame


MODELS = {
    'gazetr': (load_gazetr, GazeTRData, predict_gazetr),
    'gaze3d': (load_gaze3d, Gaze3DData, predict_gaze3d),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--only', nargs='*', help='restrict to these dataset names')
    parser.add_argument('--limit', type=int, help='only first N images per folder (debug)')
    args = parser.parse_args()

    load, DataCls, predict = MODELS[args.model]
    out_dir = os.path.join(RESULTS_DIR, args.model)
    os.makedirs(out_dir, exist_ok=True)
    net = load(args.device)

    jobs = [j for j in collect_jobs() if not args.only or j[0] in args.only]
    print(f'{len(jobs)} folders to process with {args.model}')
    for ds, tech, folder in jobs:
        out_csv = os.path.join(out_dir, f'{args.model}_{ds}_{tech}.csv')
        if os.path.exists(out_csv) and not args.limit:
            print(f'[done] {out_csv}')
            continue
        files = list_images(folder)[:args.limit]
        loader = DataLoader(DataCls(folder, files), batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False, pin_memory=True)
        vecs = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'{args.model} | {ds} | {tech}'):
                vecs.append(predict(net, batch.to(args.device, non_blocking=True)))
        vecs = np.concatenate(vecs)
        pitch, yaw = vec_to_pitch_yaw(vecs)
        df = pd.DataFrame({'img': [os.path.splitext(f)[0] for f in files],
                           'gaze_x': vecs[:, 0], 'gaze_y': vecs[:, 1], 'gaze_z': vecs[:, 2],
                           'pitch_deg': pitch, 'yaw_deg': yaw})
        if args.limit:
            out_csv = out_csv.replace('.csv', f'_limit{args.limit}.csv')
        df.to_csv(out_csv, index=False)
        print(f'[saved] {out_csv} ({len(df)} images)')


if __name__ == '__main__':
    main()
