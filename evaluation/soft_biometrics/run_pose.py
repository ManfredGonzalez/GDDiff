"""Run head pose estimation (SemiUHPE or 6DRepNet360) on the original aligned
datasets and on every de-identified baseline, one CSV per (dataset, technique).

Same folders and naming as run_gaze.py. Images are aligned face crops, so the
whole image is used as the head crop (no detector). Each CSV stores the
predicted 3x3 rotation matrix (r00..r22) plus yaw / pitch / roll in degrees,
computed with each repo's own Euler convention.

Usage (env: gaze):
    python gaze_estimation/run_pose.py --model semiuhpe
    python gaze_estimation/run_pose.py --model 6drepnet360
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gaze import MASTER, collect_jobs, list_images  # noqa: E402

RESULTS_DIR = os.path.join(MASTER, 'deid-toolkit/root_dir/results/pose')
IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class FolderData(Dataset):
    def __init__(self, folder, files, transform):
        self.folder, self.files, self.transform = folder, files, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.transform(Image.open(os.path.join(self.folder, self.files[i])).convert('RGB'))


# ------------------------------ SemiUHPE ------------------------------- #
# image.py: RGB head crop resized to 224x224, ImageNet normalisation
SEMIUHPE_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                  transforms.Normalize(**IMAGENET)])


def load_semiuhpe(device):
    # same head as get_EfficientNet_V2 in SemiUHPE/src/networks.py (num_classes=9, matrix-Fisher A)
    net = torchvision.models.efficientnet_v2_s()
    net.classifier = nn.Sequential(
        nn.Dropout(0.2), nn.Linear(1280, 512), nn.BatchNorm1d(512), nn.ReLU6(inplace=True),
        nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU6(inplace=True), nn.Linear(128, 9))
    ckpt = torch.load(os.path.join(MASTER, 'SemiUHPE/checkpoints/DAD-WildHead-EffNetV2-S-best.pth'),
                      map_location='cpu')
    net.load_state_dict(ckpt['model_state_dict'])  # agent.net, as used by image.py
    return net.to(device).eval()


def predict_semiuhpe(net, batch):
    # batch_torch_A_to_R (src/fisher/fisher_utils.py): nearest rotation to A via SVD
    A = net(batch).reshape(-1, 3, 3).double().cpu()
    U, _, V = torch.svd(A)
    s3 = torch.det(U @ V.transpose(1, 2))
    U = torch.cat((U[:, :, :2], U[:, :, 2:] * s3[:, None, None]), -1)
    R = (U @ V.transpose(1, 2)).numpy()
    # Euler angles exactly as in image.py
    ang = Rotation.from_matrix(np.transpose(R, (0, 2, 1))).as_euler('xyz', degrees=True)
    wrap = lambda a: (a + 180.0) % 360.0 - 180.0
    yaw, pitch, roll = wrap(ang[:, 1]), wrap(ang[:, 0] - 180.0), wrap(ang[:, 2])
    return R, yaw, pitch, roll


# ----------------------------- 6DRepNet360 ----------------------------- #
# test.py: Resize(256) + CenterCrop(224), ImageNet normalisation
SIXD_TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                              transforms.Normalize(**IMAGENET)])


def load_6drepnet360(device):
    sys.path.insert(0, os.path.join(MASTER, '6DRepNet360/sixdrepnet360'))
    from test import SixDRepNet360
    net = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6)
    state = torch.load(os.path.join(MASTER, '6DRepNet360/checkpoints/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth'),
                       map_location='cpu')
    net.load_state_dict(state.get('model_state_dict', state))
    return net.to(device).eval()


def predict_6drepnet360(net, batch):
    import utils as sixd_utils  # 6DRepNet360/sixdrepnet360/utils.py (on sys.path after load)
    R = net(batch)
    e = sixd_utils.compute_euler_angles_from_rotation_matrices(R) * 180 / math.pi
    pitch, yaw, roll = (e[:, i].cpu().numpy() for i in range(3))
    return R.cpu().double().numpy(), yaw, pitch, roll


MODELS = {
    'semiuhpe': (load_semiuhpe, SEMIUHPE_TF, predict_semiuhpe),
    '6drepnet360': (load_6drepnet360, SIXD_TF, predict_6drepnet360),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--only', nargs='*', help='restrict to these dataset names')
    args = parser.parse_args()

    load, tf, predict = MODELS[args.model]
    out_dir = os.path.join(RESULTS_DIR, args.model)
    os.makedirs(out_dir, exist_ok=True)
    net = load(args.device)

    jobs = [j for j in collect_jobs() if not args.only or j[0] in args.only]
    print(f'{len(jobs)} folders to process with {args.model}')
    for ds, tech, folder in jobs:
        out_csv = os.path.join(out_dir, f'{args.model}_{ds}_{tech}.csv')
        if os.path.exists(out_csv):
            print(f'[done] {out_csv}')
            continue
        files = list_images(folder)
        loader = DataLoader(FolderData(folder, files, tf), batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False, pin_memory=True)
        Rs, yaws, pitches, rolls = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'{args.model} | {ds} | {tech}'):
                R, y, p, r = predict(net, batch.to(args.device, non_blocking=True))
                Rs.append(R.reshape(-1, 9)); yaws.append(y); pitches.append(p); rolls.append(r)
        Rs = np.concatenate(Rs)
        df = pd.DataFrame(Rs, columns=[f'r{i}{j}' for i in range(3) for j in range(3)])
        df.insert(0, 'img', [os.path.splitext(f)[0] for f in files])
        df['yaw_deg'], df['pitch_deg'], df['roll_deg'] = np.concatenate(yaws), np.concatenate(pitches), np.concatenate(rolls)
        df.to_csv(out_csv, index=False)
        print(f'[saved] {out_csv} ({len(df)} images)')


if __name__ == '__main__':
    main()
