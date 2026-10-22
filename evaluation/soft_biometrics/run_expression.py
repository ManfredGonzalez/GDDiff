"""Run facial expression recognition (QCS or FMAE) on the original aligned
datasets and on every de-identified baseline, one CSV per (dataset, technique).

Same folders and naming as run_gaze.py. Both models are the authors' RAF-DB
checkpoints, so they predict the same 7 RAF-DB classes. Each CSV stores the
softmax probabilities per class and the predicted class.

  QCS  (AAAI 2025): QCS/checkpoints/RAF-DB_QCS/[12-14]-[23-47]-model_best.pth (92.50% on RAF-DB)
  FMAE (FG 2025, FMAE-IAT repo): FMAE-IAT/checkpoints/FMAE_RAFDB.pth (ViT-L). The IAT
       (identity adversarial) checkpoints are only released for action units, not expressions.

Usage (env: gaze):
    python gaze_estimation/run_expression.py --model qcs
    python gaze_estimation/run_expression.py --model fmae
"""
import argparse
import os
import pickle
import sys
import types
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gaze import MASTER, collect_jobs, list_images  # noqa: E402

RESULTS_DIR = os.path.join(MASTER, 'deid-toolkit/root_dir/results/expression')
IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# RAF-DB label order used by QCS (main_QCS_raf_db.py confusion-matrix labels)
QCS_CLASSES = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']
# FMAE uses torchvision ImageFolder on RAF-DB folders 1..7 (same RAF-DB order)
FMAE_CLASSES = QCS_CLASSES
# The aligned images include hair/background while RAF-DB faces are tight crops;
# keep the central fraction of the image. Chosen by 7-class accuracy on the RaFD
# originals (contempt excluded): QCS 68.7% at 0.65 (51% uncropped),
# FMAE 75.5% at 0.7 (74% uncropped).
CENTER_CROP = {'qcs': 0.65, 'fmae': 0.7}


class CenterFraction:
    def __init__(self, frac):
        self.frac = frac

    def __call__(self, img):
        w, h = img.size
        kw, kh = int(w * self.frac), int(h * self.frac)
        left, top = (w - kw) // 2, (h - kh) // 2
        return img.crop((left, top, left + kw, top + kh))


class FolderData(Dataset):
    def __init__(self, folder, files, transform):
        self.folder, self.files, self.transform = folder, files, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.transform(Image.open(os.path.join(self.folder, self.files[i])).convert('RGB'))


# --------------------------------- QCS --------------------------------- #
def qcs_transform(frac):
    # main_QCS_raf_db.py 'test' transform
    return transforms.Compose([CenterFraction(frac), transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize(**IMAGENET)])


def _stub_pickle_module():
    # the checkpoint also pickles training-log objects (RecorderMeter_matrix); only weights are needed
    class _Stub:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            pass

    class Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            try:
                return super().find_class(mod, name)
            except Exception:
                return _Stub

    pm = types.ModuleType('stub_pickle')
    pm.Unpickler, pm.load = Unpickler, pickle.load
    return pm


def load_qcs(device):
    sys.path.insert(0, os.path.join(MASTER, 'QCS'))
    from models import QCS_7cls_raf_db as qcs_mod
    real_load = torch.load
    try:  # constructor torch.loads a hard-coded Windows pretrain path; full weights are loaded below
        qcs_mod.torch.load = lambda *a, **k: {}
        net = qcs_mod.pyramid_trans_expr(img_size=224, num_classes=7)
    finally:
        qcs_mod.torch.load = real_load
    ckpt = torch.load(os.path.join(MASTER, 'QCS/checkpoints/RAF-DB_QCS/[12-14]-[23-47]-model_best.pth'),
                      map_location='cpu', pickle_module=_stub_pickle_module(), weights_only=False)
    state = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in ckpt['state_dict'].items()}
    net.load_state_dict(state, strict=True)
    return net.to(device).eval()


def predict_qcs(net, batch):
    return net(batch, None, None, None)  # inference keeps only the base branch -> logits [B, 7]


# --------------------------------- FMAE -------------------------------- #
def fmae_transform(frac):
    # util/datasets.py eval transform of the RAFDB_finetune branch: Resize(256, bicubic) + CenterCrop(224)
    return transforms.Compose([CenterFraction(frac),
                               transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                               transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(**IMAGENET)])


def load_fmae(device):
    from timm.models.vision_transformer import VisionTransformer
    # vit_large_patch16 with global average pooling (no cls token) + fc_norm, as in models_vit.py
    net = VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=7,
                            global_pool='avg', fc_norm=True)
    state = torch.load(os.path.join(MASTER, 'FMAE-IAT/checkpoints/FMAE_RAFDB.pth'), map_location='cpu')['model']
    state = {k: v for k, v in state.items() if not k.startswith('head.')}  # unused timm head; AU_head is the FER head
    state['head.weight'], state['head.bias'] = state.pop('AU_head.weight'), state.pop('AU_head.bias')
    net.load_state_dict(state, strict=True)
    return net.to(device).eval()


def predict_fmae(net, batch):
    return net(batch)


MODELS = {
    'qcs': (load_qcs, qcs_transform, predict_qcs, QCS_CLASSES),
    'fmae': (load_fmae, fmae_transform, predict_fmae, FMAE_CLASSES),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--only', nargs='*', help='restrict to these dataset names')
    args = parser.parse_args()

    load, make_tf, predict, classes = MODELS[args.model]
    tf = make_tf(CENTER_CROP[args.model])
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
        probs = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'{args.model} | {ds} | {tech}'):
                probs.append(torch.softmax(predict(net, batch.to(args.device, non_blocking=True)), 1).cpu().numpy())
        probs = np.concatenate(probs)
        df = pd.DataFrame(probs, columns=[f'p_{c}' for c in classes])
        df.insert(0, 'img', [os.path.splitext(f)[0] for f in files])
        df['pred'] = [classes[i] for i in probs.argmax(1)]
        df.to_csv(out_csv, index=False)
        print(f'[saved] {out_csv} ({len(df)} images)')


if __name__ == '__main__':
    main()
