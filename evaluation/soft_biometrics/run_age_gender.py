"""Run age / gender estimation on the original aligned datasets and on every
de-identified baseline, one CSV per (dataset, technique). Same folders and
naming as run_gaze.py.

  MiVOLO    (age + gender) face-only volo_d1, IMDB-clean, MiVOLO/checkpoints/model_imdb_age_gender_4.22.pth.tar
            -> needs env `mivolo` (MiVOLO pins timm 0.8.13.dev0)
  HyCnXtViT (age)          ConvNeXt+ViT hybrid, MORPH II (MAE 2.26), HyCnXtViT-for-Age-Estimation/weights/
  FairFace  (gender)       ResNet-34 res34_fair_align_multi_7_20190809.pt. The authors' Drive folder is gone
            (404); this copy is from the HuggingFace user anning01/fairface (unverified mirror),
            loaded with weights_only=True.

Each CSV: img, age (years) and/or p_male, p_female, gender.

Usage:
    conda activate mivolo; python gaze_estimation/run_age_gender.py --model mivolo
    conda activate gaze;   python gaze_estimation/run_age_gender.py --model hycnxtvit
    conda activate gaze;   python gaze_estimation/run_age_gender.py --model fairface
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gaze import MASTER, collect_jobs, list_images  # noqa: E402

RESULTS_DIR = os.path.join(MASTER, 'deid-toolkit/root_dir/results/age_gender')
IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Central fraction of the aligned image kept before each model's own preprocessing
# (the aligned crops include hair/background; the models were trained on tighter faces).
# Chosen on the originals: MiVOLO and FairFace by gender accuracy on RaFD (MiVOLO 99.4% at 0.9,
# FairFace 93.5% at 0.6 vs 86.0% uncropped); HyCnXtViT (no age labels available) by agreement
# with MiVOLO's ages (corr 0.82 XM2VTS / 0.69 LFW at 0.7).
CENTER_CROP = {'mivolo': 0.9, 'hycnxtvit': 0.7, 'fairface': 0.6}


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


# -------------------------------- MiVOLO ------------------------------- #
def mivolo_transform(frac):
    # mivolo/data/misc.py prepare_classification_images: letterbox to 224 (square input -> plain
    # resize), RGB, /255, ImageNet mean/std
    return transforms.Compose([CenterFraction(frac), transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize(**IMAGENET)])


MIVOLO_META = {}


def load_mivolo(device):
    root = os.path.join(MASTER, 'MiVOLO')
    sys.path.insert(0, root)
    from mivolo.model.create_timm_model import create_model
    ckpt_path = os.path.join(root, 'checkpoints/model_imdb_age_gender_4.22.pth.tar')
    meta = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    MIVOLO_META.update(min_age=meta['min_age'], max_age=meta['max_age'], avg_age=meta['avg_age'])
    net = create_model('mivolo_d1_224', num_classes=3, in_chans=3, pretrained=False, checkpoint_path=ckpt_path)
    return net.to(device).eval()


def predict_mivolo(net, batch):
    out = net(batch)
    p = out[:, :2].softmax(-1).cpu().numpy()  # index 0 = male, 1 = female (mi_volo.py fill_in_results)
    m = MIVOLO_META
    age = (out[:, 2] * (m['max_age'] - m['min_age']) + m['avg_age']).cpu().numpy()
    return {'age': age, 'p_male': p[:, 0], 'p_female': p[:, 1]}


# ------------------------------- HyCnXtViT ----------------------------- #
def hycnxtvit_transform(frac):
    # src/cnxtvit/data.py eval_transform: RGB, Resize((224, 224)), ImageNet mean/std
    return transforms.Compose([CenterFraction(frac), transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize(**IMAGENET)])


class HyCnXtViT(nn.Module):
    """ConvNeXT_Transformer from src/cnxtvit/model.py, rebuilt without pytorch_lightning.
    ConvNeXt (depths 3,3,9,3; dims 96..768) -> 7x7x768 map reshaped to 196 tokens of 192 ->
    timm vit_tiny_patch16_224 (patch_embed removed) -> LayerNorm, FC 128, FC 1 (age in years)."""

    def __init__(self):
        super().__init__()
        import importlib.util
        import timm
        # load convnext.py by path: the cnxtvit package __init__ imports pytorch_lightning
        path = os.path.join(MASTER, 'HyCnXtViT-for-Age-Estimation/src/cnxtvit/convnext.py')
        spec = importlib.util.spec_from_file_location('cnxtvit_convnext', path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # its timm @register_model decorators look the module up here
        spec.loader.exec_module(mod)
        ConvNeXt = mod.ConvNeXt
        cnx = ConvNeXt(in_chans=3, num_classes=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                       drop_path_rate=0.0, layer_scale_init_value=1e-6, head_init_scale=1.0)
        self.ConvNeXt = nn.Module()
        self.ConvNeXt.downsample_layers, self.ConvNeXt.stages, self.ConvNeXt.norm = \
            cnx.downsample_layers, cnx.stages, cnx.norm
        self.ConvNeXt.fc = nn.Sequential(nn.Linear(768, 256), nn.Linear(256, 1))  # unused at inference
        vit = timm.create_model('vit_tiny_patch16_224', pretrained=False)
        vit.head = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 128), nn.Linear(128, 1))
        vit.patch_embed = nn.Identity()
        self.Transformer = nn.Module()
        self.Transformer.vit = vit

    def forward(self, x):
        for i in range(4):
            x = self.ConvNeXt.downsample_layers[i](x)
            x = self.ConvNeXt.stages[i](x)
        x = torch.reshape(x, (x.shape[0], 192, 196)).permute(0, 2, 1)
        return self.Transformer.vit(x)


def load_hycnxtvit(device):
    net = HyCnXtViT()
    ckpt = torch.load(os.path.join(MASTER, 'HyCnXtViT-for-Age-Estimation/weights/'
                                   'MORPH2_CNNXT_VIT_preT_MORPH2_FC2_batch128_lossfunc_Adaptive_epoch_499_mae_2.26.ckpt'),
                      map_location='cpu', weights_only=False)
    state = {k[len('model.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
    net.load_state_dict(state, strict=True)
    return net.to(device).eval()


def predict_hycnxtvit(net, batch):
    return {'age': net(batch)[:, 0].cpu().numpy()}


# ------------------------------- FairFace ------------------------------ #
def fairface_transform(frac):
    # predict.py: Resize((224, 224)), ImageNet mean/std on the dlib face chip (RGB)
    return transforms.Compose([CenterFraction(frac), transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize(**IMAGENET)])


def load_fairface(device):
    net = torchvision.models.resnet34()
    net.fc = nn.Linear(net.fc.in_features, 18)  # 7 race + 2 gender + 9 age bins
    state = torch.load(os.path.join(MASTER, 'FairFace/fair_face_models/hf_mirror_anning01/'
                                    'res34_fair_align_multi_7_20190809.pt'), map_location='cpu', weights_only=True)
    net.load_state_dict(state, strict=True)
    return net.to(device).eval()


def predict_fairface(net, batch):
    p = net(batch)[:, 7:9].softmax(-1).cpu().numpy()  # predict.py: gender 0 = Male, 1 = Female
    return {'p_male': p[:, 0], 'p_female': p[:, 1]}


MODELS = {
    'mivolo': (load_mivolo, mivolo_transform, predict_mivolo),
    'hycnxtvit': (load_hycnxtvit, hycnxtvit_transform, predict_hycnxtvit),
    'fairface': (load_fairface, fairface_transform, predict_fairface),
}


def run_folder(net, predict, tf, folder, files, batch_size=64, num_workers=6, device='cuda'):
    loader = DataLoader(FolderData(folder, files, tf), batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=True)
    out = {}
    with torch.no_grad():
        for batch in loader:
            for k, v in predict(net, batch.to(device, non_blocking=True)).items():
                out.setdefault(k, []).append(v)
    df = pd.DataFrame({k: np.concatenate(v) for k, v in out.items()})
    if 'p_male' in df:
        df['gender'] = np.where(df['p_male'] >= df['p_female'], 'male', 'female')
    df.insert(0, 'img', [os.path.splitext(f)[0] for f in files])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--only', nargs='*', help='restrict to these dataset names')
    args = parser.parse_args()

    load, make_tf, predict = MODELS[args.model]
    tf = make_tf(CENTER_CROP[args.model])
    out_dir = os.path.join(RESULTS_DIR, args.model)
    os.makedirs(out_dir, exist_ok=True)
    net = load(args.device)

    jobs = [j for j in collect_jobs() if not args.only or j[0] in args.only]
    print(f'{len(jobs)} folders to process with {args.model}')
    for ds, tech, folder in tqdm(jobs, desc=args.model):
        out_csv = os.path.join(out_dir, f'{args.model}_{ds}_{tech}.csv')
        if os.path.exists(out_csv):
            continue
        df = run_folder(net, predict, tf, folder, list_images(folder), args.batch_size, args.num_workers, args.device)
        df.to_csv(out_csv, index=False)
        print(f'[saved] {out_csv} ({len(df)} images)')


if __name__ == '__main__':
    main()
