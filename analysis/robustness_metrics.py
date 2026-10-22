"""Metrics for the face-detection robustness study (reviewer R2.3).

Conditions are the robust_<name> runs written by analysis/results_detection/run_robustness.sh
(box jitter on clean inputs; occluded inputs with the unmodified pipeline). For each
condition and output branch (mean = Mean-GDDiff, gauss, Lapl) at lambda = 0.5:

  --part identity   (env: adaface; run from the deid-toolkit root so AdaFace finds its weights)
      AdaFace IR-50 exactly as the toolkit (whole aligned image -> 112x112 Lanczos):
      verification AUC and mean genuine cosine on the subset's existing evaluation pairs
      (original img vs de-identified img_b), and cos(original_i, output_i) per image.
      For occlusion, the same for the occluded input itself (no de-identification).
  --part quality    (env: gddiff)
      MSE and LPIPS (AlexNet) between each output and the pipeline input (256x256),
      Haar detection rate on the input, and IoU of the box used vs the MTCNN reference.
  --part summary    merges both into robustness_summary.csv and prints the tables.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLKIT = os.path.expanduser('~/Desktop/master_research/deid-toolkit')
SAMPLES = os.path.join(REPO, 'exp/image_samples')
SETS = {'rafd': 'rafd', 'lfw': 'LFW_unified_orig'}
CONDITIONS = {  # name -> (input set, box spec)
    'base': ('clean', None), 'shift005': ('clean', 'shift:0.05'), 'shift014': ('clean', 'shift:0.14'),
    'shift025': ('clean', 'shift:0.25'), 'shift040': ('clean', 'shift:0.40'), 'scale060': ('clean', 'scale:0.6'),
    # scale 0.8 / 1.25 and forced fallback were dropped for the submission deadline (the fallback case is
    # covered by the full-dataset detection study); occlusions were run on 100 images (50 RaFD + 50 LFW)
    'occl_eyes': ('occl_eyes', None), 'occl_lower': ('occl_lower', None), 'occl_patch': ('occl_patch', None),
}
BRANCHES = ['mean', 'gauss', 'Lapl']


def output_path(cond, sub, branch, name):
    return os.path.join(SAMPLES, f'robust_{cond}', sub, f'0.5_{branch}', name + '.jpg')


# ------------------------------- identity ------------------------------ #
def part_identity(args):
    from sklearn.metrics import roc_auc_score
    import torch
    sys.path.insert(0, os.path.join(TOOLKIT, 'root_dir/evaluation/identity_verification/AdaFace'))
    os.chdir(TOOLKIT)  # inference.py holds the checkpoint path relative to the toolkit root
    from inference import load_pretrained_model, to_input
    model = load_pretrained_model('ir_50').cuda().eval()
    cache = {}

    def emb(path):
        if path not in cache:
            img = Image.open(path).convert('RGB').resize((112, 112), Image.Resampling.LANCZOS)
            with torch.no_grad():
                f, _ = model(to_input(img).cuda())
            cache[path] = f[0].cpu().numpy()
        return cache[path]

    subset = pd.read_csv(os.path.join(args.inputs, 'subset.csv'))
    rows, per_image = [], []
    for sub, ds in SETS.items():
        names = set(subset[subset.set == sub].img)
        pairs = pd.read_csv(os.path.join(TOOLKIT, 'root_dir/results', f'adaface_optimized_{ds}_0.5_mean_GauDeiDiff.csv'))
        pairs['b'] = pairs.img_b.map(lambda f: os.path.splitext(f)[0])
        pairs = pairs[pairs.b.isin(names)]
        orig = lambda f: os.path.join(TOOLKIT, 'root_dir/datasets/aligned', ds, f)
        for cond, (inp, _) in CONDITIONS.items():
            sources = [(b, lambda n, b=b: output_path(cond, sub, b, n)) for b in BRANCHES]
            if inp != 'clean':  # the occluded input itself, before de-identification
                sources.append(('input', lambda n, inp=inp: os.path.join(args.inputs, inp, sub, n + '.jpg')))
            for branch, path_of in sources:
                ok = [os.path.exists(path_of(b)) for b in pairs.b]
                p = pairs[ok]
                if p.empty:
                    continue
                s = np.array([float(emb(orig(a)) @ emb(path_of(b))) for a, b in zip(p.img, p.b)])
                y = p.ground_truth.astype(int).values
                self_pairs = [(n, float(emb(orig(n + '.jpg')) @ emb(path_of(n)))) for n in sorted(names)
                              if os.path.exists(path_of(n))]
                per_image += [dict(set=sub, condition=cond, branch=branch, img=n, self_cos=c) for n, c in self_pairs]
                self_cos = [c for _, c in self_pairs]
                rows.append(dict(set=sub, condition=cond, branch=branch, n_pairs=len(p),
                                 auc=roc_auc_score(y, s), genuine_cos=s[y == 1].mean(),
                                 self_cos=np.mean(self_cos), n_images=len(self_cos)))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'robustness_identity.csv'), index=False)
    pd.DataFrame(per_image).to_csv(os.path.join(args.out, 'robustness_identity_per_image.csv'), index=False)


# ------------------------------- quality ------------------------------- #
def perturb_bbox(bbox, spec, image_name, size=256):
    """Same as guided_diffusion/diffusion.py perturb_bbox (duplicated to avoid importing the sampler)."""
    x, y, w, h = bbox
    if spec == 'fallback':
        s = int(size * 0.65)
        return (size // 2 - s // 2, size // 2 - s // 2, s // 2 * 2, s // 2 * 2)
    kind, value = spec.split(':')
    value = float(value)
    cx, cy = x + w / 2, y + h / 2
    if kind == 'shift':
        angle = int(hashlib.md5(image_name.encode()).hexdigest(), 16) % 3600 / 3600 * 2 * np.pi
        step = value * np.sqrt(w * h)
        cx, cy = cx + step * np.cos(angle), cy + step * np.sin(angle)
    else:
        w, h = w * value, h * value
    x0, y0 = int(round(max(0, cx - w / 2))), int(round(max(0, cy - h / 2)))
    x1, y1 = int(round(min(size, cx + w / 2))), int(round(min(size, cy + h / 2)))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def part_quality(args):
    import cv2
    import lpips
    import torch
    sys.path.insert(0, os.path.join(REPO, 'analysis'))
    from detection_study import iou, pipeline_input, fallback_box
    loss = lpips.LPIPS(net='alex').cuda()
    det = pd.read_csv(os.path.join(REPO, 'analysis/results_detection/detection_study.csv'))
    det = det.set_index(['dataset', 'img'])
    cascade = cv2.CascadeClassifier(os.path.join(REPO, 'haarcascade_frontalface_default.xml'))
    subset = pd.read_csv(os.path.join(args.inputs, 'subset.csv'))
    to_t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
    rows = []
    for _, r in subset.iterrows():
        sub, ds, name = r['set'], r['dataset'], r['img']
        d = det.loc[(ds, name)]
        haar_box = (d.used_x, d.used_y, d.used_w, d.used_h)
        ref = (d.ref_x, d.ref_y, d.ref_w, d.ref_h) if d.mtcnn_found else None
        for cond, (inp, spec) in CONDITIONS.items():
            bgr = pipeline_input(os.path.join(args.inputs, inp, sub, name + '.jpg'))
            faces = cascade.detectMultiScale(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            detected = faces is not None and len(faces) > 0
            box = tuple(int(v) for v in faces[0]) if detected else fallback_box()
            if spec:
                box = perturb_bbox(haar_box, spec, name + '.jpg')
            inp_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for branch in BRANCHES:
                p = output_path(cond, sub, branch, name)
                if not os.path.exists(p):
                    continue
                out = np.array(Image.open(p).convert('RGB').resize((256, 256), Image.BILINEAR))
                with torch.no_grad():
                    lp = float(loss(to_t(out), to_t(inp_rgb)))
                rows.append(dict(set=sub, condition=cond, branch=branch, img=name, haar_detected=detected,
                                 box_iou=iou(box, ref) if ref is not None else np.nan,
                                 mse=float(((out / 255.0 - inp_rgb / 255.0) ** 2).mean()), lpips=lp))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'robustness_quality.csv'), index=False)


# ------------------------------- summary ------------------------------- #
def part_summary(args):
    idn = pd.read_csv(os.path.join(args.out, 'robustness_identity.csv'))
    q = pd.read_csv(os.path.join(args.out, 'robustness_quality.csv'))
    qa = q.groupby(['set', 'condition', 'branch']).agg(
        haar_detected=('haar_detected', 'mean'), box_iou=('box_iou', 'mean'),
        mse=('mse', 'mean'), lpips=('lpips', 'mean')).reset_index()
    df = idn.merge(qa, on=['set', 'condition', 'branch'], how='left')
    df.to_csv(os.path.join(args.out, 'robustness_summary.csv'), index=False)
    order = list(CONDITIONS)
    pd.set_option('display.width', 220)
    for branch in ['mean', 'input']:
        d = df[df.branch == branch].copy()
        if d.empty:
            continue
        d['condition'] = pd.Categorical(d.condition, order, ordered=True)
        cols = ['auc', 'genuine_cos', 'self_cos', 'haar_detected', 'box_iou', 'mse', 'lpips']
        print(f'\n=== branch: {branch} ===')
        print(d.pivot_table(index='condition', columns='set', values=[c for c in cols if c in d], observed=True)
              .round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['identity', 'quality', 'summary'], required=True)
    ap.add_argument('--inputs', required=True, help='folder written by make_robustness_inputs.py')
    ap.add_argument('--out', default=os.path.join(REPO, 'analysis/results_detection'))
    args = ap.parse_args()
    args.inputs, args.out = os.path.abspath(args.inputs), os.path.abspath(args.out)
    {'identity': part_identity, 'quality': part_quality, 'summary': part_summary}[args.part](args)


if __name__ == '__main__':
    main()
