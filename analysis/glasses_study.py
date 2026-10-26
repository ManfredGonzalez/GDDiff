"""Eyeglasses preservation (reviewer R3): are synthetic eyeglasses (GlassesGAN) kept after de-identification?

300 random GlassesGAN images per dataset (RaFD, XM2VTS; extra/prepare.py) were de-identified with GDDiff
(lambda 0.1 and 0.5: Mean-, Gauss- and Lpl-GDDiff) and with FAMS (official code and weights, default settings).
Eyeglasses are detected zero-shot with CLIP ViT-L/14 (prompt ensembles "wearing glasses" vs. "without glasses");
the same detector on the inputs with and without the added glasses checks its sensitivity and specificity.

Usage (env: face-anon-simple):  python analysis/glasses_study.py -> results_glasses/{per_image,summary}.csv
"""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X = os.path.join(REPO, 'exp/robustness_work/extra')
S = os.path.join(REPO, 'exp/image_samples')
OUT = os.path.join(REPO, 'analysis/results_glasses')
SETS = ['rafd', 'xm2vts']
POS = ['a photo of a person wearing glasses', 'a photo of a person wearing eyeglasses', 'a face with spectacles']
NEG = ['a photo of a person without glasses', 'a photo of a person not wearing eyeglasses', 'a face without spectacles']
CONDITIONS = {  # condition -> folder per set
    'input without glasses': lambda s: os.path.join(X, 'glasses_in/plain', s),
    'input with glasses': lambda s: os.path.join(X, 'glasses_in/glasses', s),
    'Mean-GDDiff (0.1)': lambda s: os.path.join(S, 'glasses300_0.1', s, '0.1_mean'),
    'Mean-GDDiff (0.5)': lambda s: os.path.join(S, 'glasses300_0.5', s, '0.5_mean'),
    'Gauss-GDDiff': lambda s: os.path.join(S, 'glasses300_0.5', s, '0.5_gauss'),
    'Lpl-GDDiff': lambda s: os.path.join(S, 'glasses300_0.5', s, '0.5_Lapl'),
    'FAMS': lambda s: os.path.join(X, 'fams_glasses', s),
}


def main():
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').cuda().eval()
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    with torch.no_grad():
        t = model.get_text_features(**proc(text=POS + NEG, return_tensors='pt', padding=True).to('cuda'))
        t = t / t.norm(dim=-1, keepdim=True)
        t = torch.stack([t[:len(POS)].mean(0), t[len(POS):].mean(0)])
        t = t / t.norm(dim=-1, keepdim=True)
    subset = pd.read_csv(os.path.join(X, 'glasses_subset.csv'))
    rows = []
    for cond, folder_of in CONDITIONS.items():
        for s in SETS:
            folder = folder_of(s)
            files = {os.path.splitext(f)[0]: f for f in os.listdir(folder)} if os.path.isdir(folder) else {}
            names = [n for n in subset[subset.set == s].img if n in files]
            for i in range(0, len(names), 32):
                batch = names[i:i + 32]
                ims = [Image.open(os.path.join(folder, files[n])).convert('RGB') for n in batch]
                with torch.no_grad():
                    v = model.get_image_features(**proc(images=ims, return_tensors='pt').to('cuda'))
                    v = v / v.norm(dim=-1, keepdim=True)
                    p = (model.logit_scale.exp() * v @ t.T).softmax(-1)[:, 0].cpu().numpy()
                rows += [dict(condition=cond, set=s, img=n, p_glasses=float(q)) for n, q in zip(batch, p)]
            print(cond, s, len(names), flush=True)
    df = pd.DataFrame(rows)
    df['glasses'] = (df.p_glasses >= 0.5).astype(int)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, 'per_image.csv'), index=False)

    from scipy.stats import binomtest
    piv = df.pivot_table(index=['set', 'img'], columns='condition', values='glasses')
    summ = []
    for cond in CONDITIONS:
        if cond not in piv:
            continue
        for s in SETS + ['all']:
            sub = piv if s == 'all' else piv.loc[s]
            ok = sub[list(dict.fromkeys([cond, 'input with glasses']))].dropna()
            row = dict(condition=cond, set=s, n=len(ok), detected=ok[cond].mean(),
                       retained=ok.loc[ok['input with glasses'] == 1, cond].mean())
            if cond not in ('input with glasses', 'input without glasses') and 'FAMS' in sub and cond != 'FAMS':
                pair = sub[[cond, 'FAMS']].dropna()  # exact McNemar vs. FAMS on the same images
                b, c = int(((pair[cond] == 1) & (pair.FAMS == 0)).sum()), int(((pair[cond] == 0) & (pair.FAMS == 1)).sum())
                row['p_vs_FAMS'] = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
            summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    pd.set_option('display.width', 200)
    print('\nfraction of images with eyeglasses detected (CLIP ViT-L/14 zero-shot); '
          'retained = among inputs whose added glasses are detected')
    print(summ.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
