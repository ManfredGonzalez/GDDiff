"""Existing GDDiff results split by how well the face detector did on each input.

Uses the per-image detection outcome from detection_study.py and the toolkit's
existing per-pair AdaFace scores and per-image MSE; no new GDDiff runs.

Strata (of the de-identified image's input):
    good      Haar box found, IoU >= 0.5 with the MTCNN reference box
    offset    Haar box found, IoU <  0.5 (misplaced box)
    fallback  no Haar detection -> centred 65% fallback box
Per stratum: verification AUC on the de-identified pairs (lower = more private),
the same pairs' AUC on the originals (control for intrinsic difficulty), mean
genuine cosine similarity, and mean MSE.

Usage (env: gddiff):
    python analysis/stratify_by_detection.py analysis/results_detection/detection_study.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RESULTS = os.path.expanduser('~/Desktop/master_research/deid-toolkit/root_dir/results')
DATASETS = ['CALFW', 'LFW_unified_orig', 'rafd', 'xm2vts_aligned']
TECHS = ['0.1_mean_GauDeiDiff', '0.2_mean_GauDeiDiff', '0.5_mean_GauDeiDiff', 'gauss_GauDeiDiff', 'lpl_GauDeiDiff']


def stratum(r):
    if r.fallback:
        return 'fallback'
    if not r.mtcnn_found:
        return 'no_reference'
    return 'good' if r.iou >= 0.5 else 'offset'


def auc(y, s):
    return roc_auc_score(y, s) if len(set(y)) == 2 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('detection_csv')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    det = pd.read_csv(args.detection_csv)
    det['stratum'] = det.apply(stratum, axis=1)
    rows = []
    for ds in DATASETS:
        s_of = dict(zip(det[det.dataset == ds].img, det[det.dataset == ds].stratum))
        for tech in TECHS:
            pairs = pd.read_csv(os.path.join(RESULTS, f'adaface_optimized_{ds}_{tech}.csv'))
            pairs['stratum'] = pairs.img_b.map(lambda f: s_of.get(os.path.splitext(f)[0]))
            # file names are not consistently cased (e.g. mse_CALFW_Lpl_GauDeiDiff.csv)
            mse_name = {f.lower(): f for f in os.listdir(RESULTS)}.get(f'mse_{ds}_{tech}.csv'.lower())
            mse = (pd.read_csv(os.path.join(RESULTS, mse_name)) if mse_name
                   else pd.DataFrame(columns=['img', 'mse']))
            mse['stratum'] = mse.img.map(lambda f: s_of.get(os.path.splitext(f)[0]))
            for st, p in pairs.groupby('stratum'):
                y = p.ground_truth.astype(int)
                rows.append(dict(dataset=ds, technique=tech, stratum=st,
                                 n_images=len({os.path.splitext(f)[0] for f in p.img_b}), n_pairs=len(p),
                                 auc_deid=auc(y, p.cossim), auc_original=auc(y, p.cossim_originals),
                                 genuine_cos=p[p.ground_truth].cossim.mean(),
                                 mse=mse[mse.stratum == st].mse.mean()))
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
    pd.set_option('display.width', 220)
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
