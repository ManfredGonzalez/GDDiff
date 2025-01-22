import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from tqdm import tqdm
import random
import math
import glob
import os

# Define a range of thresholds from the minimum to the maximum score
def calculate_thresholds(genuine_scores, impostor_scores, num_thresholds=100):
    min_score = min(min(genuine_scores), min(impostor_scores))
    max_score = max(max(genuine_scores), max(impostor_scores))
    return np.linspace(min_score, max_score, num_thresholds)

# Compute FAR, FRR, and EER with Bootstrapping
def calculate_far_frr(threshold, genuine_scores, impostor_scores):
    far = np.sum(np.array(impostor_scores) >= threshold) / len(impostor_scores)
    frr = np.sum(np.array(genuine_scores) < threshold) / len(genuine_scores)
    return far, frr

def bootstrap_eer(thresholds, genuine_scores, impostor_scores, n_bootstrap=1000):
    far_values = []
    frr_values = []
    auc_values = []

    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping Progress"):
        genuine_sample = [random.choice(genuine_scores) for _ in range(len(genuine_scores))]
        impostor_sample = [random.choice(impostor_scores) for _ in range(len(impostor_scores))]

        fars = []
        frrs = []
        for threshold in thresholds:
            far, frr = calculate_far_frr(threshold, genuine_sample, impostor_sample)
            fars.append(far)
            frrs.append(frr)

        roc_auc = auc(fars, 1 - np.array(frrs))
        auc_values.append(roc_auc)

        far_values.append(fars)
        frr_values.append(frrs)

    far_values = np.array(far_values)
    frr_values = np.array(frr_values)
    auc_values = np.array(auc_values)

    return far_values, frr_values, auc_values

# Plot single ROC curve with uncertainty
def plot_single_roc_with_uncertainty(csv_files,title, n_bootstrap=1000):

    # Separate files by technique
    mask_ddpg_files = [file for file in csv_files if not 'cpp-deid' in file[0]]
    cpp_deid_files = [file for file in csv_files if 'cpp-deid' in file[0]]


    # Plot Mask-DDPG curves
    for idx, (path, label) in enumerate(mask_ddpg_files):
        data = pd.read_csv(path)
        genuine_scores = data[data['ground_truth'] == 1]['cossim'].tolist()
        impostor_scores = data[data['ground_truth'] == 0]['cossim'].tolist()

        thresholds = calculate_thresholds(genuine_scores, impostor_scores)
        far_values, frr_values, auc_values = bootstrap_eer(thresholds, genuine_scores, impostor_scores, n_bootstrap)

        mean_far = np.mean(far_values, axis=0)
        mean_frr = np.mean(frr_values, axis=0)

        far_std = np.std(far_values, axis=0)
        frr_std = np.std(frr_values, axis=0)

        roc_auc = np.mean(auc_values)
        auc_uncertainty = np.std(auc_values)
        print(label +' AUC:'+ str(roc_auc) +' uncer: '+ str(auc_uncertainty))

        

    # Plot CPP-DeID curves
    for idx, (path, label) in enumerate(cpp_deid_files):
        data = pd.read_csv(path)
        genuine_scores = data[data['ground_truth'] == 1]['cossim'].tolist()
        impostor_scores = data[data['ground_truth'] == 0]['cossim'].tolist()

        thresholds = calculate_thresholds(genuine_scores, impostor_scores)
        far_values, frr_values, auc_values = bootstrap_eer(thresholds, genuine_scores, impostor_scores, n_bootstrap)

        mean_far = np.mean(far_values, axis=0)
        mean_frr = np.mean(frr_values, axis=0)

        far_std = np.std(far_values, axis=0)
        frr_std = np.std(frr_values, axis=0)

        roc_auc = np.mean(auc_values)
        auc_uncertainty = np.std(auc_values)
        print(label +' AUC:'+ str(roc_auc) +' uncer: '+ str(auc_uncertainty))
csv_files = [#pairs: path, label
    ('/home/tico/Desktop/master_research/deid-toolkit/root_dir/results/adaface_optimized_arface_cpp-deid.csv','arface_cpp-deid'),
    ('/home/tico/Desktop/master_research/deid-toolkit/root_dir/results/adaface_optimized_ck+_fix_cpp-deid.csv', 'ck+_fix_cpp-deid')
]
plot_single_roc_with_uncertainty(csv_files,'ROC of LFW_benchmark, adaface embeddings', n_bootstrap=1000)