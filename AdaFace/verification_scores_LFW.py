import pandas as pd
import torch
import os
from PIL import Image
import pickle
import torch.nn.functional as F
from face_alignment import align
from inference import load_pretrained_model, to_input
from tqdm import tqdm  # Import tqdm for progress bars
import shutil
# Load the CSV data
cd = os.getcwd()
df = pd.read_csv('exp/datasets/LFW_benchmark_pairs_mapping.csv')
features_dirpath1 = 'AdaFace/temp_feats1'
features_dirpath2 = 'AdaFace/temp_feats2'
image1_dir_path='exp/datasets/celeba_hq/LFW_benchmark_aligned'
image2_dir_path='exp/image_samples/LFW_benchmark_aligned'
#image2_dir_path='exp/datasets/celeba_hq/LFW_benchmark_aligned'
deid_impostors = False #parameter that says if you want to use deid comparisions in impostor pairs

image1_path = image1_dir_path+'/{0}.png'
image2_path = image2_dir_path+'/{0}.png'
previous_bboxes=[]
if not os.path.exists(image1_dir_path):
    raise('Directory does not exits: ',image1_dir_path)
if not os.path.exists(image2_dir_path):
    raise('Directory does not exits: ',image2_dir_path)

# Load the pre-trained model
model = load_pretrained_model('ir_50').cuda()
model.eval()

# Function to save features to a file
def save_features(filepath, features):
    with open(filepath, 'wb') as f:
        pickle.dump(features.cpu(), f)

# Function to load features from a file
def load_features(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f).cuda()

# Function to compute or retrieve features for an image
def get_or_compute_features(image_path, features_dirpath, model, mask_face=False):
    image_name = os.path.basename(image_path)
    feature_filepath = os.path.join(features_dirpath, f"{image_name}.pkl")

    if os.path.exists(feature_filepath):
        return load_features(feature_filepath)

    image = Image.open(image_path).convert('RGB')
    aligned_rgb_img = align.get_aligned_face(image_path, rgb_pil_image=image,previous_bboxes=previous_bboxes)
    if aligned_rgb_img is None:
        return None

    bgr_input = to_input(aligned_rgb_img).cuda()
    with torch.no_grad():
        feature, _ = model(bgr_input)

    save_features(feature_filepath, feature)
    return feature

# Function to compute the cosine similarity score
def compute_similarity(feature1, feature2):
    cosine_similarity = F.cosine_similarity(feature1, feature2)
    similarity_score = (cosine_similarity + 1) / 2
    return similarity_score.item()

# Directory where features will be saved/loaded from
os.makedirs(features_dirpath1, exist_ok=True)
os.makedirs(features_dirpath2, exist_ok=True)
# Initialize a list to store similarity scores
similarity_scores = []
failed_pairs = []
# Process each row in the DataFrame
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing pairs"):
    image1_id = 'img_'+row['Image1_ID']
    image2_id = 'img_'+row['Image2_ID']
    isGenuine = row['IsGenuine']

    feature1 = get_or_compute_features(image1_path.format(image1_id), features_dirpath1, model)
    if isGenuine or deid_impostors:
        feature2 = get_or_compute_features(image2_path.format(image2_id), features_dirpath2, model)
    else:
        feature2 = get_or_compute_features(image1_path.format(image2_id), features_dirpath2, model)#take the image from the original paths

    if feature1 is None or feature2 is None:
        # If either feature is None, save the pair for later processing
        failed_pairs.append((index, image1_id, image2_id, isGenuine))
        similarity_scores.append(None)  # Placeholder for missing similarity score
        continue

    similarity = compute_similarity(feature1, feature2)
    similarity_scores.append(similarity)

# Add the similarity scores as a new column
df['Similarity'] = similarity_scores

# Save the updated DataFrame to a new CSV file
df.to_csv('data_with_similarity.csv', index=False)

# Process failed pairs if bounding boxes are populated
if previous_bboxes:
    for index, image1_id, image2_id, isGenuine in tqdm(failed_pairs, desc="Processing failed pairs"):
        feature1 = get_or_compute_features(image1_path.format(image1_id), features_dirpath1, model)
        if isGenuine or deid_impostors:
            feature2 = get_or_compute_features(image2_path.format(image2_id), features_dirpath2, model)
        else:
            feature2 = get_or_compute_features(image1_path.format(image2_id), features_dirpath2, model)#take the image from the original paths

        if feature1 is not None and feature2 is not None:
            similarity = compute_similarity(feature1, feature2)
            df.at[index, 'Similarity'] = similarity  # Update similarity in DataFrame

# Save the final DataFrame with updated similarities
df.to_csv('data_with_similarity_final.csv', index=False)
shutil.rmtree(features_dirpath1)
shutil.rmtree(features_dirpath2)