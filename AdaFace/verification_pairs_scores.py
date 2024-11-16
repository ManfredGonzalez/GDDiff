import os
import glob
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from face_alignment import align
from inference import load_pretrained_model, to_input
import shutil

import torch
import torch.nn.functional as F
import pickle
from PIL import Image
import cv2
import numpy as np
from face_alignment import mtcnn
mtcnn_model = mtcnn.MTCNN(device='cuda:0', crop_size=(112, 112))

# Directory paths
deid_images_dirpath = '/home/tico/Desktop/master_research/DDPG/exp/image_samples/rafd-frontal-GDDPG'
#deid_images_dirpath = '/home/tico/Desktop/master_research/rafd_deidentified/cpp_deid_0.5'
#deid_images_dirpath = '/home/tico/Desktop/master_research/DDPG/exp/datasets/celeba_hq/rafd-frontal'
original_images_dirpath = '/home/tico/Desktop/master_research/DDPG/exp/datasets/celeba_hq/rafd-frontal'
deid_features_dirpath = '/home/tico/Desktop/master_research/DDPG/exp/features/deid'
#deid_features_dirpath = '/home/tico/Desktop/master_research/DDPG/exp/features/original'
original_features_dirpath = '/home/tico/Desktop/master_research/DDPG/exp/features/original'

face_detector_path = 'haarcascade_frontalface_default.xml'
with_mask = False

# Create directories for features if they don't exist
os.makedirs(deid_features_dirpath, exist_ok=True)
os.makedirs(original_features_dirpath, exist_ok=True)

# Get all image paths for jpg, png, jpeg using glob
deid_images = glob.glob(f"{deid_images_dirpath}/*.[jJpP][pPnN][gGeE]")
original_images = glob.glob(f"{original_images_dirpath}/*.[jJpP][pPnN][gGeE]")

# Extract image filenames from paths and find common images
deid_filenames = set(os.path.basename(path) for path in deid_images)
original_filenames = set(os.path.basename(path) for path in original_images)
common_images = deid_filenames.intersection(original_filenames)

assert len(common_images) == len(deid_images) == len(original_images), "The directories do not have the same set of images."

# Optimized approach with a progress bar

file_pairs = []
processed_pairs = set()

# Iterate over product with tqdm for the progress bar
for deid_image, original_image in tqdm(product(deid_images, original_images), total=len(deid_images) * len(original_images), desc="Processing Pairs"):
    deid_image_name = os.path.basename(deid_image)
    original_image_name = os.path.basename(original_image)

    deid_id = deid_image_name.split("_")[1]
    original_id = original_image_name.split("_")[1]

    pair_key = tuple(sorted([deid_image_name, original_image_name]))
    if deid_image_name != original_image_name and pair_key not in processed_pairs:
        processed_pairs.add(pair_key)
        is_genuine = (deid_id == original_id)
        file_pairs.append((deid_image_name, original_image_name, is_genuine))

# Convert to pandas DataFrame for easy visualization
df_pairs_optimized = pd.DataFrame(file_pairs, columns=['Deid_Image', 'Original_Image', 'Genuine'])

# Load the pretrained model (ensure the model is on GPU)
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
    
face_detector = cv2.CascadeClassifier(face_detector_path)  
def get_face_bbox(face_detector,image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_detector.detectMultiScale(gray)
    x, y, w, h = faces[0]
    return x, y, w, h
# Function to preprocess and extract features, or load them if already computed
def get_or_compute_features(image_path, features_dirpath, mask_face = False):
    image_name = os.path.basename(image_path)
    feature_filepath = os.path.join(features_dirpath, f"{image_name}.pkl")

    # Check if the features have already been computed and saved
    if os.path.exists(feature_filepath):
        return load_features(feature_filepath)
    
    image = Image.open(image_path).convert('RGB')
    if mask_face:
        boxes, landmarks = mtcnn_model.detect_faces(image, mtcnn_model.min_face_size,
                                  mtcnn_model.thresholds, 
                                  mtcnn_model.nms_thresholds, 
                                  mtcnn_model.factor)
        '''
        Bounding Box Coordinates:
            x1: The x-coordinate of the top-left corner.
            y1: The y-coordinate of the top-left corner.
            x2: The x-coordinate of the bottom-right corner.
            y2: The y-coordinate of the bottom-right corner.

        Confidence Score:
            score: The confidence score for the bounding box, representing the likelihood of a face being in this region.
        '''
        try:
            # Find the index of the bounding box with the highest score
            max_score_index = np.argmax(boxes[:, 4])  # The score is in the last column

            # Extract the bounding box with the highest score
            best_bounding_box = boxes[max_score_index]
        except Exception as e:
            print('Face detection Failed due to error.')
            print(e)
            best_bounding_box = None
        if best_bounding_box is not None:
            x1, y1, x2, y2, score = best_bounding_box.astype(int)
            # Create a mask with the same shape as the image, initialized to zeros (black)
            mask = np.zeros_like(image)
            image = np.array(image)
            # Copy only the face region from the original image to the mask
            mask[y1:y2, x1:x2] = image[y1:y2, x1:x2]
            # Now `mask` contains the face in color and the rest of the image as black
            # Convert back to PIL format if needed
            masked_image = Image.fromarray(mask)
            #masked_image.save(os.path.basename(image_path), "PNG")
            # Compute the features if not available
            aligned_rgb_img = align.get_aligned_face(image_path,rgb_pil_image=masked_image)
        else:
            aligned_rgb_img = align.get_aligned_face(image_path,rgb_pil_image=image)
    else:
        # Compute the features if not available
        aligned_rgb_img = align.get_aligned_face(image_path,rgb_pil_image=image)
    if aligned_rgb_img is None:
        print('Alignment failed at image: ', image_path)
        #Compute the embeddings without mask
        aligned_rgb_img = align.get_aligned_face(image_path,rgb_pil_image=Image.fromarray(image))
    bgr_input = to_input(aligned_rgb_img).cuda()
    with torch.no_grad():
        feature, _ = model(bgr_input)

    # Save the computed features for future use
    save_features(feature_filepath, feature)
    return feature
# Function to compute the cosine similarity score
def compute_similarity(feature1, feature2):
    cosine_similarity = F.cosine_similarity(feature1, feature2)
    # Standardize the cosine similarity score to range [0, 1]
    similarity_score = (cosine_similarity + 1) / 2
    return similarity_score.item()

# Initialize a list to store similarity scores
similarity_scores = []

# Loop through the dataframe and compute similarities
for idx, row in tqdm(df_pairs_optimized.iterrows(), total=len(df_pairs_optimized), desc="Calculating Similarities"):
    deid_image_path = f"{deid_images_dirpath}/{row['Deid_Image']}"
    original_image_path = f"{original_images_dirpath}/{row['Original_Image']}"

    # Get or compute features for both images
    feature_deid = get_or_compute_features(deid_image_path, deid_features_dirpath,
                                            mask_face = with_mask)
    feature_original = get_or_compute_features(original_image_path, original_features_dirpath,
                                                mask_face = with_mask)

    # Compute similarity and add to the list
    similarity_score = compute_similarity(feature_deid, feature_original)
    similarity_scores.append(similarity_score)

# Add the similarity scores to the dataframe
df_pairs_optimized['AdaFace_Similarity'] = similarity_scores

method = os.path.basename(deid_images_dirpath)
# Save the dataframe as a CSV
if with_mask:
    df_pairs_optimized.to_csv(f'optimized_pairs_with_similarity_{method}_masked.csv', index=False)
else:
    df_pairs_optimized.to_csv(f'optimized_pairs_with_similarity_{method}.csv', index=False)

shutil.rmtree(deid_features_dirpath)
shutil.rmtree(original_features_dirpath)

