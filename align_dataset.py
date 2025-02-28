import numpy as np
import pickle
import torch
import PIL
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from hashlib import md5
import csv
import os
import scipy
import scipy.ndimage
import tempfile
from tqdm.auto import tqdm
import cv2
import dlib
def pil_to_cv2(pil_image):
    # Convert PIL Image to a NumPy array
    img_np = np.array(pil_image)
    
    # If the image has an alpha channel (RGBA), remove it
    if img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]
    
    # Convert RGB to BGR (OpenCV's default color format)
    cv2_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return cv2_image
# download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = dlib.shape_predictor('./model_dlib/shape_predictor_68_face_landmarks.dat')
face_detector2 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Global variables to store rectangles for averaging and standard deviation
rectangles_list = []
def get_face_bbox(face_detector, image):
    """
    Detects a face in the image and returns bounding box coordinates as dlib.rectangle objects.
    
    :param face_detector: A face detector object from OpenCV (e.g., cv2.CascadeClassifier)
    :param image: Input PIL image
    :return: A list of bounding boxes as dlib.rectangle objects
    """
    # Convert the PIL image to OpenCV format
    cv2_image = pil_to_cv2(image)

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image using OpenCV's face detector
    faces = face_detector.detectMultiScale(gray)
    
    # Convert each bounding box to dlib.rectangle format
    dlib_rectangles = []
    for (x, y, w, h) in faces:
        dlib_rectangles.append(dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h))
    
    return dlib_rectangles
def get_average_bbox_and_std():
    # Calculate average and standard deviation if we have more than one rectangle
    if len(rectangles_list) > 1:
        rect_coords = np.array([[r.left(), r.top(), r.right(), r.bottom()] for r in rectangles_list])
        avg_rect_coords = np.mean(rect_coords, axis=0)
        std_dev_rect_coords = np.std(rect_coords, axis=0)
        avg_rect = dlib.rectangle(
            int(avg_rect_coords[0]), int(avg_rect_coords[1]),
            int(avg_rect_coords[2]), int(avg_rect_coords[3])
        )
        std_dev_rect = std_dev_rect_coords  # Standard deviation in each coordinate
    else:
        avg_rect = rectangles_list[0]
        std_dev_rect = np.zeros(4)  # No deviation with only one rectangle
    return avg_rect,std_dev_rect

def get_landmark(filepath):
#def get_landmark(pil_image):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    try:
        img = dlib.load_rgb_image(filepath)
    except RuntimeError: # PPM files not supported by dlib
        import cv2
        img = cv2.imread(filepath)
    # Create a temporary file to save the image
    '''with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
        # Save the PIL image as a PNG to the temporary file
        pil_image.save(temp_file.name, format="PNG")
        
        # Load the image using dlib's load_rgb_image
        img = dlib.load_rgb_image(temp_file.name)'''
    dets = detector(img, 1)
    if len(dets)==0:
        dets = get_face_bbox(face_detector2, img)
    if len(dets)==0:
        avg_det, std_det = get_average_bbox_and_std()
        print("Average Detection for image {}: Left: {} Top: {} Right: {} Bottom: {} STD: {}".format(
            filepath, avg_det.left(), avg_det.top(), avg_det.right(), avg_det.bottom(), std_det))
        dets = [avg_det]
    else:
        # Use the first detected face for consistency
        det = dets[0]
        rectangles_list.append(det)  # Append detected rectangle for statistics

    #print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))


    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    # lm is a shape=(68,2) np.array
    return lm

    
def align_face(filepath):
    """
    :param filepath: str
    :return: PIL Image
    """

    lm = get_landmark(filepath)
    if lm is None:
        return None
    
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2


    # read image
    img = PIL.Image.open(filepath)

    output_size=1024
    transform_size=4096
    enable_padding=True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.LANCZOS)

    # Save aligned image.
    return img
def align_faces_in_directory(target_dir):
    """
    Aligns all images in the target directory and saves them in a new directory with '_aligned' added to the name.

    :param target_dir: str - Path to the directory containing the images to be aligned.
    """
    # Create output directory
    output_dir = target_dir + "_aligned"
    os.makedirs(output_dir, exist_ok=True)

    # List all image files in the directory (supports various image formats)
    image_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Process each image with a progress bar
    for image_name in tqdm(image_files, desc="Aligning images"):
        image_path = os.path.join(target_dir, image_name)
        
        # Try to align the face in the image
        aligned_image = align_face(image_path)
        
        if aligned_image is not None:
            # Save aligned image to output directory
            aligned_image_path = os.path.join(output_dir, image_name)
            aligned_image.save(aligned_image_path)
        else:
            print(f"Warning: No face detected or alignment failed for '{image_name}'.")

    print(f"All aligned images are saved in '{output_dir}'.")
align_faces_in_directory('/home/tico/Desktop/master_classes/IBB/project/Mask_DeID_DDPG/exp/datasets/CALFW_benchmark')