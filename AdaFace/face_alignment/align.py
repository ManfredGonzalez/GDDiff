import sys
import os

from face_alignment import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime
mtcnn_model = mtcnn.MTCNN(device='cuda:0', crop_size=(112, 112))

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_aligned_face(image_path, rgb_pil_image=None,previous_bboxes=None):
    if image_path == 'exp/datasets/celeba_hq/LFW_benchmark_aligned/img_336f215a9a8df055fd581e90729a5bb5.png':
        print('here')
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
        img = rgb_pil_image
        #img.show()
    # find face
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1,previous_bboxes=previous_bboxes)
        face = faces[0]
    except Exception as e:
        #print('Face detection Failed due to error.')
        face = None
    #face.save(os.path.basename(image_path), "PNG")
    return face


