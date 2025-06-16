import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

CANCER_SET=r"..\\dataset\\test_dataset_patches\\tissu_cancereux"
NON_CANCER_SET= r"..\\dataset\\test_dataset_patches\\tissus_saint"
NON_CANCER_TARGET_FOLDER = r"..\\dataset\\filtered_out_test_patches\\tissus_saint"
CANCER_TARGET_FOLDER = r"..\\dataset\\filtered_out_test_patches\\tissu_cancereux"
WHITE_THRESHOLD=230
WHITE_RATIO=0.8
MOVE_INSTEAD_OF_COPY=True

def is_mostly_white(image, white_thresh=230, white_ratio=0.8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white_pixels = np.sum(gray > white_thresh)
    total_pixels = gray.size
    return (white_pixels / total_pixels) > white_ratio

def copy_white_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in tqdm(os.listdir(input_dir)):
        path = os.path.join(input_dir, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        if is_mostly_white(img, WHITE_THRESHOLD, WHITE_RATIO):
            shutil.copy2(path, os.path.join(output_dir, filename))
            
            
def move_white_images(input_dir, output_dir):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        src = os.path.join(input_dir, filename)
        dst = os.path.join(output_dir, filename)

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        if not os.path.exists(src):
            continue

        img = cv2.imread(src)
        if img is None:
            continue
        if is_mostly_white(img, WHITE_THRESHOLD, WHITE_RATIO):
            shutil.move(src, dst)

if MOVE_INSTEAD_OF_COPY:
    move_white_images(CANCER_SET, 'white-patches')
    move_white_images(NON_CANCER_SET, 'white-patches-non-cancer')
else:
    copy_white_images(CANCER_SET, 'white-patches')
    copy_white_images(NON_CANCER_SET, 'white-patches-non-cancer')
