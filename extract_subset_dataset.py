import os
import random
import shutil
from tqdm import tqdm

CANCER_SET = r"..\dataset\train_dataset_patches_full\tissus_saint"
TARGET_FOLDER = r"..\dataset\train_dataset_patches\tissu_sain"

def sample_images(input_dir, output_dir, sample_size=17159):
    os.makedirs(output_dir, exist_ok=True)
    all_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if sample_size > len(all_images):
        raise ValueError(f"Requested {sample_size} samples, but only {len(all_images)} images available.")

    sampled = random.sample(all_images, sample_size)

    for filename in tqdm(sampled, desc="Copying sampled images"):
        src = os.path.join(input_dir, filename)
        dst = os.path.join(output_dir, filename)
        shutil.copy2(src, dst)

# Run
sample_images(CANCER_SET, TARGET_FOLDER, 17159)
