"""Build degraded test images in dataset/ from the base photos there (flower.jpg, test.jpg).

    python experiments/generate_synthetic_dataset.py [--dataset PATH]

The default dataset folder is <repo root>/dataset (it is git-ignored: 11 MB of photos).
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # the pipeline package, without needing pip install
from ivp_pipeline import synthetic  # noqa: E402
from ivp_pipeline.utils import prepare  # noqa: E402

DATASET_DIR = str(HERE.parents[1] / "dataset")

def save(name, img_rgb):
    # Convert RGB back to BGR for OpenCV saving
    if img_rgb.ndim == 3:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        bgr = img_rgb
    
    path = os.path.join(DATASET_DIR, name)
    cv2.imwrite(path, bgr)
    print(f"Saved: {name}")

def load(name):
    path = os.path.join(DATASET_DIR, name)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    # 1. Base image for noise/illumination tests
    flower = load("flower.jpg")
    # Downscale so it's easily viewable and matches backend processing size
    flower, _ = prepare(flower)
    
    # Generate Denoise test images
    sp_img = synthetic.salt_pepper(flower, 0.05)
    save("synthetic_for_median_filter_salt_pepper.jpg", sp_img)
    
    gauss_img = synthetic.gaussian_noise(flower, 15.0)
    save("synthetic_for_gaussian_blur_noise.jpg", gauss_img)
    
    # Generate Contrast / Flatten test image
    ramp_img = synthetic.illumination_ramp(flower, 60)
    save("synthetic_for_flatten_illumination_ramp.jpg", ramp_img)
    
    # 2. Base image for contrast tests
    test_img = load("test.jpg")
    test_img, _ = prepare(test_img)
    
    low_con = synthetic.low_contrast(test_img, 80, 160)
    save("synthetic_for_contrast_stretch_low_con.jpg", low_con)
    
    dark_img = synthetic.apply_gamma(test_img, 2.5) # Darkens
    save("synthetic_for_gamma_correction_dark.jpg", dark_img)
    
    bright_img = synthetic.apply_gamma(test_img, 0.4) # Brightens
    save("synthetic_for_gamma_correction_bright.jpg", bright_img)
    
    # 3. Generate a pure synthetic scene for segmentation testing
    scene_gray, mask = synthetic.scene(seed=42)
    # Convert gray to RGB so it matches expected pipeline input format
    scene_rgb = cv2.cvtColor(scene_gray, cv2.COLOR_GRAY2RGB)
    save("synthetic_for_segmentation_scene.jpg", scene_rgb)
    
    # Also save the ground truth mask for reference
    mask_visual = (mask * 255).astype(np.uint8)
    save("synthetic_for_segmentation_ground_truth_mask.png", mask_visual)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=DATASET_DIR, help="folder holding flower.jpg and test.jpg")
    DATASET_DIR = ap.parse_args().dataset
    main()
