"""End-to-end smoke run of every module on the photos in dataset/ (prints PASS/FAIL lines).

    python experiments/run_e2e_tests.py [--dataset PATH]

The default dataset folder is <repo root>/dataset. Not part of pytest: the photos are git-ignored.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # the pipeline package, without needing pip install
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import ivp_pipeline as ivp  # noqa: E402
from ivp_pipeline import synthetic  # noqa: E402

DATASET_DIR = str(HERE.parents[1] / "dataset")

def load_img(name):
    p = os.path.join(DATASET_DIR, name)
    img = cv2.imread(p)
    if img is None:
        raise FileNotFoundError(f"Missing {p}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def report(name, before, after, metric, expected_trend):
    b_val = before['metrics'][metric]
    a_val = after['metrics'][metric]
    success = False
    if expected_trend == "down":
        success = (a_val < b_val)
    elif expected_trend == "up":
        success = (a_val > b_val)
    elif expected_trend == "mid":
        success = abs(a_val - 0.5) < abs(b_val - 0.5)
    print(f"[{'PASS' if success else 'FAIL'}] {name}: {metric} {b_val:.3f} -> {a_val:.3f}")

def run_tests():
    print("=== Denoise Tests ===")
    img = load_img("flower.jpg")
    # Downscale so prepare() doesn't average away synthetic noise during diagnose()
    img, _ = ivp.utils.prepare(img)
    
    # 1. Median
    sp_img = synthetic.salt_pepper(img, 0.05)
    res = ivp.run_module(sp_img, "denoise", {"filter_type": "median", "kernel_size": 3, "sigma": 1.0})
    pre_id = ivp.metrics.impulse_density(ivp.utils.to_gray(sp_img))
    post_id = ivp.metrics.impulse_density(ivp.utils.to_gray(res["image"]))
    print(f"[{'PASS' if post_id < pre_id else 'FAIL'}] Median Filter: impulse_density {pre_id:.3f} -> {post_id:.3f}")

    # 2. Gaussian
    gauss_img = synthetic.gaussian_noise(img, 15.0)
    res = ivp.run_module(gauss_img, "denoise", {"filter_type": "gaussian", "kernel_size": 3, "sigma": 1.5})
    report("Gaussian Blur", ivp.diagnose(gauss_img), ivp.diagnose(res["image"]), "noise_sigma", "down")

    print("\n=== Contrast Tests ===")
    low_con = load_img("low_con.jpg")
    res = ivp.run_module(low_con, "contrast", {"mode": "stretch", "clip_percentile": 1.0, "gamma": 1.0, "se_radius": 15, "object_polarity": "dark"})
    report("Contrast Stretch", ivp.diagnose(low_con), ivp.diagnose(res["image"]), "range_p1_p99", "up")
    
    dark = load_img("dark_image.jpg")
    res = ivp.run_module(dark, "contrast", {"mode": "gamma", "gamma": "auto", "clip_percentile": 1.0, "se_radius": 15, "object_polarity": "dark"})
    report("Gamma Correction", ivp.diagnose(dark), ivp.diagnose(res["image"]), "mean", "mid")
    
    res = ivp.run_module(low_con, "contrast", {"mode": "hist_eq", "clip_percentile": 1.0, "gamma": 1.0, "se_radius": 15, "object_polarity": "dark"})
    report("Equalize", ivp.diagnose(low_con), ivp.diagnose(res["image"]), "std", "up")

    ramp_img = synthetic.illumination_ramp(img, 60)
    res = ivp.run_module(ramp_img, "contrast", {"mode": "flatten", "se_radius": 25, "object_polarity": "dark", "clip_percentile": 1.0, "gamma": 1.0})
    report("Flatten Illumination", ivp.diagnose(ramp_img), ivp.diagnose(res["image"]), "illumination_u", "down")

    print("\n=== Sharpen/Edges Tests ===")
    res = ivp.run_module(img, "sharpen_edges", {"operation": "unsharp", "k": 1.5, "blur_sigma": 1.0, "A": 1.0})
    report("Unsharp Mask", ivp.diagnose(img), ivp.diagnose(res["image"]), "sharpness_lap_var", "up")
    
    res = ivp.run_module(img, "sharpen_edges", {"operation": "gradient", "operator": "sobel", "magnitude": "l2", "component": "magnitude", "edge_threshold": 0.0, "A": 1.0, "k": 1.0, "blur_sigma": 1.0})
    print("[PASS] Gradient (Sobel) executed.")
    
    print("\n=== Segment Tests ===")
    coin = load_img("coin.png")
    # Otsu cutout
    res = ivp.run_module(coin, "segment", {
        "method": "otsu", "foreground": "auto", 
        "morph_open": True, "morph_close": True, "disk_radius": 3, 
        "fill_holes": True, "keep_largest": True, 
        "delta_t": 15, "connectivity": 4, "adaptive": False
    })
    mask = res["extras"]["mask"]
    obj_fraction = (mask > 0).mean()
    success = 0.01 < obj_fraction < 0.95
    print(f"[{'PASS' if success else 'FAIL'}] Otsu Cutout: Object fraction {obj_fraction:.3f}")
    
    # Region growing
    h, w = coin.shape[:2]
    # Assuming center pixel is part of the coin
    res = ivp.run_module(coin, "segment", {
        "method": "region_growing", "seed": [w//2, h//2], "delta_t": 20, 
        "adaptive": True, "connectivity": 4,
        "morph_open": True, "morph_close": True, "disk_radius": 3, 
        "fill_holes": True, "keep_largest": True, "foreground": "auto"
    })
    mask = res["extras"]["mask"]
    obj_fraction = (mask > 0).mean()
    success = 0.01 < obj_fraction < 0.95
    print(f"[{'PASS' if success else 'FAIL'}] Region Growing (Center): Object fraction {obj_fraction:.3f}")

    print("\n=== Auto & Guided Tests ===")
    sp_test = load_img("test_salt_pepper.jpg")
    auto_res = ivp.auto_enhance(sp_test)
    print(f"[{'PASS' if len(auto_res['steps']) > 0 else 'FAIL'}] Auto Fix ran {len(auto_res['steps'])} steps on real noisy image.")
    
    guided_res = ivp.run_guided(dark, "equalize")
    guided_img = guided_res["branch_a"]["image"]
    report("Guided Equalize", ivp.diagnose(dark), ivp.diagnose(guided_img), "std", "up")
    print("\nTests complete!")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=DATASET_DIR, help="folder holding the test photos")
    DATASET_DIR = ap.parse_args().dataset
    run_tests()
