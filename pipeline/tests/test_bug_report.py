"""Close-out of the earlier bug report (Phase F).

Each test would FAIL if the reported bug were real. Row numbers refer to the table in CHANGES.md.
"""
import cv2
import numpy as np
import pytest
from skimage import data

import ivp_pipeline as ivp
from ivp_pipeline import synthetic as S
from ivp_pipeline.metrics import iou

CAM = data.camera()


def test_row3_auto_exposure_uses_post_fix_metrics():
    """Claim: Auto Fix decides exposure/contrast from pre-fix metrics. The gamma step's reason must
    report the mean measured AFTER the median filter, not the speck-inflated mean before it."""
    noisy = S.salt_pepper(S.apply_gamma(CAM, 2.5), 0.05, seed=3)
    r = ivp.auto_enhance(noisy)
    steps = r["steps"]
    assert steps[0]["module"] == "denoise" and steps[0]["params"]["filter_type"] == "median"
    gamma = next(s for s in steps if s["module"] == "contrast" and s["params"]["mode"] == "gamma")
    after_median = ivp.run_module(noisy, "denoise", steps[0]["params"])["image"]
    mean_after = ivp.diagnose(after_median)["metrics"]["mean"]
    mean_before = r["diagnosis_before"]["metrics"]["mean"]
    assert f"mean {mean_after:.2f}" in gamma["reason"]
    assert f"{mean_after:.2f}" != f"{mean_before:.2f}"      # the two really differ on this image


def test_row5_flatten_radius_cap_handles_large_objects():
    """Claim: the flatten SE radius is capped at 75 px, too small for large objects on a 1024 px image.
    Verify: dark disks of radius 40–60 px on a ramped 1024 x 1024 background still segment with IoU >= 0.95."""
    rng = np.random.default_rng(5)
    mask = np.zeros((1024, 1024), np.uint8)
    for _ in range(12):
        cx, cy, r = (int(v) for v in (rng.integers(80, 944), rng.integers(80, 944), rng.integers(40, 61)))
        cv2.circle(mask, (cx, cy), r, 1, -1)
    gt = mask.astype(bool)
    img = np.where(gt, 90, 170).astype(np.uint8)
    img = S.illumination_ramp(img, 40)
    g = ivp.run_guided(img, "segment")
    assert g["branch_b"] is not None
    assert g["branch_b"]["stages"][-1]["params"]["se_radius"] == 75        # the cap is in effect
    assert iou(g["branch_b"]["extras"]["mask"] > 0, gt) >= 0.95


def test_row10_constant_image_edge_warning():
    flat = np.full((64, 64), 128, np.uint8)
    r = ivp.run_module(flat, "sharpen_edges", {"operation": "gradient"})
    assert not r["image"].any() and any("zero" in w for w in r["warnings"])
    r = ivp.run_module(flat, "sharpen_edges", {"operation": "laplacian_boost", "A": 0})
    assert not r["image"].any() and any("zero" in w for w in r["warnings"])


@pytest.mark.parametrize("engine", ["opencv", "numpy"])
def test_row11_seed_on_a_spike_still_grows(engine):
    """Claim: OpenCV region growing returns a 1-px mask when the clicked pixel is a speck.
    The seed value is the 3x3 median (the background level), and the seed is forced into the region."""
    img = np.full((64, 64), 100, np.uint8)
    img[30, 30] = 255                                     # the user clicks exactly on a white speck
    r = ivp.run_module(img, "segment", {"method": "region_growing", "seed": [30, 30],
                                        "morph_open": False, "morph_close": False}, engine=engine)
    assert r["metrics"]["foreground_fraction"] > 0.99
