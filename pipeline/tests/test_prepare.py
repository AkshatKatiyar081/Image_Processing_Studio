"""Phase 1 preparation: impulse-aware downscaling (salt & pepper must survive the resize)."""
import cv2
import numpy as np
import pytest
from skimage import data

import ivp_pipeline as ivp
from ivp_pipeline import config, metrics
from ivp_pipeline import synthetic as S
from ivp_pipeline.utils import prepare, prepare_with_info


def codes(res):
    return {f["code"] for f in res["findings"]}


@pytest.fixture(scope="module")
def big_specks():
    """A 2048 x 2048 synthetic scene with 5 % salt & pepper."""
    scene = cv2.resize(S.scene(7)[0], (2048, 2048), interpolation=cv2.INTER_NEAREST)
    return S.salt_pepper(scene, 0.05, seed=1)


def test_large_specks_are_diagnosed(big_specks):
    d = ivp.diagnose(big_specks)
    assert "IMPULSE_NOISE" in codes(d)
    assert d["meta"]["resize_interp"] == "nearest"
    assert d["meta"]["impulse_density_full_res"] > config.IMPULSE_DENSITY


def test_large_specks_guided_segment_starts_with_median(big_specks):
    g = ivp.run_guided(big_specks, "segment")
    assert g["branch_b"]["stages"][0]["name"] == "denoise:median"
    assert g["meta"]["resize_interp"] == "nearest"


def test_large_specks_auto_applies_median_first(big_specks):
    r = ivp.auto_enhance(big_specks)
    first = r["steps"][0]
    assert first["module"] == "denoise" and first["params"]["filter_type"] == "median"
    assert r["meta"]["resize_interp"] == "nearest"


def test_large_specks_run_module_meta(big_specks):
    r = ivp.run_module(big_specks, "denoise", {"filter_type": "median"})
    assert r["meta"]["resize_interp"] == "nearest" and r["meta"]["scale"] == 0.5


def test_clean_large_photo_uses_area():
    photo = cv2.resize(data.astronaut(), (2048, 2048))
    d = ivp.diagnose(photo)
    assert d["meta"]["resize_interp"] == "area"
    assert "IMPULSE_NOISE" not in codes(d)


def test_small_images_are_untouched():
    img = data.camera()                          # 512 x 512
    out, scale, info = prepare_with_info(img)
    assert scale == 1.0 and info["resize_interp"] == "none"
    assert out is img or np.array_equal(out, img)
    assert ivp.diagnose(img)["meta"]["resize_interp"] == "none"


def test_prepare_still_returns_two_values():
    out, scale = prepare(np.zeros((3000, 1500), np.uint8))
    assert scale == pytest.approx(1024 / 3000) and max(out.shape) == 1024


def _impulse_density_reference(gray):
    """The original int16 formula, kept here to prove the low-memory rewrite is identical."""
    med = cv2.medianBlur(gray, 3)
    extreme = (gray <= config.EXTREME_LOW) | (gray >= config.EXTREME_HIGH)
    outlier = np.abs(gray.astype(np.int16) - med.astype(np.int16)) > config.IMPULSE_MEDIAN_DIFF
    return float((extreme & outlier).mean())


@pytest.mark.parametrize("img", [
    data.camera(),
    S.salt_pepper(data.camera(), 0.05, seed=2),
    S.salt_pepper(data.moon(), 0.2, seed=3),
    S.scene(3)[0],
    np.zeros((64, 64), np.uint8),
])
def test_impulse_density_low_memory_formula_unchanged(img):
    assert metrics.impulse_density(img) == _impulse_density_reference(img)


def test_max_side_zero_keeps_full_resolution():
    big = np.zeros((3000, 2000, 3), np.uint8)
    out, scale, info = prepare_with_info(big, max_side=0)
    assert out.shape == big.shape and scale == 1.0 and info["resize_interp"] == "none"


def test_library_default_is_full_resolution():
    """Without IVP_MAX_SIDE the pipeline processes the image at its own size (local use)."""
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if k != "IVP_MAX_SIDE"}
    code = "import ivp_pipeline.config as c; print(c.MAX_SIDE)"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "0"
