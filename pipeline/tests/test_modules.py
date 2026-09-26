"""Module behaviour, parameter validation and engine agreement through the public API."""
import numpy as np
import pytest
from skimage import data

import ivp_pipeline as ivp
from ivp_pipeline import synthetic as S
from ivp_pipeline.errors import PipelineError

RGB = data.astronaut()
GRAY = data.camera()

CASES = [
    ("contrast", {"mode": "gamma", "gamma": 0.5}),
    ("contrast", {"mode": "gamma", "gamma": "auto"}),
    ("contrast", {"mode": "stretch"}),
    ("contrast", {"mode": "flatten"}),
    ("denoise", {"filter_type": "median", "kernel_size": 5}),
    ("denoise", {"filter_type": "box", "kernel_size": 3}),
    ("sharpen_edges", {"operation": "laplacian_boost", "A": 0}),
    ("sharpen_edges", {"operation": "laplacian_boost", "A": 1.0}),
    ("sharpen_edges", {"operation": "gradient", "operator": "roberts", "edge_threshold": 0.2}),
    ("segment", {"method": "otsu", "fill_holes": True, "keep_largest": True}),
    ("segment", {"method": "region_growing", "seed": [200, 300]}),
]


@pytest.mark.parametrize("module,params", CASES)
@pytest.mark.parametrize("img", [RGB, GRAY], ids=["rgb", "gray"])
def test_engines_agree_exactly(img, module, params):
    a = ivp.run_module(img, module, params, engine="opencv")
    b = ivp.run_module(img, module, params, engine="numpy")
    assert a["image"].dtype == np.uint8 and a["image"].shape == b["image"].shape
    assert np.array_equal(a["image"], b["image"])


@pytest.mark.parametrize("module,params", [
    ("denoise", {"filter_type": "gaussian", "sigma": 1.5}),
    ("sharpen_edges", {"operation": "unsharp", "k": 1.5}),
])
def test_engines_agree_within_rounding(module, params):
    # OpenCV's Gaussian uses fixed-point arithmetic on uint8: allow +-1 gray level.
    a = ivp.run_module(RGB, module, params, engine="opencv")["image"].astype(int)
    b = ivp.run_module(RGB, module, params, engine="numpy")["image"].astype(int)
    assert np.abs(a - b).max() <= 1


def test_hist_eq_engines_differ_only_by_normalisation():
    a = ivp.run_module(GRAY, "contrast", {"mode": "hist_eq"}, "opencv")
    b = ivp.run_module(GRAY, "contrast", {"mode": "hist_eq"}, "numpy")
    assert np.abs(a["image"].astype(int) - b["image"].astype(int)).mean() < 3
    assert len(b["extras"]["lut"]) == 256


def test_result_structure():
    r = ivp.run_module(RGB, "contrast", {"mode": "stretch"})
    assert set(r) >= {"input", "image", "extras", "histograms", "metrics", "warnings", "meta"}
    assert len(r["histograms"]["before"]) == 256 and len(r["histograms"]["after"]) == 256
    assert r["input"].shape == r["image"].shape
    assert r["meta"]["params"]["clip_percentile"] == 1.0   # defaults are filled in


def test_colour_preserved_by_luma_ops():
    r = ivp.run_module(RGB, "contrast", {"mode": "gamma", "gamma": 0.6})
    import cv2
    cr_in = cv2.cvtColor(RGB, cv2.COLOR_RGB2YCrCb)[:, :, 1:].astype(int)
    cr_out = cv2.cvtColor(r["image"], cv2.COLOR_RGB2YCrCb)[:, :, 1:].astype(int)
    assert np.abs(cr_in - cr_out).mean() < 1.5    # chroma essentially untouched


def test_high_boost_kernel_and_brightness():
    from ivp_pipeline.modules.sharpen_edges import high_boost_kernel
    for A in (0.0, 1.0, 2.0):
        k = high_boost_kernel(A)
        assert k[1, 1] == A + 8 and np.isclose(k.sum(), A)
    flat = np.full((32, 32), 60, np.uint8)
    out = ivp.run_module(flat, "sharpen_edges", {"operation": "laplacian_boost", "A": 2.0})["image"]
    assert np.all(out == 120)                     # kernel sum 2 doubles a flat image


def test_gradient_axis_convention():
    img = np.zeros((32, 32), np.uint8)
    img[:, 16:] = 200                             # a VERTICAL edge
    r = ivp.run_module(img, "sharpen_edges", {"operation": "gradient"})
    assert np.abs(r["extras"]["gx"]).max() > 0 and np.abs(r["extras"]["gy"]).max() == 0


def test_otsu_segments_synthetic_scene():
    img, gt = S.scene(3)
    from ivp_pipeline.metrics import iou
    r = ivp.run_module(img, "segment", {"method": "otsu"})
    assert iou(r["extras"]["mask"] > 0, gt) > 0.97
    assert r["extras"]["cutout"].shape[2] == 4


def test_region_growing_seed_is_robust_to_impulse():
    img, gt = S.scene(4)
    noisy = S.salt_pepper(img, 0.05, seed=1)
    noisy[9, 9], noisy[9, 10], noisy[8, 9] = 255, 0, 255   # the clicked pixel and neighbours are spikes
    r = ivp.run_module(noisy, "segment", {"method": "region_growing", "seed": [9, 9],
                                         "morph_open": False, "morph_close": False})
    assert r["metrics"]["foreground_fraction"] > 0.5   # grew over the background


def test_seed_scaled_for_downscaled_images():
    big = np.zeros((2048, 2048), np.uint8)
    big[1500:1900, 1500:1900] = 200
    r = ivp.run_module(big, "segment", {"method": "region_growing", "seed": [1700, 1700],
                                        "morph_open": False, "morph_close": False})
    assert r["meta"]["scale"] == 0.5 and r["image"].shape[:2] == (1024, 1024)
    assert 0.03 < r["metrics"]["foreground_fraction"] < 0.05   # the 400x400 square only


@pytest.mark.parametrize("module,params,code", [
    ("nope", {}, "UNKNOWN_MODULE"),
    ("denoise", {"kernel_size": 4}, "BAD_PARAM"),
    ("denoise", {"kernal_size": 3}, "BAD_PARAM"),
    ("contrast", {"gamma": 9}, "BAD_PARAM"),
    ("segment", {"delta_t": 2.5}, "BAD_PARAM"),
])
def test_bad_params(module, params, code):
    with pytest.raises(PipelineError) as e:
        ivp.run_module(GRAY, module, params)
    assert e.value.code == code


def test_string_params_from_form_data_are_coerced():
    r = ivp.run_module(GRAY, "denoise", {"filter_type": "median", "kernel_size": "5"})
    assert r["meta"]["params"]["kernel_size"] == 5
    r = ivp.run_module(GRAY, "segment", {"fill_holes": "true", "disk_radius": "2"})
    assert r["meta"]["params"]["fill_holes"] is True


@pytest.mark.parametrize("bad", [np.zeros((4, 4), np.uint8), np.zeros((20, 20), np.float32),
                                 np.zeros((20, 20, 2), np.uint8)])
def test_bad_images(bad):
    with pytest.raises(PipelineError):
        ivp.run_module(bad, "denoise")


def test_rgba_and_bad_engine():
    rgba = np.dstack([RGB, np.full(RGB.shape[:2], 255, np.uint8)])
    assert ivp.run_module(rgba, "denoise")["image"].shape == RGB.shape
    with pytest.raises(PipelineError):
        ivp.run_module(RGB, "denoise", engine="torch")


def test_list_modules_schema():
    s = ivp.list_modules()
    assert set(s) == {"contrast", "denoise", "sharpen_edges", "segment"}
    for mod in s.values():
        for p in mod["params"]:
            assert {"name", "type", "default", "description"} <= set(p)
