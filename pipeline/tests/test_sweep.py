"""Degradation sweeps (accuracy vs defect strength, Direct vs Guided) and guided difference maps."""
import time

import numpy as np
import pytest
from skimage import data

import ivp_pipeline as ivp
from ivp_pipeline import synthetic as S
from ivp_pipeline.errors import PipelineError
from ivp_pipeline.guided import ONLY_A_COLOR, ONLY_B_COLOR

CAM = data.camera()
RAMP = S.illumination_ramp(S.scene(1)[0], 40)
NOISY_CAM = S.gaussian_noise(CAM, 25, seed=4)


# ------------------------------------------------------------------ sweeps
def test_segment_sweep_uses_illumination_axis_and_guided_wins():
    r = ivp.degradation_sweep(RAMP, "segment")
    assert r["axis"]["name"] == "illumination" and r["metric"] == "IoU"
    assert [p["level"] for p in r["points"]] == [0, 20, 40, 70]
    for p in r["points"]:
        assert p["guided"] >= p["direct"]
    strongest = r["points"][-1]
    assert strongest["guided"] - strongest["direct"] > 0.3


def test_edges_sweep_on_noisy_camera_guided_wins_at_high_noise():
    r = ivp.degradation_sweep(NOISY_CAM, "edges", {"operator": "sobel"})
    assert r["axis"]["name"] == "gaussian" and r["metric"] == "Edge F1"
    high = [p for p in r["points"] if p["x"] >= 20]
    assert high and all(p["guided"] > p["direct"] for p in high)


def test_edges_sweep_on_clean_camera_lines_coincide_at_zero():
    r = ivp.degradation_sweep(CAM, "edges", {"operator": "sobel"})
    first = r["points"][0]
    assert not first["gate_fired"] and abs(first["guided"] - first["direct"]) < 0.02
    assert all(p["guided"] > p["direct"] for p in r["points"] if p["level"] >= 20)


def test_operator_changes_the_sweep():
    sobel = ivp.degradation_sweep(NOISY_CAM, "edges", {"operator": "sobel"})
    roberts = ivp.degradation_sweep(NOISY_CAM, "edges", {"operator": "roberts"})
    assert [p["direct"] for p in sobel["points"]] != [p["direct"] for p in roberts["points"]]


def test_impulse_axis_for_speckled_scene():
    r = ivp.degradation_sweep(S.salt_pepper(S.scene(7)[0], 0.05, seed=2), "segment")
    assert r["axis"]["name"] == "impulse" and r["axis"]["unit"] == "%"


@pytest.mark.parametrize("img,intent,key,factor", [
    (RAMP, "segment", "illumination_u", 1),
    (NOISY_CAM, "edges", "noise_sigma", 1),
    (S.salt_pepper(S.scene(7)[0], 0.05, seed=2), "segment", "impulse_density", 100),
])
def test_first_point_is_the_users_image(img, intent, key, factor):
    r = ivp.degradation_sweep(img, intent)
    assert r["user_point_index"] == 0 and r["points"][0]["level"] == 0
    assert r["points"][0]["x"] == pytest.approx(ivp.diagnose(img)["metrics"][key] * factor, rel=0.02, abs=1e-3)


def test_large_image_sweep_keeps_noise_level():
    import cv2
    img = S.gaussian_noise(cv2.resize(CAM, (1024, 1024), interpolation=cv2.INTER_CUBIC), 15, seed=1)
    r = ivp.degradation_sweep(img, "edges")
    assert r["sweep_scale"] == 0.5
    assert r["points"][0]["x"] == pytest.approx(ivp.diagnose(img)["metrics"]["noise_sigma"], rel=0.1)


def test_sweep_rejects_other_intents():
    with pytest.raises(PipelineError) as e:
        ivp.degradation_sweep(CAM, "sharpen")
    assert e.value.code == "UNKNOWN_INTENT"


@pytest.mark.slow
def test_sweep_runtime_512():
    t0 = time.perf_counter()
    ivp.degradation_sweep(NOISY_CAM, "edges")
    ivp.degradation_sweep(RAMP, "segment")
    assert time.perf_counter() - t0 < 5.0


# ------------------------------------------------------------------ difference maps
def _colour_count(img, colour):
    return int(np.all(img == np.array(colour, np.uint8), axis=-1).sum())


@pytest.mark.parametrize("img,intent,params", [
    (NOISY_CAM, "edges", None),
    (NOISY_CAM, "edges", {"operator": "roberts", "edge_threshold": 0.3}),
    (RAMP, "segment", None),
    (S.salt_pepper(S.scene(7)[0], 0.05, seed=2), "segment", None),
])
def test_difference_map_matches_counts(img, intent, params):
    r = ivp.run_guided(img, intent, params)
    assert r["branch_b"] is not None
    d = r["difference_map"]
    assert d.shape == img.shape[:2] + (3,) and d.dtype == np.uint8
    c = r["comparison"]
    assert _colour_count(d, ONLY_A_COLOR) == c["only_a_pixels"]
    assert _colour_count(d, ONLY_B_COLOR) == c["only_b_pixels"]
    assert r["difference_legend"]
    if intent == "edges":
        assert c["false_edge_density_b"] < c["false_edge_density_a"]
        assert {"edge_pixels_a", "edge_pixels_b"} <= set(c)
    else:
        assert {"fragmentation_a", "fragmentation_b", "eta_a", "eta_b", "n_components_a", "n_components_b"} <= set(c)


def test_no_difference_map_without_branch_b():
    r = ivp.run_guided(S.scene(1)[0], "segment")
    assert r["branch_b"] is None and "difference_map" not in r
