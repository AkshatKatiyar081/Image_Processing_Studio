"""Diagnostics (§9), quality gate + verify-after-fix (§10) and auto enhance (§11)."""
import numpy as np
import pytest
from skimage import data

import ivp_pipeline as ivp
from ivp_pipeline import synthetic as S
from ivp_pipeline.errors import PipelineError
from ivp_pipeline.metrics import iou

SCENE, GT = S.scene(1)
CAM = data.camera()


def codes(img):
    return {f["code"] for f in ivp.diagnose(img)["findings"]}


# ------------------------------------------------------------------ diagnostics
def test_clean_images_have_no_noise_findings():
    for img in (SCENE, CAM, data.coins(), data.astronaut()):
        assert not codes(img) & {"IMPULSE_NOISE", "GAUSSIAN_NOISE"}


@pytest.mark.parametrize("sigma", [10, 20])
def test_noise_estimate_tracks_truth(sigma):
    est = ivp.diagnose(S.gaussian_noise(CAM, sigma))["metrics"]["noise_sigma"]
    assert abs(est - sigma) < 0.2 * sigma + 1.5


def test_each_degradation_triggers_its_finding():
    assert "IMPULSE_NOISE" in codes(S.salt_pepper(CAM, 0.02))
    assert "GAUSSIAN_NOISE" in codes(S.gaussian_noise(CAM, 12))
    assert "UNEVEN_ILLUMINATION" in codes(S.illumination_ramp(SCENE, 40))
    assert "UNEVEN_ILLUMINATION" not in codes(SCENE)
    assert "UNDEREXPOSED" in codes(S.apply_gamma(CAM, 3.0))
    assert "LOW_CONTRAST" in codes(S.low_contrast(CAM))


# ------------------------------------------------------------------ guided
def test_clean_image_passes_gate():
    r = ivp.run_guided(SCENE, "segment")
    assert r["gate_passed"] and r["branch_b"] is None and r["verdict"]["result"] == "not_needed"


@pytest.mark.parametrize("degrade,rule", [
    (lambda im: S.salt_pepper(im, 0.05), "IMPULSE_NOISE"),
    (lambda im: S.gaussian_noise(im, 20), "GAUSSIAN_NOISE"),
    (lambda im: S.illumination_ramp(im, 40), "UNEVEN_ILLUMINATION"),
])
def test_segment_guided_fixes_each_defect(degrade, rule):
    r = ivp.run_guided(degrade(SCENE), "segment")
    assert not r["gate_passed"]
    assert [s["rule"] for s in r["branch_b"]["stages"]] == [rule]
    assert r["verdict"]["result"] == "recommended"
    iou_a = iou(r["branch_a"]["extras"]["mask"] > 0, GT)
    iou_b = iou(r["branch_b"]["extras"]["mask"] > 0, GT)
    assert iou_b > 0.97 and iou_b >= iou_a


def test_illumination_is_the_big_win():
    r = ivp.run_guided(S.illumination_ramp(SCENE, 40), "segment")
    assert iou(r["branch_a"]["extras"]["mask"] > 0, GT) < 0.5
    assert iou(r["branch_b"]["extras"]["mask"] > 0, GT) > 0.97


def test_chained_prerequisites_in_order():
    r = ivp.run_guided(S.salt_pepper(S.illumination_ramp(SCENE, 40), 0.05), "segment")
    assert [s["rule"] for s in r["branch_b"]["stages"]] == ["IMPULSE_NOISE", "UNEVEN_ILLUMINATION"]
    assert r["verdict"]["result"] == "recommended"


def test_linear_contrast_does_not_trigger_segmentation_fix():
    """The v1 flaw: a linear stretch cannot change an Otsu mask, so no rule should fire."""
    r = ivp.run_guided(S.low_contrast(SCENE, 100, 140), "segment")
    assert r["gate_passed"]


def test_false_alarm_is_reported_as_no_benefit():
    r = ivp.run_guided(CAM, "segment")      # natural scene: U_bg is high but it is not lighting
    assert r["verdict"]["result"] == "no_benefit"


@pytest.mark.parametrize("intent,img,params", [
    ("edges", S.gaussian_noise(CAM, 25), None),
    ("edges", S.gaussian_noise(CAM, 15), {"operator": "roberts"}),
    ("edges", S.salt_pepper(CAM, 0.02), None),
    ("sharpen", S.gaussian_noise(CAM, 12), None),
    ("equalize", S.gaussian_noise(S.apply_gamma(CAM, 3.0), 12), None),
])
def test_other_intents_recommended(intent, img, params):
    r = ivp.run_guided(img, intent, params)
    assert not r["gate_passed"] and r["verdict"]["result"] == "recommended"


def test_sobel_tolerates_mild_noise_but_roberts_does_not():
    """Calibrated gate: Sobel's built-in smoothing copes with σ≈13, Roberts (2x2) does not."""
    noisy = S.gaussian_noise(CAM, 13)
    assert ivp.run_guided(noisy, "edges", {"operator": "sobel"})["gate_passed"]
    assert not ivp.run_guided(noisy, "edges", {"operator": "roberts"})["gate_passed"]


def test_same_target_params_in_both_branches():
    r = ivp.run_guided(S.gaussian_noise(CAM, 25), "edges", {"operator": "prewitt", "edge_threshold": 0.3})
    assert r["branch_a"]["meta"]["params"] == r["branch_b"]["meta"]["params"]


def test_bad_intent_and_params():
    with pytest.raises(PipelineError):
        ivp.run_guided(CAM, "blur")
    with pytest.raises(PipelineError):
        ivp.run_guided(CAM, "sharpen", {"operation": "gradient"})


# ------------------------------------------------------------------ auto
def test_auto_enhance_denoises_and_brightens():
    clean = S.apply_gamma(CAM, 2.5)
    noisy = S.salt_pepper(clean, 0.02)
    r = ivp.auto_enhance(noisy)
    modules = [s["module"] for s in r["steps"]]
    assert modules[0] == "denoise" and "contrast" in modules
    assert "sharpen_edges" not in modules          # never sharpen after finding noise
    assert r["diagnosis_after"]["metrics"]["impulse_density"] < 0.001
    assert r["diagnosis_after"]["metrics"]["mean"] > r["diagnosis_before"]["metrics"]["mean"]


def test_auto_enhance_clean_image_only_sharpens():
    r = ivp.auto_enhance(CAM)
    assert [s["module"] for s in r["steps"]] == ["sharpen_edges"]
