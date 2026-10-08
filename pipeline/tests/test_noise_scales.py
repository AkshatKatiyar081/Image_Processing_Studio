"""Enlarged noise (screenshots of noisy images): detected at half scale, fixed with larger filters."""
import cv2
import numpy as np
import pytest
from skimage import data

import ivp_pipeline as ivp
from ivp_pipeline import config as C
from ivp_pipeline import metrics as Q
from ivp_pipeline import synthetic as S
from ivp_pipeline.diagnostics import diagnose_prepared, gaussian_noise, impulse_noise


def screenshot(img, factor=1.5):
    """A noisy web image as a screenshot: saved by the site as JPEG 75, then shown enlarged."""
    web = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1], cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    return cv2.resize(web, (round(w * factor), round(h * factor)), interpolation=cv2.INTER_LINEAR)


def codes(img):
    return {f["code"]: f.get("scale") for f in ivp.diagnose(img)["findings"]}


@pytest.mark.parametrize("factor", [1.5, 2.0])
def test_enlarged_salt_and_pepper_is_found(factor):
    shot = screenshot(S.salt_pepper(data.camera(), 0.10, seed=2), factor)
    m = diagnose_prepared(shot)["metrics"]
    assert m["impulse_density"] < C.IMPULSE_DENSITY           # the pixel-scale test misses it ...
    assert codes(shot).get("IMPULSE_NOISE") == "enlarged"      # ... the half-scale test does not
    assert impulse_noise(m)["median_ksize"] in (5, 7)


def test_enlarged_grain_is_found():
    shot = screenshot(S.gaussian_noise(data.camera(), 25, seed=1), 1.5)
    m = diagnose_prepared(shot)["metrics"]
    assert m["noise_sigma"] < C.NOISE_MODERATE                 # smooth between neighbouring pixels
    assert codes(shot).get("GAUSSIAN_NOISE") == "enlarged"
    assert gaussian_noise(m)["sigma"] == C.COARSE_GAUSS_SIGMA


@pytest.mark.parametrize("name", ["camera", "astronaut", "coffee", "page", "chelsea", "moon"])
@pytest.mark.parametrize("factor", [1.0, 1.5, 2.0])
def test_clean_images_and_clean_screenshots_have_no_noise_finding(name, factor):
    img = getattr(data, name)()
    shot = img if factor == 1.0 else screenshot(img, factor)
    found = codes(shot)
    assert "IMPULSE_NOISE" not in found and "GAUSSIAN_NOISE" not in found, found


def test_pixel_scale_findings_are_unchanged():
    assert codes(S.salt_pepper(data.camera(), 0.05, seed=1))["IMPULSE_NOISE"] == "pixel"
    assert codes(S.gaussian_noise(data.camera(), 20, seed=1))["GAUSSIAN_NOISE"] == "pixel"


def test_speck_density_counts_softened_specks_but_not_lines():
    flat = np.full((200, 200), 100, np.uint8)
    lines = flat.copy()
    lines[::10, :] = 220                                      # long thin strokes, like text
    assert Q.speck_density(lines) == 0.0
    specks = flat.copy()
    rng = np.random.default_rng(0)
    ys, xs = rng.integers(2, 198, 300), rng.integers(2, 198, 300)
    specks[ys, xs] = 190                                       # softened (not pure white) single-pixel specks
    assert Q.speck_density(specks) > 0.005
    assert Q.impulse_density(specks) == 0.0                    # the strict 0/255 test does not see them


def test_auto_fix_removes_enlarged_specks_with_a_bigger_median_and_does_not_sharpen():
    shot = screenshot(S.salt_pepper(data.camera(), 0.10, seed=2), 1.5)
    r = ivp.auto_enhance(shot)
    med = [s for s in r["steps"] if s["params"].get("filter_type") == "median"]
    assert med and med[0]["params"]["kernel_size"] >= 5 and "enlarged" in med[0]["reason"]
    assert not any(s["module"] == "sharpen_edges" for s in r["steps"])
    after = r["diagnosis_after"]["metrics"]
    assert after["speck_density_coarse"] < C.SPECK_COARSE_DENSITY / 4
    assert "IMPULSE_NOISE" not in {f["code"] for f in r["diagnosis_after"]["findings"]}


def test_auto_fix_smooths_enlarged_grain_with_sigma_1_5():
    shot = screenshot(S.gaussian_noise(data.camera(), 25, seed=1), 1.5)
    r = ivp.auto_enhance(shot)
    g = [s for s in r["steps"] if s["params"].get("filter_type") == "gaussian"]
    assert g and g[0]["params"]["sigma"] == C.COARSE_GAUSS_SIGMA and "coarse grain" in g[0]["reason"]
    assert not any(s["module"] == "sharpen_edges" for s in r["steps"])
    assert "GAUSSIAN_NOISE" not in {f["code"] for f in r["diagnosis_after"]["findings"]}


def test_guided_mode_uses_the_bigger_median_for_enlarged_specks():
    scene, gt = S.scene(3)
    shot = screenshot(S.salt_pepper(scene, 0.10, seed=2), 1.5)
    res = ivp.run_guided(shot, "segment")
    first = res["branch_b"]["stages"][0]
    assert first["rule"] == "IMPULSE_NOISE" and first["params"]["kernel_size"] >= 5
    assert res["verdict"]["result"] in ("recommended", "mixed")
