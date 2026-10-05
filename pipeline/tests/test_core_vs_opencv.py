"""Spec §13.4 — every from-scratch algorithm is validated against a reference library."""
import cv2
import numpy as np
import pytest
from skimage import data
from skimage.segmentation import flood

from ivp_pipeline.core import filtering as F
from ivp_pipeline.core import histogram as Hs
from ivp_pipeline.core import morphology as M
from ivp_pipeline.core import region as R

IMAGES = {"camera": data.camera(), "coins": data.coins(), "moon": data.moon()}


@pytest.fixture(params=list(IMAGES))
def gray(request):
    return IMAGES[request.param]


def test_otsu_matches_opencv(gray):
    t_cv, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res = Hs.otsu(gray)
    assert abs(res["threshold"] - int(t_cv)) <= 1
    assert 0.0 < res["eta"] <= 1.0


def test_otsu_constant_image():
    res = Hs.otsu(np.full((20, 20), 77, np.uint8))
    assert res["threshold"] == 77 and res["eta"] == 0.0


def test_hist_eq_textbook_formula(gray):
    out, lut = Hs.equalize(gray)
    n = np.bincount(gray.ravel(), minlength=256)
    expected = np.rint(255 * np.cumsum(n) / gray.size).astype(np.uint8)
    assert np.array_equal(lut, expected)
    assert np.array_equal(out, expected[gray])
    # OpenCV uses the (cdf - cdf_min) variant: close but not identical.
    assert np.abs(out.astype(int) - cv2.equalizeHist(gray).astype(int)).mean() < 3


@pytest.mark.parametrize("k", [3, 5, 7])
def test_median_matches_opencv(gray, k):
    assert np.array_equal(F.median_filter(gray, k), cv2.medianBlur(gray, k))


KERNELS = {
    "box3": F.box_kernel(3),
    "box7": F.box_kernel(7),
    "laplacian_boost": np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]], np.float32),
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
    "prewitt_y": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32),
    "roberts_1": np.array([[-1, 0], [0, 1]], np.float32),   # even-sized: tests anchor handling
    "roberts_2": np.array([[0, -1], [1, 0]], np.float32),
}


@pytest.mark.parametrize("name", list(KERNELS))
def test_correlation_matches_filter2d(gray, name):
    k = KERNELS[name]
    ref = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, k)
    assert np.abs(F.correlate2d(gray, k) - ref).max() < 1e-3


@pytest.mark.parametrize("sigma", [0.8, 1.0, 2.0])
def test_gaussian_matches_opencv(gray, sigma):
    k = F.gaussian_ksize(sigma)
    ref = cv2.GaussianBlur(gray.astype(np.float32), (k, k), sigma)
    assert np.abs(F.gaussian_filter(gray, sigma) - ref).max() < 1e-2


@pytest.mark.parametrize("r", [0, 1, 3, 7, 15])
def test_disk_matches_opencv(r):
    ref = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    assert np.array_equal(M.disk(r), ref)


@pytest.mark.parametrize("r", [1, 3, 6])
def test_morphology_matches_opencv(gray, r):
    se = M.disk(r)
    assert np.array_equal(M.erode(gray, se), cv2.erode(gray, se))
    assert np.array_equal(M.dilate(gray, se), cv2.dilate(gray, se))
    assert np.array_equal(M.opening(gray, se), cv2.morphologyEx(gray, cv2.MORPH_OPEN, se))
    assert np.array_equal(M.closing(gray, se), cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se))


def test_morphology_binary_mask():
    mask = (data.coins() > 107).astype(np.uint8) * 255
    se = M.disk(2)
    assert np.array_equal(M.opening(mask, se), cv2.morphologyEx(mask, cv2.MORPH_OPEN, se))


def test_fill_holes():
    mask = np.zeros((50, 50), bool)
    mask[10:40, 10:40] = True
    mask[20:30, 20:30] = False          # interior hole -> filled
    mask[0:5, 45:50] = False
    filled = M.fill_holes(mask)
    assert filled[10:40, 10:40].all()
    assert not filled[:5, :10].any()    # outside background untouched


def diamond_ring(n=11):
    """1-px diamond outline whose pixels touch only diagonally (8-connected, not 4-connected)."""
    m = np.zeros((n, n), bool)
    c = n // 2
    for i in range(c):
        m[c - i, 1 + i] = m[1 + i, c + i] = m[c + i, n - 2 - i] = m[n - 2 - i, c - i] = True
    return m


def test_fill_holes_complementary_connectivity():
    """An 8-connected object needs a 4-connected background flood, or the hole leaks out diagonally."""
    ring = diamond_ring()
    c = ring.shape[0] // 2
    assert not ring[c, c]
    assert M.fill_holes(ring, connectivity=8)[c, c]       # object 8-connected -> hole filled
    assert M.fill_holes(ring)[c, c]                        # default treats the object as 8-connected
    assert not M.fill_holes(ring, connectivity=4)[c, c]   # object 4-connected -> not a closed ring


@pytest.mark.parametrize("seed_rc,tol", [((150, 200), 20), ((20, 20), 10), ((256, 256), 30)])
def test_region_grow_matches_skimage_flood(seed_rc, tol):
    g = data.camera()
    ref = flood(g, seed_rc, tolerance=tol, connectivity=1)
    ours = R.region_grow(g, (seed_rc[1], seed_rc[0]), tol, connectivity=4, seed_mode="pixel")
    assert np.array_equal(ours, ref)


def test_region_grow_adaptive_runs_and_contains_seed():
    g = data.camera()
    m = R.region_grow(g, (200, 150), 12, adaptive=True)
    assert m[150, 200] and m.sum() > 1


def test_region_grow_adaptive_retests_rejected_pixels():
    """A pixel rejected against the early region mean must be accepted once the mean moves closer.

        10   8        seed = 0 (bottom-left), delta = 6, 4-connectivity
         0   5        10 is rejected first (|10 - 0| > 6), then 5 and 8 join (mean 4.33),
                      and 10 is re-tested from 8: |10 - 4.33| <= 6 -> accepted.
    """
    g = np.array([[10, 8], [0, 5]], np.uint8)
    m = R.region_grow(g, (0, 1), 6, connectivity=4, adaptive=True, seed_mode="pixel")
    assert m.all()
