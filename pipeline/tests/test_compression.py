"""Transform coding: DCT and Walsh–Hadamard from scratch, the compress module, DCT-vs-Hadamard comparison."""
import math

import cv2
import numpy as np
import pytest
import scipy.fft as sfft
from skimage import data

import ivp_pipeline as ivp
from ivp_pipeline.core import transforms as T
from ivp_pipeline.errors import PipelineError

CAM = data.camera()
ASTRO = data.astronaut()
COFFEE = data.coffee()
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ DCT
@pytest.mark.parametrize("f", [
    RNG.random((8, 8)) * 255, RNG.random((7, 13)) * 255, RNG.random((64, 33)) * 255,
    CAM.astype(float), CAM[:301, :457].astype(float),
], ids=["8x8", "7x13", "64x33", "camera", "camera_odd"])
def test_dct2_matches_scipy_ortho(f):
    F = T.dct2(f)
    assert np.abs(F - sfft.dctn(f, norm="ortho")).max() < 1e-8 * max(1.0, np.abs(f).max())
    assert np.abs(T.idct2(F) - sfft.idctn(F, norm="ortho")).max() < 1e-8 * max(1.0, np.abs(f).max())
    assert np.abs(T.idct2(F) - f).max() < 1e-8


def test_dct_matrix_is_orthonormal():
    C = T.dct_matrix(16)
    assert np.allclose(C @ C.T, np.eye(16))


@pytest.mark.parametrize("shape", [(64, 64), (64, 96), (63, 95)])
def test_opencv_dct_matches_numpy(shape):
    from ivp_pipeline.modules.compress import _dct_forward, _dct_inverse
    f = RNG.random(shape) * 255
    a, b = _dct_forward(f, "opencv"), _dct_forward(f, "numpy")
    if a.shape == b.shape:          # OpenCV 5 does odd sizes; OpenCV 4 pads odd sides by 1
        assert np.abs(a - b).max() < 0.05
    assert np.abs(_dct_inverse(a, shape, "opencv") - f).max() < 0.01


# ------------------------------------------------------------------ Walsh–Hadamard
@pytest.mark.parametrize("n", [1, 2, 4, 16, 64, 256])
def test_walsh_matrix_orthonormal_and_sequency_ordered(n):
    W = T.walsh_matrix(n)
    assert np.allclose(W @ W.T, np.eye(n))
    sign_changes = [int(np.count_nonzero(np.diff(np.sign(W[k])))) for k in range(n)]
    assert sign_changes == list(range(n))          # row k has exactly k sign changes


@pytest.mark.parametrize("n", [2, 8, 64, 512])
def test_fast_butterfly_equals_matrix(n):
    W = T.walsh_matrix(n)
    x = RNG.random((5, n))
    assert np.allclose(T.fwht(x, axis=1), x @ W.T)
    assert np.allclose(T.ifwht(T.fwht(x, axis=1), axis=1), x)
    assert np.allclose(T.fwht(x.T, axis=0), W @ x.T)


def test_fwht2_matrix_and_fast_paths_agree():
    f = RNG.random((128, 256))
    assert np.allclose(T.fwht2(f), T.walsh_matrix(128) @ f @ T.walsh_matrix(256).T)
    assert np.allclose(T.ifwht2(T.fwht2(f)), f)
    small = RNG.random((8, 32))
    assert np.allclose(T.ifwht2(T.fwht2(small)), small)


def test_matlab_fwht_relation():
    """MATLAB fwht(x) (sequency order, forward scaled by 1/N) = our orthonormal fwht / √N."""
    x = np.array([19.0, -1, 11, -9, -7, 13, -15, 5])
    matlab = np.array([2.0, 3, 0, 4, 0, 0, 10, 0])     # MATLAB documentation example: fwht(x)
    assert np.allclose(T.fwht(x) / math.sqrt(8), matlab)
    # independent construction: natural Hadamard rows sorted by their number of sign changes
    h = np.ones((1, 1))
    while h.shape[0] < 8:
        h = np.block([[h, h], [h, -h]])
    seq = h[np.argsort([np.count_nonzero(np.diff(np.sign(r))) for r in h])]
    assert np.allclose(seq @ x / 8, matlab)


def test_parseval_for_both_transforms():
    f = CAM[:256, :256].astype(float)
    e = np.sum(f ** 2)
    assert np.isclose(np.sum(T.dct2(f) ** 2), e) and np.isclose(np.sum(T.fwht2(f) ** 2), e)


def test_pad_to_pow2_pads_each_dimension_without_resizing():
    img = RNG.integers(0, 255, (301, 457), dtype=np.uint8)
    p, (ph, pw) = T.pad_to_pow2(img)
    assert p.shape == (512, 512) and (ph, pw) == (211, 55)
    assert np.array_equal(p[:301, :457], img)
    assert T.pad_to_pow2(np.zeros((64, 128)))[0].shape == (64, 128)
    assert [T.next_pow2(n) for n in (1, 2, 3, 5, 1024, 1025)] == [1, 2, 4, 8, 1024, 2048]


# ------------------------------------------------------------------ coefficient selection
def test_keep_percent_exact_count_and_dc():
    F = RNG.normal(0, 10, (40, 50))
    F[0, 0] = 0.0                                  # smallest magnitude, still kept
    for p in (0.1, 1, 10, 33.3, 100):
        G, mask = T.keep_percent(F, p)
        assert mask.sum() == math.ceil(p / 100 * F.size) and mask[0, 0]
        assert np.array_equal(G[mask], F[mask]) and not G[~mask].any()


def test_keep_percent_ties_are_deterministic():
    F = np.ones((10, 10))
    _, mask = T.keep_percent(F, 5)                 # 5 of 100 equal values -> first 5 in row-major order
    assert np.flatnonzero(mask).tolist() == [0, 1, 2, 3, 4]


def test_threshold_rule():
    """Threshold coding on a gray image: dct2, zero |c| < 500, idct2, then clip/round (no rescaling)."""
    F = T.dct2(CAM.astype(float))
    F[np.abs(F) < 500] = 0
    expected = np.clip(np.rint(T.idct2(F)), 0, 255).astype(np.uint8)
    out = ivp.run_module(CAM, "compress", {"method": "dct", "mode": "threshold", "threshold": 500},
                         engine="numpy")["image"]
    assert np.array_equal(out, expected)


def test_zonal_square_and_l_shapes():
    F = np.ones((64, 96))
    _, sq = T.zonal(F, 0.25, "square")                       # k = 16
    _, lshape = T.zonal(F, 0.25, "l_shape")
    assert sq.sum() == 16 * 16 and sq[:16, :16].all()
    assert lshape.sum() == 64 * 96 - (64 - 16) * (96 - 16)     # only the bottom-right block is zeroed
    assert lshape[0, 95] and lshape[63, 0] and not lshape[16, 16]


def test_energy_compaction_curve():
    f = CAM.astype(float)
    q = np.logspace(-3, 0, 40)
    d, h = T.energy_compaction(T.dct2(f), q), T.energy_compaction(T.fwht2(f), q)
    for c in (d, h):
        assert np.all(np.diff(c) >= -1e-12) and np.isclose(c[-1], 1.0)
    i5 = int(np.argmin(np.abs(q - 0.05)))
    assert d[i5] >= h[i5]


# ------------------------------------------------------------------ the compress module
def test_perfect_reconstruction_at_100_percent():
    for m in ("dct", "hadamard"):
        for img in (CAM, ASTRO[:301, :457]):
            out = ivp.run_module(img, "compress", {"method": m, "keep_percent": 100})["image"]
            assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def _hue_shift(a, b):
    """Mean circular hue difference (OpenCV hue units, 0–180) over well-saturated pixels."""
    ha, hb = cv2.cvtColor(a, cv2.COLOR_RGB2HSV), cv2.cvtColor(b, cv2.COLOR_RGB2HSV)
    m = (ha[..., 1] > 60) & (ha[..., 2] > 40)
    d = np.abs(ha[..., 0].astype(int) - hb[..., 0].astype(int))
    return float(np.minimum(d, 180 - d)[m].mean())


def test_colours_are_preserved():
    out = ivp.run_module(ASTRO, "compress", {"method": "dct", "keep_percent": 10})["image"]
    for c in range(3):
        assert abs(float(out[..., c].mean()) - float(ASTRO[..., c].mean())) < 2
    assert _hue_shift(ASTRO, out) < 2.5


def test_per_channel_rescale_shifts_colours():
    """Documents why the reconstruction is NOT rescaled per channel: that moves the channel means."""
    chans = []
    for c in range(3):
        F = T.dct2(ASTRO[..., c].astype(float))
        F[np.abs(F) < 500] = 0
        x = T.idct2(F)
        chans.append(np.clip(np.rint((x - x.min()) / (x.max() - x.min()) * 255), 0, 255).astype(np.uint8))
    buggy = np.dstack(chans)
    shifts = [abs(float(buggy[..., c].mean()) - float(ASTRO[..., c].mean())) for c in range(3)]
    assert max(shifts) > 10
    assert _hue_shift(ASTRO, buggy) > _hue_shift(
        ASTRO, ivp.run_module(ASTRO, "compress", {"mode": "threshold", "threshold": 500})["image"])


@pytest.mark.parametrize("img", [CAM, ASTRO, CAM[:301, :457], ASTRO[:301, :457]],
                         ids=["gray", "rgb", "gray_odd", "rgb_odd"])
@pytest.mark.parametrize("method", ["dct", "hadamard"])
def test_output_shape_and_type(img, method):
    r = ivp.run_module(img, "compress", {"method": method, "keep_percent": 5})
    assert r["image"].shape == img.shape and r["image"].dtype == np.uint8
    ex = r["extras"]
    assert ex["error_map"].shape == img.shape[:2] + (3,)
    assert ex["coef_before"].shape == ex["coef_after"].shape == ex["kept_mask"].shape
    if method == "hadamard" and img.shape[:2] == (301, 457):
        assert ex["coef_before"].shape == (512, 512)
        assert any("padded" in w for w in r["warnings"])


def test_metrics_are_consistent():
    m = ivp.run_module(ASTRO, "compress", {"keep_percent": 10})["metrics"]
    assert m["coefficients_total"] == 512 * 512 * 3
    assert abs(m["kept_percent"] - 10) < 0.01
    assert abs(m["compression_ratio"] - 10) < 0.01
    assert 0 < m["ssim"] < 1 and 20 < m["psnr_db"] < 50
    assert 90 < m["energy_kept_percent"] <= 100
    assert m["size_original_png_bytes"] > 0 and m["size_compressed_jpeg_q90_bytes"] > 0


@pytest.mark.parametrize("img", [CAM, ASTRO, COFFEE], ids=["camera", "astronaut", "coffee"])
@pytest.mark.parametrize("p", [5, 10])
def test_dct_beats_hadamard_on_natural_images(img, p):
    d = ivp.run_module(img, "compress", {"method": "dct", "keep_percent": p})["metrics"]["psnr_db"]
    h = ivp.run_module(img, "compress", {"method": "hadamard", "keep_percent": p})["metrics"]["psnr_db"]
    assert d >= h - 0.1


@pytest.mark.parametrize("params", [
    {"method": "dct", "keep_percent": 10},
    {"method": "hadamard", "keep_percent": 10},
    {"method": "dct", "mode": "zonal", "zonal_fraction": 0.25},
])
@pytest.mark.parametrize("img", [CAM, ASTRO], ids=["gray", "rgb"])
def test_engines_agree(img, params):
    a = ivp.run_module(img, "compress", params, engine="opencv")["image"].astype(int)
    b = ivp.run_module(img, "compress", params, engine="numpy")["image"].astype(int)
    assert np.abs(a - b).max() <= 2 and np.abs(a - b).mean() < 0.01


@pytest.mark.parametrize("params", [
    {"method": "fft"}, {"keep_percent": 0}, {"keep_percent": 101}, {"threshold": -1},
    {"zonal_shape": "circle"}, {"zonal_fraction": 0}, {"mode": "jpeg"},
])
def test_bad_params(params):
    with pytest.raises(PipelineError) as e:
        ivp.run_module(CAM, "compress", params)
    assert e.value.code == "BAD_PARAM"


# ------------------------------------------------------------------ DCT vs Hadamard
def test_compare_compression():
    r = ivp.compare_compression(COFFEE, keep_percent=10)
    assert set(r) >= {"dct", "hadamard", "rd_curve", "energy_curve", "summary", "curve_scale", "meta"}
    assert [row["keep_percent"] for row in r["rd_curve"]] == [0.5, 1, 2, 5, 10, 25, 50]
    for row in r["rd_curve"]:
        assert row["dct"]["psnr_db"] >= row["hadamard"]["psnr_db"] - 0.1
    psnrs = [row["dct"]["psnr_db"] for row in r["rd_curve"]]
    assert psnrs == sorted(psnrs)                               # more coefficients, better quality
    assert len(r["energy_curve"]) == 40 and abs(r["energy_curve"][-1]["dct"] - 100) < 1e-6
    assert r["summary"]["winner"] == "dct" and r["summary"]["psnr_margin_db"] > 0.5
    assert r["dct"]["metrics"]["error_amplification"] == r["hadamard"]["metrics"]["error_amplification"]
    assert r["curve_scale"] == pytest.approx(512 / 600)
    assert r["dct"]["metrics"]["kept_percent"] == pytest.approx(10, abs=0.01)


def test_compare_compression_bad_keep_percent():
    with pytest.raises(PipelineError):
        ivp.compare_compression(COFFEE, keep_percent=0)


def test_size_curve_real_files():
    """Real files: JPEG-style coding reaches the same quality with far fewer bits than a whole-image result."""
    r = ivp.compare_compression(COFFEE, keep_percent=10)
    sc = r["size_curve"]
    assert [p["quality"] for p in sc["jpeg"]] == [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    jpeg_bpp = [p["bpp"] for p in sc["jpeg"]]
    jpeg_db = [p["psnr_db"] for p in sc["jpeg"]]
    assert jpeg_bpp == sorted(jpeg_bpp) and jpeg_db == sorted(jpeg_db)     # more bits, more quality
    assert 10 in [p["keep_percent"] for p in sc["dct"]] and len(sc["hadamard"]) == len(sc["dct"])
    jv = r["summary"]["jpeg_vs_dct"]
    assert jv["times_smaller"] is not None and jv["times_smaller"] > 2
    assert jv["jpeg_bpp_same_quality"] < jv["dct_bpp"]


def test_size_curve_includes_users_keep_percent():
    r = ivp.compare_compression(COFFEE, keep_percent=7.5)
    assert 7.5 in [p["keep_percent"] for p in r["size_curve"]["dct"]]
    assert len(r["rd_curve"]) == 7                                            # the RD chart keeps its 7 levels


def test_jpeg_matched_to_dct_quality_is_much_smaller():
    """Three-way comparison: JPEG-style at the quality that matches the DCT result's PSNR."""
    r = ivp.compare_compression(COFFEE, keep_percent=10)
    j, d = r["jpeg"]["metrics"], r["dct"]["metrics"]
    assert j["method"] == "jpeg" and j["psnr_db"] >= d["psnr_db"] - 0.05      # at least the same quality
    s = r["summary"]["jpeg"]
    assert s["quality"] == j["quality"] and s["times_smaller_than_dct"] > 3
    assert j["file_bytes"] < d["file_bytes"] < r["hadamard"]["metrics"]["file_bytes"] * 1.5
    assert r["jpeg"]["metrics"]["error_amplification"] == d["error_amplification"]   # one shared scale


def test_match_jpeg_quality_is_the_lowest_sufficient_quality():
    from ivp_pipeline.compare_compression import match_jpeg_quality
    from ivp_pipeline.modules import compress as Z
    from ivp_pipeline.metrics import psnr
    q = match_jpeg_quality(COFFEE, 32.0)
    at = lambda k: psnr(COFFEE, Z.decode_cv2(Z.encode_jpeg_cv2(COFFEE, k, "4:2:0"), False))
    assert at(q) >= 32.0 and (q == 1 or at(q - 1) < 32.0)


def test_compare_compression_manual_jpeg_quality():
    r = ivp.compare_compression(COFFEE, keep_percent=10, jpeg_quality=90)
    assert r["jpeg"]["metrics"]["quality"] == 90 and r["summary"]["jpeg"]["matched_to_dct"] is False
    assert r["jpeg"]["metrics"]["psnr_db"] > r["dct"]["metrics"]["psnr_db"]       # q90 is better than DCT 10 %
    with pytest.raises(PipelineError):
        ivp.compare_compression(COFFEE, keep_percent=10, jpeg_quality=0)


def test_compare_compression_manual_jpeg_quality():
    r = ivp.compare_compression(COFFEE, keep_percent=10, jpeg_quality=90)
    assert r["jpeg"]["metrics"]["quality"] == 90 and r["summary"]["jpeg"]["matched_to_dct"] is False
    assert r["jpeg"]["metrics"]["psnr_db"] > r["dct"]["metrics"]["psnr_db"]       # q90 beats DCT keeping 10 %
    with pytest.raises(PipelineError):
        ivp.compare_compression(COFFEE, keep_percent=10, jpeg_quality=0)
