"""JPEG-style compression (core/jpeg.py and compress method="jpeg"): checked against a real JPEG codec."""
import io

import numpy as np
import pytest
from PIL import Image
from skimage import data

import ivp_pipeline as ivp
from ivp_pipeline.core import jpeg as J
from ivp_pipeline.errors import PipelineError
from ivp_pipeline.metrics import psnr

CAM = data.camera()
ASTRO = data.astronaut()
COFFEE = data.coffee()


def libjpeg(img, quality, subsampling="4:2:0"):
    """Encode with Pillow (libjpeg) and decode again: what a real JPEG file looks like."""
    buf = io.BytesIO()
    kw = {"subsampling": subsampling} if img.ndim == 3 else {}
    Image.fromarray(img).save(buf, "JPEG", quality=quality, **kw)
    with Image.open(io.BytesIO(buf.getvalue())) as im:
        return np.array(im.convert("RGB" if img.ndim == 3 else "L")), len(buf.getvalue()), im.quantization


@pytest.mark.parametrize("quality", [10, 50, 75, 95])
def test_quant_tables_match_libjpeg(quality):
    """Our IJG-scaled tables are exactly the ones libjpeg writes into the file (zig-zag order there)."""
    _, _, tables = libjpeg(ASTRO, quality)
    ours = sorted(J.quant_table(J.LUMA_Q50, quality).ravel().astype(int).tolist())
    assert sorted(tables[0]) == ours
    assert sorted(tables[1]) == sorted(J.quant_table(J.CHROMA_Q50, quality).ravel().astype(int).tolist())


@pytest.mark.parametrize("quality", [20, 60, 85, 94])
def test_quality_estimate_from_file(quality):
    _, _, tables = libjpeg(COFFEE, quality)
    assert J.estimate_quality(tables[0]) == quality


@pytest.mark.parametrize("img,sub,min_db", [
    (CAM, "4:4:4", 48.0),          # gray: only rounding differs (libjpeg uses an integer DCT)
    (ASTRO, "4:4:4", 44.0),
    (ASTRO, "4:2:0", 43.0),
    (COFFEE, "4:2:2", 42.0),
])
@pytest.mark.parametrize("quality", [30, 75, 90])
def test_roundtrip_matches_real_jpeg(img, sub, min_db, quality):
    ours = J.jpeg_roundtrip(img, quality, sub)["image"]
    real, _, _ = libjpeg(img, quality, sub)
    assert ours.shape == img.shape and ours.dtype == np.uint8
    assert psnr(real, ours) > min_db                       # our decoder ≈ libjpeg's
    assert abs(psnr(img, ours) - psnr(img, real)) < 0.5     # same quality loss


def test_lower_quality_means_fewer_nonzero_coefficients_and_smaller_files():
    kept, sizes, quality_db = [], [], []
    for q in (95, 75, 50, 25):
        r = ivp.run_module(ASTRO, "compress", {"method": "jpeg", "quality": q})["metrics"]
        kept.append(r["kept_percent"])
        sizes.append(r["jpeg_file_bytes"])
        quality_db.append(r["psnr_db"])
    assert kept == sorted(kept, reverse=True) and sizes == sorted(sizes, reverse=True)
    assert quality_db == sorted(quality_db, reverse=True)


def test_jpeg_beats_whole_image_dct_at_equal_quality():
    """Why the JPEG-style method exists: for the same PSNR it makes a much smaller file than saving
    a whole-image keep-% result (which still has to be stored as a full image)."""
    j = ivp.run_module(COFFEE, "compress", {"method": "jpeg", "quality": 75})["metrics"]
    d = ivp.run_module(COFFEE, "compress", {"method": "dct", "keep_percent": 10})
    _, dct_file_bytes, _ = libjpeg(d["image"], 94, "4:2:0")
    assert j["psnr_db"] >= d["metrics"]["psnr_db"] - 1.0
    assert j["jpeg_file_bytes"] < 0.6 * dct_file_bytes


@pytest.mark.parametrize("img", [CAM, ASTRO, COFFEE[:301, :457]], ids=["gray", "rgb", "rgb_odd"])
def test_engines_agree(img):
    a = ivp.run_module(img, "compress", {"method": "jpeg", "quality": 60}, engine="opencv")
    b = ivp.run_module(img, "compress", {"method": "jpeg", "quality": 60}, engine="numpy")
    assert a["image"].shape == b["image"].shape == img.shape
    assert psnr(a["image"], b["image"]) > 42
    assert a["metrics"]["kept_percent"] == b["metrics"]["kept_percent"]
    assert b["metrics"]["match_with_real_jpeg_db"] > 42


def test_jpeg_result_structure():
    r = ivp.run_module(ASTRO, "compress", {"method": "jpeg", "quality": 50, "chroma_subsampling": "4:2:2"})
    m, ex = r["metrics"], r["extras"]
    assert m["method"] == "jpeg" and m["quality"] == 50 and m["chroma_subsampling"] == "4:2:2"
    assert 0 < m["kept_percent"] < 30 and 0 < m["energy_kept_percent"] <= 100
    assert ex["coef_before"].shape == ex["coef_after"].shape == ex["kept_mask"].shape == (512, 512)
    assert ex["error_map"].shape == ASTRO.shape
    assert r["warnings"] == []


def test_frequency_layout_puts_dc_tile_top_left():
    """The DC tile of the rearranged coefficients is a tiny version of the image."""
    planes = J.jpeg_roundtrip(CAM, 75)["planes"]
    layout = J.frequency_layout(planes[0]["coef"])
    dc = layout[:64, :64]                                    # 512 / 8 = 64 blocks per side
    small = CAM.reshape(64, 8, 64, 8).mean(axis=(1, 3))
    assert np.corrcoef(dc.ravel(), small.ravel())[0, 1] > 0.999


@pytest.mark.parametrize("params", [{"method": "jpeg", "quality": 0}, {"method": "jpeg", "quality": 101},
                                    {"method": "jpeg", "chroma_subsampling": "4:1:1"}, {"zonal_shape": "lab"}])
def test_bad_params(params):
    with pytest.raises(PipelineError) as e:
        ivp.run_module(CAM, "compress", params)
    assert e.value.code == "BAD_PARAM"
