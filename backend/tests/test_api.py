"""HTTP-level tests: every endpoint, the happy paths and the error paths."""
import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from skimage import data

from app.main import app
from ivp_pipeline import synthetic as S

client = TestClient(app)


def png(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imencode(".png", img)[1].tobytes()


def decode_data_url(url):
    header, b64 = url.split(",", 1)
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return header, cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


CAM = png(data.camera())
RGB = png(data.astronaut())


def post(path, image=CAM, **form):
    form = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in form.items()}
    return client.post(path, files={"file": ("x.png", image, "image/png")}, data=form)


def test_health_and_modules():
    assert client.get("/api/health").json()["status"] == "ok"
    mods = client.get("/api/modules").json()
    assert set(mods) == {"contrast", "denoise", "sharpen_edges", "segment", "compress"}


def test_samples():
    items = client.get("/api/samples").json()
    assert len(items) >= 6 and all("url" in s and "hint" in s for s in items)
    r = client.get(items[0]["url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get("/api/samples/nope.png").status_code == 404


def test_diagnose():
    r = post("/api/diagnose", png(S.salt_pepper(data.camera(), 0.05)))
    body = r.json()
    assert r.status_code == 200
    assert "IMPULSE_NOISE" in {f["code"] for f in body["findings"]}


@pytest.mark.parametrize("module,params", [
    ("contrast", {"mode": "gamma", "gamma": "auto"}),
    ("denoise", {"filter_type": "median"}),
    ("sharpen_edges", {"operation": "gradient", "edge_threshold": 0.2}),
    ("segment", {"method": "region_growing", "seed": [100, 100]}),
])
def test_process_all_modules(module, params):
    r = post("/api/process", RGB, module=module, params=params)
    assert r.status_code == 200, r.text
    body = r.json()
    header, img = decode_data_url(body["image"])
    assert header.startswith("data:image/png") and img is not None
    assert len(body["histograms"]["before"]) == 256
    assert "gx" not in body["extras"]            # float arrays are not sent
    if module == "segment":
        _, cut = decode_data_url(body["extras"]["cutout"])
        assert cut.shape[2] == 4                 # RGBA cut-out


def test_process_numpy_engine():
    r = post("/api/process", CAM, module="denoise", params={"filter_type": "median"}, engine="numpy")
    assert r.status_code == 200 and r.json()["meta"]["engine"] == "numpy"


def test_guided_with_prerequisite():
    img = png(S.illumination_ramp(S.scene(1)[0], 40))
    body = post("/api/guided", img, intent="segment").json()
    assert body["gate_passed"] is False
    assert body["verdict"]["result"] == "recommended"
    assert body["branch_b"]["stages"][0]["rule"] == "UNEVEN_ILLUMINATION"
    assert body["branch_b"]["stages"][0]["image"].startswith("data:image/jpeg")
    assert body["branch_b"]["extras"]["mask"].startswith("data:image/png")


def test_guided_clean_image():
    body = post("/api/guided", png(S.scene(1)[0]), intent="segment").json()
    assert body["gate_passed"] is True and body["branch_b"] is None


def test_auto():
    body = post("/api/auto", png(S.apply_gamma(data.camera(), 2.5))).json()
    assert any(s["module"] == "contrast" for s in body["steps"])
    assert body["diagnosis_after"]["metrics"]["mean"] > body["diagnosis_before"]["metrics"]["mean"]


def test_large_image_returns_input_and_scale():
    big = cv2.resize(data.camera(), (2048, 2048))
    body = post("/api/process", png(big), module="contrast", params={"mode": "stretch"}).json()
    assert body["meta"]["scale"] == 0.5 and "input" in body


@pytest.mark.parametrize("path,form,code", [
    ("/api/process", {"module": "nope"}, "UNKNOWN_MODULE"),
    ("/api/process", {"module": "denoise", "params": "{bad json"}, "BAD_PARAM"),
    ("/api/process", {"module": "denoise", "params": {"kernel_size": 4}}, "BAD_PARAM"),
    ("/api/process", {"module": "denoise", "engine": "gpu"}, "BAD_ENGINE"),
    ("/api/guided", {"intent": "teleport"}, "UNKNOWN_INTENT"),
])
def test_errors(path, form, code):
    r = post(path, CAM, **form)
    assert r.status_code == 422 and r.json()["error"]["code"] == code


def test_not_an_image():
    r = post("/api/diagnose", b"hello, not an image")
    assert r.status_code == 422 and r.json()["error"]["code"] == "BAD_IMAGE"


def test_rgba_upload():
    rgba = np.dstack([data.astronaut(), np.full((512, 512), 200, np.uint8)])
    img = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))[1].tobytes()
    assert post("/api/process", img, module="denoise").status_code == 200


# ------------------------------------------------------------------ Phase P: uploads, decoding, 500s
import io

from PIL import Image

from app import codec, main


def _png_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def test_over_pixel_cap_is_rejected_without_decoding(monkeypatch):
    big = _png_bytes(Image.new("L", (6000, 5000), 128))       # 30 MP, but a tiny file
    assert len(big) < 200_000

    def must_not_decode(*a, **k):
        raise AssertionError("cv2.imdecode was called for an over-limit image")
    monkeypatch.setattr(codec.cv2, "imdecode", must_not_decode)
    r = post("/api/diagnose", big)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "TOO_LARGE" and "6000 × 5000" in err["message"] and "25 MP" in err["message"]


def test_just_under_pixel_cap_is_accepted():
    ok = _png_bytes(Image.new("L", (6000, 4000), 90))          # 24 MP
    r = post("/api/diagnose", ok)
    assert r.status_code == 200, r.text
    assert r.json()["meta"]["original_shape"] == [4000, 6000]


def test_decode_rgba_png():
    rgba = np.dstack([data.astronaut(), np.full((512, 512), 77, np.uint8)])
    out = codec.decode_upload(_png_bytes(Image.fromarray(rgba, "RGBA")))
    assert out.shape == (512, 512, 3) and np.array_equal(out, data.astronaut())


def test_decode_16bit_gray_png_keeps_exposure():
    g16 = (np.linspace(0, 20000, 64 * 64).reshape(64, 64)).astype(np.uint16)   # a dark 16-bit image
    out = codec.decode_upload(cv2.imencode(".png", g16)[1].tobytes())
    assert out.dtype == np.uint8 and out.ndim == 2
    assert np.array_equal(out, (g16 >> 8).astype(np.uint8))                     # not stretched to 0..255
    assert out.max() < 80


def test_decode_jpeg_applies_exif_orientation():
    img = Image.new("RGB", (200, 100), (200, 30, 30))          # landscape pixels
    exif = Image.Exif()
    exif[0x0112] = 6                                            # "rotate 90° CW to display"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    out = codec.decode_upload(buf.getvalue())
    assert out.shape[:2] == (200, 100)                          # displayed as portrait


def test_server_error_keeps_cors_headers(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated crash")
    monkeypatch.setattr(main.ivp, "diagnose", boom)
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/api/diagnose", files={"file": ("x.png", CAM, "image/png")},
               headers={"Origin": "https://example.vercel.app"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "INTERNAL"
    assert r.headers.get("access-control-allow-origin") in ("*", "https://example.vercel.app")


# ------------------------------------------------------------------ compression, sweeps, difference maps
@pytest.mark.parametrize("method", ["dct", "hadamard"])
def test_process_compress_rgb(method):
    r = post("/api/process", RGB, module="compress", params={"method": method, "keep_percent": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    header, img = decode_data_url(body["image"])
    assert header.startswith("data:image/png") and img.shape == (512, 512, 3)
    for k in ("coef_before", "coef_after", "kept_mask", "error_map"):
        assert body["extras"][k].startswith("data:image/png")
    assert body["metrics"]["method"] == method and 20 < body["metrics"]["psnr_db"] < 50


def test_lossless_compress_psnr_is_null_not_infinity():
    r = post("/api/process", CAM, module="compress", params={"keep_percent": 100})
    assert r.status_code == 200
    assert "Infinity" not in r.text and r.json()["metrics"]["psnr_db"] is None


@pytest.mark.parametrize("intent,image", [("segment", png(S.illumination_ramp(S.scene(1)[0], 40))),
                                          ("edges", png(S.gaussian_noise(data.camera(), 25)))],
                         ids=["segment", "edges"])
def test_sweep(intent, image):
    r = post("/api/sweep", image, intent=intent)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metric"] in ("IoU", "Edge F1") and len(body["points"]) >= 4
    assert {"level", "x", "direct", "guided", "gate_fired"} <= set(body["points"][0])
    assert "data:image" not in r.text                     # numbers only


def test_compare_compression():
    coffee = png(data.coffee())
    r = post("/api/compare-compression", coffee, keep_percent="10")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"dct", "hadamard", "rd_curve", "energy_curve", "size_curve", "summary", "meta", "curve_scale"}
    assert body["summary"]["jpeg_vs_dct"]["times_smaller"] > 1 and len(body["size_curve"]["jpeg"]) == 10
    assert len(body["rd_curve"]) == 7 and len(body["energy_curve"]) == 40
    for m in ("dct", "hadamard", "jpeg"):
        assert body[m]["image"].startswith("data:image/png")
        assert body[m]["extras"]["error_map"].startswith("data:image/png")
    assert body["summary"]["winner"] == "dct"
    assert body["summary"]["jpeg"]["times_smaller_than_dct"] > 1 and body["jpeg"]["metrics"]["method"] == "jpeg"


def test_guided_has_difference_map():
    body = post("/api/guided", png(S.gaussian_noise(data.camera(), 25)), intent="edges").json()
    assert body["difference_map"].startswith("data:image/png")
    assert body["difference_legend"] and "only_a_pixels" in body["comparison"]


@pytest.mark.parametrize("path,form,code", [
    ("/api/sweep", {"intent": "sharpen"}, "UNKNOWN_INTENT"),
    ("/api/sweep", {"intent": "teleport"}, "UNKNOWN_INTENT"),
    ("/api/compare-compression", {"keep_percent": "abc"}, "BAD_PARAM"),
    ("/api/compare-compression", {"keep_percent": "0"}, "BAD_PARAM"),
    ("/api/compare-compression", {"keep_percent": "250"}, "BAD_PARAM"),
    ("/api/process", {}, "BAD_REQUEST"),                   # missing "module" field
])
def test_new_endpoint_errors(path, form, code):
    r = post(path, CAM, **form)
    assert r.status_code == 422 and r.json()["error"]["code"] == code


def test_new_samples():
    ids = {s["id"] for s in client.get("/api/samples").json()}
    assert {"coffee", "chelsea"} <= ids


def test_keep_editing_a_gray_edge_result():
    """Bug-report row 9: A = 0 on a colour image returns a gray PNG; sending it back must work."""
    r = post("/api/process", RGB, module="sharpen_edges", params={"operation": "laplacian_boost", "A": 0}).json()
    assert any("grayscale" in w for w in r["warnings"])
    _, gray = decode_data_url(r["image"])
    assert gray.ndim == 2
    again = post("/api/process", png(gray), module="denoise", params={"filter_type": "median"})
    assert again.status_code == 200, again.text


# ------------------------------------------------------------------ image details + compressed FILE
def _jpeg_file(img, quality=92, subsampling="4:2:2", exif_orientation=None):
    buf = io.BytesIO()
    kw = {"quality": quality, "subsampling": subsampling}
    if exif_orientation:
        ex = Image.Exif()
        ex[0x0112] = exif_orientation
        kw["exif"] = ex
    Image.fromarray(img).save(buf, "JPEG", **kw)
    return buf.getvalue()


def test_inspect_jpeg():
    body = post("/api/inspect", _jpeg_file(data.coffee(), 92, "4:2:2")).json()
    assert body["format"] == "JPEG" and body["mime"] == "image/jpeg" and body["extension"] == "jpg"
    assert (body["width"], body["height"]) == (600, 400)
    assert body["jpeg_quality_estimate"] == 92 and body["chroma_subsampling"] == "4:2:2"
    assert body["file_bytes"] > 0 and body["lossless_format"] is False


def test_inspect_png_and_rotated_jpeg():
    assert post("/api/inspect", RGB).json()["format"] == "PNG"
    rotated = post("/api/inspect", _jpeg_file(data.coffee(), exif_orientation=6)).json()
    assert (rotated["width"], rotated["height"]) == (400, 600) and rotated["orientation"] == 6


def test_inspect_not_an_image():
    r = post("/api/inspect", b"not an image")
    assert r.status_code == 422 and r.json()["error"]["code"] == "BAD_IMAGE"


def _decode_file(url):
    header, b64 = url.split(",", 1)
    raw = base64.b64decode(b64)
    return header, raw, Image.open(io.BytesIO(raw))


def test_compress_jpeg_method_returns_smaller_jpeg_file():
    src = _jpeg_file(data.coffee(), 95, "4:2:2", exif_orientation=1)
    body = post("/api/compress", src, params={"method": "jpeg", "quality": 60, "chroma_subsampling": "4:2:2"}).json()
    f = body["file"]
    assert f["format"] == "JPEG" and f["extension"] == "jpg" and f["quality"] == 60
    assert f["original_bytes"] == len(src) and f["bytes"] < len(src) and f["size_percent"] < 100
    header, raw, im = _decode_file(body["image"])
    assert header.startswith("data:image/jpeg") and len(raw) == f["bytes"]
    assert im.format == "JPEG" and im.size == (600, 400)
    from PIL import JpegImagePlugin
    assert JpegImagePlugin.get_sampling(im) == 1                 # 4:2:2 kept
    assert len(im.getexif()) > 0                                 # metadata kept
    assert body["upload"]["jpeg_quality_estimate"] == 95
    assert 25 < body["metrics"]["psnr_db"] < 50 and body["warnings"] == []


def test_compress_same_format_for_transform_methods():
    jpg = post("/api/compress", _jpeg_file(data.coffee(), 90), params={"method": "dct", "keep_percent": 25}).json()
    assert jpg["file"]["format"] == "JPEG" and jpg["file"]["quality"] == 90
    assert jpg["file"]["source"] == "same_format_as_upload" and "method_psnr_db" in jpg["metrics"]
    png_up = post("/api/compress", RGB, params={"method": "dct", "keep_percent": 25}).json()
    assert png_up["file"]["format"] == "PNG" and png_up["image"].startswith("data:image/png")
    if png_up["file"]["bytes"] >= png_up["file"]["original_bytes"]:
        assert any("not smaller" in w for w in png_up["warnings"])


def test_compress_metrics_describe_the_downloaded_file():
    """PSNR reported = PSNR of the file the user downloads, measured against the input pixels."""
    src = _jpeg_file(data.coffee(), 95)
    body = post("/api/compress", src, params={"method": "jpeg", "quality": 40}).json()
    _, raw, _ = _decode_file(body["image"])
    shown = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    original = cv2.cvtColor(cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mse = np.mean((shown.astype(float) - original.astype(float)) ** 2)
    assert abs(body["metrics"]["psnr_db"] - 10 * np.log10(255 ** 2 / mse)) < 0.01


def test_compare_compression_jpeg_quality_option():
    coffee = png(data.coffee())
    auto = post("/api/compare-compression", coffee, keep_percent="10").json()["summary"]["jpeg"]
    assert auto["matched_to_dct"] is True and abs(auto["psnr_db"] - auto["dct_psnr_db"]) < 1.5
    manual = post("/api/compare-compression", coffee, keep_percent="10", jpeg_quality="85").json()
    assert manual["summary"]["jpeg"]["quality"] == 85 and manual["summary"]["jpeg"]["matched_to_dct"] is False
    assert manual["jpeg"]["metrics"]["quality"] == 85
    for bad in ("0", "101", "abc"):
        r = post("/api/compare-compression", coffee, keep_percent="10", jpeg_quality=bad)
        assert r.status_code == 422 and r.json()["error"]["code"] == "BAD_PARAM"


def test_compare_compression_jpeg_quality_option():
    coffee = png(data.coffee())
    auto = post("/api/compare-compression", coffee, keep_percent="10").json()["summary"]["jpeg"]
    assert auto["matched_to_dct"] is True and abs(auto["psnr_db"] - auto["dct_psnr_db"]) < 1.5
    manual = post("/api/compare-compression", coffee, keep_percent="10", jpeg_quality="85").json()
    assert manual["summary"]["jpeg"]["quality"] == 85 and manual["summary"]["jpeg"]["matched_to_dct"] is False
    assert manual["jpeg"]["metrics"]["quality"] == 85
    for bad in ("0", "101", "abc"):
        r = post("/api/compare-compression", coffee, keep_percent="10", jpeg_quality=bad)
        assert r.status_code == 422 and r.json()["error"]["code"] == "BAD_PARAM"
