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
    # previews: binary results (an edge map) as PNG, photos as JPEG
    expected = "data:image/png" if module == "sharpen_edges" else "data:image/jpeg"
    assert header.startswith(expected) and img is not None
    assert "input" not in body                   # the browser already has the input
    assert len(body["histograms"]["before"]) == 256
    assert "gx" not in body["extras"]            # float arrays are not sent
    if module == "segment":
        assert "cutout" not in body["extras"]    # not shown anywhere; full-size RGBA would be huge
        assert body["extras"]["mask"].startswith("data:image/png")


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


def test_large_image_reports_scale_and_sends_a_preview():
    big = cv2.resize(data.camera(), (2048, 2048))
    body = post("/api/process", png(big), module="contrast", params={"mode": "stretch"}).json()
    assert body["meta"]["scale"] == 0.5 and "input" not in body
    _, img = decode_data_url(body["image"])
    assert max(img.shape[:2]) <= codec.PREVIEW_SIDE


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
    assert header.startswith("data:image/jpeg") and img.shape == (512, 512, 3)
    for k in ("coef_before", "coef_after", "error_map"):
        assert body["extras"][k].startswith("data:image/jpeg")
    assert body["extras"]["kept_mask"].startswith("data:image/png")       # 0/255 mask: PNG is exact and tiny
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
        assert body[m]["image"].startswith("data:image/jpeg")
        assert body[m]["extras"]["error_map"].startswith("data:image/jpeg")
    assert body["summary"]["winner"] == "dct"
    assert body["summary"]["jpeg"]["times_smaller_than_dct"] > 1 and body["jpeg"]["metrics"]["method"] == "jpeg"


def test_guided_has_difference_map():
    body = post("/api/guided", png(S.gaussian_noise(data.camera(), 25)), intent="edges").json()
    assert body["difference_map"].startswith("data:image/")
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
    f = post("/api/process/file", RGB, module="sharpen_edges", params={"operation": "laplacian_boost", "A": 0})
    gray = cv2.imdecode(np.frombuffer(f.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert f.headers["x-ivp-format"] == "PNG" and gray.ndim == 2
    again = post("/api/process", f.content, module="denoise", params={"filter_type": "median"})
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


def _download(src, **form):
    """The compressed file itself, from /api/compress/file."""
    r = post("/api/compress/file", src, **form)
    assert r.status_code == 200, r.text
    return r.content


def test_compress_jpeg_method_returns_smaller_jpeg_file():
    src = _jpeg_file(data.coffee(), 95, "4:2:2", exif_orientation=1)
    body = post("/api/compress", src, params={"method": "jpeg", "quality": 60, "chroma_subsampling": "4:2:2"}).json()
    f = body["file"]
    assert f["format"] == "JPEG" and f["extension"] == "jpg" and f["quality"] == 60
    assert f["original_bytes"] == len(src) and f["bytes"] < len(src) and f["size_percent"] < 100
    assert f["downloadable"] is True and f["note"] is None
    raw = _download(src, params={"method": "jpeg", "quality": 60, "chroma_subsampling": "4:2:2"})
    im = Image.open(io.BytesIO(raw))
    assert len(raw) == f["bytes"] and im.format == "JPEG" and im.size == (600, 400)
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
    assert png_up["file"]["format"] == "PNG" and png_up["image"].startswith("data:image/jpeg")   # preview
    if png_up["file"]["bytes"] >= png_up["file"]["original_bytes"]:
        assert any("not smaller" in w for w in png_up["warnings"])


def test_compress_metrics_describe_the_downloaded_file():
    """PSNR reported = PSNR of the file the user downloads, measured against the input pixels."""
    src = _jpeg_file(data.coffee(), 95)
    body = post("/api/compress", src, params={"method": "jpeg", "quality": 40}).json()
    raw = _download(src, params={"method": "jpeg", "quality": 40})
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


# ------------------------------------------------------------------ limits, previews and full-quality files
from pathlib import Path
from urllib.parse import unquote

from app import limits


def _file_image(r):
    assert r.status_code == 200, r.text
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)


def test_health_reports_limits():
    lim = client.get("/api/health").json()["limits"]
    assert lim["max_upload_bytes"] == limits.MAX_UPLOAD_BYTES and lim["max_download_bytes"] == limits.MAX_DOWNLOAD_BYTES
    assert lim["max_pixels"] == codec.MAX_PIXELS and lim["hosted_on_vercel"] is False


def test_upload_over_the_limit_is_refused_with_the_limit_in_the_message(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100_000)
    r = post("/api/diagnose", RGB)                           # ~0.5 MB PNG
    assert r.status_code == 422 and r.json()["error"]["code"] == "TOO_LARGE"
    assert "0.1 MB upload limit" in r.json()["error"]["message"]


def test_process_file_is_full_quality_png():
    r = post("/api/process/file", RGB, module="denoise", params={"filter_type": "median"})
    img = _file_image(r)
    assert r.headers["content-type"] == "image/png" and r.headers["x-ivp-format"] == "PNG"
    assert unquote(r.headers["x-ivp-note"]) == "" and img.shape == (512, 512, 3)
    # identical pixels to running the module in the pipeline
    import ivp_pipeline as ivp
    expected = ivp.run_module(data.astronaut(), "denoise", {"filter_type": "median"})["image"]
    assert np.array_equal(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), expected)


def test_file_headers_are_readable_cross_origin():
    r = client.post("/api/process/file", files={"file": ("x.png", CAM, "image/png")},
                    data={"module": "denoise"}, headers={"Origin": "https://example.vercel.app"})
    exposed = r.headers.get("access-control-expose-headers", "").lower()
    assert all(h.lower() in exposed for h in main.FILE_HEADERS)


def test_result_file_falls_back_to_jpeg_when_png_is_over_the_limit(monkeypatch):
    monkeypatch.setattr(main, "MAX_DOWNLOAD_BYTES", 150_000)
    r = post("/api/process/file", RGB, module="denoise", params={"filter_type": "median"})
    img = _file_image(r)
    note = unquote(r.headers["x-ivp-note"])
    assert r.headers["content-type"] == "image/jpeg" and len(r.content) <= 150_000
    assert "Saved as JPEG (quality" in note and "download limit" in note and img.shape == (512, 512, 3)


def test_result_too_large_even_as_jpeg(monkeypatch):
    monkeypatch.setattr(main, "MAX_DOWNLOAD_BYTES", 5_000)
    r = post("/api/process/file", RGB, module="denoise")
    assert r.status_code == 422 and r.json()["error"]["code"] == "RESULT_TOO_LARGE"


def test_auto_file_matches_auto():
    src = png(S.apply_gamma(data.camera(), 2.5))
    img = _file_image(post("/api/auto/file", src))
    import ivp_pipeline as ivp
    assert np.array_equal(img, ivp.auto_enhance(S.apply_gamma(data.camera(), 2.5))["image"])


def test_guided_file_routes_and_mask():
    src = png(S.illumination_ramp(S.scene(1)[0], 40))
    a = _file_image(post("/api/guided/file", src, intent="segment", branch="a"))
    b = _file_image(post("/api/guided/file", src, intent="segment", branch="b"))
    mask = _file_image(post("/api/guided/file", src, intent="segment", branch="b", layer="mask"))
    assert a.shape == b.shape and not np.array_equal(a, b)
    assert mask.ndim == 2 and set(np.unique(mask)) <= {0, 255}
    clean = post("/api/guided/file", png(S.scene(1)[0]), intent="segment", branch="b")
    assert clean.status_code == 422 and clean.json()["error"]["code"] == "BAD_PARAM"
    bad = post("/api/guided/file", src, intent="segment", branch="c")
    assert bad.status_code == 422


def test_compare_compression_file_matches_the_comparison():
    coffee = png(data.coffee())
    body = post("/api/compare-compression", coffee, keep_percent="10").json()
    for m in ("dct", "hadamard", "jpeg"):
        r = post("/api/compare-compression/file", coffee, method=m, keep_percent="10")
        img = _file_image(r)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if m == "jpeg":       # the real JPEG file the comparison measured, at the matched quality
            assert r.headers["x-ivp-format"] == "JPEG" and len(r.content) == body["jpeg"]["metrics"]["file_bytes"]
            assert int(r.headers["x-ivp-quality"]) == body["summary"]["jpeg"]["quality"]
        original = data.coffee()
        mse = np.mean((rgb.astype(float) - original.astype(float)) ** 2)
        assert abs(10 * np.log10(255 ** 2 / mse) - body[m]["metrics"]["psnr_db"]) < 0.02, m
    assert post("/api/compare-compression/file", coffee, method="wavelet").status_code == 422


def test_compress_png_dct_falls_back_to_jpeg_with_a_note(monkeypatch):
    monkeypatch.setattr(main, "MAX_DOWNLOAD_BYTES", 200_000)
    body = post("/api/compress", RGB, params={"method": "dct", "keep_percent": 25}).json()
    f = body["file"]
    assert f["format"] == "JPEG" and f["bytes"] <= 200_000 and f["downloadable"] is True
    assert "Saved as JPEG" in f["note"] and any("Saved as JPEG" in w for w in body["warnings"])
    raw = _download(RGB, params={"method": "dct", "keep_percent": 25})
    assert len(raw) == f["bytes"]


def test_compress_jpeg_method_over_the_limit_is_preview_only(monkeypatch):
    monkeypatch.setattr(main, "MAX_DOWNLOAD_BYTES", 50_000)
    params = {"method": "jpeg", "quality": 100}
    body = post("/api/compress", RGB, params=params).json()
    assert body["file"]["downloadable"] is False and any("can be previewed but not downloaded" in w for w in body["warnings"])
    r = post("/api/compress/file", RGB, params=params)
    assert r.status_code == 422 and r.json()["error"]["code"] == "RESULT_TOO_LARGE"


def test_previews_are_capped(monkeypatch):
    monkeypatch.setattr(codec, "PREVIEW_SIDE", 300)
    body = post("/api/process", RGB, module="denoise").json()
    _, img = decode_data_url(body["image"])
    assert max(img.shape[:2]) == 300


PHOTO = Path(__file__).resolve().parents[2] / "dataset" / "flower.jpg"
REPLY_BUDGET = 3.5 * 1024 * 1024        # well under Vercel's 4.5 MB per reply


@pytest.mark.skipif(not PHOTO.exists(), reason="dataset/flower.jpg not available")
def test_every_reply_fits_on_vercel_for_a_12_megapixel_photo(monkeypatch):
    """The real case behind the size work: a 3072 x 2304 phone photo (2.85 MB JPEG) processed at FULL
    resolution must give replies far below 4.5 MB on every endpoint."""
    from ivp_pipeline import config
    monkeypatch.setattr(config, "MAX_SIDE", 0)
    photo = PHOTO.read_bytes()

    def send(path, **form):
        form = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in form.items()}
        r = client.post(path, files={"file": ("flower.jpg", photo, "image/jpeg")}, data=form)
        assert r.status_code == 200, (path, r.text[:300])
        assert len(r.content) < REPLY_BUDGET, (path, form, len(r.content))
        return r

    send("/api/diagnose")
    send("/api/inspect")
    for module, params in (("contrast", {"mode": "hist_eq"}), ("denoise", {"filter_type": "median"}),
                           ("sharpen_edges", {"operation": "gradient"}),
                           ("segment", {"method": "otsu"})):
        send("/api/process", module=module, params=params)
    for params in ({"method": "jpeg", "quality": 75}, {"method": "dct", "keep_percent": 10},
                   {"method": "hadamard", "keep_percent": 10}):
        body = send("/api/compress", params=params).json()
        assert body["meta"]["shape"] == [2304, 3072]          # processed at full resolution
    send("/api/auto")
    send("/api/guided", intent="segment", params={"method": "otsu"})
    send("/api/guided", intent="edges", params={"operator": "sobel", "edge_threshold": 0.2})
    send("/api/compare-compression", keep_percent="10")
