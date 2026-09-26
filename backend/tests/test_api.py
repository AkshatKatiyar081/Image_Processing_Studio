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
    assert set(mods) == {"contrast", "denoise", "sharpen_edges", "segment"}


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
