"""Image decoding/encoding and JSON-safe serialisation of pipeline results."""
from __future__ import annotations

import base64

import cv2
import numpy as np

from ivp_pipeline import PipelineError

IMAGE_EXTRAS = ("mask", "raw_mask", "cutout", "edge_map", "background")


# ------------------------------------------------------------------ decode
def decode_upload(data: bytes) -> np.ndarray:
    """Bytes of a PNG/JPEG/BMP/TIFF/WebP -> uint8 RGB (or gray) array. EXIF orientation is applied."""
    if not data:
        raise PipelineError("BAD_IMAGE", "The uploaded file is empty.")
    buf = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise PipelineError("BAD_IMAGE", "Could not read the file as an image. Use PNG, JPEG, BMP, TIFF or WebP.")
    if img.dtype != np.uint8:  # 16-bit PNG/TIFF -> 8-bit
        if img.dtype == np.uint16:
            # Linear scale 0-65535 → 0-255 preserves the image's absolute exposure
            # level so brightness diagnostics remain valid.  NORM_MINMAX always maps
            # the darkest pixel to 0 and the brightest to 255, hiding over/under-exposure.
            img = (img >> 8).astype(np.uint8)
        else:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        # IMREAD_UNCHANGED skips EXIF rotation; alpha images are PNG/WebP, which carry none.
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    if img.shape[2] == 3:
        # re-decode with IMREAD_COLOR so camera photos get their EXIF orientation applied
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # store grayscale photos as gray (saves work, and keeps the pipeline's gray paths)
        if np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 1], rgb[:, :, 2]):
            return np.ascontiguousarray(rgb[:, :, 0])
        return rgb
    return img[:, :, 0]


# ------------------------------------------------------------------ encode
def encode(img: np.ndarray, fmt: str = "png", quality: int = 90) -> str:
    """uint8 gray/RGB/RGBA array -> data URL."""
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        fmt = "png"  # JPEG has no alpha
    if fmt == "jpeg":
        ok, out = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        mime = "image/jpeg"
    else:
        ok, out = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        mime = "image/png"
    if not ok:  # pragma: no cover
        raise RuntimeError("image encoding failed")
    return f"data:{mime};base64,{base64.b64encode(out.tobytes()).decode('ascii')}"


def encode_png_bytes(img: np.ndarray) -> bytes:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, out = cv2.imencode(".png", img)
    return out.tobytes()


# ------------------------------------------------------------------ JSON safety
def jsonable(x):
    """Recursively convert NumPy scalars/tuples so FastAPI can serialise the value."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def module_result(res: dict, image_fmt: str = "png", include_input: bool = False) -> dict:
    """Serialise a run_module-style result. Float arrays (gx, gy) are dropped: too big, not displayable."""
    extras = {}
    for k, v in res.get("extras", {}).items():
        if k in IMAGE_EXTRAS and isinstance(v, np.ndarray):
            extras[k] = encode(v, "png")
        elif k == "lut":
            extras[k] = jsonable(v)
    out = {
        "image": encode(res["image"], image_fmt),
        "extras": extras,
        "histograms": res.get("histograms"),
        "metrics": jsonable(res.get("metrics", {})),
        "warnings": res.get("warnings", []),
        "meta": jsonable(res.get("meta", {})),
    }
    if "path" in res:
        out["path"] = res["path"]
    if include_input and "input" in res:
        out["input"] = encode(res["input"], "png")
    return out
