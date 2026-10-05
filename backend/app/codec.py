"""Image decoding/encoding and JSON-safe serialisation of pipeline results."""
from __future__ import annotations

import base64
import io
import os

import cv2
import numpy as np
from PIL import Image

from ivp_pipeline import PipelineError

IMAGE_EXTRAS = ("mask", "raw_mask", "cutout", "edge_map", "background",
                "coef_before", "coef_after", "kept_mask", "error_map")

# Largest image (in pixels) the server will decode. Render's free instance has 512 MB of RAM;
# a 25 MP RGB image is ~75 MB per copy. A small, highly compressible PNG can claim 20000 x 20000
# pixels (1.2 GB decoded), so the header is checked BEFORE any pixels are decoded.
MAX_PIXELS = int(os.getenv("MAX_PIXELS", "25000000"))
# We enforce our own, stricter limit above; Pillow's bomb check would only get in the way of
# reading the header of an over-limit file so we can report its size.
Image.MAX_IMAGE_PIXELS = None

ALPHA_MODES = {"RGBA", "LA", "PA", "RGBa", "La"}
GRAY16_MODES = {"I;16", "I;16B", "I;16L", "I;16N", "I"}
GRAY8_MODES = {"L", "1"}


def _header(data: bytes):
    """(width, height, mode, has_transparency) from the file header, or None if Pillow can't read it.

    Image.open is lazy: it parses the header and does not decode the pixels.
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            return im.size[0], im.size[1], im.mode, "transparency" in im.info
    except Exception:
        return None


def _imread_flag(mode: str, has_transparency: bool) -> int:
    """Pick ONE OpenCV decode flag from the header mode, so the file is decoded only once.

    alpha (RGBA, LA, PA, palette + transparency) -> IMREAD_UNCHANGED (alpha is dropped later)
    16-bit gray                                  -> IMREAD_UNCHANGED (keeps 16 bits for the >> 8 rule)
    8-bit gray                                   -> IMREAD_GRAYSCALE (applies EXIF orientation)
    everything else                              -> IMREAD_COLOR     (applies EXIF orientation)
    """
    if mode in ALPHA_MODES or has_transparency or mode in GRAY16_MODES:
        return cv2.IMREAD_UNCHANGED
    if mode in GRAY8_MODES:
        return cv2.IMREAD_GRAYSCALE
    return cv2.IMREAD_COLOR


# ------------------------------------------------------------------ decode
def decode_upload(data: bytes) -> np.ndarray:
    """Bytes of a PNG/JPEG/BMP/TIFF/WebP -> uint8 RGB (or gray) array. EXIF orientation is applied.

    Raises PipelineError("TOO_LARGE") if the header says width x height > MAX_PIXELS.
    """
    if not data:
        raise PipelineError("BAD_IMAGE", "The uploaded file is empty.")
    head = _header(data)
    if head is not None:
        w, h, mode, transparent = head
        if w * h > MAX_PIXELS:
            raise PipelineError(
                "TOO_LARGE",
                f"Image is {w} × {h} ({w * h / 1e6:.1f} MP). The limit is {MAX_PIXELS / 1e6:g} MP. "
                "Please resize it and try again.")
        flag = _imread_flag(mode, transparent)
    else:
        flag = cv2.IMREAD_UNCHANGED   # Pillow can't parse it: let OpenCV try (may still fail below)

    img = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
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
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] != 3:
        return np.ascontiguousarray(img[:, :, 0])
    # store grayscale-looking photos as gray (saves work, and keeps the pipeline's gray paths)
    if np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 1], img[:, :, 2]):
        return np.ascontiguousarray(img[:, :, 0])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)   # in place: no extra full-size copy


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
    """Recursively convert NumPy scalars/tuples so FastAPI can serialise the value.

    Non-finite floats (e.g. PSNR = inf for a lossless result) become null: JSON has no Infinity.
    """
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
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
