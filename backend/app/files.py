"""Image FILES (as opposed to pixel arrays): what an upload is, and how a result is saved.

inspect_upload() reads the header of the uploaded file (format, size, JPEG quality and chroma
subsampling, EXIF, colour profile) without decoding the pixels.
save_like() writes a result image as a real file, by default in the same format as the upload,
so the user downloads e.g. a JPEG for a JPEG.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, JpegImagePlugin

from ivp_pipeline.core.jpeg import estimate_quality

SUBSAMPLING_NAMES = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}
EXTENSIONS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
LOSSLESS = {"PNG", "BMP", "TIFF", "GIF"}
ORIENTATION_TAG = 0x0112


def inspect_upload(data: bytes) -> dict:
    """Facts about the uploaded file, read from its header (no pixel decoding)."""
    with Image.open(io.BytesIO(data)) as im:
        fmt = "JPEG" if im.format == "MPO" else (im.format or "unknown")   # MPO = phone JPEG with extras
        w, h = im.size
        exif = im.getexif()
        orientation = int(exif.get(ORIENTATION_TAG, 1) or 1)
        if orientation in (5, 6, 7, 8):          # stored sideways; shown rotated
            w, h = h, w
        info = {
            "format": fmt,
            "mime": Image.MIME.get(im.format, "application/octet-stream"),
            "extension": EXTENSIONS.get(fmt, (im.format or "img").lower()),
            "width": w, "height": h,
            "megapixels": round(w * h / 1e6, 2),
            "mode": im.mode,
            "file_bytes": len(data),
            "bits_per_pixel": round(len(data) * 8 / max(1, w * h), 3),
            "lossless_format": fmt in LOSSLESS,
            "has_exif": len(exif) > 0,
            "has_icc_profile": "icc_profile" in im.info,
            "orientation": orientation,
            "jpeg_quality_estimate": None, "chroma_subsampling": None, "progressive": None,
        }
        if fmt == "JPEG":
            tables = getattr(im, "quantization", None) or {}
            if 0 in tables:
                info["jpeg_quality_estimate"] = estimate_quality(tables[0])
            info["chroma_subsampling"] = ("gray" if im.mode == "L"
                                          else SUBSAMPLING_NAMES.get(JpegImagePlugin.get_sampling(im)))
            info["progressive"] = bool(im.info.get("progressive") or im.info.get("progression"))
        return info


def _metadata(data: bytes) -> dict:
    """EXIF (with orientation reset: our pixels are already upright) and ICC colour profile."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            exif = im.getexif()
            if ORIENTATION_TAG in exif:
                exif[ORIENTATION_TAG] = 1
            out = {}
            if len(exif):
                out["exif"] = exif.tobytes()
            if "icc_profile" in im.info:
                out["icc_profile"] = im.info["icc_profile"]
            return out
    except Exception:
        return {}


def save_like(img: np.ndarray, original: bytes, fmt: str, quality: int | None = None,
              subsampling: str | None = None) -> bytes:
    """Encode a uint8 RGB/gray image as fmt ("JPEG", "PNG" or "WEBP"), keeping the upload's EXIF
    and colour profile. JPEG uses Huffman-table optimisation (smaller file, same pixels)."""
    pil = Image.fromarray(img)
    meta = _metadata(original)
    buf = io.BytesIO()
    if fmt == "JPEG":
        kw = {"quality": int(quality or 90), "optimize": True}
        if img.ndim == 3 and subsampling in ("4:4:4", "4:2:2", "4:2:0"):
            kw["subsampling"] = subsampling
        pil.save(buf, "JPEG", **kw, **meta)
    elif fmt == "WEBP":
        pil.save(buf, "WEBP", quality=int(quality or 90), method=4, **meta)
    else:
        pil.save(buf, "PNG", optimize=True, **meta)
    return buf.getvalue()


def output_plan(info: dict, params: dict) -> dict:
    """Which file to write for a compression result.

    JPEG-style method → a JPEG at the chosen quality and colour subsampling (that IS the method).
    Whole-image DCT / Hadamard → the upload's own format; a JPEG upload is re-saved at its own
    (estimated) quality and subsampling, so the size change comes from the transform step only.
    Formats other than JPEG / PNG / WebP are saved as PNG.
    """
    if params["method"] == "jpeg":
        return {"format": "JPEG", "quality": int(params["quality"]), "subsampling": params["chroma_subsampling"],
                "source": "result_of_jpeg_method"}
    fmt = info["format"] if info["format"] in ("JPEG", "PNG", "WEBP") else "PNG"
    return {"format": fmt,
            "quality": (info.get("jpeg_quality_estimate") or 90) if fmt in ("JPEG", "WEBP") else None,
            "subsampling": info.get("chroma_subsampling") if fmt == "JPEG" else None,
            "source": "same_format_as_upload"}
