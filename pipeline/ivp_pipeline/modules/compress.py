"""Module 5 — Image compression by transform coding.

method = "jpeg": JPEG-style coding (see core/jpeg.py): YCbCr, chroma subsampling, 8x8 block DCT and
    quantisation tables scaled by a quality setting. This is how real image files get small: the
    quantisation turns most high-frequency coefficients into 0, and the file stores few, small integers.

method = "dct" / "hadamard": whole-image transform coding, to study the transforms themselves.
    Per channel (R, G, B independently, or the single gray channel):
        f (float64, 0–255) → forward transform F → keep some coefficients → inverse → clip(0, 255) → round
    DCT:            F = C · f · Cᵀ (orthonormal DCT-II), any size.
    Walsh–Hadamard: the channel is mirror-padded to power-of-two sides first, transformed in sequency
                    order (orthonormal), and the reconstruction is cropped back.
    Zeroing coefficients alone does not make a smaller file: the result is still a full image that
    an encoder has to store. That is what the "jpeg" method adds (quantisation + entropy coding).

The reconstruction is only clipped and rounded. Stretching each channel min→max on its own (a common
mistake) would shift the colour balance.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from .. import metrics as Q
from ..core import jpeg as J
from ..core import transforms as T

LUMA_WEIGHTS = (0.299, 0.587, 0.114)


# -------------------------------------------------------------------- transforms
def _dct_forward(f: np.ndarray, engine: str) -> np.ndarray:
    """DCT of one channel. OpenCV 4.x's cv2.dct only supports even sizes (OpenCV 5 also does odd
    ones); if it refuses, odd sides are mirror-padded by 1 and the reconstruction is cropped back
    in _dct_inverse. The padded transform is a slightly different (larger) basis, so on OpenCV 4
    the two engines can differ a little on odd-sized images."""
    if engine == "numpy":
        return T.dct2(f)
    f32 = f.astype(np.float32)
    try:
        return cv2.dct(f32).astype(np.float64)
    except cv2.error:
        h, w = f.shape
        return cv2.dct(np.pad(f32, ((0, h % 2), (0, w % 2)), mode="reflect")).astype(np.float64)


def _dct_inverse(F: np.ndarray, shape, engine: str) -> np.ndarray:
    if engine == "numpy":
        return T.idct2(F)
    h, w = shape
    return cv2.idct(F.astype(np.float32)).astype(np.float64)[:h, :w]   # crop: no-op unless padded


def forward(f: np.ndarray, method: str, engine: str) -> np.ndarray:
    """Coefficients of one channel. Hadamard pads to powers of two (both engines use the NumPy
    fast butterfly: OpenCV has no Walsh–Hadamard transform)."""
    if method == "dct":
        return _dct_forward(f, engine)
    padded, _ = T.pad_to_pow2(f)
    return T.fwht2(padded, dtype=np.float32)   # float32: padded sizes get big (3072 px -> 4096)


def inverse(F: np.ndarray, shape, method: str, engine: str) -> np.ndarray:
    """Reconstruction of one channel at its original shape (float64, not yet clipped)."""
    if method == "dct":
        return _dct_inverse(F, shape, engine)
    h, w = shape
    return T.ifwht2(F, dtype=np.float32)[:h, :w]


def select(F: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply the chosen coefficient-selection rule. Returns (kept coefficients, kept mask)."""
    if p["mode"] == "keep_percent":
        return T.keep_percent(F, float(p["keep_percent"]))
    if p["mode"] == "threshold":
        return T.threshold(F, float(p["threshold"]))
    return T.zonal(F, float(p["zonal_fraction"]), p["zonal_shape"])


def to_uint8(x: np.ndarray) -> np.ndarray:
    """clip(0, 255) then round — the correct reconstruction (no per-channel min–max rescale)."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def channels(img: np.ndarray) -> list[np.ndarray]:
    return [img[:, :, c].astype(np.float64) for c in range(3)] if img.ndim == 3 else [img.astype(np.float64)]


# ------------------------------------------------------------------- display maps
def log_map(F: np.ndarray, peak: float | None = None) -> np.ndarray:
    """log(1 + |F|) scaled to 0–255 (uint8). Pass `peak` to share one scale between two maps."""
    m = np.log1p(np.abs(F))
    peak = float(m.max()) if peak is None else peak
    return np.zeros(F.shape, np.uint8) if peak <= 0 else to_uint8(m * (255.0 / peak))


def display_size(m: np.ndarray) -> np.ndarray:
    """Shrink a coefficient map or kept mask to at most COEF_MAP_MAX_SIDE px (INTER_AREA).

    These maps are only looked at as small pictures; at full resolution a padded Hadamard map of a
    3072 x 2304 photo would be 4096 x 4096 (tens of MB in the response) without showing anything more.
    A shrunk kept mask shows the DENSITY of kept coefficients as gray levels.
    """
    h, w = m.shape[:2]
    side = config.COEF_MAP_MAX_SIDE
    if max(h, w) <= side:
        return m
    s = side / max(h, w)
    return cv2.resize(m, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def abs_error(original: np.ndarray, rec: np.ndarray) -> np.ndarray:
    """|original − reconstruction| averaged over the channels (float32, gray levels)."""
    e = np.abs(original.astype(np.float32) - rec.astype(np.float32))
    return e.mean(axis=2) if e.ndim == 3 else e


def error_amplification(*errors: np.ndarray) -> float:
    """Gain that makes the 99th-percentile error bright (≈ 255), between x1 and x16.
    Pass several error arrays to get ONE shared gain, so their maps can be compared."""
    p99 = max(float(np.percentile(e, 99)) for e in errors)
    if p99 <= 0:
        return 1.0
    return float(np.clip(np.floor(255.0 / p99), 1, config.ERROR_MAP_MAX_AMPLIFICATION))


def error_map(err: np.ndarray, amplification: float) -> np.ndarray:
    """Amplified absolute error, colour-mapped with the 'inferno' colormap (RGB uint8)."""
    u8 = to_uint8(err * amplification)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def _png_size(img: np.ndarray) -> int:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
    return int(cv2.imencode(".png", bgr)[1].size)


def _jpeg_size(img: np.ndarray, quality: int = 90) -> int:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
    return int(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].size)


# --------------------------------------------------------------------------- run
def run(img: np.ndarray, p: dict, engine: str):
    if p["method"] == "jpeg":
        return run_jpeg(img, p, engine)
    return run_transform(img, p, engine)


CV2_SAMPLING = {"4:2:0": "IMWRITE_JPEG_SAMPLING_FACTOR_420", "4:2:2": "IMWRITE_JPEG_SAMPLING_FACTOR_422",
                "4:4:4": "IMWRITE_JPEG_SAMPLING_FACTOR_444"}


def encode_jpeg_cv2(img: np.ndarray, quality: int, subsampling: str) -> bytes:
    """A real JPEG file written by OpenCV (libjpeg) at the given quality and chroma subsampling."""
    if img.ndim == 2:
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])[1].tobytes()
    flags = [cv2.IMWRITE_JPEG_QUALITY, int(quality),
             cv2.IMWRITE_JPEG_SAMPLING_FACTOR, getattr(cv2, CV2_SAMPLING[subsampling])]
    return cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), flags)[1].tobytes()


def decode_cv2(data: bytes, gray: bool) -> np.ndarray:
    buf = np.frombuffer(data, np.uint8)
    if gray:
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def run_jpeg(img: np.ndarray, p: dict, engine: str):
    """JPEG-style compression. Both engines compute the coefficients with our own block DCT and
    quantisation (core/jpeg.py) for the maps and counts; the image comes from our own decoder
    (engine="numpy") or from a real libjpeg round trip (engine="opencv")."""
    quality, sub = int(p["quality"]), p["chroma_subsampling"]
    res = J.jpeg_roundtrip(img, quality, sub)
    jpeg_file = encode_jpeg_cv2(img, quality, sub)
    real = decode_cv2(jpeg_file, img.ndim == 2)
    out = res["image"] if engine == "numpy" else real

    planes = res["planes"]
    total = sum(pl["coef"].size for pl in planes)
    n_kept = int(sum(np.count_nonzero(pl["nonzero"]) for pl in planes))
    energy_total = sum(float(np.sum(np.square(pl["coef"]), dtype=np.float64)) for pl in planes)
    energy_kept = sum(float(np.sum(np.square(pl["coef"][pl["nonzero"]]), dtype=np.float64)) for pl in planes)

    y = planes[0]
    before = J.frequency_layout(y["coef"])
    peak = float(np.log1p(np.abs(before)).max())
    err = abs_error(img, out)
    amp = error_amplification(err)
    extras = {
        "coef_before": display_size(log_map(before, peak)),
        "coef_after": display_size(log_map(J.frequency_layout(y["coef_hat"]), peak)),
        "kept_mask": display_size((J.frequency_layout(y["nonzero"]) * 255).astype(np.uint8)),
        "error_map": error_map(err, amp),
    }
    metrics = {
        "method": "jpeg",
        "quality": quality,
        "chroma_subsampling": sub if img.ndim == 3 else "gray",
        "coefficients_total": int(total),
        "coefficients_kept": n_kept,
        "kept_percent": round(100.0 * n_kept / total, 3),
        "compression_ratio": round(total / max(n_kept, 1), 2),
        "psnr_db": round(Q.psnr(img, out), 2),
        "ssim": round(Q.ssim(img, out), 4),
        "energy_kept_percent": round(100.0 * energy_kept / energy_total, 3) if energy_total > 0 else 100.0,
        "error_amplification": amp,
        "jpeg_file_bytes": len(jpeg_file),
        # how closely our from-scratch decoder reproduces a real JPEG decoder (higher = closer)
        "match_with_real_jpeg_db": round(min(Q.psnr(real, res["image"]), 99.0), 2),
    }
    return out, extras, metrics, []


def run_transform(img: np.ndarray, p: dict, engine: str):
    method = p["method"]
    h, w = img.shape[:2]
    gray = img.ndim == 2
    weights = (1.0,) if gray else LUMA_WEIGHTS
    recon = []
    # Per-channel results are folded into running sums straight away (luminance coefficients,
    # energies, kept mask), so only one channel's float64 arrays are alive at a time.
    y_before = y_after = union = None
    total = n_kept = 0
    energy_total = energy_kept = 0.0
    for wt, f in zip(weights, channels(img)):
        F = forward(f, method, engine)
        G, mask = select(F, p)
        recon.append(to_uint8(inverse(G, (h, w), method, engine)))
        total += F.size
        n_kept += int(np.count_nonzero(mask))
        energy_total += float(np.sum(np.square(F, dtype=np.float64)))
        energy_kept += float(np.sum(np.square(G, dtype=np.float64)))
        y_before = wt * F if y_before is None else y_before + wt * F     # Y = 0.299 R + 0.587 G + 0.114 B
        y_after = wt * G if y_after is None else y_after + wt * G
        union = mask if union is None else union | mask
        del F, G, mask
    out = recon[0] if gray else np.dstack(recon)

    peak = float(np.log1p(np.abs(y_before)).max())
    err = abs_error(img, out)
    amp = error_amplification(err)
    extras = {
        "coef_before": display_size(log_map(y_before, peak)),
        "coef_after": display_size(log_map(y_after, peak)),   # same scale as "before"
        "kept_mask": display_size((union * 255).astype(np.uint8)),
        "error_map": error_map(err, amp),
    }
    metrics = {
        "method": method,
        "coefficients_total": int(total),
        "coefficients_kept": n_kept,
        "kept_percent": round(100.0 * n_kept / total, 3),
        "compression_ratio": round(total / max(n_kept, 1), 2),
        "psnr_db": round(Q.psnr(img, out), 2),
        "ssim": round(Q.ssim(img, out), 4),
        "energy_kept_percent": round(100.0 * energy_kept / energy_total, 3) if energy_total > 0 else 100.0,
        "error_amplification": amp,
        "size_original_png_bytes": _png_size(img),
        "size_compressed_png_bytes": _png_size(out),
        "size_compressed_jpeg_q90_bytes": _jpeg_size(out),
    }

    warnings = []
    if method == "hadamard" and (not T.is_pow2(h) or not T.is_pow2(w)):
        ph, pw = T.next_pow2(h), T.next_pow2(w)
        warnings.append(f"Hadamard needs power-of-two sizes; the image was padded from {w}×{h} "
                        f"to {pw}×{ph} and cropped back.")
    if metrics["kept_percent"] < 0.5:
        warnings.append(f"Only {metrics['kept_percent']:.2f} % of the coefficients were kept, "
                        "so the result is very blurry or blocky.")
    return out, extras, metrics, warnings
