"""JPEG-style transform coding from scratch: the lossy part of baseline JPEG (ITU-T T.81).

    RGB → YCbCr → chroma subsampling → split into 8x8 blocks → level shift (−128) → 2-D DCT
        → quantise: q = round(F / Q) → dequantise: F' = q · Q → inverse DCT → +128 → upsample → RGB

Quantisation is where the information is thrown away: every coefficient is divided by a step size
from a quantisation table and rounded, so small high-frequency coefficients become 0. The tables
are the standard IJG ones (Annex K of the standard), scaled by the quality setting 1–100.
A real JPEG file then entropy-codes the integers q (zig-zag order, run-lengths, Huffman); that step
is lossless, so this module reproduces the image a JPEG decoder shows, up to small rounding and
chroma-upsampling differences between implementations.

Pixel planes and coefficients are float32: plenty for 8-bit images (real codecs use integers), and
half the memory traffic of float64 on multi-megapixel photos.
"""
from __future__ import annotations

import numpy as np

from .transforms import dct_matrix

# Annex K, Table K.1 (luminance) and K.2 (chrominance): step sizes at quality 50.
LUMA_Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]], np.float64)
CHROMA_Q50 = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]], np.float64)

# (vertical, horizontal) chroma subsampling factors
SUBSAMPLING = {"4:4:4": (1, 1), "4:2:2": (1, 2), "4:2:0": (2, 2)}


def quality_scale(quality: int) -> float:
    """IJG scaling: q < 50 → 5000 / q, otherwise 200 − 2q (in percent of the quality-50 table)."""
    q = min(max(int(quality), 1), 100)
    return 5000.0 / q if q < 50 else 200.0 - 2.0 * q


def quant_table(base: np.ndarray, quality: int) -> np.ndarray:
    """Q = floor((base · scale + 50) / 100), clipped to 1..255 (baseline JPEG)."""
    return np.clip(np.floor((base * quality_scale(quality) + 50.0) / 100.0), 1, 255)


def estimate_quality(luma_table) -> int:
    """Quality 1–100 whose scaled standard luminance table is closest to a file's table."""
    t = np.sort(np.asarray(luma_table, np.float64).ravel())
    errors = [np.abs(np.sort(quant_table(LUMA_Q50, q).ravel()) - t).sum() for q in range(1, 101)]
    return int(np.argmin(errors)) + 1


def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JFIF full-range YCbCr: Y = 0.299R + 0.587G + 0.114B, Cb/Cr centred on 128 (three planes)."""
    r, g, b = (rgb[..., c].astype(np.float32) for c in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_ycbcr, rounded and clipped to a uint8 RGB image."""
    cb, cr = cb - 128.0, cr - 128.0
    return np.stack([to_u8(y + 1.402 * cr), to_u8(y - 0.344136 * cb - 0.714136 * cr), to_u8(y + 1.772 * cb)], axis=-1)


def _pad_to(x: np.ndarray, mh: int, mw: int) -> np.ndarray:
    """Replicate the last row/column so the size is a multiple of (mh, mw), as JPEG encoders do."""
    h, w = x.shape
    return np.pad(x, ((0, (-h) % mh), (0, (-w) % mw)), mode="edge")


def downsample(c: np.ndarray, fy: int, fx: int) -> np.ndarray:
    """Average fy x fx blocks of a chroma plane (its size must be a multiple of fy, fx)."""
    h, w = c.shape
    return c.reshape(h // fy, fy, w // fx, fx).mean(axis=(1, 3))


def _upsample2(c: np.ndarray, axis: int) -> np.ndarray:
    """Double one axis with a triangular filter (libjpeg "fancy upsampling"):
    the two outputs of sample i are 3/4·c[i] + 1/4·c[i−1] and 3/4·c[i] + 1/4·c[i+1] (edges replicated).
    This is smoother than repeating each sample and matches what real JPEG decoders show."""
    c = np.moveaxis(c, axis, 0)
    prev = np.concatenate([c[:1], c[:-1]])
    nxt = np.concatenate([c[1:], c[-1:]])
    out = np.empty((2 * c.shape[0],) + c.shape[1:], c.dtype)
    out[0::2] = 0.75 * c + 0.25 * prev
    out[1::2] = 0.75 * c + 0.25 * nxt
    return np.moveaxis(out, 0, axis)


def upsample(c: np.ndarray, fy: int, fx: int) -> np.ndarray:
    """Bring a subsampled chroma plane back to full size (factor 1 or 2 per axis)."""
    if fy == 2:
        c = _upsample2(c, 0)
    if fx == 2:
        c = _upsample2(c, 1)
    return c


def to_blocks(x: np.ndarray) -> np.ndarray:
    """(H, W) with H, W multiples of 8 → (H/8, W/8, 8, 8)."""
    h, w = x.shape
    return x.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)


def from_blocks(b: np.ndarray) -> np.ndarray:
    n, m = b.shape[:2]
    return b.transpose(0, 2, 1, 3).reshape(n * 8, m * 8)


def block_dct(blocks: np.ndarray) -> np.ndarray:
    """2-D DCT of every 8x8 block at once: F = C · B · Cᵀ (orthonormal DCT-II, as in JPEG)."""
    c = dct_matrix(8).astype(blocks.dtype)
    return c @ blocks @ c.T


def block_idct(coefs: np.ndarray) -> np.ndarray:
    c = dct_matrix(8).astype(coefs.dtype)
    return c.T @ coefs @ c


def frequency_layout(coefs: np.ndarray) -> np.ndarray:
    """Rearrange (H/8, W/8, 8, 8) block coefficients so all coefficients of frequency (u, v) form
    one tile: the result is 8 x 8 tiles of size (H/8, W/8), with the DC tile (a tiny copy of the
    image) at the top-left and the highest frequencies at the bottom-right."""
    n, m = coefs.shape[:2]
    return coefs.transpose(2, 0, 3, 1).reshape(8 * n, 8 * m)


def code_plane(plane: np.ndarray, table: np.ndarray) -> dict:
    """Level shift, block DCT, quantise, dequantise and inverse DCT one plane (size multiple of 8)."""
    f = block_dct(to_blocks(plane - np.float32(128.0)))
    table = table.astype(f.dtype)
    q = np.rint(f / table)
    f_hat = q * table
    rec = from_blocks(block_idct(f_hat)) + np.float32(128.0)
    return {"coef": f, "coef_hat": f_hat, "nonzero": q != 0, "rec": rec}


def jpeg_roundtrip(img: np.ndarray, quality: int = 75, subsampling: str = "4:2:0") -> dict:
    """Run the lossy JPEG steps on a uint8 RGB or gray image.

    Returns {"image": uint8 reconstruction, "planes": [code_plane results, Y first],
    "luma_table", "chroma_table"}.
    """
    fy, fx = SUBSAMPLING[subsampling]
    lq, cq = quant_table(LUMA_Q50, quality), quant_table(CHROMA_Q50, quality)
    h, w = img.shape[:2]
    if img.ndim == 2:
        y = code_plane(_pad_to(img.astype(np.float32), 8, 8), lq)
        return {"image": to_u8(y["rec"][:h, :w]), "planes": [y], "luma_table": lq, "chroma_table": cq}

    my, mx = 8 * fy, 8 * fx                        # MCU size: chroma blocks must also be whole
    planes = [_pad_to(c, my, mx) for c in rgb_to_ycbcr(img)]
    y = code_plane(planes[0], lq)
    chroma = [code_plane(downsample(p, fy, fx), cq) for p in planes[1:]]
    del planes
    cb, cr = (upsample(c["rec"], fy, fx)[:h, :w] for c in chroma)
    out = ycbcr_to_rgb(y["rec"][:h, :w], cb, cr)
    return {"image": out, "planes": [y] + chroma, "luma_table": lq, "chroma_table": cq}
