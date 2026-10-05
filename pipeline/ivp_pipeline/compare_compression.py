"""DCT vs Walsh–Hadamard at the SAME coefficient budget, and both vs JPEG-style coding by FILE SIZE
(Compare tab → Compression).

Both whole-image methods keep the same percentage of their largest coefficients (mode
"keep_percent"), so that comparison is fair: equal number of numbers kept, different basis functions.
The DCT's smooth cosines match natural images better than the ±1 square waves of Walsh–Hadamard, so
the DCT packs more of the energy into fewer coefficients and reconstructs with a higher PSNR.

Keeping fewer coefficients does not by itself make a smaller FILE, though: the result is still a full
image that has to be stored. The size curve therefore compares real files: JPEG-style coding
(quantisation + entropy coding, 8x8 blocks) at several qualities, against the whole-image results
saved as near-lossless JPEGs (quality 95, 4:4:4), every point measured on the decoded file.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config
from . import metrics as Q
from .core import transforms as T
from .modules import compress as Z
from .pipeline import step

METHODS = ("dct", "hadamard")                 # whole-image transforms (same keep-% budget)
ALL_METHODS = METHODS + ("jpeg",)            # + JPEG-style coding at the matching quality
JPEG_QUALITIES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95)
SAVE_QUALITY, SAVE_SUBSAMPLING = 95, "4:4:4"     # how whole-image results are stored for the size curve


def _curve_copy(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Copy at most CURVE_MAX_SIDE px (INTER_AREA) used for the curves, and its scale."""
    h, w = img.shape[:2]
    if max(h, w) <= config.CURVE_MAX_SIDE:
        return img, 1.0
    s = config.CURVE_MAX_SIDE / max(h, w)
    return cv2.resize(img, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA), s


def _file_point(original: np.ndarray, img: np.ndarray, quality: int, subsampling: str) -> dict:
    """Save img as a real JPEG; return its size in bits per pixel and the decoded file's quality."""
    data = Z.encode_jpeg_cv2(img, quality, subsampling)
    shown = Z.decode_cv2(data, img.ndim == 2)
    h, w = img.shape[:2]
    return {"bytes": len(data), "bpp": round(len(data) * 8 / (h * w), 4),
            "psnr_db": round(min(Q.psnr(original, shown), 99.0), 3), "ssim": round(Q.ssim(original, shown), 4)}


def jpeg_bpp_for_quality(jpeg_points: list[dict], psnr_db: float) -> float | None:
    """Bits per pixel the JPEG-style curve needs to reach psnr_db (linear interpolation), or None."""
    pts = sorted(jpeg_points, key=lambda p: p["psnr_db"])
    xs = [p["psnr_db"] for p in pts]
    if not pts or psnr_db < xs[0] or psnr_db > xs[-1]:
        return None
    return float(np.interp(psnr_db, xs, [p["bpp"] for p in pts]))


def curves(img: np.ndarray, engine: str = "opencv", keep_percent: float | None = None) -> dict:
    """Rate–distortion, energy-compaction and file-size curves.

    Each forward transform is computed ONCE per channel and reused for every keep-percent level.
    rd_curve:     [{keep_percent, dct: {psnr_db, ssim}, hadamard: {psnr_db, ssim}}]  (before saving)
    energy_curve: [{percent, dct, hadamard}] with the share of energy (in %) captured by the
                  largest `percent` % of the coefficients (orthonormal, summed over channels).
    size_curve:   {jpeg: [{quality, bpp, psnr_db, ssim, bytes}],
                   dct / hadamard: [{keep_percent, bpp, psnr_db, ssim, bytes}]}  (real files, decoded)
    """
    small, scale = _curve_copy(img)
    h, w = small.shape[:2]
    coefs = {m: [Z.forward(f, m, engine) for f in Z.channels(small)] for m in METHODS}

    levels = sorted(set(config.RD_KEEP_PERCENTS) | ({float(keep_percent)} if keep_percent else set()))
    rd, size = [], {"jpeg": [], "dct": [], "hadamard": []}
    for p in levels:
        row = {"keep_percent": p}
        for m in METHODS:
            rec = [Z.to_uint8(Z.inverse(T.keep_percent(F, p)[0], (h, w), m, engine)) for F in coefs[m]]
            out = np.dstack(rec) if small.ndim == 3 else rec[0]
            row[m] = {"psnr_db": round(min(Q.psnr(small, out), 99.0), 3), "ssim": round(Q.ssim(small, out), 4)}
            size[m].append({"keep_percent": p, **_file_point(small, out, SAVE_QUALITY, SAVE_SUBSAMPLING)})
        if p in config.RD_KEEP_PERCENTS:
            rd.append(row)
    for q in JPEG_QUALITIES:
        size["jpeg"].append({"quality": q, **_file_point(small, small, q, "4:2:0")})

    fractions = np.logspace(-3, 0, config.ENERGY_CURVE_POINTS)
    energy = {m: T.energy_compaction(coefs[m], fractions) for m in METHODS}
    energy_curve = [{"percent": round(100 * float(q), 4), "dct": round(100 * float(energy["dct"][i]), 4),
                     "hadamard": round(100 * float(energy["hadamard"][i]), 4)}
                    for i, q in enumerate(fractions)]
    return {"rd_curve": rd, "energy_curve": energy_curve, "size_curve": size, "curve_scale": scale,
            "size_curve_note": f"Whole-image results saved as JPEG quality {SAVE_QUALITY}, {SAVE_SUBSAMPLING} "
                               "(near-lossless); JPEG-style at each quality with 4:2:0. "
                               "Every point is measured on the decoded file."}


def match_jpeg_quality(img: np.ndarray, target_psnr: float, subsampling: str = "4:2:0") -> int:
    """Lowest JPEG quality (1–100) whose decoded file reaches target_psnr (binary search; PSNR rises
    with quality). Returns 100 if even quality 100 falls short."""
    def psnr_at(q):
        return Q.psnr(img, Z.decode_cv2(Z.encode_jpeg_cv2(img, q, subsampling), img.ndim == 2))
    lo, hi = 1, 100
    if psnr_at(hi) < target_psnr:
        return 100
    while lo < hi:
        mid = (lo + hi) // 2
        if psnr_at(mid) >= target_psnr:
            hi = mid
        else:
            lo = mid + 1
    return lo


def compare_compression_prepared(img: np.ndarray, keep_percent: float = 10.0, engine: str = "opencv",
                                 jpeg_quality: int | None = None) -> dict:
    """Run DCT and Hadamard with the same keep_percent on a prepared image, JPEG-style coding, plus the curves.

    jpeg_quality=None ("auto"): JPEG-style runs at the lowest quality that matches the DCT result's PSNR,
    so the two can be compared at equal image quality. An int 1–100 uses that quality instead.

    All three error maps are re-drawn with ONE shared amplification so their brightness is comparable.
    `file_bytes` is the size of a real file: DCT / Hadamard results saved as near-lossless JPEGs
    (quality 95, 4:4:4), JPEG-style as the JPEG it produces.
    """
    res = {m: step(img, "compress", {"method": m, "mode": "keep_percent", "keep_percent": keep_percent}, engine)
           for m in METHODS}
    matched = jpeg_quality is None
    if matched:
        target = res["dct"]["metrics"]["psnr_db"]
        q = match_jpeg_quality(img, target if np.isfinite(target) else 99.0)
    else:
        q = int(jpeg_quality)
    res["jpeg"] = step(img, "compress", {"method": "jpeg", "quality": q, "chroma_subsampling": "4:2:0"}, engine)

    for m in METHODS:
        res[m]["metrics"]["file_bytes"] = len(Z.encode_jpeg_cv2(res[m]["image"], SAVE_QUALITY, SAVE_SUBSAMPLING))
    res["jpeg"]["metrics"]["file_bytes"] = res["jpeg"]["metrics"]["jpeg_file_bytes"]

    errs = {m: Z.abs_error(img, res[m]["image"]) for m in ALL_METHODS}
    amp = Z.error_amplification(*errs.values())
    for m in ALL_METHODS:
        res[m]["extras"]["error_map"] = Z.error_map(errs[m], amp)
        res[m]["metrics"]["error_amplification"] = amp

    d, hd = res["dct"]["metrics"]["psnr_db"], res["hadamard"]["metrics"]["psnr_db"]
    if not (np.isfinite(d) and np.isfinite(hd)):
        winner, margin = "tie", 0.0
    else:
        margin = round(abs(d - hd), 2)
        winner = "tie" if margin < 0.05 else ("dct" if d > hd else "hadamard")
    out = {m: res[m] for m in ALL_METHODS}
    out.update(curves(img, engine, keep_percent))
    out["keep_percent"] = keep_percent

    # Same quality, how many bytes? DCT at the user's setting vs the JPEG-style curve.
    dct_pt = next(p for p in out["size_curve"]["dct"] if p["keep_percent"] == float(keep_percent))
    jpeg_bpp = jpeg_bpp_for_quality(out["size_curve"]["jpeg"], dct_pt["psnr_db"])
    out["summary"] = {
        "winner": winner, "psnr_margin_db": margin, "keep_percent": keep_percent,
        "psnr_db": {"dct": d, "hadamard": hd},
        "jpeg_vs_dct": {
            "dct_bpp": dct_pt["bpp"], "dct_file_psnr_db": dct_pt["psnr_db"],
            "jpeg_bpp_same_quality": None if jpeg_bpp is None else round(jpeg_bpp, 4),
            "times_smaller": None if not jpeg_bpp else round(dct_pt["bpp"] / jpeg_bpp, 2),
        },
        # the full-size images: DCT result vs JPEG-style at the quality matching its PSNR
        "jpeg": {
            "quality": q, "matched_to_dct": matched, "psnr_db": res["jpeg"]["metrics"]["psnr_db"],
            "dct_psnr_db": res["dct"]["metrics"]["psnr_db"],
            "file_bytes": res["jpeg"]["metrics"]["file_bytes"], "dct_file_bytes": res["dct"]["metrics"]["file_bytes"],
            "times_smaller_than_dct": round(res["dct"]["metrics"]["file_bytes"] / max(1, res["jpeg"]["metrics"]["file_bytes"]), 2),
        },
    }
    return out
