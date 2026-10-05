"""One-click auto enhance for the casual view (spec §11)."""
from __future__ import annotations

import numpy as np

from . import config as C
from .diagnostics import diagnose_prepared
from .pipeline import step
from .utils import hist_list


def auto_enhance_prepared(img: np.ndarray, engine: str) -> dict:
    """img must come from utils.prepare(), whose impulse-aware resize keeps salt-and-pepper visible."""
    before = diagnose_prepared(img)
    d = before["metrics"]

    cur, steps, skip_sharpen = img, [], False

    def apply(module, params, reason):
        nonlocal cur, d
        res = step(cur, module, params, engine)
        cur = res["image"]
        steps.append({"module": module, "params": res["meta"]["params"], "reason": reason,
                      "metrics": res["metrics"]})
        d = diagnose_prepared(cur)["metrics"]

    if d["impulse_density"] > C.IMPULSE_DENSITY:
        apply("denoise", {"filter_type": "median", "kernel_size": 5 if d["impulse_density"] > C.IMPULSE_HEAVY else 3},
              f"salt-and-pepper noise on {d['impulse_density']:.2%} of pixels")
        skip_sharpen = True
    if d["noise_sigma"] > C.NOISE_STRONG:
        apply("denoise", {"filter_type": "gaussian", "sigma": 1.0},
              f"sensor noise σ ≈ {d['noise_sigma']:.1f}")
        skip_sharpen = True
    elif d["noise_sigma"] > C.NOISE_MODERATE:
        # Moderate noise (σ in NOISE_MODERATE..NOISE_STRONG): apply a lighter
        # Gaussian (σ 0.6) instead of skipping denoising entirely.  This is gentler
        # than the strong-noise path (σ 1.0) so fine detail is preserved while
        # grain visible to the eye is still meaningfully reduced.
        apply("denoise", {"filter_type": "gaussian", "sigma": 0.6},
              f"moderate sensor noise σ ≈ {d['noise_sigma']:.1f} (light smoothing)")
        skip_sharpen = True
    if d["mean"] < C.DARK_MEAN or d["mean"] > C.BRIGHT_MEAN:
        apply("contrast", {"mode": "gamma", "gamma": "auto"},
              f"{'under' if d['mean'] < C.DARK_MEAN else 'over'}exposed (mean {d['mean']:.2f})")
    if d["range_p1_p99"] < C.LOW_RANGE:
        apply("contrast", {"mode": "stretch", "clip_percentile": C.AUTO_STRETCH_CLIP},
              f"narrow intensity range ({d['range_p1_p99']:.0f} levels)")
    if not skip_sharpen:
        apply("sharpen_edges", {"operation": "unsharp", "k": C.AUTO_UNSHARP_K,
                                "blur_sigma": C.AUTO_UNSHARP_SIGMA},
              "mild unsharp masking for crispness (skipped whenever noise was found)")

    return {
        "image": cur,
        "steps": steps,
        "diagnosis_before": before,
        "diagnosis_after": diagnose_prepared(cur),
        "histograms": {"before": hist_list(img), "after": hist_list(cur)},
    }
