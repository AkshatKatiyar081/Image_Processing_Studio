"""One-click auto enhance for the casual view (spec §11)."""
from __future__ import annotations

import numpy as np

from . import config as C
from .diagnostics import diagnose_prepared, gaussian_noise, impulse_noise
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

    imp = impulse_noise(d)
    if imp:
        k = imp["median_ksize"]
        apply("denoise", {"filter_type": "median", "kernel_size": k},
              f"salt-and-pepper noise on {d['impulse_density']:.2%} of pixels" if imp["scale"] == "pixel" else
              f"enlarged salt-and-pepper specks ({imp['value']:.1%} at half scale, e.g. a screenshot): "
              f"{k}×{k} median, a 3×3 one is too small for them")
        skip_sharpen = True
    gn = gaussian_noise(d)       # measured again after the median: removing specks usually lowers σ too
    if gn:
        # pixel scale: σ 1.0 for strong noise, a lighter σ 0.6 for moderate noise (keeps fine detail);
        # enlarged grain (2–3 px wide): σ 1.5
        if gn["scale"] == "pixel":
            reason = (f"sensor noise σ ≈ {d['noise_sigma']:.1f}" if gn["strong"] else
                      f"moderate sensor noise σ ≈ {d['noise_sigma']:.1f} (light smoothing)")
        else:
            reason = f"coarse grain σ ≈ {gn['value']:.1f} at half scale (enlarged noise, e.g. a screenshot)"
        apply("denoise", {"filter_type": "gaussian", "sigma": gn["sigma"]}, reason)
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
              "mild unsharp masking for crispness (skipped whenever noise was found, at any scale)")

    return {
        "image": cur,
        "steps": steps,
        "diagnosis_before": before,
        "diagnosis_after": diagnose_prepared(cur),
        "histograms": {"before": hist_list(img), "after": hist_list(cur)},
    }
