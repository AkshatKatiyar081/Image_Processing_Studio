"""Study: why noise in SCREENSHOTS is missed, and which multi-scale measures catch it.

A screenshot of a noisy web image is usually RESIZED (browser zoom, display scaling 125–150 %), often after
the site saved it as a lossy JPEG. Enlarging smears each noise grain / speck over 2–3 pixels, so the
pixel-scale measures (Immerkær σ, 3x3-median impulse test) stop seeing it.

This script builds labelled images: clean sources × {clean, Gaussian σ 8/15/25, salt & pepper 1/5/10/20 %}
× capture {native, web JPEG q75, shown 0.6x, shown 1.5x, shown 2x, shown 1.5x + saved JPEG q90}, and prints
per-capture distributions of the candidate measures. Its output chose the multi-scale rules in
ivp_pipeline.metrics / config (run calibrate_gate.py for the final confusion matrices).

Run:  python experiments/noise_scales_study.py
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from common import RESULTS, natural_images, synthetic_scenes, texture_images, to_u8_gray

from ivp_pipeline import metrics as Q
from ivp_pipeline import synthetic as S

DATASET = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dataset")
PHOTOS = ["flower.jpg", "test.jpg", "image.png", "coin.png", "low_con.jpg", "bright_image.jpg", "dark_image.jpg"]
SCALES = (1, 2, 3)


def dataset_photos(max_side: int = 768) -> dict[str, np.ndarray]:
    out = {}
    for name in PHOTOS:
        p = os.path.join(DATASET, name)
        if not os.path.exists(p):
            continue
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        s = min(1.0, max_side / max(g.shape))
        if s < 1:
            g = cv2.resize(g, (round(g.shape[1] * s), round(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
        out["ds_" + os.path.splitext(name)[0]] = g
    return out


def sources() -> dict[str, np.ndarray]:
    src = {**natural_images(), **texture_images(), **dataset_photos()}
    src.update({n: im for n, im, _ in synthetic_scenes(6, texture=4.0)})
    return src


def jpeg(img, q):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)


def resize(img, f):
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)), interpolation=interp)


CAPTURES = {
    "native": lambda im: im,
    "web_jpeg": lambda im: jpeg(im, 75),
    "shown_0.6x": lambda im: resize(jpeg(im, 75), 0.6),
    "shown_1.5x": lambda im: resize(jpeg(im, 75), 1.5),
    "shown_2x": lambda im: resize(jpeg(im, 75), 2.0),
    "shown_1.5x_jpeg": lambda im: jpeg(resize(jpeg(im, 75), 1.5), 90),
}
DEGRADATIONS = {"clean": lambda im: im}
DEGRADATIONS.update({f"gauss{s}": (lambda s: lambda im: S.gaussian_noise(im, s, seed=1))(s) for s in (8, 15, 25)})
DEGRADATIONS.update({f"sp{int(d * 100)}": (lambda d: lambda im: S.salt_pepper(im, d, seed=2))(d)
                     for d in (0.01, 0.05, 0.10, 0.20)})


def decimate(g, f):
    return np.ascontiguousarray(g[::f, ::f]) if f > 1 else g


def speck_density(g, diff=50, max_area=2):
    """Pixels that differ from their 3x3 median by > diff AND form tiny isolated groups (≤ max_area px).
    Unlike impulse_density it does NOT require pure black/white, so softened specks count; the size test
    keeps edges and text strokes (long connected outliers) out."""
    out = (cv2.absdiff(g, cv2.medianBlur(g, 3)) > diff).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_area
    return float(small[lab].mean())


def features(g):
    f = {}
    for s in SCALES:
        d = decimate(g, s)
        f[f"sigma_{s}"] = Q.noise_sigma(d)
        f[f"impulse_{s}"] = Q.impulse_density(d)
        f[f"speck_{s}"] = speck_density(d)
        for diff in (70, 90):
            f[f"speck{diff}_{s}"] = speck_density(d, diff)
    return f


def main():
    rows = []
    for name, im in sources().items():
        im = to_u8_gray(im)
        for dname, deg in DEGRADATIONS.items():
            noisy = deg(im)
            for cname, cap in CAPTURES.items():
                rows.append({"source": name, "deg": dname, "capture": cname, **features(cap(noisy))})
    # optional: a real screenshot of a noisy web image (not part of the repository), via NOISY_SCREENSHOT=path
    shot = os.getenv("NOISY_SCREENSHOT")
    if shot and os.path.exists(shot):
        rows.append({"source": "real_screenshot", "deg": "sp_real", "capture": "screenshot",
                     **features(cv2.imread(shot, cv2.IMREAD_GRAYSCALE))})
    import csv
    path = os.path.join(RESULTS, "noise_scales_study.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} images → {path}")


if __name__ == "__main__":
    main()
