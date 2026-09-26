"""Visual Direct-vs-Guided examples for the report and viva (results/examples.png).

Run:  python experiments/make_examples.py
"""
from __future__ import annotations

import os

import numpy as np
from skimage import data

from common import INK, RESULTS

import ivp_pipeline as ivp
from ivp_pipeline import metrics as Q
from ivp_pipeline import synthetic as S


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene, gt = S.scene(101, polarity="dark", texture=4.0)
    cam = data.camera()
    cases = [
        ("Uneven lighting → Otsu", S.illumination_ramp(scene, 40), "segment", None, gt),
        ("Salt & pepper 10 % → Otsu", S.salt_pepper(scene, 0.10, seed=3), "segment", None, gt),
        ("Gaussian σ = 25 → Sobel edges", S.gaussian_noise(cam, 25, seed=4), "edges",
         {"operator": "sobel", "edge_threshold": 0.2}, None),
    ]
    fig, axes = plt.subplots(len(cases), 3, figsize=(11, 3.3 * len(cases)), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for row, (title, img, intent, params, truth) in zip(axes, cases):
        r = ivp.run_guided(img, intent, params)
        a, b = r["branch_a"], r["branch_b"]
        key = "mask" if intent == "segment" else "edge_map"
        if truth is not None:
            sa = f"IoU {Q.iou(a['extras'][key] > 0, truth):.2f}"
            sb = f"IoU {Q.iou(b['extras'][key] > 0, truth):.2f}"
        else:
            c = r["verdict"]["checks"][0]
            sa, sb = f"false edges {c['a']:.1%}", f"false edges {c['b']:.1%}"
        panels = [(img, f"Input — {title}"),
                  (a["extras"][key], f"Branch A (direct) · {sa}"),
                  (b["extras"][key], f"Branch B ({' → '.join(b['path'][:-1])}) · {sb}")]
        for ax, (im, t) in zip(row, panels):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(t, fontsize=9, color=INK, loc="left")
            ax.axis("off")
    path = os.path.join(RESULTS, "examples.png")
    fig.savefig(path, dpi=130)
    print("saved", path)


if __name__ == "__main__":
    main()
