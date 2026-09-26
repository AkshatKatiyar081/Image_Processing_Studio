"""Spec §13 — Direct (Branch A) vs Guided (Branch B) across a degradation grid, plus runtime.

Writes:
  results/benchmark.csv        every individual run
  results/benchmark.md         summary tables for the report
  results/direct_vs_guided.png headline figure
  results/runtime.md           NumPy vs OpenCV engine timings at 1024 x 1024

Run:  python experiments/run_benchmark.py            (about a minute)
"""
from __future__ import annotations

import csv
import os
import time
from collections import defaultdict

import cv2
import numpy as np
from skimage import data

from common import (DIRECT_COLOR, GUIDED_COLOR, INK, INK_MUTED, RESULTS, natural_images,
                    style_axes, synthetic_scenes)

import ivp_pipeline as ivp
from ivp_pipeline import metrics as Q
from ivp_pipeline import synthetic as S

SEG_GRID = {
    "Salt & pepper density (%)": ("sp", [1, 5, 10, 20], lambda im, v: S.salt_pepper(im, v / 100, seed=v)),
    "Gaussian noise σ": ("gauss", [5, 10, 20, 30], lambda im, v: S.gaussian_noise(im, v, seed=v)),
    "Illumination ramp (± gray levels)": ("ramp", [20, 40, 70], lambda im, v: S.illumination_ramp(im, v)),
}
EDGE_IMAGES = ["camera", "coins", "astronaut", "coffee", "chelsea"]


def seg_benchmark(rows):
    scenes = synthetic_scenes(10, texture=4.0)
    for axis, (key, levels, fn) in SEG_GRID.items():
        for v in levels:
            for name, img, gt in scenes:
                r = ivp.run_guided(fn(img, v), "segment")
                a = Q.iou(r["branch_a"]["extras"]["mask"] > 0, gt)
                b_res = r["branch_b"] or r["branch_a"]
                b = Q.iou(b_res["extras"]["mask"] > 0, gt)
                rows.append({"task": "segment", "degradation": key, "level": v, "image": name,
                             "metric": "IoU", "direct": round(a, 4), "guided": round(b, 4),
                             "gate_fired": not r["gate_passed"], "verdict": r["verdict"]["result"]})


def edge_benchmark(rows):
    nat = natural_images()
    params = {"operator": "sobel", "edge_threshold": 0.2}
    for name in EDGE_IMAGES:
        clean = nat[name]
        ref = ivp.run_module(clean, "sharpen_edges", {"operation": "gradient", **params})["extras"]["edge_map"] > 0
        for key, levels, fn in [("gauss", [5, 10, 20, 30], lambda im, v: S.gaussian_noise(im, v, seed=v)),
                                ("sp", [1, 5], lambda im, v: S.salt_pepper(im, v / 100, seed=v))]:
            for v in levels:
                r = ivp.run_guided(fn(clean, v), "edges", params)
                def f1(res):
                    mag = np.hypot(res["extras"]["gx"], res["extras"]["gy"])
                    return Q.edge_f1(mag > 0.2 * mag.max(), ref)
                a = f1(r["branch_a"])
                b = f1(r["branch_b"]) if r["branch_b"] else a
                rows.append({"task": "edges", "degradation": key, "level": v, "image": name,
                             "metric": "edge F1", "direct": round(a, 4), "guided": round(b, 4),
                             "gate_fired": not r["gate_passed"], "verdict": r["verdict"]["result"]})


def reference_benchmark(rows):
    """Sharpen and equalize: PSNR against the same operation applied to the clean image."""
    nat = natural_images()
    for name in EDGE_IMAGES:
        clean = nat[name]
        ref = ivp.run_module(clean, "sharpen_edges", {"operation": "unsharp"})["image"]
        for v in (5, 10, 20):
            r = ivp.run_guided(S.gaussian_noise(clean, v, seed=v), "sharpen")
            a = Q.psnr(ref, r["branch_a"]["image"])
            b = Q.psnr(ref, (r["branch_b"] or r["branch_a"])["image"])
            rows.append({"task": "sharpen", "degradation": "gauss", "level": v, "image": name,
                         "metric": "PSNR dB", "direct": round(a, 2), "guided": round(b, 2),
                         "gate_fired": not r["gate_passed"], "verdict": r["verdict"]["result"]})
        dark = S.apply_gamma(clean, 3.0)
        ref = ivp.run_module(dark, "contrast", {"mode": "hist_eq"})["image"]
        for v in (5, 10, 15):
            r = ivp.run_guided(S.gaussian_noise(dark, v, seed=v), "equalize")
            a = Q.psnr(ref, r["branch_a"]["image"])
            b = Q.psnr(ref, (r["branch_b"] or r["branch_a"])["image"])
            rows.append({"task": "equalize", "degradation": "gauss_dark", "level": v, "image": name,
                         "metric": "PSNR dB", "direct": round(a, 2), "guided": round(b, 2),
                         "gate_fired": not r["gate_passed"], "verdict": r["verdict"]["result"]})


def summarise(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["degradation"], r["level"], r["metric"])].append(r)
    table = []
    for (task, deg, lvl, metric), rs in groups.items():
        table.append({"task": task, "degradation": deg, "level": lvl, "metric": metric, "n": len(rs),
                      "direct": float(np.mean([x["direct"] for x in rs])),
                      "guided": float(np.mean([x["guided"] for x in rs])),
                      "gate_fired": float(np.mean([x["gate_fired"] for x in rs])),
                      "recommended": float(np.mean([x["verdict"] == "recommended" for x in rs]))})
    return table


def write_markdown(table):
    md = ["# Direct vs Guided — benchmark", "",
          "Mean over images at each level. *Gate fired* = share of images where the quality gate "
          "prescribed a prerequisite; *recommended* = share where verify-after-fix confirmed the gain. "
          "When the gate does not fire, Guided = Direct.", ""]
    for task, title in [("segment", "Segmentation (10 synthetic scenes, exact masks) — IoU"),
                        ("edges", "Edge detection (5 natural images; reference = Sobel edges of the clean image) — F1"),
                        ("sharpen", "Sharpening (reference = unsharp mask of the clean image) — PSNR dB"),
                        ("equalize", "HE of a dark image (reference = HE of the clean dark image) — PSNR dB")]:
        md += [f"## {title}", "", "| Degradation | Level | Direct | Guided | Gain | Gate fired | Recommended |",
               "|---|---|---|---|---|---|---|"]
        for t in [t for t in table if t["task"] == task]:
            fmt = "{:.3f}" if t["metric"] != "PSNR dB" else "{:.2f}"
            md.append(f"| {t['degradation']} | {t['level']} | {fmt.format(t['direct'])} | "
                      f"{fmt.format(t['guided'])} | {fmt.format(t['guided'] - t['direct'])} | "
                      f"{t['gate_fired']:.0%} | {t['recommended']:.0%} |")
        md.append("")
    with open(os.path.join(RESULTS, "benchmark.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


def plot(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("segment", key, axis, "IoU") for axis, (key, _, _) in SEG_GRID.items()]
    panels.append(("edges", "gauss", "Gaussian noise σ", "Edge F1"))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    titles = ["Segmentation vs salt & pepper", "Segmentation vs Gaussian noise",
              "Segmentation vs uneven lighting", "Sobel edges vs Gaussian noise"]
    for ax, (task, key, xlabel, ylabel), title in zip(axes, panels, titles):
        pts = sorted([t for t in table if t["task"] == task and t["degradation"] == key],
                     key=lambda t: t["level"])
        x = [p["level"] for p in pts]
        for series, colour, lab in [("direct", DIRECT_COLOR, "Direct"), ("guided", GUIDED_COLOR, "Guided")]:
            y = [p[series] for p in pts]
            ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=6,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=lab)
            ax.annotate(lab, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.set_facecolor("#fcfcfb")
        ax.set_xticks(x)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(min(x) - (max(x) - min(x)) * 0.08, max(x) + (max(x) - min(x)) * 0.25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2, fontsize=9,
               labelcolor=INK_MUTED)
    fig.suptitle("Direct operator vs guided 2-stage pipeline", x=0.01, ha="left",
                 fontsize=12, color=INK)
    path = os.path.join(RESULTS, "direct_vs_guided.png")
    fig.savefig(path, dpi=160)
    print("saved", path)


def runtime():
    img = cv2.resize(data.astronaut(), (1024, 1024), interpolation=cv2.INTER_CUBIC)
    cases = [("contrast", {"mode": "hist_eq"}), ("contrast", {"mode": "flatten"}),
             ("denoise", {"filter_type": "median", "kernel_size": 5}),
             ("denoise", {"filter_type": "gaussian", "sigma": 1.5}),
             ("sharpen_edges", {"operation": "unsharp"}),
             ("sharpen_edges", {"operation": "gradient", "operator": "sobel"}),
             ("segment", {"method": "otsu"}),
             ("segment", {"method": "region_growing", "seed": [512, 700]})]
    md = ["# Runtime (1024 × 1024 RGB, median of 3 runs, 2 vCPU container)", "",
          "| Module | Params | OpenCV engine (ms) | NumPy engine (ms) |", "|---|---|---|---|"]
    for mod, p in cases:
        t = {}
        for eng in ("opencv", "numpy"):
            ts = []
            for _ in range(3):
                t0 = time.perf_counter(); ivp.run_module(img, mod, p, eng); ts.append(time.perf_counter() - t0)
            t[eng] = np.median(ts) * 1000
        md.append(f"| {mod} | `{p}` | {t['opencv']:.0f} | {t['numpy']:.0f} |")
    for intent in ("segment", "edges"):
        t0 = time.perf_counter(); ivp.run_guided(S.salt_pepper(img, 0.05), intent)
        md.append(f"| guided:{intent} | salt & pepper 5 % | {(time.perf_counter() - t0) * 1000:.0f} | – |")
    t0 = time.perf_counter(); ivp.diagnose(img)
    md.append(f"| diagnose | – | {(time.perf_counter() - t0) * 1000:.0f} | – |")
    with open(os.path.join(RESULTS, "runtime.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


def main():
    rows = []
    t0 = time.time()
    seg_benchmark(rows)
    edge_benchmark(rows)
    reference_benchmark(rows)
    with open(os.path.join(RESULTS, "benchmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    table = summarise(rows)
    write_markdown(table)
    plot(table)
    runtime()
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
