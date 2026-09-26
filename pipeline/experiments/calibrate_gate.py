"""Spec §13.3 — calibrate the quality-gate thresholds on images with known labels.

For each diagnostic (noise σ, impulse density ρ, illumination unevenness U_bg):
  positives = clean images degraded with THAT defect,
  negatives = clean images + images degraded with OTHER defects (the detector must ignore those).
We sweep the threshold, report the one with the best F1 and the confusion matrix at the
current config value. Writes results/calibration.md and results/calibration.csv.

Run:  python experiments/calibrate_gate.py
"""
from __future__ import annotations

import csv
import os

import numpy as np

from common import RESULTS, natural_images, synthetic_scenes, texture_images

from ivp_pipeline import config as C
from ivp_pipeline import metrics as Q
from ivp_pipeline import synthetic as S


def build_samples():
    clean = list(natural_images().items()) + list(texture_images().items())
    clean += [(n, im) for n, im, _ in synthetic_scenes(6, texture=4.0)]
    samples = []   # (source, degradation, image)
    for name, im in clean:
        samples.append((name, "clean", im))
        for s in (8, 15, 25):
            samples.append((name, f"gauss{s}", S.gaussian_noise(im, s, seed=1)))
        for d in (0.01, 0.05, 0.10, 0.20):
            samples.append((name, f"sp{int(d * 100)}", S.salt_pepper(im, d, seed=2)))
        for r in (40, 70):
            samples.append((name, f"ramp{r}", S.illumination_ramp(im, r)))
        samples.append((name, "lowcontrast", S.low_contrast(im)))
    return samples


def label(deg: str, detector: str):
    """True / False = positive / negative; None = excluded (the defect this detector can't be judged on)."""
    if detector == "noise_sigma":
        if deg.startswith("gauss"):
            return True
        return None if deg.startswith("sp") else False   # impulses legitimately inflate σ: excluded
    if detector == "impulse_density":
        return deg.startswith("sp")
    if detector == "illumination_u":
        return deg.startswith("ramp")
    raise ValueError(detector)


def score(detector: str, img: np.ndarray) -> float:
    if detector == "noise_sigma":
        return Q.noise_sigma(img)
    if detector == "impulse_density":
        return Q.impulse_density(img)
    return Q.illumination_unevenness(img)[0]


def confusion(values, labels, t):
    pred = values > t
    tp = int((pred & labels).sum()); fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum()); tn = int((~pred & ~labels).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1}


def main():
    samples = build_samples()
    detectors = {"noise_sigma": C.NOISE_MODERATE, "impulse_density": C.IMPULSE_DENSITY,
                 "illumination_u": C.ILLUMINATION_U}
    rows, md = [], ["# Quality-gate calibration", "",
                    f"{len(samples)} labelled images: 8 natural + 3 texture + 6 synthetic sources, each clean "
                    "and degraded (Gaussian σ 8/15/25, salt & pepper 1/5/10/20 %, ramp ±40/±70, low contrast).", ""]
    for det, current in detectors.items():
        vals, labs, srcs = [], [], []
        for src, deg, im in samples:
            lab = label(deg, det)
            if lab is None:
                continue
            vals.append(score(det, im)); labs.append(lab); srcs.append((src, deg))
        vals, labs = np.array(vals), np.array(labs)
        cands = np.unique(np.concatenate([vals, [current]]))
        best_t, best = current, confusion(vals, labs, current)
        for t in cands:
            c = confusion(vals, labs, t)
            if c["f1"] > best["f1"] + 1e-9:
                best_t, best = float(t), c
        # prefer a threshold halfway into the gap above the best cut (more robust than the exact value)
        above = vals[vals > best_t]
        suggested = (best_t + above.min()) / 2 if above.size else best_t
        cur = confusion(vals, labs, current)
        rows.append({"detector": det, "current": current, "suggested": round(float(suggested), 5),
                     **{f"current_{k}": round(v, 3) for k, v in cur.items()},
                     **{f"best_{k}": round(v, 3) for k, v in best.items()}})
        md += [f"## `{det}`", "",
               f"Positive range: {vals[labs].min():.4g} – {vals[labs].max():.4g}; "
               f"negative range: {vals[~labs].min():.4g} – {vals[~labs].max():.4g}", "",
               "| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|---|",
               f"| current = {current:g} | {cur['tp']} | {cur['fp']} | {cur['fn']} | {cur['tn']} | "
               f"{cur['precision']:.3f} | {cur['recall']:.3f} | {cur['f1']:.3f} |",
               f"| best-F1 ≈ {suggested:.4g} | {best['tp']} | {best['fp']} | {best['fn']} | {best['tn']} | "
               f"{best['precision']:.3f} | {best['recall']:.3f} | {best['f1']:.3f} |", ""]
        fps = [f"{s}/{d}" for (s, d), v, l in zip(srcs, vals, labs) if (v > current) and not l]
        fns = [f"{s}/{d}" for (s, d), v, l in zip(srcs, vals, labs) if (v <= current) and l]
        if fps:
            md.append(f"False alarms at current threshold: {', '.join(fps[:20])}{' …' if len(fps) > 20 else ''}")
        if fns:
            md.append(f"Misses at current threshold: {', '.join(fns[:20])}{' …' if len(fns) > 20 else ''}")
        md.append("")

    md += ["## Interpretation", "",
           "* **Noise σ** separates cleanly; the only false alarms come from the *grass* texture, whose "
           "fine random detail is statistically similar to noise (a known limit of any single-image estimator).",
           "* **Impulse density**: the calibrated threshold (0.75 %) is now in `config.py`. Heavy Gaussian "
           "noise (σ = 25) produces a few genuine 0/255 outliers, which is where the remaining overlap comes from.",
           "* **Illumination U_bg** has perfect recall but low precision on natural photos: large scene regions "
           "(sky vs ground) look like uneven lighting to a background estimate. Note that *coins* and *page* "
           "are counted as false alarms only because their clean versions are labelled negative — both are "
           "textbook examples of genuinely uneven illumination. The threshold is kept at 0.15 on purpose: "
           "verify-after-fix turns false alarms into a harmless `no_benefit` verdict, while a miss would hide "
           "the largest improvement the pipeline can make.", ""]
    with open(os.path.join(RESULTS, "calibration.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(RESULTS, "calibration.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
