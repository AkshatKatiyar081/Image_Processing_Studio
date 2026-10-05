"""Degradation sweep for the Compare tab: how accuracy falls as a defect gets worse, Direct vs Guided.

The user's image is degraded further at fixed levels (fixed random seed), and at every level the
guided comparison runs again. Each branch is scored against a REFERENCE result computed on the
unmodified image, so the chart shows how quickly each route breaks down:

    segment: IoU of the object mask vs the reference mask
    edges:   edge F1 (1-px tolerance) of the thresholded edge map vs the reference edge map

The x value of each point is the MEASURED defect (noise σ, speck density or unevenness U) of the
degraded image, so the user's own image (level 0) sits at its true position on the axis.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config as C
from . import metrics as Q
from . import synthetic as S
from .diagnostics import diagnose_prepared
from .errors import PipelineError
from .guided import resolve_target, run_guided_prepared
from .modules.sharpen_edges import edge_map_from

SWEEP_INTENTS = ("segment", "edges")
SWEEP_MAX_SIDE = 512
SEED = 1234

AXES = {
    "illumination": {"key": "illumination_u", "label": "Uneven lighting U", "unit": "",
                     "levels": [0, 20, 40, 70], "level_label": "ramp ± gray levels"},
    "impulse": {"key": "impulse_density", "label": "Salt & pepper density", "unit": "%",
                "levels": [0, 1, 5, 10, 20], "level_label": "% of pixels"},
    "gaussian": {"key": "noise_sigma", "label": "Noise σ", "unit": "gray levels",
                 "levels": [0, 5, 10, 20, 30], "level_label": "added σ"},
}


def _degrade(img: np.ndarray, axis: str, level: float) -> np.ndarray:
    if level == 0:
        return img
    if axis == "illumination":
        return S.illumination_ramp(img, level)
    if axis == "impulse":
        return S.salt_pepper(img, level / 100.0, seed=SEED)
    return S.gaussian_noise(img, level, seed=SEED)


def _x_value(img: np.ndarray, axis: str) -> float:
    m = diagnose_prepared(img)["metrics"]
    v = m[AXES[axis]["key"]]
    return round(float(v) * (100.0 if axis == "impulse" else 1.0), 4)


def choose_axis(intent: str, codes: set[str]) -> str:
    """Pick the degradation that matches the defect the user's image actually has."""
    if intent == "segment":
        if "UNEVEN_ILLUMINATION" in codes:
            return "illumination"
        if "IMPULSE_NOISE" in codes:
            return "impulse"
        return "gaussian"
    return "impulse" if "IMPULSE_NOISE" in codes else "gaussian"


def _sweep_copy(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Copy at most SWEEP_MAX_SIDE px, shrunk with NEAREST-neighbour.

    Nearest-neighbour keeps each pixel's own noise, so the measured noise σ and speck density of
    the copy match the full image (area averaging would roughly halve σ at 2x, and the "your image"
    point would then disagree with the health card). Lighting gradients are smooth and survive too.
    """
    h, w = img.shape[:2]
    if max(h, w) <= SWEEP_MAX_SIDE:
        return img, 1.0
    s = SWEEP_MAX_SIDE / max(h, w)
    small = cv2.resize(img, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_NEAREST)
    return small, s


def _score(intent: str, branch: dict, ref, t: float) -> float:
    if intent == "segment":
        return round(Q.iou(branch["extras"]["mask"] > 0, ref), 4)
    mag = np.hypot(branch["extras"]["gx"], branch["extras"]["gy"])
    return round(Q.edge_f1(edge_map_from(mag, t), ref, tolerance=1), 4)


def degradation_sweep_prepared(img: np.ndarray, intent: str, params: dict | None = None,
                               engine: str = "opencv") -> dict:
    if intent not in SWEEP_INTENTS:
        raise PipelineError("UNKNOWN_INTENT", f"sweep intent must be one of {list(SWEEP_INTENTS)}, got {intent!r}")
    small, sweep_scale = _sweep_copy(img)
    if params and params.get("seed") is not None and sweep_scale != 1.0:
        params = {**params, "seed": [float(params["seed"][0]) * sweep_scale, float(params["seed"][1]) * sweep_scale]}
    _, target = resolve_target(intent, params)
    t = (target.get("edge_threshold") or C.DEFAULT_EDGE_T) if intent == "edges" else None

    base = run_guided_prepared(small, intent, params, engine)
    codes = {f["code"] for f in base["diagnosis"]["findings"]}
    axis = choose_axis(intent, codes)

    best = base["branch_b"] or base["branch_a"]
    if intent == "segment":
        ref = best["extras"]["mask"] > 0
    else:
        ref = edge_map_from(np.hypot(best["extras"]["gx"], best["extras"]["gy"]), t)

    points = []
    for level in AXES[axis]["levels"]:
        degraded = _degrade(small, axis, level)
        g = base if level == 0 else run_guided_prepared(degraded, intent, params, engine)
        a = _score(intent, g["branch_a"], ref, t)
        b = _score(intent, g["branch_b"], ref, t) if g["branch_b"] is not None else a
        points.append({"level": level, "x": _x_value(degraded, axis), "direct": a, "guided": b,
                       "gate_fired": g["branch_b"] is not None})

    route = "the guided (fix-first) result" if base["branch_b"] is not None else "the direct result"
    return {
        "intent": intent,
        "axis": {"name": axis, "key": AXES[axis]["key"], "label": AXES[axis]["label"],
                 "unit": AXES[axis]["unit"], "level_label": AXES[axis]["level_label"]},
        "metric": "IoU" if intent == "segment" else "Edge F1",
        "points": points,
        "user_point_index": 0,
        "reference_note": f"Scored against {route} on your unmodified image, the best result available "
                          "without a ground truth.",
        "sweep_scale": sweep_scale,
        "edge_threshold": t,
    }
