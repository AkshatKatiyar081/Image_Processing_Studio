"""Shared image sets and plotting style for the experiments."""
from __future__ import annotations

import os
import sys

import numpy as np
from skimage import color, data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ivp_pipeline import synthetic as S  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

# Validated categorical slots 1-2 (dataviz reference palette, light mode)
DIRECT_COLOR = "#eb6834"   # orange  — Branch A (direct)
GUIDED_COLOR = "#2a78d6"   # blue    — Branch B (guided)
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def to_u8_gray(img):
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])
    if img.dtype != np.uint8:
        img = np.clip(np.rint(img * 255 if img.max() <= 1 else img), 0, 255).astype(np.uint8)
    return img


def natural_images() -> dict[str, np.ndarray]:
    """Clean real photos from skimage.data (grayscale)."""
    names = ["camera", "coins", "moon", "page", "astronaut", "coffee", "chelsea", "rocket"]
    return {n: to_u8_gray(getattr(data, n)()) for n in names}


def texture_images() -> dict[str, np.ndarray]:
    """Clean but highly textured images — the classic false-alarm case for noise detectors."""
    return {n: to_u8_gray(getattr(data, n)()) for n in ["brick", "grass", "gravel"]}


def synthetic_scenes(n: int = 10, texture: float = 0.0):
    """[(name, image, mask)] alternating bright/dark objects."""
    out = []
    for s in range(n):
        pol = "bright" if s % 2 == 0 else "dark"
        img, gt = S.scene(seed=100 + s, polarity=pol, texture=texture)
        out.append((f"scene{s}_{pol}", img, gt))
    return out


def style_axes(ax):
    import matplotlib.pyplot as plt  # noqa: F401
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
