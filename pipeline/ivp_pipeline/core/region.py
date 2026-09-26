"""Seeded region growing from scratch (Unit 6 / Exp 09).

Iterative breadth-first search with collections.deque — never recursion, because
Python's recursion limit (~1000) breaks on regions of a few thousand pixels.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def _neighbour_steps(connectivity: int):
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    raise ValueError("connectivity must be 4 or 8")


def bfs_region(accept: np.ndarray, seeds, connectivity: int = 4) -> np.ndarray:
    """Connected set of `accept` pixels reachable from `seeds` (flat indices).

    Seeds are always included, even if accept[seed] is False.
    """
    h, w = accept.shape
    acc = accept.ravel()
    visited = np.zeros(h * w, dtype=bool)
    steps = _neighbour_steps(connectivity)
    q = deque()
    for s in np.atleast_1d(seeds):
        s = int(s)
        if not visited[s]:
            visited[s] = True
            q.append(s)
    while q:
        idx = q.popleft()
        y, x = divmod(idx, w)
        for dy, dx in steps:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                n = ny * w + nx
                if not visited[n] and acc[n]:
                    visited[n] = True
                    q.append(n)
    return visited.reshape(h, w)


def seed_value(gray: np.ndarray, x: int, y: int, mode: str = "median3") -> float:
    """Reference intensity f_s around the seed.

    median3 (default): 3x3 median — a salt/pepper spike at or next to the click cannot poison it.
    mean3: 3x3 mean. pixel: the seed pixel itself (matches skimage.segmentation.flood).
    """
    if mode == "pixel":
        return float(gray[y, x])
    y0, y1 = max(0, y - 1), min(gray.shape[0], y + 2)
    x0, x1 = max(0, x - 1), min(gray.shape[1], x + 2)
    win = gray[y0:y1, x0:x1]
    return float(np.median(win)) if mode == "median3" else float(win.mean())


def region_grow(gray: np.ndarray, seed_xy: tuple[int, int], delta: float,
                connectivity: int = 4, adaptive: bool = False,
                seed_mode: str = "median3") -> np.ndarray:
    """Grow a region from seed (x, y) = (column, row).

    Fixed predicate:    Q(p) = |f(p) - f_s| <= delta
    Adaptive predicate: Q(p) = |f(p) - mean(R)| <= delta, with mean(R) updated as pixels join.
    Returns a bool mask.
    """
    h, w = gray.shape
    x, y = int(seed_xy[0]), int(seed_xy[1])
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"seed {seed_xy} is outside the {w}x{h} image")
    fs = seed_value(gray, x, y, seed_mode)
    seed = y * w + x

    if not adaptive:
        accept = np.abs(gray.astype(np.float32) - fs) <= delta
        return bfs_region(accept, [seed], connectivity)

    g = gray.ravel().astype(np.float64)
    visited = np.zeros(h * w, dtype=bool)
    in_region = np.zeros(h * w, dtype=bool)
    steps = _neighbour_steps(connectivity)
    total, count = fs, 1  # region statistics start from the seed reference value
    visited[seed] = in_region[seed] = True
    q = deque([seed])
    while q:
        idx = q.popleft()
        cy, cx = divmod(idx, w)
        for dy, dx in steps:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                n = ny * w + nx
                if not visited[n]:
                    visited[n] = True  # always mark on first encounter to avoid re-checking
                    if abs(g[n] - total / count) <= delta:
                        in_region[n] = True
                        total += g[n]
                        count += 1
                        q.append(n)
    return in_region.reshape(h, w)
