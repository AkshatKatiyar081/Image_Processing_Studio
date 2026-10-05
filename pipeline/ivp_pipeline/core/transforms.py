"""Orthonormal 2-D transforms for transform coding, from scratch.

Both transforms are written as matrix products with an orthonormal basis matrix T:

    forward  F = T_M · f · T_Nᵀ          inverse  f = T_Mᵀ · F · T_N

Because T is orthonormal (T · Tᵀ = I), the inverse is just the transpose, and Parseval's
theorem holds: sum(f²) = sum(F²). That makes the energy in a set of coefficients directly
comparable between the DCT and the Walsh–Hadamard transform.

All functions work on single-channel float arrays on the 0–255 scale (NOT 0–1), so a
threshold such as 500 means the same thing as in MATLAB's dct2 on a uint8 image.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


# ---------------------------------------------------------------------------- DCT
@lru_cache(maxsize=16)
def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II basis C (n x n, float64), one basis function per row.

    C[k, x] = α(k) · cos(π (2x + 1) k / (2n)),   α(0) = √(1/n),  α(k) = √(2/n) for k ≥ 1.
    The returned array is read-only (it is cached and shared).
    """
    k = np.arange(n, dtype=np.float64)[:, None]
    x = np.arange(n, dtype=np.float64)[None, :]
    c = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    c[0, :] *= math.sqrt(1.0 / n)
    c[1:, :] *= math.sqrt(2.0 / n)
    c.setflags(write=False)
    return c


def dct2(f: np.ndarray) -> np.ndarray:
    """2-D DCT: F = C_M · f · C_Nᵀ. Same as MATLAB dct2 and scipy.fft.dctn(f, norm="ortho")."""
    m, n = f.shape
    return dct_matrix(m) @ np.asarray(f, np.float64) @ dct_matrix(n).T


def idct2(F: np.ndarray) -> np.ndarray:
    """Inverse 2-D DCT: f = C_Mᵀ · F · C_N. Same as MATLAB idct2."""
    m, n = F.shape
    return dct_matrix(m).T @ np.asarray(F, np.float64) @ dct_matrix(n)


# ------------------------------------------------------------------ Walsh–Hadamard
def is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def next_pow2(n: int) -> int:
    """Smallest power of two >= n."""
    return 1 << max(0, int(n) - 1).bit_length()


def _bitreverse(k: int, bits: int) -> int:
    r = 0
    for _ in range(bits):
        r = (r << 1) | (k & 1)
        k >>= 1
    return r


@lru_cache(maxsize=16)
def sequency_order(n: int) -> np.ndarray:
    """order[k] = natural-order (Sylvester) row that becomes sequency-order row k.

    order[k] = bitreverse(gray(k)) with gray(k) = k XOR (k >> 1), using log2(n) bits.
    Row k of the sequency-ordered matrix has exactly k sign changes.
    """
    bits = n.bit_length() - 1
    order = np.array([_bitreverse(k ^ (k >> 1), bits) for k in range(n)], dtype=np.intp)
    order.setflags(write=False)
    return order


@lru_cache(maxsize=16)
def walsh_matrix(n: int) -> np.ndarray:
    """Orthonormal Walsh matrix in SEQUENCY order (n x n float64; n must be a power of 2).

    Sylvester construction (natural order): H_1 = [1],  H_2m = [[H_m, H_m], [H_m, -H_m]].
    The rows are then reordered into sequency order (see sequency_order) and scaled by 1/√n.
    MATLAB's fwht uses the same (sequency) ordering but scales the forward transform by 1/N:
    fwht(x) = (this matrix · x) / √N.
    """
    if not is_pow2(n):
        raise ValueError(f"Walsh–Hadamard needs a power-of-two size, got {n}")
    h = np.ones((1, 1), np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    w = h[sequency_order(n)] / math.sqrt(n)
    w.setflags(write=False)
    return w


@lru_cache(maxsize=32)
def _hadamard_natural(n: int, dtype_name: str) -> np.ndarray:
    """Unscaled Sylvester Hadamard matrix (natural order), ±1 entries, read-only."""
    h = np.ones((1, 1), np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    h = h.astype(dtype_name)
    h.setflags(write=False)
    return h


def _wht_natural_rows(y: np.ndarray) -> np.ndarray:
    """Unnormalised WHT (natural order) of every row of a 2-D array, via a Kronecker factorisation.

    Sylvester matrices factor as H_(a·b) = H_a ⊗ H_b. Reshaping a row of length n = a·b into an
    a x b block X (row-major), (H_a ⊗ H_b) · x = vec(H_a · X · H_b). With a ≈ b ≈ √n this is two
    small matrix products per row (≈ 2√n multiply-adds per value), which BLAS runs much faster
    than the log2(n) memory sweeps of the butterfly for big images (same result; tested).
    """
    rows, n = y.shape
    k = n.bit_length() - 1
    a = 1 << (k // 2)
    b = n // a
    ha, hb = _hadamard_natural(a, y.dtype.name), _hadamard_natural(b, y.dtype.name)
    return (ha @ y.reshape(rows, a, b) @ hb).reshape(rows, n)


def _butterfly_inplace(y: np.ndarray) -> np.ndarray:
    """In-place unnormalised fast WHT along the last axis of a C-contiguous array (natural order)."""
    n = y.shape[-1]
    lead = y.shape[:-1]
    h = 1
    while h < n:
        v = y.reshape(*lead, n // (2 * h), 2, h)
        a, b = v[..., 0, :], v[..., 1, :]      # views into y
        a += b                                 # a' = a + b
        b *= -2.0
        b += a                                 # b' = (a + b) - 2b = a - b
        h *= 2
    return y


def _fwht_natural_last_axis(x: np.ndarray) -> np.ndarray:
    """Unnormalised fast Walsh–Hadamard transform along the last axis, natural (Sylvester) order.

    Iterative butterflies: at each of the log2(n) stages, pairs (a, b) that are h apart become
    (a + b, a - b). Cost O(n log n) per row instead of O(n²) for the matrix product.
    """
    return _butterfly_inplace(np.array(x, dtype=np.float64))   # a fresh copy, transformed in place


def fwht(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Orthonormal sequency-ordered WHT along one axis via the fast butterfly (= walsh_matrix(n) · x)."""
    x = np.moveaxis(np.asarray(x, np.float64), axis, -1)
    n = x.shape[-1]
    if not is_pow2(n):
        raise ValueError(f"Walsh–Hadamard needs a power-of-two size, got {n}")
    y = _fwht_natural_last_axis(x)[..., sequency_order(n)] / math.sqrt(n)
    return np.moveaxis(y, -1, axis)


def ifwht(y: np.ndarray, axis: int = -1) -> np.ndarray:
    """Inverse of fwht (= walsh_matrix(n)ᵀ · y): undo the sequency reordering, then butterfly."""
    y = np.moveaxis(np.asarray(y, np.float64), axis, -1)
    n = y.shape[-1]
    if not is_pow2(n):
        raise ValueError(f"Walsh–Hadamard needs a power-of-two size, got {n}")
    natural = np.empty_like(y)
    natural[..., sequency_order(n)] = y
    x = _fwht_natural_last_axis(natural) / math.sqrt(n)
    return np.moveaxis(x, -1, axis)


FAST_WHT_FROM = 64   # use the fast factorised transform instead of the full matrix product from this size up


def fwht2(f: np.ndarray, dtype=np.float64) -> np.ndarray:
    """2-D orthonormal Walsh–Hadamard (sequency order): F = W_M · f · W_Nᵀ.

    Fast path (from 64 px): transform every row (_wht_natural_rows), transpose, put the rows into
    sequency order, transform the rows again (= the original columns), transpose back, reorder,
    and scale once by 1/√(M·N). Reordering whole rows is a contiguous copy, much faster than
    gathering columns. dtype=np.float32 halves memory and time for big images (≈ 1e-7 relative error).
    """
    m, n = f.shape
    if max(m, n) < FAST_WHT_FROM:
        return walsh_matrix(m) @ np.asarray(f, np.float64) @ walsh_matrix(n).T
    y = _wht_natural_rows(np.asarray(f, dtype=dtype))      # along n, natural order
    t = np.ascontiguousarray(y.T)[sequency_order(n)]       # (n, m), rows in sequency order
    del y
    t = _wht_natural_rows(t)                               # along m, natural order
    y = np.ascontiguousarray(t.T)[sequency_order(m)]       # (m, n), rows in sequency order
    y *= 1.0 / math.sqrt(m * n)
    return y


def ifwht2(F: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Inverse 2-D Walsh–Hadamard: f = W_Mᵀ · F · W_N (undo each reordering, then butterfly)."""
    m, n = F.shape
    if max(m, n) < FAST_WHT_FROM:
        return walsh_matrix(m).T @ np.asarray(F, np.float64) @ walsh_matrix(n)
    y = np.empty((m, n), dtype)
    y[sequency_order(m)] = F                               # rows back to natural order (axis m)
    t = _wht_natural_rows(np.ascontiguousarray(y.T))       # along m (H is symmetric)
    del y
    u = np.empty_like(t)
    u[sequency_order(n)] = t                               # natural order along n
    del t
    y = _wht_natural_rows(np.ascontiguousarray(u.T))       # along n
    y *= 1.0 / math.sqrt(m * n)
    return y


def pad_to_pow2(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad each dimension SEPARATELY (bottom/right) up to its next power of two. Never resizes.

    Uses mirror padding ("reflect"), which avoids an artificial edge at the border; "edge" is used
    if a dimension is too small to reflect. Returns (padded, (pad_rows, pad_cols)); crop back with
    padded[:h, :w]. A rectangular 2^a x 2^b result is fine.
    """
    h, w = img.shape[:2]
    ph, pw = next_pow2(h) - h, next_pow2(w) - w
    if ph == 0 and pw == 0:
        return img, (0, 0)
    pads = [(0, ph), (0, pw)] + [(0, 0)] * (img.ndim - 2)
    mode = "reflect" if (ph < h and pw < w) else "edge"
    return np.pad(img, pads, mode=mode), (ph, pw)


# ------------------------------------------------------------- coefficient selection
def keep_percent(F: np.ndarray, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep the ceil(p/100 · size) largest-magnitude coefficients; zero the rest.

    The DC term F[0, 0] is always kept (it is ranked first). Ties in |F| at the cut-off are
    broken by position (row-major order), so the result is deterministic.
    Instead of sorting all coefficients (O(n log n)), np.partition finds the count-th largest
    magnitude in O(n); everything above it is kept, plus the first tied values by position.
    Returns (masked F, kept mask).
    """
    n = F.size
    count = min(n, max(1, math.ceil(p / 100.0 * n)))
    key = np.abs(F).ravel().astype(np.float64)
    key[0] = np.inf                                       # DC first, always kept
    cut = np.partition(key, n - count)[n - count]         # the count-th largest magnitude
    mask = key > cut
    need = count - int(np.count_nonzero(mask))
    mask[np.flatnonzero(key == cut)[:need]] = True        # ties: lowest positions first
    mask = mask.reshape(F.shape)
    return np.where(mask, F, 0.0), mask


def threshold(F: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Threshold coding: zero every coefficient with |F| < t. The DC term is always kept."""
    mask = np.abs(F) >= t
    mask[0, 0] = True
    return np.where(mask, F, 0.0), mask


def zonal(F: np.ndarray, k_fraction: float, shape: str = "square") -> tuple[np.ndarray, np.ndarray]:
    """Zonal coding: keep low-frequency coefficients by position, k = round(k_fraction · min(M, N)).

    shape="square":  keep rows < k AND cols < k (the classic zonal mask, a k x k square).
    shape="l_shape": zero only the block where row >= k AND col >= k, so the kept region is
                     L-shaped (all low rows plus all low columns).
    """
    m, n = F.shape
    k = max(1, round(k_fraction * min(m, n)))
    rows = np.arange(m)[:, None] < k
    cols = np.arange(n)[None, :] < k
    if shape == "square":
        mask = rows & cols
    elif shape == "l_shape":
        mask = rows | cols                     # = NOT (row >= k AND col >= k)
    else:
        raise ValueError(f"zonal shape must be 'square' or 'l_shape', got {shape!r}")
    return np.where(mask, F, 0.0), mask


def energy_compaction(coefs, fractions) -> np.ndarray:
    """Share of the total energy captured by the largest coefficients, for each fraction kept.

    coefs: one 2-D orthonormal coefficient array, or a list of them (one per RGB channel).
    For each channel, F² is sorted in descending order; at fraction q the top ceil(q · size)
    coefficients of every channel are kept, and their energy is summed over the channels.
    Returns an array of shares in [0, 1] (1.0 at q = 1, by Parseval = the image's energy).
    """
    if isinstance(coefs, np.ndarray) and coefs.ndim == 2:
        coefs = [coefs]
    fractions = np.asarray(fractions, np.float64)
    captured = np.zeros(len(fractions))
    total = 0.0
    for F in coefs:
        e = np.sort((np.asarray(F, np.float64) ** 2).ravel())[::-1]
        cum = np.concatenate(([0.0], np.cumsum(e)))
        counts = np.clip(np.ceil(fractions * e.size).astype(int), 0, e.size)
        captured += cum[counts]
        total += cum[-1]
    return captured / total if total > 0 else np.ones(len(fractions))
