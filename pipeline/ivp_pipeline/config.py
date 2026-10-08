"""All tunable constants in one place (spec §9, §10, §11).

Diagnostic thresholds start at the spec's initial values; experiments/calibrate_gate.py
reports calibrated values — copy them here once you are happy with them.
"""

import os

# Phase 1
# Images are downscaled so max(H, W) <= MAX_SIDE. 0 = process the full resolution (default, for running
# locally). Set the environment variable IVP_MAX_SIDE=1024 on a small server such as Render's free tier
# (512 MB RAM): full-resolution photos can exceed its memory and take much longer.
MAX_SIDE = int(os.getenv("IVP_MAX_SIDE", "0"))

# §9 diagnostics
EXTREME_LOW, EXTREME_HIGH = 5, 250
IMPULSE_MEDIAN_DIFF = 50        # |Y - median3(Y)| must exceed this to count as an impulse
NOISE_EDGE_PERCENTILE = 80      # Immerkær: ignore pixels above this Sobel-magnitude percentile
BG_RADIUS_FRACTION = 10         # U_bg background SE radius = max(H, W) / this

DARK_MEAN, BRIGHT_MEAN = 0.30, 0.70
CLIPPED_FRACTION = 0.05
LOW_RANGE = 100                 # P99 - P1 below this = low contrast
NOISE_MODERATE = 5.0            # sigma_n (gray levels)
NOISE_STRONG = 8.0
IMPULSE_DENSITY = 0.0075        # 0.75 % — calibrated (was 0.5 %; best F1 0.993 vs 0.971)
# ILLUMINATION_U is deliberately kept below its best-F1 value (~0.25): a false alarm only costs
# an extra Branch B that verify-after-fix labels "no_benefit", while a miss hides the biggest win.
IMPULSE_HEAVY = 0.10            # use a 5x5 median above this

# Enlarged noise (typical of SCREENSHOTS: the browser or display scaling enlarged a noisy image, smearing each
# grain / speck over 2–3 pixels, so the pixel-scale measures above miss it). The same measures are taken on a
# half-scale copy (every 2nd pixel), where the grains are pixel-sized again. Calibrated on 1153 images
# (experiments/noise_scales_study.py; calibrate_gate.py has the confusion matrices):
#   specks: a pixel differing from its 3x3 median by > IMPULSE_MEDIAN_DIFF that forms a tiny isolated group
#   (≤ SPECK_MAX_AREA px) — no pure black/white needed, since enlarging softens the specks to grey.
SPECK_MAX_AREA = 2
SPECK_COARSE_DENSITY = 0.023    # half-scale speck density above this = enlarged salt & pepper
SPECK_COARSE_MAX_SIGMA = 10.0   # ... unless the image has heavy pixel-scale grain (its tails look like specks)
SPECK_COARSE_HEAVY = 0.05       # 7x7 median instead of 5x5 above this
#   grain: noise σ of the half-scale copy above NOISE_COARSE_MIN AND above NOISE_COARSE_RATIO x the pixel-scale σ
#   (enlarged noise is smooth between neighbouring pixels but random again at half scale; fine detail and
#   textures raise both values together, so the ratio keeps them out).
NOISE_COARSE_MIN = 5.0
NOISE_COARSE_RATIO = 3.5
COARSE_MEDIAN_KSIZE, COARSE_MEDIAN_KSIZE_HEAVY = 5, 7
COARSE_GAUSS_SIGMA = 1.5        # enlarged grain is 2–3 px wide: sigma 1.0 leaves part of it
ILLUMINATION_U = 0.15
LOW_ETA = 0.70

# §10 guided mode
# Noise level (estimated σ_n) above which Gaussian pre-smoothing is prescribed, per intent.
# Set from the benchmark: below these, smoothing destroys more detail than the noise does.
# Edge break-even points (tolerant edge F1 vs clean-image edges, 5 natural images):
#   Sobel/Prewitt ≈ σ 15 (their 3x3 masks already smooth), Roberts ≈ σ 12 (2x2, no smoothing).
NOISE_GATE = {"segment": 8.0, "sharpen": 8.0, "equalize": 8.0}
NOISE_GATE_EDGES = {"sobel": 15.0, "prewitt": 15.0, "roberts": 12.0}
ETA_GAIN_MIN = 0.02             # flatten must raise eta by at least this much
GUIDED_GAUSS_SIGMA = 1.0
GUIDED_SHARPEN_GAUSS_SIGMA = 0.8
FLAT_REGION_SIGMA = 2.0         # smoothing used to define "flat regions" for the false-edge proxy
FLAT_REGION_PERCENTILE = 50
DEFAULT_EDGE_T = 0.2            # edge threshold used by the proxy when the user left it at 0

# §11 auto enhance
AUTO_UNSHARP_K = 0.6
AUTO_UNSHARP_SIGMA = 1.0
AUTO_STRETCH_CLIP = 1.0
GAMMA_MIN, GAMMA_MAX = 0.3, 3.0

# Flattening
FLATTEN_DOWNSCALE_ABOVE_R = 25  # estimate the background at 1/4 scale when R exceeds this

# Compression (JPEG-style block coding, whole-image DCT / Walsh–Hadamard)
ERROR_MAP_MAX_AMPLIFICATION = 16   # error maps are scaled so their 99th percentile is bright, up to x16
RD_KEEP_PERCENTS = (0.5, 1, 2, 5, 10, 25, 50)   # rate–distortion curve points (% coefficients kept)
CURVE_MAX_SIDE = 512               # RD / energy curves are computed on a copy at most this big
COEF_MAP_MAX_SIDE = 1024           # coefficient maps / kept mask are display images: shrunk to at most this
ENERGY_CURVE_POINTS = 40           # log-spaced fractions from 0.1 % to 100 %
