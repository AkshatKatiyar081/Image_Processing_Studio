"""All tunable constants in one place (spec §9, §10, §11).

Diagnostic thresholds start at the spec's initial values; experiments/calibrate_gate.py
reports calibrated values — copy them here once you are happy with them.
"""

# Phase 1
MAX_SIDE = 1024                 # images are downscaled so max(H, W) <= this

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
