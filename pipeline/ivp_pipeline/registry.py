"""Parameter schemas for every module (spec §5–§8) and validation.

list_modules() exposes these schemas so the dashboard can build its controls from them.
"""
from __future__ import annotations

import copy
import math

from .errors import PipelineError


def _enum(name, options, default, desc, when=None):
    return {"name": name, "type": "enum", "options": options, "default": default,
            "description": desc, "when": when}


def _num(name, typ, lo, hi, default, desc, step=None, when=None, allow=None):
    d = {"name": name, "type": typ, "min": lo, "max": hi, "default": default,
         "description": desc, "when": when}
    if step is not None:
        d["step"] = step
    if allow:
        d["also_allowed"] = allow
    return d


def _bool(name, default, desc, when=None):
    return {"name": name, "type": "bool", "default": default, "description": desc, "when": when}


# "when" = {param: [values]} -> the control is only relevant in those modes (UI hint).
SCHEMAS: dict[str, dict] = {
    "contrast": {
        "title": "Contrast, exposure & illumination",
        "syllabus": "Unit 2 (point & histogram processing), Unit 4, Unit 7; Exp 04, 07",
        "params": [
            _enum("mode", ["hist_eq", "gamma", "stretch", "flatten"], "hist_eq",
                  "Intensity transform to apply (on luminance Y only)"),
            _num("gamma", "float", 0.10, 3.00, 1.00,
                 "Power-law exponent: <1 brightens, >1 darkens. 'auto' maps the mean to mid-gray",
                 step=0.05, when={"mode": ["gamma"]}, allow=["auto"]),
            _num("clip_percentile", "float", 0.0, 5.0, 1.0,
                 "Percent of pixels saturated at each end before stretching",
                 step=0.5, when={"mode": ["stretch"]}),
            _num("se_radius", "int", 5, 75, 35,
                 "Disk radius for background estimation; must exceed object radius",
                 step=1, when={"mode": ["flatten"]}),
            _enum("object_polarity", ["auto", "bright", "dark"], "auto",
                  "Bright objects -> opening; dark objects -> closing", when={"mode": ["flatten"]}),
        ],
    },
    "denoise": {
        "title": "Noise removal",
        "syllabus": "Unit 2 (smoothing & order-statistic filters); Exp 03",
        "params": [
            _enum("filter_type", ["median", "box", "gaussian"], "median",
                  "Median for salt & pepper; box/Gaussian for sensor noise"),
            _enum("kernel_size", [3, 5, 7], 3, "Window size K x K",
                  when={"filter_type": ["median", "box"]}),
            _num("sigma", "float", 0.5, 3.0, 1.0,
                 "Gaussian standard deviation (kernel size = 2*ceil(3*sigma)+1)",
                 step=0.1, when={"filter_type": ["gaussian"]}),
        ],
    },
    "sharpen_edges": {
        "title": "Sharpening & edges",
        "syllabus": "Unit 2 (sharpening), Unit 5 (gradients); Exp 03, 08",
        "params": [
            _enum("operation", ["laplacian_boost", "unsharp", "gradient"], "unsharp",
                  "Sharpen (Laplacian high-boost / unsharp masking) or extract edges"),
            _num("A", "float", 0.0, 3.0, 1.0,
                 "High-boost factor: 0 = edges only, 1 = sharpen, >1 = boost (also brightens)",
                 step=0.1, when={"operation": ["laplacian_boost"]}),
            _num("k", "float", 0.2, 3.0, 1.0, "Unsharp gain: 1 = unsharp, >1 = high-boost",
                 step=0.1, when={"operation": ["unsharp"]}),
            _num("blur_sigma", "float", 0.5, 3.0, 1.0, "Blur used to build the unsharp mask",
                 step=0.1, when={"operation": ["unsharp"]}),
            _enum("operator", ["sobel", "prewitt", "roberts"], "sobel", "Gradient mask pair",
                  when={"operation": ["gradient"]}),
            _enum("magnitude", ["l2", "l1"], "l2", "sqrt(Gx^2+Gy^2) or |Gx|+|Gy|",
                  when={"operation": ["gradient"]}),
            _enum("component", ["magnitude", "gx", "gy"], "magnitude",
                  "What to display (Roberts: gx/gy are the two diagonals)",
                  when={"operation": ["gradient"]}),
            _num("edge_threshold", "float", 0.0, 1.0, 0.0,
                 "Binary edge map threshold as a fraction of max magnitude (0 = off)",
                 step=0.05, when={"operation": ["gradient"]}),
        ],
    },
    "segment": {
        "title": "Segmentation & morphological cleanup",
        "syllabus": "Unit 6 (thresholding, region growing), Unit 4 (morphology); Exp 07, 09",
        "params": [
            _enum("method", ["otsu", "region_growing"], "otsu", "Global Otsu or seeded region growing"),
            _enum("foreground", ["auto", "bright", "dark"], "auto",
                  "Which Otsu class is the object (auto = the smaller class)",
                  when={"method": ["otsu"]}),
            {"name": "seed", "type": "point", "default": None,
             "description": "(x, y) in ORIGINAL image pixels; default = image centre",
             "when": {"method": ["region_growing"]}},
            _num("delta_t", "int", 1, 50, 15, "Similarity tolerance ΔT", step=1,
                 when={"method": ["region_growing"]}),
            _enum("connectivity", [4, 8], 4, "Neighbourhood used for growing",
                  when={"method": ["region_growing"]}),
            _bool("adaptive", False, "Compare with the running region mean instead of the seed",
                  when={"method": ["region_growing"]}),
            _bool("morph_open", True, "Opening: remove specks smaller than the disk"),
            _bool("morph_close", True, "Closing: fill holes/gaps smaller than the disk"),
            _num("disk_radius", "int", 1, 15, 3, "Radius of the disk structuring element", step=1),
            _bool("fill_holes", False, "Fill enclosed background regions"),
            _bool("keep_largest", False, "Keep only the largest connected component"),
        ],
    },
    "compress": {
        "title": "Image compression",
        "syllabus": "Transforms: DCT, JPEG-style block coding, Walsh–Hadamard",
        "params": [
            _enum("method", ["jpeg", "dct", "hadamard"], "dct",
                  "jpeg: 8x8 block DCT + quantisation (how real image files get small). "
                  "dct / hadamard: whole-image transform, keep only some coefficients"),
            _num("quality", "int", 1, 100, 75,
                 "JPEG quality: scales the quantisation tables. Lower = coarser steps = smaller file",
                 step=1, when={"method": ["jpeg"]}),
            _enum("chroma_subsampling", ["4:2:0", "4:2:2", "4:4:4"], "4:2:0",
                  "How much colour detail is kept: 4:2:0 stores colour at half width and half height",
                  when={"method": ["jpeg"]}),
            _enum("mode", ["keep_percent", "threshold", "zonal"], "keep_percent",
                  "How coefficients are chosen: largest N %, a magnitude threshold, or low frequencies by position",
                  when={"method": ["dct", "hadamard"]}),
            _num("keep_percent", "float", 0.1, 100.0, 10.0,
                 "Percent of coefficients kept (the largest in magnitude; DC is always kept)",
                 step=0.1, when={"method": ["dct", "hadamard"], "mode": ["keep_percent"]}),
            _num("threshold", "float", 0.0, 5000.0, 500.0,
                 "Zero every coefficient with |F| below this (on the 0–255 pixel scale)",
                 step=10, when={"method": ["dct", "hadamard"], "mode": ["threshold"]}),
            _num("zonal_fraction", "float", 0.01, 1.0, 0.125,
                 "Zone size k as a fraction of the shorter side",
                 step=0.01, when={"method": ["dct", "hadamard"], "mode": ["zonal"]}),
            _enum("zonal_shape", ["square", "l_shape"], "square",
                  "square: keep rows < k and cols < k. l_shape: zero only the block where rows ≥ k and cols ≥ k",
                  when={"method": ["dct", "hadamard"], "mode": ["zonal"]}),
        ],
    },
}

MODULES = tuple(SCHEMAS)


def list_modules() -> dict:
    return copy.deepcopy(SCHEMAS)


def defaults(module: str) -> dict:
    return {p["name"]: p["default"] for p in SCHEMAS[module]["params"]}


def validate(module: str, params: dict | None) -> dict:
    """Fill defaults, coerce types and range-check. Raises PipelineError on any problem."""
    if module not in SCHEMAS:
        raise PipelineError("UNKNOWN_MODULE", f"module must be one of {list(MODULES)}, got {module!r}")
    params = dict(params or {})
    spec = {p["name"]: p for p in SCHEMAS[module]["params"]}
    unknown = set(params) - set(spec)
    if unknown:
        raise PipelineError("BAD_PARAM", f"unknown parameter(s) for {module}: {sorted(unknown)}")

    out = {}
    for name, p in spec.items():
        v = params.get(name, p["default"])
        try:
            out[name] = _coerce(p, v)
        except (TypeError, ValueError) as e:
            raise PipelineError("BAD_PARAM", f"{module}.{name}: {e}") from None
    return out


def _coerce(p: dict, v):
    t = p["type"]
    if t == "enum":
        # accept "5" for 5 when options are ints (form data arrives as strings)
        if isinstance(p["options"][0], int) and isinstance(v, str) and v.lstrip("-").isdigit():
            v = int(v)
        if v not in p["options"]:
            raise ValueError(f"must be one of {p['options']}, got {v!r}")
        return v
    if t == "bool":
        if isinstance(v, str):
            if v.lower() in ("true", "1", "yes"):
                return True
            if v.lower() in ("false", "0", "no"):
                return False
            raise ValueError(f"expected a boolean, got {v!r}")
        return bool(v)
    if t in ("int", "float"):
        if v in p.get("also_allowed", []):
            return v
        num = float(v)
        if not math.isfinite(num):
            raise ValueError("must be finite")
        if t == "int":
            if num != int(num):
                raise ValueError(f"must be an integer, got {v!r}")
            num = int(num)
        if not (p["min"] <= num <= p["max"]):
            raise ValueError(f"must be in [{p['min']}, {p['max']}], got {v!r}")
        return num
    if t == "point":
        if v is None:
            return None
        if len(v) != 2:
            raise ValueError("seed must be [x, y]")
        return (float(v[0]), float(v[1]))
    raise ValueError(f"unsupported type {t}")
