"""Quality gate & dual-route comparison (spec §10).

Branch A runs the target operation directly. Branch B first applies the prerequisite
fixes that the diagnostics call for (median -> Gaussian -> flatten, re-diagnosing after
each), then runs the SAME target with the SAME parameters. Verify-after-fix then checks
that Branch B measurably beats Branch A.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from . import metrics as Q
from . import registry
from .diagnostics import diagnose_prepared
from .errors import PipelineError
from .modules.sharpen_edges import edge_map_from
from .pipeline import execute, step
from .utils import to_gray

INTENTS = ("segment", "edges", "sharpen", "equalize")


def resolve_target(intent: str, params: dict | None) -> tuple[str, dict]:
    params = dict(params or {})
    if intent == "segment":
        return "segment", registry.validate("segment", params)
    if intent == "edges":
        params.setdefault("operation", "gradient")
        p = registry.validate("sharpen_edges", params)
        if p["operation"] != "gradient":
            raise PipelineError("BAD_PARAM", "intent 'edges' requires operation='gradient'")
        return "sharpen_edges", p
    if intent == "sharpen":
        p = registry.validate("sharpen_edges", params)
        if p["operation"] == "gradient":
            raise PipelineError("BAD_PARAM", "intent 'sharpen' needs operation 'unsharp' or 'laplacian_boost'")
        if p["operation"] == "laplacian_boost" and p["A"] == 0:
            raise PipelineError("BAD_PARAM", "A = 0 is edge extraction; use intent 'edges' instead")
        return "sharpen_edges", p
    if intent == "equalize":
        params.setdefault("mode", "hist_eq")
        p = registry.validate("contrast", params)
        if p["mode"] != "hist_eq":
            raise PipelineError("BAD_PARAM", "intent 'equalize' requires mode='hist_eq'")
        return "contrast", p
    raise PipelineError("UNKNOWN_INTENT", f"intent must be one of {list(INTENTS)}, got {intent!r}")


def plan_and_condition(img: np.ndarray, intent: str, engine: str, target: dict | None = None):
    """Apply the §10 prerequisite chain. Returns (conditioned image, stages, initial diagnosis)."""
    diag0 = diagnose_prepared(img)
    d = diag0["metrics"]
    cur, stages = img, []

    def add(module, params, rule, reason):
        nonlocal cur, d
        res = step(cur, module, params, engine)
        cur = res["image"]
        stages.append({"name": f"{module}:{params.get('filter_type', params.get('mode'))}",
                       "module": module, "params": res["meta"]["params"], "rule": rule,
                       "reason": reason, "image": cur})
        d = diagnose_prepared(cur)["metrics"]  # re-diagnose: an earlier fix may solve later problems

    # 1) impulse noise -> median
    if intent in ("segment", "edges", "sharpen") and d["impulse_density"] > C.IMPULSE_DENSITY:
        k = 5 if d["impulse_density"] > C.IMPULSE_HEAVY else 3
        add("denoise", {"filter_type": "median", "kernel_size": k}, "IMPULSE_NOISE",
            f"impulse density {d['impulse_density']:.2%} > {C.IMPULSE_DENSITY:.1%}")

    # 2) Gaussian noise -> Gaussian smoothing
    if intent == "edges":
        limit = C.NOISE_GATE_EDGES[(target or {}).get("operator", "sobel")]
    else:
        limit = C.NOISE_GATE[intent]
    noisy = d["noise_sigma"] > limit
    # HE amplifies noise wherever the CDF is steep, regardless of overall
    # brightness.  The previous guard (only pre-smooth dark images) was too
    # restrictive — a bright but noisy image would skip the Gaussian step and
    # come out with amplified grain.  Let `noisy` stand unmodified for all intents.
    if noisy:
        sigma = C.GUIDED_SHARPEN_GAUSS_SIGMA if intent == "sharpen" else C.GUIDED_GAUSS_SIGMA
        add("denoise", {"filter_type": "gaussian", "sigma": sigma}, "GAUSSIAN_NOISE",
            f"noise σ ≈ {d['noise_sigma']:.1f} > {limit:g}"
            + (" in a dark image" if intent == "equalize" else ""))

    # 3) uneven illumination -> morphological flattening (segmentation only)
    if intent == "segment" and d["illumination_u"] > C.ILLUMINATION_U:
        radius = int(min(75, max(5, max(img.shape[:2]) // C.BG_RADIUS_FRACTION)))
        add("contrast", {"mode": "flatten", "se_radius": radius}, "UNEVEN_ILLUMINATION",
            f"background unevenness U = {d['illumination_u']:.2f} > {C.ILLUMINATION_U}")
    return cur, stages, diag0


def _checks(intent, stages, a, b, original, target_params):
    """Verify-after-fix: one check per distinct criterion implied by the stages that ran."""
    rules = {s["rule"] for s in stages}
    checks = []

    def add(name, va, vb, passed, better):
        checks.append({"criterion": name, "a": round(float(va), 5), "b": round(float(vb), 5),
                       "better": better, "passed": bool(passed)})

    if intent == "segment":
        if rules & {"IMPULSE_NOISE", "GAUSSIAN_NOISE"}:
            fa = Q.fragmentation(a["extras"]["raw_mask"] > 0)
            fb = Q.fragmentation(b["extras"]["raw_mask"] > 0)
            add("fragmentation (islands + holes, before cleanup)", fa, fb, fb < fa, "lower")
        if "UNEVEN_ILLUMINATION" in rules:
            ea, eb = a["metrics"]["eta"], b["metrics"]["eta"]
            add("otsu separability η", ea, eb, eb >= ea + C.ETA_GAIN_MIN, "higher")
    elif intent == "edges":
        flat = Q.flat_region_mask(to_gray(original))
        t = target_params["edge_threshold"] or C.DEFAULT_EDGE_T
        def density(r):
            mag = np.hypot(r["extras"]["gx"], r["extras"]["gy"])
            return Q.edge_density(edge_map_from(mag, t), flat)
        da, db = density(a), density(b)
        add(f"false-edge density in flat regions (T={t:g})", da, db, db < da, "lower")
    else:  # sharpen / equalize: how much noise ends up in the output
        na, nb = Q.noise_sigma(to_gray(a["image"])), Q.noise_sigma(to_gray(b["image"]))
        add("noise σ in output", na, nb, nb < na, "lower")
    return checks


def run_guided_prepared(img: np.ndarray, intent: str, params: dict | None, engine: str) -> dict:
    module, target = resolve_target(intent, params)
    conditioned, stages, diag0 = plan_and_condition(img, intent, engine, target)
    branch_a = execute(img, module, target, engine)
    branch_a["path"] = [f"{module}:target"]

    if not stages:
        return {"gate_passed": True, "intent": intent, "diagnosis": diag0,
                "branch_a": branch_a, "branch_b": None,
                "verdict": {"result": "not_needed", "checks": []}}

    branch_b = execute(conditioned, module, target, engine)
    branch_b["path"] = [s["name"] for s in stages] + [f"{module}:target"]
    branch_b["stages"] = stages
    checks = _checks(intent, stages, branch_a, branch_b, img, target)
    n_pass = sum(c["passed"] for c in checks)
    result = "recommended" if n_pass == len(checks) else ("mixed" if n_pass else "no_benefit")
    return {"gate_passed": False, "intent": intent, "diagnosis": diag0,
            "branch_a": branch_a, "branch_b": branch_b,
            "verdict": {"result": result, "checks": checks}}
