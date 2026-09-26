# Session Changes — Test 1

**Date:** 2026-09-27
**Scope:** Full pipeline audit + bug fixes

---

## 1. Layout Fix — Frontend stretched to fill screen

**File:** `frontend/src/App.jsx`

**Bug:** The two-column layout was not expanding to fill the full browser width. Large empty white space appeared on the right side of the screen.

**Root cause:** The `Layout` grid component and the outer content wrapper both lacked `w-full`.

**Fix:**
```diff
- <div className="grid gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
+ <div className="grid w-full gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">

- <div className="mx-auto max-w-[1500px] space-y-4 px-4 py-4">
+ <div className="mx-auto w-full max-w-[1500px] space-y-4 px-4 py-4">
```

---

## 2. Auto Fix — Salt-and-pepper spikes not detected (two-layer smoothing)

### Bug 2a — Browser downscales with bilinear smoothing before upload
**File:** `frontend/src/lib/image.js`

**Root cause:** `prepareImage()` drew the image onto a canvas at 1024px max before uploading. `canvas.drawImage` uses bilinear interpolation which blends each isolated spike with neighbours — spikes gone before the server even receives the file.

**Fix:** Remove the client-side resize entirely. Send the original file at full resolution. Only re-encode unsupported formats without resizing.

```diff
- const needsReencode = scale < 1 || !LOSSLESS_KEEP.includes(blob.type);
- if (needsReencode) {
-   width = Math.round(w0 * scale); height = Math.round(h0 * scale);
-   ctx.drawImage(bmp, 0, 0, width, height);  // bilinear smoothing kills spikes
- }
+ const needsReencode = !LOSSLESS_KEEP.includes(blob.type);
+ if (needsReencode) {
+   canvas.width = w0; canvas.height = h0;   // full resolution, no resize
+   ctx.drawImage(bmp, 0, 0, w0, h0);
+ }
+ // display dimensions computed separately — thumbnail only, not what gets uploaded
```

### Bug 2b — `impulse_density` measured after `INTER_AREA` downscale
**File:** `pipeline/ivp_pipeline/__init__.py`

**Root cause:** Even if the browser sent the original file, the backend `prepare()` uses `cv2.INTER_AREA` which averages pixels. The `impulse_density` check ran after `prepare()`, so spikes were already gone.

**Fix:** Measure `impulse_density` BEFORE calling `prepare()`:

```diff
 def auto_enhance(img, engine="opencv"):
     check_engine(engine)
-    work, scale = prepare(img)
-    pre_id = metrics.impulse_density(to_gray(img)) if scale < 1.0 else None
+    pre_id = metrics.impulse_density(to_gray(img))   # measure BEFORE downscaling
+    work, scale = prepare(img)
     res = auto_enhance_prepared(work, engine, pre_scale_impulse_density=pre_id)
```

### Bug 2c — `auto_enhance_prepared` had no parameter for pre-scale measurement
**File:** `pipeline/ivp_pipeline/auto.py`

**Fix:** Added `pre_scale_impulse_density` parameter; use `max()` of pre- and post-scale values:

```diff
-def auto_enhance_prepared(img, engine):
-    before = diagnose_prepared(img)
-    d = before["metrics"]
+def auto_enhance_prepared(img, engine, pre_scale_impulse_density=None):
+    before = diagnose_prepared(img)
+    d = dict(before["metrics"])
+    if pre_scale_impulse_density is not None and pre_scale_impulse_density > d["impulse_density"]:
+        d["impulse_density"] = pre_scale_impulse_density
```

---

## 3. Auto Fix — Moderate noise (σ 5–8) produced zero denoising steps

**File:** `pipeline/ivp_pipeline/auto.py`

**Bug:** The `elif d["noise_sigma"] > C.NOISE_MODERATE` branch only set `skip_sharpen = True` with no denoising. An image with σ = 7.7 (visible grain, just below NOISE_STRONG = 8.0) would show "Nothing needed fixing."

**Fix:** Apply a light Gaussian (σ = 0.6) for moderate noise:

```diff
 elif d["noise_sigma"] > C.NOISE_MODERATE:
-    skip_sharpen = True  # no denoising applied at all
+    apply("denoise", {"filter_type": "gaussian", "sigma": 0.6},
+          f"moderate sensor noise σ ≈ {d['noise_sigma']:.1f} (light smoothing)")
+    skip_sharpen = True
```

---

## 4. Guided Mode — Equalize noise gate too restrictive

**File:** `pipeline/ivp_pipeline/guided.py`

**Bug:** Gaussian pre-smooth before HE was only applied when the image was both noisy AND dark (`noisy and d["mean"] < C.DARK_MEAN`). HE amplifies noise regardless of brightness, so bright noisy images skipped pre-smoothing.

**Fix:** Remove the brightness condition entirely:

```diff
-if intent == "equalize":
-    noisy = noisy and d["mean"] < C.DARK_MEAN
+# HE amplifies noise wherever the CDF is steep, regardless of brightness.
+# Removed the dark-image-only gate — let `noisy` stand unmodified.
```

---

## 5. Sharpen/Edges — A=0 on colour input silently changes output to grayscale

**File:** `pipeline/ivp_pipeline/modules/sharpen_edges.py`

**Bug:** `laplacian_boost` with `A=0` returns 2-D grayscale even for colour inputs, with no warning. Chaining further operations would treat the result as a gray image.

**Fix:** Added a user-facing warning; also imported `is_color`:

```diff
+from ..utils import apply_on_luma, is_color, to_gray

 if A == 0.0:
     out = _normalise(np.abs(_corr(gray_in, k, engine)))
+    if is_color(img):
+        warnings.append(
+            "A = 0 produces a grayscale edge image. "
+            "If you keep editing this result it will be treated as a gray image."
+        )
```

---

## 6. Sharpen/Edges — Zero-response on constant image produces silent black output

**File:** `pipeline/ivp_pipeline/modules/sharpen_edges.py`

**Bug:** Constant-value images produce zero Laplacian/gradient response. `_normalise()` returns all-black with no user warning.

**Fix:** Added warnings after each `_normalise()` call when result is all-zero:

```diff
 out = _normalise(np.abs(_corr(gray_in, k, engine)))
+if out.max() == 0:
+    warnings.append("Laplacian response is zero (constant-value image); output is all-black.")

 out = _normalise(view)  # gradient path
+if out.max() == 0:
+    warnings.append("Gradient magnitude is zero (constant-value image); output is all-black.")
```

---

## 7. Segment — fill_holes ignored the user's connectivity setting

**Files:** `pipeline/ivp_pipeline/modules/segment.py`, `pipeline/ivp_pipeline/core/morphology.py`

**Bug:** `_fill_holes()` always used 4-connectivity flood-fill, even when the user chose 8-connectivity for region growing, leaving thin diagonal holes unfilled.

**Fix:** Thread `connectivity` from validated params all the way through:

```diff
# segment.py
-def _fill_holes(mask, engine):
-    cv2.floodFill(..., flags=4)
-    M.fill_holes(mask)
+def _fill_holes(mask, engine, connectivity=4):
+    cv2.floodFill(..., flags=connectivity)
+    M.fill_holes(mask, connectivity)

-if p["fill_holes"]: out = _fill_holes(out, engine)
+if p["fill_holes"]: out = _fill_holes(out, engine, int(p["connectivity"]))

# morphology.py
-def fill_holes(mask):
-    reachable = bfs_region(background, seeds, connectivity=4)
+def fill_holes(mask, connectivity=4):
+    reachable = bfs_region(background, seeds, connectivity=connectivity)
```

---

## 8. Adaptive Region Growing — O(n²) re-checking of rejected neighbours

**File:** `pipeline/ivp_pipeline/core/region.py`

**Bug:** In adaptive BFS, rejected neighbours were never marked `visited`, so every frontier pixel that shared a border with them would re-test them. O(perimeter²) redundant work on large noisy images.

**Fix:** Mark every neighbour visited on first encounter regardless of acceptance:

```diff
 if not visited[n]:
-    if abs(g[n] - total / count) <= delta:
-        visited[n] = in_region[n] = True
+    visited[n] = True                  # always mark on first encounter
+    if abs(g[n] - total / count) <= delta:
+        in_region[n] = True
         total += g[n]; count += 1; q.append(n)
```

---

## 9. Codec — 16-bit images normalised to full range, hiding exposure info

**File:** `backend/app/codec.py`

**Bug:** 16-bit PNG/TIFF files were converted with `cv2.NORM_MINMAX`, always mapping min→0 and max→255. An over/underexposed 16-bit image always reported ~50% brightness to diagnostics.

**Fix:** Use a linear bit-shift for uint16 to preserve absolute exposure:

```diff
 if img.dtype != np.uint8:
-    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
+    if img.dtype == np.uint16:
+        img = (img >> 8).astype(np.uint8)   # 0-65535 → 0-255, preserves exposure
+    else:
+        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
```

---

## 10. Pipeline Package — Switched to editable install

**Command:**
```bash
pip install -e ./pipeline
```

**Reason:** The original `pip install ./pipeline` copied files to `site-packages`. Source edits had no effect on the running server until reinstall. Editable install makes `uvicorn --reload` pick up all `ivp_pipeline/` changes immediately.

---

## Files Changed Summary

| File | Change |
|---|---|
| `frontend/src/App.jsx` | Added `w-full` to Layout grid and content wrapper |
| `frontend/src/lib/image.js` | Removed client-side downscale; send original file to server |
| `pipeline/ivp_pipeline/__init__.py` | Added `metrics`, `to_gray` imports; measure `impulse_density` before `prepare()` |
| `pipeline/ivp_pipeline/auto.py` | Added `pre_scale_impulse_density` param; light Gaussian for moderate noise |
| `pipeline/ivp_pipeline/guided.py` | Removed dark-image-only equalize noise gate |
| `pipeline/ivp_pipeline/modules/sharpen_edges.py` | Added `is_color` import; warnings for grayscale output and zero response |
| `pipeline/ivp_pipeline/modules/segment.py` | Thread `connectivity` through `_fill_holes` |
| `pipeline/ivp_pipeline/core/morphology.py` | Added `connectivity` param to `fill_holes` |
| `pipeline/ivp_pipeline/core/region.py` | Fixed adaptive BFS to mark rejected neighbours visited immediately |
| `backend/app/codec.py` | Preserve 16-bit exposure with bit-shift instead of NORM_MINMAX |
