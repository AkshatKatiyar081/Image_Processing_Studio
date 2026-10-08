# Quality-gate calibration

187 labelled images: 8 natural + 3 texture + 6 synthetic sources, each clean and degraded (Gaussian σ 8/15/25, salt & pepper 1/5/10/20 %, ramp ±40/±70, low contrast).

## `noise_sigma`

Positive range: 7.68 – 27.64; negative range: 0.04915 – 10.53

| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| current = 5 | 51 | 3 | 0 | 65 | 0.944 | 1.000 | 0.971 |
| best-F1 ≈ 6.34 | 51 | 3 | 0 | 65 | 0.944 | 1.000 | 0.971 |

False alarms at current threshold: grass/clean, grass/ramp40, grass/ramp70

## `impulse_density`

Positive range: 0.007904 – 0.199; negative range: 0 – 0.01039

| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| current = 0.0075 | 68 | 1 | 0 | 118 | 0.986 | 1.000 | 0.993 |
| best-F1 ≈ 0.007702 | 68 | 1 | 0 | 118 | 0.986 | 1.000 | 0.993 |

False alarms at current threshold: page/gauss25

## `illumination_u`

Positive range: 0.2588 – 1; negative range: 0.003922 – 0.7569

| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| current = 0.15 | 34 | 58 | 0 | 95 | 0.370 | 1.000 | 0.540 |
| best-F1 ≈ 0.251 | 34 | 38 | 0 | 115 | 0.472 | 1.000 | 0.642 |

False alarms at current threshold: camera/clean, camera/gauss8, camera/gauss15, camera/gauss25, camera/sp1, camera/sp5, camera/sp10, camera/sp20, camera/lowcontrast, coins/clean, coins/gauss8, coins/gauss15, coins/gauss25, coins/sp1, coins/sp5, coins/sp10, coins/sp20, page/clean, page/gauss8, page/gauss15 …

## Full noise rules, including screenshots

`grain_any_scale` and `specks_any_scale` are the rules the Health card uses (diagnostics.gaussian_noise / impulse_noise): the pixel-scale tests above OR the half-scale tests for ENLARGED noise. Evaluated on the 187 images above plus 204 screenshot-like images (saved as JPEG 75, then shown 1.5x or 2x larger; clean, Gaussian σ 15/25, salt & pepper 5/10/20 %).

| Rule | Images | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| `grain_any_scale` | normal (119) | 51 | 4 | 0 | 64 | 0.927 | 1.000 | 0.962 |
|  | false alarms: page/ramp40, grass/clean, grass/ramp40, grass/ramp70 | | | | | | | |
| `grain_any_scale` | screenshots (102) | 68 | 1 | 0 | 33 | 0.986 | 1.000 | 0.993 |
|  | false alarms: grass/clean_shot1.5x | | | | | | | |
| `specks_any_scale` | normal (187) | 68 | 2 | 0 | 117 | 0.971 | 1.000 | 0.986 |
|  | false alarms: page/gauss25, gravel/gauss8 | | | | | | | |
| `specks_any_scale` | screenshots (204) | 102 | 1 | 0 | 101 | 0.990 | 1.000 | 0.995 |
|  | false alarms: grass/gauss25_shot1.5x | | | | | | | |

With the pixel-scale tests alone, the same noisy screenshots are caught only 0/68 times (grain) and 0/102 times (specks).

## Interpretation

* **Noise σ** separates cleanly; the only false alarms come from the *grass* texture, whose fine random detail is statistically similar to noise (a known limit of any single-image estimator).
* **Impulse density**: the calibrated threshold (0.75 %) is now in `config.py`. Heavy Gaussian noise (σ = 25) produces a few genuine 0/255 outliers, which is where the remaining overlap comes from.
* **Illumination U_bg** has perfect recall but low precision on natural photos: large scene regions (sky vs ground) look like uneven lighting to a background estimate. Note that *coins* and *page* are counted as false alarms only because their clean versions are labelled negative — both are textbook examples of genuinely uneven illumination. The threshold is kept at 0.15 on purpose: verify-after-fix turns false alarms into a harmless `no_benefit` verdict, while a miss would hide the largest improvement the pipeline can make.
* **Screenshots** (enlarged noise) are caught by the half-scale tests: specks smeared over 2–3 px and grain that is smooth between neighbouring pixels are pixel-sized again at half scale. Their remaining false alarms are the same grass/gravel textures that already fool the pixel-scale tests.
