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

## Interpretation

* **Noise σ** separates cleanly; the only false alarms come from the *grass* texture, whose fine random detail is statistically similar to noise (a known limit of any single-image estimator).
* **Impulse density**: the calibrated threshold (0.75 %) is now in `config.py`. Heavy Gaussian noise (σ = 25) produces a few genuine 0/255 outliers, which is where the remaining overlap comes from.
* **Illumination U_bg** has perfect recall but low precision on natural photos: large scene regions (sky vs ground) look like uneven lighting to a background estimate. Note that *coins* and *page* are counted as false alarms only because their clean versions are labelled negative — both are textbook examples of genuinely uneven illumination. The threshold is kept at 0.15 on purpose: verify-after-fix turns false alarms into a harmless `no_benefit` verdict, while a miss would hide the largest improvement the pipeline can make.
