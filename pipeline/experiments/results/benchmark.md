# Direct vs Guided — benchmark

Mean over images at each level. *Gate fired* = share of images where the quality gate prescribed a prerequisite; *recommended* = share where verify-after-fix confirmed the gain. When the gate does not fire, Guided = Direct.

## Segmentation (10 synthetic scenes, exact masks) — IoU

| Degradation | Level | Direct | Guided | Gain | Gate fired | Recommended |
|---|---|---|---|---|---|---|
| sp | 1 | 0.977 | 0.990 | 0.014 | 100% | 100% |
| sp | 5 | 0.903 | 0.988 | 0.085 | 100% | 100% |
| sp | 10 | 0.726 | 0.985 | 0.259 | 100% | 100% |
| sp | 20 | 0.260 | 0.973 | 0.713 | 100% | 100% |
| gauss | 5 | 0.992 | 0.992 | 0.000 | 0% | 0% |
| gauss | 10 | 0.992 | 0.989 | -0.002 | 100% | 100% |
| gauss | 20 | 0.972 | 0.986 | 0.014 | 100% | 100% |
| gauss | 30 | 0.934 | 0.981 | 0.047 | 100% | 100% |
| ramp | 20 | 0.992 | 0.992 | 0.000 | 0% | 0% |
| ramp | 40 | 0.201 | 0.992 | 0.792 | 100% | 100% |
| ramp | 70 | 0.140 | 0.992 | 0.852 | 100% | 100% |

## Edge detection (5 natural images; reference = Sobel edges of the clean image) — F1

| Degradation | Level | Direct | Guided | Gain | Gate fired | Recommended |
|---|---|---|---|---|---|---|
| gauss | 5 | 0.988 | 0.988 | 0.000 | 0% | 0% |
| gauss | 10 | 0.961 | 0.961 | 0.000 | 0% | 0% |
| gauss | 20 | 0.793 | 0.876 | 0.083 | 100% | 100% |
| gauss | 30 | 0.588 | 0.821 | 0.233 | 100% | 100% |
| sp | 1 | 0.795 | 0.914 | 0.118 | 100% | 100% |
| sp | 5 | 0.510 | 0.913 | 0.403 | 100% | 100% |

## Sharpening (reference = unsharp mask of the clean image) — PSNR dB

| Degradation | Level | Direct | Guided | Gain | Gate fired | Recommended |
|---|---|---|---|---|---|---|
| gauss | 5 | 28.95 | 28.95 | 0.00 | 0% | 0% |
| gauss | 10 | 23.03 | 25.70 | 2.67 | 100% | 100% |
| gauss | 20 | 17.37 | 24.04 | 6.67 | 100% | 100% |

## HE of a dark image (reference = HE of the clean dark image) — PSNR dB

| Degradation | Level | Direct | Guided | Gain | Gate fired | Recommended |
|---|---|---|---|---|---|---|
| gauss_dark | 5 | 21.45 | 21.45 | 0.00 | 0% | 0% |
| gauss_dark | 10 | 18.06 | 18.46 | 0.40 | 100% | 100% |
| gauss_dark | 15 | 16.00 | 17.73 | 1.73 | 100% | 100% |
