# Runtime (1024 × 1024 RGB, median of 3 runs, 2 vCPU container)

| Module | Params | OpenCV engine (ms) | NumPy engine (ms) |
|---|---|---|---|
| contrast | `{'mode': 'hist_eq'}` | 41 | 37 |
| contrast | `{'mode': 'flatten'}` | 82 | 95 |
| denoise | `{'filter_type': 'median', 'kernel_size': 5}` | 78 | 777 |
| denoise | `{'filter_type': 'gaussian', 'sigma': 1.5}` | 74 | 136 |
| sharpen_edges | `{'operation': 'unsharp'}` | 69 | 80 |
| sharpen_edges | `{'operation': 'gradient', 'operator': 'sobel'}` | 19 | 31 |
| segment | `{'method': 'otsu'}` | 78 | 89 |
| segment | `{'method': 'region_growing', 'seed': [512, 700]}` | 55 | 69 |
| guided:segment | salt & pepper 5 % | 622 | – |
| guided:edges | salt & pepper 5 % | 421 | – |
| diagnose | – | 138 | – |