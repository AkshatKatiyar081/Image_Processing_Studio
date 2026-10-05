# Makes `pytest` (not only `python -m pytest`) find the ivp_pipeline package from the repo root.
import os

# The tests exercise the downscaling path (impulse-aware resize, seed scaling, scale = 0.5 ...), so they
# run with the hosted-server setting. The library default is IVP_MAX_SIDE=0 (full resolution).
os.environ.setdefault("IVP_MAX_SIDE", "1024")
