"""Built-in demo images. Each one is chosen to show off a particular feature."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from skimage import data

from ivp_pipeline import synthetic as S

from .codec import encode_png_bytes

# id, title, hint shown in the UI, builder
_SPECS = [
    ("coins", "Coins", "Real uneven lighting. Try Compare → Segment",
     lambda: data.coins()),
    ("shapes_uneven", "Shapes, uneven light", "Otsu fails directly, works after flattening",
     lambda: S.illumination_ramp(S.scene(101, polarity="dark", texture=4.0)[0], 50)),
    ("shapes_saltpepper", "Shapes, salt & pepper", "Median filter before segmenting",
     lambda: S.salt_pepper(S.scene(7, texture=4.0)[0], 0.08, seed=3)),
    ("camera_noisy", "Cameraman, noisy", "Try Compare → Find edges",
     lambda: S.gaussian_noise(data.camera(), 25, seed=4)),
    ("astronaut_dark", "Astronaut, too dark", "Try Auto Fix or gamma",
     lambda: S.apply_gamma(data.astronaut(), 2.4)),
    ("camera", "Cameraman", "Clean classic test image",
     lambda: data.camera()),
    ("astronaut", "Astronaut", "Clean colour image",
     lambda: data.astronaut()),
    ("moon", "Moon", "Low contrast. Try histogram equalization",
     lambda: data.moon()),
    ("page", "Page", "Scanned text with shading",
     lambda: data.page()),
    ("coffee", "Coffee", "Colourful. Try Compression",
     lambda: data.coffee()),
    ("chelsea", "Chelsea the cat", "Fur texture. Try Compression",
     lambda: data.chelsea()),
]


@lru_cache(maxsize=1)
def catalogue() -> dict[str, dict]:
    out = {}
    for sid, title, hint, build in _SPECS:
        img = np.ascontiguousarray(build())
        out[sid] = {"id": sid, "title": title, "hint": hint,
                    "width": int(img.shape[1]), "height": int(img.shape[0]),
                    "png": encode_png_bytes(img)}
    return out


def listing() -> list[dict]:
    return [{k: v for k, v in s.items() if k != "png"} | {"url": f"/api/samples/{s['id']}.png"}
            for s in catalogue().values()]
