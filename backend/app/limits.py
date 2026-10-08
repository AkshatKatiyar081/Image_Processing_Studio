"""Size limits for uploads and downloads, in one place (also reported to the dashboard by /api/health).

Vercel functions refuse request and response bodies over 4.5 MB, so on Vercel (where the VERCEL environment
variable is set automatically) both limits default to 4.2 MiB ≈ 4.4 MB, leaving room for the multipart
envelope. Elsewhere uploads default to 15 MB. The download limit defaults to the upload limit, so every file
the server returns (a result for "Keep editing", for example) can be uploaded again.

Env vars: MAX_UPLOAD_MB, MAX_DOWNLOAD_MB (MiB, i.e. × 1024 × 1024).
"""
from __future__ import annotations

import os

MIB = 1024 * 1024
ON_VERCEL = bool(os.getenv("VERCEL"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "4.2" if ON_VERCEL else "15"))
MAX_DOWNLOAD_MB = float(os.getenv("MAX_DOWNLOAD_MB", str(MAX_UPLOAD_MB)))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * MIB)
MAX_DOWNLOAD_BYTES = int(MAX_DOWNLOAD_MB * MIB)


def mb(n: int) -> str:
    """Bytes as the dashboard shows them (1 MB = 1024 × 1024 bytes)."""
    return f"{n / MIB:.1f} MB"
