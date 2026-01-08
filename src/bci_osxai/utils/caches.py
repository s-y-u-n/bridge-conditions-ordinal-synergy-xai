from __future__ import annotations

import os
from pathlib import Path


def ensure_writable_caches(cache_dir: str | Path = ".cache") -> None:
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    mpl_dir = cache_dir / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")

