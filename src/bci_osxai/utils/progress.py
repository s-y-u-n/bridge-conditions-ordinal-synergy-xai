from __future__ import annotations

import sys
from typing import Iterable, Iterator, Optional, TypeVar


T = TypeVar("T")


def tqdm_wrap(iterable: Iterable[T], *, desc: str, enabled: bool) -> Iterable[T]:
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm  # noqa: PLC0415
    except Exception:
        return iterable
    # progress to stderr so stdout can stay off
    return tqdm(iterable, desc=desc, file=sys.stderr, leave=False)


def is_progress_enabled(config: dict | None) -> bool:
    if not config:
        return False
    progress = config.get("progress", {})
    if isinstance(progress, dict):
        return bool(progress.get("enabled", False))
    return bool(progress)
