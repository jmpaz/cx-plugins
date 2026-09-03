from __future__ import annotations

import contextlib
from typing import Generator


@contextlib.contextmanager
def download_lane() -> Generator[None]:
    try:
        from contextualize.concurrency import download_lane as _download_lane
    except Exception:
        yield
        return
    with _download_lane():
        yield
