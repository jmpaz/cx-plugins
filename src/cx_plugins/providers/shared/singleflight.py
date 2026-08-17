from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import TypeVar

from .progress import record_progress

T = TypeVar("T")


class KeyedSingleflight:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[object, Future[object]] = {}

    def run(
        self,
        key: object,
        work: Callable[[], T],
        *,
        provider: str,
        operation: str,
        target: str | None = None,
    ) -> T:
        with self._lock:
            future = self._calls.get(key)
            leader = future is None
            if future is None:
                future = Future()
                self._calls[key] = future
        if not leader:
            record_progress(provider, operation, "coalesced", target=target)
            return future.result()
        try:
            result = work()
        except BaseException as exc:
            future.set_exception(exc)
            raise
        else:
            future.set_result(result)
            return result
        finally:
            with self._lock:
                if self._calls.get(key) is future:
                    del self._calls[key]


singleflight = KeyedSingleflight()
