from __future__ import annotations

import sys
import threading

from cx_plugins.providers.shared.concurrency import download_lane


def test_download_lane_bounds_peak_occupancy(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXTUALIZE_MEDIA_DOWNLOAD_JOBS", "1")

    lock = threading.Lock()
    active = 0
    peak = 0

    def _work() -> None:
        nonlocal active, peak
        with download_lane():
            with lock:
                active += 1
                peak = max(peak, active)
            threading.Event().wait(0.01)
            with lock:
                active -= 1

    threads = [threading.Thread(target=_work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak == 1


def test_download_lane_is_a_noop_without_contextualize(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "contextualize.concurrency", None)

    entered = False
    with download_lane():
        entered = True

    assert entered
