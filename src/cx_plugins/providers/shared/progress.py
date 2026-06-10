from __future__ import annotations

def record_progress(
    provider: str,
    operation: str,
    outcome: str,
    *,
    target: str | None = None,
    detail: str | None = None,
    count: int | None = None,
    size_bytes: int | None = None,
) -> None:
    try:
        from contextualize.progress import record_progress as _record_progress
    except Exception:
        return
    _record_progress(
        provider,
        operation,
        outcome,
        target=target,
        detail=detail,
        count=count,
        size_bytes=size_bytes,
    )
