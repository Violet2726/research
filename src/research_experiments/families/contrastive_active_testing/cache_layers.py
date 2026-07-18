"""带可审计只读前身层的 CATCH 角色感知缓存路由。"""

from __future__ import annotations

import threading

from research_experiments.core.execution.cache import CachedResponse, RequestCache


class ReadThroughRequestCache:
    """Write to the active cache while allowing exact-key reads from one predecessor."""

    def __init__(
        self,
        primary: RequestCache,
        *,
        primary_namespace: str,
        fallback: RequestCache | None = None,
        fallback_namespace: str | None = None,
        fallbacks: list[tuple[RequestCache, str]] | None = None,
    ) -> None:
        self.primary = primary
        self.primary_namespace = primary_namespace
        self.fallback = fallback
        self.fallback_namespace = fallback_namespace
        self.fallbacks = list(fallbacks or [])
        if fallback is not None:
            self.fallbacks.insert(0, (fallback, str(fallback_namespace or "predecessor")))
        self._sources: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, cache_key: str) -> CachedResponse | None:
        record = self.primary.get(cache_key)
        source = self.primary_namespace
        if record is None:
            for fallback, fallback_namespace in self.fallbacks:
                record = fallback.get(cache_key)
                if record is not None:
                    source = fallback_namespace
                    break
        with self._lock:
            self._sources[cache_key] = source if record is not None else "network"
        return record

    def put(self, record: CachedResponse) -> None:
        self.primary.put(record)
        with self._lock:
            self._sources[record.cache_key] = self.primary_namespace

    def delete(self, cache_key: str) -> None:
        self.primary.delete(cache_key)

    def source_for(self, cache_key: str) -> str:
        with self._lock:
            return self._sources.get(cache_key, "network")
