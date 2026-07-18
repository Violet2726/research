from __future__ import annotations

from research_experiments.core.execution.cache import CachedResponse
from research_experiments.families.contrastive_active_testing.cache_layers import ReadThroughRequestCache


class Cache:
    def __init__(self, records=None) -> None:
        self.records = dict(records or {})
        self.deleted = []

    def get(self, key):
        return self.records.get(key)

    def put(self, record):
        self.records[record.cache_key] = record

    def delete(self, key):
        self.deleted.append(key)


def _record(key: str) -> CachedResponse:
    return CachedResponse(key, "{}", "{}", 200, 1.0, None)


def test_read_through_cache_reads_v1_but_writes_only_v2() -> None:
    primary = Cache()
    fallback = Cache({"shared": _record("shared")})
    cache = ReadThroughRequestCache(
        primary,
        primary_namespace="catch-dev-v2",
        fallback=fallback,
        fallback_namespace="catch-dev-v1",
    )
    assert cache.get("shared") is not None
    assert cache.source_for("shared") == "catch-dev-v1"

    cache.put(_record("new"))
    assert "new" in primary.records
    assert "new" not in fallback.records
    assert cache.source_for("new") == "catch-dev-v2"
