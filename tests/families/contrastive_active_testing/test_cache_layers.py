from __future__ import annotations

from research_experiments.core.execution.cache import CachedResponse
from research_experiments.families.contrastive_active_testing.cache_layers import ReadThroughRequestCache
from research_experiments.families.contrastive_active_testing.run.execute import CatchEndpoint


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


def test_pair_and_direct_judges_use_predecessor_cache() -> None:
    baseline = object()
    intervention = object()
    endpoint = CatchEndpoint(
        backbone=object(),
        provider=object(),  # type: ignore[arg-type]
        baseline_cache=baseline,
        intervention_cache=intervention,
        throttle=object(),  # type: ignore[arg-type]
        cache_namespace="catch-dev-kernel_d1_v3",
        baseline_cache_namespace=("catch-dev-cert_v2", "catch-dev-cert_v1"),
    )
    assert endpoint.cache_for_role("direct_judge") is baseline
    assert endpoint.cache_for_role("pair_judge") is baseline
    assert endpoint.cache_lookup_namespaces_for_role("pair_judge") == (
        "catch-dev-kernel_d1_v3",
        "catch-dev-cert_v2",
        "catch-dev-cert_v1",
    )
    assert endpoint.cache_for_role("kernel_atomic_verifier") is intervention
