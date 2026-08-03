from __future__ import annotations

from pathlib import Path

from research_experiments.core.execution.cache import CachedResponse, RequestCacheRouter
from research_experiments.families.contrastive_active_testing.run.execute import CatchEndpoint


def test_identical_requests_from_different_experiments_share_one_global_shard(tmp_path: Path) -> None:
    first_router = RequestCacheRouter(tmp_path)
    first = first_router.for_request_target(
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="bbeh",
    )
    first.put(CachedResponse("shared", "{}", '{"assistant_text":"A"}', 1))
    first_router.close()

    second_router = RequestCacheRouter(tmp_path)
    second = second_router.for_request_target(
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="bbeh",
    )
    assert second.get("shared") is not None
    assert "namespaces" not in second.db_path.parts
    second_router.close()


def test_all_catch_roles_use_the_same_global_cache() -> None:
    cache = object()
    endpoint = CatchEndpoint(
        backbone=object(),
        provider=object(),  # type: ignore[arg-type]
        cache=cache,
        throttle=object(),  # type: ignore[arg-type]
    )
    assert endpoint.cache_for_role("stage_a_solver") is cache
    assert endpoint.cache_for_role("d4_source_compiler") is cache
    assert endpoint.cache_lookup_namespaces_for_role("pair_judge") == ("global_cache",)
