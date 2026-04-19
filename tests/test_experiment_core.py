from __future__ import annotations

from pathlib import Path
import json
import time

from experiment_core.cache import CachedResponse, RequestCache, json_dump
from experiment_core.config import load_benchmark_config, load_model_catalog, parse_model_ref
from experiment_core.datasets import generate_split_manifests, load_split_ids, select_samples
from experiment_core.fallbacks import extract_fallback_answer
from experiment_core.parsing import parse_model_output
from experiment_core.rate_limits import SlidingWindowRateLimiter


def test_parse_model_ref() -> None:
    assert parse_model_ref("dashscope/qwen-turbo") == ("dashscope", "qwen-turbo")


def test_load_model_catalog() -> None:
    catalog = load_model_catalog()
    assert catalog
    assert all("/" in key for key in catalog)


def test_generate_and_load_split_manifests(tmp_path: Path) -> None:
    source_path = tmp_path / "gsm8k.jsonl"
    source_path.write_text(
        "\n".join(
            [
                json.dumps({"question": "1+1?", "answer": "#### 2"}, ensure_ascii=False),
                json.dumps({"question": "2+2?", "answer": "#### 4"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "benchmark.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "Toy GSM8K"',
                'slug = "toy_gsm8k"',
                'loader = "gsm8k_jsonl"',
                f'source_path = "{source_path.as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "toy"',
                'question_field = "question"',
                'answer_field = "answer"',
                'smoke_size = 1',
                'pilot_size = 2',
                'main_size = 2',
                'random_seed = 42',
                'notes = ""',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark = load_benchmark_config(benchmark_path)
    created = generate_split_manifests([benchmark], tmp_path / "splits")
    assert created
    smoke_ids = load_split_ids("toy_gsm8k", "smoke20_seed42", tmp_path / "splits")
    samples = select_samples(benchmark, "smoke20_seed42", tmp_path / "splits")
    assert len(smoke_ids) == 1
    assert [sample.sample_id for sample in samples] == smoke_ids


def test_parse_model_output_and_fallbacks() -> None:
    parsed, status = parse_model_output('{"reasoning":"short","final_answer":"yes"}')
    assert status == "direct_json"
    assert parsed["final_answer"] == "yes"

    fallback = extract_fallback_answer("strategyqa", "The answer is yes.")
    assert fallback is not None
    assert fallback[0]["final_answer"] == "yes"


def test_request_cache_round_trip(tmp_path: Path) -> None:
    cache = RequestCache(tmp_path / "requests.sqlite")
    record = CachedResponse(
        cache_key="abc",
        payload_json=json_dump({"a": 1}),
        response_json=json_dump({"b": 2}),
        http_status=200,
        latency_ms=12.5,
        provider_request_id="req_1",
    )
    cache.put(record)
    loaded = cache.get("abc")
    cache.close()
    assert loaded == record


def test_rate_limiter_without_waiting() -> None:
    limiter = SlidingWindowRateLimiter(requests_per_minute=100, tokens_per_minute=1000)
    started = time.monotonic()
    limiter.acquire(10)
    limiter.acquire(10)
    assert time.monotonic() - started < 1.0
