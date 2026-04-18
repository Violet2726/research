"""数据集加载与样本标准化。

所有 benchmark 最终都会被转换成统一的 ``DatasetSample`` 结构，
这样提示构造、模型调用和评测逻辑就不需要关心底层文件格式差异。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any

import pyarrow.parquet as pq

from api_baselines.config import BenchmarkConfig


@dataclass(frozen=True)
class DatasetSample:
    """单个样本的统一表示。"""

    dataset: str
    sample_id: str
    question: str
    reference_answer: str
    prompt_context: str
    metadata: dict[str, Any]


def load_samples(config: BenchmarkConfig) -> list[DatasetSample]:
    """根据 benchmark 配置选择对应 loader，并返回标准化样本列表。"""
    loader_map = {
        "gsm8k_jsonl": _load_gsm8k,
        "strategyqa_json": _load_strategyqa,
        "hotpotqa_parquet": _load_hotpotqa,
    }
    return loader_map[config.loader](config)


def _load_gsm8k(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载 GSM8K JSONL，并抽取 ``####`` 后的最终答案作为金标。"""
    path = Path(config.source_path)
    samples: list[DatasetSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            samples.append(
                DatasetSample(
                    dataset=config.slug,
                    sample_id=f"{config.sample_id_prefix}-{index:05d}",
                    question=record["question"].strip(),
                    reference_answer=_extract_gsm8k_gold(record["answer"]),
                    prompt_context="",
                    metadata={"raw_index": index},
                )
            )
    return samples


def _load_strategyqa(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载 StrategyQA JSON，并把布尔答案标准化成 yes/no。"""
    path = Path(config.source_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples: list[DatasetSample] = []
    for index, record in enumerate(payload):
        answer = "yes" if bool(record["answer"]) else "no"
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=str(record.get("qid") or f"{config.sample_id_prefix}-{index:05d}"),
                question=record["question"].strip(),
                reference_answer=answer,
                prompt_context="",
                metadata={"raw_index": index, "term": record.get("term")},
            )
        )
    return samples


def _load_hotpotqa(config: BenchmarkConfig) -> list[DatasetSample]:
    """加载 HotpotQA parquet，并把上下文段落渲染成可直接放入提示词的文本。"""
    table = pq.read_table(config.source_path)
    payload = table.to_pylist()
    samples: list[DatasetSample] = []
    for index, record in enumerate(payload):
        sample_id = record.get("id") or f"{config.sample_id_prefix}-{index:05d}"
        samples.append(
            DatasetSample(
                dataset=config.slug,
                sample_id=sample_id,
                question=str(record["question"]).strip(),
                reference_answer=str(record["answer"]).strip(),
                prompt_context=_render_hotpot_context(record["context"]),
                metadata={
                    "raw_index": index,
                    "type": record.get("type"),
                    "level": record.get("level"),
                },
            )
        )
    return samples


def _render_hotpot_context(context: dict[str, Any]) -> str:
    """把 HotpotQA 的结构化上下文压平成带标题的多段文本。"""
    titles = context["title"]
    paragraphs = context["sentences"]
    rendered: list[str] = []
    for title, sentences in zip(titles, paragraphs, strict=False):
        joined = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
        rendered.append(f"[{title}] {joined}")
    return "\n".join(rendered)


def _extract_gsm8k_gold(answer: str) -> str:
    """从 GSM8K 标注答案中提取最终数字，便于与模型输出对齐。"""
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
    if match:
        return match.group(1).replace(",", "")
    return answer.strip()
