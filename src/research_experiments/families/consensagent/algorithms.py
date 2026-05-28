"""CONSENSAGENT 核心算法。

本模块实现论文中的关键算法：
- 触发机制（停滞触发 t0、答案互换触发 t1、解释相似度触发 t2）
- 一致性分数计算
- Phase 4 团队答案聚合（论文 Equation 1）
- 谄媚率计算

论文参考：
- t0（停滞）：多数 agent 在连续轮次中保持相同答案，且未达成共识
- t1（答案互换）：多数 agent 在轮次间互换答案 → 潜在谄媚
- t2（解释相似度）：agent 复制他人答案时解释余弦相似度 > 80%
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TriggerState:
    """触发机制的状态。"""

    stagnation_triggered: bool = False
    sycophancy_triggered: bool = False
    trigger_round: int | None = None
    trigger_type: str | None = None  # "stagnation" | "answer_swap" | "explanation_similarity"


@dataclass(frozen=True)
class ConsistencyScore:
    """一致性分数。"""

    score: float
    majority_count: int
    total_count: int
    is_consensus: bool
    majority_answer: str = ""


def compute_consistency_score(answers: list[str]) -> ConsistencyScore:
    """计算一组答案的一致性分数。

    论文定义：一致性分数 = 最常见答案的出现次数 / 总 agent 数。
    """
    if not answers:
        return ConsistencyScore(score=0.0, majority_count=0, total_count=0, is_consensus=False)

    from collections import Counter

    counter = Counter(answers)
    majority_answer, majority_count = counter.most_common(1)[0]
    total_count = len(answers)
    score = majority_count / total_count
    is_consensus = majority_count == total_count

    return ConsistencyScore(
        score=score,
        majority_count=majority_count,
        total_count=total_count,
        is_consensus=is_consensus,
        majority_answer=majority_answer,
    )


def detect_stagnation(
    round_history: list[list[dict[str, Any]]],
    stagnation_threshold: int = 2,
    current_round_answers: list[dict[str, Any]] | None = None,
) -> bool:
    """检测辩论停滞（论文 t0 触发条件）。

    论文定义：当多数 agent 在连续 stagnation_threshold 轮中保持相同答案，
    且未达成共识时，触发停滞。

    包含 current_round_answers 以提前一轮检测。
    """
    # 构建完整历史（含当前轮）
    all_rounds = list(round_history)
    if current_round_answers:
        all_rounds.append(current_round_answers)

    # 需要 stagnation_threshold+1 个数据点才能做 stagnation_threshold 次连续比较
    if len(all_rounds) < stagnation_threshold + 1:
        return False

    recent = all_rounds[-(stagnation_threshold + 1):]
    for i in range(1, len(recent)):
        prev_answers = {a.get("agent_id"): str(a.get("answer", "")) for a in recent[i - 1]}
        curr_answers = {a.get("agent_id"): str(a.get("answer", "")) for a in recent[i]}
        unchanged = sum(
            1 for aid in prev_answers
            if prev_answers[aid] == curr_answers.get(aid, "")
        )
        # 多数 agent 保持相同答案
        if unchanged <= len(prev_answers) / 2:
            return False

    return True


def detect_sycophancy_swapping(
    prev_round_answers: list[dict[str, Any]],
    current_round_answers: list[dict[str, Any]],
) -> bool:
    """检测答案互换谄媚（论文 t1 触发条件）。

    论文定义：当多数 agent 在轮次间互换答案时，判定为潜在谄媚行为。
    互换意味着 agent A 和 agent B 在相邻轮次中交换了彼此的答案，
    这表明它们可能在复制对方而非独立推理。
    """
    if len(prev_round_answers) < 2 or len(current_round_answers) < 2:
        return False

    prev_map = {a.get("agent_id"): str(a.get("answer", "")) for a in prev_round_answers}
    curr_map = {a.get("agent_id"): str(a.get("answer", "")) for a in current_round_answers}

    agent_ids = list(prev_map.keys())
    swapped_agents: set[int] = set()

    for i, id_a in enumerate(agent_ids):
        if id_a in swapped_agents:
            continue
        for id_b in agent_ids[i + 1 :]:
            if id_b in swapped_agents:
                continue
            prev_a, prev_b = prev_map[id_a], prev_map[id_b]
            curr_a, curr_b = curr_map.get(id_a, ""), curr_map.get(id_b, "")
            # 检测互换：A 和 B 交换了答案，且原答案不同
            if (
                prev_a != prev_b
                and prev_a == curr_b
                and prev_b == curr_a
            ):
                swapped_agents.update([id_a, id_b])

    # 多数 agent 卷入互换
    return len(swapped_agents) > len(agent_ids) / 2


def detect_sycophancy_copycat(
    current_round_answers: list[dict[str, Any]],
    consistency_threshold: float = 0.8,
) -> bool:
    """检测复制型谄媚（论文 t2 触发条件的简化版）。

    论文定义：当 agent 复制他人答案且解释余弦相似度 > 80% 时触发。
    由于本实现不依赖本地 embedding，我们使用简化条件：
    当超过半数 agent 改变到多数答案且一致性超过阈值时，判定为复制型谄媚。

    这是对论文 t2 的工程近似，完整实现需引入余弦相似度计算。
    """
    if len(current_round_answers) < 2:
        return False

    answers = [str(a.get("answer", "")) for a in current_round_answers]
    consistency = compute_consistency_score(answers)

    if consistency.score <= consistency_threshold:
        return False

    majority_answer = consistency.majority_answer
    changed_to_majority = 0
    for agent_data in current_round_answers:
        prev_answer = str(agent_data.get("previous_answer", ""))
        current_answer = str(agent_data.get("answer", ""))
        if prev_answer and prev_answer != current_answer and current_answer == majority_answer:
            changed_to_majority += 1

    return changed_to_majority > len(current_round_answers) / 2


def check_triggers(
    round_history: list[list[dict[str, Any]]],
    current_round_answers: list[dict[str, Any]],
    stagnation_threshold: int = 2,
    sycophancy_consistency_threshold: float = 0.8,
    check_sycophancy_on_consensus: bool = True,
) -> TriggerState:
    """检查所有触发条件（论文 t0、t1、t2）。

    根据论文，触发条件可以在达成共识时也激活——
    如果共识是通过谄媚或停滞达成的，仍需触发。
    """
    round_idx = len(round_history)

    # 计算当前一致性
    answers = [str(a.get("answer", "")) for a in current_round_answers]
    consistency = compute_consistency_score(answers)

    # t0：停滞触发 — 多数 agent 连续保持相同答案
    stagnation = detect_stagnation(round_history, stagnation_threshold, current_round_answers)
    if stagnation and not consistency.is_consensus:
        return TriggerState(
            stagnation_triggered=True,
            trigger_round=round_idx,
            trigger_type="stagnation",
        )

    # t1：答案互换触发 — 多数 agent 互换答案
    if len(round_history) >= 1:
        prev_round = round_history[-1]
        if detect_sycophancy_swapping(prev_round, current_round_answers):
            if check_sycophancy_on_consensus or not consistency.is_consensus:
                return TriggerState(
                    sycophancy_triggered=True,
                    trigger_round=round_idx,
                    trigger_type="answer_swap",
                )

    # t2：复制型谄媚 — 多数 agent 改变到多数答案
    if detect_sycophancy_copycat(current_round_answers, sycophancy_consistency_threshold):
        if check_sycophancy_on_consensus or not consistency.is_consensus:
            return TriggerState(
                sycophancy_triggered=True,
                trigger_round=round_idx,
                trigger_type="copycat",
            )

    return TriggerState()


def aggregate_weighted_answer(
    agent_answers: list[dict[str, Any]],
    initial_answers: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, int]]:
    """Phase 4 置信度加权聚合。

    按答案分组累加置信度，选择总置信度最高的答案。
    （论文 Equation 1 的简化版——完整版需要 Phase 3 prompt 优化配合，
    无 Phase 3 时简单置信度求和在实践中更稳定）
    """
    if not agent_answers:
        return "", {}

    answer_confidence: dict[str, float] = {}
    answer_counts: dict[str, int] = {}

    for agent_data in agent_answers:
        answer = str(agent_data.get("answer", "")).strip()
        confidence = float(agent_data.get("confidence", 0.5))
        if not answer:
            continue
        answer_confidence[answer] = answer_confidence.get(answer, 0.0) + confidence
        answer_counts[answer] = answer_counts.get(answer, 0) + 1

    if not answer_confidence:
        return "", {}

    best_answer = max(answer_confidence.items(), key=lambda x: x[1])[0]
    return best_answer, answer_counts


def compute_sycophancy_rate(
    round_history: list[list[dict[str, Any]]],
    consistency_threshold: float = 0.8,
) -> float:
    """计算谄媚率。

    论文定义：谄媚率 = 检测到谄媚（t1 或 t2）的轮数 / 总轮数。
    注意论文排除首轮即达成共识的实例。
    """
    if not round_history:
        return 0.0

    sycophancy_count = 0
    for i, round_answers in enumerate(round_history):
        # t1：答案互换检测
        if i > 0 and detect_sycophancy_swapping(round_history[i - 1], round_answers) or detect_sycophancy_copycat(round_answers, consistency_threshold):
            sycophancy_count += 1

    return sycophancy_count / len(round_history)


def _get_majority_answer(answers: list[str]) -> str:
    """获取多数答案。"""
    from collections import Counter

    if not answers:
        return ""
    counter = Counter(answers)
    return counter.most_common(1)[0][0]
