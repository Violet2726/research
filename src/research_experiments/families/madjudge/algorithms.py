"""MADJudge 核心算法。

实现论文 "Multi-Agent Debate for LLM Judges with Adaptive Stability Detection" 的核心机制：
- Beta-Binomial 混合模型估计 judges 的正确率分布（论文 Section 5.1-5.2）
- 基于 KS 检验的自适应稳定性检测（论文 Section 5.3）
- Majority Vote 聚合

论文：arXiv:2510.12697
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import comb, betaln


@dataclass(frozen=True)
class StabilityState:
    """稳定性检测状态。"""

    is_stable: bool = False
    ks_statistic: float = 1.0
    consecutive_stable_rounds: int = 0
    current_round: int = 0
    correct_count: int = 0
    total_agents: int = 0


@dataclass(frozen=True)
class BetaBinomialParams:
    """Beta-Binomial 混合模型参数（论文 Equation 5-7）。"""

    weight: float  # 混合权重 w_t
    alpha1: float  # 第一个 Beta 分布的 alpha 参数
    beta1: float  # 第一个 Beta 分布的 beta 参数
    alpha2: float  # 第二个 Beta 分布的 alpha 参数
    beta2: float  # 第二个 Beta 分布的 beta 参数


def beta_binomial_pmf(s: int, k: int, alpha: float, beta: float) -> float:
    """计算 Beta-Binomial 分布的 PMF。

    论文定义：BB(s; k, α, β) = C(k, s) * B(s + α, k - s + β) / B(α, β)

    Args:
        s: 正确答案数量
        k: 总 judges 数量
        alpha: Beta 分布的 alpha 参数
        beta: Beta 分布的 beta 参数

    Returns:
        概率质量
    """
    log_pmf = (
        np.log(comb(k, s, exact=True))
        + betaln(s + alpha, k - s + beta)
        - betaln(alpha, beta)
    )
    return float(np.exp(log_pmf))


def mixture_beta_binomial_loglik(
    params: np.ndarray,
    observations: np.ndarray,
    k: int,
) -> float:
    """计算混合 Beta-Binomial 模型的负对数似然。

    论文 Equation 6：L(θ_t) = Σ_j log[w * BB(s_j; k, α1, β1) + (1-w) * BB(s_j; k, α2, β2)]

    Args:
        params: [w, alpha1, beta1, alpha2, beta2]
        observations: 观察到的正确答案数数组
        k: 总 judges 数量

    Returns:
        负对数似然（用于最小化）
    """
    w, alpha1, beta1, alpha2, beta2 = params

    # 确保参数有效
    if w < 0 or w > 1 or alpha1 <= 0 or beta1 <= 0 or alpha2 <= 0 or beta2 <= 0:
        return 1e10

    loglik = 0.0
    for s in observations:
        pmf1 = beta_binomial_pmf(int(s), k, alpha1, beta1)
        pmf2 = beta_binomial_pmf(int(s), k, alpha2, beta2)
        mixture_pmf = w * pmf1 + (1 - w) * pmf2

        if mixture_pmf <= 0:
            loglik += -100  # 惩罚
        else:
            loglik += np.log(mixture_pmf)

    return -loglik


def estimate_beta_binomial_params(
    observations: list[int],
    k: int,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> BetaBinomialParams:
    """使用 EM 算法估计混合 Beta-Binomial 模型参数。

    论文 Section 5.2：使用 EM 算法估计参数 θ_t = {w_t, α1_t, β1_t, α2_t, β2_t}
    使用 L-BFGS-B 优化方法。

    Args:
        observations: 观察到的正确答案数列表
        k: 总 judges 数量
        max_iter: 最大迭代次数
        tol: 收敛阈值（论文使用 10^-6）

    Returns:
        估计的参数
    """
    if not observations or k == 0:
        return BetaBinomialParams(weight=0.5, alpha1=1.0, beta1=1.0, alpha2=1.0, beta2=1.0)

    obs_array = np.array(observations, dtype=float)

    # 多次随机初始化，选择最佳结果
    best_params = None
    best_loglik = float("inf")

    for _ in range(5):
        # 随机初始化
        init_w = np.random.uniform(0.3, 0.7)
        init_alpha1 = np.random.uniform(1, 5)
        init_beta1 = np.random.uniform(1, 5)
        init_alpha2 = np.random.uniform(1, 5)
        init_beta2 = np.random.uniform(1, 5)

        init_params = np.array([init_w, init_alpha1, init_beta1, init_alpha2, init_beta2])

        # 边界约束
        bounds = [
            (0.01, 0.99),  # w
            (0.1, 50.0),  # alpha1
            (0.1, 50.0),  # beta1
            (0.1, 50.0),  # alpha2
            (0.1, 50.0),  # beta2
        ]

        try:
            result = minimize(
                mixture_beta_binomial_loglik,
                init_params,
                args=(obs_array, k),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": max_iter, "ftol": tol},
            )

            if result.fun < best_loglik:
                best_loglik = result.fun
                best_params = result.x
        except Exception:
            continue

    if best_params is None:
        return BetaBinomialParams(weight=0.5, alpha1=1.0, beta1=1.0, alpha2=1.0, beta2=1.0)

    w, alpha1, beta1, alpha2, beta2 = best_params
    return BetaBinomialParams(
        weight=float(w),
        alpha1=float(alpha1),
        beta1=float(beta1),
        alpha2=float(alpha2),
        beta2=float(beta2),
    )


def compute_cdf(params: BetaBinomialParams, x_points: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """计算混合 Beta-Binomial 分布的 CDF。

    论文 Equation 7：P_t(θ) = w_t * Beta(θ; α1_t, β1_t) + (1 - w_t) * Beta(θ; α2_t, β2_t)

    Args:
        params: 混合模型参数
        x_points: CDF 计算点（默认 0-1 之间 1000 个点）

    Returns:
        (x_points, cdf_values)
    """
    if x_points is None:
        x_points = np.linspace(0, 1, 1000)

    # 混合 Beta 分布的 CDF
    cdf1 = stats.beta.cdf(x_points, params.alpha1, params.beta1)
    cdf2 = stats.beta.cdf(x_points, params.alpha2, params.beta2)
    mixture_cdf = params.weight * cdf1 + (1 - params.weight) * cdf2

    return x_points, mixture_cdf


def compute_ks_statistic(
    params_t: BetaBinomialParams,
    params_t_minus_1: BetaBinomialParams,
) -> float:
    """计算两轮之间的 Kolmogorov-Smirnov 统计量。

    论文 Equation 8：D_t = sup |F_t(θ) - F_{t-1}(θ)|

    Args:
        params_t: 当前轮的参数
        params_t_minus_1: 上一轮的参数

    Returns:
        KS 统计量
    """
    x_points = np.linspace(0, 1, 1000)
    _, cdf_t = compute_cdf(params_t, x_points)
    _, cdf_t_minus_1 = compute_cdf(params_t_minus_1, x_points)

    # KS 统计量 = 最大绝对差
    ks_stat = float(np.max(np.abs(cdf_t - cdf_t_minus_1)))
    return ks_stat


def check_stability_batch(
    round_majority_counts: list[int],
    k: int,
    ks_threshold: float = 0.05,
    consecutive_stable_required: int = 2,
    previous_params: BetaBinomialParams | None = None,
    consecutive_stable_count: int = 0,
) -> tuple[StabilityState, BetaBinomialParams]:
    """跨题批次级稳定性检测（论文 Section 5.3 的正确实现）。

    论文的 Beta-Binomial 模型跨所有题目聚合观测：
    S_t = 每题的 majority count（作为正确答案数的代理）
    用这些观测拟合混合 Beta-Binomial 分布，比较相邻轮次的分布差异。

    Args:
        round_majority_counts: 当前轮所有题目的 majority count 列表
        k: agent 数量
        ks_threshold: KS 统计量阈值（论文使用 0.05）
        consecutive_stable_required: 需要连续稳定的轮数（论文使用 2）
        previous_params: 上一轮的模型参数
        consecutive_stable_count: 已连续稳定的轮数

    Returns:
        (稳定性状态, 当前轮的模型参数)
    """
    if not round_majority_counts:
        return (
            StabilityState(is_stable=False, ks_statistic=1.0),
            BetaBinomialParams(weight=0.5, alpha1=1.0, beta1=1.0, alpha2=1.0, beta2=1.0),
        )

    # 用当前轮所有题目的 majority count 拟合 Beta-Binomial
    current_params = estimate_beta_binomial_params(round_majority_counts, k)

    # 如果没有上一轮参数，无法计算 KS 统计量
    if previous_params is None:
        return (
            StabilityState(
                is_stable=False,
                ks_statistic=1.0,
                consecutive_stable_rounds=0,
                current_round=0,
                correct_count=sum(round_majority_counts),
                total_agents=k * len(round_majority_counts),
            ),
            current_params,
        )

    # 计算 KS 统计量
    ks_stat = compute_ks_statistic(current_params, previous_params)

    # 检查是否稳定
    is_stable = ks_stat < ks_threshold

    # 更新连续稳定轮数
    new_consecutive_stable = consecutive_stable_count + 1 if is_stable else 0

    return (
        StabilityState(
            is_stable=new_consecutive_stable >= consecutive_stable_required,
            ks_statistic=ks_stat,
            consecutive_stable_rounds=new_consecutive_stable,
            current_round=0,
            correct_count=sum(round_majority_counts),
            total_agents=k * len(round_majority_counts),
        ),
        current_params,
    )


def compute_majority_count(answers: list[str]) -> int:
    """计算一组答案中多数答案的出现次数。"""
    from collections import Counter
    if not answers:
        return 0
    counter = Counter(answers)
    return counter.most_common(1)[0][1]


def aggregate_majority_vote(answers: list[str]) -> tuple[str, dict[str, int]]:
    """Majority Vote 聚合。

    论文使用 SoM (Majority Vote) 作为聚合方法。

    Args:
        answers: 答案列表

    Returns:
        (多数答案, 答案计数)
    """
    if not answers:
        return "", {}

    from collections import Counter
    counter = Counter(answers)
    majority_answer = counter.most_common(1)[0][0]
    return majority_answer, dict(counter)
