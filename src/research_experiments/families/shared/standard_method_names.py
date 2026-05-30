"""项目级标准对比方法命名。

这个模块只负责维护跨实验重复出现、且需要严格对齐语义的标准方法名。
横切分析层与治理测试应优先引用这里的常量，避免在多个文件中散落硬编码字符串。
"""

from __future__ import annotations

COT_1 = "cot_1"
SC_5 = "sc_5"
SC_6 = "sc_6"

MV_3 = "mv_3"
MV_4 = "mv_4"
MV_5 = "mv_5"
MV_6 = "mv_6"
MV_7 = "mv_7"

MAD_3A_R1 = "mad_3a_r1"
MAD_3A_R2 = "mad_3a_r2"
MAD_FIXED_R1 = "mad_fixed_r1"
MAD_FIXED_R2 = "mad_fixed_r2"
MAD_FIXED_R3 = "mad_fixed_r3"
VANILLA_MAD_R1_FINAL_VOTE = "vanilla_mad_r1_final_vote"
MV_3_INITIAL = "mv_3_initial"

STANDARD_NO_COMM_BASELINES: frozenset[str] = frozenset(
    {
        COT_1,
        SC_5,
        SC_6,
        MV_3,
        MV_4,
        MV_5,
        MV_6,
        MV_7,
        MV_3_INITIAL,
    }
)

STANDARD_VANILLA_MAD_BASELINES: frozenset[str] = frozenset(
    {
        MAD_3A_R1,
        MAD_3A_R2,
        MAD_FIXED_R1,
        MAD_FIXED_R2,
        MAD_FIXED_R3,
        VANILLA_MAD_R1_FINAL_VOTE,
    }
)

STANDARD_COMPARATOR_METHODS: frozenset[str] = (
    STANDARD_NO_COMM_BASELINES | STANDARD_VANILLA_MAD_BASELINES
)


def is_standard_comparator_method(method_name: str) -> bool:
    """判断一个方法名是否属于项目要求统一语义的标准对比方法。"""

    return method_name in STANDARD_COMPARATOR_METHODS
