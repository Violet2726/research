"""标准比较器常量的兼容导出层。"""

from __future__ import annotations

from research_experiments.families.shared.comparator_registry import (
    COT_1,
    MAD_3A_R1,
    MAD_3A_R2,
    MAD_FIXED_R1,
    MAD_FIXED_R2,
    MAD_FIXED_R3,
    MV_3,
    MV_4,
    MV_5,
    MV_6,
    MV_7,
    SC_5,
    SC_6,
    STANDARD_COMPARATOR_METHODS,
    STANDARD_NO_COMM_BASELINES,
    STANDARD_VANILLA_MAD_BASELINES,
    VANILLA_MAD_R1_FINAL_VOTE,
    is_standard_comparator_method,
)

__all__ = [
    "COT_1",
    "SC_5",
    "SC_6",
    "MV_3",
    "MV_4",
    "MV_5",
    "MV_6",
    "MV_7",
    "MAD_3A_R1",
    "MAD_3A_R2",
    "MAD_FIXED_R1",
    "MAD_FIXED_R2",
    "MAD_FIXED_R3",
    "VANILLA_MAD_R1_FINAL_VOTE",
    "STANDARD_NO_COMM_BASELINES",
    "STANDARD_VANILLA_MAD_BASELINES",
    "STANDARD_COMPARATOR_METHODS",
    "is_standard_comparator_method",
]
