"""baseline_compare 使用的固定方法清单。"""

from __future__ import annotations

from research_experiments.family_runtime.comparators import (
    COT_1,
    MAD_3A_R1,
    MAD_3A_R2,
    MAD_5A_R1,
    SC_3,
    SC_5,
)

CONTROL_METHOD_ORDER = [COT_1, SC_3, SC_5]
MAD_METHOD_ORDER = [MAD_3A_R1, MAD_3A_R2, MAD_5A_R1]
METHOD_ORDER = [*CONTROL_METHOD_ORDER, *MAD_METHOD_ORDER]
