"""项目级标准比较器与 family-local 方法命名注册表。"""

from __future__ import annotations

COT_1 = "cot_1"
SC_3 = "sc_3"
SC_5 = "sc_5"
SC_6 = "sc_6"
SC_7 = "sc_7"

MV_3 = "mv_3"
MV_4 = "mv_4"
MV_5 = "mv_5"
MV_6 = "mv_6"
MV_7 = "mv_7"

MAD_3A_R1 = "mad_3a_r1"
MAD_3A_R2 = "mad_3a_r2"
MAD_5A_R1 = "mad_5a_r1"
MAD_FIXED_R1 = "mad_fixed_r1"
MAD_FIXED_R2 = "mad_fixed_r2"
MAD_FIXED_R3 = "mad_fixed_r3"
VANILLA_MAD_R1_FINAL_VOTE = "vanilla_mad_r1_final_vote"

DMAD_SINGLE_COT = "dmad_single_cot"
DMAD_SINGLE_SBP = "dmad_single_sbp"
DMAD_SINGLE_POT_L2M = "dmad_single_pot_l2m"
DMAD_SINGLE_COT_SC = "dmad_single_cot_sc"
DMAD_SINGLE_SBP_SC = "dmad_single_sbp_sc"
DMAD_SINGLE_POT_L2M_SC = "dmad_single_pot_l2m_sc"
DMAD_SELF_REFINE = "dmad_self_refine"
DMAD_SELF_CONTRAST = "dmad_self_contrast"
DMAD_MRP = "dmad_mrp"

STANDARD_NO_COMM_BASELINES: frozenset[str] = frozenset(
    {
        COT_1,
        SC_3,
        SC_5,
        SC_6,
        SC_7,
        MV_3,
        MV_4,
        MV_5,
        MV_6,
        MV_7,
    }
)

STANDARD_VANILLA_MAD_BASELINES: frozenset[str] = frozenset(
    {
        MAD_3A_R1,
        MAD_3A_R2,
        MAD_5A_R1,
        MAD_FIXED_R1,
        MAD_FIXED_R2,
        MAD_FIXED_R3,
        VANILLA_MAD_R1_FINAL_VOTE,
    }
)

STANDARD_COMPARATOR_METHODS: frozenset[str] = STANDARD_NO_COMM_BASELINES | STANDARD_VANILLA_MAD_BASELINES

DMAD_FAMILY_LOCAL_METHODS: frozenset[str] = frozenset(
    {
        DMAD_SINGLE_COT,
        DMAD_SINGLE_SBP,
        DMAD_SINGLE_POT_L2M,
        DMAD_SINGLE_COT_SC,
        DMAD_SINGLE_SBP_SC,
        DMAD_SINGLE_POT_L2M_SC,
        DMAD_SELF_REFINE,
        DMAD_SELF_CONTRAST,
        DMAD_MRP,
    }
)

PROTECTED_STANDARD_BARE_TOKENS: frozenset[str] = frozenset({"cot", "sc", "mv", "mad", "vanilla_mad"})

STANDARD_COMPARATOR_ALIASES: dict[str, str] = {
    "chain_of_thought": COT_1,
    "majority_vote_3": MV_3,
    "majority_vote_4": MV_4,
    "majority_vote_5": MV_5,
    "majority_vote_6": MV_6,
    "majority_vote_7": MV_7,
    "standard_mad_r1": MAD_3A_R1,
    "standard_mad_r2": MAD_3A_R2,
}


def canonical_standard_method_name(method_name: str) -> str | None:
    """把标准比较器别名解析为唯一 canonical method_name。"""
    normalized = str(method_name or "").strip().lower()
    if normalized in STANDARD_COMPARATOR_METHODS:
        return normalized
    return STANDARD_COMPARATOR_ALIASES.get(normalized)


def is_standard_comparator_method(method_name: str) -> bool:
    """判断某个 method_name 是否属于全局标准比较器。"""
    return canonical_standard_method_name(method_name) is not None


def is_protected_standard_bare_name(method_name: str) -> bool:
    """判断某个名字是否落在受保护的标准比较器裸命名空间。"""
    return str(method_name or "").strip().lower() in PROTECTED_STANDARD_BARE_TOKENS


def is_family_local_method_name(method_name: str) -> bool:
    """判断某个名字是否显式属于 family-local 方法命名空间。"""
    normalized = str(method_name or "").strip().lower()
    return normalized in DMAD_FAMILY_LOCAL_METHODS or normalized.startswith("dmad_")
