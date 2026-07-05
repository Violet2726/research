"""CRED-CVS 共享证书结果类型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CertificateValidation:
    valid: bool
    certificate_type: str
    normalized_answer: str
    challenger_pass: bool
    leader_pass: bool
    failure_reason: str
    checker_runtime_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
