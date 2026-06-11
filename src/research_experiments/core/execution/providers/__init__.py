"""共享 provider 适配层导出入口。"""

from __future__ import annotations

import httpx

from research_experiments.core.execution.providers.client import (
    OpenAICompatibleProvider,
    execute_completion_request,
)
from research_experiments.core.execution.providers.normalization import (
    ProviderRequestError,
    ProviderResponse,
)
from research_experiments.core.execution.providers.normalization import (
    extract_message_channels as _extract_message_channels,
)
from research_experiments.core.execution.providers.payloads import build_payload

__all__ = [
    "ProviderResponse",
    "ProviderRequestError",
    "_extract_message_channels",
    "httpx",
    "OpenAICompatibleProvider",
    "build_payload",
    "execute_completion_request",
]
