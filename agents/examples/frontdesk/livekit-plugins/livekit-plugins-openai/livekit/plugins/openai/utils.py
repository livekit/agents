from __future__ import annotations

import os
from collections.abc import Awaitable
from typing import Callable, Union

from livekit.agents.inference.llm import to_fnc_ctx

AsyncAzureADTokenProvider = Callable[[], Union[str, Awaitable[str]]]


def get_base_url(base_url: str | None) -> str:
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return base_url


__all__ = ["get_base_url", "to_fnc_ctx", "AsyncAzureADTokenProvider"]
