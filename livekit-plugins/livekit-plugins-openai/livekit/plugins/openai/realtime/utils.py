"""Compatibility exports for OpenAI Realtime conversion helpers."""

from typing import Any

from livekit.agents.llm._realtime import openai_utils as _impl
from livekit.agents.llm._realtime.openai_utils import *  # noqa: F403


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)
