"""Compatibility exports for OpenAI Realtime conversion helpers, read when used: an agents build
whose OpenAI realtime client runs in its core has none."""

from typing import Any


def __getattr__(name: str) -> Any:
    from livekit.agents.llm._realtime import openai_utils as _impl

    return getattr(_impl, name)
