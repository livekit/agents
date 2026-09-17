"""Compatibility exports for GPT-Live wire types."""

from typing import Any

from livekit.agents.llm._realtime import gpt_live_types as _impl
from livekit.agents.llm._realtime.gpt_live_types import *  # noqa: F403


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)
