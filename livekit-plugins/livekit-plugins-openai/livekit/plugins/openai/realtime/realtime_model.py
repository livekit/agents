"""Direct-provider OpenAI Realtime API.

The protocol engine lives in :mod:`livekit.agents.llm._realtime.openai` so hosted
and direct-provider clients share one implementation.
"""

from typing import Any

from livekit.agents.llm._realtime import openai as _impl
from livekit.agents.llm._realtime.openai import *  # noqa: F403
from livekit.agents.llm._realtime.openai import (
    _DiscardedGeneration as _DiscardedGeneration,
    _is_fatal_error as _is_fatal_error,
    _MessageGeneration as _MessageGeneration,
)


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)
