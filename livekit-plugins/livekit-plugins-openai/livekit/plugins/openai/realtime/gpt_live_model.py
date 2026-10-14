"""Direct-provider OpenAI GPT-Live API.

The protocol engine lives in :mod:`livekit.agents.llm._realtime.gpt_live` so hosted
and direct-provider clients share one implementation.
"""

from typing import Any

from livekit.agents.llm._realtime import gpt_live as _impl
from livekit.agents.llm._realtime.gpt_live import *  # noqa: F403


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)
