"""Direct-provider OpenAI Realtime API.

The protocol engine lives in :mod:`livekit.agents.llm._realtime` so hosted
and direct-provider clients share one implementation.
"""

from typing import Any

from livekit.agents.llm import _realtime as _impl
from livekit.agents.llm._realtime import *  # noqa: F403


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)
