"""Direct-provider OpenAI Realtime API.

The protocol engine lives in :mod:`livekit.agents.llm._realtime.openai` so hosted
and direct-provider clients share one implementation.
"""

from livekit.agents.llm._realtime import openai as _impl
from livekit.agents.llm._realtime.openai import (
    DEFAULT_VOICE as DEFAULT_VOICE,
    NUM_CHANNELS as NUM_CHANNELS,
    OPENAI_BASE_URL as OPENAI_BASE_URL,
    SAMPLE_RATE as SAMPLE_RATE,
    RealtimeSession as RealtimeSession,
    _DiscardedGeneration as _DiscardedGeneration,
    _is_fatal_error as _is_fatal_error,
    _MessageGeneration as _MessageGeneration,
    lk_oai_debug as lk_oai_debug,
    process_base_url as process_base_url,
)


class RealtimeModel(_impl.RealtimeModel):
    """OpenAI Realtime API model, ready to pass to ``AgentSession(llm=)``."""
