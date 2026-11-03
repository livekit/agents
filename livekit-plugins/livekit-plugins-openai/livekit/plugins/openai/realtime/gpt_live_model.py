"""Direct-provider OpenAI GPT-Live API.

The protocol engine lives in :mod:`livekit.agents.llm._realtime.gpt_live` so hosted
and direct-provider clients share one implementation.
"""

from typing import Any

from livekit.agents import llm
from livekit.agents.llm._realtime import gpt_live as _impl
from livekit.agents.llm._realtime.gpt_live import (
    DEFAULT_BACKEND_MODEL as DEFAULT_BACKEND_MODEL,
    DEFAULT_MODEL as DEFAULT_MODEL,
    DEFAULT_VOICE as DEFAULT_VOICE,
    NUM_CHANNELS as NUM_CHANNELS,
    OPENAI_BASE_URL as OPENAI_BASE_URL,
    SAMPLE_RATE as SAMPLE_RATE,
    GPTLiveDelegation as GPTLiveDelegation,
    GPTLiveSession as GPTLiveSession,
    GPTLiveVoices as GPTLiveVoices,
    ResponsesDelegationOptions as ResponsesDelegationOptions,
    Role as Role,
    lk_oai_debug as lk_oai_debug,
)

from ..tools import OpenAITool


class GPTLiveModel(_impl.GPTLiveModel):
    """OpenAI GPT-Live full-duplex voice model, ready to pass to ``AgentSession(llm=)``."""

    def _provider_tool_config(self, tool: llm.ProviderTool) -> dict[str, Any] | None:
        return tool.to_dict() if isinstance(tool, OpenAITool) else None
