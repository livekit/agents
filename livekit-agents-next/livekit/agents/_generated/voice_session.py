# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/session.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..types import NOT_GIVEN

if TYPE_CHECKING:
    from livekit.agents import VAD
    from livekit.agents._core import AgentNodes, VoiceAgent
    from livekit.agents.llm import LLM
    from livekit.agents.stt import STT
    from livekit.agents.tts import TTS
    from livekit.agents.voice.agent_session import _DynamicInstructions

    from .._core import ChatItem
    from ..types import NotGivenOr
    from .turn import TurnHandlingOptions


@dataclass(frozen=True, kw_only=True, slots=True)
class UpdateAgentArgs:
    """The agent a host `Agent` describes (`AgentActivity._frame_kwargs`), as `update_agent` and
    `run_voice_task` install it. Each provider is three-state: absent keeps the leg, `None` removes
    it, an object replaces it. `chat_ctx` replaces when present. The tools are one toolset,
    replaced together because the runner is per-agent. `turn_handling` is the agent layer.
    `llm_node`, `tts_node` and `stt_node` are the agent's overrides, each over the leg given beside
    it: a leg kept from the previous agent keeps that agent's node. `transcription_node` and
    `realtime_audio_output_node` replace the previous agent's.
    """

    instructions: str
    dynamic_instructions: _DynamicInstructions | None = None
    agent: str
    """The new agent's id."""
    agent_host: object | None = None
    """The new agent's `Agent`, handed back by the events that name it."""
    tools: list[dict[str, object]] | None = None
    tools_host: object | None = None
    voice_agent: VoiceAgent
    """The agent as its tools are hosted: its definition (retry budgets) and its `AgentHost`."""
    chat_ctx: list[ChatItem] | None = None
    llm: NotGivenOr[LLM | None] = NOT_GIVEN
    tts: NotGivenOr[TTS | None] = NOT_GIVEN
    stt: NotGivenOr[STT | None] = NOT_GIVEN
    vad: NotGivenOr[VAD | None] = NOT_GIVEN
    turn_handling: TurnHandlingOptions | None = None
    min_consecutive_speech_delay: float | None = None
    """The agent's `min_consecutive_speech_delay` in seconds; `None` defers to the session's."""
    use_tts_aligned_transcript: bool | None = None
    """The agent's `use_tts_aligned_transcript`; `None` defers to the session's."""
    expressive: NotGivenOr[bool | dict[str, object] | None] = NOT_GIVEN
    """The agent's `expressive`; `NOT_GIVEN` defers to the session's."""
    llm_node: AgentNodes | None = None
    tts_node: AgentNodes | None = None
    stt_node: AgentNodes | None = None
    transcription_node: AgentNodes | None = None
    realtime_audio_output_node: AgentNodes | None = None
