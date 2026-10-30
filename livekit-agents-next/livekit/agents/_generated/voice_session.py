# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/session.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..types import NOT_GIVEN

if TYPE_CHECKING:
    from livekit.agents import VAD
    from livekit.agents._core import (
        AgentNodes,
        ConsoleIo,
        DuplexRealtimeModel,
        HostRealtimeModel,
        InterruptionDetector,
        OpenAIRealtimeModel,
        RealtimeFallbackModel,
        VoiceAgent,
        _core_VoiceStore,
    )
    from livekit.agents._core_view import TurnDetector
    from livekit.agents._generated.expressive import ExpressiveOptions
    from livekit.agents.job import JobContext
    from livekit.agents.llm import LLM
    from livekit.agents.stt import STT
    from livekit.agents.tts import TTS
    from livekit.agents.voice.agent_session import VoiceSession, _AgentHooks, _DynamicInstructions
    from livekit.agents.voice.io import AudioInput, AudioOutput, TextOutput, VideoInput
    from livekit.agents.voice.remote_session import RemoteSessionAccess

    from ..types import NotGivenOr
    from .agent_run import FunctionToolInfo
    from .chat_items import ChatItem
    from .keyterms import STTContextOptions
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
    """The agent as its tools are hosted: its definition (retry budgets) and its callbacks."""
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
    expressive: NotGivenOr[bool | ExpressiveOptions | None] = NOT_GIVEN
    """The agent's `expressive`; `NOT_GIVEN` defers to the session's."""
    llm_node: AgentNodes | None = None
    tts_node: AgentNodes | None = None
    stt_node: AgentNodes | None = None
    transcription_node: AgentNodes | None = None
    realtime_audio_output_node: AgentNodes | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class VoiceSessionArgs:
    """What `_core.VoiceSession` is assembled from: the session's legs, hooks, nodes and options as
    `VoiceSession(...)` and `start()` hand them over.
    """

    instructions: str | None = None
    dynamic_instructions: _DynamicInstructions | None = None
    agent: str = ""
    agent_host: object | None = None
    chat_ctx: list[ChatItem] | None = None
    storage: _core_VoiceStore | None = None
    stt: STT | None = None
    vad: VAD | None = None
    llm: LLM | None = None
    realtime: DuplexRealtimeModel | HostRealtimeModel | OpenAIRealtimeModel | RealtimeFallbackModel | None = None
    tts: TTS | None = None
    on_event: VoiceSession | None = None
    """Receives each session event, on the calling loop."""
    hooks: _AgentHooks | None = None
    agent_turn_handling: TurnHandlingOptions | None = None
    """The first agent's own `turn_handling`, `min_consecutive_speech_delay` in seconds and
    `use_tts_aligned_transcript`: the agent layer of the stack.
    """
    agent_min_consecutive_speech_delay: float | None = None
    agent_use_tts_aligned_transcript: bool | None = None
    turn_detector: TurnDetector | None = None
    """The turn detector over the user's audio (`turn_detection`)."""
    interruption_detector: InterruptionDetector | None = None
    """The adaptive-interruption detector: an object means the host's gate said yes."""
    room_io_config: dict[str, object]
    """`RoomOptions._to_core()`, resolved per leg."""
    tools: list[FunctionToolInfo] | None = None
    """Every tool of the first agent and the session, as `_core_setup_tools` reported them."""
    tools_host: object | None = None
    voice_agent: VoiceAgent | None = None
    """The first agent as its tools are hosted: its definition and its `AgentCallbacks`."""
    transcription_node: AgentNodes | None = None
    realtime_audio_output_node: AgentNodes | None = None
    llm_node: AgentNodes | None = None
    tts_node: AgentNodes | None = None
    stt_node: AgentNodes | None = None
    host_audio_input: AudioInput | None = None
    """The host's own io legs, replacing the room's."""
    host_audio_output: AudioOutput | None = None
    host_video_input: VideoInput | None = None
    video_sampler: tuple[float, float] | None = None
    """`VoiceActivityVideoSampler(speaking_fps, silent_fps)`'s two rates; `None`: every frame."""
    host_video_sampler: bool = False
    """A custom `video_sampler` callable: the host pumps and gates; the session reads no camera."""
    host_transcription_output: TextOutput | None = None
    using_default_vad: bool = False
    hook_bound: float | None = None
    """The end-of-turn hook's bound, in seconds; `None` is no hook and no wait."""
    max_unrecoverable_errors: int = 3
    stt_context: STTContextOptions | None = None
    keyterm_llm: LLM | None = None
    keyterm_tools: object | None = None
    agent_expressive: NotGivenOr[bool | ExpressiveOptions | None] = NOT_GIVEN
    """The first agent's own `expressive`; `NOT_GIVEN` defers to the session's."""
    session_id: str | None = None
    """Prefixes the ids the session mints for chat items; `None` mints a random one."""
    remote_session: RemoteSessionAccess | None = None
    """Serve the remote session protocol (`lk agent simulate`, the dashboard, `lk agent
    console`) on the room's byte streams, or on `console`.
    """
    console: ConsoleIo | None = None
    """`lk agent console`'s socket: the session's audio input and output, and the remote
    session's transport.
    """
    job: JobContext | None = None
    """The job whose room this is: `start()` connects it beside the opening."""
