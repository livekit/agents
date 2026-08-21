from __future__ import annotations

import asyncio
import contextvars
import heapq
import json
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from opentelemetry import context as otel_context, trace

from livekit import rtc
from livekit.agents.llm.realtime import (
    MessageGeneration,
    _UserMessageSyncStatus,
)
from livekit.agents.metrics.base import Metadata

from .. import inference, llm, stt, tts, utils, vad
from ..llm.chat_context import Instructions
from ..llm.realtime_fallback_adapter import _FallbackRealtimeSession
from ..llm.tool_context import (
    StopResponse,
    ToolFlag,
    get_fnc_tool_names,
)
from ..log import logger
from ..metrics import (
    EOUMetrics,
    LLMMetrics,
    RealtimeModelMetrics,
    STTMetrics,
    TTSMetrics,
    VADMetrics,
)
from ..telemetry import otel_metrics, trace_types, tracer, utils as trace_utils
from ..tokenize.basic import split_words
from ..types import NOT_GIVEN, FlushSentinel, NotGivenOr
from ..utils.misc import is_given
from ._utils import _set_participant_attributes
from .agent import (
    Agent,
    ModelSettings,
    _get_activity_task_info,
    _set_activity_task_info,
)
from .audio_recognition import (
    AudioRecognition,
    RecognitionHooks,
    _EndOfTurnInfo,
    _PreemptiveGenerationInfo,
    _STTPipeline,
)
from .endpointing import create_endpointing
from .events import (
    AgentFalseInterruptionEvent,
    AgentState,
    AgentStateChangedEvent,
    EotPredictionEvent,
    ErrorEvent,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    SessionUsageUpdatedEvent,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
    UserTranscriptionTimeoutEvent,
    UserTurnExceededEvent,
    _AgentBackchannelOpportunityEvent,
)
from .generation import (
    ToolExecutionOutput,
    _AudioOutput,
    _ForwardOutput,
    _inject_running_tool_calls,
    _interrupted_tool_output,
    _strip_assistant_markup,
    _strip_running_tool_calls,
    _TextOutput,
    _time_to_first_sentence,
    _TTSGenerationData,
    forward_generation,
    perform_audio_forwarding,
    perform_llm_inference,
    perform_text_forwarding,
    perform_tool_executions,
    perform_tts_inference,
    remove_expressive_instructions,
    remove_instructions,
    update_expressive_instructions,
    update_instructions,
)
from .speech_handle import DEFAULT_INPUT_DETAILS, InputDetails, SpeechHandle
from .tool_executor import _resolve_async_tool_options, _RunningTasks, _ToolExecutor
from .turn import (
    EndpointingOptions,
    PreemptiveGenerationOptions,
    RealtimeInputMode,
    TurnDetectionMode,
    _resolve_endpointing,
    _StreamingTurnDetector,
    _StreamingTurnDetectorStream,
)

if TYPE_CHECKING:
    from ..llm import mcp
    from .agent_session import AgentSession, ExpressiveOptions


_AgentActivityContextVar = contextvars.ContextVar["AgentActivity"]("agents_activity")
_SpeechHandleContextVar = contextvars.ContextVar["SpeechHandle"]("agents_speech_handle")
_IdleHoldContextVar = contextvars.ContextVar[bool]("agents_idle_hold", default=False)


async def _aligned_transcript_or_text(
    timed_texts: AsyncIterable[str], text: AsyncIterable[str]
) -> AsyncGenerator[str, None]:
    """Forward the TTS-aligned transcript, falling back to the spoken text.

    ``aligned_transcript`` says the TTS was asked for word timings, not that this
    model/language pair returns any. When none arrive the aligned stream is simply
    empty, and forwarding it alone would leave the turn with no transcript.
    """
    aligned = False
    async for timed_text in timed_texts:
        aligned = True
        yield timed_text

    if not aligned:
        logger.warning(
            "no aligned transcript was returned from tts, "
            "forwarding the generated text without word timings"
        )
        async for chunk in text:
            yield chunk


def _transcripts_equivalent(first: str, second: str | None) -> bool:
    if first == second:
        return True
    if second is None:
        return False

    first_words = [
        word.casefold()
        for word, _, _ in split_words(first, ignore_punctuation=True, split_character=True)
    ]
    second_words = [
        word.casefold()
        for word, _, _ in split_words(second, ignore_punctuation=True, split_character=True)
    ]
    return bool(first_words) and first_words == second_words


class ActivityClosedError(Exception):
    """Raised by ``wait_for_idle`` when the target activity/session has closed."""


@dataclass
class _OnEnterData:
    session: AgentSession
    agent: Agent


_OnEnterContextVar = contextvars.ContextVar["_OnEnterData"]("agents_activity_on_enter")


@dataclass
class _ReusableResources:
    stt_pipeline: _STTPipeline | None = None
    rt_session: llm.RealtimeSession | None = None
    turn_detector_stream: _StreamingTurnDetectorStream | None = None

    async def cleanup(self) -> None:
        tasks = []
        if self.stt_pipeline is not None:
            tasks.append(self.stt_pipeline.aclose())
            self.stt_pipeline = None
        if self.rt_session is not None:
            tasks.append(self.rt_session.aclose())
            self.rt_session = None
        if self.turn_detector_stream is not None:
            tasks.append(self.turn_detector_stream.aclose())
            self.turn_detector_stream = None

        if tasks:
            outputs = await asyncio.gather(*tasks, return_exceptions=True)
            for output in outputs:
                if isinstance(output, Exception):
                    logger.error("error cleaning up reusable resources", exc_info=output)


@dataclass
class _PreemptiveGeneration:
    speech_handle: SpeechHandle
    user_message: llm.ChatMessage
    info: _PreemptiveGenerationInfo
    chat_ctx: llm.ChatContext
    tools: list[llm.Tool | llm.Toolset]
    tool_choice: llm.ToolChoice | None
    created_at: float


@dataclass
class _PausedSpeechInfo:
    handle: SpeechHandle
    agent_state: AgentState
    timeout: float


@dataclass(frozen=True)
class _TurnPolicyDiagnostic:
    level: Literal["info", "warning"]
    message: str


@dataclass(frozen=True)
class _ResolvedRealtimeTurnPolicy:
    input_mode: RealtimeInputMode
    configured_turn_detection: TurnDetectionMode | None
    turn_detection: TurnDetectionMode | None
    turn_detection_explicit: bool
    server_turn_detection_enabled: bool
    input_owner: Literal["framework", "provider"]
    interruption_owner: Literal["disabled", "framework", "provider"]
    finalize_empty_transcript_on_timeout: bool
    diagnostics: tuple[_TurnPolicyDiagnostic, ...] = ()


def _resolve_realtime_turn_policy(
    *,
    input_mode: RealtimeInputMode,
    llm_model: llm.LLM | llm.RealtimeModel | None,
    configured_turn_detection: TurnDetectionMode | None,
    turn_detection_explicit: bool,
    vad_model: vad.VAD | None,
    stt_model: stt.STT | None,
    using_default_vad: bool,
    interruption_detection_configured: bool,
    allow_interruptions: bool,
    preserve_existing_stt_boundary: bool = False,
    active_server_turn_detection_enabled: bool | None = None,
) -> _ResolvedRealtimeTurnPolicy:
    """Resolve every owner of a user turn without mutating live session state."""

    diagnostics: list[_TurnPolicyDiagnostic] = []
    realtime_model = llm_model if isinstance(llm_model, llm.RealtimeModel) else None
    caps = realtime_model.capabilities if realtime_model is not None else None

    if input_mode == "text" and configured_turn_detection == "realtime_llm":
        raise ValueError(
            "turn_detection='realtime_llm' is incompatible with realtime text input; "
            "use 'stt', 'manual', or None for automatic STT turn detection"
        )

    desired_server_detection = False
    if caps is not None:
        if input_mode == "text":
            desired_server_detection = caps.turn_detection and not caps.can_disable_turn_detection
        elif not caps.turn_detection or not caps.can_disable_turn_detection:
            desired_server_detection = caps.turn_detection
        elif turn_detection_explicit and configured_turn_detection == "realtime_llm":
            desired_server_detection = True
        elif turn_detection_explicit and configured_turn_detection == "manual":
            desired_server_detection = False
        elif vad_model is not None and (
            (
                turn_detection_explicit
                and (
                    isinstance(configured_turn_detection, _StreamingTurnDetector)
                    or configured_turn_detection == "vad"
                )
            )
            or interruption_detection_configured
        ):
            desired_server_detection = False
        else:
            desired_server_detection = True

    server_detection = (
        desired_server_detection
        if active_server_turn_detection_enabled is None
        else active_server_turn_detection_enabled
    )
    if (
        active_server_turn_detection_enabled is not None
        and turn_detection_explicit
        and desired_server_detection != active_server_turn_detection_enabled
    ):
        diagnostics.append(
            _TurnPolicyDiagnostic(
                "warning",
                "changing turn_detection at runtime does not update a realtime model's "
                "server-side turn detection (resolved at session start); it stays "
                f"{'enabled' if active_server_turn_detection_enabled else 'disabled'} "
                "for this session.",
            )
        )

    resolved_detection = configured_turn_detection
    unavailable_detector = False
    if isinstance(resolved_detection, _StreamingTurnDetector):
        if vad_model is None:
            diagnostics.append(
                _TurnPolicyDiagnostic(
                    "warning",
                    "TurnDetector requires a VAD model. Pass vad=inference.VAD() to "
                    "AgentSession/Agent or turn_detection=None to disable the default TurnDetector",
                )
            )
            resolved_detection = None
            unavailable_detector = True
        elif server_detection:
            if turn_detection_explicit:
                diagnostics.append(
                    _TurnPolicyDiagnostic(
                        "warning",
                        "turn_detection is a TurnDetector, but the LLM is a RealtimeModel "
                        "with server-side turn detection enabled, ignoring the turn_detection setting",
                    )
                )
            resolved_detection = None
    elif resolved_detection is None or isinstance(resolved_detection, str):
        mode = resolved_detection if isinstance(resolved_detection, str) else None
        if mode == "vad" and vad_model is None:
            diagnostics.append(
                _TurnPolicyDiagnostic(
                    "warning",
                    "turn_detection is set to 'vad', but no VAD model is provided. "
                    "Pass a VAD instance, e.g. Agent(vad=silero.VAD.load())",
                )
            )
            mode = None
            unavailable_detector = True
        if mode == "stt" and stt_model is None:
            diagnostics.append(
                _TurnPolicyDiagnostic(
                    "warning",
                    "turn_detection is set to 'stt', but no STT model is provided. "
                    "Pass an STT instance, e.g. Agent(stt=deepgram.STT())",
                )
            )
            mode = None
            unavailable_detector = True

        if realtime_model is not None:
            if mode == "realtime_llm" and not server_detection:
                diagnostics.append(
                    _TurnPolicyDiagnostic(
                        "warning",
                        "turn_detection is set to 'realtime_llm', but the LLM is not a "
                        "RealtimeModel or the server-side turn detection is not supported/enabled, "
                        "ignoring the turn_detection setting",
                    )
                )
                mode = None
            if mode == "stt" and input_mode != "text":
                diagnostics.append(
                    _TurnPolicyDiagnostic(
                        "warning",
                        "turn_detection is set to 'stt', but the LLM is a RealtimeModel, "
                        "ignoring the turn_detection setting",
                    )
                )
                mode = None
            elif mode and mode != "realtime_llm" and server_detection:
                diagnostics.append(
                    _TurnPolicyDiagnostic(
                        "warning",
                        f"turn_detection is set to '{mode}', but the LLM is a RealtimeModel "
                        "and server-side turn detection enabled, ignoring the turn_detection setting",
                    )
                )
                mode = None
            if (
                not server_detection
                and vad_model is not None
                and not using_default_vad
                and mode is None
            ):
                mode = "vad"
        elif mode == "realtime_llm":
            diagnostics.append(
                _TurnPolicyDiagnostic(
                    "warning",
                    "turn_detection is set to 'realtime_llm', but the LLM is not a RealtimeModel",
                )
            )
            mode = None
        resolved_detection = mode
    # Legacy turn detectors expose predict_end_of_turn() instead of stream(). They
    # have no VAD prerequisite, so a valid configured instance remains authoritative.

    if (
        input_mode == "text"
        and resolved_detection is None
        and vad_model is None
        and stt_model is not None
        and (stt_model.capabilities.streaming or preserve_existing_stt_boundary)
        and (not turn_detection_explicit or unavailable_detector)
    ):
        diagnostics.append(
            _TurnPolicyDiagnostic(
                "info",
                "using STT end-of-speech events for realtime text input because no usable "
                "VAD or turn detector is configured",
            )
        )
        resolved_detection = "stt"

    if (
        vad_model is None
        and stt_model is not None
        and not stt_model.capabilities.streaming
        and isinstance(llm_model, llm.LLM)
        and allow_interruptions
        and resolved_detection is None
    ):
        diagnostics.append(
            _TurnPolicyDiagnostic(
                "warning",
                "VAD is not set. Enabling VAD is recommended when using LLM and "
                "non-streaming STT for more responsive interruption handling.",
            )
        )

    if server_detection and not allow_interruptions:
        raise ValueError(
            "the RealtimeModel uses a server-side turn detection, allow_interruptions cannot "
            "be False, disable turn_detection in the RealtimeModel and use VAD on the "
            "AgentSession instead"
        )

    input_owner: Literal["framework", "provider"] = "provider" if server_detection else "framework"
    interruption_owner: Literal["disabled", "framework", "provider"] = (
        "disabled" if not allow_interruptions else input_owner
    )
    return _ResolvedRealtimeTurnPolicy(
        input_mode=input_mode,
        configured_turn_detection=configured_turn_detection,
        turn_detection=resolved_detection,
        turn_detection_explicit=turn_detection_explicit,
        server_turn_detection_enabled=server_detection,
        input_owner=input_owner,
        interruption_owner=interruption_owner,
        finalize_empty_transcript_on_timeout=(
            realtime_model is not None and not server_detection and resolved_detection != "manual"
        ),
        diagnostics=tuple(diagnostics),
    )


_RealtimeTurnState = Literal[
    "open",
    "sealed",
    "input_submitted",
    "generation_pending",
    "generation_created",
    "settled",
]


@dataclass
class _RealtimeTurnTransaction:
    turn_id: int
    policy: _ResolvedRealtimeTurnPolicy
    state: _RealtimeTurnState = "open"
    disposition: Literal["active", "abandoned"] = "active"
    frames: list[rtc.AudioFrame] = field(default_factory=list)
    input_observed: bool = False
    provider_activity_started: bool = False
    ready_fut: asyncio.Future[_RealtimeTurnTransaction] | None = None
    speech_handle: SpeechHandle | None = None
    generation_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None

    @property
    def token(self) -> _RealtimeTurnTransaction:
        return self

    @property
    def activity_started(self) -> bool:
        return self.provider_activity_started

    @activity_started.setter
    def activity_started(self, value: bool) -> None:
        self.provider_activity_started = value

    @property
    def turn_complete(self) -> bool:
        return self.state != "open"

    @turn_complete.setter
    def turn_complete(self, value: bool) -> None:
        if value and self.state == "open":
            self.state = "sealed"
        elif not value and self.state == "sealed":
            self.state = "open"

    @property
    def input_submitted(self) -> bool:
        return self.state in (
            "input_submitted",
            "generation_pending",
            "generation_created",
            "settled",
        )


_RealtimeAudioInputOwner = _RealtimeTurnTransaction | asyncio.Future[_RealtimeTurnTransaction]


# NOTE: AgentActivity isn't exposed to the public API
class AgentActivity(RecognitionHooks):
    def __init__(self, agent: Agent, sess: AgentSession) -> None:
        self._agent, self._session = agent, sess
        self._rt_session: llm.RealtimeSession | None = None
        self._realtime_spans: utils.BoundedDict[str, trace.Span] | None = None
        self._audio_recognition: AudioRecognition | None = None
        self._lock = asyncio.Lock()
        self._realtime_chat_ctx_lock = asyncio.Lock()
        self._pending_realtime_user_message_ids: set[str] = set()
        self._provider_transcription_item_ids: set[str] = set()
        self._bounded_close_user_message_ids: set[str] = set()
        self._tool_choice: llm.ToolChoice | None = None

        self._started = False
        self._closed = False
        self._scheduling_paused = True
        self._new_turns_blocked = False

        self._current_speech: SpeechHandle | None = None
        self._active_realtime_generation: llm.GenerationCreatedEvent | None = None
        self._speech_q: list[tuple[int, float, SpeechHandle]] = []
        self._user_silence_event: asyncio.Event = asyncio.Event()
        self._user_silence_event.set()

        # for false interruption handling
        self._paused_speech: _PausedSpeechInfo | None = None
        self._false_interruption_timer: asyncio.TimerHandle | None = None
        # the timeout elapsed while a turn decision was still open; the resume waits on it
        self._false_interruption_pending: bool = False
        self._cancel_speech_pause_task: asyncio.Task[None] | None = None

        self._stt_eos_received: bool = False

        # fired when a speech_task finishes or when a new speech_handle is scheduled
        # this is used to wake up the main task when the scheduling state changes
        self._q_updated = asyncio.Event()

        self._scheduling_atask: asyncio.Task[None] | None = None
        self._user_turn_completed_atask: asyncio.Task[None] | None = None
        self._speech_tasks: list[asyncio.Task[Any]] = []

        self._preemptive_generation: _PreemptiveGeneration | None = None
        self._preemptive_generation_count: int = 0
        self._authorization_allowed = asyncio.Event()
        self._authorization_allowed.set()

        self._drain_blocked_tasks: set[asyncio.Task[Any]] = set()
        self._mcp_tools: list[mcp.MCPToolset] = []

        # activity-scoped executor: cancels cancellable tools / awaits the rest on drain,
        # and delivers replies to this activity's agent
        if is_given(self._agent._async_tool_options):
            activity_options = _resolve_async_tool_options(self._agent._async_tool_options)
        else:
            activity_options = self._session._async_tool_options
        self._tool_executor = _ToolExecutor(
            owning_activity=self, async_tool_options=activity_options
        )

        self._user_turn_exceeded_atask: asyncio.Task[None] | None = None
        self._user_turn_exceeded_locked: bool = False

        self._on_enter_task: asyncio.Task | None = None
        self._on_exit_task: asyncio.Task | None = None

        self._realtime_input_mode = self._resolve_realtime_input_mode()
        configured_turn_detection = (
            self._agent.turn_detection
            if is_given(self._agent.turn_detection)
            else self._session.turn_detection
        )
        turn_detection_explicit = (
            is_given(self._agent.turn_detection) or self._session._turn_detection_explicit
        )
        self._turn_policy = self._resolve_turn_policy(
            configured_turn_detection=configured_turn_detection,
            turn_detection_explicit=turn_detection_explicit,
        )
        self._rt_turn_detection_enabled = self._turn_policy.server_turn_detection_enabled
        self._turn_detection = self._turn_policy.turn_detection
        self._turn_detection_metrics_source: inference.TurnDetector | None = None
        self._validate_realtime_input_mode()
        if (
            isinstance(self.llm, llm.RealtimeModel)
            and not self._rt_turn_detection_enabled
            and self.llm.capabilities.turn_detection
        ):
            logger.info(
                "client-side turn-taking is configured, disabling realtime server-side "
                "turn detection."
            )

        self._next_realtime_turn_id = 0
        self._realtime_turn = self._new_realtime_turn()
        self._deferred_realtime_turns: deque[_RealtimeTurnTransaction] = deque()

        self._interruption_detector: inference.AdaptiveInterruptionDetector | None = (
            self._resolve_interruption_detection()
        )
        self._interruption_detection_enabled: bool = self._interruption_detector is not None
        self._interruption_detected: bool = False

        # this allows taking over audio interruption temporarily until interruption is detected
        self._interruption_by_audio_activity_enabled = (
            self._turn_policy.input_owner == "framework"
            and self._turn_detection not in ("manual", "realtime_llm")
        )
        self._default_interruption_by_audio_activity_enabled = (
            self._interruption_by_audio_activity_enabled
        )

        # speeches that audio playout finished but not done because of tool calls
        self._background_speeches: set[SpeechHandle] = set()

        # placeholder used to hold a RunResult open while waiting for a realtime
        # model to auto-generate a tool reply (auto_tool_reply_generation=True).
        self._pending_auto_tool_reply_fut: asyncio.Future[None] | None = None

    def _resolve_realtime_input_mode(self) -> RealtimeInputMode:
        mode = self._agent._turn_handling.get(
            "realtime_input_mode", self._session.options.realtime_input_mode
        )
        if mode not in ("audio", "text"):
            raise ValueError("turn_handling.realtime_input_mode must be either 'audio' or 'text'")
        return mode

    def _validate_realtime_input_mode(self) -> None:
        if self._realtime_input_mode == "audio":
            return

        if not isinstance(self.llm, llm.RealtimeModel):
            raise ValueError("realtime_input_mode='text' requires a RealtimeModel")
        self._validate_text_mode_stt_path(
            resolved_stt=self.stt,
            resolved_vad=self.vad,
            resolved_turn_detection=self._turn_policy.turn_detection,
            resolved_turn_detection_explicit=self._turn_policy.turn_detection_explicit,
            configured_turn_detection=self._turn_policy.configured_turn_detection,
        )
        if not self.llm.capabilities.mutable_chat_context:
            raise ValueError(
                "realtime_input_mode='text' requires a RealtimeModel with mutable chat context"
            )

        if self._rt_turn_detection_enabled:
            raise ValueError(
                "realtime_input_mode='text' requires server-side turn detection to be disabled"
            )

    def _validate_text_mode_stt_path(
        self,
        *,
        resolved_stt: stt.STT | None,
        resolved_vad: vad.VAD | None,
        resolved_turn_detection: TurnDetectionMode | None,
        resolved_turn_detection_explicit: bool,
        configured_turn_detection: TurnDetectionMode | None,
    ) -> None:
        if resolved_stt is None:
            raise ValueError("realtime_input_mode='text' requires an external STT")
        if resolved_stt.capabilities.streaming or resolved_vad is not None:
            return
        if type(self._agent).stt_node is Agent.stt_node:
            raise ValueError(
                "realtime_input_mode='text' with a non-streaming STT requires a VAD when "
                "using the default Agent.stt_node; add a VAD, pass a streaming or "
                "stt.StreamAdapter-wrapped STT, or override Agent.stt_node and explicitly "
                "configure turn_detection"
            )
        compatible_explicit_detection = (
            resolved_turn_detection_explicit
            and configured_turn_detection in (None, "manual", "stt")
        )
        if resolved_turn_detection != "stt" and not compatible_explicit_detection:
            raise ValueError(
                "realtime_input_mode='text' with a custom Agent.stt_node, non-streaming STT, "
                "and no VAD requires explicit turn_detection"
            )

    def _resolve_turn_policy(
        self,
        *,
        configured_turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        turn_detection_explicit: NotGivenOr[bool] = NOT_GIVEN,
        vad_model: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        stt_model: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        llm_model: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        preserve_existing_stt_boundary: bool = False,
        active_server_turn_detection_enabled: NotGivenOr[bool] = NOT_GIVEN,
    ) -> _ResolvedRealtimeTurnPolicy:
        if is_given(configured_turn_detection):
            effective_turn_detection = configured_turn_detection
            effective_explicit = (
                turn_detection_explicit
                if is_given(turn_detection_explicit)
                else configured_turn_detection is not None
            )
        elif is_given(self._agent.turn_detection):
            effective_turn_detection = self._agent.turn_detection
            effective_explicit = True
        else:
            effective_turn_detection = self._session.turn_detection
            effective_explicit = self._session._turn_detection_explicit

        effective_vad = vad_model if is_given(vad_model) else self.vad
        effective_stt = stt_model if is_given(stt_model) else self.stt
        effective_llm = llm_model if is_given(llm_model) else self.llm
        policy = _resolve_realtime_turn_policy(
            input_mode=self._realtime_input_mode,
            llm_model=effective_llm,
            configured_turn_detection=effective_turn_detection,
            turn_detection_explicit=effective_explicit,
            vad_model=effective_vad,
            stt_model=effective_stt,
            using_default_vad=self.using_default_vad if not is_given(vad_model) else False,
            interruption_detection_configured=(
                is_given(self._agent.interruption_detection)
                or is_given(self._session.interruption_detection)
            ),
            allow_interruptions=self.allow_interruptions,
            preserve_existing_stt_boundary=preserve_existing_stt_boundary,
            active_server_turn_detection_enabled=(
                active_server_turn_detection_enabled
                if is_given(active_server_turn_detection_enabled)
                else None
            ),
        )
        for diagnostic in policy.diagnostics:
            if diagnostic.level == "warning":
                logger.warning(diagnostic.message)
            else:
                logger.info(diagnostic.message)
        return policy

    def _apply_turn_policy(self, policy: _ResolvedRealtimeTurnPolicy) -> None:
        if self._turn_detection == "manual" or policy.turn_detection == "manual":
            self._cancel_false_interruption_timer()

        self._turn_policy = policy
        self._rt_turn_detection_enabled = policy.server_turn_detection_enabled
        self._turn_detection = policy.turn_detection
        if self._started and not self._new_turns_blocked:
            self._set_turn_detection_metrics_source(
                self._turn_detection
                if isinstance(self._turn_detection, inference.TurnDetector)
                else None
            )
        self._default_interruption_by_audio_activity_enabled = (
            policy.input_owner == "framework"
            and policy.turn_detection not in ("manual", "realtime_llm")
        )
        if (
            self._realtime_turn.state == "open"
            and not self._realtime_turn.frames
            and not self._realtime_turn.input_observed
            and not self._realtime_turn.provider_activity_started
        ):
            self._realtime_turn = self._new_realtime_turn()

    def _set_turn_detection_metrics_source(self, detector: inference.TurnDetector | None) -> None:
        if self._turn_detection_metrics_source is detector:
            return

        if self._turn_detection_metrics_source is not None:
            self._turn_detection_metrics_source.off("metrics_collected", self._on_metrics_collected)
        self._turn_detection_metrics_source = detector
        if detector is not None:
            detector.on("metrics_collected", self._on_metrics_collected)

    def _new_realtime_turn(self) -> _RealtimeTurnTransaction:
        self._next_realtime_turn_id += 1
        return _RealtimeTurnTransaction(
            turn_id=self._next_realtime_turn_id,
            policy=self._turn_policy,
        )

    @property
    def _rt_audio_input_token(self) -> _RealtimeTurnTransaction:
        """Compatibility view of the current framework-owned turn identity."""

        return self._realtime_turn

    @property
    def _rt_audio_input_sealed(self) -> bool:
        return self._realtime_turn.turn_complete

    @property
    def _rt_user_activity_started(self) -> bool:
        return self._realtime_turn.provider_activity_started

    @property
    def _deferred_realtime_audio_inputs(self) -> deque[_RealtimeTurnTransaction]:
        """Compatibility view of FIFO transactions waiting for provider ownership."""

        return self._deferred_realtime_turns

    @property
    def scheduling_paused(self) -> bool:
        return self._scheduling_paused

    @property
    def session(self) -> AgentSession:
        return self._session

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def interruption_enabled(self) -> bool:
        return self._interruption_detection_enabled

    @property
    def mcp_servers(self) -> list[mcp.MCPServer] | None:
        return (
            self._agent.mcp_servers
            if is_given(self._agent.mcp_servers)
            else self._session.mcp_servers
        )

    @property
    def allow_interruptions(self) -> bool:
        return (
            self._agent.allow_interruptions
            if is_given(self._agent.allow_interruptions)
            else self._session.options.interruption["enabled"]
        )

    @property
    def endpointing_opts(self) -> EndpointingOptions:
        overrides: EndpointingOptions = {
            **self.session._opts.endpointing_overrides,
            **(self._agent._turn_handling.get("endpointing") or EndpointingOptions()),  # type: ignore[typeddict-item]
        }
        return _resolve_endpointing(overrides, turn_detection=self._turn_detection)

    @property
    def preemptive_generation_opts(self) -> PreemptiveGenerationOptions:
        # session is always fully resolved; agent-level keys override it
        agent_preemptive = self._agent._turn_handling.get("preemptive_generation", {})
        session_preemptive = self._session.options.preemptive_generation
        return PreemptiveGenerationOptions(**{**session_preemptive, **agent_preemptive})

    @property
    def realtime_input_mode(self) -> RealtimeInputMode:
        return self._realtime_input_mode

    @property
    def min_endpointing_delay(self) -> float:
        # this resolves to the fixed value from either agent or session instead of the dynamic one
        return self.endpointing_opts["min_delay"]

    @property
    def max_endpointing_delay(self) -> float:
        # this resolves to the fixed value from either agent or session instead of the dynamic one
        return self.endpointing_opts["max_delay"]

    @property
    def realtime_llm_session(self) -> llm.RealtimeSession | None:
        return self._rt_session

    @property
    def current_speech(self) -> SpeechHandle | None:
        return self._current_speech

    @property
    def tools(
        self,
    ) -> list[llm.Tool | llm.Toolset]:
        from .tool_executor import cancel_task, get_running_tasks, has_cancellable_tool

        tools = self._session.tools + self._agent.tools + self._mcp_tools
        # auto-expose cancel_task / get_running_tasks when any tool opts in via
        # ToolFlag.CANCELLABLE. always-on (not per-turn) so the LLM-visible
        # schema stays stable across turns and the prompt cache stays warm
        if has_cancellable_tool(tools):
            tools = [*tools, cancel_task, get_running_tasks]
        return tools

    @property
    def min_consecutive_speech_delay(self) -> float:
        return (
            self._agent.min_consecutive_speech_delay
            if is_given(self._agent.min_consecutive_speech_delay)
            else self._session.options.min_consecutive_speech_delay
        )

    @property
    def use_tts_aligned_transcript(self) -> bool:
        use_aligned_transcript = (
            self._agent.use_tts_aligned_transcript
            if is_given(self._agent.use_tts_aligned_transcript)
            else self._session.options.use_tts_aligned_transcript
        )

        return use_aligned_transcript is True

    async def update_instructions(self, instructions: str) -> None:
        self._agent._instructions = instructions

        # Record the configuration change
        config_update = llm.AgentConfigUpdate(
            instructions=instructions,
        )
        self._agent._chat_ctx.insert(config_update)
        self._session._chat_ctx.insert(config_update)

        if self._rt_session is not None:
            await self._rt_session.update_instructions(instructions)
        else:
            update_instructions(
                self._agent._chat_ctx, instructions=instructions, add_if_missing=True
            )

    async def update_tools(self, tools: list[llm.Tool | llm.Toolset]) -> None:
        # Compute tool diff before updating
        old_tool_names = set(get_fnc_tool_names(self._agent._tools))
        new_tool_names = set(get_fnc_tool_names(tools))
        tools_added = list(new_tool_names - old_tool_names) or None
        tools_removed = list(old_tool_names - new_tool_names) or None

        tools = list({tool.id: tool for tool in tools}.values())
        self._agent._tools = tools

        # Record the configuration change (skip if no visible diff)
        if tools_added or tools_removed:
            config_update = llm.AgentConfigUpdate(
                tools_added=tools_added,
                tools_removed=tools_removed,
            )
            config_update._tools = llm.ToolContext(tools).flatten()
            self._agent._chat_ctx.insert(config_update)
            self._session._chat_ctx.insert(config_update)

        if self._rt_session is not None:
            await self._rt_session.update_tools(llm.ToolContext(self.tools).flatten())

        if isinstance(self.llm, llm.LLM):
            # for realtime LLM, we assume the server will remove unvalid tool messages
            await self.update_chat_ctx(self._agent._chat_ctx.copy(tools=tools))

    async def update_chat_ctx(
        self, chat_ctx: llm.ChatContext, *, exclude_invalid_function_calls: bool = True
    ) -> None:
        chat_ctx = chat_ctx.copy(tools=self.tools if exclude_invalid_function_calls else NOT_GIVEN)

        if self._rt_session is not None:
            async with self._realtime_chat_ctx_lock:
                self._agent._chat_ctx = chat_ctx
                remove_instructions(chat_ctx)
                await self._rt_session.update_chat_ctx(chat_ctx)
        else:
            self._agent._chat_ctx = chat_ctx
            update_instructions(
                chat_ctx, instructions=self._agent.instructions, add_if_missing=True
            )

    def update_options(
        self,
        *,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        endpointing_opts: NotGivenOr[EndpointingOptions] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        session_turn_detection_explicit: NotGivenOr[bool] = NOT_GIVEN,
        # deprecated
        min_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
        max_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(min_endpointing_delay) or is_given(max_endpointing_delay):
            logger.warning(
                "min_endpointing_delay and max_endpointing_delay are deprecated, use endpointing instead"
            )
            endpointing_opts = EndpointingOptions(
                mode=self.endpointing_opts["mode"],
                min_delay=min_endpointing_delay
                if is_given(min_endpointing_delay)
                else self.endpointing_opts["min_delay"],
                max_delay=max_endpointing_delay
                if is_given(max_endpointing_delay)
                else self.endpointing_opts["max_delay"],
                alpha=self.endpointing_opts["alpha"],
            )

        if utils.is_given(tool_choice):
            self._tool_choice = tool_choice

        if self._rt_session is not None:
            self._rt_session.update_options(tool_choice=self._tool_choice)

        if utils.is_given(turn_detection):
            policy = self._resolve_turn_policy(
                configured_turn_detection=turn_detection,
                turn_detection_explicit=(
                    session_turn_detection_explicit
                    if is_given(session_turn_detection_explicit)
                    else turn_detection is not None
                ),
                active_server_turn_detection_enabled=self._rt_turn_detection_enabled,
            )
            if self._realtime_input_mode == "text":
                self._validate_text_mode_stt_path(
                    resolved_stt=self.stt,
                    resolved_vad=self.vad,
                    resolved_turn_detection=policy.turn_detection,
                    resolved_turn_detection_explicit=policy.turn_detection_explicit,
                    configured_turn_detection=policy.configured_turn_detection,
                )
            if self._audio_recognition is not None:
                self._audio_recognition._check_vad_silence_requirement(
                    detector=(
                        policy.turn_detection
                        if isinstance(policy.turn_detection, _StreamingTurnDetector)
                        else None
                    ),
                    vad=self.vad,
                )
            self._apply_turn_policy(policy)
            turn_detection = policy.turn_detection

        if self._audio_recognition:
            self._audio_recognition._update_options(
                endpointing=create_endpointing(endpointing_opts)
                if is_given(endpointing_opts)
                else NOT_GIVEN,
                turn_detection=turn_detection,
                finalize_empty_transcript_on_timeout=(
                    self._turn_policy.finalize_empty_transcript_on_timeout
                    if is_given(turn_detection)
                    else NOT_GIVEN
                ),
            )

    def _update_models(
        self,
        *,
        new_stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        new_vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        new_llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        new_tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
    ) -> None:
        # A RealtimeModel owns a live session; reject before mutating to stay all-or-nothing.
        if is_given(new_llm) and (
            isinstance(new_llm, llm.RealtimeModel) or isinstance(self.llm, llm.RealtimeModel)
        ):
            raise RuntimeError(
                "cannot swap to or from a RealtimeModel while the agent is running, "
                "use AgentSession.update_agent() instead"
            )

        prospective_stt = new_stt if is_given(new_stt) else self.stt
        prospective_vad = new_vad if is_given(new_vad) else self.vad
        policy: _ResolvedRealtimeTurnPolicy | None = None
        if is_given(new_stt) or is_given(new_vad) or is_given(new_llm):
            policy = self._resolve_turn_policy(
                stt_model=new_stt,
                vad_model=new_vad,
                llm_model=new_llm,
                preserve_existing_stt_boundary=(
                    is_given(new_stt)
                    and prospective_stt is not None
                    and type(self._agent).stt_node is not Agent.stt_node
                    and self._turn_policy.turn_detection == "stt"
                ),
                active_server_turn_detection_enabled=self._rt_turn_detection_enabled,
            )
            if self._realtime_input_mode == "text":
                self._validate_text_mode_stt_path(
                    resolved_stt=prospective_stt,
                    resolved_vad=prospective_vad,
                    resolved_turn_detection=policy.turn_detection,
                    resolved_turn_detection_explicit=policy.turn_detection_explicit,
                    configured_turn_detection=policy.configured_turn_detection,
                )

        # Validate the complete prospective policy before mutating any model or listener.
        if policy is not None and self._audio_recognition is not None:
            self._audio_recognition._check_vad_silence_requirement(
                detector=(
                    policy.turn_detection
                    if isinstance(policy.turn_detection, _StreamingTurnDetector)
                    else None
                ),
                vad=prospective_vad,
            )

        if is_given(new_stt):
            old_stt = self.stt
            if isinstance(old_stt, stt.STT):
                old_stt.off("metrics_collected", self._on_metrics_collected)
                old_stt.off("error", self._on_error)
                self._session.off("conversation_item_added", old_stt._push_conversation_item)

            self._agent._stt = new_stt
            resolved_stt = self.stt
            if self._audio_recognition is not None:
                self._audio_recognition._update_stt(
                    self._agent.stt_node if resolved_stt else None,
                    model=resolved_stt.model if isinstance(resolved_stt, stt.STT) else None,
                    provider=resolved_stt.provider if isinstance(resolved_stt, stt.STT) else None,
                    reset_context=True,
                )
            self._session._keyterm_detector.swap_stt(resolved_stt)

            if isinstance(resolved_stt, stt.STT):
                resolved_stt.prewarm()
                resolved_stt.on("metrics_collected", self._on_metrics_collected)
                resolved_stt.on("error", self._on_error)
                forward_chat_ctx = self._session._opts.stt_context_options["forward_chat_context"]
                if resolved_stt.capabilities.chat_context and forward_chat_ctx:
                    self._session.on(
                        "conversation_item_added", resolved_stt._push_conversation_item
                    )

        if is_given(new_vad):
            old_vad = self.vad
            if isinstance(old_vad, vad.VAD):
                old_vad.off("metrics_collected", self._on_metrics_collected)

            self._agent._vad = new_vad
            if self._audio_recognition is not None:
                self._audio_recognition._update_vad(self.vad)
            if isinstance(self.vad, vad.VAD):
                self.vad.on("metrics_collected", self._on_metrics_collected)

        if is_given(new_llm):
            old_llm = self.llm
            if isinstance(old_llm, llm.LLM):
                old_llm.off("metrics_collected", self._on_metrics_collected)
                old_llm.off("error", self._on_error)

            self._agent._llm = new_llm  # llm_node reads activity.llm per generation
            if isinstance(self.llm, llm.LLM):
                self.llm.prewarm()
                self.llm.on("metrics_collected", self._on_metrics_collected)
                self.llm.on("error", self._on_error)

        if policy is not None:
            self._apply_turn_policy(policy)
            if self._audio_recognition is not None:
                self._audio_recognition._update_options(
                    turn_detection=policy.turn_detection,
                    finalize_empty_transcript_on_timeout=(
                        policy.finalize_empty_transcript_on_timeout
                    ),
                )

        if is_given(new_tts):
            old_tts = self.tts
            if isinstance(old_tts, tts.TTS):
                old_tts.off("metrics_collected", self._on_metrics_collected)
                old_tts.off("error", self._on_error)

            self._agent._tts = new_tts  # tts_node reads activity.tts per synthesis
            if isinstance(self.tts, tts.TTS):
                self.tts.prewarm()
                self.tts.on("metrics_collected", self._on_metrics_collected)
                self.tts.on("error", self._on_error)

    def _create_speech_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        speech_handle: SpeechHandle | None = None,
        name: str | None = None,
    ) -> asyncio.Task[Any]:
        """
        This method must only be used for tasks that "could" create a new SpeechHandle.
        When draining, every task created with this method will be awaited.
        """
        # https://github.com/python/cpython/pull/31837 alternative impl
        tk = _AgentActivityContextVar.set(self)
        tk1 = None
        if speech_handle is not None:
            tk1 = _SpeechHandleContextVar.set(speech_handle)

        # Capture the current OpenTelemetry context to ensure proper span nesting
        current_context = otel_context.get_current()

        # Create a wrapper coroutine that runs in the captured context
        async def _context_aware_coro() -> Any:
            # Attach the captured context before running the original coroutine
            token = otel_context.attach(current_context)
            try:
                return await coro
            finally:
                otel_context.detach(token)

        task = asyncio.create_task(_context_aware_coro(), name=name)
        self._speech_tasks.append(task)
        task.add_done_callback(lambda _: self._speech_tasks.remove(task))

        _set_activity_task_info(task, speech_handle=speech_handle)

        if speech_handle is not None:
            # mark a speech_handle as done, if every "linked" tasks are done
            speech_handle._tasks.append(task)

            def _mark_done_if_needed(_: asyncio.Task) -> None:
                if all(task.done() for task in speech_handle._tasks):
                    speech_handle._mark_done()

            task.add_done_callback(_mark_done_if_needed)

        task.add_done_callback(lambda _: self._wake_up_scheduling_task())
        _AgentActivityContextVar.reset(tk)

        if tk1:
            _SpeechHandleContextVar.reset(tk1)

        return task

    async def start(self, *, reuse_resources: _ReusableResources | None = None) -> None:
        # `start` must only be called by AgentSession

        async with self._lock:
            if self._started:
                return

            start_span = tracer.start_span(
                "start_agent_activity",
                attributes={trace_types.ATTR_AGENT_LABEL: self.agent.label},
            )
            try:
                self._agent._activity = self

                with tracer.use_span(start_span, end_on_exit=False):
                    if isinstance(self.llm, llm.LLM):
                        self.llm.prewarm()

                    if isinstance(self.stt, stt.STT):
                        self.stt.prewarm()

                    if isinstance(self.tts, tts.TTS):
                        self.tts.prewarm()

                # one-shot — not re-run on resume, so toolsets and MCP connections
                # survive pause/resume
                await self._setup_toolsets()

                # don't use start_span for _start_session, avoid nested user/assistant turns
                await self._start_session(reuse_resources=reuse_resources)
                self._started = True

                @tracer.start_as_current_span(
                    "on_enter",
                    context=trace.set_span_in_context(start_span),
                    attributes={trace_types.ATTR_AGENT_LABEL: self._agent.label},
                )
                @utils.log_exceptions(logger=logger)
                async def _traceable_on_enter() -> None:
                    data = _OnEnterData(session=self._session, agent=self._agent)
                    try:
                        tk = _OnEnterContextVar.set(data)
                        await self._agent.on_enter()
                    finally:
                        _OnEnterContextVar.reset(tk)

                self._on_enter_task = task = self._create_speech_task(
                    _traceable_on_enter(), name="AgentTask_on_enter"
                )
                _set_activity_task_info(task, inline_task=True)
            finally:
                start_span.end()

    async def _detach_reusable_resources(self, new_activity: AgentActivity) -> _ReusableResources:
        """Detach reusable resources for handoff to *new_activity*."""
        resources = _ReusableResources()

        try:
            # stt pipeline; only reuse with the default stt_node, a custom override may
            # access the old self.session/activity inside the yield loop after detach
            if (
                self._audio_recognition
                and self.stt is not None
                and type(self.agent).stt_node is Agent.stt_node
                and type(new_activity.agent).stt_node is Agent.stt_node
                and self.stt is new_activity.stt
            ):
                resources.stt_pipeline = await self._audio_recognition._detach_stt()

            # reuse the stream during a handoff whenever we can
            if (
                self._audio_recognition
                and isinstance(self._turn_detection, _StreamingTurnDetector)
                and self._turn_detection is new_activity._turn_detection
            ):
                resources.turn_detector_stream = self._audio_recognition._detach_turn_detector()

            # rt session
            if (
                self._rt_session is not None
                and isinstance(self.llm, llm.RealtimeModel)
                and self.llm is new_activity.llm
            ):
                # context update is supported or chat context is equivalent
                reusable = self.llm.capabilities.mutable_chat_context or (
                    self._rt_session.chat_ctx.copy(
                        exclude_instructions=True, exclude_handoff=True, exclude_config_update=True
                    ).is_equivalent(
                        new_activity.agent.chat_ctx.copy(
                            exclude_instructions=True,
                            exclude_handoff=True,
                            exclude_config_update=True,
                        )
                    )
                )
                # instructions update is supported or instructions are the same
                reusable = reusable and (
                    self.llm.capabilities.mutable_instructions
                    or self.agent.instructions == new_activity.agent.instructions
                )
                # tools update is supported or tools are the same
                reusable = reusable and (
                    self.llm.capabilities.mutable_tools
                    or llm.ToolContext(self.tools) == llm.ToolContext(new_activity.tools)
                )
                # only reuse if the new activity resolves server-side turn detection the same way
                reusable = reusable and (
                    self._rt_turn_detection_enabled == new_activity._rt_turn_detection_enabled
                )
                # input routing is activity-owned; switching between raw audio and external
                # text requires a fresh provider session with no buffered turn state.
                reusable = reusable and (
                    self._realtime_input_mode == new_activity._realtime_input_mode
                )

                if reusable:
                    # detach: remove event listeners but don't close the session
                    self._rt_session.off("generation_created", self._on_generation_created)
                    self._rt_session.off("input_speech_started", self._on_input_speech_started)
                    self._rt_session.off("input_speech_stopped", self._on_input_speech_stopped)
                    self._rt_session.off(
                        "input_audio_transcription_completed",
                        self._on_input_audio_transcription_completed,
                    )
                    self._rt_session.off("metrics_collected", self._on_metrics_collected)
                    self._rt_session.off("remote_item_added", self._on_remote_item_added)
                    self._rt_session.off("error", self._on_error)
                    if isinstance(self._rt_session, _FallbackRealtimeSession):
                        self._rt_session._agent_session = None
                    resources.rt_session = self._rt_session
                    self._rt_session = None  # prevent _close_session from closing it

        except Exception:
            # avoid leaking resources
            await resources.cleanup()
            raise

        return resources

    async def _setup_toolsets(self) -> None:
        """Build MCP toolsets, scope each AsyncToolset, and call ``setup()`` on all."""
        from ..llm.async_toolset import AsyncToolset

        assert self._lock.locked(), "_setup_toolsets must run under the activity lock."

        if self.mcp_servers:
            from ..llm.mcp import MCPToolset

            self._mcp_tools = [
                MCPToolset(id=utils.shortuuid("mcp_toolset_"), mcp_server=server)
                for server in self.mcp_servers
            ]

        session_toolsets = [t for t in self._session.tools if isinstance(t, llm.Toolset)]
        agent_toolsets = [t for t in self._agent.tools if isinstance(t, llm.Toolset)]
        mcp_toolsets: list[llm.Toolset] = list(self._mcp_tools)

        all_toolsets = session_toolsets + agent_toolsets + mcp_toolsets
        if not all_toolsets:
            return

        # session.tools → session-scoped (survives handoff); agent.tools → activity-scoped
        for ts in llm.ToolContext(session_toolsets).toolsets:
            if isinstance(ts, AsyncToolset):
                ts._attach_activity(activity=None, session=self._session)
        for ts in llm.ToolContext(agent_toolsets).toolsets:
            if isinstance(ts, AsyncToolset):
                ts._attach_activity(activity=self, session=self._session)

        @utils.log_exceptions(logger=logger)
        async def _do_setup(toolset: llm.Toolset) -> None:
            await toolset.setup()

        await asyncio.gather(
            *(_do_setup(ts) for ts in all_toolsets),
            return_exceptions=True,
        )

    async def _start_session(self, *, reuse_resources: _ReusableResources | None = None) -> None:
        assert self._lock.locked(), "_start_session should only be used when locked."

        if isinstance(self.llm, llm.LLM):
            self.llm.on("metrics_collected", self._on_metrics_collected)
            self.llm.on("error", self._on_error)

        if isinstance(self.stt, stt.STT):
            self.stt.on("metrics_collected", self._on_metrics_collected)
            self.stt.on("error", self._on_error)

        if isinstance(self.tts, tts.TTS):
            self.tts.on("metrics_collected", self._on_metrics_collected)
            self.tts.on("error", self._on_error)

        if isinstance(self.vad, vad.VAD):
            self.vad.on("metrics_collected", self._on_metrics_collected)

        if isinstance(self._interruption_detector, inference.AdaptiveInterruptionDetector):
            self._interruption_detector.on("metrics_collected", self._on_metrics_collected)
            self._interruption_detector.on("error", self._on_error)
            self._interruption_detector.on("overlapping_speech", self._on_overlap_speech_ended)

        self._set_turn_detection_metrics_source(
            self._turn_detection
            if isinstance(self._turn_detection, inference.TurnDetector)
            else None
        )

        # keyterm detection runs its own LLM, surface its usage
        self._session._keyterm_detector.on("metrics_collected", self._on_metrics_collected)

        if isinstance(self.llm, llm.RealtimeModel):
            rt_reused = reuse_resources is not None and reuse_resources.rt_session is not None
            if rt_reused:
                assert reuse_resources and reuse_resources.rt_session is not None
                logger.debug("reusing realtime session from previous activity")
                self._rt_session = reuse_resources.rt_session
                reuse_resources.rt_session = None  # ownership transferred

                # clear any stale audio/generation state
                self._rt_session.interrupt()
                self._clear_realtime_input()
            else:
                # disable only when we resolved it off AND the model can (guards a model whose
                # turn detection is explicitly pinned off from a spurious disable)
                turn_detection_disabled = (
                    not self._rt_turn_detection_enabled
                    and self.llm.capabilities.can_disable_turn_detection
                )
                self._rt_session = self.llm.session(turn_detection_disabled=turn_detection_disabled)
                logger.debug("created new realtime session for activity, id=%s", self._rt_session)

            self._rt_session.on("generation_created", self._on_generation_created)
            self._rt_session.on("input_speech_started", self._on_input_speech_started)
            self._rt_session.on("input_speech_stopped", self._on_input_speech_stopped)
            self._rt_session.on(
                "input_audio_transcription_completed",
                self._on_input_audio_transcription_completed,
            )
            self._rt_session.on("metrics_collected", self._on_metrics_collected)
            self._rt_session.on("remote_item_added", self._on_remote_item_added)
            self._rt_session.on("error", self._on_error)

            # the fallback adapter's session needs the AgentSession to drive interrupt/generate_reply on swap
            if isinstance(self._rt_session, _FallbackRealtimeSession):
                self._rt_session._agent_session = self._session

            remove_instructions(self._agent._chat_ctx)

            capabilities = self.llm.capabilities
            reset_instructions = reset_chat_ctx = reset_tools = True
            if rt_reused:
                # skip the update if the session is reused and no mid-session update is supported
                # this means the content is the same as the previous session
                reset_instructions = capabilities.mutable_instructions
                reset_chat_ctx = capabilities.mutable_chat_context
                reset_tools = capabilities.mutable_tools

            await self._rt_session._update_session(
                instructions=self._render_realtime_instructions(self._agent.instructions)
                if reset_instructions
                else NOT_GIVEN,
                chat_ctx=self._agent.chat_ctx if reset_chat_ctx else NOT_GIVEN,
                tools=llm.ToolContext(self.tools).flatten() if reset_tools else NOT_GIVEN,
            )

            self._realtime_spans = utils.BoundedDict[str, trace.Span](maxsize=100)
            if not capabilities.audio_output and not self.tts and self._session.output.audio:
                logger.error(
                    "audio output is enabled but RealtimeModel has no audio modality "
                    "and no TTS is set. Either enable audio modality in the RealtimeModel "
                    "or set a TTS model."
                )

        elif isinstance(self.llm, llm.LLM):
            try:
                update_instructions(
                    self._agent._chat_ctx,
                    instructions=self._agent.instructions,
                    add_if_missing=True,
                )
            except ValueError:
                logger.exception("failed to update the instructions")

        # Record initial agent configuration (skip if empty)
        initial_tools = get_fnc_tool_names(self.tools) or None
        instr = self._agent.instructions
        # collapse modality variants for the record; audio-first matches the
        # update_instructions default for voice sessions
        initial_instructions = (
            instr.render(modality="audio") if isinstance(instr, Instructions) else instr
        )
        if initial_instructions or initial_tools:
            initial_config = llm.AgentConfigUpdate(
                instructions=initial_instructions,
                tools_added=initial_tools,
            )
            initial_config._tools = llm.ToolContext(self.tools).flatten()
            self._agent._chat_ctx.insert(initial_config)
            self._session._chat_ctx.insert(initial_config)

        await self._resume_scheduling_task()
        # skip default vad when llm does not need it
        wired_vad = self.vad
        if wired_vad is not None and self.using_default_vad and self._rt_turn_detection_enabled:
            wired_vad = None
        self._audio_recognition = AudioRecognition(
            self._session,
            hooks=self,
            stt=self._agent.stt_node if self.stt else None,
            vad=wired_vad,
            using_default_vad=self.using_default_vad,
            interruption_detection=self._interruption_detector,
            endpointing=create_endpointing(self.endpointing_opts),
            turn_detection=self._turn_detection,
            stt_model=self.stt.model if self.stt else None,
            stt_provider=self.stt.provider if self.stt else None,
            finalize_empty_transcript_on_timeout=(
                self._turn_policy.finalize_empty_transcript_on_timeout
            ),
        )
        stt_pipeline = reuse_resources.stt_pipeline if reuse_resources else None
        turn_detector_stream = reuse_resources.turn_detector_stream if reuse_resources else None
        if stt_pipeline is not None:
            logger.debug("reusing STT pipeline from previous activity")
        if turn_detector_stream is not None:
            logger.debug("reusing turn detector stream from previous activity")
        self._audio_recognition._start(
            stt_pipeline=stt_pipeline,
            turn_detector_stream=turn_detector_stream,
        )
        if reuse_resources:
            # ownership transferred to the new AudioRecognition
            reuse_resources.stt_pipeline = None
            reuse_resources.turn_detector_stream = None

        if isinstance(self.stt, stt.STT):
            # bind the session's keyterm detector to this activity's STT (detection uses its
            # own LLM, configured via stt_context_options, not the agent's)
            self._session._keyterm_detector.start(self._session, stt=self.stt)

            # forward conversation turns to STTs that consume context natively; gated by the STT's
            # capability and the session's forward_chat_context toggle. stateless, activity-scoped.
            forward_chat_ctx = self._session._opts.stt_context_options["forward_chat_context"]
            if self.stt.capabilities.chat_context and forward_chat_ctx:
                self._session.on("conversation_item_added", self.stt._push_conversation_item)

    @tracer.start_as_current_span("drain_agent_activity")
    async def drain(
        self, *, new_activity: AgentActivity | None = None
    ) -> _ReusableResources | None:
        # `drain` must only be called by AgentSession
        # AgentSession makes sure there is always one agent available to the users.
        current_span = trace.get_current_span()
        current_span.set_attribute(trace_types.ATTR_AGENT_LABEL, self._agent.label)

        @tracer.start_as_current_span(
            "on_exit", attributes={trace_types.ATTR_AGENT_LABEL: self._agent.label}
        )
        @utils.log_exceptions(logger=logger)
        async def _traceable_on_exit() -> None:
            await self._agent.on_exit()

        async with self._lock:
            if self._on_exit_task is None:
                self._on_exit_task = task = self._create_speech_task(
                    _traceable_on_exit(), name="AgentTask_on_exit"
                )
                _set_activity_task_info(task, inline_task=True)

            self._cancel_preemptive_generation()

            try:
                await self._on_exit_task
            except Exception:
                pass  # already logged by @log_exceptions

            shutdown_callback_tasks = (
                [
                    task
                    for task in self._speech_tasks
                    if not task.done()
                    and task.get_name() == "AgentActivity._user_turn_completed_task"
                ]
                if self._session._closing
                else None
            )
            await self._pause_scheduling_task(blocked_tasks=shutdown_callback_tasks)

            # detach after speech tasks are done but before _close_session
            if new_activity is not None:
                try:
                    return await self._detach_reusable_resources(new_activity)
                except BaseException:
                    logger.exception("failed to detach reusable resources")

            return None

    async def _pause_scheduling_task(
        self, *, blocked_tasks: list[asyncio.Task] | None = None
    ) -> None:
        assert self._lock.locked(), "_finalize_main_task should only be used when locked."

        if self._scheduling_paused:
            return

        await self._session._keyterm_detector.aclose()

        self._scheduling_paused = True
        if blocked_tasks:
            self._add_drain_blocked_tasks(blocked_tasks)
        self._wake_up_scheduling_task()

        if self._scheduling_atask is not None:
            # When pausing/draining, we ensure that all speech_tasks complete fully.
            # This means that even if the SpeechHandle themselves have finished,
            # we still wait for the entire execution (e.g function_tools)
            await asyncio.shield(self._scheduling_atask)

    def _add_drain_blocked_tasks(self, tasks: list[asyncio.Task[Any]]) -> None:
        # tasks blocked on an agent handoff are excluded from the drain wait,
        # otherwise drain would wait for them while the handoff waits for drain's lock
        self._drain_blocked_tasks.update(tasks)
        self._wake_up_scheduling_task()

    async def _resume_scheduling_task(self) -> None:
        assert self._lock.locked(), "_finalize_main_task should only be used when locked."

        if not self._scheduling_paused:
            return

        self._scheduling_paused = False
        self._new_turns_blocked = False
        self._drain_blocked_tasks.clear()
        self._scheduling_atask = asyncio.create_task(
            self._scheduling_task(), name="_scheduling_task"
        )

    async def resume(self, *, reuse_resources: _ReusableResources | None = None) -> None:
        # `resume` must only be called by AgentSession

        async with self._lock:
            span = tracer.start_span(
                "resume_agent_activity",
                attributes={trace_types.ATTR_AGENT_LABEL: self.agent.label},
            )
            try:
                await self._start_session(reuse_resources=reuse_resources)
            finally:
                span.end()

    def _wake_up_scheduling_task(self) -> None:
        self._q_updated.set()

    async def pause(
        self,
        *,
        blocked_tasks: list[asyncio.Task],
        new_activity: AgentActivity | None = None,
    ) -> _ReusableResources | None:
        # `pause` must only be called by AgentSession

        # When draining, the tasks that have done the "premption" must be ignored.
        # They will most likely block until the Agent transition is done. So we must not
        # wait for them to avoid deadlocks.

        # When resuming, the AgentSession.update_agent must use the same AgentActivity instance!
        async with self._lock:
            if self._closed:
                # already closed by the session close
                return None

            span = tracer.start_span(
                "pause_agent_activity",
                attributes={trace_types.ATTR_AGENT_LABEL: self._agent.label},
            )

            resources: _ReusableResources | None = None
            try:
                await self._pause_scheduling_task(blocked_tasks=blocked_tasks)

                # detach after speech tasks are done but before _close_session
                if new_activity is not None:
                    resources = await self._detach_reusable_resources(new_activity)

                await self._close_session()
            except BaseException:
                if resources is not None:
                    await resources.cleanup()
                raise
            finally:
                span.end()

            return resources

    async def _close_session(self) -> None:
        assert self._lock.locked(), "_close_session should only be used when locked."

        # Freeze turn admission before either recognition or provider teardown can race a
        # bounded callback. Existing bounded text may still settle into local history.
        self._new_turns_blocked = True
        close_deadline = asyncio.get_running_loop().time() + max(
            0.0, self._session.options.session_close_transcript_timeout
        )

        if isinstance(self.llm, llm.LLM):
            self.llm.off("metrics_collected", self._on_metrics_collected)
            self.llm.off("error", self._on_error)

        if isinstance(self.llm, llm.RealtimeModel) and self._rt_session is not None:
            self._rt_session.off("generation_created", self._on_generation_created)
            self._rt_session.off("input_speech_started", self._on_input_speech_started)
            self._rt_session.off("input_speech_stopped", self._on_input_speech_stopped)
            self._rt_session.off(
                "input_audio_transcription_completed",
                self._on_input_audio_transcription_completed,
            )
            self._rt_session.off("metrics_collected", self._on_metrics_collected)
            self._rt_session.off("remote_item_added", self._on_remote_item_added)
            self._rt_session.off("error", self._on_error)
            if isinstance(self._rt_session, _FallbackRealtimeSession):
                self._rt_session._agent_session = None

        if isinstance(self.stt, stt.STT):
            self.stt.off("metrics_collected", self._on_metrics_collected)
            self.stt.off("error", self._on_error)
            self._session.off("conversation_item_added", self.stt._push_conversation_item)

        if isinstance(self.tts, tts.TTS):
            self.tts.off("metrics_collected", self._on_metrics_collected)
            self.tts.off("error", self._on_error)

        if isinstance(self.vad, vad.VAD):
            self.vad.off("metrics_collected", self._on_metrics_collected)

        if isinstance(self._interruption_detector, inference.AdaptiveInterruptionDetector):
            self._interruption_detector.off("metrics_collected", self._on_metrics_collected)
            self._interruption_detector.off("error", self._on_error)
            self._interruption_detector.off("overlapping_speech", self._on_overlap_speech_ended)

        self._set_turn_detection_metrics_source(None)

        self._session._keyterm_detector.off("metrics_collected", self._on_metrics_collected)

        # A bounded deferred EOU may be waiting for its provider-input turn. Release only that
        # local waiter before abandoning the provider queue; the admission gate above ensures it
        # records the transcript without committing or generating on the closing session.
        for transaction in self._deferred_realtime_turns:
            if transaction.ready_fut is not None and not transaction.ready_fut.done():
                transaction.ready_fut.set_result(transaction)
        self._reset_realtime_audio_input_state()

        if self._realtime_spans is not None:
            self._realtime_spans.clear()

        # Begin closing the provider so a blocked receive loop can unwind, while the admission
        # gate above lets an already-bounded recognition task retain its text without starting
        # provider work. Recognition teardown happens only after that task has settled.
        rt_close_task: asyncio.Task[None] | None = None

        def _observe_realtime_close(task: asyncio.Task[None]) -> None:
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("failed to close realtime session")

        if self._rt_session is not None:
            rt_close_task = asyncio.create_task(
                self._rt_session.aclose(), name="AgentActivity.close_realtime_session"
            )
            rt_close_task.add_done_callback(_observe_realtime_close)

        current_task = asyncio.current_task()
        cancellation_requested: set[asyncio.Task[None]] = set()

        def _observe_bounded_turn(task: asyncio.Task[None]) -> None:
            try:
                task.result()
            except (asyncio.CancelledError, Exception):
                # The bounded task logs its own failure. Retrieving the result prevents a late
                # callback completion from surfacing as an unhandled task exception.
                pass

        async def _settle_bounded_turn(task: asyncio.Task[None] | None) -> None:
            if task is None or task is current_task:
                return
            if task.done():
                _observe_bounded_turn(task)
                return

            remaining = max(0.0, close_deadline - asyncio.get_running_loop().time())
            done, _ = await asyncio.wait({task}, timeout=remaining)
            if task in done:
                _observe_bounded_turn(task)
                return
            if task.done():
                _observe_bounded_turn(task)
                return

            if task not in cancellation_requested:
                cancellation_requested.add(task)
                logger.warning(
                    "timed out waiting for on_user_turn_completed during shutdown; "
                    "continuing cleanup",
                    extra={"timeout": self._session.options.session_close_transcript_timeout},
                )
                task.cancel()
                task.add_done_callback(_observe_bounded_turn)

        bounded_turn_tasks = {
            task
            for task in self._speech_tasks
            if task.get_name() == "AgentActivity._user_turn_completed_task"
        }
        if self._user_turn_completed_atask is not None:
            bounded_turn_tasks.add(self._user_turn_completed_atask)
        for bounded_turn_task in bounded_turn_tasks:
            await _settle_bounded_turn(bounded_turn_task)

        if rt_close_task is not None:
            await rt_close_task

        latest_bounded_turn_tasks = {
            task
            for task in self._speech_tasks
            if task.get_name() == "AgentActivity._user_turn_completed_task"
        }
        if self._user_turn_completed_atask is not None:
            latest_bounded_turn_tasks.add(self._user_turn_completed_atask)
        for bounded_turn_task in latest_bounded_turn_tasks - bounded_turn_tasks:
            await _settle_bounded_turn(bounded_turn_task)

        if self._audio_recognition is not None:
            await self._audio_recognition._aclose()

        await self._cancel_speech_pause(
            old_task=self._cancel_speech_pause_task,
            interrupt=False,  # don't interrupt the paused speech, it's managed by _pause_scheduling_task
        )
        self._cancel_speech_pause_task = None

    async def aclose(self) -> None:
        # `aclose` must only be called by AgentSession

        async with self._lock:
            if self._closed:
                return

            self._closed = True
            self._cancel_preemptive_generation()
            await self._session._keyterm_detector.aclose()

            # on_exit_task should be awaited in `drain`
            self._on_exit_task = None

            # cancel cancellable tools and await the rest before teardown
            await self._tool_executor.drain()

            await self._close_session()
            await asyncio.gather(*self._interrupt_background_speeches(force=False))

            if self._scheduling_atask is not None:
                await utils.aio.cancel_and_wait(self._scheduling_atask)

            # session-scoped toolsets are closed by the session; this only closes
            # the agent's own toolsets + MCP — all of which outlive pause
            toolsets = self._mcp_tools + [
                tool for tool in self._agent.tools if isinstance(tool, llm.Toolset)
            ]
            if toolsets:
                await asyncio.gather(
                    *(toolset.aclose() for toolset in toolsets), return_exceptions=True
                )

            # final sweep: anything non-cancellable that survived drain dies here
            await self._tool_executor.aclose()

            self._agent._activity = None

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if not self._started:
            return

        aec_warmup_active: bool = (
            self._session.agent_state == "speaking"
            and self._session._aec_warmup_remaining > 0
            and self._session._aec_warmup_timer is not None
        )

        uninterruptible_speech_active: bool = (
            self._current_speech is not None
            and not self._current_speech.done()
            and not self._current_speech.interrupted
            and not self._current_speech.allow_interruptions
            and self._session.options.interruption["discard_audio_if_uninterruptible"]
        )

        should_discard: bool = aec_warmup_active or uninterruptible_speech_active

        # When discarding, substitute silence on the paths that would otherwise
        # see contaminated/echoed audio (STT, realtime model) so the downstream
        # stream stays continuous. VAD, AMD and the interruption detector keep
        # receiving the real frame so they can still react to the user.
        stt_frame: rtc.AudioFrame | None = None
        if should_discard:
            stt_frame = utils.audio.silence_frame_like(frame)

        if (
            self._rt_session is not None
            and self._realtime_input_mode == "audio"
            and not self._new_turns_blocked
        ):
            model_frame = stt_frame if stt_frame is not None else frame
            if self._rt_audio_input_sealed:
                self._get_deferred_realtime_audio_input().frames.append(model_frame)
            else:
                self._realtime_turn.input_observed = True
                self._rt_session.push_audio(model_frame)

        if self._audio_recognition is not None:
            self._audio_recognition._push_audio(frame, stt_frame=stt_frame)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        if not self._started:
            return

        if self._rt_session is not None:
            self._rt_session.push_video(frame)

    def say(
        self,
        text: str | AsyncIterable[str],
        *,
        audio: NotGivenOr[AsyncIterable[rtc.AudioFrame]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        add_to_chat_ctx: bool = True,
    ) -> SpeechHandle:
        if (
            not is_given(audio)
            and not self.tts
            and not (isinstance(self.llm, llm.RealtimeModel) and self.llm.capabilities.supports_say)
            and self._session.output.audio
            and self._session.output.audio_enabled
        ):
            raise RuntimeError(
                "trying to generate speech from text without a TTS model or a RealtimeSession that supports say(); "
                "add a TTS model to AgentSession to enable say()"
            )

        if self._rt_turn_detection_enabled and allow_interruptions is False:
            logger.warning(
                "the RealtimeModel uses a server-side turn detection, allow_interruptions cannot be False when using VoiceAgent.say(), "  # noqa: E501
                "disable turn_detection in the RealtimeModel and use VAD on the AgentTask/VoiceAgent instead"  # noqa: E501
            )
            allow_interruptions = NOT_GIVEN

        handle = SpeechHandle.create(
            allow_interruptions=allow_interruptions
            if is_given(allow_interruptions)
            else self.allow_interruptions
        )
        self._session.emit(
            "speech_created",
            SpeechCreatedEvent(speech_handle=handle, user_initiated=True, source="say"),
        )

        if (
            self._rt_session is not None
            and not is_given(audio)
            and not self.tts
            and isinstance(self.llm, llm.RealtimeModel)
            and self.llm.capabilities.supports_say
        ):
            if not add_to_chat_ctx:
                logger.warning(
                    "add_to_chat_ctx=False is not supported when say() uses a RealtimeModel; "
                    "the message will still be added to the chat context"
                )
            self._create_speech_task(
                self._realtime_reply_task(
                    speech_handle=handle,
                    text=text,
                    model_settings=ModelSettings(),
                ),
                speech_handle=handle,
                name="AgentActivity.realtime_say",
            )
        else:
            task = self._create_speech_task(
                self._tts_task(
                    speech_handle=handle,
                    text=text,
                    audio=audio or None,
                    add_to_chat_ctx=add_to_chat_ctx,
                    model_settings=ModelSettings(),
                ),
                speech_handle=handle,
                name="AgentActivity.tts_say",
            )
            task.add_done_callback(self._on_pipeline_reply_done)
        self._schedule_speech(handle, SpeechHandle.SPEECH_PRIORITY_NORMAL)
        return handle

    def _generate_reply(
        self,
        *,
        user_message: NotGivenOr[llm.ChatMessage | None] = NOT_GIVEN,
        chat_ctx: NotGivenOr[llm.ChatContext | None] = NOT_GIVEN,
        instructions: NotGivenOr[str | Instructions] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[str]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        schedule_speech: bool = True,
        input_details: InputDetails = DEFAULT_INPUT_DETAILS,
        realtime_audio_input_owner: _RealtimeAudioInputOwner | None = None,
        commit_realtime_audio: bool = False,
    ) -> SpeechHandle:
        if self._rt_turn_detection_enabled and allow_interruptions is False:
            logger.warning(
                "the RealtimeModel uses a server-side turn detection, allow_interruptions cannot be False when using VoiceAgent.generate_reply(), "  # noqa: E501
                "disable turn_detection in the RealtimeModel and use VAD on the AgentTask/VoiceAgent instead"  # noqa: E501
            )
            allow_interruptions = NOT_GIVEN

        if self.llm is None:
            raise RuntimeError("trying to generate reply without an LLM model")

        task = asyncio.current_task()
        if not is_given(tool_choice) and task is not None:
            if task_info := _get_activity_task_info(task):
                if task_info.function_call is not None:
                    # when generate_reply is called inside a function_tool, set tool_choice to None by default  # noqa: E501
                    tool_choice = "none"

        all_tools = self.tools.copy()

        # resolve tool names to Tool objects if tools param is given
        resolved_tools: NotGivenOr[list[llm.Tool | llm.Toolset]] = NOT_GIVEN
        if is_given(tools):
            tool_ctx = llm.ToolContext(all_tools)
            toolset_dict = {t.id: t for t in tool_ctx.toolsets}
            tool_dict = {t.id: t for t in tool_ctx.flatten()}
            resolved_tools = list[llm.Tool | llm.Toolset]()
            for name in set(tools):
                tool = toolset_dict.get(name) or tool_dict.get(name)
                if tool is None:
                    raise ValueError(
                        f"tool '{name}' not found in agent's registered tools. "
                        f"Available tools: {list(tool_ctx.function_tools.keys())}"
                    )
                resolved_tools.append(tool)

        handle = SpeechHandle.create(
            allow_interruptions=allow_interruptions
            if is_given(allow_interruptions)
            else self.allow_interruptions,
            input_details=input_details,
        )
        self._session.emit(
            "speech_created",
            SpeechCreatedEvent(speech_handle=handle, user_initiated=True, source="generate_reply"),
        )

        if isinstance(self.llm, llm.RealtimeModel):
            if (
                realtime_audio_input_owner is None
                and self._rt_session is not None
                and self._realtime_input_mode == "audio"
                and input_details.modality == "audio"
                and (self._rt_user_activity_started or self._rt_audio_input_sealed)
            ):
                realtime_audio_input_owner = self._rt_audio_input_token
            if isinstance(realtime_audio_input_owner, _RealtimeTurnTransaction):
                realtime_audio_input_owner.speech_handle = handle
            elif isinstance(realtime_audio_input_owner, asyncio.Future):
                for transaction in self._deferred_realtime_turns:
                    if transaction.ready_fut is realtime_audio_input_owner:
                        transaction.speech_handle = handle
                        break
            self._create_speech_task(
                self._realtime_reply_task(
                    speech_handle=handle,
                    realtime_audio_input_owner=realtime_audio_input_owner,
                    commit_realtime_audio=commit_realtime_audio,
                    user_message=user_message if is_given(user_message) else None,
                    instructions=self._render_realtime_instructions(instructions)
                    if instructions
                    else None,
                    tools=resolved_tools if is_given(resolved_tools) else None,
                    model_settings=ModelSettings(tool_choice=tool_choice),
                ),
                speech_handle=handle,
                name="AgentActivity.realtime_reply",
            )

        elif isinstance(self.llm, llm.LLM):
            task = self._create_speech_task(
                self._pipeline_reply_task(
                    speech_handle=handle,
                    chat_ctx=chat_ctx or self._agent._chat_ctx,
                    tools=resolved_tools if is_given(resolved_tools) else all_tools,
                    new_message=user_message if is_given(user_message) else None,
                    instructions=instructions or None,
                    model_settings=ModelSettings(
                        tool_choice=tool_choice
                        if utils.is_given(tool_choice) or self._tool_choice is None
                        else self._tool_choice
                    ),
                ),
                speech_handle=handle,
                name="AgentActivity.pipeline_reply",
            )
            task.add_done_callback(self._on_pipeline_reply_done)

        if schedule_speech:
            self._schedule_speech(handle, SpeechHandle.SPEECH_PRIORITY_NORMAL)

        return handle

    def _cancel_preemptive_generation(self) -> None:
        if self._preemptive_generation is not None:
            self._preemptive_generation.speech_handle._cancel()
            self._preemptive_generation = None

    def _pause_authorization(self) -> None:
        self._authorization_allowed.clear()

    def _resume_authorization(self) -> None:
        self._authorization_allowed.set()

    def _interrupt_background_speeches(self, force: bool = False) -> list[SpeechHandle]:
        interrupted_speeches: list[SpeechHandle] = []
        for speech in self._background_speeches:
            if force or speech.allow_interruptions:
                interrupted_speeches.append(speech.interrupt(force=force))

        return interrupted_speeches

    def interrupt(self, *, force: bool = False) -> asyncio.Future[None]:
        """Interrupt the current speech generation and any queued speeches.

        A queued speech that disallows interruptions keeps playing, along with the ones
        behind it, unless ``force`` is set.

        Returns:
            An asyncio.Future that completes when the interruption is fully processed
            and chat context has been updated

        Raises:
            RuntimeError: If the speech currently playing disallows interruptions and
                ``force`` is False.
        """
        self._cancel_preemptive_generation()

        future = asyncio.Future[None]()

        interrupted_speeches = self._interrupt_background_speeches(force=force)

        if self._current_speech is not None:
            self._current_speech.interrupt(force=force)
            interrupted_speeches.append(self._current_speech)

        if self._rt_session is not None:
            self._rt_session.interrupt()

        # _speech_q is a heap, so its list order is not the order it pops in
        for _, _, speech in sorted(self._speech_q, key=lambda item: (item[0], item[1])):
            try:
                speech.interrupt(force=force)
            except RuntimeError:
                # the speeches behind this one are going to play, so stopping
                # here keeps the conversation contiguous
                logger.warning(
                    "a queued speech does not allow interruptions and will play after the "
                    "interruption, use interrupt(force=True) to interrupt it as well",
                    extra={"speech_id": speech.id},
                )
                break

            interrupted_speeches.append(speech)

        if not interrupted_speeches:
            future.set_result(None)
        else:

            def on_playout_done(_: SpeechHandle) -> None:
                if not future.done() and all(speech.done() for speech in interrupted_speeches):
                    future.set_result(None)

            for speech in interrupted_speeches:
                speech.add_done_callback(on_playout_done)

        return future

    def _start_realtime_user_activity(self) -> None:
        if (
            self._rt_session is None
            or self._realtime_input_mode != "audio"
            or self._new_turns_blocked
        ):
            return

        if self._realtime_turn.policy.input_owner == "provider":
            # Activity notification is a public provider hook, not an ownership transfer.
            # Provider-owned turns stay outside framework seal/defer/advance state.
            self._rt_session.start_user_activity()
            return

        if self._realtime_turn.turn_complete:
            self._get_deferred_realtime_audio_input().provider_activity_started = True
            return
        if self._realtime_turn.provider_activity_started:
            return

        # Set the latch before calling the provider so a synchronous callback cannot start the
        # same logical activity twice. Roll it back if the provider rejects the notification.
        self._realtime_turn.provider_activity_started = True
        try:
            self._rt_session.start_user_activity()
        except BaseException:
            self._realtime_turn.provider_activity_started = False
            raise

    def _get_deferred_realtime_audio_input(self) -> _RealtimeTurnTransaction:
        if not self._deferred_realtime_turns or self._deferred_realtime_turns[-1].turn_complete:
            transaction = self._new_realtime_turn()
            transaction.ready_fut = asyncio.get_running_loop().create_future()
            self._deferred_realtime_turns.append(transaction)
        return self._deferred_realtime_turns[-1]

    def _seal_realtime_audio_input(self) -> asyncio.Future[_RealtimeTurnTransaction]:
        if not self._realtime_turn.turn_complete:
            self._realtime_turn.state = "sealed"
            ready_fut = asyncio.get_running_loop().create_future()
            ready_fut.set_result(self._realtime_turn)
            self._realtime_turn.ready_fut = ready_fut
            return ready_fut

        deferred = self._get_deferred_realtime_audio_input()
        deferred.state = "sealed"
        assert deferred.ready_fut is not None
        return deferred.ready_fut

    def _advance_realtime_audio_input(self) -> None:
        previous = self._realtime_turn
        previous.provider_activity_started = False
        previous.state = "settled"
        if self._rt_session is None or not self._deferred_realtime_turns:
            self._realtime_turn = self._new_realtime_turn()
            return

        deferred = self._deferred_realtime_turns.popleft()
        self._realtime_turn = deferred
        ready_fut = deferred.ready_fut
        assert ready_fut is not None
        try:
            if deferred.provider_activity_started:
                # Advance to the already-observed next VAD turn only after the previous
                # provider input was completed or discarded.
                self._rt_session.start_user_activity()
            # Audio reaches this path before external VAD can report activity. Preserve those
            # frames just as the unsealed path forwards frames before start_user_activity().
            for frame in deferred.frames:
                self._rt_session.push_audio(frame)
            if not ready_fut.done():
                ready_fut.set_result(deferred)
        except BaseException as e:
            try:
                self._rt_session.clear_audio()
            except BaseException:
                logger.exception("failed to clear realtime input after deferred replay failure")

            deferred.disposition = "abandoned"
            deferred.state = "settled"
            self._realtime_turn = self._new_realtime_turn()
            if not ready_fut.done():
                ready_fut.set_exception(e)
                # Replay can fail before EOU owns this future. Mark it observed while retaining
                # the same exception for any waiter that already captured it.
                ready_fut.exception()
            self._abort_deferred_realtime_audio()
            raise

    def _abort_deferred_realtime_audio(self) -> None:
        while self._deferred_realtime_turns:
            deferred = self._deferred_realtime_turns.popleft()
            deferred.disposition = "abandoned"
            deferred.state = "settled"
            assert deferred.ready_fut is not None
            if not deferred.ready_fut.done():
                deferred.ready_fut.cancel()

    def _reset_realtime_audio_input_state(self) -> None:
        self._realtime_turn.disposition = "abandoned"
        self._realtime_turn.state = "settled"
        self._realtime_turn = self._new_realtime_turn()
        self._abort_deferred_realtime_audio()

    def _discard_latest_realtime_audio_input(self) -> None:
        """Discard the newest logical input without disturbing older provider-current audio."""

        if self._deferred_realtime_turns:
            deferred = self._deferred_realtime_turns[-1]
            if deferred.policy.input_owner != "framework":
                return
            if deferred.input_submitted:
                return
            self._deferred_realtime_turns.pop()
            deferred.disposition = "abandoned"
            deferred.state = "settled"
            assert deferred.ready_fut is not None
            if not deferred.ready_fut.done():
                deferred.ready_fut.cancel()
            return
        if (
            self._realtime_turn.policy.input_owner != "framework"
            or self._realtime_turn.input_submitted
        ):
            return
        self._clear_realtime_input()

    def _clear_realtime_input(self, *, advance_deferred: bool = True) -> None:
        if self._rt_session is None:
            self._reset_realtime_audio_input_state()
            return

        clear_error: BaseException | None = None
        try:
            self._rt_session.clear_audio()
        except BaseException as e:
            clear_error = e

        self._realtime_turn.disposition = "abandoned"
        try:
            if advance_deferred and self._realtime_input_mode == "audio":
                self._advance_realtime_audio_input()
            else:
                self._realtime_turn.state = "settled"
                self._realtime_turn = self._new_realtime_turn()
        except BaseException:
            if clear_error is None:
                raise
            logger.exception("failed to advance deferred input after clear_audio failed")

        if clear_error is not None:
            raise clear_error

    def _clear_realtime_input_if_owned(self, input_owner: _RealtimeAudioInputOwner | None) -> None:
        if input_owner is None:
            return

        input_token: _RealtimeTurnTransaction
        if isinstance(input_owner, asyncio.Future):
            if not input_owner.done():
                for deferred in tuple(self._deferred_realtime_turns):
                    if deferred.ready_fut is input_owner:
                        self._deferred_realtime_turns.remove(deferred)
                        deferred.disposition = "abandoned"
                        deferred.state = "settled"
                        input_owner.cancel()
                        return
                return
            try:
                input_token = input_owner.result()
            except BaseException:
                return
        else:
            input_token = input_owner

        if input_token.policy.input_owner != "framework":
            return
        if input_token is not self._realtime_turn:
            return
        self._clear_realtime_input()

    def clear_user_turn(self) -> None:
        if self._audio_recognition:
            self._audio_recognition._clear_user_turn()

        if self._rt_session is None or self._realtime_input_mode != "audio":
            return

        if self._realtime_turn.policy.input_owner == "provider":
            # Explicit caller intent is delegated to the current input owner without entering
            # framework transaction, deferred-input, or advancement machinery.
            self._rt_session.clear_audio()
        else:
            self._discard_latest_realtime_audio_input()

    def commit_user_turn(
        self, *, transcript_timeout: float, stt_flush_duration: float, skip_reply: bool = False
    ) -> asyncio.Future[str]:
        reply_already_triggered = False
        if self._rt_session is not None and self._realtime_input_mode == "audio":
            if self._turn_policy.input_owner == "provider":
                # Manual commit is an explicit provider boundary even when the provider owns
                # detection. It never enters framework seal/defer/clear machinery.
                self._rt_session.commit_audio()
                if not skip_reply:
                    self._session.generate_reply(input_modality="audio")
                reply_already_triggered = True
            elif skip_reply:
                # Do not make deliberately skipped input model-visible. This also handles the
                # no-STT case, where no end-of-turn callback will arrive to discard the buffer.
                # Mark the provider disposition complete so a later external-STT callback keeps
                # the transcript but does not claim another realtime audio input.
                self._discard_latest_realtime_audio_input()
                reply_already_triggered = True
            else:
                # Commit the provider audio and trigger the reply before waiting for an external
                # STT flush. The EOT marker below prevents a second trigger without overloading
                # the caller's skip_reply intent. A consecutive turn stays bound to its deferred
                # input until the preceding generation releases the provider buffer.
                self._start_realtime_user_activity()
                input_ready_fut = self._seal_realtime_audio_input()
                try:
                    if input_ready_fut.done():
                        # The current input is ready, so preserve the established synchronous
                        # manual-commit behavior and let AgentSession track the returned handle.
                        transaction = input_ready_fut.result()
                        transaction.state = "input_submitted"
                        self._rt_session.commit_audio()
                        transaction.state = "generation_pending"
                        handle = self._session.generate_reply(input_modality="audio")
                        if isinstance(handle, SpeechHandle):
                            transaction.speech_handle = handle
                    else:
                        # A reply handle must exist now so interruption/cancellation can discard
                        # this exact deferred input before it becomes provider-visible.
                        handle = self._generate_reply(
                            input_details=InputDetails(modality="audio"),
                            realtime_audio_input_owner=input_ready_fut,
                            commit_realtime_audio=True,
                        )
                        for transaction in self._deferred_realtime_turns:
                            if transaction.ready_fut is input_ready_fut:
                                transaction.speech_handle = handle
                                break
                        if run_state := self._session._global_run_state:
                            run_state._watch_handle(handle)
                except BaseException:
                    self._clear_realtime_input_if_owned(input_ready_fut)
                    raise
                reply_already_triggered = True

        assert self._audio_recognition is not None
        return self._audio_recognition._commit_user_turn(
            audio_detached=not self._session.input.audio_enabled,
            transcript_timeout=transcript_timeout,
            stt_flush_duration=stt_flush_duration,
            skip_reply=skip_reply,
            reply_already_triggered=reply_already_triggered,
        )

    def _schedule_speech(self, speech: SpeechHandle, priority: int, force: bool = False) -> None:
        # when force=True, we still allow to schedule a new speech even if
        # `pause_speech_scheduling` is waiting for the schedule_task to drain.
        # This allows for tool responses to be generated before the AgentActivity is finalized.

        if self._scheduling_paused and not force:
            speech.interrupt(force=True)
            raise RuntimeError(
                "cannot schedule new speech, the speech scheduling is draining/pausing, the speech will be cancelled"
            )

        if self._scheduling_atask and self._scheduling_atask.done():
            logger.warning(
                "attempting to schedule a new SpeechHandle, but the scheduling_task is not running, the speech will be cancelled"
            )
            speech.interrupt(force=True)
            return

        while True:
            try:
                # negate the priority to make it a max heap
                heapq.heappush(self._speech_q, (-priority, time.perf_counter_ns(), speech))
                break
            except TypeError:
                # handle TypeError when identical timestamps cause speech comparison failure
                # with perf_counter_ns(), collisions should be rare
                pass

        speech._mark_scheduled()
        self._wake_up_scheduling_task()

    @utils.log_exceptions(logger=logger)
    async def _scheduling_task(self) -> None:
        last_playout_ts = 0.0
        while True:
            await self._q_updated.wait()
            self._q_updated.clear()

            while self._speech_q:
                _, _, speech = heapq.heappop(self._speech_q)
                if speech.done():
                    # skip done speech (interrupted when it's in the queue)
                    self._current_speech = None
                    continue
                self._current_speech = speech
                if self.min_consecutive_speech_delay > 0.0:
                    delay = self.min_consecutive_speech_delay - (time.time() - last_playout_ts)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    # check again if speech is done after sleep delay
                    if speech.done():
                        # skip done speech (interrupted during delay)
                        self._current_speech = None
                        continue
                speech._authorize_generation()
                await speech._wait_for_generation()

                if self._paused_speech and self._paused_speech.handle is self._current_speech:
                    # clear paused speech after generation done
                    self._paused_speech = None
                    self._cancel_false_interruption_timer()
                    if (audio_output := self._session.output.audio) and audio_output.can_pause:
                        audio_output.resume()
                self._current_speech = None
                last_playout_ts = time.time()

            # if we're draining/pasuing and there are no more speech tasks, we can exit.
            # only speech tasks can bypass draining to create a tool response (see `_schedule_speech`)  # noqa: E501

            blocked_handles: list[SpeechHandle] = []
            for task in self._drain_blocked_tasks:
                info = _get_activity_task_info(task)
                if not info:
                    logger.error("blocked task without activity info; skipping.")
                    continue

                if not info.speech_handle:
                    continue  # on_enter/on_exit

                blocked_handles.append(info.speech_handle)

            to_wait: list[asyncio.Task] = []
            for task in self._speech_tasks:
                if task in self._drain_blocked_tasks:
                    continue

                info = _get_activity_task_info(task)
                if info and info.speech_handle in blocked_handles:
                    continue

                to_wait.append(task)

            if self._scheduling_paused and len(to_wait) == 0:
                break

    async def wait_for_idle(
        self, *, wait_for_agent: bool = True, wait_for_user: bool = True
    ) -> None:
        """Wait until this activity has no in-flight agent or user work.

        Raises ``ActivityClosedError`` if the activity has terminally closed.
        """
        agent_active = True
        user_active = True

        async def _wait_for_eou() -> None:
            # eou is part of the user turn and may spawn a new speech handle,
            # so an in-flight eou keeps both the user and the agent active.
            nonlocal user_active
            if (
                self._audio_recognition
                and (eou_task := self._audio_recognition._end_of_turn_task)
                and not eou_task.done()
            ):
                user_active = True
                await asyncio.shield(eou_task)

            if (user_turn_task := self._user_turn_completed_atask) and not user_turn_task.done():
                user_active = True
                await asyncio.shield(user_turn_task)

        while (wait_for_agent and agent_active) or (wait_for_user and user_active):
            if self._closed or self._session._closing:
                raise ActivityClosedError(f"activity {self.agent.label} is closing")

            if wait_for_agent:
                await _wait_for_eou()

                if self._current_speech is None and not self._speech_q:
                    agent_active = False
                else:
                    agent_active = True
                    if (speech := self._current_speech) and speech._generations:
                        await speech._wait_for_generation()
                    await asyncio.sleep(0)

            if wait_for_user:
                if self._audio_recognition and self._audio_recognition._speaking:
                    user_active = True
                    await self._audio_recognition._wait_for_user_silence()
                else:
                    user_active = False

                await _wait_for_eou()

            if self._session._user_turn_claims > 0:
                # `AgentSession.claim_user_turn` is holding idle open
                await self._session._user_turn_released.wait()
                agent_active = wait_for_agent
                user_active = wait_for_user

            if self._session._idle_holds > 0 and not _IdleHoldContextVar.get():
                # another caller holds `_wait_for_idle_and_hold` — block until release
                await self._session._idle_released.wait()
                agent_active = wait_for_agent
                user_active = wait_for_user

        if self._closed or self._session._closing:
            raise ActivityClosedError(f"activity {self.agent.label} is closing")

    # -- Realtime Session events --

    def _on_metrics_collected(
        self,
        ev: STTMetrics | TTSMetrics | VADMetrics | LLMMetrics | RealtimeModelMetrics,
    ) -> None:
        if (speech_handle := _SpeechHandleContextVar.get(None)) and (
            isinstance(ev, LLMMetrics) or isinstance(ev, TTSMetrics)
        ):
            ev.speech_id = speech_handle.id
        if (
            isinstance(ev, RealtimeModelMetrics)
            and self._realtime_spans is not None
            and (realtime_span := self._realtime_spans.pop(ev.request_id, None))
        ):
            trace_utils.record_realtime_metrics(realtime_span, ev)
        self._session._usage_collector.collect(ev)
        otel_metrics.collect_usage(ev)
        self._session.emit("metrics_collected", MetricsCollectedEvent(metrics=ev))
        self._session.emit(
            "session_usage_updated",
            SessionUsageUpdatedEvent(usage=self._session.usage),
        )

    def _on_remote_item_added(self, ev: llm.RemoteItemAddedEvent) -> None:
        # A provider may echo a client context update before update_chat_ctx() resolves. The
        # authoritative finalized item is committed (and emitted) after that update is confirmed.
        if ev.item.id in self._pending_realtime_user_message_ids:
            return

        # add the remote item to the local chat context as a placeholder
        local_chat_ctx = self._agent._chat_ctx
        if local_chat_ctx.get_by_id(ev.item.id) is not None:
            return

        # only add placeholders for server-initiated items (responses, function calls),
        # which always append at the end of the conversation. client-initiated items
        # (from update_chat_ctx) already exist in _agent._chat_ctx and go local→remote,
        # so they don't need placeholders.
        last_item_id = local_chat_ctx.items[-1].id if local_chat_ctx.items else None
        if ev.previous_item_id is None or ev.previous_item_id == last_item_id:
            local_chat_ctx.items.append(ev.item.model_copy())

    def _on_error(
        self,
        error: llm.LLMError
        | stt.STTError
        | tts.TTSError
        | llm.RealtimeModelError
        | inference.InterruptionDetectionError,
    ) -> None:
        if isinstance(error, llm.LLMError):
            error_event = ErrorEvent(error=error, source=self.llm)
            self._session.emit("error", error_event)
        elif isinstance(error, llm.RealtimeModelError):
            error_event = ErrorEvent(error=error, source=self.llm)
            self._session.emit("error", error_event)
        elif isinstance(error, stt.STTError):
            error_event = ErrorEvent(error=error, source=self.stt)
            self._session.emit("error", error_event)
        elif isinstance(error, tts.TTSError):
            error_event = ErrorEvent(error=error, source=self.tts)
            self._session.emit("error", error_event)
        elif isinstance(error, inference.InterruptionDetectionError):
            if not error.recoverable:
                self._fallback_to_vad_interruption(error)
            return

        self._session._on_error(error)

    def _on_overlap_speech_ended(self, ev: inference.OverlappingSpeechEvent) -> None:
        self._interruption_detected = ev.is_interruption
        self._session.emit("overlapping_speech", ev)

    def _on_input_speech_started(self, _: llm.InputSpeechStartedEvent) -> None:
        if self.vad is None or self.using_default_vad:
            self._session._update_user_state("speaking")
            if self._audio_recognition:
                self._audio_recognition._on_start_of_speech(
                    started_at=time.time(),
                    user_speaking_span=self._session._user_speaking_span,
                )

        try:
            self.interrupt()  # input_speech_started is also interrupting on the serverside realtime session  # noqa: E501
        except RuntimeError:
            # only out of sync when the server cancelled its own response, with client-side turn
            # taking an uninterruptible speech is expected
            if self._rt_turn_detection_enabled:
                logger.exception(
                    "RealtimeAPI input_speech_started, but current speech is not interruptable, this should never happen!"  # noqa: E501
                )

    def _on_input_speech_stopped(self, ev: llm.InputSpeechStoppedEvent) -> None:
        if self.vad is None or self.using_default_vad:
            if self._audio_recognition:
                self._audio_recognition._on_end_of_speech(
                    ended_at=time.time(),
                    user_speaking_span=self._session._user_speaking_span,
                )

            self._session._update_user_state("listening")

        if ev.user_transcription_enabled:
            self._session._user_input_transcribed(
                UserInputTranscribedEvent(transcript="", is_final=False)
            )

    def _on_input_audio_transcription_completed(self, ev: llm.InputTranscriptionCompleted) -> None:
        self._session._user_input_transcribed(
            UserInputTranscribedEvent(
                transcript=ev.transcript,
                is_final=ev.is_final,
                item_id=ev.item_id,
            )
        )

        if ev.is_final:
            if self.stt is None and ev.transcript and (amd := self._session._amd) is not None:
                amd._on_transcript(ev.transcript)

            msg = llm.ChatMessage(role="user", content=[ev.transcript], id=ev.item_id)
            if ev.turn_started_at is not None:
                # a provider may withhold the final transcript until its reply has finished
                # generating, which would otherwise stamp the turn after the reply it prompted
                msg.created_at = ev.turn_started_at
                msg.metrics = {"started_speaking_at": ev.turn_started_at}
            if self._discard_matching_bounded_close_user_message(msg):
                return
            self._provider_transcription_item_ids.add(msg.id)
            self._agent._chat_ctx._upsert_item(msg)
            self._session._conversation_item_added(msg)

    def _on_generation_created(self, ev: llm.GenerationCreatedEvent) -> None:
        self._active_realtime_generation = ev
        if ev.user_initiated:
            # user_initiated generations are directly handled inside _realtime_reply_task
            return

        if self._scheduling_paused or self._new_turns_blocked:
            # TODO(theomonnom): should we "forward" this new turn to the next agent?
            logger.warning("skipping new realtime generation, the speech scheduling is not running")
            return

        handle = SpeechHandle.create(
            allow_interruptions=self.allow_interruptions,
            input_details=InputDetails(modality="audio"),
        )
        self._session.emit(
            "speech_created",
            SpeechCreatedEvent(speech_handle=handle, user_initiated=False, source="generate_reply"),
        )

        self._create_speech_task(
            self._realtime_generation_task(
                speech_handle=handle,
                generation_ev=ev,
                model_settings=ModelSettings(),
            ),
            speech_handle=handle,
            name="AgentActivity.realtime_generation",
        )

        if (fut := self._pending_auto_tool_reply_fut) and not fut.done():
            if (run_state := self._session._global_run_state) is not None and not run_state.done():
                run_state._watch_handle(handle)
            self._pending_auto_tool_reply_fut = None
            fut.set_result(None)

        self._schedule_speech(handle, SpeechHandle.SPEECH_PRIORITY_NORMAL)

    def _interrupt_by_audio_activity(
        self, *, ignore_user_transcript_until: float | None = None
    ) -> None:
        """
        Interrupt the current speech or generation, and optionally ignore the user transcript until the given timestamp.

        Args:
            ignore_user_transcript_until: The timestamp until which the user transcript should be ignored.
                If None, the user transcript will be ignored until the current time.
        """
        if not self._interruption_by_audio_activity_enabled:
            return

        if self._session._aec_warmup_remaining > 0 and self._session._aec_warmup_timer is not None:
            # disable interruption from audio activity while aec warmup is active
            return

        if self._rt_turn_detection_enabled:
            # ignore if realtime model has turn detection enabled
            return

        interruption_options = self._session.options.interruption
        if (
            self.stt is not None
            and interruption_options["min_words"] > 0
            and self._audio_recognition is not None
        ):
            text = self._audio_recognition._current_transcript

            # TODO(long): better word splitting for multi-language
            if len(split_words(text, split_character=True)) < interruption_options["min_words"]:
                return

        self._start_realtime_user_activity()

        if (
            self._current_speech is not None
            and not self._current_speech.interrupted
            and self._current_speech.allow_interruptions
        ):
            # reset the false interruption timer
            self._cancel_false_interruption_timer()

            # only interrupt if not already interrupting
            if (
                self._audio_recognition
                and not self._audio_recognition._endpointing.overlapping
                and self._session.agent_state == "speaking"
            ):
                self._audio_recognition._on_start_of_speech(
                    started_at=time.time(),
                )

            if self._pause_enabled():
                assert (timeout := interruption_options["false_interruption_timeout"]) is not None
                assert (audio_output := self._session.output.audio) is not None

                # EOS arms false-interruption resume. A final transcript or a
                # replying turn commit interrupts the paused handle.
                self._update_paused_speech(self._current_speech, timeout)
                audio_output.pause()
                self._session._update_agent_state("listening")
                if self._audio_recognition:
                    self._audio_recognition._on_end_of_agent_speech(
                        ignore_user_transcript_until=ignore_user_transcript_until or time.time()
                    )
                if self.interruption_enabled:
                    self._restore_interruption_by_audio_activity()
            else:
                if self._rt_session is not None:
                    self._rt_session.interrupt()

                self._current_speech.interrupt()

    # region recognition hooks

    def on_start_of_speech(
        self,
        ev: vad.VADEvent | None,
        speech_start_time: float,
    ) -> None:
        self._session._update_user_state("speaking", last_speaking_time=speech_start_time)
        if self._audio_recognition:
            self._audio_recognition._on_start_of_speech(
                started_at=speech_start_time,
                speech_duration=ev.speech_duration if ev else 0.0,
                user_speaking_span=self._session._user_speaking_span,
            )
        self._user_silence_event.clear()
        self._stt_eos_received = False
        self._interruption_detected = False

        current_speech_active = (
            self._current_speech is not None
            and not self._current_speech.done()
            and not self._current_speech.interrupted
        )
        if not current_speech_active:
            self._start_realtime_user_activity()

        # cancel the timer when user starts speaking but leave the paused state unchanged
        self._cancel_false_interruption_timer()

        if (
            self._session.agent_state != "speaking"
            and self._pause_enabled()
            and (current_speech := self._current_speech) is not None
            and not current_speech.interrupted
            and current_speech.allow_interruptions
            and (self._paused_speech is None or self._paused_speech.handle is not current_speech)
        ):
            # EOS arms false-interruption resume. A final transcript or a
            # replying turn commit interrupts the paused handle.
            assert (audio_output := self._session.output.audio) is not None

            self._update_paused_speech(current_speech, timeout=0)
            audio_output.pause()

    def on_end_of_speech(self, ev: vad.VADEvent | None) -> None:
        speech_end_time = time.time()
        if ev:
            speech_end_time = speech_end_time - ev.silence_duration - ev.inference_duration
        else:
            self._stt_eos_received = True

        if self._audio_recognition:
            self._audio_recognition._on_end_of_speech(
                ended_at=speech_end_time,
                user_speaking_span=self._session._user_speaking_span,
                interruption=self._interruption_detected
                if self._interruption_detection_enabled
                else NOT_GIVEN,
            )

        self._session._update_user_state(
            "listening",
            last_speaking_time=speech_end_time,
        )
        self._user_silence_event.set()

        if self._paused_speech:
            self._start_false_interruption_timer(self._paused_speech.timeout)

    def on_transcription_timeout(self, *, speech_duration: float, turn_start: float) -> None:
        self._session.emit(
            "user_transcription_timeout",
            UserTranscriptionTimeoutEvent(
                speech_duration=speech_duration,
                vad_speech_started_at=turn_start,
            ),
        )

    def on_vad_inference_done(self, ev: vad.VADEvent) -> None:
        if self._turn_detection in ("manual", "realtime_llm"):
            # ignore vad inference done event if turn_detection is manual or realtime_llm
            return

        active_speech = ev.speech_duration >= self._session.options.interruption["min_duration"]
        if active_speech and (
            self._turn_detection != "stt"
            or not self._stt_eos_received
            or ev.raw_accumulated_silence == 0
        ):
            # STT may send EOS before VAD EOS, we only interrupt if:
            # 1. turn detection is not STT; or
            # 2. STT EOS hasn't been received yet; or
            # 3. VAD speech is still ongoing
            self._interrupt_by_audio_activity()

        if (
            ev.speaking
            # allow some silence between utterances during active speech
            and ev.raw_accumulated_silence <= self.min_endpointing_delay / 2
        ):
            self._user_silence_event.clear()
        else:
            self._user_silence_event.set()

    def on_backchannel_confirmed(self) -> None:
        # clear the buffered backchannel audio so it can't prefix the next committed turn
        if (
            self._interruption_detection_enabled
            and self._rt_session is not None
            and self._realtime_input_mode == "audio"
            and self._turn_detection not in ("manual", "realtime_llm")
        ):
            self._discard_latest_realtime_audio_input()

    def on_interruption(self, ev: inference.OverlappingSpeechEvent) -> None:
        # restore interruption by audio activity and then immediately interrupt
        self._restore_interruption_by_audio_activity()
        self._interrupt_by_audio_activity(
            ignore_user_transcript_until=ev.overlap_started_at or ev.detected_at
        )
        # flush held transcripts again if possible
        if self._audio_recognition:
            self._audio_recognition._on_end_of_agent_speech(
                ignore_user_transcript_until=ev.overlap_started_at or ev.detected_at
            )

    def on_interim_transcript(self, ev: stt.SpeechEvent, *, speaking: bool | None) -> None:
        if (
            self._realtime_input_mode == "audio"
            and isinstance(self.llm, llm.RealtimeModel)
            and self.llm.capabilities.user_transcription
        ):
            # skip stt transcription if user_transcription is enabled on the realtime model
            return

        self._session._user_input_transcribed(
            UserInputTranscribedEvent(
                language=ev.alternatives[0].language,
                transcript=ev.alternatives[0].text,
                is_final=False,
                speaker_id=ev.alternatives[0].speaker_id,
            ),
        )

        if (
            self.vad is None
            and ev.alternatives[0].text
            and self._turn_detection
            not in (
                "manual",
                "realtime_llm",
            )
        ):
            self._interrupt_by_audio_activity()

            if (
                speaking is False
                and self._paused_speech
                and (timeout := self._session.options.interruption["false_interruption_timeout"])
                is not None
            ):
                # schedule a resume timer if interrupted after end_of_speech
                self._start_false_interruption_timer(timeout)

    def on_final_transcript(self, ev: stt.SpeechEvent, *, speaking: bool | None = None) -> None:
        if (
            self._realtime_input_mode == "audio"
            and isinstance(self.llm, llm.RealtimeModel)
            and self.llm.capabilities.user_transcription
        ):
            # skip stt transcription if user_transcription is enabled on the realtime model
            return

        self._session._user_input_transcribed(
            UserInputTranscribedEvent(
                language=ev.alternatives[0].language,
                transcript=ev.alternatives[0].text,
                is_final=True,
                speaker_id=ev.alternatives[0].speaker_id,
            ),
        )
        # agent speech might not be interrupted if VAD failed and a final transcript is received
        # we call _interrupt_by_audio_activity (idempotent) to pause the speech, if possible
        # which will also be immediately interrupted

        if self._audio_recognition and self._turn_detection not in (
            "manual",
            "realtime_llm",
        ):
            self._interrupt_by_audio_activity()

            if (
                speaking is False
                and self._paused_speech
                and (timeout := self._session.options.interruption["false_interruption_timeout"])
                is not None
            ):
                # schedule a resume timer if interrupted after end_of_speech
                self._start_false_interruption_timer(timeout)

        self._cancel_speech_pause_task = asyncio.create_task(
            self._cancel_speech_pause(old_task=self._cancel_speech_pause_task)
        )

    def on_preemptive_generation(self, info: _PreemptiveGenerationInfo) -> None:
        preemptive_opts = self.preemptive_generation_opts
        if (
            not preemptive_opts["enabled"]
            or self._scheduling_paused
            or self._new_turns_blocked
            or (self._current_speech is not None and not self._current_speech.interrupted)
            or not isinstance(self.llm, llm.LLM)
        ):
            return

        self._cancel_preemptive_generation()

        if (
            info.started_speaking_at is not None
            and time.time() - info.started_speaking_at > preemptive_opts["max_speech_duration"]
        ):
            return

        if self._preemptive_generation_count >= preemptive_opts["max_retries"]:
            return

        self._preemptive_generation_count += 1

        user_message = llm.ChatMessage(
            role="user",
            content=[info.new_transcript],
            transcript_confidence=info.transcript_confidence,
        )

        chat_ctx = self._agent.chat_ctx.copy()
        speech_handle = self._generate_reply(
            # we need to send in the original user_message because metrics are injected later on
            user_message=user_message,
            chat_ctx=chat_ctx,
            schedule_speech=False,
            input_details=InputDetails(modality="audio"),
        )

        self._preemptive_generation = _PreemptiveGeneration(
            speech_handle=speech_handle,
            user_message=user_message,
            info=info,
            chat_ctx=chat_ctx.copy(),
            tools=self.tools.copy(),
            tool_choice=self._tool_choice,
            created_at=time.time(),
        )

    def on_eot_prediction(self, ev: EotPredictionEvent) -> None:
        if (host := self._session._session_host) is not None:
            host._on_eot_prediction(ev)

    def on_agent_backchannel_opportunity(self, ev: _AgentBackchannelOpportunityEvent) -> None:
        # TODO: consume the backchannel opportunity internally (e.g. trigger a
        # backchannel phrase). Kept internal for now — not surfaced as a public event.
        pass

    def on_end_of_turn(self, info: _EndOfTurnInfo) -> bool:
        # IMPORTANT: This method is sync to avoid it being cancelled by the AudioRecognition
        # We explicitly create a new task here

        # TODO: @chenghao-mou replace this direct call with the public `eot_prediction`
        # event once feat/AGT-2520-multimodal-EOU lands.
        if (amd := self._session._amd) is not None and amd._on_end_of_turn(info):
            # cancel post-verdict preemptive and new generations
            self._cancel_preemptive_generation()
            info.skip_reply = True

        if self._scheduling_paused or self._new_turns_blocked:
            self._cancel_preemptive_generation()
            if (
                not info.reply_already_triggered
                and self._rt_session is not None
                and self._realtime_input_mode == "audio"
            ):
                # External activity may already have streamed provider audio. Discard it when
                # this turn cannot be scheduled so it cannot prefix the next turn.
                self._discard_latest_realtime_audio_input()
            logger.warning(
                "skipping user input, speech scheduling is paused",
                extra={"lk.pii.user_input": info.new_transcript},
            )

            if self._session._closing:
                user_message = llm.ChatMessage(
                    role="user",
                    content=[info.new_transcript],
                    transcript_confidence=info.transcript_confidence,
                )
                user_message.metrics = self._init_metrics_from_end_of_turn(info)
                self._commit_bounded_user_message_locally(
                    user_message,
                    provider_reply_already_triggered=info.reply_already_triggered,
                )

            # TODO(theomonnom): should we "forward" this new turn to the next agent/activity?
            return True

        # Minimum-word retention only applies when STT produced text to retain. An authoritative
        # empty timeout must settle buffered realtime audio instead of leaving it open.
        if (
            self.stt is not None
            and self._turn_detection != "manual"
            and self._current_speech is not None
            and self._current_speech.allow_interruptions
            and not self._current_speech.interrupted
            and self._session.options.interruption["min_words"] > 0
            and bool(info.new_transcript.strip())
            and len(split_words(info.new_transcript, split_character=True))
            < self._session.options.interruption["min_words"]
        ):
            self._cancel_preemptive_generation()
            # AudioRecognition retains this transcript for the next endpointing verdict.
            # Keep the matching provider audio under the same logical input owner as well.
            return False

        # avoid interruption if backchannel is detected with realtime model. an unjudged overlap
        # commits: interrupting the agent is recoverable, discarding a real user turn is not
        if (
            self.stt is None
            and self._turn_detection != "manual"
            and isinstance(self.llm, llm.RealtimeModel)
            and not self._rt_turn_detection_enabled
            and self._interruption_detection_enabled
            and info.backchannel_over_agent
        ):
            logger.debug("skipping user input, realtime backchannel detected")
            self._cancel_preemptive_generation()
            # no transcript to gatekeep for realtime barge-in — drop the backchannel turn
            # and clear the buffered audio so it can't leak into the next committed turn
            if not info.reply_already_triggered and self._rt_session is not None:
                self._discard_latest_realtime_audio_input()
            return False

        # a replying turn interrupts the paused speech, so cancel the resume that would race it —
        # but the reply task returns before that in these two cases, so leave the resume armed
        if not info.skip_reply and not self._rt_turn_detection_enabled:
            self._cancel_false_interruption_timer()

        audio_input_ready_fut: asyncio.Future[_RealtimeTurnTransaction] | None = None
        if (
            self._rt_session is not None
            and self._realtime_input_mode == "audio"
            and not self._rt_turn_detection_enabled
            and not info.reply_already_triggered
        ):
            audio_input_ready_fut = self._seal_realtime_audio_input()

        old_task = self._user_turn_completed_atask
        self._user_turn_completed_atask = self._create_speech_task(
            self._user_turn_completed_task(old_task, info, audio_input_ready_fut),
            name="AgentActivity._user_turn_completed_task",
        )
        return True

    async def _interrupt_current_speech_for_user_turn(
        self,
        *,
        user_input: str,
        audio_input_token: _RealtimeTurnTransaction | None,
    ) -> bool:
        current_speech = self._current_speech
        if current_speech is None:
            return True
        if not current_speech.allow_interruptions:
            logger.warning(
                "skipping reply to user input, current speech generation cannot be interrupted",
                extra={"lk.pii.user_input": user_input},
            )
            if self._rt_session is not None and self._realtime_input_mode == "audio":
                self._clear_realtime_input_if_owned(audio_input_token)
            return False

        await self._cancel_speech_pause(self._cancel_speech_pause_task)
        await current_speech.interrupt()

        if self._rt_session is not None:
            self._rt_session.interrupt()
        return True

    @utils.log_exceptions(logger=logger)
    async def _user_turn_completed_task(
        self,
        old_task: asyncio.Task[None] | None,
        info: _EndOfTurnInfo,
        audio_input_ready_fut: asyncio.Future[_RealtimeTurnTransaction] | None = None,
    ) -> None:
        user_message = llm.ChatMessage(
            role="user",
            content=[info.new_transcript],
            transcript_confidence=info.transcript_confidence,
        )
        bounded_user_message = user_message
        metrics_report: llm.MetricsReport = self._init_metrics_from_end_of_turn(info)
        user_message.metrics = metrics_report

        audio_input_token: _RealtimeTurnTransaction | None = None
        try:
            if old_task is not None:
                # Wait without propagating cancellation between predecessor and successor.
                await asyncio.wait({old_task})
                if not old_task.cancelled():
                    old_task.result()

            if audio_input_ready_fut is not None:
                audio_input_token = await asyncio.shield(audio_input_ready_fut)
        except asyncio.CancelledError:
            if self._session._closing and self._new_turns_blocked:
                self._commit_bounded_user_message_locally(
                    bounded_user_message,
                    provider_reply_already_triggered=info.reply_already_triggered,
                )
            self._clear_realtime_input_if_owned(audio_input_ready_fut)
            raise
        except BaseException:
            self._clear_realtime_input_if_owned(audio_input_ready_fut)
            raise
        self._preemptive_generation_count = 0
        defer_speech_interruption = (
            self._realtime_input_mode == "text" and not info.new_transcript.strip()
        )

        # When the audio recognition detects the end of a user turn:
        #  - check if realtime model server-side turn detection is enabled
        #  - check if there is no current generation happening
        #  - cancel the current generation if it allows interruptions (otherwise skip this current
        #  turn)
        #  - generate a reply to the user input

        # Empty realtime text turns first give user code a chance to supply deliberate model
        # input. Until then, they must not interrupt a valid response or background speech.

        # A turn that was already bounded before close/handoff remains locally observable, but
        # the shutdown boundary must not submit input or begin new provider work.
        if self._scheduling_paused or self._new_turns_blocked:
            logger.warning(
                "skipping on_user_turn_completed, speech scheduling is paused",
                extra={"lk.pii.user_input": info.new_transcript},
            )
            if self._session._closing:
                self._commit_bounded_user_message_locally(
                    user_message,
                    provider_reply_already_triggered=info.reply_already_triggered,
                )
            if self._rt_session is not None and self._realtime_input_mode == "audio":
                self._clear_realtime_input_if_owned(audio_input_token)
            return

        if not defer_speech_interruption:
            await asyncio.gather(*self._interrupt_background_speeches(force=False))

        if isinstance(self.llm, llm.RealtimeModel):
            if self._rt_turn_detection_enabled:
                return

            if self._rt_session is not None:
                if info.skip_reply:
                    self._rt_session._exclude_chat_ctx_item_from_replay(user_message.id)
                    if self._realtime_input_mode == "audio":
                        self._clear_realtime_input_if_owned(audio_input_token)
                    # A skipped finalized transcript remains observable in local conversation
                    # history even when the explicit text-input mode owns no provider audio.
                    if info.new_transcript != "":
                        self._commit_user_message_locally(user_message)
                    return
                if info.reply_already_triggered:
                    # Manual audio commit already started exactly one provider generation. Keep
                    # the finalized external transcript for local observability, but do not touch
                    # the provider input or create a duplicate reply.
                    if (
                        self._realtime_input_mode == "audio"
                        and not self.llm.capabilities.user_transcription
                        and info.new_transcript != ""
                    ):
                        self._commit_user_message_locally(user_message)
                    return
                if self._realtime_input_mode == "audio":
                    if audio_input_token is not self._rt_audio_input_token:
                        return

        if info.skip_reply:
            if info.new_transcript != "":
                self._commit_user_message_locally(user_message)
            return

        if not defer_speech_interruption and not await self._interrupt_current_speech_for_user_turn(
            user_input=info.new_transcript,
            audio_input_token=audio_input_token,
        ):
            return

        # create a temporary mutable chat context to pass to on_user_turn_completed
        # the user can edit it for the current generation, but changes will not be kept inside the
        # Agent.chat_ctx
        temp_mutable_chat_ctx = self._agent.chat_ctx.copy()
        start_time = time.perf_counter()
        try:
            await self._agent.on_user_turn_completed(
                temp_mutable_chat_ctx, new_message=user_message
            )
        except asyncio.CancelledError:
            if self._session._closing and self._new_turns_blocked:
                self._commit_bounded_user_message_locally(
                    bounded_user_message,
                    provider_reply_already_triggered=info.reply_already_triggered,
                )
            if self._rt_session is not None and self._realtime_input_mode == "audio":
                self._clear_realtime_input_if_owned(audio_input_token)
            raise
        except StopResponse:
            if self._rt_session is not None and self._realtime_input_mode == "audio":
                self._clear_realtime_input_if_owned(audio_input_token)
            return  # ignore this turn
        except Exception:
            logger.exception("error occurred during on_user_turn_completed")
            if self._rt_session is not None and self._realtime_input_mode == "audio":
                self._clear_realtime_input_if_owned(audio_input_token)
            return

        on_user_turn_completed_delay = time.perf_counter() - start_time
        metrics_report["on_user_turn_completed_delay"] = on_user_turn_completed_delay

        if isinstance(self.llm, llm.RealtimeModel):
            if self._realtime_input_mode == "audio":
                # Native realtime audio remains the model input in the default mode.
                user_message = None  # type: ignore
            elif not (user_message.raw_text_content or "").strip():
                # No model-consumable input exists in text mode. The callback above still had
                # an opportunity to replace an empty STT result with deliberate text.
                return
        elif self.llm is None:
            return  # skip response if no llm is set

        if self._scheduling_paused or self._new_turns_blocked:
            logger.warning(
                "skipping reply to user input, speech scheduling is paused",
                extra={"lk.pii.user_input": info.new_transcript},
            )
            if self._session._closing:
                self._commit_bounded_user_message_locally(
                    bounded_user_message,
                    provider_reply_already_triggered=info.reply_already_triggered,
                )
            if self._rt_session is not None and self._realtime_input_mode == "audio":
                self._clear_realtime_input_if_owned(audio_input_token)
            return

        if defer_speech_interruption:
            await asyncio.gather(*self._interrupt_background_speeches(force=False))
            assert user_message is not None
            if not await self._interrupt_current_speech_for_user_turn(
                user_input=user_message.raw_text_content or "",
                audio_input_token=audio_input_token,
            ):
                return
            if self._scheduling_paused or self._new_turns_blocked:
                logger.warning(
                    "skipping reply to user input, speech scheduling is paused",
                    extra={"lk.pii.user_input": user_message.raw_text_content},
                )
                if self._session._closing:
                    self._commit_bounded_user_message_locally(
                        user_message,
                        provider_reply_already_triggered=info.reply_already_triggered,
                    )
                return

        speech_handle: SpeechHandle | None = None
        if preemptive := self._preemptive_generation:
            # make sure the on_user_turn_completed didn't change some request parameters
            # otherwise invalidate the preemptive generation
            if (
                _transcripts_equivalent(
                    preemptive.info.new_transcript, user_message.raw_text_content
                )
                and preemptive.chat_ctx.is_equivalent(temp_mutable_chat_ctx)
                and preemptive.tools == self.tools
                and preemptive.tool_choice == self._tool_choice
            ):
                speech_handle = preemptive.speech_handle

                # The pipeline task retains the ChatMessage created for preemptive generation.
                # Reconcile it with the finalized message before scheduling so conversation
                # history keeps the final transcript and on_user_turn_completed edits.
                preemptive.user_message.content = user_message.content.copy()
                preemptive.user_message.transcript_confidence = user_message.transcript_confidence
                preemptive.user_message.metrics = metrics_report
                self._schedule_speech(speech_handle, priority=SpeechHandle.SPEECH_PRIORITY_NORMAL)
                logger.debug(
                    "using preemptive generation",
                    extra={"preemptive_lead_time": time.time() - preemptive.created_at},
                )
            else:
                logger.warning(
                    "preemptive generation invalidated after `on_user_turn_completed` because "
                    "the transcript, chat context, tools, or tool choice changed",
                )
                preemptive.speech_handle._cancel()

            self._preemptive_generation = None

        if speech_handle is None:
            # Ensure the new message is passed to generate_reply
            # This preserves the original message_id, making it easier for users to track responses
            speech_handle = self._generate_reply(
                user_message=user_message,
                chat_ctx=temp_mutable_chat_ctx,
                input_details=InputDetails(
                    modality="text" if self._realtime_input_mode == "text" else "audio"
                ),
                realtime_audio_input_owner=audio_input_token,
                commit_realtime_audio=(
                    self._realtime_input_mode == "audio" and audio_input_token is not None
                ),
            )

        if self._user_turn_completed_atask != asyncio.current_task():
            # If a new user turn has already started, interrupt this one since it's now outdated
            # (We still create the SpeechHandle and the generate_reply coroutine, otherwise we may
            # lose data like the beginning of a user speech).
            # await the interrupt to make sure user message is added to the chat context before the new task starts
            await speech_handle.interrupt()

        metadata: Metadata | None = None
        if isinstance(self._turn_detection, str):
            metadata = Metadata(model_name="unknown", model_provider=self._turn_detection)
        elif self._turn_detection is not None:
            metadata = Metadata(
                model_name=self._turn_detection.model, model_provider=self._turn_detection.provider
            )

        eou_metrics = EOUMetrics(
            timestamp=time.time(),
            end_of_utterance_delay=info.metrics.end_of_turn_delay or 0.0,
            transcription_delay=info.metrics.transcription_delay or 0.0,
            on_user_turn_completed_delay=on_user_turn_completed_delay,
            speech_id=speech_handle.id,
            metadata=metadata,
        )
        self._session.emit("metrics_collected", MetricsCollectedEvent(metrics=eou_metrics))

    def on_user_turn_exceeded(self, ev: UserTurnExceededEvent) -> None:
        if self._scheduling_paused or self._new_turns_blocked:
            logger.warning(
                "skipping user turn exceeded, speech scheduling is paused",
                extra={
                    "num_words": ev.accumulated_word_count,
                    "duration": ev.duration,
                },
            )
            return

        if self._user_turn_exceeded_locked:
            return  # user callback is executing, drop

        # cancel previous wait phase (if still waiting for EOU result)
        if self._user_turn_exceeded_atask is not None:
            self._user_turn_exceeded_atask.cancel()

        self._user_turn_exceeded_atask = self._create_speech_task(
            self._user_turn_exceeded_task(ev),
            name="AgentActivity._user_turn_exceeded_task",
        )

    @utils.log_exceptions(logger=logger)
    async def _user_turn_exceeded_task(self, ev: UserTurnExceededEvent) -> None:
        agent_speaking_fut = asyncio.Future[None]()

        def _on_agent_state_changed(state_ev: AgentStateChangedEvent) -> None:
            if state_ev.new_state == "speaking" and not agent_speaking_fut.done():
                agent_speaking_fut.set_result(None)

        if self._session.agent_state == "speaking":
            agent_speaking_fut.set_result(None)
        else:
            self._session.on("agent_state_changed", _on_agent_state_changed)

        # wait for the EOU-triggered agent response (cancellable by the new user turn exceeded event)
        wait_inactive = asyncio.ensure_future(
            self.wait_for_idle(wait_for_agent=True, wait_for_user=False)
        )
        try:
            done, _ = await asyncio.wait(
                (agent_speaking_fut, wait_inactive), return_when=asyncio.FIRST_COMPLETED
            )
            if agent_speaking_fut in done:
                # agent started speaking, skip the user turn exceeded event
                return
        finally:
            self._session.off("agent_state_changed", _on_agent_state_changed)
            if not wait_inactive.done():
                wait_inactive.cancel()

        # re-check after the wait phase: if a handoff started in the meantime,
        # don't fire the callback on this now-stale activity.
        if self._scheduling_paused or self._new_turns_blocked:
            return

        # custom callback, locked - don't cancel user's callback
        logger.debug(
            "user turn limit exceeded",
            extra={"num_words": ev.accumulated_word_count, "duration": ev.duration},
        )
        self._user_turn_exceeded_locked = True
        try:
            await self._agent.on_user_turn_exceeded(ev)
        except Exception:
            logger.exception("error in on_user_turn_exceeded callback")
        finally:
            self._user_turn_exceeded_locked = False
            self._user_turn_exceeded_atask = None

    # AudioRecognition is calling this method to retrieve the chat context before running the TurnDetector model  # noqa: E501
    def retrieve_chat_ctx(self) -> llm.ChatContext:
        return self._agent.chat_ctx

    # endregion

    def _render_realtime_instructions(self, instructions: str | Instructions) -> str:
        """Resolve instructions to a plain string for the realtime session.

        Realtime instructions are session-level (there is no per-turn modality
        resolution like the pipeline path), so modality-specific ``Instructions``
        resolve to the realtime model's output modality.
        """
        if isinstance(instructions, Instructions):
            assert isinstance(self.llm, llm.RealtimeModel)
            modality: Literal["audio", "text"] = (
                "audio" if self.llm.capabilities.audio_output else "text"
            )
            return instructions.render(modality=modality)
        return instructions

    def _resolve_expressive_options(self) -> ExpressiveOptions | None:
        """Resolve the effective expressive setting. Returns None if disabled.

        The agent's ``expressive`` overrides the session's when set, matching how the
        agent's ``llm``/``tts`` override the session models.

        Expressive mode requires two things:
        - the inference gateway TTS (``livekit.agents.inference.TTS``): the markup
          normalization/conversion and expressive chunking run there, so direct
          provider plugins would receive unconverted markup.
        - a TTS that actually declares a markup dialect (``llm_instructions()`` is
          not ``None``): gateway providers without one (e.g. ``rime``, ``deepgram``)
          get no markup instructions, so no tags can appear in the stream — leaving
          it "active" would enable xml-aware chunking with nothing to chunk and
          re-introduce the stray-``<`` streaming stall.
        """
        from .agent_session import DEFAULT_EXPRESSIVE_OPTIONS, resolve_expressive_options

        if not isinstance(self.tts, inference.TTS) or self.tts.markup.llm_instructions() is None:
            return None

        expr = (
            self._agent.expressive
            if is_given(self._agent.expressive)
            else self._session._expressive
        )
        if not expr and not isinstance(expr, dict):
            return None
        # speech_steering renders per-provider delivery guidelines on top of the
        # provider-agnostic default; explicit templates override
        provider_key = self.tts.markup._provider_key() if self.tts else ""
        return resolve_expressive_options(
            expr if isinstance(expr, dict) else {},
            provider_key=provider_key,
            default=DEFAULT_EXPRESSIVE_OPTIONS,
        )

    def _inject_expressive_instructions(
        self,
        chat_ctx: llm.ChatContext,
        options: ExpressiveOptions,
        speech_handle: SpeechHandle,
    ) -> None:
        """Inject the TTS markup guide into the chat context."""

        def _to_instructions(v: Instructions | str) -> Instructions:
            return v if isinstance(v, Instructions) else Instructions(v)

        turn_modality = speech_handle.input_details.modality if speech_handle else None

        tts_instructions = (
            self.tts.markup.llm_instructions(speech_steering=options.get("speech_steering"))
            if self.tts
            else None
        )
        if tts_instructions:
            tts_template = _to_instructions(options["tts_instructions_template"])
            text = tts_template.render(
                modality=turn_modality,
                data={
                    "tts": {
                        "markup": {
                            "llm_instructions": tts_instructions,
                        },
                    },
                },
            )
            if text.strip():
                # keyed message: re-injection replaces last turn's guide instead of
                # stacking copies, and an expressive-off turn removes it again
                update_expressive_instructions(chat_ctx, text=text)

    @property
    def _no_pending_speech(self) -> bool:
        return not self._speech_q and (not self._current_speech or self._current_speech.done())

    def _on_pipeline_reply_done(self, _: asyncio.Task[None]) -> None:
        if self._no_pending_speech:
            # a speech awaiting its tool executions keeps the agent busy: stay in
            # "thinking" so the user-away timer isn't armed mid-tool (#6904)
            self._session._update_agent_state(
                "thinking" if self._background_speeches else "listening"
            )
            if self._audio_recognition:
                self._audio_recognition._on_end_of_agent_speech(
                    ignore_user_transcript_until=time.time()
                )
            if self.interruption_enabled:
                self._restore_interruption_by_audio_activity()

    @utils.log_exceptions(logger=logger)
    async def _tts_task(
        self,
        speech_handle: SpeechHandle,
        text: str | AsyncIterable[str],
        audio: AsyncIterable[rtc.AudioFrame] | None,
        add_to_chat_ctx: bool,
        model_settings: ModelSettings,
    ) -> None:
        with tracer.start_as_current_span(
            "agent_turn", context=self._session._root_span_context
        ) as current_span:
            current_span.set_attribute(trace_types.ATTR_AGENT_TURN_ID, speech_handle._generation_id)
            if parent_id := speech_handle._parent_generation_id:
                current_span.set_attribute(trace_types.ATTR_AGENT_PARENT_TURN_ID, parent_id)
            speech_handle._agent_turn_context = otel_context.get_current()

            await self._tts_task_impl(
                speech_handle=speech_handle,
                text=text,
                audio=audio,
                add_to_chat_ctx=add_to_chat_ctx,
                model_settings=model_settings,
            )

    async def _tts_task_impl(
        self,
        speech_handle: SpeechHandle,
        text: str | AsyncIterable[str],
        audio: AsyncIterable[rtc.AudioFrame] | None,
        add_to_chat_ctx: bool,
        model_settings: ModelSettings,
    ) -> None:
        current_span = trace.get_current_span(context=speech_handle._agent_turn_context)
        current_span.set_attribute(trace_types.ATTR_SPEECH_ID, speech_handle.id)

        tr_output = (
            self._session.output.transcription
            if self._session.output.transcription_enabled
            else None
        )
        audio_output = self._session.output.audio if self._session.output.audio_enabled else None

        # See discussion in https://github.com/livekit/agents/issues/4432
        authorization_tasks: list[asyncio.Future[Any]] = [
            asyncio.ensure_future(speech_handle._wait_for_authorization()),
            asyncio.ensure_future(self._authorization_allowed.wait()),
        ]
        if speech_handle.allow_interruptions:
            authorization_tasks.append(asyncio.ensure_future(self._user_silence_event.wait()))
        await speech_handle.wait_if_not_interrupted(authorization_tasks)
        speech_handle._clear_authorization()

        if speech_handle.interrupted:
            current_span.set_attribute(trace_types.ATTR_SPEECH_INTERRUPTED, True)
            await utils.aio.cancel_and_wait(*authorization_tasks)
            return

        text_source: AsyncIterable[str] | None = None
        audio_source: AsyncIterable[str] | None = None

        tee: utils.aio.itertools.Tee[str] | None = None
        if isinstance(text, AsyncIterable):
            tee = utils.aio.itertools.tee(text, 2)
            text_source, audio_source = tee
        elif isinstance(text, str):

            async def _read_text() -> AsyncIterable[str]:
                yield text

            text_source = _read_text()
            audio_source = _read_text()

        tts_task: asyncio.Task[Any] | None = None
        forward_audio_task: asyncio.Task[Any] | None = None
        forward_text_task: asyncio.Task[Any] | None = None
        started_speaking_at: float | None = None
        stopped_speaking_at: float | None = None
        started_forwarding_at: float | None = None

        def _on_first_frame(fut: asyncio.Future[float] | asyncio.Future[None]) -> None:
            """
            Callback to update the agent state when the first frame is captured:
            1. _AudioOutput.first_frame_fut (float)
            2. _TextOutput.first_text_fut (None)
            """
            nonlocal started_speaking_at, started_forwarding_at
            try:
                started_speaking_at = fut.result() or time.time()
                started_forwarding_at = (
                    audio_out.started_forwarding_at
                    if audio_out and audio_out.started_forwarding_at is not None
                    else started_speaking_at
                )
            except BaseException:
                return

            self._session._update_agent_state(
                "speaking",
                start_time=started_speaking_at,
                otel_context=speech_handle._agent_turn_context,
            )
            if self._audio_recognition:
                self._audio_recognition._on_start_of_agent_speech(started_at=started_speaking_at)
            if self.interruption_enabled:
                self._disable_vad_interruption_soon()

        audio_out: _AudioOutput | None = None
        tts_gen_data: _TTSGenerationData | None = None
        if audio_output is not None:
            if audio is None:
                # generate audio using TTS
                tts_task, tts_gen_data = perform_tts_inference(
                    node=self._agent.tts_node,
                    input=audio_source,
                    model_settings=model_settings,
                    text_transforms=self._session.options.tts_text_transforms,
                    model=self.tts.model if self.tts else None,
                    provider=self.tts.provider if self.tts else None,
                )
                if (
                    self.use_tts_aligned_transcript
                    and (tts := self.tts)
                    and (tts.capabilities.aligned_transcript or not tts.capabilities.streaming)
                    and (timed_texts := await tts_gen_data.timed_texts_fut)
                ):
                    text_source = _aligned_transcript_or_text(timed_texts, text_source)

                forward_audio_task, audio_out = perform_audio_forwarding(
                    audio_output=audio_output,
                    tts_output=tts_gen_data.audio_ch,
                    reconcile_playout_pause=lambda: self._reconcile_playout_pause(speech_handle),
                )
            else:
                # use the provided audio
                forward_audio_task, audio_out = perform_audio_forwarding(
                    audio_output=audio_output,
                    tts_output=audio,
                    reconcile_playout_pause=lambda: self._reconcile_playout_pause(speech_handle),
                )

            audio_out.first_frame_fut.add_done_callback(_on_first_frame)

        # text output
        tr_node = self._agent.transcription_node(text_source, model_settings)
        tr_node_result = await tr_node if asyncio.iscoroutine(tr_node) else tr_node
        text_out: _TextOutput | None = None
        if tr_node_result is not None:
            forward_text_task, text_out = perform_text_forwarding(
                text_output=tr_output,
                source=tr_node_result,
            )
            if audio_output is None:
                # update the agent state based on text if no audio output
                text_out.first_text_fut.add_done_callback(_on_first_frame)

        all_tasks: list[asyncio.Future[Any]] = [
            t for t in (tts_task, forward_audio_task, forward_text_task) if t is not None
        ]
        await speech_handle.wait_if_not_interrupted(all_tasks)

        # check for errors in generation/forwarding tasks (e.g. missing audio file)
        for task in (tts_task, forward_audio_task, forward_text_task):
            if task is not None and task.done() and not task.cancelled():
                if exc := task.exception():
                    raise exc

        if audio_output is not None:
            await speech_handle.wait_if_not_interrupted(
                [asyncio.ensure_future(audio_output.wait_for_playout())]
            )

        stopped_speaking_at = time.time()
        current_span.set_attribute(trace_types.ATTR_SPEECH_INTERRUPTED, speech_handle.interrupted)
        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(*all_tasks)

            if audio_output is not None:
                audio_output.clear_buffer()
                await audio_output.wait_for_playout()

        if tee is not None:
            await tee.aclose()

        # use synchronized transcript when available after interruption
        forwarded_text = text_out.text if text_out else ""
        if speech_handle.interrupted and audio_output is not None:
            playback_ev = await audio_output.wait_for_playout()

            if (
                audio_out is not None
                and audio_out.first_frame_fut.done()
                and not audio_out.first_frame_fut.cancelled()
            ):
                if playback_ev.synchronized_transcript is not None:
                    forwarded_text = playback_ev.synchronized_transcript
            else:
                forwarded_text = ""
        current_span.set_attribute(trace_types.ATTR_RESPONSE_TEXT, forwarded_text)

        if forwarded_text and add_to_chat_ctx:
            assistant_metrics: llm.MetricsReport = {}

            if tts_gen_data and tts_gen_data.ttfb is not None:
                assistant_metrics["tts_node_ttfb"] = tts_gen_data.ttfb

            if stopped_speaking_at and started_speaking_at:
                assistant_metrics["started_speaking_at"] = started_speaking_at
                assistant_metrics["stopped_speaking_at"] = stopped_speaking_at

                if started_forwarding_at is not None:
                    assistant_metrics["playback_latency"] = (
                        started_speaking_at - started_forwarding_at
                    )

            msg = self._agent._chat_ctx.add_message(
                role="assistant",
                content=forwarded_text,
                interrupted=speech_handle.interrupted,
                metrics=assistant_metrics,
            )
            speech_handle._item_added([msg])
            self._session._conversation_item_added(msg)

        if self._session.agent_state == "speaking":
            self._session._update_agent_state(
                "thinking" if self._background_speeches else "listening"
            )
            if self._audio_recognition:
                self._audio_recognition._on_end_of_agent_speech(
                    ignore_user_transcript_until=time.time()
                )
            if self.interruption_enabled:
                self._restore_interruption_by_audio_activity()

        if audio_out is not None and not audio_out.first_frame_fut.done():
            audio_out.first_frame_fut.cancel()

    def _on_enter_ignored_tools(self, tool_ctx: llm.ToolContext) -> list[llm.Tool]:
        """Tools flagged IGNORE_ON_ENTER, when this reply runs inside on_enter."""
        on_enter_data = _OnEnterContextVar.get(None)
        if (
            on_enter_data is None
            or on_enter_data.agent != self._agent
            or on_enter_data.session != self._session
        ):
            return []
        return [
            tool
            for tool in tool_ctx.flatten()
            if isinstance(tool, llm.RawFunctionTool | llm.FunctionTool)
            and tool.info.flags & ToolFlag.IGNORE_ON_ENTER
        ]

    @utils.log_exceptions(logger=logger)
    async def _pipeline_reply_task(
        self,
        *,
        speech_handle: SpeechHandle,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool | llm.Toolset],
        model_settings: ModelSettings,
        new_message: llm.ChatMessage | None = None,
        instructions: str | Instructions | None = None,
        _previous_user_metrics: llm.MetricsReport | None = None,
    ) -> None:
        with tracer.start_as_current_span(
            "agent_turn", context=self._session._root_span_context
        ) as current_span:
            current_span.set_attribute(trace_types.ATTR_AGENT_TURN_ID, speech_handle._generation_id)
            if parent_id := speech_handle._parent_generation_id:
                current_span.set_attribute(trace_types.ATTR_AGENT_PARENT_TURN_ID, parent_id)
            speech_handle._agent_turn_context = otel_context.get_current()

            await self._pipeline_reply_task_impl(
                speech_handle=speech_handle,
                chat_ctx=chat_ctx,
                tools=tools,
                model_settings=model_settings,
                new_message=new_message,
                instructions=instructions,
                _previous_user_metrics=_previous_user_metrics,
            )

    async def _pipeline_reply_task_impl(
        self,
        *,
        speech_handle: SpeechHandle,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool | llm.Toolset],
        model_settings: ModelSettings,
        new_message: llm.ChatMessage | None = None,
        instructions: str | Instructions | None = None,
        _previous_user_metrics: llm.MetricsReport | None = None,
    ) -> None:
        from .agent import ModelSettings

        current_span = trace.get_current_span(context=speech_handle._agent_turn_context)
        current_span.set_attribute(trace_types.ATTR_SPEECH_ID, speech_handle.id)
        if instructions is not None:
            turn_modality = speech_handle.input_details.modality
            instr_trace = (
                instructions.render(modality=turn_modality)
                if isinstance(instructions, Instructions)
                else instructions
            )
            current_span.set_attribute(trace_types.ATTR_INSTRUCTIONS, instr_trace)
        if new_message:
            current_span.set_attribute(
                trace_types.ATTR_USER_INPUT, new_message.raw_text_content or ""
            )

        if (room_io := self._session._room_io) and room_io.room.isconnected():
            _set_participant_attributes(current_span, room_io.room.local_participant)

        audio_output = self._session.output.audio if self._session.output.audio_enabled else None
        text_output = (
            self._session.output.transcription
            if self._session.output.transcription_enabled
            else None
        )
        chat_ctx = chat_ctx.copy()
        tool_ctx = llm.ToolContext(tools)
        tool_ctx._exclude(self._on_enter_ignored_tools(tool_ctx))

        if new_message is not None:
            chat_ctx.insert(new_message)

        # resolve modality-specific instructions for this turn
        turn_modality = speech_handle.input_details.modality
        if instructions is not None:
            instr_text = (
                instructions.render(modality=turn_modality)
                if isinstance(instructions, Instructions)
                else instructions
            )
            chat_ctx.add_message(role="system", content=[instr_text])
        elif isinstance(self._agent.instructions, Instructions):
            update_instructions(
                chat_ctx,
                instructions=self._agent.instructions,
                modality=turn_modality,
                add_if_missing=False,
            )

        # inject expressive instructions (TTS markup guide + speaker context)
        _expr_opts = self._resolve_expressive_options()
        if _expr_opts is not None:
            self._inject_expressive_instructions(chat_ctx, _expr_opts, speech_handle)
        else:
            # expressive is off for this turn (toggled off via update_options, an agent
            # override, or a handoff to a TTS without a markup dialect): remove the
            # injected markup guide and scrub markup left in past assistant turns so
            # the LLM isn't instructed or few-shotted into emitting tags nothing
            # downstream converts or strips — an unsupported tag would reach the TTS
            # as literal text and be spoken.
            remove_expressive_instructions(chat_ctx)
            _strip_assistant_markup(chat_ctx)
            if chat_ctx is not self._agent._chat_ctx:
                # user turns run on a copy of the agent's history; clean the stored
                # history too so stale markup doesn't survive into future snapshots
                remove_expressive_instructions(self._agent._chat_ctx)
                _strip_assistant_markup(self._agent._chat_ctx)

        # TODO(theomonnom): since pause is closing STT/LLM/TTS, we have issues for SpeechHandle still in queue  # noqa: E501
        # I should implement a retry mechanism?

        # a tool still running from a previous turn isn't in this turn's ctx, so the model
        # re-issues it and duplicates side effects. inject an in-progress placeholder so it
        # leaves the call alone. mutating chat_ctx directly (not a copy) keeps a custom
        # llm_node's edits; the placeholder is stripped again before chat_ctx is forwarded.
        _inject_running_tool_calls(
            chat_ctx,
            [task.ctx.function_call for task in _RunningTasks.get(self._session, {}).values()],
        )

        tasks: list[asyncio.Task[Any]] = []
        llm_task, llm_gen_data = perform_llm_inference(
            node=self._agent.llm_node,
            chat_ctx=chat_ctx,
            tool_ctx=tool_ctx,
            model_settings=model_settings,
            model=self.llm.model if self.llm else None,
            provider=self.llm.provider if self.llm else None,
        )
        tasks.append(llm_task)

        def _on_llm_task_done(task: asyncio.Task[bool]) -> None:
            # Surface a genuine LLM failure (not interruption/cancellation) so it
            # propagates through SpeechHandle.exception() and RunResult (i.e.
            # session.run()); this also retrieves the task exception (no "never
            # retrieved" warning).
            if task.cancelled():
                return
            if (exc := task.exception()) is not None:
                speech_handle._error = exc

        llm_task.add_done_callback(_on_llm_task_done)

        # split the LLM text on FlushSentinel into segments, each spoken independently;
        # without a FlushSentinel there is a single (continuous) segment
        @dataclass
        class _SpeechSegment:
            text: utils.aio.Chan[str]  # spoken text for this segment
            tts: _TTSGenerationData | None = None  # audio + timed transcript, when enabled

        segment_ch = utils.aio.Chan[_SpeechSegment]()

        @utils.log_exceptions(logger=logger)
        async def _produce_segments() -> None:
            await llm_gen_data.started_fut  # keep the tts span under the llm span
            current: _SpeechSegment | None = None
            tts_text: utils.aio.Chan[str] | None = None
            prev_tts_task: asyncio.Task[bool] | None = None

            async def _start_segment() -> _SpeechSegment:
                # start this segment's tts; one inference at a time (await the previous),
                # but the next starts during the previous segment's playout, not after
                nonlocal tts_text, prev_tts_task
                tts_data: _TTSGenerationData | None = None
                tts_text = None
                if audio_output is not None:
                    if prev_tts_task is not None:
                        await prev_tts_task
                    tts_text = utils.aio.Chan[str]()
                    prev_tts_task, tts_data = perform_tts_inference(
                        node=self._agent.tts_node,
                        input=tts_text,
                        model_settings=model_settings,
                        text_transforms=self._session.options.tts_text_transforms,
                        model=self.tts.model if self.tts else None,
                        provider=self.tts.provider if self.tts else None,
                    )
                    tasks.append(prev_tts_task)
                seg = _SpeechSegment(text=utils.aio.Chan[str](), tts=tts_data)
                segment_ch.send_nowait(seg)
                return seg

            def _end_segment() -> None:
                nonlocal current, tts_text
                if current is not None:
                    current.text.close()
                if tts_text is not None:
                    tts_text.close()  # let this segment's TTS inference finish
                current, tts_text = None, None

            try:
                async for chunk in llm_gen_data.text_ch:
                    if isinstance(chunk, FlushSentinel):
                        _end_segment()
                        continue
                    text = chunk
                    if not text:
                        continue
                    if current is None:
                        current = await _start_segment()
                    current.text.send_nowait(text)
                    if tts_text is not None:
                        tts_text.send_nowait(text)
            finally:
                _end_segment()
                segment_ch.close()

        # start synthesis preemptively (before the speech is scheduled) when enabled;
        # otherwise it starts right after scheduling below
        preemptive_opts = self.preemptive_generation_opts
        synthesize_task: asyncio.Task[None] | None = None
        if (
            audio_output is not None
            and preemptive_opts["enabled"]
            and preemptive_opts["preemptive_tts"]
        ):
            synthesize_task = asyncio.create_task(
                _produce_segments(), name="AgentActivity.pipeline_reply.produce_segments"
            )
            tasks.append(synthesize_task)

        wait_for_scheduled = asyncio.ensure_future(speech_handle._wait_for_scheduled())
        await speech_handle.wait_if_not_interrupted([wait_for_scheduled])

        # add new message to chat context if the speech is scheduled

        user_metrics: llm.MetricsReport | None = _previous_user_metrics
        if new_message is not None and speech_handle.scheduled:
            self._agent._chat_ctx.insert(new_message)
            self._session._conversation_item_added(new_message)
            user_metrics = new_message.metrics

        if speech_handle.interrupted:
            current_span.set_attribute(trace_types.ATTR_SPEECH_INTERRUPTED, True)
            await utils.aio.cancel_and_wait(*tasks, wait_for_scheduled)
            return

        # start synthesis now if it wasn't started preemptively
        if synthesize_task is None:
            synthesize_task = asyncio.create_task(
                _produce_segments(), name="AgentActivity.pipeline_reply.produce_segments"
            )
            tasks.append(synthesize_task)

        self._session._update_agent_state("thinking")

        authorization_tasks: list[asyncio.Future[Any]] = [
            asyncio.ensure_future(speech_handle._wait_for_authorization()),
            asyncio.ensure_future(self._authorization_allowed.wait()),
        ]
        if speech_handle.allow_interruptions:
            authorization_tasks.append(asyncio.ensure_future(self._user_silence_event.wait()))
        await speech_handle.wait_if_not_interrupted(authorization_tasks)
        speech_handle._clear_authorization()

        if speech_handle.interrupted:
            current_span.set_attribute(trace_types.ATTR_SPEECH_INTERRUPTED, True)
            await utils.aio.cancel_and_wait(*tasks, *authorization_tasks)
            return

        reply_started_at = time.time()

        # In expressive mode the LLM emits markup the TTS interprets as audio directives.
        # The raw text (markup intact) flows into chat history and the tts_node; the
        # transcript sinks strip it from the room transcript downstream and surface the
        # leading expression as the lk.expression attribute (see TranscriptMarkupStripper).
        started_speaking_at: float | None = None
        stopped_speaking_at: float | None = None
        started_forwarding_at: float | None = None
        first_tts_gen_data: _TTSGenerationData | None = None  # first segment's tts, for metrics

        def _on_first_frame(
            fut: asyncio.Future[float] | asyncio.Future[None],
            audio_out: _AudioOutput | None = None,
        ) -> None:
            """
            Callback to update the agent state when the first frame is captured:
            1. _AudioOutput.first_frame_fut (float)
            2. _TextOutput.first_text_fut (None)
            """
            nonlocal started_speaking_at, started_forwarding_at
            # only the first segment's first frame should trigger state transitions
            if started_speaking_at is not None:
                return
            try:
                started_speaking_at = fut.result() or time.time()
                started_forwarding_at = (
                    audio_out.started_forwarding_at
                    if audio_out and audio_out.started_forwarding_at is not None
                    else started_speaking_at
                )
            except BaseException:
                return

            # purely used for realtime console rendering (metrics are shown
            # as soon as the agent starts speaking, before playout finishes)
            early_metrics: llm.MetricsReport = {}
            if llm_gen_data.ttft is not None:
                early_metrics["llm_node_ttft"] = llm_gen_data.ttft
            if first_tts_gen_data and first_tts_gen_data.ttfb is not None:
                early_metrics["tts_node_ttfb"] = first_tts_gen_data.ttfb
            early_metrics["playback_latency"] = started_speaking_at - started_forwarding_at
            if user_metrics and "stopped_speaking_at" in user_metrics:
                early_metrics["e2e_latency"] = (
                    started_speaking_at - user_metrics["stopped_speaking_at"]
                )
            self._session._early_assistant_metrics = early_metrics

            self._session._update_agent_state(
                "speaking",
                start_time=started_speaking_at,
                otel_context=speech_handle._agent_turn_context,
            )

            if self._audio_recognition:
                self._audio_recognition._on_start_of_agent_speech(started_at=started_speaking_at)
            if self.interruption_enabled:
                self._disable_vad_interruption_soon()

        # messages in RunResult are ordered by the `created_at` field
        def _tool_execution_started_cb(fnc_call: llm.FunctionCall) -> None:
            # function call is created during LLM generation, might be before the speech is authorized
            # reset the `created_at` to the start time of the tool execution
            fnc_call.created_at = time.time()
            speech_handle._item_added([fnc_call])

        def _tool_execution_completed_cb(out: ToolExecutionOutput) -> None:
            speech_handle._item_added([out.fnc_call_out])

        # start to execute tools (only after play())
        exe_task, tool_output = perform_tool_executions(
            session=self._session,
            speech_handle=speech_handle,
            tool_ctx=tool_ctx,
            tool_choice=model_settings.tool_choice,
            function_stream=llm_gen_data.function_ch,
            tool_execution_started_cb=_tool_execution_started_cb,
            tool_execution_completed_cb=_tool_execution_completed_cb,
        )

        # use the TTS-aligned timing for the transcript instead of the raw text, when
        # the TTS supports it (resolved per segment below)
        use_aligned_transcript = bool(
            audio_output is not None
            and self.use_tts_aligned_transcript
            and (tts := self.tts)
            and (tts.capabilities.aligned_transcript or not tts.capabilities.streaming)
        )

        # forward each segment serially: its audio must finish playing before the next
        # one starts (matches the realtime per-message behavior)
        segment_outputs: list[_ForwardOutput] = []
        read_transcript_from_tts = False

        async def _next_segment() -> _SpeechSegment | None:
            # wake promptly on interruption so a slow LLM (no segment produced yet)
            # doesn't delay teardown
            recv = asyncio.ensure_future(segment_ch.recv())
            await speech_handle.wait_if_not_interrupted([recv])
            if speech_handle.interrupted:
                await utils.aio.cancel_and_wait(recv)
                return None
            try:
                return recv.result()
            except utils.aio.ChanClosed:
                return None

        while (segment := await _next_segment()) is not None:
            if first_tts_gen_data is None:
                first_tts_gen_data = segment.tts

            transcript: AsyncIterable[str] = segment.text
            if (
                segment.tts is not None
                and use_aligned_transcript
                and (timed_texts := await segment.tts.timed_texts_fut)
            ):
                # the channel exists as soon as the node streams, before any timed
                # word has arrived, so a TTS that advertises alignment it doesn't
                # deliver would leave the segment with no transcript at all
                transcript = _aligned_transcript_or_text(timed_texts, segment.text)
                read_transcript_from_tts = True

            tr_node = self._agent.transcription_node(transcript, model_settings)
            text_source = await tr_node if asyncio.iscoroutine(tr_node) else tr_node
            audio_source = segment.tts.audio_ch if segment.tts else None

            out = await forward_generation(
                speech_handle=speech_handle,
                audio_output=audio_output,
                text_output=text_output,
                audio_source=audio_source,
                text_source=text_source,
                on_first_frame=_on_first_frame,
                reconcile_playout_pause=lambda: self._reconcile_playout_pause(speech_handle),
            )
            segment_outputs.append(out)
            if speech_handle.interrupted:
                break

        stopped_speaking_at = time.time()
        assistant_metrics: llm.MetricsReport = {}

        if self.llm:
            assistant_metrics["llm_metadata"] = self.llm.metrics_metadata
        if self.tts:
            assistant_metrics["tts_metadata"] = self.tts.metrics_metadata

        if llm_gen_data.ttft is not None:
            assistant_metrics["llm_node_ttft"] = llm_gen_data.ttft

        if llm_gen_data.tps is not None:
            assistant_metrics["llm_node_tps"] = llm_gen_data.tps

        if (ttfs := _time_to_first_sentence(llm_gen_data, first_tts_gen_data)) is not None:
            assistant_metrics["llm_node_ttfs"] = ttfs

        if first_tts_gen_data and first_tts_gen_data.ttfb is not None:
            assistant_metrics["tts_node_ttfb"] = first_tts_gen_data.ttfb

        if stopped_speaking_at and started_speaking_at:
            assistant_metrics["started_speaking_at"] = started_speaking_at
            assistant_metrics["stopped_speaking_at"] = stopped_speaking_at

            if started_forwarding_at is not None:
                assistant_metrics["playback_latency"] = started_speaking_at - started_forwarding_at

            if user_metrics and "stopped_speaking_at" in user_metrics:
                e2e_latency = started_speaking_at - user_metrics["stopped_speaking_at"]
                assistant_metrics["e2e_latency"] = e2e_latency
                current_span.set_attribute(trace_types.ATTR_E2E_LATENCY, e2e_latency)

        current_span.set_attribute(trace_types.ATTR_SPEECH_INTERRUPTED, speech_handle.interrupted)

        forwarded_text = "".join(out.forwarded_text for out in segment_outputs)
        if speech_handle.interrupted:
            # forward_generation already cleared the buffer and waited for playout
            await utils.aio.cancel_and_wait(*tasks)
        elif read_transcript_from_tts and any(
            out.played != "skipped" and out.text_out is not None and not out.text_out.text
            for out in segment_outputs
        ):
            logger.warning(
                "`use_tts_aligned_transcript` is enabled but no agent transcript was returned from tts"
            )

        if forwarded_text:
            # forwarded_text carries the raw LLM output including any expressive markup
            # (the transcript forwarder strips it only for the room transcript), so the
            # markup lives directly on the stored assistant message.
            extra_kwargs: dict = {}
            if llm_gen_data.generated_extra:
                extra_kwargs["extra"] = llm_gen_data.generated_extra
            msg = chat_ctx.add_message(
                role="assistant",
                content=forwarded_text,
                id=llm_gen_data.id,
                interrupted=speech_handle.interrupted,
                created_at=reply_started_at,
                metrics=assistant_metrics,
                **extra_kwargs,
            )
            self._agent._chat_ctx.insert(msg)
            self._session._conversation_item_added(msg)
            speech_handle._item_added([msg])
            current_span.set_attribute(trace_types.ATTR_RESPONSE_TEXT, forwarded_text)

        if not speech_handle.interrupted and len(tool_output.output) > 0:
            self._session._update_agent_state("thinking")
            if self._audio_recognition:
                self._audio_recognition._on_end_of_agent_speech(
                    ignore_user_transcript_until=time.time()
                )
            if self.interruption_enabled:
                self._restore_interruption_by_audio_activity()
        elif self._session.agent_state == "speaking":
            # a running tool keeps the agent busy; "listening" would arm the away timer
            tool_running = not speech_handle.interrupted and not exe_task.done()
            self._session._update_agent_state("thinking" if tool_running else "listening")
            if self._audio_recognition:
                self._audio_recognition._on_end_of_agent_speech(
                    ignore_user_transcript_until=time.time()
                )
            if self.interruption_enabled:
                self._restore_interruption_by_audio_activity()

        for out in segment_outputs:
            if out.audio_out is not None and not out.audio_out.first_frame_fut.done():
                out.audio_out.first_frame_fut.cancel()

        speech_handle._mark_generation_done()  # mark the playout done before waiting for the tool execution  # noqa: E501

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(exe_task)

            # commit results of tools that finished despite the interruption (#3702), so
            # the next inference doesn't run them again
            interrupted_calls: list[llm.FunctionCall] = []
            interrupted_fnc_outputs: list[llm.FunctionCallOutput] = []
            for sanitized_out in tool_output.output:
                interrupted_calls.append(sanitized_out.fnc_call)
                interrupted_fnc_outputs.append(_interrupted_tool_output(sanitized_out))

            if interrupted_tool_messages := interrupted_calls + interrupted_fnc_outputs:
                self._session.emit(
                    "function_tools_executed",
                    FunctionToolsExecutedEvent(
                        function_calls=interrupted_calls,
                        function_call_outputs=interrupted_fnc_outputs,
                    ),
                )
                self._agent._chat_ctx.insert(interrupted_tool_messages)
                self._session._tool_items_added(interrupted_tool_messages)
            return

        # wait for the tool execution to complete
        self._background_speeches.add(speech_handle)
        try:
            await exe_task
        finally:
            self._background_speeches.discard(speech_handle)

        # important: no agent output should be used after this point

        if len(tool_output.output) > 0:
            max_steps_reached = speech_handle.num_steps >= self._session.options.max_tool_steps + 1

            if max_steps_reached:
                logger.warning(
                    "maximum number of function calls steps reached, "
                    "generating final response with tool_choice='none'",
                    extra={"speech_id": speech_handle.id},
                )

            speech_handle._num_steps += 1

            new_calls: list[llm.FunctionCall] = []
            new_fnc_outputs: list[llm.FunctionCallOutput] = []
            new_agent_task: Agent | None = None
            ignore_task_switch = False
            fnc_executed_ev = FunctionToolsExecutedEvent(
                function_calls=[], function_call_outputs=[]
            )
            for sanitized_out in tool_output.output:
                new_calls.append(sanitized_out.fnc_call)
                new_fnc_outputs.append(sanitized_out.fnc_call_out)

                fnc_executed_ev.function_calls.append(sanitized_out.fnc_call)
                fnc_executed_ev.function_call_outputs.append(sanitized_out.fnc_call_out)

                if new_agent_task is not None and sanitized_out.agent_task is not None:
                    logger.error("expected to receive only one AgentTask from the tool executions")
                    ignore_task_switch = True
                    # TODO(long): should we mark the function call as failed to notify the LLM?

                new_agent_task = sanitized_out.agent_task

            if new_agent_task and not ignore_task_switch:
                fnc_executed_ev._handoff_required = True

            self._session.emit("function_tools_executed", fnc_executed_ev)

            draining = self.scheduling_paused
            if fnc_executed_ev._handoff_required and new_agent_task and not ignore_task_switch:
                self._session.update_agent(new_agent_task)
                draining = True

            tool_messages = new_calls + new_fnc_outputs
            # commit now so results survive even if the reply speech never runs (#3702)
            if tool_messages:
                self._agent._chat_ctx.insert(tool_messages)
                self._session._tool_items_added(tool_messages)

            if fnc_executed_ev.has_tool_reply and not speech_handle.interrupted:
                # forwarding chat_ctx to the tool reply: drop the in-progress placeholders
                # (the next turn re-injects from the live running set)
                _strip_running_tool_calls(chat_ctx)

                # refresh conversation items added during tool execution: a tool that
                # awaits an inline AgentTask runs a whole sub-conversation, merged into
                # the agent's chat_ctx at handoff-return - this turn's snapshot predates
                # it. Without the refresh, the tool response is generated blind to what
                # was actually said inside the tool call (and re-asks captured fields).
                # Conversational analog of the update_instructions() refresh below.
                # tool_messages must be added first so merge() dedups them by id.
                chat_ctx.items.extend(tool_messages)
                chat_ctx.merge(self._agent._chat_ctx, exclude_instructions=True)

                # refresh instructions in chat_ctx so that any update_instructions()
                # calls made inside tool functions are reflected in the tool response
                # generation (fixes #4242)
                update_instructions(
                    chat_ctx,
                    instructions=self._agent._instructions,
                    modality=speech_handle.input_details.modality,
                    add_if_missing=False,
                )

                tool_response_task = self._create_speech_task(
                    self._pipeline_reply_task(
                        speech_handle=speech_handle,
                        chat_ctx=chat_ctx,
                        tools=tools,
                        model_settings=ModelSettings(
                            # Avoid setting tool_choice to "required" or a specific function when
                            # passing tool response back to the LLM.
                            # Force tool_choice="none" when max steps reached to guarantee
                            # a final text response instead of silently stopping.
                            tool_choice="none"
                            if max_steps_reached or draining or model_settings.tool_choice == "none"
                            else "auto",
                        ),
                        # in case the current reply only generated tools (no speech), re-use the current user_metrics for the next
                        # tool response generation
                        _previous_user_metrics=user_metrics if not forwarded_text else None,
                    ),
                    speech_handle=speech_handle,
                    name="AgentActivity.pipeline_reply",
                )
                tool_response_task.add_done_callback(self._on_pipeline_reply_done)
                self._schedule_speech(
                    speech_handle, SpeechHandle.SPEECH_PRIORITY_NORMAL, force=True
                )

    async def _sync_realtime_user_message(
        self, *, speech_handle: SpeechHandle, user_message: llm.ChatMessage
    ) -> bool:
        assert self._rt_session is not None, "rt_session is not available"
        async with self._realtime_chat_ctx_lock:
            chat_ctx = self._rt_session.chat_ctx.copy()
            chat_ctx._upsert_item(user_message)
            self._pending_realtime_user_message_ids.add(user_message.id)
            try:
                sync_result = await self._rt_session._sync_user_message(chat_ctx, user_message.id)
            except Exception as error:
                logger.exception("failed to synchronize the user turn before generating the reply")
                speech_handle._mark_done(error=error)
                raise
            finally:
                self._pending_realtime_user_message_ids.discard(user_message.id)

            if sync_result.status == _UserMessageSyncStatus.REJECTED:
                sync_error = sync_result.error or llm.RealtimeError(
                    "realtime provider rejected the finalized user turn"
                )
                logger.error(
                    "failed to synchronize the user turn before generating the reply",
                    extra={"error": str(sync_error)},
                )
                speech_handle._mark_done(error=sync_error)
                return False

            if sync_result.status == _UserMessageSyncStatus.UNKNOWN:
                logger.warning(
                    "user turn synchronization acknowledgement is unknown; preserving "
                    "best-effort realtime behavior",
                    extra={"error": str(sync_result.error) if sync_result.error else None},
                )

            self._commit_realtime_user_message(user_message)
        return True

    def _provider_transcription_matches_bounded_message(
        self,
        provider_message: llm.ChatMessage,
        bounded_message: llm.ChatMessage,
    ) -> bool:
        if not _transcripts_equivalent(
            bounded_message.raw_text_content or "", provider_message.raw_text_content
        ):
            return False

        bounded_started_at: float | None = None
        if bounded_message.metrics is not None:
            value = bounded_message.metrics.get("started_speaking_at")
            if isinstance(value, (int, float)):
                bounded_started_at = float(value)

        provider_started_at: float | None = None
        if provider_message.metrics is not None:
            value = provider_message.metrics.get("started_speaking_at")
            if isinstance(value, (int, float)):
                provider_started_at = float(value)
        return (
            bounded_started_at is None
            or (provider_started_at or provider_message.created_at) >= bounded_started_at
        )

    def _has_matching_provider_transcription(self, bounded_message: llm.ChatMessage) -> bool:
        for item_id in self._provider_transcription_item_ids:
            item = self._agent._chat_ctx.get_by_id(item_id)
            if isinstance(
                item, llm.ChatMessage
            ) and self._provider_transcription_matches_bounded_message(item, bounded_message):
                return True
        return False

    def _discard_matching_bounded_close_user_message(
        self, provider_message: llm.ChatMessage
    ) -> bool:
        for item_id in tuple(self._bounded_close_user_message_ids):
            item = self._agent._chat_ctx.get_by_id(item_id)
            if isinstance(
                item, llm.ChatMessage
            ) and self._provider_transcription_matches_bounded_message(provider_message, item):
                self._bounded_close_user_message_ids.discard(item_id)
                return True
        return False

    def _commit_bounded_user_message_locally(
        self,
        user_message: llm.ChatMessage,
        *,
        provider_reply_already_triggered: bool = False,
    ) -> None:
        # Server-detected audio is already represented by the provider-owned conversation item.
        # A concurrent external recognizer may finish during close, but it must not create a
        # second user item for the same turn.
        if (
            isinstance(self.llm, llm.RealtimeModel)
            and self._realtime_input_mode == "audio"
            and self._turn_policy.input_owner == "provider"
        ):
            return

        # Empty candidates are hook opportunities, not conversation items. This helper is called
        # from several close/cancellation paths; the delegated commit is idempotent by message ID.
        if not (user_message.raw_text_content or "").strip():
            return

        if (
            self._session._closing
            and provider_reply_already_triggered
            and isinstance(self.llm, llm.RealtimeModel)
            and self._realtime_input_mode == "audio"
            and self.llm.capabilities.user_transcription
        ):
            # A manual provider commit and the external STT final can finish in either order.
            # Whichever copy lands first represents this bounded turn; the later equivalent
            # transcript is discarded without suppressing a close-only external fallback.
            if self._has_matching_provider_transcription(user_message):
                return
            self._bounded_close_user_message_ids.add(user_message.id)

        self._commit_user_message_locally(user_message)

    def _commit_user_message_locally(self, user_message: llm.ChatMessage) -> None:
        self._commit_realtime_user_message(user_message)

    def _commit_realtime_user_message(self, user_message: llm.ChatMessage) -> None:
        is_new_message = self._agent._chat_ctx.get_by_id(user_message.id) is None
        self._agent._chat_ctx._upsert_item(user_message)
        if is_new_message:
            self._session._conversation_item_added(user_message)

    def _interrupt_created_realtime_generation_if_owned(
        self,
        generation_fut: asyncio.Future[llm.GenerationCreatedEvent],
        input_token: _RealtimeTurnTransaction | None,
    ) -> None:
        """Settle a completed generation future and stop only its still-owned output."""

        assert self._rt_session is not None, "rt_session is not available"
        if not generation_fut.done() or generation_fut.cancelled():
            return

        # Retrieving an error is required even during cancellation so a synchronously failed
        # provider future cannot surface later as an unhandled exception.
        if generation_fut.exception() is not None:
            return

        generation_ev = generation_fut.result()
        input_still_owned = input_token is None or input_token is self._realtime_turn
        if input_token is not None and input_still_owned:
            input_token.state = "generation_created"
        if input_still_owned and self._active_realtime_generation is generation_ev:
            # Text turns use the exact generation identity directly. Audio also requires its
            # transaction to remain current so a stale owner cannot interrupt newer output.
            self._rt_session.interrupt()

    @utils.log_exceptions(logger=logger)
    async def _realtime_reply_task(
        self,
        *,
        speech_handle: SpeechHandle,
        model_settings: ModelSettings,
        realtime_audio_input_owner: _RealtimeAudioInputOwner | None = None,
        commit_realtime_audio: bool = False,
        tools: list[llm.Tool | llm.Toolset] | None = None,
        user_message: llm.ChatMessage | None = None,
        instructions: str | None = None,
        tool_reply: bool = False,
        text: str | AsyncIterable[str] | None = None,
    ) -> None:
        assert self._rt_session is not None, "rt_session is not available"

        # Native server-detected audio responses are handled by _realtime_generation_task.
        authorization_tasks: list[asyncio.Future[Any]] = [
            asyncio.ensure_future(speech_handle._wait_for_authorization()),
            asyncio.ensure_future(self._authorization_allowed.wait()),
        ]
        if speech_handle.allow_interruptions:
            authorization_tasks.append(asyncio.ensure_future(self._user_silence_event.wait()))
        try:
            await speech_handle.wait_if_not_interrupted(authorization_tasks)
        except asyncio.CancelledError:
            if realtime_audio_input_owner is not None:
                self._clear_realtime_input_if_owned(realtime_audio_input_owner)
            raise
        finally:
            await utils.aio.cancel_and_wait(*authorization_tasks)
        if speech_handle.interrupted:
            if realtime_audio_input_owner is not None:
                self._clear_realtime_input_if_owned(realtime_audio_input_owner)
            return

        realtime_audio_input_token: _RealtimeTurnTransaction | None
        if isinstance(realtime_audio_input_owner, asyncio.Future):
            try:
                await speech_handle.wait_if_not_interrupted([realtime_audio_input_owner])
            except asyncio.CancelledError:
                self._clear_realtime_input_if_owned(realtime_audio_input_owner)
                raise
            if speech_handle.interrupted:
                self._clear_realtime_input_if_owned(realtime_audio_input_owner)
                return
            if realtime_audio_input_owner.cancelled():
                return
            realtime_audio_input_token = realtime_audio_input_owner.result()
        else:
            realtime_audio_input_token = realtime_audio_input_owner

        if realtime_audio_input_token is not None:
            realtime_audio_input_token.speech_handle = speech_handle

        if (
            realtime_audio_input_token is not None
            and realtime_audio_input_token is not self._rt_audio_input_token
        ):
            return

        if text is not None:
            try:
                generation_ev = await self._rt_session.say(text)
            except llm.RealtimeError as e:
                logger.error("failed to say text: %s", str(e))
                speech_handle._mark_done(error=e)
                return

            await self._realtime_generation_task(
                speech_handle=speech_handle,
                generation_ev=generation_ev,
                model_settings=model_settings,
            )
            return

        # inside on_enter, hide flagged tools even when no tools= was passed (fall back to self.tools)
        turn_tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN
        tool_ctx = llm.ToolContext(tools if tools is not None else self.tools)
        on_enter_ignored = self._on_enter_ignored_tools(tool_ctx)
        if tools is not None or on_enter_ignored:
            tool_ctx._exclude(on_enter_ignored)
            turn_tools = tool_ctx.flatten()

        generate_reply_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        ori_tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN
        ori_tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN
        try:
            if not (
                per_response_tool_choice
                := self._rt_session.realtime_model.capabilities.per_response_tool_choice
            ):
                # update the tool and tool choice at the session level if they are specified
                if (
                    is_given(model_settings.tool_choice)
                    and model_settings.tool_choice != self._tool_choice
                ):
                    ori_tool_choice = self._tool_choice
                    self._rt_session.update_options(tool_choice=model_settings.tool_choice)

                if is_given(turn_tools):
                    ori_tools = self._rt_session.tools.flatten()
                    await self._rt_session.update_tools(turn_tools)

            rt_session = self._rt_session
            assert rt_session is not None

            async def _sync_and_start_generation() -> (
                asyncio.Future[llm.GenerationCreatedEvent] | None
            ):
                if (
                    realtime_audio_input_token is not None
                    and realtime_audio_input_token is not self._rt_audio_input_token
                ):
                    return None
                # Provider text is sent only after authorization and all awaited configuration
                # work. Once synchronized, call generate_reply without another scheduling point so
                # the provider can never retain a pending text turn without a settled request.
                if user_message is not None and not await self._sync_realtime_user_message(
                    speech_handle=speech_handle, user_message=user_message
                ):
                    return None
                if (
                    realtime_audio_input_token is not None
                    and realtime_audio_input_token is not self._rt_audio_input_token
                ):
                    return None
                if commit_realtime_audio:
                    assert realtime_audio_input_token is not None
                    rt_session.commit_audio()
                    realtime_audio_input_token.state = "input_submitted"
                if realtime_audio_input_token is not None:
                    realtime_audio_input_token.state = "generation_pending"
                generation_fut = rt_session.generate_reply(
                    instructions=instructions or NOT_GIVEN,
                    tool_choice=(
                        model_settings.tool_choice if per_response_tool_choice else NOT_GIVEN
                    ),
                    tools=(turn_tools if per_response_tool_choice else NOT_GIVEN),
                )
                if realtime_audio_input_token is not None:
                    realtime_audio_input_token.generation_fut = generation_fut
                return generation_fut

            start_generation_task = asyncio.create_task(
                _sync_and_start_generation(),
                name="AgentActivity.sync_and_start_realtime_generation",
            )
            try:
                generate_reply_fut = await asyncio.shield(start_generation_task)
            except asyncio.CancelledError:
                try:
                    cancelled_generation_fut = await start_generation_task
                except Exception:
                    logger.exception("failed to settle realtime input during cancellation")
                else:
                    if cancelled_generation_fut is not None and not cancelled_generation_fut.done():
                        cancelled_generation_fut.cancel()
                    elif (
                        cancelled_generation_fut is not None
                        and not cancelled_generation_fut.cancelled()
                    ):
                        self._interrupt_created_realtime_generation_if_owned(
                            cancelled_generation_fut, realtime_audio_input_token
                        )
                raise
            if generate_reply_fut is None:
                return
            await speech_handle.wait_if_not_interrupted([generate_reply_fut])
            if speech_handle.interrupted:
                # cancel the pending generation; the plugin emits response.cancel
                if not generate_reply_fut.done():
                    generate_reply_fut.cancel()
                elif not generate_reply_fut.cancelled():
                    self._interrupt_created_realtime_generation_if_owned(
                        generate_reply_fut, realtime_audio_input_token
                    )
                return

            try:
                generation_ev = await generate_reply_fut
            except llm.RealtimeError as e:
                logger.error(
                    "failed to generate a reply%s: %s",
                    " after tool execution" if tool_reply else "",
                    str(e),
                )
                speech_handle._mark_done(error=e)
                self._session._update_agent_state("listening")
                return

            if realtime_audio_input_token is not None:
                realtime_audio_input_token.state = "generation_created"
            if (
                realtime_audio_input_token is not None
                and realtime_audio_input_token is self._rt_audio_input_token
            ):
                try:
                    # A provider can keep a completed input trigger in flight until it emits
                    # generation_created. Keep the next activity quarantined until that
                    # acknowledgement so it cannot supersede the reply just requested.
                    self._advance_realtime_audio_input()
                except Exception:
                    # The provider already created the preceding generation. A failure while
                    # replaying later input belongs to that later turn and must not discard the
                    # valid response now in progress.
                    logger.exception(
                        "failed to advance deferred realtime input after generation started"
                    )

            # _realtime_generation_task will clear the authorization
            await self._realtime_generation_task(
                speech_handle=speech_handle,
                generation_ev=generation_ev,
                model_settings=model_settings,
                instructions=instructions,
            )
        finally:
            # Direct task cancellation must settle the provider-side request as well. SpeechHandle
            # interruption already follows this path above, but session close/task teardown may
            # cancel the coroutine itself while generate_reply() is still pending.
            if generate_reply_fut is not None and not generate_reply_fut.done():
                generate_reply_fut.cancel()
            # reset tool_choice and tools
            if is_given(ori_tool_choice):
                try:
                    self._rt_session.update_options(tool_choice=ori_tool_choice)
                except Exception:
                    logger.exception("failed to reset tool_choice")

            if is_given(ori_tools):
                try:
                    await self._rt_session.update_tools(ori_tools)
                except Exception:
                    logger.exception("failed to reset tools")

            # A setup failure may discard only the audio input captured by this reply task.
            if realtime_audio_input_token is not None:
                try:
                    self._clear_realtime_input_if_owned(realtime_audio_input_token)
                except Exception:
                    logger.exception(
                        "failed to clear realtime input after generation setup failure"
                    )

    @utils.log_exceptions(logger=logger)
    async def _realtime_generation_task(
        self,
        *,
        speech_handle: SpeechHandle,
        generation_ev: llm.GenerationCreatedEvent,
        model_settings: ModelSettings,
        instructions: str | None = None,
    ) -> None:
        with tracer.start_as_current_span(
            "agent_turn", context=self._session._root_span_context
        ) as current_span:
            current_span.set_attribute(trace_types.ATTR_AGENT_TURN_ID, speech_handle._generation_id)
            if parent_id := speech_handle._parent_generation_id:
                current_span.set_attribute(trace_types.ATTR_AGENT_PARENT_TURN_ID, parent_id)
            speech_handle._agent_turn_context = otel_context.get_current()

            try:
                await self._realtime_generation_task_impl(
                    speech_handle=speech_handle,
                    generation_ev=generation_ev,
                    model_settings=model_settings,
                    instructions=instructions,
                )
            finally:
                if self._active_realtime_generation is generation_ev:
                    self._active_realtime_generation = None

    async def _realtime_generation_task_impl(
        self,
        *,
        speech_handle: SpeechHandle,
        generation_ev: llm.GenerationCreatedEvent,
        model_settings: ModelSettings,
        instructions: str | None = None,
    ) -> None:
        current_span = trace.get_current_span(context=speech_handle._agent_turn_context)
        current_span.set_attribute(trace_types.ATTR_SPEECH_ID, speech_handle.id)

        room_io = self._session._room_io
        if room_io and room_io.room.isconnected():
            _set_participant_attributes(current_span, room_io.room.local_participant)

        assert self._rt_session is not None, "rt_session is not available"
        assert isinstance(self.llm, llm.RealtimeModel), "llm is not a realtime model"

        current_span.set_attributes(
            {
                trace_types.ATTR_GEN_AI_OPERATION_NAME: "chat",
                trace_types.ATTR_GEN_AI_PROVIDER_NAME: self.llm.provider,
                trace_types.ATTR_GEN_AI_REQUEST_MODEL: self.llm.model,
            }
        )
        if self._realtime_spans is not None and generation_ev.response_id:
            self._realtime_spans[generation_ev.response_id] = current_span

        audio_output = self._session.output.audio if self._session.output.audio_enabled else None
        text_output = (
            self._session.output.transcription
            if self._session.output.transcription_enabled
            else None
        )
        tool_ctx = llm.ToolContext(self.tools)

        authorization_tasks: list[asyncio.Future[Any]] = [
            asyncio.ensure_future(speech_handle._wait_for_authorization()),
            asyncio.ensure_future(self._authorization_allowed.wait()),
        ]
        if speech_handle.allow_interruptions:
            authorization_tasks.append(asyncio.ensure_future(self._user_silence_event.wait()))
        try:
            await speech_handle.wait_if_not_interrupted(authorization_tasks)
        finally:
            await utils.aio.cancel_and_wait(*authorization_tasks)
        speech_handle._clear_authorization()

        if speech_handle.interrupted:
            current_span.set_attribute(trace_types.ATTR_SPEECH_INTERRUPTED, True)
            return  # TODO(theomonnom): remove the message from the serverside history

        started_speaking_at: float | None = None
        stopped_speaking_at: float | None = None
        started_forwarding_at: float | None = None

        def _on_first_frame(
            fut: asyncio.Future[float] | asyncio.Future[None], audio_out: _AudioOutput | None = None
        ) -> None:
            """
            Callback to update the agent state when the first frame is captured:
            1. _AudioOutput.first_frame_fut (float)
            2. _TextOutput.first_text_fut (None)
            """
            nonlocal started_speaking_at, started_forwarding_at
            # only the first message's first frame should trigger state transitions
            if started_speaking_at is not None:
                return
            try:
                started_speaking_at = fut.result() or time.time()
                started_forwarding_at = (
                    audio_out.started_forwarding_at
                    if audio_out and audio_out.started_forwarding_at is not None
                    else started_speaking_at
                )
            except BaseException:
                return

            self._session._update_agent_state(
                "speaking",
                start_time=started_speaking_at,
                otel_context=speech_handle._agent_turn_context,
            )
            if self._audio_recognition:
                self._audio_recognition._on_start_of_agent_speech(started_at=started_speaking_at)
            if self.interruption_enabled:
                self._disable_vad_interruption_soon()

        tasks: list[asyncio.Task[Any]] = []
        tees: list[utils.aio.itertools.Tee[Any]] = []

        read_transcript_from_tts = False

        # multiple message items may be produced for a single realtime response
        # (e.g. GPT-Realtime-2.0). We process each one serially: push frames,
        # flush, wait_for_playout
        @dataclass
        class _MsgOutput:
            msg: MessageGeneration
            out: _ForwardOutput

        message_outputs: list[_MsgOutput] = []

        async def _process_one_message(msg: MessageGeneration) -> _MsgOutput:
            """Resolve a message's audio/text sources, then forward and wait for playout."""
            nonlocal read_transcript_from_tts
            assert isinstance(self.llm, llm.RealtimeModel)

            msg_modalities = await msg.modalities
            tts_text_input: AsyncIterable[str] | None = None
            if "audio" not in msg_modalities and self.tts:
                if self.llm.capabilities.audio_output:
                    logger.warning(
                        "text response received from realtime API, falling back to use a TTS model."  # noqa: E501
                    )
                tee = utils.aio.itertools.tee(msg.text_stream, 2)
                tts_text_input, tr_text_input = tee
                tees.append(tee)
            else:
                tr_text_input = msg.text_stream.__aiter__()

            audio_source: AsyncIterable[rtc.AudioFrame] | None = None
            if audio_output is not None:
                if tts_text_input is not None:
                    tts_task, tts_gen_data = perform_tts_inference(
                        node=self._agent.tts_node,
                        input=tts_text_input,
                        model_settings=model_settings,
                        text_transforms=self._session.options.tts_text_transforms,
                        model=self.tts.model if self.tts else None,
                        provider=self.tts.provider if self.tts else None,
                    )

                    if (
                        self.use_tts_aligned_transcript
                        and (tts := self.tts)
                        and (tts.capabilities.aligned_transcript or not tts.capabilities.streaming)
                        and (timed_texts := await tts_gen_data.timed_texts_fut)
                    ):
                        tr_text_input = _aligned_transcript_or_text(timed_texts, tr_text_input)
                        read_transcript_from_tts = True

                    tasks.append(tts_task)
                    audio_source = tts_gen_data.audio_ch
                elif "audio" in msg_modalities:
                    realtime_audio = self._agent.realtime_audio_output_node(
                        msg.audio_stream, model_settings
                    )
                    audio_source = (
                        await realtime_audio
                        if asyncio.iscoroutine(realtime_audio)
                        else realtime_audio
                    )
                elif self.llm.capabilities.audio_output:
                    logger.error(
                        "Text message received from Realtime API with audio modality. "
                        "This usually happens when text chat context is synced to the API. "
                        "Try to add a TTS model as fallback or use text modality with TTS instead."  # noqa: E501
                    )
                else:
                    logger.warning(
                        "audio output is enabled but neither tts nor realtime audio is available",
                    )

            tr_node = self._agent.transcription_node(tr_text_input, model_settings)
            text_source = await tr_node if asyncio.iscoroutine(tr_node) else tr_node

            out = await forward_generation(
                speech_handle=speech_handle,
                audio_output=audio_output,
                text_output=text_output,
                audio_source=audio_source,
                text_source=text_source,
                on_first_frame=_on_first_frame,
                reconcile_playout_pause=lambda: self._reconcile_playout_pause(speech_handle),
            )
            return _MsgOutput(msg=msg, out=out)

        @utils.log_exceptions(logger=logger)
        async def _process_messages() -> None:
            async for msg in generation_ev.message_stream:
                if speech_handle.interrupted:
                    # remaining messages are left out of message_outputs so
                    # update_chat_ctx below removes them server-side.
                    break
                entry = await _process_one_message(msg)
                message_outputs.append(entry)
                if entry.out.played == "partial":
                    break

        process_msg_task = asyncio.create_task(
            _process_messages(), name="AgentActivity.realtime_generation.process_messages"
        )
        tasks.append(process_msg_task)

        # read function calls
        fnc_tee = utils.aio.itertools.tee(generation_ev.function_stream, 2)
        fnc_stream, fnc_stream_for_tracing = fnc_tee
        tees.append(fnc_tee)
        function_calls: list[llm.FunctionCall] = []

        async def _read_fnc_stream() -> None:
            async for fnc in fnc_stream_for_tracing:
                function_calls.append(fnc)

        tasks.append(
            asyncio.create_task(
                _read_fnc_stream(),
                name="AgentActivity.realtime_generation.read_fnc_stream",
            )
        )

        # messages in RunResult are ordered by the `created_at` field
        def _tool_execution_started_cb(fnc_call: llm.FunctionCall) -> None:
            # function call is created during the realtime generation, before the assistant
            # message it belongs to is placed at `started_speaking_at`
            # reset the `created_at` to the start time of the tool execution
            fnc_call.created_at = time.time()
            speech_handle._item_added([fnc_call])
            self._agent._chat_ctx._upsert_item(fnc_call)
            self._session._tool_items_added([fnc_call])

        def _tool_execution_completed_cb(out: ToolExecutionOutput) -> None:
            speech_handle._item_added([out.fnc_call_out])

        exe_task, tool_output = perform_tool_executions(
            session=self._session,
            speech_handle=speech_handle,
            tool_ctx=tool_ctx,
            tool_choice=model_settings.tool_choice,
            function_stream=fnc_stream,
            tool_execution_started_cb=_tool_execution_started_cb,
            tool_execution_completed_cb=_tool_execution_completed_cb,
        )

        await speech_handle.wait_if_not_interrupted([*tasks])

        current_span.set_attribute(trace_types.ATTR_SPEECH_INTERRUPTED, speech_handle.interrupted)
        current_span.set_attribute(
            trace_types.ATTR_RESPONSE_FUNCTION_CALLS,
            json.dumps([fnc.model_dump(exclude={"type", "created_at"}) for fnc in function_calls]),
        )

        # _process_messages handles its own playout waits and interrupt cleanup
        await process_msg_task

        if audio_output is not None:
            self._session._update_agent_state(
                "thinking" if self._background_speeches else "listening"
            )
            if self._audio_recognition:
                self._audio_recognition._on_end_of_agent_speech(
                    ignore_user_transcript_until=time.time()
                )
            if self.interruption_enabled:
                self._restore_interruption_by_audio_activity()
            current_span.set_attribute(
                trace_types.ATTR_SPEECH_INTERRUPTED, speech_handle.interrupted
            )

        stopped_speaking_at = time.time()

        def _create_assistant_message(
            message_id: str, forwarded_text: str, interrupted: bool
        ) -> llm.ChatMessage:
            assistant_metrics: llm.MetricsReport = {}

            if generation_ev.response_id:
                assistant_metrics["provider_request_ids"] = [generation_ev.response_id]

            if stopped_speaking_at and started_speaking_at:
                assistant_metrics["started_speaking_at"] = started_speaking_at
                assistant_metrics["stopped_speaking_at"] = stopped_speaking_at

                if started_forwarding_at is not None:
                    assistant_metrics["playback_latency"] = (
                        started_speaking_at - started_forwarding_at
                    )

            msg = llm.ChatMessage(
                role="assistant",
                content=[forwarded_text],
                id=message_id,
                interrupted=interrupted,
            )
            if started_speaking_at is not None:
                msg.created_at = started_speaking_at
            msg.metrics = assistant_metrics
            return msg

        if (
            not speech_handle.interrupted
            and read_transcript_from_tts
            and any(
                e.out.played != "skipped" and e.out.text_out is not None and not e.out.text_out.text
                for e in message_outputs
            )
        ):
            logger.warning(
                "`use_tts_aligned_transcript` is enabled but no agent transcript was returned from tts"  # noqa: E501
            )

        # create assistant message per generated message
        trace_text_parts: list[str] = []
        any_skipped = False
        for entry in message_outputs:
            if entry.out.played == "skipped":
                any_skipped = True
                continue

            msg_interrupted = entry.out.played == "partial"
            forwarded_text = entry.out.forwarded_text

            if msg_interrupted and self.llm.capabilities.message_truncation:
                msg_modalities = await entry.msg.modalities
                self._rt_session.truncate(
                    message_id=entry.msg.message_id,
                    modalities=msg_modalities,
                    audio_end_ms=int(entry.out.playback_position * 1000),
                    audio_transcript=forwarded_text,
                )

            if not forwarded_text:
                continue

            trace_text_parts.append(forwarded_text)
            chat_msg = _create_assistant_message(
                message_id=entry.msg.message_id,
                forwarded_text=forwarded_text,
                interrupted=msg_interrupted,
            )
            self._agent._chat_ctx._upsert_item(chat_msg)
            speech_handle._item_added([chat_msg])
            self._session._conversation_item_added(chat_msg)

        if trace_text_parts:
            current_span.set_attribute(trace_types.ATTR_RESPONSE_TEXT, "\n".join(trace_text_parts))

        # sync local chat ctx to the realtime server to remove any items the
        # model added but the user never heard (interrupted before we pulled
        # them, or message_outputs entries left in "skipped")
        if speech_handle.interrupted and any_skipped and self.llm.capabilities.mutable_chat_context:
            try:
                await self._rt_session.update_chat_ctx(self._agent._chat_ctx)
            except llm.RealtimeError as e:
                logger.warning(
                    "failed to sync chat context to remove never-played messages",
                    extra={"error": str(e)},
                )

        for entry in message_outputs:
            if entry.out.audio_out is not None and not entry.out.audio_out.first_frame_fut.done():
                entry.out.audio_out.first_frame_fut.cancel()

        for tee in tees:
            await tee.aclose()
        speech_handle._mark_generation_done()

        if speech_handle.interrupted:
            await utils.aio.cancel_and_wait(exe_task)

            # commit results of tools that finished despite the interruption, as the pipeline
            # task does. the calls are already recorded, so each one answers or the model waits
            interrupted_calls: list[llm.FunctionCall] = []
            interrupted_fnc_outputs: list[llm.FunctionCallOutput] = []
            for sanitized_out in tool_output.output:
                interrupted_calls.append(sanitized_out.fnc_call)
                interrupted_fnc_outputs.append(_interrupted_tool_output(sanitized_out))

            if interrupted_fnc_outputs:
                self._session.emit(
                    "function_tools_executed",
                    FunctionToolsExecutedEvent(
                        function_calls=interrupted_calls,
                        function_call_outputs=interrupted_fnc_outputs,
                    ),
                )
                self._agent._chat_ctx.insert(interrupted_fnc_outputs)
                self._session._tool_items_added(interrupted_fnc_outputs)

                # unlike the pipeline, a realtime model holds the call open server-side
                chat_ctx = self._rt_session.chat_ctx.copy()
                chat_ctx.items.extend(interrupted_fnc_outputs)
                try:
                    await self._rt_session.update_chat_ctx(chat_ctx)
                except llm.RealtimeError as e:
                    logger.warning(
                        "failed to sync the tool results of an interrupted generation",
                        extra={"error": str(e)},
                    )
            return

        # wait for the tool execution to complete
        tool_output.first_tool_started_fut.add_done_callback(
            lambda _: self._session._update_agent_state("thinking")
        )

        self._background_speeches.add(speech_handle)
        try:
            await exe_task
        finally:
            self._background_speeches.discard(speech_handle)

        # important: no agent output should be used after this point

        tool_reply_expected = False
        if len(tool_output.output) > 0:
            speech_handle._num_steps += 1

            new_fnc_outputs: list[llm.FunctionCallOutput] = []
            fnc_executed_ev = FunctionToolsExecutedEvent(
                function_calls=[], function_call_outputs=[]
            )
            new_agent_task: Agent | None = None
            ignore_task_switch = False

            for sanitized_out in tool_output.output:
                fnc_executed_ev.function_calls.append(sanitized_out.fnc_call)
                fnc_executed_ev.function_call_outputs.append(sanitized_out.fnc_call_out)

                new_fnc_outputs.append(sanitized_out.fnc_call_out)

                # add tool output to the chat context
                self._agent._chat_ctx._upsert_item(sanitized_out.fnc_call_out)
                self._session._tool_items_added([sanitized_out.fnc_call_out])

                if new_agent_task is not None and sanitized_out.agent_task is not None:
                    logger.error(
                        "expected to receive only one Agent from the tool executions",
                    )
                    ignore_task_switch = True

                new_agent_task = sanitized_out.agent_task

            if new_agent_task and not ignore_task_switch:
                fnc_executed_ev._handoff_required = True

            self._session.emit("function_tools_executed", fnc_executed_ev)

            draining = self.scheduling_paused
            if fnc_executed_ev._handoff_required and new_agent_task and not ignore_task_switch:
                self._session.update_agent(new_agent_task)
                draining = True

            if len(new_fnc_outputs) > 0:
                # wait all speeches played before updating the tool output and generating the response
                # most realtime models don't support generating multiple responses at the same time
                while self._current_speech or self._speech_q:
                    if (
                        self._current_speech
                        and not self._current_speech.done()
                        and self._current_speech is not speech_handle
                    ):
                        await self._current_speech
                    else:
                        await asyncio.sleep(0)

                # if the realtime model auto-generates the tool reply, install a
                # placeholder so the active RunResult waits for that reply
                auto_reply_fut: asyncio.Future[None] | None = None
                if (
                    self._rt_session.capabilities.auto_tool_reply_generation
                    and fnc_executed_ev.has_tool_reply
                    and self._pending_auto_tool_reply_fut is None
                    and (run_state := self._session._global_run_state) is not None
                    and not run_state.done()
                ):
                    auto_reply_fut = asyncio.get_event_loop().create_future()
                    self._pending_auto_tool_reply_fut = auto_reply_fut
                    llm_label = self.llm._label

                    async def _wait_for_auto_tool_reply() -> None:
                        try:
                            await asyncio.wait_for(asyncio.shield(auto_reply_fut), 5.0)
                        except asyncio.TimeoutError:
                            logger.warning(
                                "timed out waiting for realtime auto tool reply from %s",
                                llm_label,
                            )
                        finally:
                            if self._pending_auto_tool_reply_fut is auto_reply_fut:
                                self._pending_auto_tool_reply_fut = None

                    task = asyncio.create_task(_wait_for_auto_tool_reply())
                    run_state._watch_handle(task)

                chat_ctx = self._rt_session.chat_ctx.copy()
                chat_ctx.items.extend(new_fnc_outputs)
                try:
                    await self._rt_session.update_chat_ctx(chat_ctx)
                except llm.RealtimeError as e:
                    logger.warning(
                        "failed to update chat context before generating the function calls results",  # noqa: E501
                        extra={"error": str(e)},
                    )
                    if auto_reply_fut is not None and not auto_reply_fut.done():
                        if self._pending_auto_tool_reply_fut is auto_reply_fut:
                            self._pending_auto_tool_reply_fut = None
                        auto_reply_fut.set_result(None)

            tool_reply_expected = fnc_executed_ev.has_tool_reply
            if tool_reply_expected and not self._rt_session.capabilities.auto_tool_reply_generation:
                self._rt_session.interrupt()

                self._create_speech_task(
                    self._realtime_reply_task(
                        speech_handle=speech_handle,
                        model_settings=ModelSettings(
                            # Avoid setting tool_choice to "required" or a specific function when
                            # passing tool response back to the LLM
                            tool_choice="none"
                            if draining or model_settings.tool_choice == "none"
                            else "auto",
                        ),
                        tool_reply=True,
                    ),
                    speech_handle=speech_handle,
                    name="AgentActivity.realtime_reply",
                )
                self._schedule_speech(
                    speech_handle, SpeechHandle.SPEECH_PRIORITY_NORMAL, force=True
                )

        # no reply follows, so nothing else clears the "thinking" the tool asserted
        if not tool_reply_expected and self._no_pending_speech:
            self._session._update_agent_state(
                "thinking" if self._background_speeches else "listening"
            )

    def _update_paused_speech(self, speech_handle: SpeechHandle, timeout: float) -> None:
        """Record that ``speech_handle`` is paused.

        If already paused for this handle, only ``timeout`` is updated — the
        ``agent_state`` captured at first pause is preserved, so the resume
        path restores the correct state even across multiple calls.
        """
        if self._paused_speech and self._paused_speech.handle is speech_handle:
            self._paused_speech.timeout = timeout
        else:
            self._paused_speech = _PausedSpeechInfo(
                handle=speech_handle,
                agent_state=self._session.agent_state,
                timeout=timeout,
            )

    def _pause_enabled(self) -> bool:
        interruption_options = self._session.options.interruption
        return bool(
            interruption_options["resume_false_interruption"]
            and interruption_options["false_interruption_timeout"] is not None
            and self._session.output.audio_enabled
            and self._session.output.audio
            and self._session.output.audio.can_pause
        )

    def _reconcile_playout_pause(self, speech_handle: SpeechHandle) -> None:
        """Preserve, apply, or release a speech pause before forwarding audio."""
        audio_output = self._session.output.audio
        pause_is_allowed = (
            self._pause_enabled()
            and not speech_handle.interrupted
            and speech_handle.allow_interruptions
        )
        pause_is_valid = (
            self._paused_speech is not None
            and self._paused_speech.handle is speech_handle
            and pause_is_allowed
        )
        if pause_is_valid:
            # a paused playout stay paused regardless of forwarding status
            return

        # clear stale _paused_speech ref
        if self._paused_speech is not None:
            self._cancel_false_interruption_timer()
            self._paused_speech = None

        if (
            pause_is_allowed
            and self._session.agent_state != "speaking"
            and not self._user_silence_event.is_set()
        ):
            assert audio_output is not None
            # SOS arrived before this handle became current so we pause here
            # EOS/transcripts/turn commit will resolve the pause eventually
            self._update_paused_speech(speech_handle, timeout=0)
            audio_output.pause()
            return

        if audio_output is not None:
            audio_output.resume()

    def _cancel_false_interruption_timer(self) -> None:
        if self._false_interruption_timer is not None:
            self._false_interruption_timer.cancel()
            self._false_interruption_timer = None
        self._false_interruption_pending = False

    def _start_false_interruption_timer(self, timeout: float) -> None:
        self._cancel_false_interruption_timer()

        def _on_false_interruption() -> None:
            if self._paused_speech is None or (
                self._current_speech and self._current_speech is not self._paused_speech.handle
            ):
                # already new speech is scheduled, do nothing
                self._paused_speech = None
                return

            resumed = False
            if (
                self._session.options.interruption["resume_false_interruption"]
                and (audio_output := self._session.output.audio)
                and audio_output.can_pause
                and not self._paused_speech.handle.done()
            ):
                self._session._update_agent_state(
                    self._paused_speech.agent_state,
                    otel_context=self._paused_speech.handle._agent_turn_context,
                )
                if self._audio_recognition and self._paused_speech.agent_state == "speaking":
                    self._audio_recognition._on_start_of_agent_speech(started_at=time.time())
                if self.interruption_enabled:
                    self._disable_vad_interruption_soon()
                audio_output.resume()
                resumed = True
                logger.debug("resumed false interrupted speech", extra={"timeout": timeout})

            self._session.emit(
                "agent_false_interruption", AgentFalseInterruptionEvent(resumed=resumed)
            )

            self._paused_speech = None
            self._false_interruption_timer = None

        def _on_turn_settled(settled: asyncio.Task[None]) -> None:
            if not self._false_interruption_pending:
                return  # the turn committed, or the pause was resolved another way

            recognition = self._audio_recognition
            eot_task = recognition._end_of_turn_task if recognition else None
            # a newer decision superseded this one (e.g. an stt final re-armed the bounce)
            if eot_task is not None and eot_task is not settled and not eot_task.done():
                eot_task.add_done_callback(_on_turn_settled)
                return

            self._false_interruption_pending = False
            if settled.cancelled() or (recognition is not None and recognition._closing.is_set()):
                return  # torn down instead of decided; closing releases the pause itself

            _on_false_interruption()

        def _on_timeout() -> None:
            self._false_interruption_timer = None

            # an open turn decision owns the paused speech: it either commits and interrupts it
            # or drops it, and only then is the interruption known to be false
            eot_task = (
                self._audio_recognition._end_of_turn_task if self._audio_recognition else None
            )
            if eot_task is not None and not eot_task.done():
                self._false_interruption_pending = True
                eot_task.add_done_callback(_on_turn_settled)
                return

            _on_false_interruption()

        self._false_interruption_timer = self._session._loop.call_later(timeout, _on_timeout)

    async def _cancel_speech_pause(
        self, old_task: asyncio.Task[None] | None = None, *, interrupt: bool = True
    ) -> None:
        """Clear a speech pause and optionally interrupt its handle.

        Final STT transcripts and committed turns that generate replies use
        ``interrupt=True``. Activity shutdown uses ``interrupt=False`` because the
        scheduling task owns the speech.
        """
        if old_task is not None:
            try:
                await old_task
            except Exception:
                # Don't let a failed prior task poison subsequent turn completions.
                # This can happen when _wait_for_generation raises because
                # the paused speech had no active generation (race condition).
                logger.debug("previous _cancel_speech_pause task failed, ignoring")

        self._cancel_false_interruption_timer()

        if not self._paused_speech:
            return

        if (
            interrupt
            and not self._paused_speech.handle.interrupted
            and self._paused_speech.handle.allow_interruptions
        ):
            self._paused_speech.handle.interrupt()
            # ensure the generation is done — but only if a generation
            # was actually started; a paused speech that was never
            # authorized won't have an active generation future.
            if self._paused_speech.handle._generations:
                await self._paused_speech.handle._wait_for_generation()
        self._paused_speech = None

        if (
            self._session.options.interruption["resume_false_interruption"]
            and self._session.output.audio
        ):
            self._session.output.audio.resume()

    def _disable_vad_interruption_soon(self) -> None:
        """Disable VAD interruption after the backchannel boundary expires."""
        if self._audio_recognition and self._audio_recognition._backchannel_boundary_active:

            def _disable_vad_interruption() -> None:
                # only disable it if the agent is still speaking
                if (
                    self._session.agent_state == "speaking"
                    and self._interruption_by_audio_activity_enabled
                ):
                    logger.trace("backchannel boundary expired")
                    self._interruption_by_audio_activity_enabled = False

            self._audio_recognition._backchannel_boundary_callback = _disable_vad_interruption
        else:
            self._interruption_by_audio_activity_enabled = False

    def _restore_interruption_by_audio_activity(self) -> None:
        if self._audio_recognition:
            self._audio_recognition._cancel_backchannel_boundary()

        self._interruption_by_audio_activity_enabled = (
            self._default_interruption_by_audio_activity_enabled
        )

    def _fallback_to_vad_interruption(
        self, error: inference.InterruptionDetectionError | None = None
    ) -> None:
        """Degrade gracefully from adaptive interruption to VAD-based interruption.

        Called when the adaptive interruption detector encounters an unrecoverable error.
        Re-enables audio-activity interruption so VAD events can trigger interruptions,
        and flushes any held transcripts that were waiting on the detector.
        """
        if not self._interruption_detection_enabled:
            return

        self._interruption_detection_enabled = False
        self._restore_interruption_by_audio_activity()

        if isinstance(self._interruption_detector, inference.AdaptiveInterruptionDetector):
            self._interruption_detector.off("metrics_collected", self._on_metrics_collected)
            self._interruption_detector.off("error", self._on_error)
            self._interruption_detector.off("overlapping_speech", self._on_overlap_speech_ended)

        if self._audio_recognition:
            # this also releases any held transcripts
            self._audio_recognition._update_interruption_detection(None)

        logger.info(
            "adaptive interruption disabled due to unrecoverable error, "
            "falling back to VAD-based interruption",
            extra={
                "error": str(error.error) if error is not None else None,
                "label": error.label if error is not None else None,
            },
        )

    def _init_metrics_from_end_of_turn(self, info: _EndOfTurnInfo) -> llm.MetricsReport:
        metrics_report: llm.MetricsReport = {}
        if self.stt:
            metrics_report["stt_metadata"] = self.stt.metrics_metadata
        if info.metrics.started_speaking_at is not None:
            metrics_report["started_speaking_at"] = info.metrics.started_speaking_at

        if info.metrics.stopped_speaking_at is not None:
            metrics_report["stopped_speaking_at"] = info.metrics.stopped_speaking_at

        if info.metrics.transcription_delay is not None:
            metrics_report["transcription_delay"] = info.metrics.transcription_delay

        if info.metrics.end_of_turn_delay is not None:
            metrics_report["end_of_turn_delay"] = info.metrics.end_of_turn_delay

        return metrics_report

    # move them to the end to avoid shadowing the same named modules for mypy
    @property
    def _text_only(self) -> bool:
        # text simulations run without audio: no STT/TTS/VAD
        return self._session._text_only

    @property
    def vad(self) -> vad.VAD | None:
        if self._text_only:
            return None
        return self._agent.vad if is_given(self._agent.vad) else self._session.vad

    @property
    def using_default_vad(self) -> bool:
        if is_given(self._agent.vad):
            return False
        return self._session._using_default_vad

    def _resolve_interruption_detection(self) -> inference.AdaptiveInterruptionDetector | None:
        realtime_llm = self.llm if isinstance(self.llm, llm.RealtimeModel) else None
        if realtime_llm is not None:
            # realtime commits turns manually; barge-in withholds the commit, so no STT is needed
            can_gatekeep = not self._rt_turn_detection_enabled
        else:
            # the STT pipeline gatekeeps by holding and flushing transcripts
            can_gatekeep = (
                self.stt is not None
                and self.stt.capabilities.aligned_transcript
                and self.stt.capabilities.streaming
            )

        if (
            not can_gatekeep
            or self.vad is None
            or self._turn_detection in ("manual", "realtime_llm")
        ):
            if (
                is_given(self._agent.interruption_detection)
                and self._agent.interruption_detection == "adaptive"
            ) or (
                is_given(self._session.interruption_detection)
                and self._session.interruption_detection == "adaptive"
            ):
                logger.warning(
                    "interruption_detection is provided, but it's not compatible with the current configuration and will be disabled"
                )
            return None

        if not self.allow_interruptions:
            return None

        if (
            is_given(self._agent.interruption_detection)
            and self._agent.interruption_detection == "vad"
        ):
            return None
        if (
            is_given(self._session.interruption_detection)
            and self._session.interruption_detection == "vad"
        ):
            return None

        if (
            not is_given(self._agent.interruption_detection)
            and not is_given(self._session.interruption_detection)
            and not utils.is_hosted()
            and not utils.is_dev_mode()
        ):
            logger.info("adaptive interruption is disabled by default in production mode")
            return None

        try:
            detector = inference.AdaptiveInterruptionDetector()
        except ValueError as e:
            logger.warning("failed to create AdaptiveInterruptionDetector", extra={"error": str(e)})
            return None

        return detector

    @property
    def stt(self) -> stt.STT | None:
        if self._text_only:
            return None
        return self._agent.stt if is_given(self._agent.stt) else self._session.stt

    @property
    def llm(self) -> llm.LLM | llm.RealtimeModel | None:
        return self._agent.llm if is_given(self._agent.llm) else self._session.llm

    @property
    def tts(self) -> tts.TTS | None:
        if self._text_only:
            return None
        return self._agent.tts if is_given(self._agent.tts) else self._session.tts
