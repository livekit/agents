"""Native voice-session construction: public configuration and callback signatures, their
adaptation to native arguments, session startup and the remote session protocol."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import weakref
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Callable, Coroutine
from typing import TYPE_CHECKING, Any, Literal, cast

from typing_extensions import TypeVar

from livekit import rtc
from livekit.agents._generated.recording_options import RecordingOptions as RecordingOptions

from .. import _bindings, _core, inference, llm, stt, tts
from .._bindings import TurnDetector, tool_specs
from .._generated.connect import SessionConnectOptions as SessionConnectOptions
from .._generated.keyterms import STTContextOptions
from .._generated.session_options import SessionKwargs
from .._generated.session_owner import ProviderBindings
from .._generated.session_wire import DEFAULT_HOOK_BOUND, DEFAULT_SILENT_FPS, DEFAULT_SPEAKING_FPS
from .._generated.vad import VAD
from .._generated.voice_session import UpdateAgentArgs, VoiceCommandOriginHook
from ..inference.llm import llm_from_model_string
from ..llm import (
    LLM,
    ChatContext,
    RealtimeModel,
    function_tool,
)
from ..log import logger
from ..types import (
    NOT_GIVEN,
    FlushSentinel,
    NotGivenOr,
    TranscriptChunk,
)
from ..utils import is_dev_mode, is_given, is_hosted, shortuuid
from . import Agent, ModelSettings, _BindingContextVar, _VoiceBinding, io
from ._session_callbacks import _DynamicInstructions, _hook_name, _origin, _SessionNodes
from .events import (
    Context_T,
    RunContext,
    UserTurnExceededEvent,
)
from .io import SpeechHandle

if TYPE_CHECKING:
    from .._core import VoiceAgent as _NativeAgent
    from .._generated.session_options import SessionLegs, UpdateKwargs
    from ..storage import Thread
    from .agent_session import VoiceSession
    from .io import EndpointingOptions, TurnDetectionMode
import time
from typing import TYPE_CHECKING

from .._bindings import realtime_model
from .._generated.voice_session import VoiceSessionArgs
from ..job import get_job_context
from .events import CloseReason
from .room_io import Room as _RtcRoom, RoomOptions

if TYPE_CHECKING:
    from .agent_session import VoiceSession
from typing import TYPE_CHECKING, ClassVar

from google.protobuf.json_format import MessageToDict

from .._generated.remote_session_wire import STATE_OPTION_FIELDS
from .._generated.usage import Usage
from ..version import __version__

if TYPE_CHECKING:
    from .agent_session import VoiceSession


TextTransforms = Literal["filter_markdown", "filter_emoji"]
"""A transform the core applies to the text it sends to the TTS."""


# on_exit and the other callbacks are awaited by the core this long before it moves on;
# on_user_turn_completed is awaited until it returns
_HOOK_BOUND = DEFAULT_HOOK_BOUND

# the session's pipeline callbacks (voice-session.md "Customizing the pipeline"): each receives
# the call's `RunContext`, whose `agent` is the agent holding the call
_C = TypeVar("_C")
_Item = TypeVar("_Item")
_Node = AsyncIterable[_Item] | Coroutine[Any, Any, AsyncIterable[_Item] | None]
LLMNode = Callable[
    [RunContext[_C], ChatContext, list[llm.Tool], ModelSettings],
    AsyncIterable[llm.ChatChunk | str | FlushSentinel]
    | Coroutine[Any, Any, AsyncIterable[llm.ChatChunk | str | FlushSentinel] | str | None],
]
STTNode = Callable[
    [RunContext[_C], AsyncIterable[rtc.AudioFrame], ModelSettings], _Node[stt.SpeechEvent]
]
TTSNode = Callable[[RunContext[_C], AsyncIterable[str], ModelSettings], _Node[rtc.AudioFrame]]
TranscriptionNode = Callable[
    [RunContext[_C], AsyncIterable[TranscriptChunk], ModelSettings], _Node[str | TranscriptChunk]
]
UserTurnCompletedHook = Callable[
    [RunContext[_C], ChatContext, llm.ChatMessage], Coroutine[Any, Any, None]
]
UserTurnExceededHook = Callable[[RunContext[_C], UserTurnExceededEvent], Coroutine[Any, Any, None]]


class VoiceActivityVideoSampler:
    """Sample the camera at ``speaking_fps`` while the user speaks, ``silent_fps`` otherwise."""

    def __init__(
        self, *, speaking_fps: float = DEFAULT_SPEAKING_FPS, silent_fps: float = DEFAULT_SILENT_FPS
    ) -> None:
        self.speaking_fps = speaking_fps
        self.silent_fps = silent_fps


_CallbackT = TypeVar("_CallbackT", bound=Callable[..., object])


for _type in (RecordingOptions, VoiceActivityVideoSampler):
    _type.__module__ = "livekit.agents.voice.agent_session"


_OPTION_KEYS = (SessionKwargs.__required_keys__ | SessionKwargs.__optional_keys__) - {
    "conn_options"
}
_NODES = ("llm_node", "tts_node", "stt_node", "transcription_node")
_MODELS = ("stt", "tts", "vad")


def initialize(
    session: VoiceSession[Context_T],
    *,
    stt: NotGivenOr[stt.STT | str] = NOT_GIVEN,
    vad: NotGivenOr[VAD | None] = NOT_GIVEN,
    llm: NotGivenOr[llm.LLM | llm.RealtimeModel | str] = NOT_GIVEN,
    tts: NotGivenOr[tts.TTS | str] = NOT_GIVEN,
    tools: NotGivenOr[list[llm.Tool | llm.Toolset]] = NOT_GIVEN,
    context: Context_T | None = None,
    thread: Thread | None = None,
    video_sampler: NotGivenOr[VoiceActivityVideoSampler | None] = NOT_GIVEN,
    conn_options: NotGivenOr[SessionConnectOptions] = NOT_GIVEN,
    turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
    endpointing_opts: NotGivenOr[EndpointingOptions] = NOT_GIVEN,
    stt_context_options: NotGivenOr[STTContextOptions | None] = NOT_GIVEN,
    **options: Any,
) -> None:
    """A voice session over the Rust core.

    Args:
        stt, vad, llm, tts: The providers, or a LiveKit Inference model string. ``vad``
            defaults to ``inference.VAD(model="silero")``; ``vad=None`` opts out.
        tools: Tools shared by every agent of the session.
        context, thread: Read by tools through ``RunContext``.
        video_sampler: How often camera frames reach the model; ``None`` sends every frame.
        turn_detection, endpointing_opts: The session's turn completion and endpointing
            policy, as ``update_options`` changes them.
        options: The session options the core resolves: ``turn_handling``,
            ``stt_context_options``, ``max_tool_steps``, ``use_tts_aligned_transcript``,
            ``tts_text_transforms``, ``min_consecutive_speech_delay``, ``expressive``,
            ``aec_warmup_duration``, ``ivr_detection``, ``user_away_timeout``,
            ``transcription_timeout``, ``session_close_transcript_timeout``.
    """
    if unknown := options.keys() - _OPTION_KEYS:
        raise TypeError(f"unexpected VoiceSession options: {sorted(unknown)}")
    listening = dict(options.get("turn_handling") or {})
    if is_given(turn_detection):
        listening["turn_detection"] = turn_detection
    if is_given(endpointing_opts):
        listening["endpointing"] = endpointing_opts
    if listening:
        options["turn_handling"] = listening
    session._conn_options = conn_options or SessionConnectOptions()
    session._options = _core.SessionOptions(
        cast(SessionKwargs, {**_given(**options), "conn_options": session._conn_options})
    )
    session._stt_context = stt_context_options if is_given(stt_context_options) else None

    session._stt = inference.STT.from_model_string(stt) if isinstance(stt, str) else stt or None
    session._llm = llm_from_model_string(llm) if isinstance(llm, str) else llm or None
    session._tts = inference.TTS.from_model_string(tts) if isinstance(tts, str) else tts or None
    session._tools = tools if is_given(tools) else []
    turn_handling = options.get("turn_handling") or {}
    detection = turn_handling.get("turn_detection", NOT_GIVEN)
    defaults = _core.default_session_models(
        is_given(vad), is_given(detection), isinstance(session._llm, RealtimeModel)
    )
    session._using_default_vad = defaults.vad_model is not None
    session._vad = (
        inference.VAD(model=defaults.vad_model)
        if defaults.vad_model is not None
        else vad
        if is_given(vad)
        else None
    )
    if not is_given(detection):
        detection = inference.TurnDetector() if defaults.turn_detector else None
    # the gateway detector's stream is the core's own detector; a plugin's is seen through a view
    session._turn_detector = (
        None
        if isinstance(detection, str | None)
        else detection.stream()
        if isinstance(detection, inference.TurnDetector)
        else TurnDetector(detection)
    )
    session._turn_detection_mode = detection if isinstance(detection, str) else None
    session._interruption_mode = (turn_handling.get("interruption") or {}).get("mode")
    session._video_sampler = (
        video_sampler if is_given(video_sampler) else VoiceActivityVideoSampler()
    )

    # `ctx.context` on this call; `None` when the session has no application context
    session._context = cast("Context_T", context)
    session._thread = thread
    # `ctx.execution_id` on this call: one conversational task across its turns
    session._execution_id = shortuuid("execution_")
    session._store = None

    session._owner = _core._core_SessionOwner(session._options)
    session._room = None
    session._amd = None
    session._options.update_providers(
        {"stt": session._stt, "tts": session._tts, "vad": session._vad}
    )
    # an `update_agent` the core has not answered yet: the agent the session is handing to
    session._chat_ctx = ChatContext.empty()
    session._agent_state = "initializing"
    session._user_state = "listening"
    session._input = io.AgentInput(session)
    session._output = io.AgentOutput(session)
    # what a job's session report reads
    session._started_at = None
    session._event_snapshots = deque()
    session._event_retention = _core._core_EventRetention()
    session._recording_options = cast(
        RecordingOptions,
        dataclasses.asdict(
            _core.plan_session_start(job_default=False, another_session_is_primary=False).recording
        ),
    )
    # the session's span once started, which the job's RPC handlers parent under
    session._root_span_context = None
    session._text_input_cb = None
    # the session's audio recording, written by the core; finished once it closes
    session._recorder = None
    # the pipeline callbacks registered on this session, by step name
    session._steps = {}
    session._recording_path = None
    session._audio_input_options = None
    session._tasks = set()
    # the handles the core announced, for their done callbacks (``speech_done``)
    session._speech_handles = weakref.WeakValueDictionary()


class SessionAdapter:
    def __init__(self, session: VoiceSession[Any]) -> None:
        self.session = session

    def _register(self, step: str, f: Callable[..., Any]) -> None:
        session = self.session
        if session._core is not None:
            raise RuntimeError(f"register {step} before the session starts")
        if step in session._steps:
            raise RuntimeError(f"this session already has a {step}")
        session._steps[step] = f

    def _interruption_detector(
        self, agent: Agent, model: Any
    ) -> inference.AdaptiveInterruptionDetector | None:
        """The adaptive interruption detector, when the reference's gate builds one."""
        session = self.session
        stt_caps = session._stt.capabilities if session._stt is not None else None
        has_vad = session._vad is not None
        realtime = isinstance(model, RealtimeModel)
        verdict = _core.resolve_interruption_detector(
            stt_aligned_transcript=bool(stt_caps.aligned_transcript) if stt_caps else None,
            stt_streaming=stt_caps.streaming if stt_caps else None,
            realtime=realtime,
            rt_server_turn_detection=realtime
            and _core.rt_server_turn_detection(
                caps_turn_detection=model.capabilities.turn_detection,
                can_disable_turn_detection=model.capabilities.can_disable_turn_detection,
                vad=has_vad,
                turn_detection=session._turn_detection_mode,
                streaming_detector=session._turn_detector is not None,
                interruption_given=bool(session._interruption_mode),
            ),
            has_vad=has_vad,
            turn_detection=session._turn_detection_mode,
            allow_interruptions=session.options["interruption"]["enabled"],
            agent_mode=None,
            session_mode=session._interruption_mode,
            hosted=is_hosted(),
            dev_mode=is_dev_mode(),
        )
        if verdict["message"]:
            log = logger.warning if verdict["level"] == "warning" else logger.info
            log(verdict["message"])
        if not verdict["build"]:
            return None
        try:
            return inference.AdaptiveInterruptionDetector()
        except ValueError as e:
            logger.warning("failed to create AdaptiveInterruptionDetector", extra={"error": str(e)})
            return None

    def _pool(self, agent: Agent) -> list[llm.Tool | llm.Toolset]:
        session = self.session
        return [*agent.tools, *session._tools]

    def _voice_agent(self, agent: Agent, *, task: bool = False) -> _NativeAgent:
        """``agent`` as its callbacks run on this session: its definition over its tools and the
        session's it lacks, its ``AgentCallbacks`` and the context they run in, whose binding is this
        session's."""
        session = self.session
        from .agent_run import _voice_tools

        captured = contextvars.copy_context()
        binding = _VoiceBinding(session, agent, task=task)
        captured.run(_BindingContextVar.set, binding)
        return _core.VoiceAgent(
            agent._definition(_voice_tools(binding)),
            agent._host,
            (asyncio.get_running_loop(), captured),
            toolset_scope=id(agent),
        )

    def _frame(self, agent: Agent, *, task: bool = False) -> dict[str, Any]:
        """``agent`` as the core installs it (``update_agent``'s keywords); the core has its
        toolsets acquired by the native registry before Python reflects their tools. ``task``:
        it runs a ``VoiceTask``."""
        session = self.session
        # its tools, instructions and hooks are what the core runs from now on
        agent._config.freeze()
        pool = self._pool(agent)
        frame: dict[str, Any] = {
            "instructions": agent._config.instructions,
            "dynamic_instructions": (
                _DynamicInstructions(session, agent) if agent._config.dynamic_instructions else None
            ),
            "agent": agent.name,
            "agent_host": agent,
            "tools_host": pool,
            "voice_agent": self._voice_agent(agent, task=task),
        }
        # every leg a node may wrap is given, so the previous frame's node never stays on it; the
        # session's STT and TTS stay as they are, under the agent's nodes
        binding = _VoiceBinding(session, agent)
        frame["stt"] = _bindings.core_leg(binding.stt)
        model = binding.llm
        if is_given(agent._llm) or not isinstance(model, llm.RealtimeModel):
            frame["llm"] = _bindings.core_leg(model)
        frame["tts"] = _bindings.core_leg(binding.tts)
        return {**frame, **self._nodes(agent)}

    def _nodes(self, agent: Agent) -> dict[str, Any]:
        """The registered nodes, as the core opens them for ``agent``, each call in its
        binding."""
        session = self.session
        registered = [n for n in _NODES if n in session._steps]
        if not registered:
            return {}
        context = contextvars.copy_context()
        context.run(_BindingContextVar.set, _VoiceBinding(session, agent))
        # taken in the agent's binding: each node call runs in a copy of it
        return dict.fromkeys(registered, context.run(_core.AgentNodes, _SessionNodes(session)))

    def _legs(self) -> SessionLegs:
        session = self.session
        return {
            "stt": session._stt is not None,
            "vad": session._vad is not None,
            "using_default_vad": session._using_default_vad,
            "llm": isinstance(session._llm, llm.LLM),
            "tts": session._tts is not None,
            "realtime": isinstance(session._llm, llm.RealtimeModel),
            "detector": None,
            "adaptive_detector": False,
            "audio_enabled": True,
            "audio_output": True,
            "sink_can_pause": True,
        }

    def _install_models(self, models: dict[str, Any]) -> None:
        session = self.session
        stt_model: NotGivenOr[stt.STT | None] = models.get("stt", NOT_GIVEN)
        tts_model: NotGivenOr[tts.TTS | None] = models.get("tts", NOT_GIVEN)
        vad_model: NotGivenOr[VAD | None] = models.get("vad", NOT_GIVEN)
        if is_given(stt_model):
            session._stt = stt_model
        if is_given(tts_model):
            session._tts = tts_model
        if is_given(vad_model):
            session._vad = vad_model
            session._using_default_vad = False
        # Plugin proxies capture the running host loop. Before startup, retain only SDK objects;
        # the startup adapter binds their effective values when the native owner asks for them.
        if models and session._core is not None:
            nodes = self._nodes(session.current_agent)
            session._owner.install_providers(
                ProviderBindings(
                    stt=_bindings.core_leg(stt_model) if is_given(stt_model) else NOT_GIVEN,
                    tts=_bindings.core_leg(tts_model) if is_given(tts_model) else NOT_GIVEN,
                    vad=_bindings.core_leg(vad_model) if is_given(vad_model) else NOT_GIVEN,
                    stt_node=nodes.get("stt_node") if is_given(stt_model) else None,
                    tts_node=nodes.get("tts_node") if is_given(tts_model) else None,
                )
            )

    @staticmethod
    def _split(options: dict[str, Any]) -> tuple[dict[str, Any], UpdateKwargs]:
        """``options`` as its providers and its settings."""
        models = {k: v for k, v in options.items() if k in _MODELS}
        return models, cast("UpdateKwargs", {k: v for k, v in options.items() if k not in _MODELS})

    def update_agent(self, agent: Agent, *, chat_ctx: ChatContext | None = None) -> None:
        """Hand the conversation to ``agent``; the session and its history stay. ``chat_ctx`` is
        the context ``agent`` works on instead of the conversation carried forward."""
        session = self.session
        frame = None
        if session._core is not None:
            fields = self._frame(agent)
            if chat_ctx is not None:
                fields["chat_ctx"] = chat_ctx.items
            frame = UpdateAgentArgs(**fields)
        session._owner.update_agent(agent, frame)
        if session._core is None and chat_ctx is not None:
            session._chat_ctx = chat_ctx.copy()

    def say(
        self,
        text: str | AsyncIterable[str],
        *,
        audio: NotGivenOr[AsyncIterable[rtc.AudioFrame]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        add_to_chat_ctx: bool = True,
    ) -> SpeechHandle:
        session = self.session
        interruptions = allow_interruptions if is_given(allow_interruptions) else None
        origin = _origin()
        if is_given(audio):
            assert isinstance(text, str), "say(audio=) takes the words it speaks as a str"
            core = session._running.say_audio(
                text,
                audio,
                allow_interruptions=interruptions,
                add_to_chat_ctx=add_to_chat_ctx,
                origin=origin,
            )
        elif isinstance(text, str):
            core = session._running.say(
                text,
                allow_interruptions=interruptions,
                add_to_chat_ctx=add_to_chat_ctx,
                origin=origin,
            )
        else:
            core = session._running.say_stream(
                text,
                allow_interruptions=interruptions,
                add_to_chat_ctx=add_to_chat_ctx,
                origin=origin,
            )
        return SpeechHandle(core)

    def generate_reply(
        self,
        *,
        user_input: NotGivenOr[str] = NOT_GIVEN,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[str]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
    ) -> SpeechHandle:
        """Reply now. ``tools`` narrows the offered tools to these names for this reply."""
        session = self.session
        toolset: dict[str, Any] = {}
        if is_given(tools):
            pool = self._pool(session.current_agent)
            specs = tool_specs(pool)
            specs = [specs[index] for index in _core.reply_tool_indices(specs, tools)]
            toolset = {
                "tools": specs,
                "tools_host": pool,
                "voice_agent": self._voice_agent(session.current_agent),
            }
        core = session._running.generate_reply(
            user_input=user_input if is_given(user_input) else None,
            instructions=instructions if is_given(instructions) else None,
            allow_interruptions=allow_interruptions if is_given(allow_interruptions) else None,
            tool_choice=tool_choice if is_given(tool_choice) else None,
            chat_ctx=chat_ctx.items if is_given(chat_ctx) else None,
            on_enter=isinstance(_origin(), VoiceCommandOriginHook)
            and _hook_name.get(None) == "on_enter",
            origin=_origin(),
            **toolset,
        )
        return SpeechHandle(core)

    async def _start_ivr_detection(self, transcript: str | None = None) -> None:
        """Start IVR detection: the DTMF tool joins the session's tools and the core's navigator
        runs, counting ``transcript`` (AMD's greeting) as heard."""
        session = self.session
        from ..beta import send_dtmf_events

        if send_dtmf_events not in session._tools:
            session._tools.append(send_dtmf_events)
        session._running.ivr_start(
            tools=tool_specs([send_dtmf_events]), tools_host=[send_dtmf_events]
        )
        if transcript is not None:
            session._running.ivr_final(transcript)

    def configure(self, **options: Any) -> None:
        session = self.session
        models, turn = self._split(_provider_options(options))
        self._install_models(
            session._owner.update_options(
                turn, models, interruption_mode_given="mode" in (turn.get("interruption") or {})
            )
        )

    @contextlib.asynccontextmanager
    async def override(self, **options: Any) -> AsyncIterator[None]:
        session = self.session
        models, turn = self._split(_provider_options(options))
        # One override identity for turn settings and runtime-owned provider references.
        layer = None
        failure: BaseException | None = None
        try:
            override = session._owner.override_options(
                turn, models, interruption_mode_given="mode" in (turn.get("interruption") or {})
            )
            layer = override.layer
            self._install_models(override.providers)
            yield
        except BaseException as error:
            failure = error
            raise
        finally:
            # a release never raises: a closed session has nothing left to restore
            if layer is not None:
                try:
                    self._install_models(session._owner.release_override(layer))
                except _core.SessionClosing as refused:
                    # a closed session has nothing left to restore; the block's own error is the
                    # one to raise
                    if failure is not None:
                        failure.add_note(f"restoring the overridden providers: {refused}")


def _provider_options(options: dict[str, Any]) -> dict[str, Any]:
    options = _given(**options)
    if "stt" in options:
        options["stt"] = _stt_model(options["stt"])
    if "tts" in options:
        options["tts"] = _tts_model(options["tts"])
    return options


def _given(**options: Any) -> dict[str, Any]:
    return {k: v for k, v in options.items() if is_given(v)}


def _stt_model(model: NotGivenOr[stt.STT | str | None]) -> NotGivenOr[stt.STT | None]:
    return inference.STT.from_model_string(model) if isinstance(model, str) else model


def _tts_model(model: NotGivenOr[tts.TTS | str | None]) -> NotGivenOr[tts.TTS | None]:
    return inference.TTS.from_model_string(model) if isinstance(model, str) else model


@function_tool(name="record_keyterms")
async def _record_keyterms(pending: list[str], confirm: list[str], remove: list[str]) -> None:
    """Update the STT keyterms based on the latest transcript.

    Args:
        pending: Distinctive terms seen but not yet trusted — tracked, not applied.
        confirm: Pending terms the transcript has now corroborated — applied.
        remove: Only a spelling the user corrected away; applied terms are otherwise sticky.
    """
    # only used to elicit a structured tool call; never executed


def _detection_llm(stt_context: STTContextOptions | None) -> LLM | None:
    """The detection LLM ``stt_context_options`` asks for: the ``LLM`` given, else the model the
    core names (``_core.resolve_keyterm_detection``) through the inference gateway. ``None`` with
    detection off, or when the model cannot be created (detection is then skipped)."""
    config = (stt_context or {}).get("keyterm_detection")
    resolved = _core.resolve_keyterm_detection(config)
    if not resolved.enabled:
        return None
    if (model := resolved.model) is None:
        llm = config["llm"]  # type: ignore[index]
        return llm if isinstance(llm, LLM) else None
    try:
        from ..inference import LLM as InferenceLLM

        return InferenceLLM.from_model_string(model)
    except Exception:  # noqa: BLE001 — never let detection setup break the session
        logger.warning("keyterm detection: could not create detection LLM %r; skipping", model)
        return None


class SessionHost:
    def __init__(
        self,
        session: VoiceSession[Any],
        agent: Agent,
        room: NotGivenOr[rtc.Room | None],
        room_options: NotGivenOr[RoomOptions],
        chat_ctx: ChatContext | None,
    ) -> None:
        self.session = session
        self.agent = agent
        self.room = room
        self.room_options = room_options
        self.chat_ctx = chat_ctx
        thread = session._thread
        self.storage = _core._core_SessionStorage(
            thread._storage.threads if thread is not None else None,
            thread.id if thread is not None else None,
            thread._agent_name if thread is not None else agent.name,
            agent.name,
            chat_ctx.items if chat_ctx is not None else None,
        )

    async def prepare(self) -> _core._core_SessionBindings:
        from ._session_callbacks import _AgentHooks

        session, agent = self.session, self.agent
        room, room_options, chat_ctx = self.room, self.room_options, self.chat_ctx
        console = ConsoleCall.claim()
        job_ctx = get_job_context(required=False)
        if job_ctx is not None:
            session._event_retention.claim(job_ctx.job.id)
        if console is not None:
            room = None
        elif not is_given(room):
            room = job_ctx.room if job_ctx is not None else None
        if chat_ctx is not None:
            session._chat_ctx = chat_ctx.copy()
        options = room_options if is_given(room_options) else RoomOptions()
        session._audio_input_options = options.get_audio_input_options()
        text_input = options.get_text_input_options()
        session._text_input_cb = text_input.text_input_cb if text_input is not None else None
        # the job's session (on its room, or the console's) records and reports for the job and
        # closes with it; the primary one, and the console's, serve the remote session protocol
        in_job = job_ctx is not None and (console is not None or room is job_ctx.room)
        hosting = console is not None
        primary = False
        if in_job:
            assert job_ctx is not None
            primary = job_ctx._claim_session(session)
            plan = _core.plan_session_start(
                job_default=job_ctx.job.enable_recording,
                another_session_is_primary=not primary,
            )
            session._recording_options = cast(RecordingOptions, dataclasses.asdict(plan.recording))
            hosting = hosting or plan.hosting
            job_ctx.init_recording(session._recording_options)

        from ..storage import Thread, _StorageHandle
        from ..storage.context import ContextStore

        storage = session._owner.storage
        assert storage is not None
        session._store = ContextStore(activation=storage.activation, scope_id=storage.scope_id)
        if session._thread is None and (namespace := storage.namespace) is not None:
            assert storage.thread_id is not None
            session._thread = Thread(
                _StorageHandle(namespace), storage.thread_id, agent_name=agent.name
            )
        if (saved := storage.history) is not None:
            session._chat_ctx = ChatContext(list(saved))

        hooks = _AgentHooks(session)
        frame = session._frame(agent)
        model = frame.get("llm", session._llm)
        # an OpenAI-protocol model or a fallback chain runs in the core; any other one is a host
        # realtime model
        realtime = (
            realtime_model(model, lambda: session._adapter._pool(session.current_agent))
            if isinstance(model, llm.RealtimeModel)
            else None
        )
        sampler = session._video_sampler
        keyterm_llm = _detection_llm(session._stt_context)
        session._room = room
        args = VoiceSessionArgs(
            instructions=frame["instructions"],
            dynamic_instructions=frame["dynamic_instructions"],
            agent=frame["agent"],
            agent_host=frame["agent_host"],
            chat_ctx=session._chat_ctx.items,
            stt=frame["stt"],
            vad=_bindings.core_leg(session._vad),
            llm=None if isinstance(model, llm.RealtimeModel) else model,
            realtime=realtime,
            tts=frame["tts"],
            host_audio_input=session._input.audio,
            host_video_input=session._input.video,
            host_audio_output=session._output.audio,
            host_transcription_output=session._output.transcription,
            on_event=session,
            hooks=hooks,
            session_toolsets=hooks._core_toolset_resources(session._tools),
            tools_host=frame["tools_host"],
            voice_agent=frame["voice_agent"],
            room_io_config=options._to_core(),
            video_sampler=(sampler.speaking_fps, sampler.silent_fps) if sampler else None,
            using_default_vad=session._using_default_vad,
            turn_detector=session._turn_detector,
            hook_bound=_HOOK_BOUND,
            max_unrecoverable_errors=session._conn_options.max_unrecoverable_errors,
            interruption_detector=session._adapter._interruption_detector(agent, model),
            stt_context=session._stt_context,
            keyterm_llm=_bindings.core_leg(keyterm_llm) if keyterm_llm is not None else None,
            keyterm_tools=[_record_keyterms] if keyterm_llm is not None else None,
            **{node: frame[node] for node in _NODES if node in frame},
            remote_session=RemoteSessionAccess(session) if hosting else None,
            console=console,
            # the core connects the job's room beside its opening
            job=job_ctx if in_job and console is None else None,
            # the job ends once its primary session closes
            job_run=job_ctx._run if primary and job_ctx is not None else None,
        )
        session._started_at = time.time()
        if in_job:
            assert job_ctx is not None
            if session._recording_options["audio"] or (console is not None and ConsoleCall.record):
                session._recording_path = job_ctx.session_directory / "audio.ogg"
                session._recorder = _core.SessionRecorder(str(session._recording_path))
            job_ctx.add_shutdown_callback(lambda: session._aclose(CloseReason.JOB_SHUTDOWN))
        rtc_room = _RtcRoom.of(room) if room is not None else None
        return _core._core_SessionBindings(
            room=_core._core_create_room_session(rtc_room), args=args, recorder=session._recorder
        )

    async def started(self) -> None:
        if self.session.options["ivr_detection"]:
            await self.session._start_ivr_detection()

    async def closed(self) -> None:
        session = self.session
        if session._store is not None:
            session._store._close()
        audio_in = session._audio_input_options
        if audio_in is not None and isinstance(audio_in.noise_cancellation, rtc.FrameProcessor):
            audio_in.noise_cancellation._close()


class ConsoleCall:
    """The ``console`` subcommand's call: ``lk agent console`` owns the microphone, the speakers
    and the TUI, and reaches the agent over one socket, which the first session started takes as
    its audio and its remote session's transport."""

    addr: ClassVar[str | None] = None
    """The CLI's ``host:port`` while the subcommand runs."""
    record: ClassVar[bool] = False

    @classmethod
    def claim(cls) -> _core.ConsoleIo | None:
        if cls.addr is None:
            return None
        addr, cls.addr = cls.addr, None
        return _core.ConsoleIo(addr)


class RemoteSessionAccess:
    """What a remote client asks of ``session`` that needs its objects."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session

    async def _core_chat_history(self) -> list[llm.ChatItem]:
        return self._session.history.items

    async def _core_agent_info(self) -> tuple[str, str | None, list[str], list[llm.ChatItem]]:
        agent = self._session.current_agent
        tools = list(llm.ToolContext(agent.tools).function_tools)
        return agent.name, agent._config.instructions, tools, self._session.history.items

    async def _core_run_input(self, text: str) -> tuple[list[llm.ChatItem] | None, str | None]:
        try:
            await self._session.interrupt(force=True)
            result = await self._session._run(user_input=text)
        except Exception as e:
            return None, str(e)
        return result.new_messages(), None

    async def _core_session_state(self) -> tuple[str, str, str, dict[str, str], float | None]:
        options = self._session.options
        return (
            self._session.agent_state,
            self._session.user_state,
            self._session.current_agent.name,
            {k: str(options.get(k)) for k in STATE_OPTION_FIELDS},
            self._session._started_at,
        )

    async def _core_rtc_stats(self) -> tuple[list[Any], list[Any]] | None:
        job_ctx = get_job_context(required=False)
        if job_ctx is None or not job_ctx.room.isconnected():
            return None
        stats = await job_ctx.room.get_rtc_stats()
        return (
            [MessageToDict(s) for s in stats.publisher_stats],
            [MessageToDict(s) for s in stats.subscriber_stats],
        )

    async def _core_session_usage(self) -> Usage:
        return self._session.usage

    def _core_framework_info(self) -> tuple[str, str]:
        return "python", __version__

    async def _core_update_io(self, update: tuple[bool | None, ...]) -> None:
        audio_in, _, audio_out, _, transcription = update
        if audio_in is not None:
            self._session.input.set_audio_enabled(audio_in)
        if audio_out is not None:
            self._session.output.set_audio_enabled(audio_out)
        if transcription is not None:
            self._session.output.set_transcription_enabled(transcription)

    async def _core_finalize_simulation(
        self, provisional: tuple[bool, str]
    ) -> tuple[tuple[bool, str] | None, str | None]:

        from ..evals import SimulationVerdict

        job_ctx = get_job_context(required=False)
        sim_ctx = job_ctx.simulation_context if job_ctx is not None else None
        if job_ctx is None or sim_ctx is None:
            return None, None
        try:
            success, reason = provisional
            sim_ctx._begin_finalize(
                simulator_verdict=SimulationVerdict(success=success, reason=reason),
            )
            if (fnc := job_ctx._simulation_end_fnc) is not None:
                if asyncio.iscoroutine(out := fnc(sim_ctx)):
                    await out
        except Exception as e:
            logger.exception("error while executing the on_simulation_end callback")
            return None, str(e)
        verdict = sim_ctx.user_verdict
        return ((verdict.success, verdict.reason) if verdict else None), None
