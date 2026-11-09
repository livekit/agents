from __future__ import annotations

import asyncio
import contextvars
from collections import deque
from collections.abc import AsyncIterable, Callable, Coroutine
from typing import TYPE_CHECKING, Any, ClassVar, cast

from google.protobuf.json_format import MessageToDict
from typing_extensions import TypeVar

from livekit import rtc

from .. import _bindings, _core, inference, llm, stt, tts
from .._bindings import tool_specs
from .._generated.connect import SessionConnectOptions as SessionConnectOptions
from .._generated.keyterms import STTContextOptions
from .._generated.remote_session import (
    AgentInfo,
    FrameworkInfo,
    IoUpdate,
    RemoteSessionState,
    RtcStats,
)
from .._generated.remote_session_wire import STATE_OPTION_FIELDS
from .._generated.session_options import SessionKwargs
from .._generated.session_owner import ProviderBindings
from .._generated.session_wire import DEFAULT_HOOK_BOUND
from .._generated.vad import VAD
from .._generated.video_sampler import VoiceActivityVideoSampler
from .._generated.voice_session import UpdateAgentArgs, VoiceSessionArgs
from ..inference.llm import llm_from_model_string
from ..job import get_job_context
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
from ..version import __version__
from . import Agent, ModelSettings, _BindingContextVar, _VoiceBinding, io
from ._session_callbacks import _DynamicInstructions, _SessionNodes
from .events import (
    Context_T,
    RunContext,
    UserTurnExceededEvent,
)
from .room_io import RoomOptions

if TYPE_CHECKING:
    from .._core import VoiceAgent as _NativeAgent
    from .._generated.session_options import SessionLegs
    from ..evals import SimulationVerdict
    from ..storage import Thread
    from .agent_session import VoiceSession


# pipeline callbacks; each receives the call's `RunContext`, whose `agent` holds the call
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


_CallbackT = TypeVar("_CallbackT", bound=Callable[..., object])


_NODES = ("llm_node", "tts_node", "stt_node", "transcription_node")


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
    stt_context_options: NotGivenOr[STTContextOptions | None] = NOT_GIVEN,
    options: SessionKwargs,
) -> None:
    session._conn_options = conn_options or SessionConnectOptions()
    session._options = _core.SessionOptions(options, session._conn_options)
    session._stt_context = stt_context_options if is_given(stt_context_options) else None

    stt_model = inference.STT.from_model_string(stt) if isinstance(stt, str) else stt or None
    session._llm = llm_from_model_string(llm) if isinstance(llm, str) else llm or None
    tts_model = inference.TTS.from_model_string(tts) if isinstance(tts, str) else tts or None
    session._tools = tools if is_given(tools) else []
    turn_handling = options.get("turn_handling") or {}
    detection = turn_handling.get("turn_detection", NOT_GIVEN)
    defaults = _core.default_session_models(
        is_given(vad), is_given(detection), isinstance(session._llm, RealtimeModel)
    )
    # the VAD the session created itself, if any
    session._default_vad = (
        inference.VAD(model=defaults.vad_model) if defaults.vad_model is not None else None
    )
    vad_model = session._default_vad or (vad if is_given(vad) else None)
    if not is_given(detection):
        detection = inference.TurnDetector() if defaults.turn_detector else None
    # SessionOptions already rejected any other detector type
    session._turn_detector = (
        detection.stream() if isinstance(detection, inference.TurnDetector) else None
    )
    session._video_sampler = (
        video_sampler if is_given(video_sampler) else VoiceActivityVideoSampler()
    )

    # `ctx.context` on this call; `None` when the session has no application context
    session._context = cast("Context_T", context)
    session._thread = thread
    # `ctx.execution_id` on this call: one conversational task across its turns
    session._execution_id = shortuuid("execution_")
    session._store = None

    session._native = _core._core_SessionOwner(session._options)
    session._room = None
    session._amd = None
    # the native options own the providers: `session.stt`, `tts` and `vad` read them
    session._options.update_providers({"stt": stt_model, "tts": tts_model, "vad": vad_model})
    session._input = io.AgentInput(session)
    session._output = io.AgentOutput(session)
    # read by the job's session report
    session._event_snapshots = deque()
    # the job's RPC handlers parent their spans under this once started
    session._root_span_context = None
    session._text_input_cb = None
    session._steps = {}
    session._audio_input_options = None
    session._tasks = set()


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
        session = self.session
        stt_caps = session.stt.capabilities if session.stt is not None else None
        has_vad = session.vad is not None
        realtime = isinstance(model, RealtimeModel)
        options = session._options
        verdict = _core.resolve_interruption_detector(
            stt_aligned_transcript=bool(stt_caps.aligned_transcript) if stt_caps else None,
            stt_streaming=stt_caps.streaming if stt_caps else None,
            realtime=realtime,
            rt_server_turn_detection=realtime
            and _core.rt_server_turn_detection(
                caps_turn_detection=model.capabilities.turn_detection,
                can_disable_turn_detection=model.capabilities.can_disable_turn_detection,
                vad=has_vad,
                turn_detection=options.turn_detection,
                streaming_detector=session._turn_detector is not None,
                interruption_given=options.interruption_mode is not None,
            ),
            has_vad=has_vad,
            turn_detection=options.turn_detection,
            allow_interruptions=session.options["interruption"]["enabled"],
            agent_mode=None,
            session_mode=options.interruption_mode,
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

    def _frame(self, agent: Agent, *, task: bool = False) -> UpdateAgentArgs:
        session = self.session
        agent._config.freeze()
        # give every leg a node may wrap, so the previous frame's node never stays on it
        binding = _VoiceBinding(session, agent)
        model = binding.llm
        return UpdateAgentArgs(
            instructions=agent._config.instructions,
            dynamic_instructions=(
                _DynamicInstructions(session, agent) if agent._config.dynamic_instructions else None
            ),
            agent=agent.name,
            agent_host=agent,
            tools_host=self._pool(agent),
            voice_agent=self._voice_agent(agent, task=task),
            stt=_bindings.core_leg(binding.stt),
            llm=(
                _bindings.core_leg(model)
                if is_given(agent._llm) or not isinstance(model, llm.RealtimeModel)
                else NOT_GIVEN
            ),
            tts=_bindings.core_leg(binding.tts),
            **self._nodes(agent),
        )

    def _nodes(self, agent: Agent) -> dict[str, Any]:
        session = self.session
        registered = [n for n in _NODES if n in session._steps]
        if not registered:
            return {}
        context = contextvars.copy_context()
        context.run(_BindingContextVar.set, _VoiceBinding(session, agent))
        # created in the agent's binding: each node call runs in a copy of it
        return dict.fromkeys(registered, context.run(_core.AgentNodes, _SessionNodes(session)))

    def _legs(self) -> SessionLegs:
        session = self.session
        return {
            "stt": session.stt is not None,
            "vad": session.vad is not None,
            "using_default_vad": self._using_default_vad(),
            "llm": isinstance(session._llm, llm.LLM),
            "tts": session.tts is not None,
            "realtime": isinstance(session._llm, llm.RealtimeModel),
        }

    def _using_default_vad(self) -> bool:
        session = self.session
        return session._default_vad is not None and session.vad is session._default_vad

    async def _start_ivr_detection(self, transcript: str | None = None) -> None:
        session = self.session
        from ..beta import send_dtmf_events

        if send_dtmf_events not in session._tools:
            session._tools.append(send_dtmf_events)
        session._running.ivr_start(
            tools=tool_specs([send_dtmf_events]), tools_host=[send_dtmf_events]
        )
        if transcript is not None:
            session._running.ivr_final(transcript)


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
    except Exception:  # noqa: BLE001 - never let detection setup break the session
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
        if console is not None:
            room = None
        elif not is_given(room):
            room = job_ctx.room if job_ctx is not None else None
        options = room_options if is_given(room_options) else RoomOptions()
        session._audio_input_options = options.get_audio_input_options()
        text_input = options.get_text_input_options()
        session._text_input_cb = text_input.text_input_cb if text_input is not None else None
        # the job's session runs on its room, or the console's
        job = (
            job_ctx
            if job_ctx is not None and (console is not None or room is job_ctx.room)
            else None
        )

        from ..storage.context import ContextStore

        storage = session._native.storage
        assert storage is not None
        session._store = ContextStore(activation=storage.activation, scope_id=storage.scope_id)

        hooks = _AgentHooks(session)
        frame = session._frame(agent)
        model = frame.llm if is_given(frame.llm) else session._llm
        realtime = model._core_leg() if isinstance(model, llm.RealtimeModel) else None
        keyterm_llm = _detection_llm(session._stt_context)
        session._room = room
        args = VoiceSessionArgs(
            instructions=frame.instructions,
            dynamic_instructions=frame.dynamic_instructions,
            agent=frame.agent,
            agent_host=frame.agent_host,
            # None: a stored call starts on its lane's history
            chat_ctx=chat_ctx.items if chat_ctx is not None else None,
            stt=frame.stt if is_given(frame.stt) else None,
            vad=_bindings.core_leg(session.vad),
            llm=None if isinstance(model, llm.RealtimeModel) else model,
            realtime=realtime,
            realtime_tools=lambda: session._adapter._pool(session.current_agent),
            tts=frame.tts if is_given(frame.tts) else None,
            host_audio_input=session._input.audio,
            host_video_input=session._input.video,
            host_audio_output=session._output.audio,
            host_transcription_output=session._output.transcription,
            on_event=session,
            hooks=hooks,
            session_toolsets=hooks._core_toolset_resources(session._tools),
            tools_host=frame.tools_host,
            voice_agent=frame.voice_agent,
            room_io_config=options,
            video_sampler=session._video_sampler,
            using_default_vad=session._adapter._using_default_vad(),
            turn_detector=session._turn_detector,
            hook_bound=DEFAULT_HOOK_BOUND,
            max_unrecoverable_errors=session._conn_options.max_unrecoverable_errors,
            interruption_detector=session._adapter._interruption_detector(agent, model),
            stt_context=session._stt_context,
            keyterm_llm=_bindings.core_leg(keyterm_llm) if keyterm_llm is not None else None,
            keyterm_tools=[_record_keyterms] if keyterm_llm is not None else None,
            llm_node=frame.llm_node,
            tts_node=frame.tts_node,
            stt_node=frame.stt_node,
            transcription_node=frame.transcription_node,
            remote_session=RemoteSessionAccess(session),
            console=console,
            job=job,
            # the job ends once its primary session closes
            job_run=job._run if job is not None else None,
        )
        return _core._core_SessionBindings(
            room=_core._core_create_room_session(room),
            args=args,
            record_audio=console is not None and ConsoleCall.record,
        )

    def bind_providers(self, providers: dict[str, Any]) -> ProviderBindings:
        session = self.session
        nodes = session._adapter._nodes(session.current_agent)

        def leg(slot: str) -> Any:
            return _bindings.core_leg(providers[slot]) if slot in providers else NOT_GIVEN

        return ProviderBindings(
            stt=leg("stt"),
            tts=leg("tts"),
            vad=leg("vad"),
            stt_node=nodes.get("stt_node") if "stt" in providers else None,
            tts_node=nodes.get("tts_node") if "tts" in providers else None,
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
    # `lk agent console` reaches the agent over one socket; the first session started claims it
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
    def __init__(self, session: VoiceSession[Any]) -> None:
        self.session = session

    async def _core_agent_info(self) -> AgentInfo:
        agent = self.session.current_agent
        return AgentInfo(
            id=agent.name,
            instructions=agent._config.instructions,
            tools=list(llm.ToolContext(agent.tools).function_tools),
            chat_ctx=self.session.history.items,
        )

    async def _core_run_input(self, text: str) -> list[llm.ChatItem]:
        from .run_result import _events

        await self.session.interrupt(force=True)
        result = await self.session._run(user_input=text)
        # the reply to `text`: the run's events, without the input itself (1.x `RunResult.events`)
        return [event.item for event in _events(result.new_messages())]

    async def _core_session_state(self) -> RemoteSessionState:
        options = self.session.options
        return RemoteSessionState(
            agent_state=self.session.agent_state,
            user_state=self.session.user_state,
            agent_id=self.session.current_agent.name,
            options={k: str(options.get(k)) for k in STATE_OPTION_FIELDS},
            started_at=self.session._native.started_at,
        )

    async def _core_rtc_stats(self) -> RtcStats | None:
        job_ctx = get_job_context(required=False)
        if job_ctx is None or not job_ctx.room.isconnected():
            return None
        stats = await job_ctx.room.get_rtc_stats()
        return RtcStats(
            publisher_stats=[MessageToDict(s) for s in stats.publisher_stats],
            subscriber_stats=[MessageToDict(s) for s in stats.subscriber_stats],
        )

    def _core_framework_info(self) -> FrameworkInfo:
        return FrameworkInfo(sdk="python", sdk_version=__version__)

    async def _core_update_io(self, update: IoUpdate) -> None:
        if update.input_audio is not None:
            self.session.input.set_audio_enabled(update.input_audio)
        if update.output_audio is not None:
            self.session.output.set_audio_enabled(update.output_audio)
        if update.output_transcription is not None:
            self.session.output.set_transcription_enabled(update.output_transcription)

    async def _core_finalize_simulation(
        self, provisional: SimulationVerdict
    ) -> SimulationVerdict | None:
        job_ctx = get_job_context(required=False)
        sim_ctx = job_ctx.simulation_context if job_ctx is not None else None
        if job_ctx is None or sim_ctx is None:
            return None
        sim_ctx._begin_finalize(simulator_verdict=provisional)
        if (fnc := job_ctx._simulation_end_fnc) is not None:
            if asyncio.iscoroutine(out := fnc(sim_ctx)):
                await out
        return sim_ctx.user_verdict
