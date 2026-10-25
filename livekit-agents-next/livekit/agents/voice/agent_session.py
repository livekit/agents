"""``LiveSession``: the developer's handle on a session the Rust core runs.

The core (``_core.LiveSession``) owns the whole session: turn detection, scheduling, generation,
tools' ordering, interruptions, the room's io. This class turns the constructor's arguments into
the core's, forwards each call as one command, mirrors the core's events into the public event
classes and answers the core's hook calls by running the ``Agent``'s methods.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import time
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from typing_extensions import TypedDict, TypeVar

from livekit import rtc

from .. import _core, inference, llm, stt, tts, vad
from .._core_view import from_core_item, to_core_items, tool_specs
from .._generated.expressive import ExpressiveOptions, NonverbalOptions, SpeechSteeringOptions
from .._generated.session_dispatch import IGNORED_EVENTS, PUBLIC_EVENTS
from ..job import get_job_context
from ..llm import ChatContext, StopResponse
from ..llm.chat_context import Instructions
from ..log import logger
from ..metrics import AgentSessionUsage
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils.misc import is_given
from . import io
from .agent import Agent, _BindingContextVar, _VoiceBinding
from .events import (
    AgentEvent,
    AgentState,
    AgentStateChangedEvent,
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    ErrorEvent,
    EventTypes,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    RunContext,
    SpeechCreatedEvent,
    ToolExecutionUpdatedEvent,
    UserInputTranscribedEvent,
    UserState,
    UserStateChangedEvent,
)
from .speech_handle import SpeechHandle
from .tool_executor import ToolRunner

if TYPE_CHECKING:
    from ..storage.resolve import Thread
    from .room_io import RoomOptions

__all__ = [
    "LiveSession",
    "ExpressiveOptions",
    "NonverbalOptions",
    "RecordingOptions",
    "SessionConnectOptions",
    "SpeechSteeringOptions",
    "VoiceActivityVideoSampler",
]

Userdata_T = TypeVar("Userdata_T", default=Any)

# the hook running in this context, for `generate_reply(on_enter=)`
_hook_name: contextvars.ContextVar[str] = contextvars.ContextVar("lk_hook_name")

# on_user_turn_completed and on_exit are awaited by the core this long before it moves on
_HOOK_BOUND = 15.0

# the session options and their defaults; the core resolves them (`_core.SessionOptions`)
_OPTION_DEFAULTS: dict[str, Any] = {
    "turn_handling": NOT_GIVEN,
    "stt_context_options": NOT_GIVEN,
    "keyterms_options": NOT_GIVEN,
    "tool_handling": NOT_GIVEN,
    "max_tool_steps": 3,
    "use_tts_aligned_transcript": NOT_GIVEN,
    "tts_text_transforms": NOT_GIVEN,
    "min_consecutive_speech_delay": 0.0,
    "expressive": False,
    "aec_warmup_duration": NOT_GIVEN,
    "ivr_detection": False,
    "user_away_timeout": 15.0,
    "transcription_timeout": None,
    "session_close_transcript_timeout": 2.0,
}

# the deprecated turn kwargs the core migrates into `turn_handling`
_DEPRECATED_TURN_KWARGS = (
    "min_endpointing_delay",
    "max_endpointing_delay",
    "false_interruption_timeout",
    "turn_detection",
    "discard_audio_if_uninterruptible",
    "min_interruption_duration",
    "min_interruption_words",
    "allow_interruptions",
    "resume_false_interruption",
    "agent_false_interruption_timeout",
    "preemptive_generation",
)


class RecordingOptions(TypedDict, total=False):
    """Which recording features are active; keys default to ``True`` when not specified."""

    audio: bool
    traces: bool
    logs: bool
    transcript: bool
    redaction: bool


@dataclass
class SessionConnectOptions:
    stt_conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    llm_conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    tts_conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    max_unrecoverable_errors: int = 3
    """Maximum number of consecutive unrecoverable errors from stt, llm, or tts."""


class VoiceActivityVideoSampler:
    """Sample the camera at ``speaking_fps`` while the user speaks, ``silent_fps`` otherwise."""

    def __init__(self, *, speaking_fps: float = 1.0, silent_fps: float = 0.3) -> None:
        self.speaking_fps = speaking_fps
        self.silent_fps = silent_fps


class LiveSession(rtc.EventEmitter[EventTypes], Generic[Userdata_T]):
    def __init__(
        self,
        *,
        stt: NotGivenOr[stt.STT | str] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | str] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | str] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool | llm.Toolset]] = NOT_GIVEN,
        userdata: NotGivenOr[Userdata_T] = NOT_GIVEN,
        deps: Any = None,
        state: Any = None,
        thread: Thread | None = None,
        video_sampler: NotGivenOr[VoiceActivityVideoSampler | None] = NOT_GIVEN,
        conn_options: NotGivenOr[SessionConnectOptions] = NOT_GIVEN,
        **options: Any,
    ) -> None:
        """A voice session over the Rust core.

        Args:
            stt, vad, llm, tts: The providers, or a LiveKit Inference model string. ``vad``
                defaults to ``inference.VAD(model="silero")``; ``vad=None`` opts out.
            tools: Tools shared by every agent of the session.
            userdata, deps, state, thread: Read by tools through ``RunContext``.
            video_sampler: How often camera frames reach the model; ``None`` sends every frame.
            options: The session options the core resolves: ``turn_handling``,
                ``stt_context_options``, ``tool_handling``, ``max_tool_steps``,
                ``use_tts_aligned_transcript``, ``tts_text_transforms``,
                ``min_consecutive_speech_delay``, ``expressive``, ``aec_warmup_duration``,
                ``ivr_detection``, ``user_away_timeout``, ``transcription_timeout``,
                ``session_close_transcript_timeout`` and the deprecated turn keywords.
        """
        super().__init__()
        if unknown := options.keys() - _OPTION_DEFAULTS.keys() - set(_DEPRECATED_TURN_KWARGS):
            raise TypeError(f"unexpected LiveSession options: {sorted(unknown)}")
        deprecated = {k: options.pop(k, NOT_GIVEN) for k in _DEPRECATED_TURN_KWARGS}
        self._conn_options = conn_options or SessionConnectOptions()
        self._options = _core.SessionOptions(
            {**_OPTION_DEFAULTS, **options, "deprecated": deprecated, "conn_options": conn_options}
        )

        self._stt = inference.STT.from_model_string(stt) if isinstance(stt, str) else stt or None
        self._llm = inference.llm_from_model_string(llm) if isinstance(llm, str) else llm or None
        self._tts = inference.TTS.from_model_string(tts) if isinstance(tts, str) else tts or None
        self._using_default_vad = not is_given(vad)
        self._vad = inference.VAD(model="silero") if not is_given(vad) else vad
        self._tools = tools if is_given(tools) else []
        self._video_sampler = (
            video_sampler if is_given(video_sampler) else VoiceActivityVideoSampler()
        )

        self._userdata: Userdata_T | None = userdata if is_given(userdata) else None
        self._deps = deps
        self._state = state
        self._thread = thread

        self._core: _core.LiveSession = None  # type: ignore[assignment]
        self._agent: Agent | None = None
        # every agent the core may name, by the id it was installed under
        self._agents: dict[str, Agent] = {}
        self._speeches: dict[str, SpeechHandle] = {}
        self._fillers: dict[int, Callable[[int], Any] | None] = {}
        self._next_filler = 0
        self._chat_ctx = ChatContext.empty()
        self._agent_state: AgentState = "initializing"
        self._user_state: UserState = "listening"
        self._input = io.AgentInput(self)
        self._output = io.AgentOutput(self)
        self._usage = AgentSessionUsage(model_usage=[])
        # what a job's session report reads
        self._started_at: float | None = None
        self._recorded_events: list[AgentEvent] = []
        self._recording_options: RecordingOptions = {
            "audio": False,
            "traces": False,
            "logs": False,
            "transcript": False,
            "redaction": False,
        }
        self._tasks: set[asyncio.Task[Any]] = set()

    # --- state ---------------------------------------------------------------------------

    @property
    def userdata(self) -> Userdata_T:
        if self._userdata is None:
            raise ValueError("LiveSession userdata is not set")
        return self._userdata

    @userdata.setter
    def userdata(self, value: Userdata_T) -> None:
        self._userdata = value

    @property
    def deps(self) -> Any:
        return self._deps

    @property
    def state(self) -> Any:
        return self._state

    @property
    def thread(self) -> Thread | None:
        return self._thread

    @property
    def stt(self) -> stt.STT | None:
        return self._stt

    @property
    def llm(self) -> llm.LLM | llm.RealtimeModel | None:
        return self._llm

    @property
    def tts(self) -> tts.TTS | None:
        return self._tts

    @property
    def vad(self) -> vad.VAD | None:
        return self._vad

    @property
    def tools(self) -> list[llm.Tool | llm.Toolset]:
        return list(self._tools)

    @property
    def options(self) -> dict[str, Any]:
        """The configuration the core runs on, current the instant a change is sent."""
        if self._core is not None:
            return self._core.resolved_options()  # type: ignore[no-any-return]
        return self._options.resolved(self._legs())  # type: ignore[no-any-return]

    @property
    def conn_options(self) -> SessionConnectOptions:
        return self._conn_options

    @property
    def input(self) -> io.AgentInput:
        return self._input

    @property
    def output(self) -> io.AgentOutput:
        return self._output

    @property
    def history(self) -> ChatContext:
        return self._chat_ctx

    @property
    def keyterms(self) -> list[str]:
        return self._core.keyterms() or [] if self._core is not None else []

    @property
    def agent_state(self) -> AgentState:
        return self._agent_state

    @property
    def user_state(self) -> UserState:
        return self._user_state

    @property
    def current_agent(self) -> Agent:
        if self._agent is None:
            raise RuntimeError("LiveSession isn't running")
        return self._agent

    @property
    def usage(self) -> AgentSessionUsage:
        return self._usage

    @property
    def _running(self) -> _core.LiveSession:
        if self._core is None:
            raise RuntimeError("LiveSession isn't running")
        return self._core

    # --- lifecycle -----------------------------------------------------------------------

    async def start(
        self,
        agent: Agent,
        *,
        room: NotGivenOr[rtc.Room | _core.FakeRoom | _core.OfflineRoom] = NOT_GIVEN,
        room_options: NotGivenOr[RoomOptions] = NOT_GIVEN,
        chat_ctx: llm.ChatContext | None = None,
        capture_run: bool = False,
    ) -> None:
        """Start the session on ``room`` (the job's room by default, an offline room without
        one) with ``agent`` driving."""
        if self._core is not None:
            return
        if not is_given(room):
            job_ctx = get_job_context(required=False)
            room = job_ctx.room if job_ctx is not None else _core.OfflineRoom("console")
        if chat_ctx is not None:
            self._chat_ctx = chat_ctx.copy()

        frame = self._frame(agent)
        realtime = frame.get("llm") if isinstance(frame.get("llm"), llm.RealtimeModel) else None
        sampler = self._video_sampler
        self._core = _core.LiveSession(
            room,
            instructions=frame["instructions"],
            agent=frame["agent"],
            chat_ctx=to_core_items(self._chat_ctx.items),
            stt=frame.get("stt", self._stt),
            vad=frame.get("vad", self._vad),
            llm=None if realtime else frame.get("llm", self._llm),
            realtime_host=realtime,
            tts=frame.get("tts", self._tts),
            host_audio_input=self._input.audio,
            host_video_input=self._input.video,
            host_audio_output=self._output.audio,
            host_transcription_output=self._output.transcription,
            on_event=self,
            options=self._options,
            agent_turn_handling=frame.get("turn_handling"),
            tools=frame["tools"],
            tools_host=frame["tools_host"],
            tool_runner=frame["tool_runner"],
            room_io_config=room_options._to_core() if is_given(room_options) else None,
            video_sampler=(sampler.speaking_fps, sampler.silent_fps) if sampler else None,
            using_default_vad=self._using_default_vad,
            hook_bound=_HOOK_BOUND,
            max_unrecoverable_errors=self._conn_options.max_unrecoverable_errors,
        )
        self._agent = agent
        self._started_at = time.time()
        await self._core.start(capture_run=capture_run)

    async def aclose(self) -> None:
        await self._close(drain=False)

    async def drain(self) -> None:
        await self._running.drain()

    def shutdown(self, *, drain: bool = True) -> None:
        self._spawn(self._close(drain=drain))

    async def _close(self, *, drain: bool) -> None:
        if self._core is None:
            return
        await self._core.aclose(reason=CloseReason.USER_INITIATED.value, drain=drain)
        await self._core.close_io()
        for task in list(self._tasks):
            task.cancel()

    async def __aenter__(self) -> LiveSession[Userdata_T]:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # --- commands ------------------------------------------------------------------------

    def say(
        self,
        text: str | AsyncIterable[str],
        *,
        audio: NotGivenOr[AsyncIterable[rtc.AudioFrame]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        add_to_chat_ctx: bool = True,
    ) -> SpeechHandle:
        kwargs = {
            "allow_interruptions": allow_interruptions if is_given(allow_interruptions) else None,
            "add_to_chat_ctx": add_to_chat_ctx,
            "origin": _origin(),
        }
        if is_given(audio):
            assert isinstance(text, str), "say(audio=) takes the words it speaks as a str"
            core, pusher = self._running.say_audio(text, **kwargs)
            self._pump(audio, pusher)
        elif isinstance(text, str):
            core = self._running.say(text, **kwargs)
        else:
            core, pusher = self._running.say_stream(**kwargs)
            self._pump(text, pusher)
        return SpeechHandle(core)

    def _pump(self, source: AsyncIterable[Any], pusher: Any) -> None:
        async def _run() -> None:
            try:
                async for item in source:
                    if not await pusher.push(item):
                        break
            finally:
                pusher.end()

        self._spawn(_run())

    def generate_reply(
        self,
        *,
        user_input: NotGivenOr[str] = NOT_GIVEN,
        instructions: NotGivenOr[str | Instructions] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[str]] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
    ) -> SpeechHandle:
        """Reply now. ``tools`` narrows the offered tools to these names for this reply."""
        toolset: dict[str, Any] = {}
        if is_given(tools):
            pool = self._pool(self.current_agent)
            specs = [s for s in tool_specs(pool) if s["name"] in tools]
            toolset = {
                "tools": specs,
                "tools_host": pool,
                "tool_runner": ToolRunner(self, self.current_agent, pool),
            }
        if isinstance(instructions, Instructions):
            instructions = instructions.render(modality="audio")
        core = self._running.generate_reply(
            user_input=user_input if is_given(user_input) else None,
            instructions=instructions if is_given(instructions) else None,
            allow_interruptions=allow_interruptions if is_given(allow_interruptions) else None,
            tool_choice=tool_choice if is_given(tool_choice) else None,
            chat_ctx=to_core_items(chat_ctx.items) if is_given(chat_ctx) else None,
            on_enter=_origin() == "hook" and _hook_name.get(None) == "on_enter",
            origin=_origin(),
            **toolset,
        )
        return SpeechHandle(core)

    def interrupt(self, *, force: bool = False) -> asyncio.Future[None]:
        return asyncio.ensure_future(self._running.interrupt(force=force))

    def clear_user_turn(self) -> None:
        self._running.clear_user_turn()

    async def commit_user_turn(
        self,
        *,
        transcript_timeout: float = 2.0,
        stt_flush_duration: float = 2.0,
        skip_reply: bool = False,
    ) -> str:
        return await self._running.commit_user_turn(  # type: ignore[no-any-return]
            transcript_timeout=transcript_timeout,
            stt_flush_duration=stt_flush_duration,
            skip_reply=skip_reply,
        )

    async def wait_for_idle(self) -> None:
        await self._running.wait_for_idle()

    def update_agent(self, agent: Agent) -> None:
        """Hand the conversation to ``agent``; the session and its history stay."""
        self._spawn(self._running.update_agent(**self._frame(agent)))
        self._agent = agent

    def update_options(self, **update: Any) -> None:
        """Change the turn options of a running session (``UpdateKwargs``)."""
        if self._core is None:
            self._options.update(update)
        else:
            self._core.update_turn_options(update)

    # --- frames --------------------------------------------------------------------------

    def _pool(self, agent: Agent) -> list[llm.Tool | llm.Toolset]:
        return [*agent.tools, *self._tools]

    def _frame(self, agent: Agent) -> dict[str, Any]:
        """``agent`` as the core installs it (``update_agent``'s keywords)."""
        self._agents[agent.id] = agent
        pool = self._pool(agent)
        frame: dict[str, Any] = {
            "instructions": _instructions(agent.instructions),
            "agent": agent.id,
            "tools": tool_specs(pool),
            "tools_host": pool,
            "tool_runner": ToolRunner(self, agent, pool),
            "turn_handling": agent._turn_handling or None,
        }
        for leg in ("stt", "vad", "llm", "tts"):
            if is_given(value := getattr(agent, f"_{leg}")):
                frame[leg] = value
        return frame

    def _legs(self) -> dict[str, Any]:
        return {
            "stt": self._stt is not None,
            "vad": self._vad is not None,
            "using_default_vad": self._using_default_vad,
            "llm": isinstance(self._llm, llm.LLM),
            "tts": self._tts is not None,
            "realtime": isinstance(self._llm, llm.RealtimeModel),
            "detector": None,
            "adaptive_detector": False,
            "audio_enabled": True,
            "audio_output": True,
            "sink_can_pause": True,
        }

    # --- the core's events ---------------------------------------------------------------

    def _core_event(self, wire: dict[str, Any]) -> None:
        kind = wire["type"]
        if kind in IGNORED_EVENTS:
            return
        if (public := PUBLIC_EVENTS.get(kind)) is not None:
            cls, name = public
            self.emit(name, cls(**{k: v for k, v in wire.items() if k != "type"}))
            return
        handler = _HANDLERS.get(kind)
        if handler is not None:
            handler(self, wire)

    def _core_stream_closed(self, wire: dict[str, Any]) -> None:
        self._speeches.clear()

    def _on_hook(self, wire: dict[str, Any]) -> None:
        self._spawn(_answer_hook(self, wire))

    def _on_speech_created(self, wire: dict[str, Any]) -> None:
        handle = SpeechHandle(wire["speech_handle"])
        self._speeches[handle.id] = handle
        self.emit(
            "speech_created",
            SpeechCreatedEvent(
                user_initiated=wire["user_initiated"], source=wire["source"], speech_handle=handle
            ),
        )

    def _on_speech_done(self, wire: dict[str, Any]) -> None:
        self._speeches.pop(wire["speech_id"], None)

    def _on_item_added(self, wire: dict[str, Any]) -> None:
        item = from_core_item(wire["item"])
        self._chat_ctx.insert(item)
        if item.type == "message":
            self.emit("conversation_item_added", ConversationItemAddedEvent(item=item))

    def _on_agent_state(self, wire: dict[str, Any]) -> None:
        self._agent_state = wire["new_state"]
        self.emit(
            "agent_state_changed",
            AgentStateChangedEvent(old_state=wire["old_state"], new_state=wire["new_state"]),
        )

    def _on_user_state(self, wire: dict[str, Any]) -> None:
        self._user_state = wire["new_state"]
        self.emit(
            "user_state_changed",
            UserStateChangedEvent(old_state=wire["old_state"], new_state=wire["new_state"]),
        )

    def _on_transcript(self, wire: dict[str, Any]) -> None:
        self.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(
                transcript=wire["transcript"],
                is_final=wire["is_final"],
                language=wire.get("language"),
            ),
        )

    def _on_tool_update(self, wire: dict[str, Any]) -> None:
        self.emit(
            "tool_execution_updated",
            ToolExecutionUpdatedEvent.model_validate(
                {"update": {k: _host_value(v) for k, v in wire["update"].items()}}
            ),
        )

    def _on_metrics(self, wire: dict[str, Any]) -> None:
        self.emit("metrics_collected", MetricsCollectedEvent(metrics=wire["metrics"]))

    def _on_error(self, wire: dict[str, Any]) -> None:
        logger.error("session error", extra={"source": wire["source"], "error": str(wire["error"])})
        self.emit(
            "error",
            ErrorEvent(error=wire["error"], source=wire["source"]),
        )

    def _on_close(self, wire: dict[str, Any]) -> None:
        self.emit("close", CloseEvent(reason=CloseReason(wire["reason"]), error=None))

    def emit(self, event: EventTypes, arg: AgentEvent) -> None:  # type: ignore[override]
        self._recorded_events.append(arg)
        super().emit(event, arg)

    # --- speech handles, fillers, tasks ----------------------------------------------------

    def _speech(self, speech_id: str) -> SpeechHandle | None:
        return self._speeches.get(speech_id)

    def _open_filler(self, source: Callable[[int], Any] | None) -> int:
        self._next_filler += 1
        self._fillers[self._next_filler] = source
        return self._next_filler

    def _close_filler(self, filler: int) -> None:
        self._fillers.pop(filler, None)

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task


def __getattr__(name: str) -> Any:
    if name == "AgentSession":
        # deprecated alias, kept one release
        return LiveSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _host_value(value: Any) -> Any:
    return from_core_item(value) if isinstance(value, _core.ChatItem) else value


def _instructions(value: str | Instructions) -> str | tuple[str, str | None, str | None]:
    if isinstance(value, Instructions):
        return (value.common, value.audio, value.text)
    return str(value)


def _origin() -> str | None:
    """What the running hook or tool sends its commands as; ``None`` from outside the session."""
    binding = _BindingContextVar.get(None)
    return binding.origin if isinstance(binding, _VoiceBinding) else None


@contextlib.contextmanager
def _in_hook(hook: str, session: LiveSession[Any], agent: Agent) -> Any:
    binding = _BindingContextVar.set(_VoiceBinding(session, agent, origin="hook"))
    name = _hook_name.set(hook)
    try:
        yield
    finally:
        _hook_name.reset(name)
        _BindingContextVar.reset(binding)


async def _answer_hook(session: LiveSession[Any], wire: dict[str, Any]) -> None:
    """Run the hook the core called and answer it, exactly once."""
    core, hook, hook_id = session._core, wire["hook"], wire["id"]
    try:
        if hook in ("on_enter", "on_exit"):
            agent = session._agents[wire["agent"]]
            with _in_hook(hook, session, agent):
                if hook == "on_enter":
                    await agent._enter(RunContext(function_call=None, session=session))
                else:
                    await agent.on_exit()
            core.hook_returned(hook_id)
        elif hook == "on_user_turn_completed":
            await _turn_completed(session, wire)
        elif hook == "describe_agent":
            agent = wire["agent"]
            if agent is None:
                core.hook_failed(hook_id)
                return
            core.hook_described(hook_id, **session._frame(agent))
            session._agent = agent
        elif hook == "function_tools_executed":
            ev = FunctionToolsExecutedEvent(
                function_calls=[from_core_item(c) for c in wire["function_calls"]],
                function_call_outputs=[from_core_item(o) for o in wire["function_call_outputs"]],
            )
            ev._handoff_required = wire["handoff_required"]
            session.emit("function_tools_executed", ev)
            core.hook_tools_executed(
                hook_id,
                [o.reply_required for o in ev.function_call_outputs],
                ev.has_agent_handoff,
            )
        elif hook == "filler_source":
            source = session._fillers.get(wire["filler"])
            out = source(wire["step"]) if source is not None else None
            if asyncio.iscoroutine(out):
                out = await out
            if isinstance(out, SpeechHandle):
                core.hook_filler(hook_id, spoke=True)
            else:
                core.hook_filler(hook_id, text=out)
        else:
            # render_tool_reply: unanswered, the core uses its default templates
            core.hook_failed(hook_id)
    except Exception:
        logger.exception(f"error in the {hook} hook")
        core.hook_failed(hook_id)


async def _turn_completed(session: LiveSession[Any], wire: dict[str, Any]) -> None:
    agent = session.current_agent
    turn_ctx = session.history.copy()
    message = llm.ChatMessage(
        role="user",
        content=[wire["new_transcript"]],
        transcript_confidence=wire["transcript_confidence"],
    )
    before = [i.id for i in turn_ctx.items]
    stop = False
    with _in_hook("on_user_turn_completed", session, agent):
        try:
            await agent.on_user_turn_completed(turn_ctx, message)
        except StopResponse:
            stop = True
    changed = [i.id for i in turn_ctx.items] != before
    session._core.hook_turn_done(
        wire["id"],
        user_text=message.text_content if message.text_content != wire["new_transcript"] else None,
        items=to_core_items(turn_ctx.items) if changed else None,
        stop=stop,
    )


_HANDLERS: dict[str, Callable[[LiveSession[Any], dict[str, Any]], None]] = {
    "hook_call": LiveSession._on_hook,
    "speech_created": LiveSession._on_speech_created,
    "speech_done": LiveSession._on_speech_done,
    "conversation_item_added": LiveSession._on_item_added,
    "agent_state_changed": LiveSession._on_agent_state,
    "user_state_changed": LiveSession._on_user_state,
    "user_input_transcribed": LiveSession._on_transcript,
    "tool_execution_updated": LiveSession._on_tool_update,
    "metrics_collected": LiveSession._on_metrics,
    "error": LiveSession._on_error,
    "close": LiveSession._on_close,
}
