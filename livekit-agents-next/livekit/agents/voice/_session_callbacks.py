"""Application hooks, Python async iterators and provider callback adapters."""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable
from typing import TYPE_CHECKING, Any

from livekit import rtc

from .. import _core, llm, stt, tts
from .._core_view import (
    chat_request,
    tool_specs,
)
from .._generated.agent_run import FunctionToolInfo
from ..llm import (
    ChatContext,
    StopResponse,
)
from ..log import logger
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    USERDATA_TIMED_TRANSCRIPT,
    USERDATA_TTS_STARTED_TIME,
    FlushSentinel,
    TranscriptChunk,
)
from .agent import (
    Agent,
    ModelSettings,
    _BindingContextVar,
    _VoiceBinding,
)
from .agent_run import tool_infos
from .events import (
    FunctionToolsExecutedEvent,
    RunContext,
    UserTurnExceededEvent,
)
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from .agent_session import UserTurnExceededHook, VoiceSession

_hook_name: contextvars.ContextVar[str] = contextvars.ContextVar("lk_hook_name")
_NodeCallsVar: contextvars.ContextVar[Any] = contextvars.ContextVar("lk_node_calls")


class _DynamicInstructions:
    """An agent's ``@agent.instructions`` functions, rendered by the core before each model
    request of a reply (``_core_render``)."""

    def __init__(self, session: VoiceSession[Any], agent: Agent) -> None:
        self._session = session
        self._agent = agent

    async def _core_render(self, speech: _core.SpeechHandle) -> str:
        ctx = RunContext(
            function_call=None, session=self._session, speech_handle=SpeechHandle._of(speech)
        )
        return await self._agent._render_dynamic_instructions(ctx)


def _live_binding(operation: str) -> _VoiceBinding:
    """The live session interaction the current hook or tool runs in; raises outside one."""
    binding = _BindingContextVar.get(None)
    if not isinstance(binding, _VoiceBinding):
        raise RuntimeError(
            f"{operation} needs a live VoiceSession: await it inside a tool of the agent "
            "holding the call"
        )
    return binding


def _origin() -> str | None:
    """What the running hook or tool sends its commands as; ``None`` from outside the session."""
    binding = _BindingContextVar.get(None)
    return binding.origin if isinstance(binding, _VoiceBinding) else None


@contextlib.contextmanager
def _in_hook(hook: str, session: VoiceSession[Any], agent: Agent) -> Any:
    binding = _BindingContextVar.set(_VoiceBinding(session, agent, origin="hook"))
    name = _hook_name.set(hook)
    try:
        yield
    finally:
        _hook_name.reset(name)
        _BindingContextVar.reset(binding)


class _AgentHooks:
    """The agent's hooks as the core awaits them (``_core.SessionHooks``): each runs in the agent's
    binding and returns the core's answer; ``render_tool_reply`` is the core's default."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session
        self._toolsets = _core._core_ToolsetScopes()

    def _core_toolset_scopes(self) -> _core._core_ToolsetScopes:
        return self._toolsets

    def _core_toolset_resources(self, tools: Iterable[llm.Tool | llm.Toolset]) -> list[llm.Toolset]:
        return [tool for tool in tools if isinstance(tool, llm.Toolset)]

    async def _core_on_enter(self, agent: Agent) -> None:
        with _in_hook("on_enter", self._session, agent):
            await agent._enter(RunContext(function_call=None, session=self._session))

    async def _core_on_exit(self, agent: Agent) -> None:
        with _in_hook("on_exit", self._session, agent):
            await agent.on_exit()

    async def _core_on_user_turn_exceeded(self, agent: Agent, ev: UserTurnExceededEvent) -> None:
        session = self._session
        hook: UserTurnExceededHook[Any] = session._steps.get(
            "on_user_turn_exceeded", session.default.on_user_turn_exceeded
        )
        with _in_hook("on_user_turn_exceeded", session, agent):
            await hook(RunContext(function_call=None, session=session), ev)

    def _core_tool_infos(self, tools: list[llm.Tool | llm.Toolset]) -> list[FunctionToolInfo]:
        infos = {info.spec["name"]: info for info in tool_infos(llm.ToolContext(tools))}
        return [
            infos.get(spec["name"]) or FunctionToolInfo(spec=spec) for spec in tool_specs(tools)
        ]

    async def _core_closing(self) -> None:
        if self._session._amd is not None:
            await self._session._amd.aclose()

    async def _core_on_user_turn_completed(
        self,
        agent: Agent,
        chat_ctx: list[llm.ChatItem],
        new_transcript: str,
        transcript_confidence: float,
    ) -> tuple[str | None, list[llm.ChatItem] | None, bool]:
        session = self._session
        turn_ctx = ChatContext(list(chat_ctx))
        message = llm.ChatMessage(
            role="user", content=[new_transcript], transcript_confidence=transcript_confidence
        )
        before = [i.id for i in turn_ctx.items]
        stop = False
        if (hook := session._steps.get("on_user_turn_completed")) is not None:
            with _in_hook("on_user_turn_completed", session, agent):
                try:
                    await hook(RunContext(function_call=None, session=session), turn_ctx, message)
                except StopResponse:
                    stop = True
        changed = [i.id for i in turn_ctx.items] != before
        return message.text_content, turn_ctx.items if changed else None, stop

    async def _core_function_tools_executed(
        self,
        function_calls: list[llm.ChatItem],
        function_call_outputs: list[llm.ChatItem],
    ) -> list[bool]:
        ev = FunctionToolsExecutedEvent(
            function_calls=list(function_calls),
            function_call_outputs=list(function_call_outputs),
        )
        self._session.emit("function_tools_executed", ev)
        return [o.reply_required for o in ev.function_call_outputs]

    async def _core_filler_source(self, source: Callable[[int], Any], step: int) -> Any:
        out = source(step)
        return await out if inspect.isawaitable(out) else out

    def _core_hook_failed(self, hook: str, error: BaseException | None) -> None:
        logger.error(f"error in the {hook} hook", exc_info=error)


class _SessionNodes:
    """The session's registered nodes as the core calls them (``_core.AgentNodes``), each call in
    a copy of the agent's binding: one completion, one spoken segment, one recognition stream, one
    segment's transcript. The core feeds what they read (its
    streams, read with ``recv()``) and drains what they return."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session

    def _call(self, step: str, *args: Any, settings: ModelSettings | None = None) -> Any:
        ctx = RunContext(function_call=None, session=self._session)
        return self._session._steps[step](ctx, *args, settings or ModelSettings())

    def llm(self, req: _core.ChatRequest) -> AsyncIterator[Any]:
        chat_ctx, tools = chat_request(req)
        choice = req.tool_choice
        settings = ModelSettings(tool_choice=NOT_GIVEN if choice is None else choice)
        return _node_output(lambda: self._call("llm_node", chat_ctx, tools, settings=settings))

    def tts(self, text: Any, calls: Any) -> AsyncIterator[Any]:
        def node() -> Any:
            _NodeCallsVar.set(calls)
            return self._call("tts_node", _node_input(text))

        return _node_output(node, lambda frame: tts.SynthesizedAudio(frame=frame, request_id=""))

    def stt(self, audio: Any) -> AsyncIterator[Any]:
        return _node_output(lambda: self._call("stt_node", _node_input(audio)))

    def transcription(self, text: Any) -> AsyncIterator[Any]:
        words = _node_input(text)
        return _node_output(lambda: self._call("transcription_node", words), _transcript_chunk)


class DefaultPipeline:
    """``session.default``: the built-in pipeline steps, with the callbacks' signatures."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session

    async def llm_node(
        self,
        ctx: RunContext[Any],
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterator[llm.ChatChunk | str | FlushSentinel]:
        """A request to the session's LLM."""
        model = self._session.llm
        assert isinstance(model, llm.LLM), "llm_node needs an LLM (not a realtime model)"
        conn_options = self._session.conn_options.llm_conn_options or DEFAULT_API_CONNECT_OPTIONS
        async with model.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            tool_choice=model_settings.tool_choice,
            conn_options=conn_options,
        ) as stream:
            async for chunk in stream:
                yield chunk

    def stt_node(
        self,
        ctx: RunContext[Any],
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> AsyncIterator[stt.SpeechEvent]:
        """The session's STT; one that does not stream is wrapped with its VAD."""
        session = self._session
        if session.stt is None:
            raise RuntimeError("stt_node called but no STT is available")
        return _core.stt_node(
            session.stt, audio, conn_options=session.conn_options.stt_conn_options, vad=session.vad
        )

    def tts_node(
        self, ctx: RunContext[Any], text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        """The session's TTS; one that does not stream is fed sentence by sentence."""
        session = self._session
        if session.tts is None:
            raise RuntimeError("tts_node called but no TTS is available")
        conn_options = session.conn_options.tts_conn_options
        calls = _NodeCallsVar.get(None)
        frames = _core.tts_node(session.tts, text, conn_options=conn_options, calls=calls)
        return (_stamped(frame) async for frame in frames)

    async def transcription_node(
        self,
        ctx: RunContext[Any],
        text: AsyncIterable[TranscriptChunk],
        model_settings: ModelSettings,
    ) -> AsyncIterator[str | TranscriptChunk]:
        """The transcript as spoken."""
        async for delta in text:
            yield delta

    async def on_user_turn_exceeded(self, ctx: RunContext[Any], ev: UserTurnExceededEvent) -> None:
        """A short reply cutting in on the person, from the turn's transcript."""
        await self._session.generate_reply(
            user_input=ev.transcript,
            **_core.user_turn_exceeded_reply(),
        )


def _stamped(node_frame: Any) -> rtc.AudioFrame:
    """A default `tts_node` frame as the reference yields it: its `userdata` carries the words it
    covers and when its synthesis started (a `perf_counter()` stamp)."""
    frame: rtc.AudioFrame = node_frame.frame
    if node_frame.timed_transcript:
        frame.userdata[USERDATA_TIMED_TRANSCRIPT] = node_frame.timed_transcript
    if node_frame.started_time is not None:
        elapsed = time.time() - node_frame.started_time
        frame.userdata[USERDATA_TTS_STARTED_TIME] = time.perf_counter() - elapsed
    return frame


async def _node_input(stream: Any, item: Callable[[Any], Any] = lambda x: x) -> AsyncIterator[Any]:
    """A core stream as the node reads it."""
    while (value := await stream.recv()) is not None:
        yield item(value)


def _transcript_chunk(value: str | TranscriptChunk) -> TranscriptChunk:
    """Normalize plain text once; existing chunks retain every metadata field."""
    return TranscriptChunk(value) if isinstance(value, str) else value


async def _node_output(
    node: Callable[[], Any], item: Callable[[Any], Any] = lambda x: x
) -> AsyncIterator[Any]:
    """What a node returned, read as the reference's node task reads it: an awaitable is awaited
    first, an async iterable is read to its end (and closed with this one), a single value is the
    one item, ``None`` is none."""
    out = node()
    if inspect.isawaitable(out):
        out = await out
    if out is None:
        return
    if not isinstance(out, AsyncIterable):
        yield item(out)
        return
    iterator = aiter(out)
    try:
        async for value in iterator:
            yield item(value)
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()
