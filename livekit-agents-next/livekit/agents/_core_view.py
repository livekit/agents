"""What the Rust core calls on a host object that is not a generated base, and the values it
crosses with.

A core proxy calls ``_core_<method>`` on the object it was handed. The provider bases (``stt.STT``,
``tts.TTS``, ``llm.LLM``, ``vad.VAD``, ``llm.RealtimeModel`` and their streams) carry that
surface themselves, generated with them. An io leg is seen through its view here (the proxy's
declared ``view``), a room through ``duck(obj, cls)``, ``cls`` naming the base class the core
expects; the duplex and turn-detection views answer over a plugin's public API.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Never, Protocol, cast

from livekit import rtc

from . import _core, llm
from ._generated import duplex, realtime_wire, rtc_wire
from ._generated.llm_fallback import AvailabilityChanged
from ._generated.tool_options import ProviderToolDeclaration, ToolSpec
from ._generated.turn_events import EotPrediction
from ._generated.types import ProviderId
from .llm.chat_context import ChatItem
from .llm.tool_context import (
    ProviderTool,
    Tool,
    ToolFlag,
    Toolset,
)
from .types import NOT_GIVEN, NotGivenOr, TranscriptChunk
from .voice import io
from .voice.io import _StreamingTurnDetector


def duck(obj: Any, cls: type) -> Any:
    """The view of ``obj`` as the core's ``cls``: called by the core, once per handed object."""
    for base, view in _views().items():
        if issubclass(cls, base):
            return view(obj)
    raise TypeError(f"no core view of {type(obj).__name__} as {cls.__qualname__}")


def chat_request(req: _core.ChatRequest) -> tuple[llm.ChatContext, list[Tool]]:
    """The context and the offered tools of a request the core makes."""
    if req.chat_ctx_host is not None:
        chat_ctx = cast(llm.ChatContext, req.chat_ctx_host)
    else:
        chat_ctx = llm.ChatContext.from_dict(req.chat_ctx)
    pools = [req.tools_host, *req.tools_host_extra]
    held = [cast("Iterable[Tool]", pool) for pool in pools if pool is not None]
    tools = offered_tools([t for pool in held for t in pool], req.tool_names)
    names = {
        tool.info.name
        for tool in tools
        if llm.is_function_tool(tool) or llm.is_raw_function_tool(tool)
    }
    for schema in req.function_schemas:
        if schema["name"] not in names:
            tools.append(llm.function_tool(raw_schema=schema)(_refuse_native_tool_call))
            names.add(schema["name"])
    return chat_ctx, tools


def _refuse_native_tool_call(*args: object, **kwargs: object) -> Never:
    raise RuntimeError("this tool is executed by the native agent and cannot be called directly")


def realtime_model(model: llm.RealtimeModel, tools: Callable[[], Sequence[Tool | Toolset]]) -> Any:
    """``model`` as the core takes it: its own model for an OpenAI-protocol model or a fallback
    chain, else a plugin's model bound to the session's tools."""
    from .llm._realtime import RealtimeModel as OpenAIRealtimeModel
    from .llm.realtime import RealtimeModelFallbackAdapter

    if isinstance(model, OpenAIRealtimeModel):
        return model._core_native()
    if isinstance(model, RealtimeModelFallbackAdapter):
        return model._core_native([realtime_model(m, tools) for m in model._models])
    if isinstance(model, llm.DuplexRealtimeAdapter):
        return _core.DuplexRealtimeModel(Duplex(model, tools), audio_timeout=model._audio_timeout)
    return _core.HostRealtimeModel(model._core_with_tools(tools))


def forward_session_events(
    session: llm.RealtimeSession | llm.DuplexSession,
    put: Callable[[realtime_wire.Event], None],
) -> None:
    """A realtime or duplex session's caller events, reconnects and errors, as the core reads
    them."""
    session.on("input_speech_started", lambda _: put(realtime_wire.InputSpeechStartedEvent()))
    session.on(
        "input_speech_stopped",
        lambda ev: put(dataclasses.replace(ev)),
    )
    session.on(
        "input_audio_transcription_completed",
        lambda ev: put(dataclasses.replace(ev)),
    )
    session.on(
        "session_reconnected", lambda _: put(realtime_wire.RealtimeSessionReconnectedEvent())
    )
    session.on("metrics_collected", lambda ev: put(realtime_wire.RealtimeMetrics(metrics=ev)))
    session.on(
        "error",
        lambda ev: put(
            realtime_wire.HostRealtimeError(
                label=ev.label,
                message=str(ev.error),
                recoverable=ev.recoverable,
                timestamp=ev.timestamp,
            )
        ),
    )


class Duplex:
    """A ``DuplexRealtimeAdapter``'s model as the core's duplex host: the core gates and segments
    the output; ``tools`` names the tool objects the core refers to by name."""

    def __init__(
        self, adapter: llm.DuplexRealtimeAdapter, tools: Callable[[], Sequence[Tool | Toolset]]
    ) -> None:
        self._adapter = adapter
        self._tools = tools

    def _core_label(self) -> str:
        return self._adapter.duplex_model.label

    def _core_capabilities(self) -> duplex.DuplexCapabilities:
        return dataclasses.replace(self._adapter.duplex_model.capabilities)

    def _core_gate(self) -> llm.AudioGate | None:
        """The gate the adapter or the model declares; ``None``: the adaptive gate."""
        return (
            self._adapter._gate()
            if self._adapter._gate
            else self._adapter.duplex_model.audio_gate()
        )

    async def _core_session(self) -> _DuplexSession:
        """A new plugin session."""
        return _DuplexSession(self._adapter.duplex_model.session(), self._tools)


class _DuplexSession:
    """A plugin's ``DuplexSession`` flattened: the core applies the updates in order and pulls
    the output audio; the session's callbacks report one event at a time."""

    def __init__(
        self, session: llm.DuplexSession, tools: Callable[[], Sequence[Tool | Toolset]]
    ) -> None:
        self._session = session
        self._tools = tools
        self._audio = aiter(session.audio_stream)
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        put = self._events.put_nowait
        forward_session_events(session, put)
        session.on(
            "transcript_delta",
            lambda ev: put(dataclasses.replace(ev)),
        )
        session.on("function_call", lambda call: put(realtime_wire.FunctionCallReady(call=call)))

    def _core_push_audio(self, frame: rtc.AudioFrame) -> None:
        self._session.push_audio(frame)

    async def _core_configure(
        self, instructions: str | None, items: list[ChatItem] | None, names: list[str]
    ) -> None:
        await self._session._update_session(
            instructions=instructions or NOT_GIVEN,
            chat_ctx=_chat_ctx(items),
            tools=offered_tools(self._tools(), names),
        )

    async def _core_update_instructions(self, instructions: str) -> None:
        await self._session._update_instructions(instructions)

    async def _core_append_items(self, items: list[ChatItem]) -> None:
        await self._session._append_items(items)

    async def _core_update_tools(self, names: list[str]) -> None:
        await self._session._update_tools(offered_tools(self._tools(), names))

    async def _core_generate(
        self, instructions: str | None, tool_choice: llm.ToolChoice | None
    ) -> None:
        self._session._generate_reply(
            instructions=instructions or NOT_GIVEN,
            tool_choice=NOT_GIVEN if tool_choice is None else tool_choice,
        )

    async def _core_next_audio(self) -> duplex.DuplexAudioFrame | None:
        try:
            item = await anext(self._audio)
        except StopAsyncIteration:
            return None
        return dataclasses.replace(item)

    async def _core_next_event(self) -> Any:
        return await self._events.get()

    async def _core_close_plugin(self) -> None:
        await self._session.aclose()

    async def _core_aclose(self) -> None:
        await self._session.aclose()
        self._events.put_nowait(None)


def _chat_ctx(items: list[ChatItem] | None) -> NotGivenOr[llm.ChatContext]:
    return NOT_GIVEN if items is None else llm.ChatContext(items)


class FallbackListener(Protocol):
    """A fallback adapter as its chain reports to it: an entry, by its index in the adapter's
    list, went unavailable or recovered; every entry is unavailable."""

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None: ...

    def _core_all_unavailable(self) -> None: ...


class TurnDetector:
    """A ``turn_detection`` object as the core's end-of-turn detector: its ``stream()`` hears
    the user's audio and predicts the end of their turn."""

    def __init__(self, detector: _StreamingTurnDetector) -> None:
        if not hasattr(detector, "stream"):
            raise TypeError(
                f"turn_detection={detector!r} is not an audio turn detector (one with stream()); "
                "use livekit.agents.inference.TurnDetector"
            )
        self._detector = detector
        self._stream = detector.stream()

    def _core_id(self) -> ProviderId:
        cls = type(self._detector)
        label = f"{cls.__module__}.{cls.__name__}"
        return {"label": label, "model": self._detector.model, "provider": self._detector.provider}

    def _core_push_audio(self, frame: rtc.AudioFrame) -> None:
        self._stream.push_audio(frame)

    def _core_cancel(self, timed_out: bool) -> None:
        self._stream.cancel_inference(timed_out=timed_out)

    def _core_flush(self, reason: str | None) -> None:
        self._stream.flush(reason)

    async def _core_predict(self, timeout: float | None) -> EotPrediction:
        ev = await self._stream.predict()
        return EotPrediction(
            end_of_turn_probability=ev.end_of_turn_probability,
            last_speaking_time=ev.last_speaking_time,
            detection_delay=ev.detection_delay,
            inference_duration=ev.inference_duration,
            backchannel_probability=ev.backchannel_probability,
        )

    async def _core_aclose(self) -> None:
        await self._stream.aclose()


# --- io legs ----------------------------------------------------------------------------------


class _AudioInput:
    def __init__(self, obj: io.AudioInput) -> None:
        self._obj = obj

    def _core_label(self) -> str:
        return self._obj.label

    async def _core_next_frame(self) -> Any:
        try:
            return await self._obj.__anext__()
        except StopAsyncIteration:
            return None

    def _core_on_attached(self) -> None:
        self._obj.on_attached()

    def _core_on_detached(self) -> None:
        self._obj.on_detached()


class _VideoInput(_AudioInput):
    async def _core_next_frame(self) -> Any:
        frame = await super()._core_next_frame()
        if frame is None:
            return None
        return rtc_wire.VideoFrameRef(
            frame=frame, width=frame.width, height=frame.height, timestamp_us=frame.timestamp_us
        )


class _AudioOutput:
    """``io.AudioOutput`` as a core sink, and as the listener of a core leg it wraps: the sink's
    playback reports go into its ``PlaybackTracker``, the leg's into its ``on_playback_*``."""

    def __init__(self, obj: io.AudioOutput) -> None:
        self._obj = obj

    def _core_label(self) -> str:
        return self._obj.label

    def _core_tracker(self) -> _core.PlaybackTracker:
        return self._obj._tracker

    def _core_capabilities(self) -> io.AudioOutputCapabilities:
        return io.AudioOutputCapabilities(pause=self._obj.can_pause)

    def _core_sample_rate(self) -> int | None:
        return self._obj.sample_rate

    def _core_begin_segment(self, segment: io.PlaybackSegment) -> None:
        self._obj.begin_segment(segment)

    async def _core_capture(self, frame: rtc.AudioFrame) -> None:
        await self._obj.capture_frame(frame)

    def _core_flush_segment(self) -> None:
        self._obj.flush()

    def _core_clear_buffer(self) -> None:
        self._obj.clear_buffer()

    def _core_on_attached(self) -> None:
        self._obj.on_attached()

    def _core_on_detached(self) -> None:
        self._obj.on_detached()

    def _core_pause(self) -> None:
        self._obj.pause()

    def _core_resume(self) -> None:
        self._obj.resume()

    def _core_on_playback_started(self, segment: io.PlaybackSegment, created_at: float) -> None:
        self._obj.on_playback_started(segment=segment, created_at=created_at)

    def _core_on_playback_progressed(
        self, segment: io.PlaybackSegment, started_at: float, offset: float, duration: float
    ) -> None:
        self._obj.on_playback_progressed(
            segment=segment, started_at=started_at, offset=offset, duration=duration
        )

    def _core_on_playback_finished(
        self,
        segment: io.PlaybackSegment,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: str | None,
    ) -> None:
        self._obj.on_playback_finished(
            segment=segment,
            playback_position=playback_position,
            interrupted=interrupted,
            synchronized_transcript=synchronized_transcript,
        )


class _TextOutput:
    def __init__(self, obj: io.TextOutput) -> None:
        self._obj = obj

    def _core_label(self) -> str:
        return self._obj.label

    async def _core_capture_timed(self, text: TranscriptChunk) -> None:
        await self._obj.capture_transcript(text)

    async def _core_capture_text(self, text: str) -> None:
        await self._obj.capture_text(text)

    def _core_flush(self) -> None:
        self._obj.flush()

    def _core_on_attached(self) -> None:
        self._obj.on_attached()

    def _core_on_detached(self) -> None:
        self._obj.on_detached()


@functools.cache
def _views() -> dict[type, Any]:
    from .voice.room_io import Room

    return {
        io.AudioInput: _AudioInput,
        io.VideoInput: _VideoInput,
        io.AudioOutput: _AudioOutput,
        io.TextOutput: _TextOutput,
        Room: Room.of,
    }


# --- tools ------------------------------------------------------------------------------------


def tool_specs(tools: Sequence[Tool | Toolset]) -> list[ToolSpec]:
    """Project function schemas and provider declarations through the core's tool boundary."""
    from .llm.utils import tool_schema

    specs: list[ToolSpec] = []
    for tool in llm.ToolContext(tools).flatten():
        ignore_on_enter = False
        provider_tool: ProviderToolDeclaration | None
        if isinstance(tool, ProviderTool):
            name, description, parameters = tool.id, "", None
            provider_tool = {"provider": tool.provider, "payload": tool.to_dict()}
        elif llm.is_function_tool(tool) or llm.is_raw_function_tool(tool):
            name, description, parameters = tool_schema(tool)
            ignore_on_enter = ToolFlag.IGNORE_ON_ENTER in tool.info.flags
            provider_tool = None
        else:
            continue
        spec: ToolSpec = {
            "name": name,
            "description": description,
            "json_schema": parameters,
            "ignore_on_enter": ignore_on_enter,
        }
        if provider_tool is not None:
            spec["provider_tool"] = provider_tool
        specs.append(spec)
    return specs


def offered_tools(pool: Sequence[Tool | Toolset], names: Sequence[str]) -> list[Tool]:
    """The tools of ``pool`` a request offers: the named function tools and every provider tool."""
    ctx = llm.ToolContext(pool)
    named = [t for n in names if (t := ctx.get_function_tool(n)) is not None]
    return [*named, *(t for t in ctx.flatten() if isinstance(t, ProviderTool))]
