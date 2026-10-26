"""What the Rust core calls on a host object, and the values it crosses with.

A core proxy calls ``_core_<method>`` on the object it was handed. An object without that surface
(every plugin and io leg) is seen through ``duck(obj, cls)``: ``cls`` names the base class the
core expects, and the view here answers for it over the object's public API. Records the core
reads by attribute (``SpeechEvent``, ``ChatChunk``, ``SynthesizedAudio``, the capabilities) are
the plugin's own objects; chat items cross as ``_core.ChatItem``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import TypeAdapter

from . import _core, llm, stt, tts, vad
from .llm.chat_context import AudioContent, ChatItem, ImageContent
from .llm.tool_context import (
    FunctionTool,
    ProviderTool,
    RawFunctionTool,
    Tool,
    ToolFlag,
    Toolset,
)
from .llm.utils import function_arguments_to_pydantic_model
from .types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions
from .voice import io


def duck(obj: Any, cls: type) -> Any:
    """The view of ``obj`` as the core's ``cls``: called by the core, once per handed object."""
    for base, view in _views().items():
        if issubclass(cls, base):
            return view(obj)
    raise TypeError(f"no core view of {type(obj).__name__} as {cls.__qualname__}")


def _identity(obj: Any) -> dict[str, str]:
    return {"label": obj.label, "model": obj.model, "provider": obj.provider}


# --- providers --------------------------------------------------------------------------------


class _Provider:
    """A provider view; ``_core_with_conn_options`` binds the budget its calls carry."""

    def __init__(self, obj: Any, conn: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> None:
        self._obj = obj
        self._conn = conn

    def _core_with_conn_options(self, conn: dict[str, Any]) -> Any:
        return type(self)(self._obj, APIConnectOptions(**conn))


class _Stt(_Provider):
    _obj: stt.STT

    def _core_label(self) -> str:
        return self._obj.label

    def _core_capabilities(self) -> stt.STTCapabilities:
        return self._obj.capabilities

    def _core_id(self) -> dict[str, str]:
        return _identity(self._obj)

    def _core_prewarm(self) -> None:
        self._obj.prewarm()

    def _core_update_keyterms(self, keyterms: list[str]) -> None:
        self._obj._update_session_keyterms(keyterms)

    def _core_push_conversation_item(self, role: str, text: str) -> None:
        from .voice.events import ConversationItemAddedEvent

        item = llm.ChatMessage(role=role, content=[text])  # type: ignore[arg-type]
        self._obj._push_conversation_item(ConversationItemAddedEvent(item=item))

    async def _core_recognize(self, buffer: list[Any], language: str | None) -> stt.SpeechEvent:
        return await self._obj.recognize(
            buffer, language=language or NOT_GIVEN, conn_options=self._conn
        )

    async def _core_stream(
        self, language: str | None, start_time_offset: float | None
    ) -> _InputStream | None:
        if not self._obj.capabilities.streaming:
            return None
        stream = self._obj.stream(language=language or NOT_GIVEN, conn_options=self._conn)
        if start_time_offset is not None:
            stream.start_time_offset = start_time_offset
        return _InputStream(stream, stream.push_frame)


class _Tts(_Provider):
    _obj: tts.TTS

    def _core_label(self) -> str:
        return self._obj.label

    def _core_capabilities(self) -> tts.TTSCapabilities:
        return self._obj.capabilities

    def _core_id(self) -> dict[str, str]:
        return _identity(self._obj)

    def _core_prewarm(self) -> None:
        self._obj.prewarm()

    def _core_sample_rate(self) -> int:
        return self._obj.sample_rate

    def _core_num_channels(self) -> int:
        return self._obj.num_channels

    async def _core_synthesize(self, text: str) -> tts.ChunkedStream:
        return self._obj.synthesize(text, conn_options=self._conn)

    async def _core_stream(self) -> _InputStream:
        if not self._obj.capabilities.streaming:
            raise RuntimeError(f"{self._obj.label} has no duplex path")
        stream = self._obj.stream(conn_options=self._conn)
        return _InputStream(stream, stream.push_text)


class _Vad:
    def __init__(self, obj: vad.VAD) -> None:
        self._obj = obj

    def _core_label(self) -> str:
        return self._obj._label

    def _core_capabilities(self) -> vad.VADCapabilities:
        return self._obj.capabilities

    def _core_id(self) -> dict[str, str]:
        return {"label": self._obj._label, "model": self._obj.model, "provider": self._obj.provider}

    def _core_prewarm(self) -> None:
        pass

    async def _core_stream(self) -> _InputStream:
        stream = self._obj.stream()
        return _InputStream(stream, stream.push_frame)


class _InputStream:
    """A plugin stream as the core drives it: ``_core_push`` in, ``__anext__`` out. The core's
    end of input may arrive after it closed the stream, and a plugin stream raises on that."""

    def __init__(self, stream: Any, push: Any) -> None:
        self._stream = stream
        self._core_push = push
        self._ended = False

    def flush(self) -> None:
        if not self._ended:
            self._stream.flush()

    def end_input(self) -> None:
        if not self._ended:
            self._ended = True
            self._stream.end_input()

    async def __anext__(self) -> Any:
        return await self._stream.__anext__()

    async def aclose(self) -> None:
        self._ended = True
        await self._stream.aclose()


class _Llm(_Provider):
    _obj: llm.LLM

    def _core_label(self) -> str:
        return self._obj.label

    def _core_model(self) -> str:
        return self._obj.model

    def _core_id(self) -> dict[str, str]:
        return _identity(self._obj)

    def _core_prewarm(self) -> None:
        self._obj.prewarm()

    async def _core_chat(self, req: _core.ChatRequest) -> llm.LLMStream:
        chat_ctx, tools = chat_request(req)
        choice = req.tool_choice
        return self._obj.chat(
            chat_ctx=chat_ctx,
            conn_options=self._conn,
            tools=tools,
            tool_choice=NOT_GIVEN if choice is None else choice,
            parallel_tool_calls=(
                NOT_GIVEN if req.parallel_tool_calls is None else req.parallel_tool_calls
            ),
            extra_kwargs=req.extra or NOT_GIVEN,
        )


def chat_request(req: _core.ChatRequest) -> tuple[llm.ChatContext, list[Tool]]:
    """The context and the offered tools of a request the core makes."""
    if req.chat_ctx_host is not None:
        chat_ctx = _core.resolve_host_ref(req.chat_ctx_host)
    else:
        chat_ctx = llm.ChatContext.from_dict(req.chat_ctx)
    pools = [req.tools_host, *req.tools_host_extra]
    tools = offered_tools(
        [t for key in pools if key is not None for t in _core.resolve_host_ref(key)],
        req.tool_names,
    )
    return chat_ctx, tools


class Realtime:
    """A plugin's ``RealtimeModel`` as the core's host realtime model. ``tools`` names the tool
    objects the core refers to by name (the running frame's pool)."""

    def __init__(self, model: llm.RealtimeModel, tools: Callable[[], Sequence[Tool | Toolset]]):
        self._model = model
        self._tools = tools

    def _core_label(self) -> str:
        return self._model.label

    def _core_capabilities(self) -> dict[str, Any]:
        return dataclasses.asdict(self._model.capabilities)

    async def _core_session(self, turn_detection_disabled: bool) -> _RealtimeSession:
        session = self._model.session(turn_detection_disabled=turn_detection_disabled)
        return _RealtimeSession(session, self._tools)


class _RealtimeSession:
    """A plugin's ``RealtimeSession`` flattened: commands in, one ``kind``-tagged event out at a
    time, each generation's message and function streams drained in order."""

    def __init__(
        self, session: llm.RealtimeSession, tools: Callable[[], Sequence[Tool | Toolset]]
    ) -> None:
        self._session = session
        self._tools = tools
        self._events: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._replies: dict[str, asyncio.Future[Any]] = {}
        # replies cancelled before they were asked for, behind a session update
        self._cancelled: set[str] = set()
        # the last session update: updates apply in order, and a reply waits for them
        self._updating: asyncio.Task[Any] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        put = self._events.put_nowait
        session.on("input_speech_started", lambda _: put({"kind": "input_speech_started"}))
        session.on(
            "input_speech_stopped",
            lambda ev: put(
                {
                    "kind": "input_speech_stopped",
                    "user_transcription_enabled": ev.user_transcription_enabled,
                }
            ),
        )
        session.on(
            "input_audio_transcription_completed",
            lambda ev: put(
                {
                    "kind": "input_transcription",
                    "item_id": ev.item_id,
                    "transcript": ev.transcript,
                    "is_final": ev.is_final,
                    "confidence": ev.confidence,
                    "turn_started_at": ev.turn_started_at,
                }
            ),
        )
        session.on(
            "generation_created",
            lambda ev: None if ev.user_initiated else self._spawn(self._generation(ev, None)),
        )
        session.on(
            "remote_item_added",
            lambda ev: put(
                {
                    "kind": "remote_item_added",
                    "previous_item_id": ev.previous_item_id,
                    "item": to_core_item(ev.item),
                }
            ),
        )
        session.on("session_reconnected", lambda _: put({"kind": "session_reconnected"}))
        session.on(
            "error",
            lambda ev: put(
                {
                    "kind": "error",
                    "label": ev.label,
                    "message": str(ev.error),
                    "recoverable": ev.recoverable,
                    "timestamp": ev.timestamp,
                }
            ),
        )

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _update(self, apply: Callable[[], Awaitable[Any]]) -> None:
        """Run a session update after the previous one: what it reads is current when it runs."""
        previous = self._updating

        async def _run() -> None:
            if previous is not None:
                await asyncio.wait([previous])
            await apply()

        self._updating = self._spawn(_run())

    def _after_updates(self, req_id: str, ask: Callable[[], asyncio.Future[Any]]) -> None:
        """Ask for a reply once the updates sent before it applied, as the reference awaits
        them: a tool's output reaches the server before the reply that reads it."""
        pending = self._updating

        def _ask() -> None:
            if req_id in self._cancelled:
                self._cancelled.discard(req_id)
                return
            self._reply(req_id, ask())

        if pending is None or pending.done():
            _ask()
            return

        async def _run() -> None:
            await asyncio.wait([pending])
            _ask()

        self._spawn(_run())

    async def _generation(self, ev: llm.GenerationCreatedEvent, req_id: str | None) -> None:
        put = self._events.put_nowait
        response_id = ev.response_id or ""
        put({"kind": "generation_created", "response_id": response_id, "req_id": req_id})
        ids = {"response_id": ev.response_id}

        async def _message(msg: llm.MessageGeneration) -> None:
            mid = {"message_id": msg.message_id, **ids}
            put({"kind": "message_started", **mid})

            async def _text() -> None:
                async for delta in msg.text_stream:
                    put({"kind": "text_delta", "delta": str(delta), **mid})

            async def _audio() -> None:
                async for frame in msg.audio_stream:
                    put({"kind": "audio_delta", "frame": frame, **mid})

            async def _modalities() -> None:
                modalities = await msg.modalities
                kind = "audio" if "audio" in modalities else "text"
                put({"kind": "message_modalities", "modalities": kind, **mid})

            await asyncio.gather(_text(), _audio(), _modalities())
            put({"kind": "message_done", **mid})

        async def _messages() -> None:
            # each message forwards as it arrives, not once the server has sent them all
            started = []
            async for m in ev.message_stream:
                started.append(asyncio.ensure_future(_message(m)))
            await asyncio.gather(*started)

        async def _calls() -> None:
            async for call in ev.function_stream:
                put({"kind": "function_call", "call": to_core_item(call)})

        await asyncio.gather(_messages(), _calls())
        put({"kind": "generation_done", "response_id": response_id})

    def _reply(self, req_id: str, fut: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
        self._replies[req_id] = fut

        def _done(fut: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            self._replies.pop(req_id, None)
            if fut.cancelled():
                return
            if (exc := fut.exception()) is not None:
                self._events.put_nowait(
                    {
                        "kind": "generate_reply_failed",
                        "req_id": req_id,
                        "message": str(exc),
                        "code": getattr(exc, "code", None),
                    }
                )
                return
            self._spawn(self._generation(fut.result(), req_id))

        fut.add_done_callback(_done)

    def _core_send(self, cmd: dict[str, Any]) -> None:
        session, kind = self._session, cmd["kind"]
        if kind == "push_audio":
            session.push_audio(cmd["frame"])
        elif kind == "generate_reply":
            choice = cmd["tool_choice"]
            self._after_updates(
                cmd["req_id"],
                lambda: session.generate_reply(
                    instructions=cmd["instructions"] or NOT_GIVEN,
                    tool_choice=NOT_GIVEN if choice is None else choice,
                ),
            )
        elif kind == "say":
            self._after_updates(cmd["req_id"], lambda: session.say(cmd["text"]))
        elif kind == "cancel_reply":
            if (fut := self._replies.pop(cmd["req_id"], None)) is not None:
                fut.cancel()
            else:
                self._cancelled.add(cmd["req_id"])
        elif kind == "interrupt":
            session.interrupt()
        elif kind == "commit_audio":
            session.commit_audio()
        elif kind == "clear_audio":
            session.clear_audio()
        elif kind == "truncate":
            transcript = cmd["audio_transcript"]
            session.truncate(
                message_id=cmd["message_id"],
                modalities=["audio", "text"] if cmd["has_audio"] else ["text"],
                audio_end_ms=cmd["audio_end_ms"],
                audio_transcript=NOT_GIVEN if transcript is None else transcript,
            )
        elif kind == "update_instructions":
            self._update(lambda: session.update_instructions(cmd["instructions"]))
        elif kind == "update_chat_ctx":
            ctx = llm.ChatContext([from_core_item(i) for i in cmd["items"]])
            self._update(lambda: session.update_chat_ctx(ctx))
        elif kind == "push_item":
            item = from_core_item(cmd["item"])

            def _pushed() -> Awaitable[None]:
                ctx = session.chat_ctx.copy()
                ctx.insert(item)
                return session.update_chat_ctx(ctx)

            self._update(_pushed)
        elif kind == "update_tools":
            tools = offered_tools(self._tools(), cmd["names"])
            self._update(lambda: session.update_tools(tools))

    async def _core_next_event(self) -> dict[str, Any] | None:
        return await self._events.get()

    async def _core_aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await self._session.aclose()
        self._events.put_nowait(None)


class TurnDetector:
    """A ``turn_detection`` object as the core's end-of-turn detector: a streaming one
    (``stream()``) hears the user's audio, a batch one scores the conversation."""

    def __init__(self, detector: Any) -> None:
        self._detector = detector
        self.streaming = hasattr(detector, "stream")
        self._stream = detector.stream() if self.streaming else None

    def _core_id(self) -> dict[str, str]:
        cls = type(self._detector)
        label = f"{cls.__module__}.{cls.__name__}"
        return {"label": label, "model": self._detector.model, "provider": self._detector.provider}

    def _core_push_audio(self, frame: Any) -> None:
        if self._stream is not None:
            self._stream.push_audio(frame)

    def _core_cancel(self, timed_out: bool) -> None:
        if self._stream is not None:
            self._stream.cancel_inference(timed_out=timed_out)

    def _core_flush(self, reason: str) -> None:
        if self._stream is not None:
            self._stream.flush(reason)

    async def _core_predict(
        self, chat_ctx: _core.TurnContext, timeout: float | None
    ) -> dict[str, Any]:
        if self._stream is not None:
            ev = await self._stream.predict()
            return {
                "end_of_turn_probability": ev.end_of_turn_probability,
                "last_speaking_time": ev.last_speaking_time,
                "detection_delay": ev.detection_delay,
                "inference_duration": ev.inference_duration,
                "backchannel_probability": ev.backchannel_probability,
            }
        ctx = llm.ChatContext.from_dict(chat_ctx.to_host_dict())
        probability = await self._detector.predict_end_of_turn(ctx, timeout=timeout)
        return {"end_of_turn_probability": probability}

    async def _core_aclose(self) -> None:
        if self._stream is not None:
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
        return _core.VideoFrameRef(frame, frame.width, frame.height, frame.timestamp_us)


class _AudioOutput:
    """``io.AudioOutput`` as a core sink, and as the listener of a core leg it wraps: the sink's
    playback reports go into its ``PlaybackTracker``, the leg's into its ``on_playback_*``."""

    def __init__(self, obj: io.AudioOutput) -> None:
        self._obj = obj

    def _core_label(self) -> str:
        return self._obj.label

    def _core_tracker(self) -> _core.PlaybackTracker:
        return self._obj._tracker

    def _core_capabilities(self) -> _core.AudioOutputCapabilities:
        return _core.AudioOutputCapabilities(pause=self._obj.can_pause)

    def _core_sample_rate(self) -> int | None:
        return self._obj.sample_rate

    def _core_next_in_chain(self) -> None:
        # the host chain is the host's: the core sees its head as one sink
        return None

    async def _core_capture(self, frame: Any) -> None:
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

    def _core_on_playback_started(self, created_at: float) -> None:
        self._obj.on_playback_started(created_at=created_at)

    def _core_on_playback_progressed(
        self, started_at: float, offset: float, duration: float
    ) -> None:
        self._obj.on_playback_progressed(started_at=started_at, offset=offset, duration=duration)

    def _core_on_playback_finished(
        self, playback_position: float, interrupted: bool, synchronized_transcript: str | None
    ) -> None:
        self._obj.on_playback_finished(
            playback_position=playback_position,
            interrupted=interrupted,
            synchronized_transcript=synchronized_transcript,
        )


class _TextOutput:
    def __init__(self, obj: io.TextOutput) -> None:
        self._obj = obj

    def _core_label(self) -> str:
        return self._obj.label

    def _core_next_in_chain(self) -> None:
        return None

    async def _core_capture_timed(self, text: Any) -> None:
        await self._obj.capture_text(text)

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
    from .voice.room_io import _core_rtc

    return {
        stt.STT: _Stt,
        tts.TTS: _Tts,
        vad.VAD: _Vad,
        llm.LLM: _Llm,
        io.AudioInput: _AudioInput,
        io.VideoInput: _VideoInput,
        io.AudioOutput: _AudioOutput,
        io.TextOutput: _TextOutput,
        _core_rtc.Room: _core_rtc.Room,
    }


# --- tools ------------------------------------------------------------------------------------


def tool_specs(tools: Sequence[Tool | Toolset]) -> list[dict[str, Any]]:
    """The core's ``ToolSpec`` of each function tool; provider tools stay in the host pool."""
    specs = []
    for tool in llm.ToolContext(tools).function_tools.values():
        if isinstance(tool, RawFunctionTool):
            schema = tool.info.raw_schema
            name, description = schema["name"], schema.get("description", "")
            parameters = schema.get("parameters", {})
        else:
            assert isinstance(tool, FunctionTool)
            name, description = tool.info.name, tool.info.description or ""
            parameters = function_arguments_to_pydantic_model(tool).model_json_schema()
        specs.append(
            {
                "name": name,
                "description": description,
                "json_schema": parameters,
                "ignore_on_enter": ToolFlag.IGNORE_ON_ENTER in tool.info.flags,
            }
        )
    return specs


def offered_tools(pool: Sequence[Tool | Toolset], names: Sequence[str]) -> list[Tool]:
    """The tools of ``pool`` a request offers: the named function tools and every provider tool."""
    ctx = llm.ToolContext(pool)
    named = [t for n in names if (t := ctx.get_function_tool(n)) is not None]
    return [*named, *(t for t in ctx.flatten() if isinstance(t, ProviderTool))]


# --- chat items -------------------------------------------------------------------------------

_ITEM = TypeAdapter(ChatItem)


def to_core_item(item: ChatItem) -> _core.ChatItem:
    fields = item.model_dump(exclude={"type", "hash"})
    match item.type:
        case "message":
            fields["content"] = [_to_core_content(c) for c in item.content]
            return _core.ChatItem.message(**fields)
        case "function_call":
            return _core.ChatItem.function_call(**fields)
        case "function_call_output":
            return _core.ChatItem.function_call_output(**fields)
        case "agent_handoff":
            return _core.ChatItem.agent_handoff(**fields)
        case "agent_config_update":
            return _core.ChatItem.agent_config_update(**fields)
    raise TypeError(f"unknown chat item {item.type!r}")


def to_core_items(items: Sequence[ChatItem]) -> list[_core.ChatItem]:
    return [to_core_item(i) for i in items]


def from_core_item(item: _core.ChatItem | dict[str, Any]) -> ChatItem:
    """A ``_core.ChatItem``, or the dict of its fields (how records carrying one emit it)."""
    fields = item if isinstance(item, dict) else item.fields()
    if fields["type"] == "message":
        fields["content"] = [_from_core_content(c) for c in fields["content"]]
    return _ITEM.validate_python(fields)


def _to_core_content(content: Any) -> _core.ChatContent:
    if isinstance(content, str):
        return _core.ChatContent.text(content)
    if isinstance(content, AudioContent):
        return _core.ChatContent.audio(frames=content.frame, transcript=content.transcript)
    assert isinstance(content, ImageContent)
    shape = {
        "id": content.id,
        "inference_width": content.inference_width,
        "inference_height": content.inference_height,
        "inference_detail": content.inference_detail,
        "mime_type": content.mime_type,
    }
    if isinstance(content.image, str):
        return _core.ChatContent.image_url(url=content.image, **shape)
    return _core.ChatContent.image_frame(frame=content.image, **shape)


def _from_core_content(content: _core.ChatContent) -> Any:
    return content.value
