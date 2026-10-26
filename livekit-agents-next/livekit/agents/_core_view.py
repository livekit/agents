"""What the Rust core calls on a host object, and the values it crosses with.

A core proxy calls ``_core_<method>`` on the object it was handed. An object without that surface
(every plugin and io leg) is seen through ``duck(obj, cls)``: ``cls`` names the base class the
core expects, and the view here answers for it over the object's public API. Records the core
reads by attribute (``SpeechEvent``, ``ChatChunk``, ``SynthesizedAudio``, the capabilities) are
the plugin's own objects; chat items cross as ``_core.ChatItem``.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
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
from .types import NOT_GIVEN
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


class _Stt:
    def __init__(self, obj: stt.STT) -> None:
        self._obj = obj

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

    def _core_with_conn_options(self, conn: Any) -> None:
        return None

    async def _core_recognize(self, buffer: list[Any], language: str | None) -> stt.SpeechEvent:
        return await self._obj.recognize(buffer, language=language or NOT_GIVEN)

    async def _core_stream(
        self, language: str | None, start_time_offset: float | None
    ) -> _InputStream | None:
        if not self._obj.capabilities.streaming:
            return None
        stream = self._obj.stream(language=language or NOT_GIVEN)
        if start_time_offset is not None:
            stream.start_time_offset = start_time_offset
        return _InputStream(stream, stream.push_frame)


class _Tts:
    def __init__(self, obj: tts.TTS) -> None:
        self._obj = obj

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

    def _core_with_conn_options(self, conn: Any) -> None:
        return None

    async def _core_synthesize(self, text: str) -> tts.ChunkedStream:
        return self._obj.synthesize(text)

    async def _core_stream(self) -> _InputStream:
        if not self._obj.capabilities.streaming:
            raise RuntimeError(f"{self._obj.label} has no duplex path")
        stream = self._obj.stream()
        return _InputStream(stream, stream.push_text)


class _Vad:
    def __init__(self, obj: vad.VAD) -> None:
        self._obj = obj

    def _core_label(self) -> str:
        return f"{type(self._obj).__module__}.{type(self._obj).__name__}"

    def _core_capabilities(self) -> vad.VADCapabilities:
        return self._obj.capabilities

    def _core_id(self) -> dict[str, str]:
        return {
            "label": self._core_label(),
            "model": self._obj.model,
            "provider": self._obj.provider,
        }

    def _core_prewarm(self) -> None:
        pass

    async def _core_stream(self) -> _InputStream:
        stream = self._obj.stream()
        return _InputStream(stream, stream.push_frame)


class _InputStream:
    """A plugin stream as the core drives it: ``_core_push`` in, ``__anext__`` out."""

    def __init__(self, stream: Any, push: Any) -> None:
        self._stream = stream
        self._core_push = push

    def flush(self) -> None:
        self._stream.flush()

    def end_input(self) -> None:
        self._stream.end_input()

    async def __anext__(self) -> Any:
        return await self._stream.__anext__()

    async def aclose(self) -> None:
        await self._stream.aclose()


class _Llm:
    def __init__(self, obj: llm.LLM) -> None:
        self._obj = obj

    def _core_label(self) -> str:
        return self._obj.label

    def _core_model(self) -> str:
        return self._obj.model

    def _core_id(self) -> dict[str, str]:
        return _identity(self._obj)

    def _core_prewarm(self) -> None:
        self._obj.prewarm()

    def _core_with_conn_options(self, conn: Any) -> None:
        return None

    async def _core_chat(self, req: _core.ChatRequest) -> llm.LLMStream:
        if req.chat_ctx_host is not None:
            chat_ctx = _core.resolve_host_ref(req.chat_ctx_host)
        else:
            chat_ctx = llm.ChatContext.from_dict(req.chat_ctx)
        pools = [req.tools_host, *req.tools_host_extra]
        tools = offered_tools(
            [t for key in pools if key is not None for t in _core.resolve_host_ref(key)],
            req.tool_names,
        )
        choice = req.tool_choice
        return self._obj.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            tool_choice=NOT_GIVEN if choice is None else choice,
            parallel_tool_calls=(
                NOT_GIVEN if req.parallel_tool_calls is None else req.parallel_tool_calls
            ),
            extra_kwargs=req.extra or NOT_GIVEN,
        )


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
