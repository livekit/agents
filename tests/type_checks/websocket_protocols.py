"""Type checks for the socket bases, run by ``.github/check_types.py``; the host tests run these
fake protocols against loopback peers (``tests/host/test_websocket_bases.py``).

A protocol declares its provider's messages (``Inbound``, TypedDicts on one tag) and its commands
(``Outbound``); each hook is typed from them. Each ``type: ignore[code]`` marks an error the type
checker must report; strict mode reports an ignore that is no longer needed.
"""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Annotated, Literal, NotRequired, TypedDict, assert_never

from pydantic import Field

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, LanguageCode, llm, stt, tts, utils
from livekit.agents.llm import realtime as llm_realtime
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr

# STT: JSON messages, binary audio up, a goodbye the provider answers with `finished`.


class Transcript(TypedDict):
    # only the fields the plugin reads: the provider's others are ignored
    type: Literal["transcript"]
    text: str
    final: bool
    confidence: NotRequired[float]


class Finished(TypedDict):
    type: Literal["finished"]
    # open: the provider adds reasons; the plugin reads the ones it knows
    reason: NotRequired[str]


SttMessage = Annotated[Transcript | Finished, Field(discriminator="type")]


class Finalize(TypedDict):
    type: Literal["finalize"]


class CloseStream(TypedDict):
    type: Literal["close"]


SttCommand = Finalize | CloseStream


class LoopbackSttStream(stt.WebSocketStream[SttMessage, SttCommand]):
    frame_ms = 50

    def __init__(self, *, stt: LoopbackStt, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._url = stt.url

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        return stt.WebSocketTarget(self._url, params={"encoding": "linear16"})

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> bytes | SttCommand:
        match item:
            case rtc.AudioFrame():
                return bytes(item.data)
            case stt.Flush():
                return Finalize(type="finalize")
            case stt.EndOfAudio():
                return CloseStream(type="close")
            case _:
                assert_never(item)

    def on_message(self, message: Transcript | Finished) -> None:
        if message["type"] == "finished":
            if message.get("reason") == "error":
                raise APIError("the provider failed the session", kind="server")
            self.report(stt.SessionFinished())
        elif message["type"] == "transcript":
            self.report(
                stt.Transcript(
                    message["text"],
                    LanguageCode("en"),
                    final=message["final"],
                    confidence=message.get("confidence", 0.0),
                )
            )
        else:
            assert_never(message)


class LoopbackStt(stt.STT):
    def __init__(self, url: str) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self.url = url

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> LoopbackSttStream:
        return LoopbackSttStream(stt=self, conn_options=conn_options)


# A binary codec (AWS event-stream style): each frame is a 4-byte header length, JSON headers,
# then the payload. The protocol decodes frames itself; nothing goes through JSON validation.


class AudioEvent(TypedDict):
    type: Literal["audio_event"]
    payload: bytes


class TranscriptEvent(TypedDict):
    type: Literal["transcript_event"]
    text: str


class ExceptionEvent(TypedDict):
    type: Literal["exception"]
    message: str


CodecMessage = TranscriptEvent | ExceptionEvent


def event_frame(headers: dict[str, str], payload: bytes) -> bytes:
    encoded = json.dumps(headers).encode()
    return struct.pack(">I", len(encoded)) + encoded + payload


class CodecSttStream(stt.WebSocketStream[CodecMessage, bytes]):
    def __init__(self, *, stt: LoopbackStt, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._url = stt.url

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        # a fresh presigned URL per connection
        return stt.WebSocketTarget(self._url, params={"X-Amz-Signature": "fresh"})

    def decode(self, frame: bytes | str) -> CodecMessage:
        if not isinstance(frame, bytes):
            raise stt.protocol_error(self, "a text frame on a binary protocol")
        (length,) = struct.unpack(">I", frame[:4])
        headers = json.loads(frame[4 : 4 + length])
        payload = frame[4 + length :]
        if headers[":message-type"] == "exception":
            return ExceptionEvent(type="exception", message=payload.decode())
        return TranscriptEvent(type="transcript_event", text=payload.decode())

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> bytes | None:
        match item:
            case rtc.AudioFrame():
                return event_frame({":event-type": "AudioEvent"}, bytes(item.data))
            case stt.EndOfAudio():
                return event_frame({":event-type": "AudioEvent"}, b"")
            case stt.Flush():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: CodecMessage) -> None:
        if message["type"] == "exception":
            # the provider's own failure event
            raise APIError(
                "the provider failed the stream", kind="server", detail=message["message"]
            )
        self.report(stt.Transcript(message["text"], LanguageCode("en"), final=True))


# TTS: text up, binary audio down; each segment is a context, and several streams share one
# socket by those contexts.


class Done(TypedDict):
    type: Literal["done"]
    context_id: str


TtsMessage = Done | bytes


class Speak(TypedDict):
    context_id: str
    text: str
    final: bool


class Cancel(TypedDict):
    context_id: str
    cancel: bool


TtsCommand = Speak | Cancel


class LoopbackMux(tts.WebSocketMultiplexer[TtsMessage, TtsCommand]):
    max_contexts = 4

    def __init__(self, url: str) -> None:
        self.url = url

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        return tts.WebSocketTarget(self.url)

    def route(self, message: Done | bytes) -> str | None:
        if isinstance(message, bytes):
            # binary audio leads with its context: `<context>|<audio>`
            return message.split(b"|", 1)[0].decode()
        return message["context_id"]

    def close_context(self, context_id: str) -> TtsCommand:
        return Cancel(context_id=context_id, cancel=True)


class LoopbackTtsStream(tts.WebSocketStream[TtsMessage, TtsCommand]):
    def __init__(self, *, tts: LoopbackTts, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = tts.mux

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> TtsCommand | None:
        match item:
            case tts.SegmentText():
                return Speak(context_id=item.segment_id, text=item.chunk.text, final=False)
            case tts.SegmentEnd():
                return Speak(context_id=item.segment_id, text="", final=True)
            case tts.SegmentStart() | tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: Done | bytes) -> None:
        if isinstance(message, bytes):
            context, audio = message.split(b"|", 1)
            self.report(tts.SegmentAudio(context.decode(), audio))
        else:
            self.report(tts.SegmentDone(message["context_id"]))


class LoopbackTts(tts.TTS):
    def __init__(self, url: str) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True), sample_rate=16000, num_channels=1
        )
        self.mux = LoopbackMux(url)

    def _stream(self, *, conn_options: APIConnectOptions) -> LoopbackTtsStream:
        return LoopbackTtsStream(tts=self, conn_options=conn_options)


# Realtime: commands through `send`, a change applied by reconnecting.


class Said(TypedDict):
    type: Literal["said"]
    text: str


class Ping(TypedDict):
    type: Literal["ping"]


class ReplyDone(TypedDict):
    type: Literal["done"]


RealtimeMessage = Annotated[Said | Ping | ReplyDone, Field(discriminator="type")]


class Pong(TypedDict):
    type: Literal["pong"]


class Say(TypedDict):
    type: Literal["say"]
    text: str


RealtimeCommand = Pong | Say


class LoopbackRealtimeSession(llm.RealtimeWebSocketSession[RealtimeMessage, RealtimeCommand]):
    """A reply the provider ``said`` is one generation, its text in pieces, until it is ``done``."""

    def __init__(self, model: LoopbackRealtime) -> None:
        self.url = model.url
        self.heard: list[str] = []
        self.instructions = ""
        self._requested: list[asyncio.Future[llm.GenerationCreatedEvent]] = []
        self._reply: tuple[utils.aio.Chan[str], utils.aio.Chan[llm.MessageGeneration]] | None = None
        super().__init__(model, conn_options=model.conn_options)

    async def connect(self, conn_options: APIConnectOptions) -> llm_realtime.WebSocketTarget:
        return llm_realtime.WebSocketTarget(self.url, params={"instructions": self.instructions})

    def encode(self, item: rtc.AudioFrame) -> bytes:
        return bytes(item.data)

    def on_message(self, message: Said | Ping | ReplyDone) -> None:
        if message["type"] == "ping":
            self.send_now(Pong(type="pong"))
        elif message["type"] == "said":
            self.heard.append(message["text"])
            if self._reply is None:
                self._reply = self._generation()
            self._reply[0].send_nowait(message["text"])
        elif message["type"] == "done":
            if self._reply is not None:
                for chan in self._reply:
                    chan.close()
                self._reply = None
        else:
            assert_never(message)

    def _generation(self) -> tuple[utils.aio.Chan[str], utils.aio.Chan[llm.MessageGeneration]]:
        text, audio = utils.aio.Chan[str](), utils.aio.Chan[rtc.AudioFrame]()
        audio.send_nowait(rtc.AudioFrame(bytes(480), 24000, 1, 240))
        audio.close()
        modalities = asyncio.get_running_loop().create_future()
        modalities.set_result(["audio", "text"])
        messages = utils.aio.Chan[llm.MessageGeneration]()
        messages.send_nowait(
            llm.MessageGeneration(
                message_id=f"msg_{len(self.heard)}",
                text_stream=text,
                audio_stream=audio,
                modalities=modalities,
            )
        )
        functions = utils.aio.Chan[llm.FunctionCall]()
        functions.close()
        requested = self._requested.pop(0) if self._requested else None
        event = llm.GenerationCreatedEvent(
            message_stream=messages,
            function_stream=functions,
            user_initiated=requested is not None,
            response_id=f"resp_{len(self.heard)}",
        )
        self.emit("generation_created", event)
        if requested is not None and not requested.done():
            requested.set_result(event)
        return text, messages

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return llm.ChatContext.empty()

    @property
    def tools(self) -> llm.ToolContext:
        return llm.ToolContext.empty()

    async def update_instructions(self, instructions: str) -> None:
        self.instructions = instructions
        self.reconnect("instructions changed")

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        pass

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        pass

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        pass

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        self.send_now(Say(type="say", text=str(instructions or "")))
        self._requested.append(asyncio.get_running_loop().create_future())
        return self._requested[-1]

    def commit_audio(self) -> None:
        pass

    def clear_audio(self) -> None:
        pass

    def interrupt(self) -> None:
        pass

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        pass


class LoopbackRealtime(llm.RealtimeModel):
    def __init__(self, url: str, conn_options: APIConnectOptions) -> None:
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=False,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
            )
        )
        self.url = url
        self.conn_options = conn_options

    def session(self, *, turn_detection_disabled: bool = False) -> LoopbackRealtimeSession:
        return LoopbackRealtimeSession(self)


# What the type checker refuses.


class WrongMessageType(stt.WebSocketStream[SttMessage, SttCommand]):
    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        return stt.WebSocketTarget("wss://provider.test")

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> None:
        return None

    # a message the protocol never declared
    def on_message(self, message: Said) -> None:  # type: ignore[override]
        pass


class MissedItem(stt.WebSocketStream[SttMessage, SttCommand]):
    # an encode that does not take every input item (Flush, EndOfAudio)
    def encode(self, item: rtc.AudioFrame) -> bytes:  # type: ignore[override]
        return bytes(item.data)

    def on_message(self, message: Transcript | Finished) -> None:
        pass


def messages_are_typed(stream: LoopbackSttStream, realtime: LoopbackRealtimeSession) -> None:
    stream.send_now(Finalize(type="finalize"))
    stream.send_now({"type": "speak"})  # type: ignore[arg-type]
    realtime.send_now(Say(type="say", text="hi"))
    realtime.send_now(Finalize(type="finalize"))  # type: ignore[arg-type]
    stream.report(stt.Ready())
    stream.report(Finished(type="finished"))  # type: ignore[arg-type]
