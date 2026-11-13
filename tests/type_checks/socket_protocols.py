"""Type checks for the socket bases, run by ``.github/check_types.py``; the host tests run these
fake protocols against loopback peers (``tests/host/test_socket_bases.py``).

A protocol declares its provider's messages (``Inbound``, TypedDicts on one tag) and its commands
(``Outbound``); each hook is typed from them. Each ``type: ignore[code]`` marks an error the type
checker must report; strict mode reports an ignore that is no longer needed.
"""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Annotated, Literal, NotRequired, TypedDict, assert_type

from pydantic import Field

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, LanguageCode, http, llm, stt, tts
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


SttMessage = Annotated[Transcript | Finished, Field(discriminator="type")]


class Finalize(TypedDict):
    type: Literal["finalize"]


class CloseStream(TypedDict):
    type: Literal["close"]


SttCommand = Finalize | CloseStream


class LoopbackSttStream(stt.SocketStream[SttMessage, SttCommand]):
    goodbye = CloseStream(type="close")
    frame_ms = 50

    def __init__(self, *, stt: LoopbackStt, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._url = stt.url

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        return http.SocketTarget(self._url, params={"encoding": "linear16"})

    def encode(
        self, item: rtc.AudioFrame | stt.RecognizeStream.Flush | http.EndOfInput
    ) -> http.Encoded[SttCommand]:
        if isinstance(item, rtc.AudioFrame):
            return bytes(item.data)
        if isinstance(item, stt.RecognizeStream.Flush):
            return Finalize(type="finalize")
        return None

    def on_message(self, message: Transcript | Finished) -> stt.SpeechEvent | http.Done:
        if message["type"] == "finished":
            return http.DONE
        kind = (
            stt.SpeechEventType.FINAL_TRANSCRIPT
            if message["final"]
            else stt.SpeechEventType.INTERIM_TRANSCRIPT
        )
        data = stt.SpeechData(
            language=LanguageCode("en"),
            text=message["text"],
            confidence=message.get("confidence", 0.0),
        )
        return stt.SpeechEvent(type=kind, alternatives=[data])


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


class CodecSttStream(stt.SocketStream[CodecMessage, bytes]):
    def __init__(self, *, stt: LoopbackStt, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._url = stt.url

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        # a fresh presigned URL per connection
        return http.SocketTarget(self._url, params={"X-Amz-Signature": "fresh"})

    def decode(self, frame: bytes | str) -> CodecMessage:
        if not isinstance(frame, bytes):
            raise http.protocol_error(self, "a text frame on a binary protocol")
        (length,) = struct.unpack(">I", frame[:4])
        headers = json.loads(frame[4 : 4 + length])
        payload = frame[4 + length :]
        if headers[":message-type"] == "exception":
            return ExceptionEvent(type="exception", message=payload.decode())
        return TranscriptEvent(type="transcript_event", text=payload.decode())

    def encode(
        self, item: rtc.AudioFrame | stt.RecognizeStream.Flush | http.EndOfInput
    ) -> http.Encoded[bytes]:
        if isinstance(item, rtc.AudioFrame):
            return event_frame({":event-type": "AudioEvent"}, bytes(item.data))
        if isinstance(item, http.EndOfInput):
            return event_frame({":event-type": "AudioEvent"}, b"")
        return None

    def on_message(self, message: CodecMessage) -> stt.SpeechEvent | http.Done:
        if message["type"] == "exception":
            # the provider's own failure event
            raise APIError(
                "the provider failed the stream", kind="server", detail=message["message"]
            )
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=LanguageCode("en"), text=message["text"])],
        )


# TTS: text up, binary audio down; several streams share one socket by context id.


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


class LoopbackMux(http.Multiplexer[TtsMessage, TtsCommand]):
    max_contexts = 4

    def __init__(self, url: str) -> None:
        self.url = url

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        return http.SocketTarget(self.url)

    def route(self, message: Done | bytes) -> str | None:
        if isinstance(message, bytes):
            # binary audio leads with its context: `<context>|<audio>`
            return message.split(b"|", 1)[0].decode()
        return message["context_id"]

    def close_context(self, context_id: str) -> http.Encoded[TtsCommand]:
        return Cancel(context_id=context_id, cancel=True)


class LoopbackTtsStream(tts.SocketStream[TtsMessage, TtsCommand]):
    def __init__(self, *, tts: LoopbackTts, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = tts.mux

    def encode(
        self, item: tts.SpeechChunk | tts.SynthesizeStream.Flush | http.EndOfInput
    ) -> http.Encoded[TtsCommand]:
        context_id = self.context_id or ""
        if isinstance(item, tts.SpeechChunk):
            return Speak(context_id=context_id, text=item.text, final=False)
        if isinstance(item, http.EndOfInput):
            return Speak(context_id=context_id, text="", final=True)
        return None

    def on_message(self, message: Done | bytes) -> bytes | http.Done:
        if isinstance(message, bytes):
            return message.split(b"|", 1)[1]
        return http.DONE


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


RealtimeMessage = Annotated[Said | Ping, Field(discriminator="type")]


class Pong(TypedDict):
    type: Literal["pong"]


class Say(TypedDict):
    type: Literal["say"]
    text: str


RealtimeCommand = Pong | Say


class LoopbackRealtimeSession(llm.RealtimeSocketSession[RealtimeMessage, RealtimeCommand]):
    def __init__(self, model: LoopbackRealtime) -> None:
        self.url = model.url
        self.heard: list[str] = []
        self.instructions = ""
        super().__init__(model, conn_options=model.conn_options)

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        return http.SocketTarget(self.url, params={"instructions": self.instructions})

    def encode(self, item: rtc.AudioFrame | http.EndOfInput) -> http.Encoded[RealtimeCommand]:
        return bytes(item.data) if isinstance(item, rtc.AudioFrame) else None

    def on_message(self, message: Said | Ping) -> None:
        if message["type"] == "ping":
            self.send_command(Pong(type="pong"))
        else:
            self.heard.append(message["text"])

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
        self.send_command(Say(type="say", text=str(instructions or "")))
        return asyncio.get_running_loop().create_future()

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


class WrongMessageType(stt.SocketStream[SttMessage, SttCommand]):
    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        return http.SocketTarget("wss://provider.test")

    def encode(
        self, item: rtc.AudioFrame | stt.RecognizeStream.Flush | http.EndOfInput
    ) -> http.Encoded[SttCommand]:
        return None

    # a message the protocol never declared
    def on_message(self, message: Said) -> None:  # type: ignore[override]
        pass


class MissedItem(stt.SocketStream[SttMessage, SttCommand]):
    # an encode that does not take every input item (Flush, EndOfInput)
    def encode(self, item: rtc.AudioFrame) -> http.Encoded[SttCommand]:  # type: ignore[override]
        return bytes(item.data)

    def on_message(self, message: Transcript | Finished) -> None:
        pass


def commands_are_typed(stream: LoopbackSttStream, realtime: LoopbackRealtimeSession) -> None:
    stream.send_command(Finalize(type="finalize"))
    stream.send_command({"type": "speak"})  # type: ignore[arg-type]
    realtime.send_command(Say(type="say", text="hi"))
    realtime.send_command(Finalize(type="finalize"))  # type: ignore[arg-type]
    assert_type(stream.on_message(Finished(type="finished")), stt.SpeechEvent | http.Done)
