"""
Uplift TTS Plugin for LiveKit, this will soon be available as a python lib
"""

from __future__ import annotations

import base64
import json
import os
import weakref
from dataclasses import dataclass
from typing import Annotated, Literal, NotRequired, TypedDict, assert_never

from pydantic import Field, JsonValue, TypeAdapter

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    tokenize,
    tts,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

# Output format options
OutputFormat = Literal[
    "PCM_22050_16",
    "WAV_22050_16",
    "WAV_22050_32",
    "MP3_22050_32",
    "MP3_22050_64",
    "MP3_22050_128",
    "OGG_22050_16",
    "ULAW_8000_8",
]

# Default configuration
DEFAULT_BASE_URL = "wss://api.upliftai.org"
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_NUM_CHANNELS = 1
DEFAULT_VOICE_ID = "v_meklc281"
DEFAULT_OUTPUT_FORMAT: OutputFormat = "MP3_22050_32"
WEBSOCKET_NAMESPACE = "/text-to-speech/multi-stream"


def get_content_type_from_output_format(output_format: OutputFormat) -> str:
    """Get MIME type based on output format"""
    if output_format == "PCM_22050_16":
        return "audio/pcm"
    elif output_format == "WAV_22050_16":
        return "audio/wav"
    elif output_format == "WAV_22050_32":
        return "audio/wav"
    elif output_format.startswith("MP3"):
        return "audio/mpeg"
    elif output_format.startswith("OGG"):
        return "audio/ogg"
    elif output_format == "ULAW_8000_8":
        return "audio/pcmu"
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def _sample_rate(output_format: OutputFormat) -> int:
    """G.711 is sent at 8 kHz and decoded without resampling; every other format at 22.05 kHz."""
    return 8000 if output_format == "ULAW_8000_8" else DEFAULT_SAMPLE_RATE


@dataclass
class VoiceSettings:
    """Voice configuration settings"""

    voice_id: str = DEFAULT_VOICE_ID
    output_format: OutputFormat = DEFAULT_OUTPUT_FORMAT


@dataclass
class _TTSOptions:
    """Internal TTS options"""

    base_url: str
    api_key: str
    voice_settings: VoiceSettings
    num_channels: int
    phrase_replacement_config_id: str | None


class TTS(tts.TTS):
    """Uplift TTS implementation for LiveKit"""

    def __init__(
        self,
        *,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice_id: str = DEFAULT_VOICE_ID,
        output_format: OutputFormat = DEFAULT_OUTPUT_FORMAT,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        phrase_replacement_config_id: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Uplift TTS.

        Args:
            base_url: Base URL for TTS service. Defaults to wss://api.upliftai.org
            api_key: API key for authentication
            voice_id: Voice ID to use. Defaults to "17"
            output_format: Audio output format. Options:
                - 'PCM_22050_16': PCM format, 22.05kHz, 16-bit
                - 'WAV_22050_16': WAV format, 22.05kHz, 16-bit
                - 'WAV_22050_32': WAV format, 22.05kHz, 32-bit
                - 'MP3_22050_32': MP3 format, 22.05kHz, 32kbps (default)
                - 'MP3_22050_64': MP3 format, 22.05kHz, 64kbps
                - 'MP3_22050_128': MP3 format, 22.05kHz, 128kbps
                - 'OGG_22050_16': OGG format, 22.05kHz, 16-bit
                - 'ULAW_8000_8': μ-law format, 8kHz, 8-bit
            num_channels: Number of audio channels. Defaults to 1 (mono)
            phrase_replacement_config_id: Optional ID for phrase replacement configuration
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=False,
            ),
            sample_rate=_sample_rate(output_format),
            num_channels=num_channels,
        )

        # Get configuration from environment if not provided
        resolved_base_url: str = (
            base_url
            if is_given(base_url)
            else os.environ.get("UPLIFTAI_BASE_URL", DEFAULT_BASE_URL)
        )
        resolved_api_key: str | None = (
            api_key if is_given(api_key) else os.environ.get("UPLIFTAI_API_KEY")
        )

        if not resolved_api_key:
            raise ValueError(
                "API key is required, either as argument or set UPLIFTAI_API_KEY environment variable"
            )

        self._opts = _TTSOptions(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            voice_settings=VoiceSettings(voice_id=voice_id, output_format=output_format),
            num_channels=num_channels,
            phrase_replacement_config_id=phrase_replacement_config_id
            if is_given(phrase_replacement_config_id)
            else None,
        )

        self._streams = weakref.WeakSet[SynthesizeStream]()

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        output_format: NotGivenOr[OutputFormat] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Update TTS configuration options.

        Args:
            voice_id: New voice ID
            output_format: New output format (see __init__ for options)
            sentence_tokenizer: What splits the text into the chunks the TTS is sent.
            pacing: A ``SentenceStreamPacer``, or ``None`` to send each chunk as soon as it is
                split.
        """
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            if is_given(voice_id):
                self._opts.voice_settings.voice_id = voice_id
            if is_given(output_format):
                self._opts.voice_settings.output_format = output_format
                # G.711 is 8 kHz: the stream's audio follows the format
                self._sample_rate = _sample_rate(output_format)

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(chunk, conn_options=conn_options)

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, opts=self._opts, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Clean up resources"""
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()


# Uplift speaks Socket.IO over Engine.IO: each frame is an Engine.IO packet (a type digit, then
# its payload); a message packet carries a Socket.IO packet (a type digit, the namespace, then
# its JSON). The server pings, the client answers with a pong.
_EIO_PING = "2"
_EIO_PONG = "3"
_EIO_MESSAGE = "4"
_SIO_CONNECT = "0"
_SIO_DISCONNECT = "1"
_SIO_EVENT = "2"
_SIO_CONNECT_ERROR = "4"


class _Ping(TypedDict):
    type: Literal["ping"]


class _Connected(TypedDict):
    type: Literal["connected"]


class _Refused(TypedDict):
    type: Literal["refused"]
    data: object


class _Disconnected(TypedDict):
    type: Literal["disconnected"]


# Uplift's own messages, each declaring only the fields read


class _Audio(TypedDict):
    type: Literal["audio"]
    requestId: str
    audio: NotRequired[str | None]


class _AudioEnd(TypedDict):
    type: Literal["audio_end"]
    requestId: str


class _Error(TypedDict):
    type: Literal["error"]
    requestId: NotRequired[str]
    message: NotRequired[str]


_Message = Annotated[_Audio | _AudioEnd | _Error, Field(discriminator="type")]
_MESSAGE: TypeAdapter[_Audio | _AudioEnd | _Error] = TypeAdapter(_Message)
_Inbound = _Ping | _Connected | _Refused | _Disconnected | _Audio | _AudioEnd | _Error
_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class SynthesizeStream(tts.WebSocketStream[_Inbound, str]):
    """Each segment is one synthesis request, its id the segment's."""

    awaits_ready = True

    def __init__(self, *, tts: TTS, opts: _TTSOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self.mime_type = get_content_type_from_output_format(opts.voice_settings.output_format)
        self._text: dict[str, list[str]] = {}

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        url = self._opts.base_url.replace("http", "ws", 1).rstrip("/")
        return tts.WebSocketTarget(
            f"{url}/socket.io/", params={"EIO": "4", "transport": "websocket"}
        )

    def setup(self) -> str:
        return _event(_SIO_CONNECT, {"token": self._opts.api_key})

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> str | None:
        match item:
            case tts.SegmentText():
                self._text.setdefault(item.segment_id, []).append(item.chunk.text)
                return None
            case tts.SegmentEnd():
                request: dict[str, object] = {
                    "type": "synthesize",
                    "requestId": item.segment_id,
                    "text": " ".join(self._text.pop(item.segment_id, [])),
                    "voiceId": self._opts.voice_settings.voice_id,
                    "outputFormat": self._opts.voice_settings.output_format,
                }
                if self._opts.phrase_replacement_config_id:
                    request["phraseReplacementConfigId"] = self._opts.phrase_replacement_config_id
                return _event(_SIO_EVENT, ["synthesize", request])
            case tts.SegmentStart() | tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def decode(self, frame: bytes | str) -> _Inbound | None:
        if not isinstance(frame, str):
            raise tts.protocol_error(self, "a binary frame")
        if frame == _EIO_PING:
            return {"type": "ping"}
        prefix = f"{_EIO_MESSAGE}{{}}{WEBSOCKET_NAMESPACE},"
        packet = next((t for t in "0124" if frame.startswith(prefix.format(t))), None)
        if packet is None:
            return None  # the open packet, another namespace
        body = frame[len(prefix.format(packet)) :]
        if packet == _SIO_CONNECT:
            return {"type": "connected"}
        if packet == _SIO_DISCONNECT:
            return {"type": "disconnected"}
        try:
            data = _JSON.validate_json(body) if body else None
            if packet == _SIO_CONNECT_ERROR:
                return {"type": "refused", "data": data}
            # an event is its name, then its arguments: Uplift's messages ride on "message"
            if not isinstance(data, list) or not data or not isinstance(data[0], str):
                raise ValueError("an event without a name")
            if data[0] == "message":
                if len(data) < 2 or not isinstance(message := data[1], dict):
                    raise ValueError("a message event without its message")
                tag = message.get("type")
                if tag in ("audio", "audio_end", "error"):
                    return _MESSAGE.validate_python(message)
            else:
                tag = data[0]
        except ValueError as e:
            raise tts.protocol_error(self, f"{e}; frame: {frame}") from e
        # an event or message type the stream does not read (one Uplift added since): named once
        tts.log_skipped("Uplift", str(tag))
        return None

    def on_message(self, message: _Inbound) -> None:
        if message["type"] == "ping":
            self.send_now(_EIO_PONG)
        elif message["type"] == "connected":
            self.report(tts.Ready())
        elif message["type"] == "refused":
            raise APIError(
                "Uplift refused the connection", kind="invalid_request", body=message["data"]
            )
        elif message["type"] == "disconnected":
            raise APIConnectionError("Uplift disconnected before synthesis completed")
        elif message["type"] == "audio":
            if audio := message.get("audio"):
                self.report(tts.SegmentAudio(message["requestId"], base64.b64decode(audio)))
        elif message["type"] == "audio_end":
            self.report(tts.SegmentDone(message["requestId"]))
        elif message["type"] == "error":
            raise APIError(
                "Uplift rejected the synthesis request", kind="invalid_request", body=message
            )
        else:
            assert_never(message)


def _event(packet: str, data: object) -> str:
    return f"{_EIO_MESSAGE}{packet}{WEBSOCKET_NAMESPACE},{json.dumps(data)}"
