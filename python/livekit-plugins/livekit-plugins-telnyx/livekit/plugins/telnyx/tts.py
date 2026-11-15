"""
*   Telnyx TTS API documentation:
    <https://developers.telnyx.com/docs/voice/programmable-voice/tts-standalone>.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
from typing import NotRequired, TypedDict, assert_never

from livekit.agents import APIConnectOptions, tts

from .common import NUM_CHANNELS, SAMPLE_RATE, TTS_ENDPOINT, get_api_key


@dataclass
class _TTSOptions:
    api_key: str
    voice: str
    base_url: str


# Telnyx's websocket: one segment a connection, its text then an empty text; the server sends
# the audio (MP3) and closes once it is spoken.


class _Audio(TypedDict):
    audio: NotRequired[str | None]


class _Text(TypedDict):
    text: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "Telnyx.NaturalHD.astra",
        api_key: str | None = None,
        base_url: str = TTS_ENDPOINT,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._opts = _TTSOptions(voice=voice, api_key=get_api_key(api_key), base_url=base_url)

    @property
    def model(self) -> str:
        return self._opts.voice

    @property
    def provider(self) -> str:
        return "telnyx"

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(chunk, conn_options=conn_options)

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        return SynthesizeStream(tts=self, opts=replace(self._opts), conn_options=conn_options)


class SynthesizeStream(tts.WebSocketStream[_Audio, _Text]):
    connection_per_segment = True
    mime_type = "audio/mp3"

    def __init__(self, *, tts: TTS, opts: _TTSOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._segment = ""

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        return tts.WebSocketTarget(
            self._opts.base_url,
            headers={"Authorization": f"Bearer {self._opts.api_key}"},
            params={"voice": self._opts.voice},
        )

    def setup(self) -> _Text:
        return _Text(text=" ")

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Text | None:
        match item:
            case tts.SegmentStart():
                self._segment = item.segment_id
                return None
            case tts.SegmentText():
                return _Text(text=item.chunk.text)
            case tts.SegmentEnd():
                # the end of the text: the server closes once it is spoken
                return _Text(text="")
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Audio) -> None:
        if audio := message.get("audio"):
            self.report(tts.SegmentAudio(self._segment, base64.b64decode(audio)))
