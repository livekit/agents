"""
*   Telnyx STT API documentation:
    <https://developers.telnyx.com/docs/voice/programmable-voice/stt-standalone>.
"""

from __future__ import annotations

import asyncio
import json
import struct
import weakref
from dataclasses import dataclass
from typing import Literal

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .common import NUM_CHANNELS, SAMPLE_RATE, STT_ENDPOINT, SessionManager, get_api_key
from .log import logger

TranscriptionEngine = Literal["telnyx", "google", "deepgram", "azure"]


@dataclass
class _STTOptions:
    api_key: str
    language: str
    transcription_engine: TranscriptionEngine
    interim_results: bool
    base_url: str
    sample_rate: int


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: str = "en",
        transcription_engine: TranscriptionEngine = "telnyx",
        interim_results: bool = True,
        api_key: str | None = None,
        base_url: str = STT_ENDPOINT,
        sample_rate: int = SAMPLE_RATE,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=interim_results,
            )
        )

        self._opts = _STTOptions(
            api_key=get_api_key(api_key),
            language=language,
            transcription_engine=transcription_engine,
            interim_results=interim_results,
            base_url=base_url,
            sample_rate=sample_rate,
        )
        self._session_manager = SessionManager(http_session)
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.transcription_engine

    @property
    def provider(self) -> str:
        return "telnyx"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        resolved_language = language if is_given(language) else self._opts.language

        stream = self.stream(language=language, conn_options=conn_options)
        try:
            frames = buffer if isinstance(buffer, list) else [buffer]
            for frame in frames:
                stream.push_frame(frame)
            stream.end_input()

            final_text = ""
            async for event in stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    if event.alternatives:
                        final_text += event.alternatives[0].text

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        language=resolved_language,
                        text=final_text,
                    )
                ],
            )
        finally:
            await stream.aclose()

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        resolved_language = language if is_given(language) else self._opts.language
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            language=resolved_language,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._session_manager.close()


def _create_streaming_wav_header(sample_rate: int, num_channels: int) -> bytes:
    """Create a WAV header for streaming with maximum possible size."""
    bytes_per_sample = 2
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = 0x7FFFFFFF
    file_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        file_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        16,
        b"data",
        data_size,
    )
    return header


class SpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        language: str,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._opts.sample_rate)
        self._stt: STT = stt
        self._language = language
        self._speaking = False

    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            wav_header = _create_streaming_wav_header(self._stt._opts.sample_rate, NUM_CHANNELS)
            await ws.send_bytes(wav_header)

            samples_per_chunk = self._stt._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._stt._opts.sample_rate,
                num_channels=NUM_CHANNELS,
                samples_per_channel=samples_per_chunk,
            )

            async for data in self._input_ch:
                if isinstance(data, rtc.AudioFrame):
                    for frame in audio_bstream.write(data.data.tobytes()):
                        await ws.send_bytes(frame.data.tobytes())
                elif isinstance(data, self._FlushSentinel):
                    for frame in audio_bstream.flush():
                        await ws.send_bytes(frame.data.tobytes())

            for frame in audio_bstream.flush():
                await ws.send_bytes(frame.data.tobytes())

            await asyncio.sleep(1.0)
            closing_ws = True
            await ws.close()

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:
                        return
                    raise APIStatusError(message="Telnyx STT WebSocket closed unexpectedly")

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        logger.debug("Telnyx STT received: %s", data)
                        self._process_stream_event(data)
                    except Exception:
                        logger.exception("Failed to process Telnyx STT message")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("Telnyx STT WebSocket error: %s", ws.exception())

        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            ws = await self._connect_ws()
            tasks = [
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
        finally:
            if ws is not None:
                await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        opts = self._stt._opts
        params = {
            "transcription_engine": opts.transcription_engine,
            "language": self._language,
            "input_format": "wav",
        }
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{opts.base_url}?{query_string}"
        headers = {"Authorization": f"Bearer {opts.api_key}"}

        try:
            ws = await asyncio.wait_for(
                self._stt._session_manager.ensure_session().ws_connect(url, headers=headers),
                self._conn_options.timeout,
            )
            logger.debug("Established Telnyx STT WebSocket connection")
            return ws
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("Failed to connect to Telnyx STT") from e

    def _process_stream_event(self, data: dict) -> None:
        transcript = data.get("transcript", "")
        is_final = data.get("is_final", False)

        if not transcript:
            return

        if not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

        alternatives = [
            stt.SpeechData(
                language=self._language,
                text=transcript,
                confidence=data.get("confidence", 0.0),
            )
        ]

        if is_final:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=alternatives,
                )
            )
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
        else:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=alternatives,
                )
            )
