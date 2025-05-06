from __future__ import annotations

from livekit.agents.types import NOT_GIVEN

"""Minimal Azure Speech-to-Text implementation for LiveKit agents.

Relies only on `aiohttp` and core LiveKit utilities. Designed to be compact yet
fully functional: it supports streaming recognition with interim and final
results in a single language.

Environment variables required:
  * AZURE_SPEECH_KEY – subscription key.
  * AZURE_SPEECH_REGION – service region (e.g. "westeurope").

For advanced features (language auto-detect, phrase hints, etc.) extend the
`_build_context` method.
"""

import os
import uuid
import json
import datetime as _dt
import asyncio
from urllib.parse import urlencode

import aiohttp
import datetime

from livekit import rtc  # type: ignore
from livekit.agents import (
    stt,
    utils,
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)
from livekit.agents.utils import AudioBuffer

__all__ = ["AzureSTT"]

_WS_ENDPOINT = (
    "wss://{region}.stt.speech.microsoft.com/speech/recognition/{mode}/cognitiveservices/v1"
)


class STT(stt.STT):
    """STT provider that talks to Azure Speech over raw WebSockets."""

    def __init__(
        self,
        *,
        language: str = "en-US",
        interim_results: bool = True,
        sample_rate: int = 16000,
        format_: str = "detailed",
        region: str | None = None,
        key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=interim_results)
        )

        self._language = language
        self._sample_rate = sample_rate
        self._interim = interim_results
        self._result_format = format_
        self._region = region or os.getenv("AZURE_SPEECH_REGION")
        self._key = key or os.getenv("AZURE_SPEECH_KEY")
        if not (self._region and self._key):
            raise ValueError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION env vars are required")

        self._session = http_session or utils.http_context.http_session()
        self._streams: set["SpeechStream"] = set()

    def _recognize_impl(
        self,
        buffer: list[AudioFrame] | rtc.AudioFrame,
        *,
        language: str | NotGiven = ...,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        pass

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------
    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "SpeechStream":
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            language=self._language,
            region=self._region,
            key=self._key,
            format_=self._result_format,
            interim_results=self._interim,
            sample_rate=self._sample_rate,
            http_session=self._session,
        )
        self._streams.add(stream)
        return stream


# -------------------------------------------------------------------------
# Streaming worker
# -------------------------------------------------------------------------


class SpeechStream(stt.SpeechStream):
    _WAV_HEADER_SENT = object()

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        language: str,
        region: str,
        key: str,
        format_: str,
        interim_results: bool,
        sample_rate: int,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self._lang = language
        self._region = region
        self._key = key
        self._format = format_
        self._interim = interim_results
        self._sample_rate = sample_rate
        self._session = http_session
        self._connection_id = uuid.uuid4().hex.upper()
        self._request_id = uuid.uuid4().hex.upper()

    # -------------------------------------------------- helpers
    def _endpoint(self) -> str:
        params = {
            "language": self._lang,
            "format": self._format,
        }
        qs = urlencode(params)
        return _WS_ENDPOINT.format(region=self._region, mode="conversation") + f"?{qs}"

    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        print(self._endpoint())
        return await asyncio.wait_for(
            self._session.ws_connect(
                self._endpoint(),
                headers={
                    "Ocp-Apim-Subscription-Key": self._key,
                    "X-ConnectionId": self._connection_id,
                },
            ),
            self._conn_options.timeout,
        )

    # -------------------------------------------------- message builders
    @staticmethod
    def _header(path: str, req_id: str) -> str:
        ts = _dt.datetime.utcnow().isoformat() + "Z"
        return f"Path: {path}\r\nX-RequestId: {req_id}\r\nX-Timestamp: {ts}\r\n"

    def _speech_config(self) -> str:
        body = {
            "context": {
                "system": {"version": "1.0.0"},
            }
        }
        return (
            self._header("speech.config", self._request_id)
            + "Content-Type: application/json; charset=utf-8\r\n\r\n"
            + json.dumps(body)
        )

    def _speech_context(self) -> str:
        body = {
            # from https://github.com/microsoft/cognitive-services-speech-sdk-js/blob/68541424c991bd273e11ff87e6baf77b1d9a836b/src/common.speech/ServiceMessages/PhraseDetection/PhraseDetectionContext.ts#L29
            "phraseDetection": {
                "language": "en-US",
                "mode": "Conversation",
                "interimResults": self._interim,
                "initialSilenceTimeout": 0,
                "trailingSilenceTimeout": 0,
                "conversation": {
                    "segmentation": {
                        "mode": "Normal",
                    }
                },
            }
        }
        return (
            self._header("speech.context", self._request_id)
            + "Content-Type: application/json; charset=utf-8\r\n\r\n"
            + json.dumps(body)
        )

    # -------------------------------------------------- run loop
    async def _run(self) -> None:  # noqa: C901 – keep concise
        ws = await self._connect()
        print(ws)
        await ws.send_str(self._speech_config())
        await ws.send_str(self._speech_context())  # new

        # micro-helper for audio chunking (50 ms per chunk)
        samples_chunk = self._sample_rate // 20
        byte_stream = utils.audio.AudioByteStream(
            sample_rate=self._sample_rate,
            num_channels=1,
            samples_per_channel=samples_chunk,
        )

        import asyncio, datetime, struct, uuid

        _CRLF = b"\r\n"

        def _wav_header(num_samples: int, sample_rate: int = 16000) -> bytes:
            """Return a minimal 44-byte PCM WAV header (16-bit, mono)."""
            byte_rate = sample_rate * 2  # 16-bit mono
            block_align = 2
            data_bytes = num_samples * 2
            riff_size = 36 + data_bytes
            return (
                b"RIFF"
                + struct.pack("<I", riff_size)
                + b"WAVEfmt "
                + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, block_align, 16)
                + b"data"
                + struct.pack("<I", data_bytes)
            )

        def _build_audio_frame(req_id: str, pcm: bytes, *, first=False) -> bytes:
            """
            Binary WS frame for Azure STT:
            2-byte big-endian length of header block,
            ASCII headers ending in CRLFCRLF,
            raw PCM/WAV bytes.
            """
            ts = datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
            hdr_lines = [
                "Path: audio",
                f"X-RequestId: {req_id}",
                f"X-Timestamp: {ts}",
            ]
            if first:
                hdr_lines.append("Content-Type: audio/x-wav")
            hdr = _CRLF.join(l.encode("ascii") for l in hdr_lines) + _CRLF * 2
            return struct.pack(">H", len(hdr)) + hdr + pcm

        async def _sender():
            """Stream audio frames to Azure Speech over `ws`."""
            wav_header_sent = False
            req_id = self._request_id or uuid.uuid4().hex.upper()

            async for item in self._input_ch:
                if not isinstance(item, rtc.AudioFrame):
                    continue  # ignore flush sentinels
                for chunk in byte_stream.write(item.data.tobytes()):
                    pcm = chunk.data.tobytes()
                    if not wav_header_sent:  # first frame
                        frame_bytes = _build_audio_frame(
                            req_id,
                            _wav_header(len(pcm), self._sample_rate) + pcm,
                            first=True,
                        )
                        wav_header_sent = True
                    else:
                        frame_bytes = _build_audio_frame(req_id, pcm)
                    await ws.send_bytes(frame_bytes)  # aiohttp expects raw bytes only

            # signal end-of-stream with zero-length audio payload
            await ws.send_bytes(_build_audio_frame(req_id, b""))

        async def _receiver():
            while True:
                msg = await ws.receive()
                print(msg)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._process_server_msg(msg.data)
                else:
                    break  # connection closed

        await asyncio.gather(_sender(), _receiver())

    # -------------------------------------------------- result processing
    def _process_server_msg(self, text: str) -> None:
        # split headers & body
        if "\r\n\r\n" in text:
            headers, body = text.split("\r\n\r\n", 1)
        else:
            return  # malformed
        path = None
        for line in headers.split("\r\n"):
            if line.lower().startswith("path:"):
                path = line.split(":", 1)[1].strip()
                break
        if path == "speech.hypothesis":
            data = json.loads(body)
            alt = stt.SpeechData(language=self._lang, text=data.get("Text", ""))
            evt = stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=[alt],
            )
            self._event_ch.send_nowait(evt)
        elif path == "speech.phrase":
            data = json.loads(body)
            alt = stt.SpeechData(
                language=self._lang, text=data.get("DisplayText") or data.get("Text", "")
            )
            # evt_type = (
            #     stt.SpeechEventType.FINAL_TRANSCRIPT
            #     if data.get("RecognitionStatus") == "Success"
            #     else stt.SpeechEventType.RECOGNITION_FAILED
            # )
            # evt = stt.SpeechEvent(
            #     type=evt_type,
            #     request_id=self._request_id,
            #     alternatives=[alt],
            # )
            # self._event_ch.send_nowait(evt)
