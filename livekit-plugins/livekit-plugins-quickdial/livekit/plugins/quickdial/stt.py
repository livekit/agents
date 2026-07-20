# Copyright 2026 Samay AI (Quickdial)
# Licensed under the Apache License, Version 2.0
"""Quickdial STT for LiveKit Agents.

* ``SpeechStream`` — realtime WebSocket ``/v1/stt/stream``: stream 16-bit PCM
  mono @ 16 kHz, send ``eos`` at end-of-utterance, receive a transcript.
* ``_recognize_impl`` — batch ``POST /v1/stt`` (multipart WAV).

Pair with a VAD (e.g. ``silero.VAD``) so end-of-utterance is detected; the plugin
emits a FINAL_TRANSCRIPT per utterance (no interim results yet).
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, replace

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .models import STTLanguages

DEFAULT_BASE_URL = "https://api.quickdial.ai"
SAMPLE_RATE = 16000
NUM_CHANNELS = 1


@dataclass
class _STTOptions:
    language: str
    sample_rate: int
    base_url: str
    api_key: str
    params: dict | None

    @property
    def ws_url(self) -> str:
        return self.base_url.replace("http", "ws", 1) + "/v1/stt/stream"


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: STTLanguages | str = "en",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = DEFAULT_BASE_URL,
        sample_rate: int = SAMPLE_RATE,
        params: NotGivenOr[dict] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        # Quickdial returns a transcript per utterance (no interim results), so we
        # run as a NON-streaming STT: the AgentSession's VAD segments speech and
        # calls _recognize_impl (POST /v1/stt) per utterance. This is the reliable
        # path; the WS SpeechStream remains available for advanced use.
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        key = api_key if is_given(api_key) else os.environ.get("QUICKDIAL_API_KEY", "")
        if not key:
            raise ValueError("Quickdial API key required — pass api_key= or set QUICKDIAL_API_KEY")
        self._opts = _STTOptions(
            language=language,
            sample_rate=sample_rate,
            base_url=base_url.rstrip("/"),
            api_key=key,
            params=params if is_given(params) else None,
        )
        self._session = http_session

    @property
    def provider(self) -> str:
        return "Quickdial"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        wav = rtc.combine_audio_frames(buffer).to_wav_bytes()
        form = aiohttp.FormData()
        form.add_field("audio", wav, filename="audio.wav", content_type="audio/wav")
        cfg = dict(self._opts.params or {})
        cfg["language"] = language if is_given(language) else self._opts.language
        form.add_field("params", json.dumps(cfg))
        try:
            async with self._ensure_session().post(
                f"{self._opts.base_url}/v1/stt",
                headers={"Authorization": f"Bearer {self._opts.api_key}"},
                data=form,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=conn_options.timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return _to_speech_event(data, self._opts.language)
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(message=e.message, status_code=e.status) from e
        except Exception as e:
            raise APIConnectionError() from e

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = replace(self._opts)
        if is_given(language):
            opts.language = language
        return SpeechStream(stt=self, opts=opts, conn_options=conn_options)


def _to_speech_event(data: dict, language: str) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                language=data.get("language", language),
                text=data.get("text", ""),
            )
        ],
    )


class SpeechStream(stt.SpeechStream):
    def __init__(self, *, stt: STT, opts: _STTOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._stt: STT = stt
        self._opts = opts

    async def _run(self) -> None:
        # Send the API key in the Authorization header (not the URL query string) so it
        # isn't captured in proxy/access logs. The Quickdial WS endpoint accepts a Bearer
        # header; the ?key= query param is only needed for browser clients that can't set
        # WS handshake headers.
        url = self._opts.ws_url
        ws_headers = {"Authorization": f"Bearer {self._opts.api_key}"}

        async def _send(ws: aiohttp.ClientWebSocketResponse) -> None:
            cfg = dict(self._opts.params or {})
            cfg["language"] = self._opts.language
            await ws.send_str(json.dumps({"params": cfg}))
            samples_50ms = self._opts.sample_rate // 20
            stream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=NUM_CHANNELS,
                samples_per_channel=samples_50ms,
            )
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    for f in stream.flush():
                        await ws.send_bytes(f.data.tobytes())
                    await ws.send_str(json.dumps({"type": "eos"}))
                    continue
                for f in stream.write(data.data.tobytes()):
                    await ws.send_bytes(f.data.tobytes())

        async def _recv(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                frame = await ws.receive()
                if frame.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break
                if frame.type != aiohttp.WSMsgType.TEXT:
                    continue
                evt = json.loads(frame.data)
                t = evt.get("type")
                if t == "transcript":
                    text = (evt.get("text") or "").strip()
                    if text:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                        )
                        self._event_ch.send_nowait(_to_speech_event(evt, self._opts.language))
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                        )
                elif t == "error":
                    raise APIStatusError(message=evt.get("message", "stt error"))

        try:
            async with self._stt._ensure_session().ws_connect(url, headers=ws_headers) as ws:
                await asyncio.gather(_send(ws), _recv(ws))
        except (APITimeoutError, APIStatusError, APIConnectionError):
            # already-typed API errors (e.g. a server "error" event) must propagate as-is
            # so non-retryable failures aren't relabeled as retryable connection errors.
            raise
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(message=e.message, status_code=e.status) from e
        except Exception as e:
            raise APIConnectionError() from e
