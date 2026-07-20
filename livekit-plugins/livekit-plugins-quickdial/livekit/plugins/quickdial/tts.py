# Copyright 2026 Samay AI (Quickdial)
# Licensed under the Apache License, Version 2.0
"""Quickdial TTS for LiveKit Agents.

Two paths, both emitting 16-bit PCM mono @ 24 kHz:

* ``ChunkedStream`` — one-shot ``POST /v1/tts`` (streaming WAV), used by ``synthesize()``.
* ``SynthesizeStream`` — realtime WebSocket ``/v1/tts/stream``, used by ``stream()``.

Auth is a Bearer API key (``qdl_live_…``). Get one at https://web.quickdial.ai.
"""
from __future__ import annotations

import asyncio
import json
import os
import weakref
from dataclasses import dataclass, replace

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import TTSVoices

DEFAULT_BASE_URL = "https://api.quickdial.ai"
DEFAULT_VOICE = "alba"
SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    voice: str
    sample_rate: int
    base_url: str
    api_key: str
    word_tokenizer: tokenize.WordTokenizer  # type: ignore  # noqa: F821
    params: dict | None

    @property
    def ws_url(self) -> str:
        return self.base_url.replace("http", "ws", 1) + "/v1/tts/stream"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: TTSVoices | str = DEFAULT_VOICE,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = DEFAULT_BASE_URL,
        sample_rate: int = SAMPLE_RATE,
        params: NotGivenOr[dict] = NOT_GIVEN,
        word_tokenizer: NotGivenOr["tokenize.WordTokenizer"] = NOT_GIVEN,  # type: ignore  # noqa: F821
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Quickdial TTS.

        Args:
            voice: voice name (see ``GET /v1/voices``). Defaults to ``alba``.
            api_key: Quickdial API key; falls back to ``QUICKDIAL_API_KEY``.
            base_url: API base, default ``https://api.quickdial.ai``.
            sample_rate: output sample rate (Quickdial streams 24 kHz).
            params: optional Pocket-TTS knobs (``temperature``, ``speed``, …).
            word_tokenizer: sentence/word tokenizer for the streaming path.
        """
        # Non-streaming: the AgentSession synthesizes the whole reply in one
        # POST /v1/tts, giving continuous audio (no per-sentence generation gaps
        # that make WS-per-flush synthesis sound choppy). The WS SynthesizeStream
        # remains available for advanced/low-latency use.
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        key = api_key if is_given(api_key) else os.environ.get("QUICKDIAL_API_KEY", "")
        if not key:
            raise ValueError(
                "Quickdial API key required — pass api_key= or set QUICKDIAL_API_KEY"
            )
        from livekit.agents import tokenize  # local import to avoid hard dep at top

        self._opts = _TTSOptions(
            voice=voice,
            sample_rate=sample_rate,
            base_url=base_url.rstrip("/"),
            api_key=key,
            word_tokenizer=word_tokenizer
            if is_given(word_tokenizer)
            else tokenize.basic.WordTokenizer(ignore_punctuation=False),
            params=params if is_given(params) else None,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def provider(self) -> str:
        return "Quickdial"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self, *, voice: NotGivenOr[str] = NOT_GIVEN, params: NotGivenOr[dict] = NOT_GIVEN
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(params):
            self._opts.params = params
        for s in self._streams:
            s.update_options(voice=voice, params=params)

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream


class ChunkedStream(tts.ChunkedStream):
    """One-shot synthesis over ``POST /v1/tts`` (streaming WAV)."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        body: dict = {"text": self._input_text, "voice": self._opts.voice, "format": "wav"}
        if self._opts.params:
            body["params"] = self._opts.params
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}/v1/tts",
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/wav",
                )
                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)
                output_emitter.flush()
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(message=e.message, status_code=e.status) from e
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Realtime synthesis over the WebSocket ``/v1/tts/stream``."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    def update_options(
        self, *, voice: NotGivenOr[str] = NOT_GIVEN, params: NotGivenOr[dict] = NOT_GIVEN
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(params):
            self._opts.params = params

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        url = f"{self._opts.ws_url}?key={self._opts.api_key}"

        async def _send(ws: aiohttp.ClientWebSocketResponse) -> None:
            # Aggregate text tokens between flush markers and synthesize ONCE per
            # flush — LiveKit expects one output segment per flush, so we must not
            # send a request per token (that yields "segments mismatch").
            pending: list[str] = []

            async def _flush() -> None:
                text = "".join(pending).strip()
                pending.clear()
                if not text:
                    return
                msg: dict = {"text": text, "voice": self._opts.voice}
                if self._opts.params:
                    msg["params"] = self._opts.params
                self._mark_started()
                await ws.send_str(json.dumps(msg))

            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    await _flush()
                else:
                    pending.append(data if isinstance(data, str) else str(data))
            await _flush()  # trailing text at end of input

        segment_open = False

        async def _recv(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal segment_open
            while True:
                frame = await ws.receive()
                if frame.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break
                if frame.type == aiohttp.WSMsgType.BINARY:
                    if not segment_open:  # streaming emitter needs an open segment
                        output_emitter.start_segment(segment_id=utils.shortuuid())
                        segment_open = True
                    output_emitter.push(frame.data)  # 16-bit PCM @ 24 kHz
                elif frame.type == aiohttp.WSMsgType.TEXT:
                    evt = json.loads(frame.data)
                    if evt.get("type") == "end":
                        if segment_open:
                            output_emitter.end_segment()
                            segment_open = False
                    elif evt.get("type") == "error":
                        raise APIStatusError(message=evt.get("message", "tts error"))

        try:
            async with self._tts._ensure_session().ws_connect(url) as ws:
                await asyncio.gather(_send(ws), _recv(ws))
                if segment_open:
                    output_emitter.end_segment()
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(message=e.message, status_code=e.status) from e
        except Exception as e:
            raise APIConnectionError() from e
