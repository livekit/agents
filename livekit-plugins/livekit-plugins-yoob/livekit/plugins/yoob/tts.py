from __future__ import annotations

import asyncio
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import DEFAULT_BASE_URL, DEFAULT_VOICE, TTS_SAMPLE_RATE, TTSModels, TTSQuality


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: str
    quality: TTSQuality | str
    instructions: str | None
    speed: float | None
    seed: int | str | None
    base_url: str
    api_key: str

    def body(self) -> dict[str, Any]:
        opts = {
            "model": self.model,
            "voice": self.voice,
            "quality": self.quality,
            "instructions": self.instructions,
            "speed": self.speed,
            "seed": self.seed,
            "response_format": "pcm",
        }
        return {k: v for k, v in opts.items() if v is not None}

    def ws_url(self) -> str:
        return self.base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + (
            "/audio/speech/stream"
        )


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        model: TTSModels | str = "yoob-tts-1",
        quality: TTSQuality | str = "best",
        instructions: str | None = None,
        speed: float | None = None,
        seed: int | str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Yoob TTS (Overtone).

        Args:
            voice: A voice id from ``GET /v1/voices`` (built-in or one of your cloned voices).
            model: ``yoob-tts-1``.
            quality: ``best`` (default) or ``fast`` (quicker first sound).
            instructions: Free-text delivery, e.g. ``"warm voice, softly spoken"``.
                Inline tags such as ``[laughing]`` or ``[whisper] ... [/whisper]`` in the text work too.
            speed: 0.5 to 2.0.
            seed: Makes renders repeatable.
            api_key: Defaults to the ``YOOB_API_KEY`` environment variable. Keys come from https://app.yoob.com.
            base_url: Defaults to ``YOOB_BASE_URL`` or https://tts-api.yoob.com/v1.
            http_session: An existing aiohttp session to reuse.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=1,
        )
        key = api_key or os.environ.get("YOOB_API_KEY")
        if not key:
            raise ValueError("Yoob API key is required: pass api_key or set YOOB_API_KEY")
        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            quality=quality,
            instructions=instructions,
            speed=speed,
            seed=seed,
            base_url=(base_url or os.environ.get("YOOB_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            api_key=key,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Yoob"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._opts.api_key}"}

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        quality: NotGivenOr[TTSQuality | str] = NOT_GIVEN,
        instructions: NotGivenOr[str | None] = NOT_GIVEN,
        speed: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(quality):
            self._opts.quality = quality
        if is_given(instructions):
            self._opts.instructions = instructions
        if is_given(speed):
            self._opts.speed = speed

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

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


def _api_error(status: int, body: str, request_id: str | None) -> APIStatusError:
    # Only the API's error message and code; the raw body is not kept, so retry logs never carry the input text.
    code = None
    message = f"Yoob speech request failed with HTTP {status}"
    try:
        err = json.loads(body).get("error") or {}
        code = err.get("code")
        message = err.get("message") or code or message
    except (ValueError, AttributeError):
        pass
    return APIStatusError(
        message, status_code=status, request_id=request_id, body={"code": code} if code else None
    )


class ChunkedStream(tts.ChunkedStream):
    """One request: POST /audio/speech with stream=true, raw 48 kHz PCM back as it renders."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}/audio/speech",
                headers=self._tts._headers(),
                json={**self._opts.body(), "input": self._input_text, "stream": True},
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=self._conn_options.timeout,
                    sock_read=self._conn_options.timeout,
                ),
            ) as resp:
                request_id = resp.headers.get("x-request-id") or utils.shortuuid()
                if resp.status >= 400:
                    raise _api_error(resp.status, await resp.text(), request_id)
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=TTS_SAMPLE_RATE,
                    num_channels=1,
                    mime_type="audio/pcm",
                )
                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)
                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Text pushed as it is written; one Yoob speech socket per segment (between flushes)."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        segments_ch = utils.aio.Chan[utils.aio.Chan[str]]()

        async def _split_segments() -> None:
            current: utils.aio.Chan[str] | None = None
            async for item in self._input_ch:
                if isinstance(item, str):
                    if not item:
                        continue
                    if current is None:
                        current = utils.aio.Chan[str]()
                        segments_ch.send_nowait(current)
                    current.send_nowait(item)
                elif isinstance(item, self._FlushSentinel):
                    if current is not None:
                        current.close()
                    current = None
            if current is not None:
                current.close()
            segments_ch.close()

        async def _run_segments() -> None:
            async for text_ch in segments_ch:
                await self._run_ws(text_ch, output_emitter)

        tasks = [asyncio.create_task(_split_segments()), asyncio.create_task(_run_segments())]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(e.message, status_code=e.status, request_id=request_id) from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(self, text_ch: utils.aio.Chan[str], output_emitter: tts.AudioEmitter) -> None:
        session = self._tts._ensure_session()
        ws = await asyncio.wait_for(
            session.ws_connect(self._opts.ws_url(), headers=self._tts._headers(), max_msg_size=0),
            self._conn_options.timeout,
        )
        try:
            await ws.send_str(json.dumps(self._opts.body()))
            ready = await asyncio.wait_for(ws.receive(), self._conn_options.timeout)
            if ready.type != aiohttp.WSMsgType.TEXT:
                raise APIStatusError(
                    "Yoob speech socket closed before it was ready",
                    status_code=ws.close_code or -1,
                )
            ready_msg = json.loads(ready.data)
            if ready_msg.get("type") == "error":
                raise _socket_error(ready_msg)
            segment_id = ready_msg.get("request_id") or utils.shortuuid()
            output_emitter.start_segment(segment_id=segment_id)

            async def send_task() -> None:
                async for text in text_ch:
                    self._mark_started()
                    await ws.send_str(json.dumps({"text": text}))
                await ws.send_str(json.dumps({"end": True}))

            async def recv_task() -> None:
                while True:
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        output_emitter.push(msg.data)
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        kind = data.get("type")
                        if kind == "error":
                            raise _socket_error(data)
                        if kind == "done":
                            output_emitter.end_segment()
                            return
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        raise APIStatusError(
                            "Yoob speech socket closed unexpectedly",
                            status_code=ws.close_code or -1,
                        )

            tasks = [asyncio.create_task(send_task()), asyncio.create_task(recv_task())]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
        finally:
            await ws.close()


def _socket_error(msg: dict[str, Any]) -> APIStatusError:
    err = msg.get("error") if isinstance(msg.get("error"), dict) else msg
    code = err.get("code") or "error"
    status = {"invalid_api_key": 401, "insufficient_credits": 402, "busy": 429}.get(code, 400)
    logger.debug("yoob speech socket error", extra={"code": code})
    return APIStatusError(
        f"Yoob: {err.get('message') or code}",
        status_code=status,
        body={"code": code},
        retryable=status >= 429,
    )
