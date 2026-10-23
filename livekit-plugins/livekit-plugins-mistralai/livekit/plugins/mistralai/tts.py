from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import uuid
import weakref
from dataclasses import dataclass, replace
from typing import Any, Literal
from urllib.parse import urlencode

import aiohttp
import httpx  # mistralai raises httpx.TimeoutException from its httpx v1 client
import numpy as np

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
from mistralai.client import Mistral, models
from mistralai.client.errors import HTTPValidationError, SDKError
from mistralai.client.utils import get_security, get_security_from_env

from .log import logger
from .models import TTSModels, TTSVoices


def _f32le_to_s16le(data: bytes) -> bytes:
    n = len(data) // 4
    floats = struct.unpack(f"<{n}f", data)
    return struct.pack(f"<{n}h", *(max(-32768, min(32767, int(s * 32767))) for s in floats))


DEFAULT_MODEL: TTSModels = "voxtral-mini-tts-latest"
DEFAULT_VOICE: TTSVoices = "en_paul_neutral"

SAMPLE_RATE: int = 24000
NUM_CHANNELS: int = 1

RESPONSE_FORMAT = Literal["mp3", "wav", "pcm", "opus", "flac"]
DEFAULT_RESPONSE_FORMAT: RESPONSE_FORMAT = "mp3"


def _is_realtime(model: str) -> bool:
    return "realtime" in model


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: TTSVoices | str | None
    response_format: RESPONSE_FORMAT
    ref_audio: str | None


class TTS(tts.TTS):
    def __init__(
        self,
        client: Mistral | None = None,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        ref_audio: NotGivenOr[str] = NOT_GIVEN,
        response_format: NotGivenOr[RESPONSE_FORMAT] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of MistralAI TTS.

        Realtime models (e.g. ``voxtral-mini-tts-realtime-3``) stream text in and audio out
        over a WebSocket, so the TTS supports ``stream()``. Other models synthesize each
        sentence over HTTP.

        Args:
            client: Optional pre-configured MistralAI client instance.
            api_key: Your Mistral AI API key. If not provided, will use the MISTRAL_API_KEY environment variable.
            model: The Mistral AI model to use for text-to-speech, default is "voxtral-mini-tts-latest".
            voice: The voice ID to use for synthesis. Mutually exclusive with ``ref_audio``. Defaults to ``en_paul_neutral`` when neither is given.
            ref_audio: Base64-encoded audio sample (3–25 s) for zero-shot voice cloning. Mutually exclusive with ``voice``.
            response_format: The audio format of synthesized speech, between ``mp3``, ``wav``, ``pcm``, ``opus`` or ``flac``. Defaults to ``mp3``. Ignored by realtime models, which always stream PCM.
            http_session: Optional aiohttp session for the realtime WebSocket.
        """
        if is_given(voice) and is_given(ref_audio):
            raise ValueError("Only one of 'voice' or 'ref_audio' may be provided, not both")

        resolved_model = model if is_given(model) else DEFAULT_MODEL
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=_is_realtime(resolved_model)),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._opts = _TTSOptions(
            model=resolved_model,
            voice=voice if is_given(voice) else (None if is_given(ref_audio) else DEFAULT_VOICE),
            ref_audio=ref_audio if is_given(ref_audio) else None,
            response_format=response_format
            if is_given(response_format)
            else DEFAULT_RESPONSE_FORMAT,
        )

        mistral_api_key = api_key if is_given(api_key) else os.environ.get("MISTRAL_API_KEY")
        if not client and not mistral_api_key:
            raise ValueError("Mistral AI API key is required. Set MISTRAL_API_KEY or pass api_key")
        self._client = client or Mistral(api_key=mistral_api_key)
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            mark_refreshed_on_get=False,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "MistralAI"

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        ref_audio: NotGivenOr[str] = NOT_GIVEN,
        response_format: NotGivenOr[RESPONSE_FORMAT] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.

        Args:
            model: The model to use for text-to-speech. Clears ``ref_audio``.
            voice: The voice ID to use for synthesis.
            ref_audio: Base64-encoded audio sample for zero-shot voice cloning. Clears ``voice``.
            response_format: The audio format of synthesized speech, between ``mp3``, ``wav``, ``pcm``, ``opus`` or ``flac``. Defaults to ``mp3``.
        """
        if is_given(voice) and is_given(ref_audio):
            raise ValueError("Only one of 'voice' or 'ref_audio' may be provided, not both")
        session_before = (self._opts.model, self._opts.voice, self._opts.ref_audio)
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
            self._opts.ref_audio = None
        if is_given(ref_audio):
            self._opts.ref_audio = ref_audio
            self._opts.voice = None
        if is_given(response_format):
            self._opts.response_format = response_format
        if (self._opts.model, self._opts.voice, self._opts.ref_audio) != session_before:
            self._pool.invalidate()

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        if not self._capabilities.streaming:
            return super().stream(conn_options=conn_options)  # type: ignore[return-value]
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        if self._capabilities.streaming:
            self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _realtime_request(self) -> tuple[str, dict[str, str]]:
        cfg = self._client.sdk_configuration
        base_url, _ = cfg.get_server_details()
        security = cfg.security() if callable(cfg.security) else cfg.security
        headers, query = get_security(get_security_from_env(security, models.Security))
        client_headers = getattr(cfg.async_client, "headers", None)
        if client_headers:
            headers |= dict(client_headers)
        params = {k: v[-1] for k, v in query.items() if v}
        params["model"] = self._opts.model
        url = base_url.rstrip("/").replace("http", "ws", 1)
        return f"{url}/v1/audio/speech/realtime?{urlencode(params)}", headers

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        url, headers = self._realtime_request()
        session: dict[str, Any] = {"streaming_mode": "responsive", "response_format": "pcm"}
        if self._opts.ref_audio is not None:
            session["ref_audio"] = self._opts.ref_audio
        else:
            session["voice"] = self._opts.voice or DEFAULT_VOICE

        try:
            ws = await asyncio.wait_for(
                self._ensure_session().ws_connect(url, headers=headers), timeout
            )
        except asyncio.TimeoutError:
            raise APIConnectionError("failed to connect to Mistral AI") from None
        except aiohttp.ClientResponseError as e:
            # RequestInfo carries the Authorization header, don't chain or format it
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError(
                f"failed to connect to Mistral AI ({type(e).__name__})"
            ) from None

        try:
            await asyncio.wait_for(self._open_session(ws, session), timeout)
        except BaseException:
            await ws.close()
            raise
        return ws

    async def _open_session(
        self, ws: aiohttp.ClientWebSocketResponse, session: dict[str, Any]
    ) -> None:
        await _expect(ws, "session.created")
        await ws.send_str(json.dumps({"type": "session.update", "session": session}))
        await _expect(ws, "session.updated")

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()


class ChunkedStream(tts.ChunkedStream):
    """ChunkedStream for MistralAI TTS. Sends the full text to the API and
    returns the synthesized audio as a single chunk."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            output_emitter.initialize(
                request_id=str(uuid.uuid4()),
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                mime_type=f"audio/{self._opts.response_format}",
            )
            if self._opts.ref_audio is not None:
                stream = await self._tts._client.audio.speech.complete_async(
                    model=self._opts.model,
                    input=self.input_text,
                    ref_audio=self._opts.ref_audio,
                    response_format=self._opts.response_format,
                    timeout_ms=int(self._conn_options.timeout * 1000),
                    stream=True,
                )
            else:
                stream = await self._tts._client.audio.speech.complete_async(
                    model=self._opts.model,
                    input=self.input_text,
                    voice_id=self._opts.voice or DEFAULT_VOICE,
                    response_format=self._opts.response_format,
                    timeout_ms=int(self._conn_options.timeout * 1000),
                    stream=True,
                )
            async for ev in stream:
                if ev.event == "speech.audio.delta":
                    data = base64.b64decode(ev.data.audio_data)
                    if self._opts.response_format == "pcm":
                        data = _f32le_to_s16le(data)
                    output_emitter.push(data)
                elif ev.event == "speech.audio.done":
                    self._set_token_usage(
                        input_tokens=ev.data.usage.prompt_tokens,
                        output_tokens=ev.data.usage.completion_tokens,
                    )

            output_emitter.flush()

        except (asyncio.TimeoutError, httpx.TimeoutException) as e:
            raise APITimeoutError() from e
        except (SDKError, HTTPValidationError) as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.headers.get("x-request-id"),
                body=e.body,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streams text to a Voxtral realtime model and audio back over a pooled WebSocket."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            stream=True,
            mime_type="audio/pcm",
        )

        segments_ch = utils.aio.Chan[utils.aio.Chan[str]]()

        async def _split_input() -> None:
            segment: utils.aio.Chan[str] | None = None
            async for data in self._input_ch:
                if isinstance(data, str):
                    if segment is None:
                        segment = utils.aio.Chan[str]()
                        segments_ch.send_nowait(segment)
                    segment.send_nowait(data)
                elif segment is not None:
                    segment.close()
                    segment = None
            if segment is not None:
                segment.close()
            segments_ch.close()

        async def _run_segments() -> None:
            async for segment in segments_ch:
                await self._run_ws(segment, output_emitter)

        tasks = [
            asyncio.create_task(_split_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(self, segment: utils.aio.Chan[str], output_emitter: tts.AudioEmitter) -> None:
        # the server never answers a flush of whitespace-only text
        first_text = ""
        async for text in segment:
            first_text += text
            if first_text.strip():
                break
        else:
            return

        output_emitter.start_segment(segment_id=utils.shortuuid())
        flushed = asyncio.Event()

        async def _send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            self._mark_started()
            await ws.send_str(json.dumps({"type": "input_text.append", "text": first_text}))
            async for text in segment:
                await ws.send_str(json.dumps({"type": "input_text.append", "text": text}))
            await ws.send_str(json.dumps({"type": "input_text.flush"}))
            flushed.set()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            remainder = b""
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive(), self._conn_options.timeout)
                except asyncio.TimeoutError:
                    # the LLM may still be producing text, only time out once the segment is flushed
                    if flushed.is_set():
                        raise
                    continue

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Mistral AI connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )
                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Mistral AI message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                msg_type = data.get("type")
                if msg_type == "speech.audio.delta":
                    # float32 samples can be split across deltas
                    audio = remainder + base64.b64decode(data["audio_data"])
                    usable = len(audio) - len(audio) % 4
                    remainder = audio[usable:]
                    samples = np.frombuffer(audio[:usable], dtype="<f4")
                    output_emitter.push(
                        (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                    )
                elif msg_type == "speech.audio.done":
                    usage = data.get("usage") or {}
                    self._set_token_usage(
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                    )
                    output_emitter.end_segment()
                    return
                elif msg_type == "error":
                    raise _status_error(data)
                else:
                    logger.warning("unexpected Mistral AI message", extra={"lk.pii.data": data})

        async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
            self._acquire_time = self._tts._pool.last_acquire_time
            self._connection_reused = self._tts._pool.last_connection_reused
            tasks = [
                asyncio.create_task(_send_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)


async def _expect(ws: aiohttp.ClientWebSocketResponse, event_type: str) -> None:
    msg = await ws.receive()
    if msg.type != aiohttp.WSMsgType.TEXT:
        raise APIStatusError(
            f"Mistral AI closed the connection before {event_type}",
            status_code=ws.close_code or -1,
            body=f"{msg.data=} {msg.extra=}",
        )
    data = json.loads(msg.data)
    if data.get("type") == "error":
        raise _status_error(data)
    if data.get("type") != event_type:
        raise APIConnectionError(f"expected {event_type} from Mistral AI, got {data.get('type')}")


def _status_error(data: dict[str, Any]) -> APIStatusError:
    error = data.get("error") or {}
    return APIStatusError(
        str(error.get("message", "Mistral AI realtime TTS error")),
        status_code=-1,
        body=data,
    )
