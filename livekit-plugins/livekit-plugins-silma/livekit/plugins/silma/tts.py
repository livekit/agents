# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import base64
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ._utils import (
    Float32Decoder,
    http_stream_url,
    is_auth_close_code,
    normalize_base_url,
    raise_for_status,
    raise_ws_error,
    split_text,
    websocket_url,
)
from .log import logger
from .models import (
    API_KEY_HEADER,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    MAX_TEXT_CHARACTERS,
    NUM_CHANNELS,
    SAMPLE_RATE,
    USER_AGENT,
    TTSModels,
    TTSVoices,
)

# How long a pooled WebSocket may sit unused before it is discarded rather than
# reused. The pool refreshes this on every acquisition, so it is an idle timeout
# and not a cap on connection age.
MAX_SESSION_DURATION = 240


class _ChunkProgress:
    """Tracks whether an utterance already produced audio.

    Held by reference so the caller can tell a failure that reached the user
    apart from one that did not, even when the receive loop raises.
    """

    __slots__ = ("pushed",)

    def __init__(self) -> None:
        self.pushed = False


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: TTSVoices | str
    creativity: float | None
    speed: float | None
    user_id: str | None
    custom_audio_id: str | None
    enable_server_pronunciation_overrides: bool
    api_key: str
    base_url: str

    def build_payload(self, text: str) -> dict[str, Any]:
        """Build the ``TTSRequest`` body for a single chunk of text."""
        payload: dict[str, Any] = {
            "model_id": self.model,
            "text": text,
            "voice_id": self.voice,
        }
        if self.creativity is not None:
            payload["creativity"] = self.creativity
        if self.speed is not None:
            payload["speed"] = self.speed
        if self.user_id is not None:
            payload["user_id"] = self.user_id
        if self.custom_audio_id is not None:
            payload["custom_audio_id"] = self.custom_audio_id
        if self.enable_server_pronunciation_overrides:
            payload["enable_server_pronunciation_overrides"] = True
        return payload


def _validate_options(
    *,
    model: str,
    voice: str,
    user_id: str | None,
    custom_audio_id: str | None,
) -> None:
    if not model or not model.strip():
        raise ValueError("model must be a non-empty string")
    if not voice or not voice.strip():
        raise ValueError("voice must be a non-empty string")
    if custom_audio_id is not None and not user_id:
        raise ValueError(
            "user_id is required when custom_audio_id is set; "
            "find your user id at https://app.silma.ai/api-keys"
        )


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = DEFAULT_MODEL,
        voice: TTSVoices | str = DEFAULT_VOICE,
        creativity: float | None = None,
        speed: float | None = None,
        user_id: str | None = None,
        custom_audio_id: str | None = None,
        enable_server_pronunciation_overrides: bool = False,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        text_pacing: tts.SentenceStreamPacer | bool = False,
        allow_insecure_base_url: bool = False,
    ) -> None:
        """Create a new instance of the SILMA TTS.

        See https://silma.ai/ for an overview and https://app.silma.ai for the
        playground, voice library and API keys.

        Args:
            model: The SILMA model id. ``silma-tts-v2-english`` for English,
                ``silma-tts-v2-msa`` for Modern Standard Arabic, or
                ``silma-tts-v2-ksa`` for the Saudi (Najdi) dialect.
            voice: The pre-defined voice id. English voices are ``james`` and
                ``emma``; Arabic voices are ``sarah``, ``salma``, ``salwa``,
                ``saja``, ``sultan``, ``salman``, ``sulaiman`` and ``salim``.
            creativity: Variance in speech prosody. Left to the server default
                when ``None``.
            speed: Speed of the generated speech. Left to the server default
                when ``None``.
            user_id: Your SILMA user id, required for pronunciation overrides
                and custom voices. Found at https://app.silma.ai/api-keys.
            custom_audio_id: The id of an uploaded custom voice to clone, e.g.
                ``voice_1769817467123`` from https://app.silma.ai/voices.
                Requires ``user_id``.
            enable_server_pronunciation_overrides: Apply the pronunciation
                overrides configured on your account at
                https://app.silma.ai/control.
            api_key: The SILMA API key. Falls back to the ``SILMA_API_KEY``
                environment variable.
            base_url: The API base URL. Defaults to
                ``https://api.silma.ai/tts/v2``.
            http_session: An existing ``aiohttp.ClientSession`` to use instead
                of creating one.
            tokenizer: The sentence tokenizer used to split streamed LLM text
                into complete utterances. Defaults to
                ``livekit.agents.tokenize.blingfire.SentenceTokenizer``.
            text_pacing: Stream pacer for the TTS. ``True`` uses the default
                pacer, ``False`` disables pacing.
            allow_insecure_base_url: Permit a plaintext ``http://`` base URL
                pointing at a non-local host. Only for trusted development
                networks.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        silma_api_key = api_key or os.environ.get("SILMA_API_KEY")
        if not silma_api_key:
            raise ValueError(
                "SILMA API key is required, either as the api_key argument or "
                "via the SILMA_API_KEY environment variable"
            )

        _validate_options(
            model=model,
            voice=voice,
            user_id=user_id,
            custom_audio_id=custom_audio_id,
        )

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            creativity=creativity,
            speed=speed,
            user_id=user_id,
            custom_audio_id=custom_audio_id,
            enable_server_pronunciation_overrides=enable_server_pronunciation_overrides,
            api_key=silma_api_key,
            base_url=normalize_base_url(base_url, allow_insecure_base_url=allow_insecure_base_url),
        )

        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=MAX_SESSION_DURATION,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )

        self._stream_pacer: tts.SentenceStreamPacer | None = None
        if text_pacing is True:
            self._stream_pacer = tts.SentenceStreamPacer()
        elif isinstance(text_pacing, tts.SentenceStreamPacer):
            self._stream_pacer = text_pacing

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "SILMA"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        try:
            return await asyncio.wait_for(
                session.ws_connect(
                    websocket_url(self._opts.base_url),
                    headers={
                        API_KEY_HEADER: self._opts.api_key,
                        "User-Agent": USER_AGENT,
                    },
                ),
                timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.WSServerHandshakeError as e:
            # Do not surface the exception message: RequestInfo can carry the
            # API key header.
            raise APIStatusError(
                "SILMA TTS WebSocket handshake failed",
                status_code=e.status,
                request_id=None,
                body=None,
                retryable=e.status not in (401, 403),
            ) from None
        except Exception as e:
            # Transport errors can embed credentials in URLs.
            raise APIConnectionError(type(e).__name__) from None

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def prewarm(self) -> None:
        self._pool.prewarm()

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        creativity: NotGivenOr[float | None] = NOT_GIVEN,
        speed: NotGivenOr[float | None] = NOT_GIVEN,
        user_id: NotGivenOr[str | None] = NOT_GIVEN,
        custom_audio_id: NotGivenOr[str | None] = NOT_GIVEN,
        enable_server_pronunciation_overrides: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """Update the synthesis options.

        Any option left unset keeps its current value. The update is atomic: if
        the resulting combination is invalid, nothing changes.

        Args:
            model: The SILMA model id.
            voice: The pre-defined voice id.
            creativity: Variance in speech prosody.
            speed: Speed of the generated speech.
            user_id: Your SILMA user id.
            custom_audio_id: The id of an uploaded custom voice to clone.
            enable_server_pronunciation_overrides: Apply account-level
                pronunciation overrides.
        """
        # Build a copy first so an invalid combination leaves the live options
        # untouched.
        updated = replace(self._opts)
        if is_given(model):
            updated.model = model
        if is_given(voice):
            updated.voice = voice
        if is_given(creativity):
            updated.creativity = creativity
        if is_given(speed):
            updated.speed = speed
        if is_given(user_id):
            updated.user_id = user_id
        if is_given(custom_audio_id):
            updated.custom_audio_id = custom_audio_id
        if is_given(enable_server_pronunciation_overrides):
            updated.enable_server_pronunciation_overrides = enable_server_pronunciation_overrides

        _validate_options(
            model=updated.model,
            voice=updated.voice,
            user_id=updated.user_id,
            custom_audio_id=updated.custom_audio_id,
        )
        self._opts = updated

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
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text over the binary waveform streaming endpoint."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        # The API caps `text` at MAX_TEXT_CHARACTERS, so long input is sent as
        # several sequential requests and concatenated into one segment.
        chunks = split_text(self._input_text, max_characters=MAX_TEXT_CHARACTERS)
        try:
            for chunk in chunks:
                await self._synthesize_chunk(chunk, output_emitter)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIStatusError:
            raise
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                "SILMA TTS request failed",
                status_code=e.status,
                request_id=None,
                body=None,
            ) from None
        except Exception as e:
            raise APIConnectionError() from e

        output_emitter.flush()

    async def _synthesize_chunk(self, text: str, output_emitter: tts.AudioEmitter) -> None:
        decoder = Float32Decoder()
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=self._conn_options.timeout,
            sock_read=self._conn_options.timeout,
        )

        async with self._tts._ensure_session().post(
            http_stream_url(self._opts.base_url),
            headers={
                API_KEY_HEADER: self._opts.api_key,
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            json=self._opts.build_payload(text),
            timeout=timeout,
        ) as resp:
            if resp.status != 200:
                raise_for_status(resp.status, await resp.text())

            content_type = resp.headers.get("Content-Type", "")
            if "json" in content_type.lower():
                # A 200 carrying JSON is an error envelope from a gateway in
                # front of the API, not audio.
                raise APIStatusError(
                    "SILMA TTS returned a non-audio response",
                    status_code=502,
                    request_id=None,
                    body={"content_type": content_type},
                )

            async for data, _ in resp.content.iter_chunks():
                pcm = decoder.decode(data)
                if pcm:
                    output_emitter.push(pcm)


class SynthesizeStream(tts.SynthesizeStream):
    """Stream synthesis over the full-duplex WebSocket endpoint."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=request_id)

        sentence_stream = self._tts._sentence_tokenizer.stream()
        if self._tts._stream_pacer:
            sentence_stream = self._tts._stream_pacer.wrap(
                sent_stream=sentence_stream,
                audio_emitter=output_emitter,
            )

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sentence_stream.flush()
                    continue

                sentence_stream.push_text(data)

            sentence_stream.end_input()

        async def _synthesis_task() -> None:
            async for ev in sentence_stream:
                # SILMA accepts complete utterances only, and caps them at
                # MAX_TEXT_CHARACTERS.
                for chunk in split_text(ev.token, max_characters=MAX_TEXT_CHARACTERS):
                    await self._synthesize_chunk(chunk, output_emitter)

            output_emitter.end_input()

        tasks = [
            asyncio.create_task(_input_task()),
            asyncio.create_task(_synthesis_task()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIStatusError:
            raise
        except APIConnectionError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await sentence_stream.aclose()
            await utils.aio.gracefully_cancel(*tasks)

    async def _synthesize_chunk(self, text: str, output_emitter: tts.AudioEmitter) -> None:
        """Send one utterance and drain its audio.

        A pooled connection may have been closed by the server since it was last
        used. When that shows up on a reused connection before any audio was
        emitted, reconnect once and try again — replaying is safe because
        nothing has reached the caller yet.
        """
        payload = json.dumps(self._opts.build_payload(text))

        for attempt in range(2):
            ws = await self._tts._pool.get(timeout=self._conn_options.timeout)
            reused = self._tts._pool.last_connection_reused
            self._acquire_time = self._tts._pool.last_acquire_time
            self._connection_reused = reused
            # Tracked by reference so it stays accurate when _receive_chunk
            # raises partway through an utterance.
            progress = _ChunkProgress()

            try:
                self._mark_started()
                await ws.send_str(payload)
                await self._receive_chunk(ws, output_emitter, progress)
            except asyncio.CancelledError:
                # Drop it, and let the cancellation propagate 
                # catching it alongside the retry logic below would swallow the
                # interruption and keep the agent talking.
                self._tts._pool.remove(ws)
                raise
            except Exception as e:
                self._tts._pool.remove(ws)
                # An error the server explicitly refused (a bad key, say) will
                # fail identically on a new connection.
                fatal = isinstance(e, APIStatusError) and not e.retryable
                if attempt == 0 and reused and not progress.pushed and not fatal:
                    logger.debug("SILMA TTS pooled connection was stale, reconnecting")
                    continue
                raise

            if ws.closed:
                self._tts._pool.remove(ws)
            else:
                self._tts._pool.put(ws)
            return

    async def _receive_chunk(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        output_emitter: tts.AudioEmitter,
        progress: _ChunkProgress,
    ) -> None:
        """Read one utterance's messages until ``completed``.

        Marks ``progress.pushed`` as soon as any audio reaches the emitter.
        """
        decoder = Float32Decoder()

        while True:
            msg = await ws.receive(timeout=self._conn_options.timeout)

            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                close_code = ws.close_code
                raise APIStatusError(
                    "SILMA TTS WebSocket closed before the utterance completed",
                    status_code=401 if is_auth_close_code(close_code) else (close_code or 1006),
                    request_id=None,
                    body=None,
                    retryable=not is_auth_close_code(close_code),
                )

            if msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError("SILMA TTS WebSocket error")

            if msg.type == aiohttp.WSMsgType.BINARY:
                # Some deployments push the waveform as binary frames rather
                # than base64 inside a JSON event.
                pcm = decoder.decode(msg.data)
                if pcm:
                    output_emitter.push(pcm)
                    progress.pushed = True
                continue

            if msg.type != aiohttp.WSMsgType.TEXT:
                logger.warning("unexpected SILMA TTS message type %s", msg.type)
                continue

            try:
                event = json.loads(msg.data)
            except ValueError:
                logger.warning("SILMA TTS sent a non-JSON text frame")
                continue

            if not isinstance(event, dict):
                continue

            audio = event.get("audio")
            if isinstance(audio, str) and audio:
                pcm = decoder.decode(base64.b64decode(audio))
                if pcm:
                    output_emitter.push(pcm)
                    progress.pushed = True

            status = event.get("status")
            if status == "failed":
                raise_ws_error(event)
            elif status == "completed":
                return
            elif status not in ("started", "streaming", None):
                logger.debug("unhandled SILMA TTS status %r", status)
