# Copyright 2025 LiveKit, Inc.
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

"""LiveKit TTS plugin for the Clevr Labs conversational speech model."""

from __future__ import annotations

import asyncio
import base64
import re
import time
from math import gcd

import httpx
import numpy as np
from scipy.signal import resample_poly

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from ._text import clean_text
from .log import logger

_DEFAULT_SERVER_URL = "https://api.theclevr.com"

# The Clevr Labs model always synthesizes at 24 kHz (and the pipeline encodes
# user audio at 24 kHz too), so the output rate is fixed, not configurable.
_OUTPUT_SAMPLE_RATE = 24000


class TTS(tts.TTS):
    """Streaming text-to-speech backed by the Clevr Labs conversational speech model.

    Audio is synthesized on the Clevr Labs servers and streamed back as PCM; no
    model runs locally. The plugin keeps a single voice consistent across a
    conversation as long as each user turn is fed back via ``add_user_turn``.

    Example::

        from livekit.plugins import clevrlabs

        tts = clevrlabs.TTS(api_key="clevr_...")

        # Wire into a LiveKit AgentSession:
        session = AgentSession(tts=tts, ...)

        # After each user turn, forward the audio so the voice stays consistent:
        tts.add_user_turn(text=transcript, audio=audio_np, sample_rate=48000)
    """

    def __init__(
        self,
        *,
        api_key: str,
        server_url: str = _DEFAULT_SERVER_URL,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        """Create a Clevr Labs TTS plugin.

        Args:
            api_key:      Clevr Labs API key (``clevr_...``), available at
                          https://theclevr.com.
            server_url:   Base URL of the Clevr Labs API. Defaults to the hosted
                          endpoint; override only to target a different deployment.
            conn_options: Connection and retry options applied to synthesis requests.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=_OUTPUT_SAMPLE_RATE,
            num_channels=1,
        )
        self._server_url = server_url.rstrip("/")
        self._conn_options = conn_options

        self._session_id: str | None = None
        self._session_started: bool = False
        self._session_lock = asyncio.Lock()
        self._pending_user_turn: dict | None = None

        self._http_client = httpx.AsyncClient(
            base_url=self._server_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(90.0, connect=10.0),
        )

    def add_user_turn(self, *, text: str, audio: np.ndarray, sample_rate: int) -> None:
        """Buffer a user turn to send with the next synthesis request.

        The server owns all conversation context in its KV cache. The client
        only needs to forward new user audio so the server can encode it.

        Args:
            text:        Transcript of what the user said.
            audio:       Float32 numpy array of the user's speech. Integer arrays
                         (e.g. int16) are accepted and normalised automatically.
            sample_rate: Sample rate of the audio array (e.g. 48000, 16000).
        """
        if not text.strip() or audio.size == 0:
            logger.debug(
                "add_user_turn: dropping empty turn (text=%r, audio_size=%d)",
                text,
                audio.size,
            )
            return

        if np.issubdtype(audio.dtype, np.integer):
            max_int = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float32) / max_int
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if sample_rate != 24000:
            g = gcd(sample_rate, 24000)
            audio = resample_poly(audio, 24000 // g, sample_rate // g).astype(np.float32)

        audio_b64 = base64.b64encode(audio.tobytes()).decode("ascii")
        self._pending_user_turn = {"speaker": 1, "text": text, "audio_b64": audio_b64}

    def start_session(self) -> None:
        """Eagerly start a session in the background.

        Call this right after construction (from an async context) so the
        session is ready before the first synthesis request. This is purely an
        optimization: if not called, the session is opened lazily on first use,
        and a failed eager start is simply retried on first use.
        """

        async def _eager_start() -> None:
            try:
                await self._ensure_session()
            except Exception:
                logger.warning(
                    "Eager Clevr session start failed; will retry on first synthesis",
                    exc_info=True,
                )

        asyncio.ensure_future(_eager_start())

    async def _ensure_session(self) -> None:
        if self._session_started:
            return

        # The lock serializes concurrent callers so only one session is opened.
        # A failed attempt leaves _session_started False, so the next call retries
        # instead of permanently caching the error.
        async with self._session_lock:
            if self._session_started:
                return
            resp = await self._http_client.post("/tts/session/start")
            resp.raise_for_status()
            data = resp.json()
            self._session_id = data["session_id"]
            self._session_started = True
            logger.info("Clevr session started: %s", self._session_id)

    async def _end_session(self) -> None:
        if not self._session_started or not self._session_id:
            return
        try:
            resp = await self._http_client.post(
                "/tts/session/end", params={"session_id": self._session_id}
            )
            resp.raise_for_status()
            logger.info("Clevr session ended: %s", self._session_id)
        except Exception:
            logger.warning("Failed to end Clevr session %s", self._session_id, exc_info=True)
        finally:
            self._session_started = False
            self._session_id = None

    async def aclose(self) -> None:
        await self._end_session()
        await self._http_client.aclose()

    @property
    def model(self) -> str:
        return "csm-1"

    @property
    def provider(self) -> str:
        return "clevr"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions | None = None
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(
            text=text, conn_options=conn_options or self._conn_options
        )

    def stream(self, *, conn_options: APIConnectOptions | None = None) -> ClevrLabsSynthesizeStream:
        return ClevrLabsSynthesizeStream(tts=self, conn_options=conn_options or self._conn_options)


class _SentenceBuffer:
    """Buffers streamed LLM tokens and emits chunks of N complete sentences."""

    def __init__(self, n: int = 3) -> None:
        self._n = n
        self._buf = ""
        self._pattern = re.compile(r"([.!?]+(?:\s+|\n+|\Z))")

    def push(self, text: str) -> list:
        self._buf += text
        chunks = []
        while True:
            matches = list(self._pattern.finditer(self._buf))
            if len(matches) < self._n:
                break
            split_idx = matches[self._n - 1].end()
            chunk = self._buf[:split_idx].strip()
            self._buf = self._buf[split_idx:]
            if chunk:
                chunks.append(chunk)
        return chunks

    def flush(self) -> str:
        text = self._buf.strip()
        self._buf = ""
        return text


class ClevrLabsSynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            stream=True,
        )
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._tts.sample_rate,
            num_channels=1,
        )
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        buf = _SentenceBuffer(n=3)
        async for data in self._input_ch:
            if isinstance(data, tts.SynthesizeStream._FlushSentinel):
                text = buf.flush()
                if text:
                    await self._synthesize_segment(text, output_emitter, audio_bstream)
            else:
                for chunk in buf.push(data):
                    await self._synthesize_segment(chunk, output_emitter, audio_bstream)

        text = buf.flush()
        if text:
            await self._synthesize_segment(text, output_emitter, audio_bstream)

        output_emitter.flush()
        output_emitter.end_segment()

    async def _synthesize_segment(
        self,
        text: str,
        output_emitter: tts.AudioEmitter,
        audio_bstream: utils.audio.AudioByteStream,
    ) -> None:
        text = clean_text(text.strip())
        if not text:
            return

        t_start = time.perf_counter()

        # Read but don't yet clear the pending user turn: if the request fails and
        # the base class retries _run(), the context must still be available.
        pending_user_turn = self._tts._pending_user_turn
        payload: dict = {
            "text": text,
            "speaker": 0,
        }
        if pending_user_turn is not None:
            payload["context"] = [pending_user_turn]

        audio_chunks: list = []
        byte_buffer = b""

        try:
            # Opening the session is inside the try so its HTTP errors are wrapped
            # as APIError too (e.g. a 401 from a bad key stays visible/retryable).
            await self._tts._ensure_session()
            payload["session_id"] = self._tts._session_id

            async with self._tts._http_client.stream(
                "POST", "/tts/synthesize/stream", json=payload
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                async for raw in resp.aiter_bytes():
                    byte_buffer += raw
                    complete = (len(byte_buffer) // 4) * 4
                    if complete == 0:
                        continue
                    chunk_np = np.frombuffer(byte_buffer[:complete], dtype=np.float32).copy()
                    byte_buffer = byte_buffer[complete:]

                    audio_chunks.append(chunk_np)
                    chunk_np = np.clip(chunk_np, -1.0, 1.0)
                    chunk_int16 = (chunk_np * 32767).astype(np.int16)
                    for frame in audio_bstream.write(chunk_int16.tobytes()):
                        output_emitter.push(frame.data.tobytes())

            for frame in audio_bstream.flush():
                output_emitter.push(frame.data.tobytes())

            # Request succeeded — only now drop the user context so it isn't
            # re-sent on the next segment (and is preserved for a retry on failure).
            if pending_user_turn is not None:
                self._tts._pending_user_turn = None

        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as e:
            raise APIStatusError(
                message=e.response.text,
                status_code=e.response.status_code,
                request_id=None,
                body=None,
            ) from None
        except httpx.TimeoutException as e:
            raise APITimeoutError() from e
        except httpx.HTTPError as e:
            raise APIConnectionError(
                f"Cannot connect to Clevr TTS server at {self._tts._server_url}. "
                "Check your server_url or visit https://theclevr.com for status."
            ) from e
        finally:
            total_ms = (time.perf_counter() - t_start) * 1000
            audio_np = (
                np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.float32)
            )
            duration_s = len(audio_np) / self._tts.sample_rate
            logger.info("[clevr] %.1fs audio in %.0fms", duration_s, total_ms)
