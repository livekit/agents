# Copyright 2023 LiveKit, Inc.
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
import json
import os
import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

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

from .audio import normalize_ref_audio
from .log import logger
from .models import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VOICE,
    NUM_CHANNELS,
    TTSModels,
    TTSVoices,
)
from .version import __version__

USER_AGENT = f"livekit-plugins-voxcpm/{__version__}"


def _normalize_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def _speech_http_url(base_url: str) -> str:
    return f"{_normalize_base_url(base_url)}/audio/speech"


def _speech_ws_url(base_url: str) -> str:
    parsed = urlparse(_normalize_base_url(base_url))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, f"{parsed.path}/audio/speech/stream", "", "", ""))


def _provider_from_base_url(base_url: str) -> str:
    try:
        return urlparse(_normalize_base_url(base_url)).netloc or "vllm-omni"
    except Exception:
        return "vllm-omni"


@dataclass
class _TTSOptions:
    base_url: str
    model: TTSModels | str
    voice: TTSVoices | str
    sample_rate: int
    api_key: str | None
    ref_audio: str | None
    ref_text: str | None
    timeout: float
    split_granularity: str

    def auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def ws_headers(self) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def build_speech_payload(self, *, text: str, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": "pcm",
            "stream": stream,
        }
        if self.ref_audio:
            payload["ref_audio"] = self.ref_audio
        if self.ref_text:
            payload["ref_text"] = self.ref_text
        return payload

    def build_ws_session_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "type": "session.config",
            "model": self.model,
            "voice": self.voice,
            "response_format": "pcm",
            "stream_audio": True,
            "split_granularity": self.split_granularity,
        }
        if self.ref_audio:
            config["ref_audio"] = self.ref_audio
        if self.ref_text:
            config["ref_text"] = self.ref_text
        return config


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        model: TTSModels | str = DEFAULT_MODEL,
        voice: TTSVoices | str = DEFAULT_VOICE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        api_key: NotGivenOr[str | None] = NOT_GIVEN,
        ref_audio: NotGivenOr[str | Path | None] = NOT_GIVEN,
        ref_text: NotGivenOr[str | None] = NOT_GIVEN,
        timeout: float = 300.0,
        split_granularity: str = "sentence",
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of VoxCPM2 TTS backed by a vLLM-Omni server.

        Args:
            base_url: OpenAI-compatible base URL for the vLLM-Omni server.
                Defaults to ``VLLM_OMNI_URL`` or ``http://127.0.0.1:8800/v1``.
            model: Model id served by vLLM-Omni (default ``openbmb/VoxCPM2``).
            voice: Preset or uploaded voice name (default ``default``).
            sample_rate: Output PCM sample rate. VoxCPM2 emits 48 kHz PCM.
            api_key: Optional bearer token when the server enforces auth.
            ref_audio: Optional reference audio path or data URI for cloning.
            ref_text: Optional transcript for the reference audio.
            timeout: Request timeout in seconds.
            split_granularity: WebSocket text split mode (``sentence`` by default).
            http_session: Optional shared aiohttp session.
            tokenizer: Sentence tokenizer for streaming synthesis.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        resolved_base_url = (
            base_url
            if is_given(base_url)
            else os.environ.get("VLLM_OMNI_URL", "http://127.0.0.1:8800/v1")
        )
        resolved_model = (
            model if model != DEFAULT_MODEL else os.environ.get("VLLM_OMNI_MODEL", model)
        )
        resolved_voice = voice if voice != DEFAULT_VOICE else os.environ.get("VOXCPM_VOICE", voice)
        resolved_api_key = api_key if is_given(api_key) else os.environ.get("VLLM_API_KEY")

        ref_audio_value: str | None = None
        if is_given(ref_audio) and ref_audio is not None:
            ref_audio_value = normalize_ref_audio(ref_audio)

        self._opts = _TTSOptions(
            base_url=_normalize_base_url(resolved_base_url),
            model=resolved_model,
            voice=resolved_voice,
            sample_rate=sample_rate,
            api_key=resolved_api_key,
            ref_audio=ref_audio_value,
            ref_text=ref_text if is_given(ref_text) else None,
            timeout=timeout,
            split_granularity=split_granularity,
        )

        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return _provider_from_base_url(self._opts.base_url)

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        ref_audio: NotGivenOr[str | Path | None] = NOT_GIVEN,
        ref_text: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(ref_audio):
            self._opts.ref_audio = normalize_ref_audio(ref_audio) if ref_audio is not None else None
        if is_given(ref_text):
            self._opts.ref_text = ref_text

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text via vLLM-Omni ``POST /v1/audio/speech`` with PCM streaming."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload = self._opts.build_speech_payload(text=self._input_text, stream=True)
        timeout = aiohttp.ClientTimeout(
            total=self._opts.timeout,
            sock_connect=self._conn_options.timeout,
        )

        try:
            async with self._tts._ensure_session().post(
                _speech_http_url(self._opts.base_url),
                headers=self._opts.auth_headers(),
                json=payload,
                timeout=timeout,
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise APIStatusError(
                        message=f"vLLM-Omni TTS request failed: {body}",
                        status_code=resp.status,
                        request_id=resp.headers.get("x-request-id"),
                        body=body,
                    )

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )

                async for chunk, _ in resp.content.iter_chunks():
                    if chunk:
                        output_emitter.push(chunk)

                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIStatusError:
            raise
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            logger.exception("voxcpm chunked stream failed")
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming synthesis via vLLM-Omni ``/v1/audio/speech/stream`` WebSocket."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=request_id)

        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            ws = await asyncio.wait_for(
                self._tts._ensure_session().ws_connect(
                    _speech_ws_url(self._opts.base_url),
                    headers=self._opts.ws_headers(),
                    heartbeat=30.0,
                    max_msg_size=0,
                ),
                self._conn_options.timeout,
            )

            await ws.send_json(self._opts.build_ws_session_config())

            sent_stream = self._tts._sentence_tokenizer.stream()

            async def input_task() -> None:
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        sent_stream.flush()
                        continue
                    sent_stream.push_text(data)
                sent_stream.end_input()

            async def sentence_task() -> None:
                async for ev in sent_stream:
                    self._mark_started()
                    await ws.send_json({"type": "input.text", "text": ev.token})
                await ws.send_json({"type": "input.done"})

            async def recv_task() -> None:
                while True:
                    msg = await ws.receive(timeout=self._opts.timeout)
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        raise APIStatusError(
                            message="vLLM-Omni WebSocket closed unexpectedly",
                            status_code=ws.close_code or -1,
                            request_id=request_id,
                            body=None,
                        )

                    if msg.type == aiohttp.WSMsgType.BINARY:
                        if msg.data:
                            output_emitter.push(msg.data)
                        continue

                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue

                    data = json.loads(msg.data)
                    msg_type = data.get("type")
                    if msg_type == "session.done":
                        break
                    if msg_type == "error":
                        raise APIStatusError(
                            message=str(data.get("message") or data),
                            status_code=500,
                            request_id=request_id,
                            body=data,
                        )

            tasks = [
                asyncio.create_task(input_task()),
                asyncio.create_task(sentence_task()),
                asyncio.create_task(recv_task()),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await sent_stream.aclose()
                await utils.aio.gracefully_cancel(*tasks)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIStatusError:
            raise
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            logger.exception("voxcpm synthesize stream failed")
            raise APIConnectionError() from e
        finally:
            if ws is not None:
                await ws.close()
            output_emitter.end_segment()
