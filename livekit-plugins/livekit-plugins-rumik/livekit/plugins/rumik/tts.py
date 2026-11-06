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
import json
import os
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    create_api_error_from_http,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .models import TTSModels

NUM_CHANNELS = 1
SAMPLE_RATE = 24000
RUMIK_BASE_URL = "https://silk-api.rumik.ai"


@dataclass
class _TTSOptions:
    model: TTSModels | str
    api_key: str
    description: str | None
    speaker: str | None
    f0_up_key: int | None
    temperature: float | None
    top_p: float | None
    top_k: int | None
    repetition_penalty: float | None
    max_new_tokens: int | None
    base_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = "muga",
        description: str | None = None,
        speaker: str | None = None,
        f0_up_key: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
        base_url: str = RUMIK_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of Rumik AI Silk TTS.

        Args:
            api_key: Your Rumik AI API key.
            model: The TTS model to use ("muga" or "mulberry").
            description: Mulberry only. Natural-language description to steer the voice.
            speaker: Mulberry only. Preset voice ("speaker_1"..."speaker_4").
            f0_up_key: Mulberry only. Pitch shift in semitones (-12...12).
            temperature: Sampling temperature.
            top_p: Nucleus sampling.
            top_k: Top-k sampling.
            repetition_penalty: Repetition penalty.
            max_new_tokens: Output length cap.
            base_url: Base URL for Rumik AI API.
            http_session: An existing ClientSession to use.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("RUMIK_API_KEY")
        if not api_key:
            raise ValueError(
                "Rumik AI API key is required, either as argument or set "
                "via the RUMIK_API_KEY environment variable"
            )

        self._opts = _TTSOptions(
            model=model,
            api_key=api_key,
            description=description,
            speaker=speaker,
            f0_up_key=f0_up_key,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            base_url=base_url,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "RumikAI"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        description: NotGivenOr[str | None] = NOT_GIVEN,
        speaker: NotGivenOr[str | None] = NOT_GIVEN,
        f0_up_key: NotGivenOr[int | None] = NOT_GIVEN,
        temperature: NotGivenOr[float | None] = NOT_GIVEN,
        top_p: NotGivenOr[float | None] = NOT_GIVEN,
        top_k: NotGivenOr[int | None] = NOT_GIVEN,
        repetition_penalty: NotGivenOr[float | None] = NOT_GIVEN,
        max_new_tokens: NotGivenOr[int | None] = NOT_GIVEN,
    ) -> None:
        """Update TTS options."""
        if is_given(model):
            self._opts.model = model
        if is_given(description):
            self._opts.description = description
        if is_given(speaker):
            self._opts.speaker = speaker
        if is_given(f0_up_key):
            self._opts.f0_up_key = f0_up_key
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(top_p):
            self._opts.top_p = top_p
        if is_given(top_k):
            self._opts.top_k = top_k
        if is_given(repetition_penalty):
            self._opts.repetition_penalty = repetition_penalty
        if is_given(max_new_tokens):
            self._opts.max_new_tokens = max_new_tokens

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
        return SynthesizeStream(tts=self, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    """HTTP-based synthesis — used when synthesize() is called directly."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            headers = {
                "Authorization": f"Bearer {self._opts.api_key}",
                "Content-Type": "application/json",
            }
            data: dict[str, Any] = {
                "model": self._opts.model,
                "text": self._input_text,
            }
            if self._opts.description is not None:
                data["description"] = self._opts.description
            if self._opts.speaker is not None:
                data["speaker"] = self._opts.speaker
            if self._opts.f0_up_key is not None:
                data["f0_up_key"] = self._opts.f0_up_key
            if self._opts.temperature is not None:
                data["temperature"] = self._opts.temperature
            if self._opts.top_p is not None:
                data["top_p"] = self._opts.top_p
            if self._opts.top_k is not None:
                data["top_k"] = self._opts.top_k
            if self._opts.repetition_penalty is not None:
                data["repetition_penalty"] = self._opts.repetition_penalty
            if self._opts.max_new_tokens is not None:
                data["max_new_tokens"] = self._opts.max_new_tokens

            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}/v1/tts",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise create_api_error_from_http(body, status=resp.status)

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/wav",
                )

                async for chunk, _ in resp.content.iter_chunks():
                    output_emitter.push(chunk)

                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise create_api_error_from_http(e.message, status=e.status) from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """WebSocket-based streaming synthesis — primary path used by the agent pipeline."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        try:
            text_buffer = ""
            async for data in self._input_ch:
                if isinstance(data, str):
                    text_buffer += data
                elif isinstance(data, self._FlushSentinel):
                    if text_buffer.strip():
                        await self._run_ws(text_buffer.strip(), output_emitter)
                    text_buffer = ""
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

    async def _run_ws(self, text: str, output_emitter: tts.AudioEmitter) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        # 1. Mint WebSocket Session Url
        headers = {"Authorization": f"Bearer {self._opts.api_key}"}
        req_body = {
            "model": self._opts.model,
            "text": text,
        }
        async with self._tts._ensure_session().post(
            f"{self._opts.base_url}/v1/tts/ws-connect",
            headers=headers,
            json=req_body,
            timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise create_api_error_from_http(body, status=resp.status)
            res_data = await resp.json()
            ws_url = res_data["ws_url"]
            token = res_data["token"]

        # 2. Connect and synthesize
        ws_endpoint = f"{ws_url}?token={token}"
        try:
            ws = await asyncio.wait_for(
                self._tts._ensure_session().ws_connect(
                    ws_endpoint,
                    timeout=aiohttp.ClientWSTimeout(
                        ws_receive=self._conn_options.timeout, ws_close=self._conn_options.timeout
                    ),
                ),
                timeout=self._conn_options.timeout,
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to Rumik AI WebSocket") from e

        try:
            self._mark_started()

            frame: dict[str, Any] = {
                "text": text,
            }
            if self._opts.description is not None:
                frame["description"] = self._opts.description
            if self._opts.speaker is not None:
                frame["speaker"] = self._opts.speaker
            if self._opts.f0_up_key is not None:
                frame["f0_up_key"] = self._opts.f0_up_key
            if self._opts.temperature is not None:
                frame["temperature"] = self._opts.temperature
            if self._opts.top_p is not None:
                frame["top_p"] = self._opts.top_p
            if self._opts.top_k is not None:
                frame["top_k"] = self._opts.top_k
            if self._opts.repetition_penalty is not None:
                frame["repetition_penalty"] = self._opts.repetition_penalty
            if self._opts.max_new_tokens is not None:
                frame["max_new_tokens"] = self._opts.max_new_tokens

            await ws.send_str(json.dumps(frame))

            # 3. Read Binary PCM frames
            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)

                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Rumik AI WebSocket closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=str(msg.data),
                    )

                if msg.type == aiohttp.WSMsgType.BINARY:
                    output_emitter.push(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(msg.data)
                    if event.get("type") == "done":
                        output_emitter.end_segment()
                        break
                    elif "error" in event:
                        raise APIConnectionError(
                            f"Rumik AI TTS error: {event.get('error', 'unknown error')}"
                        )
        finally:
            await ws.close()
