from __future__ import annotations

import asyncio
import base64
import uuid
from dataclasses import dataclass
from typing import cast

import httpx
import numpy as np
import openai
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_API_KEY = "dummy"
DEFAULT_VOICE = "UK female"
DEFAULT_SYSTEM_PROMPT = "Perform TTS. Use the UK female voice."


@dataclass
class _TTSOptions:
    voice: str
    system_prompt: str


class TTS(tts.TTS):
    """Text-to-Speech using LiquidAI LFM2.5-Audio model."""

    def __init__(
        self,
        *,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        system_prompt: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of LiquidAI TTS.

        Args:
            base_url: The base URL of the LFM2.5-Audio server (default: http://127.0.0.1:8080/v1)
            api_key: API key for authentication (default: "dummy")
            voice: Voice description (default: "UK female")
            system_prompt: System prompt for TTS (default: "Perform TTS. Use the UK female voice.")
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(
            voice=voice if is_given(voice) else DEFAULT_VOICE,
            system_prompt=system_prompt if is_given(system_prompt) else DEFAULT_SYSTEM_PROMPT,
        )

        self._client = openai.AsyncClient(
            max_retries=0,
            api_key=api_key if is_given(api_key) else DEFAULT_API_KEY,
            base_url=base_url if is_given(base_url) else DEFAULT_BASE_URL,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=60.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50, max_keepalive_connections=50, keepalive_expiry=120
                ),
            ),
        )

    @property
    def model(self) -> str:
        return "LFM2.5-Audio"

    @property
    def provider(self) -> str:
        return "LiquidAI"

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        system_prompt: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(system_prompt):
            self._opts.system_prompt = system_prompt

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            client=self._client,
        )

    async def aclose(self) -> None:
        await self._client.close()


class ChunkedStream(tts.ChunkedStream):
    """Streaming audio chunks from LiquidAI TTS."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        client: openai.AsyncClient,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._client = client
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = str(uuid.uuid4().hex)[:12]

        try:
            # Create messages for the API
            system_prompt = self._opts.system_prompt
            if self._opts.voice and self._opts.voice not in system_prompt:
                system_prompt = f"{system_prompt}\nVoice: {self._opts.voice}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.input_text},
            ]

            # Call the streaming chat completion API
            response = await self._client.chat.completions.create(
                model="LFM2.5-Audio",
                messages=messages,  # type: ignore
                stream=True,
                max_tokens=2048,
                extra_body={"reset_context": True},
                timeout=self._conn_options.timeout,
            )
            # When stream=True, the response is always an AsyncStream
            stream = cast(AsyncStream[ChatCompletionChunk], response)

            # Initialize the emitter with PCM format
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                mime_type="audio/pcm",
            )

            # Process audio chunks from the stream
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].finish_reason == "stop":
                    break

                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                # Handle audio chunks
                if hasattr(delta, "audio_chunk") and delta.audio_chunk:
                    chunk_data = delta.audio_chunk.get("data")
                    if chunk_data:
                        # Decode base64 PCM float32 data
                        pcm_bytes = base64.b64decode(chunk_data)

                        # Convert float32 samples to int16
                        float_samples = np.frombuffer(pcm_bytes, dtype=np.float32)
                        int16_samples = np.clip(float_samples * 32767, -32768, 32767).astype(
                            np.int16
                        )

                        # Push the converted audio data
                        output_emitter.push(int16_samples.tobytes())

            output_emitter.flush()

        except openai.APITimeoutError:
            raise APIConnectionError() from None
        except asyncio.CancelledError:
            raise
        except openai.APIStatusError as e:
            logger.error(f"TTS API error: {e}")
            raise APIConnectionError() from e
        except Exception as e:
            logger.error(f"TTS error: {e}")
            raise APIConnectionError() from e
