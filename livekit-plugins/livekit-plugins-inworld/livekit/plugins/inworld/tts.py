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

from __future__ import annotations

import asyncio
import base64
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass, replace
from typing import Literal, cast

from inworld_sdk import InworldClient, models

from livekit.agents import tokenize, tts, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
)
from livekit.agents.utils import shortuuid

AUDIO_ENCODING = "LINEAR16"
INWORLD_API_BASE_URL = "https://api.inworld.ai/tts/v1alpha"
MIME_TYPE = "audio/pcm"
NUM_CHANNELS = 1
SAMPLE_RATE = 24000
SPEED = 1.0
WAV_HEADER_SIZE = 44


def strip_wav_header(audio_data: bytes) -> bytes:
    if len(audio_data) <= WAV_HEADER_SIZE:
        return audio_data
    elif audio_data.startswith(b"RIFF") and audio_data[8:12] == b"WAVE":
        return audio_data[WAV_HEADER_SIZE:]
    else:
        return audio_data


@dataclass
class _TTSOptions:
    modelId: models.TTSModelIds | str | None
    languageCode: models.TTSLanguageCodes | str | None
    voice: models.TTSVoices | str | None
    speed: float
    sampleRateHertz: int
    tokenizer: tokenize.basic.SentenceTokenizer


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: models.TTSModelIds | str | None = None,
        language: models.TTSLanguageCodes | str | None = None,
        voice: models.TTSVoices | str | None = None,
        speed: float = SPEED,
        sample_rate: int = SAMPLE_RATE,
        base_url: str = INWORLD_API_BASE_URL,
        auth_type: Literal["basic", "bearer"] = "basic",
    ) -> None:
        """
        Create a new instance of Inworld AI TTS.

        Args:
            api_key (str, optional): The Inworld AI API key.
                If not provided, it will be read from the INWORLD_API_KEY environment variable.
            model (str, optional): The Inworld AI model to use.
            language (TTSLanguageCodes, str, optional): The language code for synthesis.
            voice (TTSVoices, str, optional): The voice to use.
            speed (float, optional): The speed of the voice. Defaults to 1.0.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            base_url (str, optional): The base URL for the Inworld AI API.
            auth_type (Literal["basic", "bearer"], optional): The authentication type to use.
                Defaults to "basic".
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.getenv("INWORLD_API_KEY")
        if not api_key:
            raise ValueError(
                "Inworld API key required. Set INWORLD_API_KEY or provide api_key."
            )

        self._opts = _TTSOptions(
            modelId=model,
            languageCode=language,
            voice=voice,
            speed=speed,
            sampleRateHertz=sample_rate,
            tokenizer=tokenize.basic.SentenceTokenizer(
                min_sentence_len=10,
                stream_context_len=5,
            ),
        )

        self._client = InworldClient(
            api_key=api_key,
            auth_type=auth_type,
            base_url=base_url,
        )

    def update_options(
        self,
        *,
        model: models.TTSModelIds | str | None = None,
        language: models.TTSLanguageCodes | str | None = None,
        voice: models.TTSVoices | str | None = None,
        speed: float | None = None,
        sample_rate: int | None = None,
    ) -> None:
        """
        Update the TTS configuration options.

        Args:
            model (str, optional): The Inworld AI model to use.
            language (TTSLanguageCodes, optional): The language code for synthesis.
            voice (TTSVoices, optional): The voice to use.
            speed (float, optional): The speed of the voice.
            sample_rate (int, optional): The audio sample rate in Hz.
        """
        if model is not None:
            self._opts.modelId = model
        if language is not None:
            self._opts.languageCode = language
        if voice is not None:
            self._opts.voice = voice
        if speed is not None:
            self._opts.speed = speed
        if sample_rate is not None:
            self._opts.sampleRateHertz = sample_rate

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        stream = self.stream(conn_options=conn_options)
        stream.push_text(text)
        stream.end_input()
        return cast(tts.ChunkedStream, stream)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the TTS synthesis process."""
        tokenizer_stream = self._opts.tokenizer.stream()
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=self._opts.sampleRateHertz,
            num_channels=NUM_CHANNELS,
            mime_type=MIME_TYPE,
            stream=True,
        )

        tasks = [
            asyncio.create_task(self._tokenize_input(tokenizer_stream)),
            asyncio.create_task(
                self._synthesize_sentences(tokenizer_stream, output_emitter)
            ),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)

    async def _tokenize_input(self, tokenizer_stream: tokenize.SentenceStream) -> None:
        """Feed input text to the tokenizer stream."""
        async for text in self._input_ch:
            if isinstance(text, str):
                tokenizer_stream.push_text(text)
            elif isinstance(text, self._FlushSentinel):
                tokenizer_stream.flush()

        tokenizer_stream.end_input()

    async def _synthesize_sentences(
        self,
        tokenizer_stream: tokenize.SentenceStream,
        output_emitter: tts.AudioEmitter,
    ) -> None:
        """Process sentences from tokenizer and synthesize audio."""
        output_emitter.start_segment(segment_id=shortuuid())

        try:
            async for sentence in tokenizer_stream:
                if not sentence.token.strip():
                    continue
                await self._synthesize_sentence(sentence, output_emitter)
            output_emitter.end_segment()
        except Exception as e:
            print(f"Error synthesizing segment: {e}")
            raise

    async def _synthesize_sentence(
        self, sentence: tokenize.TokenData, output_emitter: tts.AudioEmitter
    ) -> None:
        """Synthesize a single sentence to audio."""
        try:
            stream = self._create_synthesis_stream(sentence)
            await self._process_audio_stream(stream, output_emitter)
        except ValueError as e:
            if "Chunk too big" in str(e):
                print(f"Warning: Audio chunk too large, skipping: {e}")
            else:
                raise
        except Exception as e:
            print(f"Error calling Inworld API: {e}")

    def _create_synthesis_stream(
        self, sentence: tokenize.TokenData
    ) -> AsyncGenerator[models.SynthesizeSpeechResponse, None]:
        """Synthesize a stream for a sentence."""
        return self._tts._client.tts.synthesizeSpeechStream(
            input=sentence.token,
            voice=self._opts.voice,
            languageCode=self._opts.languageCode,
            modelId=self._opts.modelId,
            audioConfig=cast(
                models.AudioConfig,
                {
                    "audioEncoding": AUDIO_ENCODING,
                    "speakingRate": self._opts.speed,
                    "sampleRateHertz": self._opts.sampleRateHertz,
                },
            ),
        )

    async def _process_audio_stream(
        self,
        stream: AsyncGenerator[models.SynthesizeSpeechResponse, None],
        output_emitter: tts.AudioEmitter,
    ) -> None:
        """Process audio chunks from the synthesized stream."""
        async for chunk in stream:
            if chunk and chunk.get("audioContent"):
                try:
                    audio_data = base64.b64decode(chunk["audioContent"])
                    audio_data = strip_wav_header(audio_data)
                    if audio_data:
                        self._push_audio_chunks(audio_data, output_emitter)
                except Exception as e:
                    print(f"Error processing audio: {e}")

    def _push_audio_chunks(
        self, audio_data: bytes, output_emitter: tts.AudioEmitter
    ) -> None:
        """Push audio data to the emitter in manageable chunks."""
        max_chunk_size = 8192
        for i in range(0, len(audio_data), max_chunk_size):
            chunk_data = audio_data[i : i + max_chunk_size]
            output_emitter.push(chunk_data)
