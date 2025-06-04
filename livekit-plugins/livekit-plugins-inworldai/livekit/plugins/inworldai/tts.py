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
from dataclasses import dataclass, replace

from inworld_sdk import InworldAIClient

from livekit.agents import APIConnectionError, tokenize, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import shortuuid

from .models import (
    TTSLanguageCodes,
    TTSVoices,
)

NUM_CHANNELS = 1
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
    modelId: str
    languageCode: TTSLanguageCodes
    voice: TTSVoices
    speed: float
    sampleRateHertz: int
    tokenizer: tokenize.basic.SentenceTokenizer


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        language: TTSLanguageCodes | None = None,
        voice: TTSVoices | None = None,
        speed: float | None = None,
        sample_rate: int = 24000,
        base_url: str | None = None,
        auth_type: str | None = None,
    ) -> None:
        """
        Create a new instance of Inworld AI TTS.

        Args:
            api_key (str, optional): The Inworld AI API key.
                If not provided, it will be read from the INWORLD_API_KEY environment variable.
            model (str, optional): The Inworld AI model to use.
            language (TTSLanguageCodes, optional): The language code for synthesis.
            voice (TTSVoices, optional): The voice to use.
            speed (float, optional): The speed of the voice.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            base_url (str, optional): The base URL for the Inworld AI API.
            auth_type (str, optional): The authentication type to use.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(
            modelId=model,
            languageCode=language,
            voice=voice,
            speed=speed,
            sampleRateHertz=sample_rate,
            tokenizer=tokenize.basic.SentenceTokenizer(),
        )

        self._client = InworldAIClient(
            api_key=(api_key if api_key is not None else os.environ.get("INWORLD_API_KEY")),
            auth_type=auth_type,
            base_url=base_url,
        )

    def update_options(
        self,
        *,
        model: str | None = None,
        language: TTSLanguageCodes | None = None,
        voice: TTSVoices | None = None,
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
        conn_options: tts.APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: tts.APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: tts.APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=self._opts.sampleRateHertz,
            num_channels=NUM_CHANNELS,
            mime_type="audio/raw",
        )

        try:
            resp = await self._tts._client.tts.synthesizeSpeech(
                input=self._input_text,
                voice=self._opts.voice,
                modelId=self._opts.modelId,
                languageCode=self._opts.languageCode,
                audioConfig={
                    "audioEncoding": "LINEAR16",
                    "speakingRate": self._opts.speed,
                    "sampleRateHertz": self._opts.sampleRateHertz,
                },
            )

            if resp and resp.get("audioContent"):
                try:
                    decoded_audio = base64.b64decode(resp["audioContent"])
                    headerless_audio = strip_wav_header(decoded_audio)
                    output_emitter.push(headerless_audio)
                except Exception as e:
                    print(f"Error processing audio: {e}")
                    raise

            output_emitter.flush()

        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: tts.APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._segments_ch = utils.aio.Chan[tokenize.SentenceStream]()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=self._opts.sampleRateHertz,
            num_channels=NUM_CHANNELS,
            mime_type="audio/raw",
            stream=True,
        )

        async def _tokenize_input() -> None:
            """Tokenize text from the input_ch into sentences"""
            input_stream = None
            async for text in self._input_ch:
                if isinstance(text, str):
                    if input_stream is None:
                        input_stream = self._opts.tokenizer.stream()
                        self._segments_ch.send_nowait(input_stream)
                    input_stream.push_text(text)
                elif isinstance(text, self._FlushSentinel):
                    if input_stream is not None:
                        input_stream.end_input()
                    input_stream = None

            if input_stream is not None:
                input_stream.end_input()

            self._segments_ch.close()

        async def _process_segments() -> None:
            async for input_stream in self._segments_ch:
                await self._synthesize_segment(input_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_process_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)

    async def _synthesize_segment(
        self, input_stream: tokenize.SentenceStream, output_emitter: tts.AudioEmitter
    ) -> None:
        """Synthesize a single segment of text and push it to the output emitter."""
        output_emitter.start_segment(segment_id=shortuuid())

        try:
            async for sentence in input_stream:
                if not sentence.token.strip():
                    continue

                resp = await self._tts._client.tts.synthesizeSpeech(
                    input=sentence.token,
                    voice=self._opts.voice,
                    modelId=self._opts.modelId,
                    languageCode=self._opts.languageCode,
                    audioConfig={
                        "audioEncoding": "LINEAR16",
                        "speakingRate": self._opts.speed,
                        "sampleRateHertz": self._opts.sampleRateHertz,
                    },
                )

                if resp and resp.get("audioContent"):
                    try:
                        decoded_audio = base64.b64decode(resp["audioContent"])
                        headerless_audio = strip_wav_header(decoded_audio)
                        output_emitter.push(headerless_audio)
                    except Exception as e:
                        print(f"Error processing audio: {e}")
                        continue

            output_emitter.flush()

        except Exception as e:
            print(f"Error synthesizing segment: {e}")
            raise
