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
import os
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

from .models import TTSAudioFormat, TTSEmotionPreset, TTSEmotionType, TTSModels

API_BASE_URL = "https://api.typecast.ai/v1"
API_KEY_HEADER = "X-API-KEY"

# Typecast outputs mono 44100 Hz audio
_SAMPLE_RATE = 44100
_NUM_CHANNELS = 1

DEFAULT_MODEL: TTSModels = "ssfm-v30"
DEFAULT_AUDIO_FORMAT: TTSAudioFormat = "mp3"


def _audio_format_to_mimetype(audio_format: TTSAudioFormat) -> str:
    if audio_format == "wav":
        return "audio/wav"
    elif audio_format == "mp3":
        return "audio/mpeg"
    raise ValueError(f"Unsupported audio format: {audio_format}")


@dataclass
class _TTSOptions:
    api_key: str
    voice_id: str
    model: TTSModels | str
    language: NotGivenOr[str]
    audio_format: TTSAudioFormat
    emotion_type: TTSEmotionType
    emotion_preset: NotGivenOr[TTSEmotionPreset | str]
    emotion_intensity: NotGivenOr[float]
    audio_tempo: NotGivenOr[float]
    audio_pitch: NotGivenOr[int]
    volume: NotGivenOr[int]
    seed: NotGivenOr[int]


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: str,
        model: TTSModels | str = DEFAULT_MODEL,
        language: NotGivenOr[str] = NOT_GIVEN,
        audio_format: TTSAudioFormat = DEFAULT_AUDIO_FORMAT,
        emotion_type: TTSEmotionType = "smart",
        emotion_preset: NotGivenOr[TTSEmotionPreset | str] = NOT_GIVEN,
        emotion_intensity: NotGivenOr[float] = NOT_GIVEN,
        audio_tempo: NotGivenOr[float] = NOT_GIVEN,
        audio_pitch: NotGivenOr[int] = NOT_GIVEN,
        volume: NotGivenOr[int] = NOT_GIVEN,
        seed: NotGivenOr[int] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Typecast TTS.

        Args:
            voice_id (str): Typecast voice ID (format: ``tc_<id>``).
            model (TTSModels | str): TTS model. Defaults to ``"ssfm-v30"``.
            language (NotGivenOr[str]): ISO 639-3 language code (e.g. ``"kor"``, ``"eng"``).
                Auto-detected if omitted.
            audio_format (TTSAudioFormat): Output format, ``"wav"`` or ``"mp3"``.
                Defaults to ``"mp3"``.
            emotion_type (TTSEmotionType): ``"smart"`` for AI-inferred emotion, ``"preset"``
                for manual selection. Defaults to ``"smart"``.
            emotion_preset (NotGivenOr[TTSEmotionPreset | str]): Emotion preset name.
                Only used when ``emotion_type="preset"``.
            emotion_intensity (NotGivenOr[float]): Emotion intensity 0.0–2.0.
                Only used when ``emotion_type="preset"``.
            audio_tempo (NotGivenOr[float]): Playback speed multiplier 0.5–2.0.
            audio_pitch (NotGivenOr[int]): Pitch adjustment in semitones, -12 to +12.
            volume (NotGivenOr[int]): Output volume 0–200. Defaults to 100.
            seed (NotGivenOr[int]): Seed for reproducible generation.
            api_key (NotGivenOr[str]): Typecast API key. Can also be set via the
                ``TYPECAST_API_KEY`` environment variable.
            http_session (aiohttp.ClientSession | None): Optional custom HTTP session.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=_SAMPLE_RATE,
            num_channels=_NUM_CHANNELS,
        )

        typecast_api_key = api_key if is_given(api_key) else os.environ.get("TYPECAST_API_KEY")
        if not typecast_api_key:
            raise ValueError(
                "Typecast API key is required, either as argument or set TYPECAST_API_KEY environment variable"
            )

        self._opts = _TTSOptions(
            api_key=typecast_api_key,
            voice_id=voice_id,
            model=model,
            language=language,
            audio_format=audio_format,
            emotion_type=emotion_type,
            emotion_preset=emotion_preset,
            emotion_intensity=emotion_intensity,
            audio_tempo=audio_tempo,
            audio_pitch=audio_pitch,
            volume=volume,
            seed=seed,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Typecast"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        emotion_type: NotGivenOr[TTSEmotionType] = NOT_GIVEN,
        emotion_preset: NotGivenOr[TTSEmotionPreset | str] = NOT_GIVEN,
        emotion_intensity: NotGivenOr[float] = NOT_GIVEN,
        audio_tempo: NotGivenOr[float] = NOT_GIVEN,
        audio_pitch: NotGivenOr[int] = NOT_GIVEN,
        volume: NotGivenOr[int] = NOT_GIVEN,
        seed: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(emotion_type):
            self._opts.emotion_type = emotion_type
        if is_given(emotion_preset):
            self._opts.emotion_preset = emotion_preset
        if is_given(emotion_intensity):
            self._opts.emotion_intensity = emotion_intensity
        if is_given(audio_tempo):
            self._opts.audio_tempo = audio_tempo
        if is_given(audio_pitch):
            self._opts.audio_pitch = audio_pitch
        if is_given(volume):
            self._opts.volume = volume
        if is_given(seed):
            self._opts.seed = seed

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        pass


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the Typecast REST API endpoint."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload = _build_payload(self._opts, self._input_text)

        try:
            async with self._tts._ensure_session().post(
                f"{API_BASE_URL}/text-to-speech",
                headers={
                    API_KEY_HEADER: self._opts.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()

                if not resp.content_type.startswith("audio/"):
                    content = await resp.text()
                    raise APIError(message="Typecast returned non-audio data", body=content)

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=_SAMPLE_RATE,
                    num_channels=_NUM_CHANNELS,
                    mime_type=_audio_format_to_mimetype(self._opts.audio_format),
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


def _build_payload(opts: _TTSOptions, text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": text,
        "model": opts.model,
        "voice_id": opts.voice_id,
    }

    if is_given(opts.language):
        payload["language"] = opts.language

    # Emotion control: smart uses AI-inferred emotion; preset uses explicit emotion_preset/intensity
    prompt: dict[str, Any] = {"emotion_type": opts.emotion_type}
    if opts.emotion_type == "preset":
        if is_given(opts.emotion_preset):
            prompt["emotion_preset"] = opts.emotion_preset
        if is_given(opts.emotion_intensity):
            prompt["emotion_intensity"] = opts.emotion_intensity
    payload["prompt"] = prompt

    output: dict[str, Any] = {"audio_format": opts.audio_format}
    if is_given(opts.volume):
        output["volume"] = opts.volume
    if is_given(opts.audio_pitch):
        output["audio_pitch"] = opts.audio_pitch
    if is_given(opts.audio_tempo):
        output["audio_tempo"] = opts.audio_tempo
    payload["output"] = output

    if is_given(opts.seed):
        payload["seed"] = opts.seed

    return payload
