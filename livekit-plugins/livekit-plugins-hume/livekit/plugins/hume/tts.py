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
import base64
import json
import os
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, TypedDict

import aiohttp

from livekit.agents import APIConnectionError, APIConnectOptions, APITimeoutError, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .version import __version__


class VoiceById(TypedDict, total=False):
    id: str
    provider: VoiceProvider | None


class VoiceByName(TypedDict, total=False):
    name: str
    provider: VoiceProvider | None


class UtteranceOptions(TypedDict, total=False):
    """Options for configuring each utterance sent to Hume TTS."""

    description: str | None
    """Natural language instructions describing how the synthesized speech should sound
    (≤1000 characters)

    Including but not limited to tone, intonation, pacing, and accent (e.g., 'a soft, gentle
    voice with a strong British accent').

    - If a Voice is specified in the request, this description serves as acting instructions.
      See: https://dev.hume.ai/docs/text-to-speech-tts/acting-instructions
    - If no Voice is specified, a new voice is generated based on this description.
      See: https://dev.hume.ai/docs/text-to-speech-tts/prompting
    """
    speed: float | None
    """Speed multiplier for the synthesized speech (≥0.25, ≤3.0, default: 1.0)

    Note: Changes are non-linear - setting to 2.0 makes speech faster but not exactly twice as fast.
    """
    voice: VoiceById | VoiceByName | None
    """The name or id associated with a Voice from the Voice Library.

    See: https://dev.hume.ai/docs/text-to-speech-tts/voices
    """
    trailing_silence: float | None
    """Duration of trailing silence (in seconds) to add to each utterance. (≥0, ≤5.0, default: 0.35)
    """


class Utterance(UtteranceOptions, total=False):
    text: str


class VoiceProvider(str, Enum):
    """Voice provider for the voice library."""

    hume = "HUME_AI"
    custom = "CUSTOM_VOICE"


class AudioFormat(str, Enum):
    """Audio format for the synthesized speech."""

    mp3 = "mp3"
    wav = "wav"
    pcm = "pcm"


DEFAULT_HEADERS = {
    "X-Hume-Client-Name": "LiveKit",
    "X-Hume-Client-Version": __version__,
}
API_AUTH_HEADER = "X-Hume-Api-Key"
STREAM_PATH = "/v0/tts/stream/json"
DEFAULT_BASE_URL = "https://api.hume.ai"
SUPPORTED_SAMPLE_RATE = 48000
DEFAULT_UTTERANCE_OPTIONS = {
    "voice": {"name": "Male English Actor", "provider": VoiceProvider.hume}
}


@dataclass
class _TTSOptions:
    api_key: str
    base_url: str
    utterance_options: Utterance
    context: str | list[Utterance] | None
    instant_mode: bool | None
    audio_format: AudioFormat

    def http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        utterance_options: NotGivenOr[UtteranceOptions] = NOT_GIVEN,
        context: str | list[Utterance] | None = None,
        instant_mode: NotGivenOr[bool] = NOT_GIVEN,
        audio_format: AudioFormat = AudioFormat.mp3,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ):
        """Initialize the Hume AI TTS client. Options will be used for all future synthesis
        (until updated with update_options).

        Args:
            api_key: Hume AI API key. If not provided, will look for HUME_API_KEY environment
                variable.
            utterance_options: Options for text-to-speech synthesis. If not provided, uses
                default options.
            context: Optional context for synthesis, either as text or list of utterances.
            instant_mode: Whether to use instant mode. Defaults to True if voice specified,
                False otherwise. Requires a voice to be specified when enabled.
            audio_format: Output audio format (mp3, wav, or pcm). Defaults to mp3.
            base_url: Base URL for Hume AI API. Defaults to https://api.hume.ai
            http_session: Optional aiohttp ClientSession to use for requests.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SUPPORTED_SAMPLE_RATE,
            num_channels=1,
        )
        key = api_key or os.environ.get("HUME_API_KEY")
        if not key:
            raise ValueError("Hume API key is required via api_key or HUME_API_KEY env var")

        resolved_utterance_options = (
            utterance_options if is_given(utterance_options) else DEFAULT_UTTERANCE_OPTIONS
        )

        has_voice = resolved_utterance_options.get("voice") is not None

        # Default instant_mode is True if a voice is specified, otherwise False
        # (Hume API requires a voice for instant mode)
        if not is_given(instant_mode):
            resolved_instant_mode = has_voice
        elif instant_mode and not has_voice:
            raise ValueError("Hume TTS: instant_mode cannot be enabled without specifying a voice")
        else:
            resolved_instant_mode = instant_mode

        self._opts = _TTSOptions(
            api_key=key,
            utterance_options=resolved_utterance_options,
            context=context,
            instant_mode=resolved_instant_mode,
            audio_format=audio_format,
            base_url=base_url,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def update_options(
        self,
        *,
        utterance_options: NotGivenOr[UtteranceOptions] = NOT_GIVEN,
        context: NotGivenOr[str | list[Utterance]] = NOT_GIVEN,
        instant_mode: NotGivenOr[bool] = NOT_GIVEN,
        audio_format: NotGivenOr[AudioFormat] = NOT_GIVEN,
    ) -> None:
        """Update TTS options used for all future synthesis (until updated again)

        Args:
            utterance_options: Options for text-to-speech synthesis.
            context: Optional context for synthesis, either as text or list of utterances.
            instant_mode: Whether to use instant mode.
            audio_format: Output audio format (mp3, wav, or pcm).
        """
        if is_given(utterance_options):
            self._opts.utterance_options = utterance_options
        if is_given(context):
            self._opts.context = context
        if is_given(instant_mode):
            self._opts.instant_mode = instant_mode
        if is_given(audio_format):
            self._opts.audio_format = audio_format

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        utterance: Utterance = {**self._opts.utterance_options, "text": self._input_text}

        payload: dict[str, Any] = {
            "utterances": [utterance],
            "strip_headers": True,
            "instant_mode": self._opts.instant_mode,
            "format": {"type": self._opts.audio_format.value},
        }
        if self._opts.context is str:
            payload["context"] = {"generation_id": self._opts.context}
        elif self._opts.context is list[Utterance]:
            payload["context"] = {"utterances": self._opts.context}

        try:
            async with self._tts._ensure_session().post(
                self._opts.http_url(STREAM_PATH),
                headers={**DEFAULT_HEADERS, API_AUTH_HEADER: self._opts.api_key},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=self._conn_options.timeout),
                # large read_bufsize to avoid `ValueError: Chunk too big`
                read_bufsize=10 * 1024 * 1024,
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=SUPPORTED_SAMPLE_RATE,
                    num_channels=self._tts.num_channels,
                    mime_type=f"audio/{self._opts.audio_format.value}",
                )

                async for raw_line in resp.content:
                    line = raw_line.strip()
                    if not line:
                        continue

                    data = json.loads(line.decode())
                    audio_b64 = data.get("audio")
                    if audio_b64:
                        output_emitter.push(base64.b64decode(audio_b64))

                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError() from e
