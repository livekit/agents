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
from typing import TYPE_CHECKING, Any, cast

from fish_audio_sdk import ReferenceAudio, Session as FishAudioSession, TTSRequest  # type: ignore[import-untyped]

from livekit.agents import tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .log import logger
from .models import OutputFormat, TTSBackends

if TYPE_CHECKING:
    pass


class TTS(tts.TTS):
    """
    Fish Audio TTS implementation for LiveKit Agents.

    This plugin provides text-to-speech synthesis using Fish Audio's API.
    It supports both reference ID-based and custom reference audio-based synthesis.

    Args:
        api_key (str | None): Fish Audio API key. Can be set via argument or `FISH_API_KEY` environment variable.
        model (TTSBackends): TTS model/backend to use. Defaults to "speech-1.6".
        reference_id (str | None): Optional reference voice model ID.
        output_format (OutputFormat): Audio output format. Defaults to "mp3".
        sample_rate (int): Audio sample rate in Hz. Defaults to 24000.
        num_channels (int): Number of audio channels. Defaults to 1 (mono).
        base_url (str | None): Custom base URL for the Fish Audio API. Optional.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSBackends = "speech-1.6",
        reference_id: str | None = None,
        output_format: OutputFormat = "mp3",
        sample_rate: int = 24000,
        num_channels: int = 1,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._api_key = api_key or os.getenv("FISH_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Fish Audio API key is required, either as argument or set FISH_API_KEY environment variable"
            )

        self._model: TTSBackends = model
        self._output_format: OutputFormat = output_format
        self._reference_id = reference_id or os.getenv("FISH_AUDIO_REFERENCE_ID")
        self._base_url = base_url

        # Initialize Fish Audio session
        if base_url:
            self._session = FishAudioSession(self._api_key, base_url=base_url)
        else:
            self._session = FishAudioSession(self._api_key)

        logger.info(
            "FishAudioTTS initialized",
            extra={
                "model": self._model,
                "format": self._output_format,
                "sample_rate": sample_rate,
            },
        )

    @property
    def model(self) -> TTSBackends:
        """Get the current TTS model/backend."""
        return self._model

    @property
    def output_format(self) -> OutputFormat:
        """Get the current output format."""
        return self._output_format

    @property
    def reference_id(self) -> str | None:
        """Get the current reference voice model ID."""
        return self._reference_id

    @property
    def session(self) -> FishAudioSession:
        """Get the Fish Audio SDK session."""
        return self._session

    def update_options(
        self,
        *,
        model: TTSBackends | None = None,
        reference_id: str | None = None,
    ) -> None:
        """
        Update TTS options dynamically.

        Args:
            model (TTSBackends | None): New TTS model/backend to use.
            reference_id (str | None): New reference voice model ID.
        """
        if model is not None:
            self._model = model
            logger.debug("Updated TTS model", extra={"model": model})

        if reference_id is not None:
            self._reference_id = reference_id
            logger.debug("Updated reference ID", extra={"reference_id": reference_id})

    async def list_models(self) -> list[dict[str, Any]]:
        """
        List available voice models from Fish Audio.

        Returns:
            list[dict[str, Any]]: List of available voice models with their metadata.
        """
        result = await self._session.list_models.awaitable()
        return cast(list[dict[str, Any]], result)

    async def get_model(self, model_id: str) -> dict[str, Any]:
        """
        Get detailed information about a specific voice model.

        Args:
            model_id (str): The model ID to query.

        Returns:
            dict[str, Any]: Model information including metadata and capabilities.
        """
        result = await self._session.get_model.awaitable(model_id)
        return cast(dict[str, Any], result)

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """
        Synthesize speech from text.

        Args:
            text (str): The text to synthesize.
            conn_options (APIConnectOptions): Connection options for the API call.

        Returns:
            ChunkedStream: A stream object that will produce synthesized audio.
        """
        return ChunkedStream(tts_instance=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        """
        Close TTS resources.
        Fish Audio SDK doesn't require explicit cleanup.
        """


class ChunkedStream(tts.ChunkedStream):
    """
    ChunkedStream implementation for Fish Audio TTS.

    This class handles the actual synthesis by communicating with the Fish Audio API
    and streaming the resulting audio data through the LiveKit framework.
    """

    def __init__(
        self,
        *,
        tts_instance: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts_instance, input_text=input_text, conn_options=conn_options)
        self._tts_instance = tts_instance
        self._input_text = input_text
        self._conn_options = conn_options

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """
        Generate audio and push it to the emitter.

        Args:
            output_emitter (tts.AudioEmitter): The emitter to receive synthesized audio data.
        """
        try:
            audio_data = await asyncio.to_thread(self._generate_audio_sync)

            if not audio_data:
                logger.warning("No audio data generated from Fish Audio API")
                return

            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=self._tts_instance.sample_rate,
                num_channels=self._tts_instance.num_channels,
                mime_type=f"audio/{self._tts_instance.output_format}",
            )
            output_emitter.push(audio_data)
            output_emitter.flush()

        except Exception as e:
            logger.error(
                "Fish Audio TTS synthesis failed",
                exc_info=e,
                extra={"text_length": len(self._input_text)},
            )
            raise

    def _generate_audio_sync(self) -> bytes:
        """
        Synchronously generate audio using Fish Audio SDK.

        Returns:
            bytes: The synthesized audio data.
        """
        request = TTSRequest(
            text=self._input_text,
            reference_id=self._tts_instance.reference_id,
            format=self._tts_instance.output_format,
        )

        audio_data = bytearray()
        try:
            for chunk in self._tts_instance.session.tts(request, backend=self._tts_instance.model):
                audio_data.extend(chunk)
        except Exception as e:
            logger.error("Fish Audio SDK TTS call failed", exc_info=e)
            raise

        return bytes(audio_data)


def create_reference_audio(audio: bytes, text: str) -> ReferenceAudio:
    """
    Helper function to create a ReferenceAudio object.

    Args:
        audio (bytes): The reference audio file data.
        text (str): The transcript text of the reference audio.

    Returns:
        ReferenceAudio: A reference audio object for use in TTS requests.
    """
    return ReferenceAudio(audio=audio, text=text)
