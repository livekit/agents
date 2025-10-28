# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Oracle Corporation and/or its affiliates.
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

"""
This module is the Oracle LiveKit TTS plug-in.

Author: Keith Schnable (at Oracle Corporation)
Date: 2025-08-12
"""

from __future__ import annotations

from livekit.agents import tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .audio_cache import AudioCache
from .log import logger
from .oracle_tts import OracleTTS
from .utils import AuthenticationType

REQUIRED_LIVE_KIT_AUDIO_RATE = 24000
REQUIRED_LIVE_KIT_AUDIO_CHANNELS = 1
REQUIRED_LIVE_KIT_AUDIO_BITS = 16


class TTS(tts.TTS):
    """
    The Oracle LiveKit TTS plug-in class. This derives from livekit.agents.tts.TTS.
    """

    def __init__(
        self,
        *,
        base_url: str,  # must be specified
        compartment_id: str,  # must be specified
        authentication_type: AuthenticationType = AuthenticationType.SECURITY_TOKEN,
        authentication_configuration_file_spec: str = "~/.oci/config",
        authentication_profile_name: str = "DEFAULT",
        voice: str = "Victoria",
        audio_cache_file_path: str | None = None,
        audio_cache_maximum_utterance_length: int = 100,
        audio_cache_maximum_number_of_utterances: int = 100,
    ) -> None:
        """
        Create a new instance of the TTS class to access Oracle's TTS service. This has LiveKit dependencies.

        Args:
            base_url: Base URL. Type is str. No default (must be specified).
            compartment_id: Compartment ID. Type is str. No default (must be specified).
            authentication_type: Authentication type. Type is AuthenticationType enum (API_KEY, SECURITY_TOKEN, INSTANCE_PRINCIPAL, or RESOURCE_PRINCIPAL). Default is SECURITY_TOKEN.
            authentication_configuration_file_spec: Authentication configuration file spec. Type is str. Default is "~/.oci/config". Applies only for API_KEY or SECURITY_TOKEN.
            authentication_profile_name: Authentication profile name. Type is str. Default is "DEFAULT". Applies only for API_KEY or SECURITY_TOKEN.
            voice: Voice. Type is str. Default is "Victoria".
            audio_cache_file_path: Audio cache file path. Type is str. Default is None.
            audio_cache_maximum_utterance_length: Audio cache maximum utterance length. Type is int. Default is 100.
            audio_cache_maximum_number_of_utterances: Audio cache maximum number of utterances. Type is int. Default is 100.
        """

        capabilities = tts.TTSCapabilities(streaming=False)

        super().__init__(
            capabilities=capabilities,
            sample_rate=REQUIRED_LIVE_KIT_AUDIO_RATE,
            num_channels=REQUIRED_LIVE_KIT_AUDIO_CHANNELS,
        )

        self._oracle_tts = OracleTTS(
            base_url=base_url,
            compartment_id=compartment_id,
            authentication_type=authentication_type,
            authentication_configuration_file_spec=authentication_configuration_file_spec,
            authentication_profile_name=authentication_profile_name,
            request_id_prefix="live-kit-tts-plug-in-",
            voice=voice,
            sample_rate=REQUIRED_LIVE_KIT_AUDIO_RATE,
        )

        if audio_cache_file_path is not None:
            if not isinstance(audio_cache_file_path, str):
                raise TypeError("The audio_cache_file_path parameter must be a string.")
            audio_cache_file_path = audio_cache_file_path.strip()
            if len(audio_cache_file_path) == 0:
                raise ValueError("The audio_cache_file_path parameter must not be an empty string.")

            if not isinstance(audio_cache_maximum_utterance_length, int):
                raise TypeError(
                    "The audio_cache_maximum_utterance_length parameter must be an integer."
                )
            if audio_cache_maximum_utterance_length <= 0:
                raise ValueError(
                    "The audio_cache_maximum_utterance_length parameter must be greater than 0."
                )

            if not isinstance(audio_cache_maximum_number_of_utterances, int):
                raise TypeError(
                    "The audio_cache_maximum_number_of_utterances parameter must be an integer."
                )
            if audio_cache_maximum_number_of_utterances <= 0:
                raise ValueError(
                    "The audio_cache_maximum_number_of_utterances parameter must be greater than 0."
                )

        if audio_cache_file_path is None:
            self._audio_cache = None
        else:
            self._audio_cache = AudioCache(
                audio_cache_file_path=audio_cache_file_path,
                audio_cache_maximum_number_of_utterances=audio_cache_maximum_number_of_utterances,
            )
            self._voice = voice
            self._audio_cache_maximum_utterance_length = audio_cache_maximum_utterance_length

        logger.debug("Initialized TTS.")

    def synthesize(self, text: str, *, conn_options: DEFAULT_API_CONNECT_OPTIONS) -> ChunkedStream:
        return ChunkedStream(tts=self, text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    """
    The TTS chunked stream class. This derives from livekit.agents.tts.ChunkedStream.
    """

    def __init__(
        self, *, tts: tts.TTS, text: str, conn_options: DEFAULT_API_CONNECT_OPTIONS
    ) -> None:
        super().__init__(tts=tts, input_text=text, conn_options=conn_options)

        self._oracle_tts_livekit_plugin = tts

    async def _run(self, audio_emitter: tts.AudioEmitter) -> None:
        logger.debug("Received text from LiveKit for TTS: " + self._input_text)

        if self._oracle_tts_livekit_plugin._audio_cache is None:
            audio_bytes = None
        else:
            audio_bytes = self._oracle_tts_livekit_plugin._audio_cache.get_audio_bytes(
                text=self._input_text,
                voice=self._oracle_tts_livekit_plugin._voice,
                audio_rate=REQUIRED_LIVE_KIT_AUDIO_RATE,
                audio_channels=REQUIRED_LIVE_KIT_AUDIO_CHANNELS,
                audio_bits=REQUIRED_LIVE_KIT_AUDIO_BITS,
            )

            logger.debug("TTS is" + (" not" if audio_bytes is None else "") + " cached.")

        if audio_bytes is None:
            logger.debug("Before getting TTS audio bytes.")

            audio_bytes = await self._oracle_tts_livekit_plugin._oracle_tts.synthesize_speech(
                text=self._input_text
            )

            logger.debug("After getting TTS audio bytes.")

            audio_bytes_from_cache = False
        else:
            audio_bytes_from_cache = True

        if audio_bytes is not None:
            audio_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=REQUIRED_LIVE_KIT_AUDIO_RATE,
                num_channels=REQUIRED_LIVE_KIT_AUDIO_CHANNELS,
                mime_type="audio/pcm",
            )

            audio_emitter.push(audio_bytes)
            audio_emitter.flush()

            if (
                not audio_bytes_from_cache
                and self._oracle_tts_livekit_plugin._audio_cache is not None
                and len(self._input_text)
                <= self._oracle_tts_livekit_plugin._audio_cache_maximum_utterance_length
            ):
                self._oracle_tts_livekit_plugin._audio_cache.set_audio_bytes(
                    text=self._input_text,
                    voice=self._oracle_tts_livekit_plugin._voice,
                    audio_rate=REQUIRED_LIVE_KIT_AUDIO_RATE,
                    audio_channels=REQUIRED_LIVE_KIT_AUDIO_CHANNELS,
                    audio_bits=REQUIRED_LIVE_KIT_AUDIO_BITS,
                    audio_bytes=audio_bytes,
                )
