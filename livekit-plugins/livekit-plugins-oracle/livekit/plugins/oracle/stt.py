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
This module is the Oracle LiveKit STT plug-in.

Author: Keith Schnable (at Oracle Corporation)
Date: 2025-08-12
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from livekit import rtc
from livekit.agents import stt

from .log import logger
from .oracle_stt import OracleSTT
from .utils import AuthenticationType

REQUIRED_REAL_TIME_SPEECH_SERVICE_AUDIO_RATE = 16000
REQUIRED_REAL_TIME_SPEECH_SERVICE_IS_ACK_ENABLED = False


class STT(stt.STT):
    """
    The Oracle LiveKit STT plug-in class. This derives from livekit.agents.stt.STT.
    """

    def __init__(
        self,
        *,
        base_url: str,  # must be specified
        compartment_id: str,  # must be specified
        authentication_type: AuthenticationType = AuthenticationType.SECURITY_TOKEN,
        authentication_configuration_file_spec: str = "~/.oci/config",
        authentication_profile_name: str = "DEFAULT",
        language_code: str = "en-US",
        model_domain: str = "GENERIC",
        partial_silence_threshold_milliseconds: int = 0,
        final_silence_threshold_milliseconds: int = 2000,
        stabilize_partial_results: str = "NONE",
        punctuation: str = "NONE",
        customizations: list[dict] | None = None,
        should_ignore_invalid_customizations: bool = False,
        return_partial_results: bool = False,
    ) -> None:
        """
        Create a new instance of the STT class to access Oracle's RTS service. This has LiveKit dependencies.

        Args:
            base_url: Base URL. Type is str. No default (must be specified).
            compartment_id: Compartment ID. Type is str. No default (must be specified).
            authentication_type: Authentication type. Type is AuthenticationType enum (API_KEY, SECURITY_TOKEN, INSTANCE_PRINCIPAL, or RESOURCE_PRINCIPAL). Default is SECURITY_TOKEN.
            authentication_configuration_file_spec: Authentication configuration file spec. Type is str. Default is "~/.oci/config". Applies only for API_KEY or SECURITY_TOKEN.
            authentication_profile_name: Authentication profile name. Type is str. Default is "DEFAULT". Applies only for API_KEY or SECURITY_TOKEN.
            language_code: Language code. Type is str. Default is "en-US".
            model_domain: Model domain. Type is str. Default is "GENERIC".
            partial_silence_threshold_milliseconds: Partial silence threshold milliseconds. Type is int. Default is 0.
            final_silence_threshold_milliseconds: Final silence threshold milliseconds. Type is int. Default is 2000.
            stabilize_partial_results: Stabilize partial results. Type is str. Default is "NONE". Must be one of "NONE", "LOW", "MEDIUM", or "HIGH".
            punctuation: Punctuation. Type is str. Default is "NONE". Must be one of "NONE", "SPOKEN", or "AUTO".
            customizations: Customizations. Type is list[dict]. Default is None.
            should_ignore_invalid_customizations. Should-ignore-invalid-customizations flag. Type is bool. Default is False.
            return_partial_results. Return-partial-results flag. Type is bool. Default is False.
        """

        capabilities = stt.STTCapabilities(streaming=True, interim_results=return_partial_results)
        super().__init__(capabilities=capabilities)

        self._oracle_stt = OracleSTT(
            base_url=base_url,
            compartment_id=compartment_id,
            authentication_type=authentication_type,
            authentication_configuration_file_spec=authentication_configuration_file_spec,
            authentication_profile_name=authentication_profile_name,
            request_id_prefix="live-kit-stt-plug-in-",
            sample_rate=REQUIRED_REAL_TIME_SPEECH_SERVICE_AUDIO_RATE,
            language_code=language_code,
            model_domain=model_domain,
            is_ack_enabled=REQUIRED_REAL_TIME_SPEECH_SERVICE_IS_ACK_ENABLED,
            partial_silence_threshold_milliseconds=partial_silence_threshold_milliseconds,
            final_silence_threshold_milliseconds=final_silence_threshold_milliseconds,
            stabilize_partial_results=stabilize_partial_results,
            punctuation=punctuation,
            customizations=customizations,
            should_ignore_invalid_customizations=should_ignore_invalid_customizations,
            return_partial_results=return_partial_results,
        )

        logger.debug("Initialized STT.")

    async def get_speech_event(self) -> stt.SpeechEvent:
        speech_result_queue = self._oracle_stt.get_speech_result_queue()

        if speech_result_queue.empty():
            return None

        speech_result = await speech_result_queue.get()

        speech_data = stt.SpeechData(
            language="multi",  # this must be "multi" or 4-second delays seem to occur before any tts occurs.
            text=speech_result.text,
        )

        speech_event = stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT
            if speech_result.is_final
            else stt.SpeechEventType.INTERIM_TRANSCRIPT,
            alternatives=[speech_data],
        )

        logger.debug(
            "Returning "
            + ("final" if speech_result.is_final else "partial")
            + " speech result to LiveKit: "
            + speech_result.text
        )

        return speech_event

    # STT method.
    def stream(self, *, language=None, conn_options=None) -> SpeechStream:
        return SpeechStream(self)

    # STT method.
    async def _recognize_impl(
        self,
        audio_buffer,
        *,
        language=None,
        conn_options=None,
    ) -> stt.SpeechEvent:
        speech_data = stt.SpeechData(language="multi", text="zz")

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[speech_data]
        )

    # STT method.
    def on_start(self, participant_id: str, room_id: str):
        pass

    # STT method.
    def on_stop(self):
        pass


class SpeechStream:
    """
    The STT stream class.
    """

    def __init__(self, oracle_stt_livekit_plugin: STT):
        self._running = True
        self._queue = asyncio.Queue()

        self._oracle_stt_livekit_plugin = oracle_stt_livekit_plugin

        self._audio_resampler = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._running = False

    def __aiter__(self) -> AsyncIterator[stt.SpeechEvent]:
        return self._event_stream()

    def push_frame(self, frame: rtc.AudioFrame):
        self._queue.put_nowait(frame)

    async def _event_stream(self) -> AsyncIterator[stt.SpeechEvent]:
        while self._running:
            frame = await self._queue.get()

            logger.trace("Received audio frame data from LiveKit.")

            if frame.sample_rate != REQUIRED_REAL_TIME_SPEECH_SERVICE_AUDIO_RATE:
                if self._audio_resampler is None:
                    self._audio_resampler = rtc.AudioResampler(
                        input_rate=frame.sample_rate,
                        output_rate=REQUIRED_REAL_TIME_SPEECH_SERVICE_AUDIO_RATE,
                        quality=rtc.AudioResamplerQuality.HIGH,
                    )
                frame = self._audio_resampler.push(frame)

            frames = frame if isinstance(frame, list) else [frame]

            for frame in frames:
                audio_bytes = frame.data
                self._oracle_stt_livekit_plugin._oracle_stt.add_audio_bytes(audio_bytes)

            while True:
                speech_event = await self._oracle_stt_livekit_plugin.get_speech_event()
                if speech_event is None:
                    break
                yield speech_event
