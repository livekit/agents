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
This module wraps Oracle's STT cloud service. While it is used by the Oracle LiveKit STT plug-in,
it it completely indpendent of LiveKit and could be used in other environments besides LiveKit.

Author: Keith Schnable (at Oracle Corporation)
Date: 2025-08-12
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from urllib.parse import urlparse

from oci.ai_speech.models import (
    RealtimeMessageAckAudio,
    RealtimeMessageConnect,
    RealtimeMessageError,
    RealtimeMessageResult,
    RealtimeParameters,
)
from oci_ai_speech_realtime import RealtimeSpeechClient, RealtimeSpeechClientListener

from .log import logger
from .utils import AuthenticationType, get_config_and_signer


class OracleSTT(RealtimeSpeechClientListener):
    """
    The Oracle STT class. This class wraps the Oracle STT service.
    """

    def __init__(
        self,
        *,
        base_url: str,  # must be specified
        compartment_id: str,  # must be specified
        authentication_type: AuthenticationType = AuthenticationType.SECURITY_TOKEN,
        authentication_configuration_file_spec: str = "~/.oci/config",
        authentication_profile_name: str = "DEFAULT",
        request_id_prefix: str = "",
        sample_rate: int = 16000,
        language_code: str = "en-US",
        model_domain: str = "GENERIC",
        is_ack_enabled: bool = False,
        partial_silence_threshold_milliseconds: int = 0,
        final_silence_threshold_milliseconds: int = 2000,
        stabilize_partial_results: str = "NONE",
        punctuation: str = "NONE",
        customizations: list[dict] | None = None,
        should_ignore_invalid_customizations: bool = False,
        return_partial_results: bool = False,
    ) -> None:
        """
        Create a new instance of the OracleSTT class to access Oracle's RTS service. This has no LiveKit dependencies.

        Args:
            base_url: Base URL. Type is str. No Default (must be specified).
            compartment_id: Compartment ID. Type is str. No default (must be specified).
            authentication_type: Authentication type. Type is AuthenticationType enum (API_KEY, SECURITY_TOKEN, INSTANCE_PRINCIPAL, or RESOURCE_PRINCIPAL). Default is SECURITY_TOKEN.
            authentication_configuration_file_spec: Authentication configuration file spec. Type is str. Default is "~/.oci/config". Applies only for API_KEY or SECURITY_TOKEN.
            authentication_profile_name: Authentication profile name. Type is str. Default is "DEFAULT". Applies only for API_KEY or SECURITY_TOKEN.
            request_id_prefix: Request ID prefix. Type is str. Default is "".
            sample_rate: Sample rate. Type is int. Default is 16000.
            language_code: Language code. Type is str. Default is "en-US".
            model_domain: Model domain. Type is str. Default is "GENERIC".
            is_ack_enabled: Is-ack-enabled flag. Type is bool. Default is False.
            partial_silence_threshold_milliseconds: Partial silence threshold milliseconds. Type is int. Default is 0.
            final_silence_threshold_milliseconds: Final silence threshold milliseconds. Type is int. Default is 2000.
            stabilize_partial_results: Stabilize partial results. Type is str. Default is "NONE". Must be one of "NONE", "LOW", "MEDIUM", or "HIGH".
            punctuation: Punctuation. Type is str. Default is "NONE". Must be one of "NONE", "SPOKEN", or "AUTO".
            customizations: Customizations. Type is list[dict]. Default is None.
            should_ignore_invalid_customizations. Should-ignore-invalid-customizations flag. Type is bool. Default is False.
            return_partial_results. Return-partial-results flag. Type is bool. Default is False.
        """

        self._parameters = Parameters()
        self._parameters.base_url = base_url
        self._parameters.compartment_id = compartment_id
        self._parameters.authentication_type = authentication_type
        self._parameters.authentication_configuration_file_spec = (
            authentication_configuration_file_spec
        )
        self._parameters.authentication_profile_name = authentication_profile_name
        self._parameters.request_id_prefix = request_id_prefix
        self._parameters.sample_rate = sample_rate
        self._parameters.language_code = language_code
        self._parameters.model_domain = model_domain
        self._parameters.is_ack_enabled = is_ack_enabled
        self._parameters.partial_silence_threshold_milliseconds = (
            partial_silence_threshold_milliseconds
        )
        self._parameters.final_silence_threshold_milliseconds = final_silence_threshold_milliseconds
        self._parameters.stabilize_partial_results = stabilize_partial_results
        self._parameters.punctuation = punctuation
        self._parameters.customizations = customizations
        self._parameters.should_ignore_invalid_customizations = should_ignore_invalid_customizations
        self._parameters.return_partial_results = return_partial_results

        self.validate_parameters()

        self._audio_bytes_queue = asyncio.Queue()
        self._speech_result_queue = asyncio.Queue()

        self._real_time_speech_client = None
        self._connected = False

        asyncio.create_task(self.add_audio_bytes_background_task())

        self.real_time_speech_client_open()

        logger.debug("Initialized OracleSTT.")

    def validate_parameters(self):
        if not isinstance(self._parameters.base_url, str):
            raise TypeError("The base_url parameter must be a string.")
        self._parameters.base_url = self._parameters.base_url.strip()
        parsed = urlparse(self._parameters.base_url)
        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError("The base_url parameter must be a valid URL.")

        if not isinstance(self._parameters.compartment_id, str):
            raise TypeError("The compartment_id parameter must be a string.")
        self._parameters.compartment_id = self._parameters.compartment_id.strip()
        if len(self._parameters.compartment_id) == 0:
            raise ValueError("The compartment_id parameter must not be an empty string.")

        if not isinstance(self._parameters.authentication_type, AuthenticationType):
            raise TypeError(
                "The authentication_type parameter must be one of the AuthenticationType enum members."
            )

        if self._parameters.authentication_type in {
            AuthenticationType.API_KEY,
            AuthenticationType.SECURITY_TOKEN,
        }:
            if not isinstance(self._parameters.authentication_configuration_file_spec, str):
                raise TypeError(
                    "The authentication_configuration_file_spec parameter must be a string."
                )
            self._parameters.authentication_configuration_file_spec = (
                self._parameters.authentication_configuration_file_spec.strip()
            )
            if len(self._parameters.authentication_configuration_file_spec) == 0:
                raise ValueError(
                    "The authentication_configuration_file_spec parameter must not be an empty string."
                )

            if not isinstance(self._parameters.authentication_profile_name, str):
                raise TypeError("The authentication_profile_name parameter must be a string.")
            self._parameters.authentication_profile_name = (
                self._parameters.authentication_profile_name.strip()
            )
            if len(self._parameters.authentication_profile_name) == 0:
                raise ValueError(
                    "The authentication_profile_name parameter must not be an empty string."
                )

        if not isinstance(self._parameters.request_id_prefix, str):
            raise TypeError("The request_id_prefix parameter must be a string.")

        if not isinstance(self._parameters.sample_rate, int):
            raise TypeError("The sample_rate parameter must be an integer.")
        if self._parameters.sample_rate <= 0:
            raise ValueError("The sample_rate parameter must be greater than 0.")

        if not isinstance(self._parameters.language_code, str):
            raise TypeError("The language_code parameter must be a string.")
        self._parameters.language_code = self._parameters.language_code.strip()
        if len(self._parameters.language_code) == 0:
            raise ValueError("The language_code parameter must not be an empty string.")

        if not isinstance(self._parameters.model_domain, str):
            raise TypeError("The model_domain parameter must be a string.")
        self._parameters.model_domain = self._parameters.model_domain.strip()
        if len(self._parameters.model_domain) == 0:
            raise ValueError("The model_domain parameter must not be an empty string.")

        if not isinstance(self._parameters.is_ack_enabled, bool):
            raise TypeError("The is_ack_enabled parameter must be a boolean.")

        if not isinstance(self._parameters.final_silence_threshold_milliseconds, int):
            raise TypeError(
                "The final_silence_threshold_milliseconds parameter must be an integer."
            )
        if self._parameters.final_silence_threshold_milliseconds <= 0:
            raise ValueError(
                "The final_silence_threshold_milliseconds parameter must be greater than 0."
            )

        if not isinstance(self._parameters.return_partial_results, bool):
            raise TypeError("The return_partial_results parameter must be a boolean.")

        if self._parameters.return_partial_results:
            if not isinstance(self._parameters.partial_silence_threshold_milliseconds, int):
                raise TypeError(
                    "The partial_silence_threshold_milliseconds parameter must be an integer."
                )
            if self._parameters.partial_silence_threshold_milliseconds <= 0:
                raise ValueError(
                    "The partial_silence_threshold_milliseconds parameter must be greater than 0."
                )
        else:
            self._parameters.partial_silence_threshold_milliseconds = (
                self._parameters.final_silence_threshold_milliseconds
            )

        if not isinstance(self._parameters.stabilize_partial_results, str):
            raise TypeError("The stabilize_partial_results parameter must be a string.")
        self._parameters.stabilize_partial_results = (
            self._parameters.stabilize_partial_results.strip().upper()
        )
        if self._parameters.stabilize_partial_results not in {"NONE", "LOW", "MEDIUM", "HIGH"}:
            raise ValueError(
                "The stabilize_partial_results parameter must be 'NONE', 'LOW', 'MEDIUM', or 'HIGH'."
            )

        if not isinstance(self._parameters.punctuation, str):
            raise TypeError("The punctuation parameter must be a string.")
        self._parameters.punctuation = self._parameters.punctuation.strip().upper()
        if self._parameters.punctuation not in {"NONE", "SPOKEN", "AUTO"}:
            raise ValueError("The punctuation parameter must be 'NONE', 'SPOKEN', or 'AUTO'.")

        if self._parameters.customizations is not None and (
            not isinstance(self._parameters.customizations, list)
            or not all(isinstance(item, dict) for item in self._parameters.customizations)
        ):
            raise TypeError("The customizations parameter must be None or a list of dictionaries.")

        if not isinstance(self._parameters.should_ignore_invalid_customizations, bool):
            raise TypeError("The should_ignore_invalid_customizations parameter must be a boolean.")

    def add_audio_bytes(self, audio_bytes: bytes) -> None:
        self._audio_bytes_queue.put_nowait(audio_bytes)

    def get_speech_result_queue(self) -> asyncio.Queue:
        return self._speech_result_queue

    def real_time_speech_client_open(self) -> None:
        self.real_time_speech_client_close()

        configAndSigner = get_config_and_signer(
            authentication_type=self._parameters.authentication_type,
            authentication_configuration_file_spec=self._parameters.authentication_configuration_file_spec,
            authentication_profile_name=self._parameters.authentication_profile_name,
        )
        config = configAndSigner["config"]
        signer = configAndSigner["signer"]

        real_time_parameters = RealtimeParameters()

        real_time_parameters.encoding = "audio/raw;rate=" + str(self._parameters.sample_rate)
        real_time_parameters.language_code = self._parameters.language_code
        real_time_parameters.model_domain = self._parameters.model_domain
        real_time_parameters.is_ack_enabled = self._parameters.is_ack_enabled
        real_time_parameters.partial_silence_threshold_in_ms = (
            self._parameters.partial_silence_threshold_milliseconds
        )
        real_time_parameters.final_silence_threshold_in_ms = (
            self._parameters.final_silence_threshold_milliseconds
        )
        real_time_parameters.stabilize_partial_results = self._parameters.stabilize_partial_results
        real_time_parameters.punctuation = self._parameters.punctuation
        if self._parameters.customizations is not None:
            real_time_parameters.customizations = self._parameters._customizations
            real_time_parameters.should_ignore_invalid_customizations = (
                self._parameters.should_ignore_invalid_customizations
            )

        real_time_speech_client_listener = self

        compartment_id = self._parameters.compartment_id

        #
        #  TODO: The self._parameters.request_id_prefix parameter is never used because there is no clear
        #        way to set the opc_request_id using the RealtimeParameters and the RealtimeSpeechClient
        #        classes. The commented-out Python code just below accomplishes part of this but it doesn't
        #        seem to support the four different ways of authenticating which require both the "config"
        #        and "signer" parameters.
        #
        # import oci
        #
        # config = oci.config.from_file()
        #
        # ai_speech_client = oci.ai_speech.AIServiceSpeechClient(config)
        #
        # create_realtime_session_token_response = ai_speech_client.create_realtime_session_token(
        # create_realtime_session_token_details=oci.ai_speech.models.CreateRealtimeSessionTokenDetails(
        #     compartment_id="ocid1.test.oc1..<unique_ID>EXAMPLE-compartmentId-Value",
        #     freeform_tags={
        #         'EXAMPLE_KEY_XhhWK': 'EXAMPLE_VALUE_boU7XVY49wxJc7QtHycR'},
        #     defined_tags={
        #         'EXAMPLE_KEY_C52Iu': {
        #         'EXAMPLE_KEY_0C8XM': 'EXAMPLE--Value'}}),
        # opc_retry_token="EXAMPLE-opcRetryToken-Value",
        # opc_request_id="BRMK4LU7R2XWHOJSLH3S<unique_ID>")
        #

        self._real_time_speech_client = RealtimeSpeechClient(
            config,
            real_time_parameters,
            real_time_speech_client_listener,
            self._parameters.base_url,
            signer,
            compartment_id,
        )

        asyncio.create_task(self.connect_background_task())

    def real_time_speech_client_close(self) -> None:
        if self._real_time_speech_client is not None:
            self._real_time_speech_client.close()
            self._real_time_speech_client = None
        self._connected = False

    async def connect_background_task(self) -> None:
        await self._real_time_speech_client.connect()

    async def add_audio_bytes_background_task(self) -> None:
        while True:
            if (
                self._real_time_speech_client is not None
                and not self._real_time_speech_client.close_flag
                and self._connected
            ):
                logger.trace("Adding audio frame data to RTS SDK.")
                audio_bytes = await self._audio_bytes_queue.get()
                await self._real_time_speech_client.send_data(audio_bytes)
            else:
                await asyncio.sleep(0.010)

    # RealtimeSpeechClient method.
    def on_network_event(self, message):
        super_result = super().on_network_event(message)
        self.real_time_speech_client_open()
        return super_result

    # RealtimeSpeechClient method.
    def on_error(self, error: RealtimeMessageError):
        super_result = super().on_error(error)
        self.real_time_speech_client_open()
        return super_result

    # RealtimeSpeechClient method.
    def on_connect(self):
        return super().on_connect()

    # RealtimeSpeechClient method.
    def on_connect_message(self, connectmessage: RealtimeMessageConnect):
        self._connected = True
        return super().on_connect_message(connectmessage)

    # RealtimeSpeechClient method.
    def on_ack_message(self, ackmessage: RealtimeMessageAckAudio):
        return super().on_ack_message(ackmessage)

    # RealtimeSpeechClient method.
    def on_result(self, result: RealtimeMessageResult):
        super_result = super().on_result(result)

        transcription = result["transcriptions"][0]

        is_final = transcription["isFinal"]
        text = transcription["transcription"]

        log_message = "FINAL" if is_final else "PARTIAL"
        log_message += " utterance: " + text
        logger.debug(log_message)

        if is_final or self._parameters.return_partial_results:
            speech_result = SpeechResult(is_final, text)
            self._speech_result_queue.put_nowait(speech_result)

        return super_result

    # RealtimeSpeechClient method.
    def on_close(self, error_code: int, error_message: str):
        return super().on_close(error_code, error_message)


class Parameters:
    """
    The parameters class. This class contains all parameter information for the Oracle STT class.
    """

    base_url: str
    compartment_id: str
    authentication_type: AuthenticationType
    authentication_configuration_file_spec: str
    authentication_profile_name: str
    sample_rate: int
    language_code: str
    model_domain: str
    is_ack_enabled: bool
    partial_silence_threshold_milliseconds: int
    final_silence_threshold_milliseconds: int
    stabilize_partial_results: str
    punctuation: str
    customizations: list[dict]
    should_ignore_invalid_customizations: bool
    return_partial_results: bool


@dataclass
class SpeechResult:
    """
    The speech result class. This class contains all information related to one speech result.
    """

    is_final: bool
    text: str
