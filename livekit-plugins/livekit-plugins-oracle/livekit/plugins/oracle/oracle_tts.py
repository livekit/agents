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
This module wraps Oracle's TTS cloud service. While it is used by the Oracle LiveKit TTS plug-in,
it it completely indpendent of LiveKit and could be used in other environments besides LiveKit.

Author: Keith Schnable (at Oracle Corporation)
Date: 2025-08-12
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from urllib.parse import urlparse

import oci
from oci.ai_speech import AIServiceSpeechClient
from oci.ai_speech.models import (
    TtsOracleConfiguration,
    TtsOracleSpeechSettings,
    TtsOracleTts2NaturalModelDetails,
)

from .log import logger
from .utils import AuthenticationType, get_config_and_signer


class OracleTTS:
    """
    The Oracle TTS class. This class wraps the Oracle TTS service.
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
        voice: str = "Victoria",
        sample_rate: int = 16000,
    ) -> None:
        """
        Create a new instance of the OracleTTS class to access Oracle's TTS service. This has no LiveKit dependencies.

        Args:
            base_url: Base URL. Type is str. No default (must be specified).
            compartment_id: Compartment ID. Type is str. No default (must be specified).
            authentication_type: Authentication type. Type is AuthenticationType enum (API_KEY, SECURITY_TOKEN, INSTANCE_PRINCIPAL, or RESOURCE_PRINCIPAL). Default is SECURITY_TOKEN.
            authentication_configuration_file_spec: Authentication configuration file spec. Type is str. Default is "~/.oci/config". Applies only for API_KEY or SECURITY_TOKEN.
            authentication_profile_name: Authentication profile name. Type is str. Default is "DEFAULT". Applies only for API_KEY or SECURITY_TOKEN.
            request_id_prefix: Request ID prefix. Type is str. Default is "".
            voice: Voice. Type is str. Default is "Victoria".
            sample_rate: Sample rate. Type is int. Default is 16000.
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
        self._parameters.voice = voice
        self._parameters.sample_rate = sample_rate

        self.validate_parameters()

        configAndSigner = get_config_and_signer(
            authentication_type=self._parameters.authentication_type,
            authentication_configuration_file_spec=self._parameters.authentication_configuration_file_spec,
            authentication_profile_name=self._parameters.authentication_profile_name,
        )
        config = configAndSigner["config"]
        signer = configAndSigner["signer"]

        if signer is None:
            self._ai_service_speech_client = AIServiceSpeechClient(
                config=config, service_endpoint=self._parameters.base_url
            )
        else:
            self._ai_service_speech_client = AIServiceSpeechClient(
                config=config, service_endpoint=self._parameters.base_url, signer=signer
            )

        logger.debug("Initialized OracleTTS.")

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

        if not isinstance(self._parameters.voice, str):
            raise TypeError("The voice parameter must be a string.")
        self._parameters.voice = self._parameters.voice.strip()
        if len(self._parameters.voice) == 0:
            raise ValueError("The voice parameter must not be an empty string.")

        if not isinstance(self._parameters.sample_rate, int):
            raise TypeError("The sample_rate parameter must be an integer.")
        if self._parameters.sample_rate <= 0:
            raise ValueError("The sample_rate parameter must be greater than 0.")

    async def synthesize_speech(self, *, text: str) -> bytes:
        def sync_call():
            request_id = self._parameters.request_id_prefix + short_uuid()

            logger.debug("Before call to TTS service for: " + text)

            #
            #  this link may help if ever setting is_stream_enabled = True. this will only noticeably reduce latency
            #  if multiple sentences are passed into synthesize_speech() at a time.
            #
            #  https://confluence.oraclecorp.com/confluence/pages/viewpage.action?pageId=11517257226
            #
            response = self._ai_service_speech_client.synthesize_speech(
                synthesize_speech_details=oci.ai_speech.models.SynthesizeSpeechDetails(
                    text=text,
                    is_stream_enabled=False,
                    compartment_id=self._parameters.compartment_id,
                    configuration=TtsOracleConfiguration(
                        model_family="ORACLE",
                        model_details=TtsOracleTts2NaturalModelDetails(
                            model_name="TTS_2_NATURAL", voice_id=self._parameters.voice
                        ),
                        speech_settings=TtsOracleSpeechSettings(
                            text_type="TEXT",
                            sample_rate_in_hz=self._parameters.sample_rate,
                            output_format="PCM",
                        ),
                    ),
                ),
                opc_request_id=request_id,
            )

            logger.debug("After call to TTS service for: " + text)

            if response is None or response.status != 200:
                logger.error("Error calling TTS service for: " + text)
                return None

            #
            #  the data is in .wav file format so remove the 44-byte .wav header.
            #
            audio_bytes = response.data.content[44:]

            return audio_bytes

        return await asyncio.to_thread(sync_call)


@staticmethod
def short_uuid() -> str:
    uuid4 = uuid.uuid4()
    base64EncodedUUID = base64.urlsafe_b64encode(uuid4.bytes)
    return base64EncodedUUID.rstrip(b"=").decode("ascii")


class Parameters:
    """
    The parameters class. This class contains all parameter information for the Oracle TTS class.
    """

    base_url: str
    compartment_id: str
    authentication_type: AuthenticationType
    authentication_configuration_file_spec: str
    authentication_profile_name: str
    voice: str
    sample_rate: int
