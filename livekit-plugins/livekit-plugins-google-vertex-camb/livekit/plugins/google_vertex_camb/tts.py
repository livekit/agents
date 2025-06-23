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
import tempfile
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf
from google.cloud import aiplatform
from livekit import rtc

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .models import Language

# Constants
NUM_CHANNELS = 1
DEFAULT_SAMPLE_RATE = 24000

@dataclass
class _TTSOptions:
    """Internal options for Google Vertex AI MARS7 TTS"""

    project_id: str
    location: str
    endpoint_id: str
    credentials_path: NotGivenOr[str]
    language: Language
    audio_ref_path: NotGivenOr[str]
    ref_text: NotGivenOr[str]
    word_tokenizer: tokenize.WordTokenizer


class TTS(tts.TTS):
    """Google Vertex AI MARS7 Text-to-Speech implementation"""

    def __init__(
        self,
        *,
        endpoint_id: str,
        project_id: NotGivenOr[str] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        credentials_path: NotGivenOr[str] = NOT_GIVEN,
        language: Language = Language.EN_US,
        audio_ref_path: NotGivenOr[str] = NOT_GIVEN,
        ref_text: NotGivenOr[str] = NOT_GIVEN,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Google Vertex AI MARS7 TTS.

        Args:
            endpoint_id (str): Vertex AI endpoint ID for MARS7 model.
            project_id (NotGivenOr[str]): Google Cloud project ID. Can be set via argument or
                `GOOGLE_CLOUD_PROJECT` environment variable. Defaults to 'cambai-public'.
            location (NotGivenOr[str]): Google Cloud location. Defaults to 'us-central1'.
            credentials_path (NotGivenOr[str]): Path to service account credentials JSON file.
                Can be set via argument or `GOOGLE_APPLICATION_CREDENTIALS` environment variable.
            language (Language): Target language for synthesis. Defaults to EN_US.
            audio_ref_path (NotGivenOr[str]): Path to reference audio file for voice cloning.
            ref_text (NotGivenOr[str]): Optional transcription of reference audio for better results.
            word_tokenizer (NotGivenOr[tokenize.WordTokenizer]): Tokenizer for processing text.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        # Handle Google Cloud credentials
        if is_given(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        # Set up project and location
        resolved_project_id = (
            project_id if is_given(project_id) 
            else os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        resolved_location = (
            location if is_given(location)
            else os.environ.get("GOOGLE_CLOUD_LOCATION")
        )

        # Initialize Vertex AI
        try:
            aiplatform.init(project=resolved_project_id, location=resolved_location)
        except Exception as e:
            raise ValueError(f"Failed to initialize Vertex AI: {e}")

        if not is_given(word_tokenizer):
            word_tokenizer = tokenize.basic.WordTokenizer(
                ignore_punctuation=False  # punctuation can help for intonation
            )

        self._opts = _TTSOptions(
            project_id=resolved_project_id,
            location=resolved_location,
            endpoint_id=endpoint_id,
            credentials_path=credentials_path,
            language=language,
            audio_ref_path=audio_ref_path,
            ref_text=ref_text,
            word_tokenizer=word_tokenizer,
        )
        self._streams = weakref.WeakSet[ChunkedStream]()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """
        Synthesize text to speech using Google Vertex AI MARS7.

        Args:
            text (str): Text to synthesize
            conn_options (APIConnectOptions): Connection options

        Returns:
            ChunkedStream: Stream of synthesized audio
        """
        stream = ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close all streams and resources"""
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    """Google Vertex AI MARS7 TTS chunked stream implementation"""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts

    async def _run(self, output_emitter) -> None:
        """Run the TTS synthesis process"""
        request_id = utils.shortuuid()

        try:
            # Load reference audio if provided
            audio_ref_bytes = None
            if is_given(self._opts.audio_ref_path):
                with open(self._opts.audio_ref_path, "rb") as f:
                    audio_ref_bytes = base64.b64encode(f.read()).decode("utf-8")

            # Create the instances payload for MARS7 API
            instances = {
                "text": self._input_text,
                "language": self._opts.language.value,
            }

            # Add reference audio if provided
            if audio_ref_bytes:
                instances["audio_ref"] = audio_ref_bytes
                if is_given(self._opts.ref_text):
                    instances["ref_text"] = self._opts.ref_text

            endpoint = aiplatform.Endpoint(endpoint_name=self._opts.endpoint_id)
            data = {"instances": [instances]}
            
            response = endpoint.raw_predict(
                body=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )

            response_data = json.loads(response.content)
            if "predictions" not in response_data or not response_data["predictions"]:
                raise APIStatusError(
                    message="No predictions returned from MARS7 API",
                    status_code=500,
                    request_id=request_id,
                    body=str(response_data),
                )

            audio_bytes = base64.b64decode(response_data["predictions"][0])

            if audio_bytes:
                # Initialize the output emitter
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=DEFAULT_SAMPLE_RATE,
                    num_channels=1,
                    mime_type="audio/flac",
                )
                
                # Push the raw FLAC audio bytes directly
                output_emitter.push(audio_bytes)
                
                # Flush to complete the emission
                output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            if "credentials" in str(e).lower():
                raise APIConnectionError(f"Authentication failed: {e}") from e
            elif "endpoint" in str(e).lower():
                raise APIStatusError(
                    message=f"Endpoint error: {e}",
                    status_code=404,
                    request_id=request_id,
                    body=None,
                ) from e
            else:
                raise APIConnectionError(f"MARS7 API error: {e}") from e