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
import weakref
from dataclasses import dataclass

import aiohttp

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

from .models import Gender

# Constants
API_BASE_URL = "https://client.camb.ai/apis"
NUM_CHANNELS = 1
DEFAULT_SAMPLE_RATE = 24000


@dataclass
class Voice:
    """Camb AI voice representation"""

    id: int
    voice_name: str


@dataclass
class _TTSOptions:
    """Internal options for Camb AI TTS"""

    api_key: str
    voice_id: int
    language: int
    gender: Gender
    age: int
    base_url: str
    sample_rate: int
    word_tokenizer: tokenize.WordTokenizer


class TTS(tts.TTS):
    """Camb AI Text-to-Speech implementation"""

    def __init__(
        self,
        *,
        voice_id: int,
        language: int = 1,
        gender: Gender = Gender.NOT_KNOWN,
        age: int = 1,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Camb AI TTS.

        Args:
            voice_id (int): Voice ID from Camb AI.
            language (int): Language ID. Defaults to 1 (English).
            gender (Gender): Gender enum value. Defaults to NOT_KNOWN.
            age (int): Age parameter. Defaults to 0.
            api_key (NotGivenOr[str]): Camb AI API key. Can be set via argument or
                `CAMB_API_KEY` environment variable.
            base_url (NotGivenOr[str]): Custom base URL for the API. Optional.
            word_tokenizer (NotGivenOr[tokenize.WordTokenizer]): Tokenizer for processing text.
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests.
                Optional.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        camb_api_key = api_key if is_given(api_key) else os.environ.get("CAMB_API_KEY")
        if not camb_api_key:
            raise ValueError(
                "Camb AI API key is required, either as argument or set "
                "CAMB_API_KEY environmental variable"
            )

        if not is_given(word_tokenizer):
            word_tokenizer = tokenize.basic.WordTokenizer(
                ignore_punctuation=False  # punctuation can help for intonation
            )

        self._opts = _TTSOptions(
            voice_id=voice_id,
            language=language,
            gender=gender,
            age=age,
            api_key=camb_api_key,
            base_url=base_url if is_given(base_url) else API_BASE_URL,
            sample_rate=self.sample_rate,
            word_tokenizer=word_tokenizer,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[ChunkedStream]()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def list_voices(self) -> list[Voice]:
        """
        List available voices from Camb AI.

        Returns:
            List[Voice]: List of available voices
        """
        async with self._ensure_session().get(
            f"{self._opts.base_url}/list-voices",
            headers={"x-api-key": self._opts.api_key},
        ) as resp:
            if resp.status != 200:
                raise APIStatusError(
                    message=f"Failed to list voices: {resp.reason}",
                    status_code=resp.status,
                    request_id=None,
                    body=await resp.text(),
                )

            data = await resp.json()
            voices = []
            # The API returns a list of voices directly, not in a payload object
            if isinstance(data, list):
                for voice in data:
                    voices.append(
                        Voice(
                            id=voice["id"],
                            voice_name=voice["voice_name"],
                        )
                    )
            elif isinstance(data, dict) and "payload" in data:
                for voice in data["payload"]:
                    voices.append(
                        Voice(
                            id=voice["id"],
                            voice_name=voice["voice_name"],
                        )
                    )
            return voices

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """
        Synthesize text to speech using Camb AI.

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
            session=self._ensure_session(),
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
    """Camb AI TTS chunked stream implementation"""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        conn_options: APIConnectOptions,
        session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._session = session

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the TTS synthesis process"""
        request_id = utils.shortuuid()

        # Create the payload for Camb AI TTS API
        payload = {
            "text": self._input_text,
            "voice_id": self._opts.voice_id,
            "language": self._opts.language,
            "gender": self._opts.gender.value,
            "age": self._opts.age,
        }

        try:
            # Step 1: Create the TTS task
            async with self._session.post(
                f"{self._opts.base_url}/tts",
                headers={
                    "x-api-key": self._opts.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                if resp.status != 200:
                    raise APIStatusError(
                        message=f"Failed to create TTS task: {resp.reason}",
                        status_code=resp.status,
                        request_id=request_id,
                        body=await resp.text(),
                    )

                data = await resp.json()
                # The API returns task_id directly in the response, not in a payload object
                task_id = data.get("task_id")

                if not task_id:
                    raise APIStatusError(
                        message="No task_id returned from Camb AI API",
                        status_code=resp.status,
                        request_id=request_id,
                        body=str(data),
                    )

            # Step 2: Poll for task completion
            run_id = None
            max_attempts = 30
            attempt = 0

            while attempt < max_attempts:
                async with self._session.get(
                    f"{self._opts.base_url}/tts/{task_id}",
                    headers={"x-api-key": self._opts.api_key},
                ) as status_resp:
                    if status_resp.status != 200:
                        raise APIStatusError(
                            message=f"Failed to check TTS task status: {status_resp.reason}",
                            status_code=status_resp.status,
                            request_id=request_id,
                            body=await status_resp.text(),
                        )

                    status_data = await status_resp.json()
                    status = status_data.get("status")

                    if status == "SUCCESS":
                        run_id = status_data.get("run_id")
                        break
                    elif status == "FAILED":
                        raise APIStatusError(
                            message="TTS task failed",
                            status_code=500,
                            request_id=request_id,
                            body=str(status_data),
                        )

                    # Wait before checking again
                    await asyncio.sleep(1)
                    attempt += 1

            if not run_id:
                raise APITimeoutError("Timed out waiting for TTS task to complete")

            # Step 3: Get the audio result
            async with self._session.get(
                f"{self._opts.base_url}/tts-result/{run_id}",
                headers={"x-api-key": self._opts.api_key},
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as audio_resp:
                if audio_resp.status != 200:
                    raise APIStatusError(
                        message=f"Failed to get TTS result: {audio_resp.reason}",
                        status_code=audio_resp.status,
                        request_id=request_id,
                        body=await audio_resp.text(),
                    )

                # Create decoder for the audio stream
                decoder = utils.codecs.AudioStreamDecoder(
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                )


                # Process the audio data
                audio_data = await audio_resp.read()
                decoder.push(audio_data)
                decoder.end_input()

                # Emit the audio frames
                async for frame in decoder:
                    output_emitter.push(frame)

                output_emitter.flush()
                await decoder.aclose()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e
