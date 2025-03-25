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
from dataclasses import dataclass
from typing import Any, Callable

import aiohttp
from aiobotocore.session import AioSession, get_session

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .models import TTS_LANGUAGE, TTS_SPEECH_ENGINE
from .utils import _strip_nones, get_aws_credentials

TTS_NUM_CHANNELS: int = 1
DEFAULT_SPEECH_ENGINE: TTS_SPEECH_ENGINE = "generative"
DEFAULT_SPEECH_REGION = "us-east-1"
DEFAULT_VOICE = "Ruth"
DEFAULT_SAMPLE_RATE = 16000


@dataclass
class _TTSOptions:
    # https://docs.aws.amazon.com/polly/latest/dg/API_SynthesizeSpeech.html
    voice: NotGivenOr[str]
    speech_engine: NotGivenOr[TTS_SPEECH_ENGINE]
    speech_region: str
    sample_rate: int
    language: NotGivenOr[TTS_LANGUAGE | str]


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[TTS_LANGUAGE | str] = NOT_GIVEN,
        speech_engine: NotGivenOr[TTS_SPEECH_ENGINE] = NOT_GIVEN,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        speech_region: NotGivenOr[str] = DEFAULT_SPEECH_REGION,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        session: AioSession | None = None,
    ) -> None:
        """
        Create a new instance of AWS Polly TTS.

        ``api_key``  and ``api_secret`` must be set to your AWS Access key id and secret access key, either using the argument or by setting the
        ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY`` environmental variables.

        See https://docs.aws.amazon.com/polly/latest/dg/API_SynthesizeSpeech.html for more details on the the AWS Polly TTS.

        Args:
            Voice (TTSModels, optional): Voice ID to use for the synthesis. Defaults to "Ruth".
            language (TTS_LANGUAGE, optional): language code for the Synthesize Speech request. This is only necessary if using a bilingual voice, such as Aditi, which can be used for either Indian English (en-IN) or Hindi (hi-IN).
            sample_rate(int, optional): The audio frequency specified in Hz. Defaults to 16000.
            speech_engine(TTS_SPEECH_ENGINE, optional): The engine to use for the synthesis. Defaults to "generative".
            speech_region(str, optional): The region to use for the synthesis. Defaults to "us-east-1".
            api_key(str, optional): AWS access key id.
            api_secret(str, optional): AWS secret access key.
        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=sample_rate,
            num_channels=TTS_NUM_CHANNELS,
        )

        self._api_key, self._api_secret, self._speech_region = get_aws_credentials(
            api_key, api_secret, speech_region
        )

        self._opts = _TTSOptions(
            voice=voice,
            speech_engine=speech_engine,
            speech_region=self._speech_region,
            language=language,
            sample_rate=sample_rate,
        )
        self._session = session or get_session()

    def _get_client(self):
        return self._session.create_client(
            "polly",
            region_name=self._opts.speech_region,
            aws_access_key_id=self._api_key,
            aws_secret_access_key=self._api_secret,
        )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            text=text,
            conn_options=conn_options,
            opts=self._opts,
            get_client=self._get_client,
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        text: str,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        opts: _TTSOptions,
        get_client: Callable[[], Any],
    ) -> None:
        super().__init__(tts=tts, input_text=text, conn_options=conn_options)
        self._opts = opts
        self._get_client = get_client
        self._segment_id = utils.shortuuid()

    async def _run(self):
        request_id = utils.shortuuid()

        try:
            async with self._get_client() as client:
                params = {
                    "Text": self._input_text,
                    "OutputFormat": "mp3",
                    "Engine": self._opts.speech_engine
                    if is_given(self._opts.speech_engine)
                    else DEFAULT_SPEECH_ENGINE,
                    "VoiceId": self._opts.voice if is_given(self._opts.voice) else DEFAULT_VOICE,
                    "TextType": "text",
                    "SampleRate": str(self._opts.sample_rate),
                    "LanguageCode": self._opts.language if is_given(self._opts.language) else None,
                }
                response = await client.synthesize_speech(**_strip_nones(params))
                if "AudioStream" in response:
                    decoder = utils.codecs.AudioStreamDecoder(
                        sample_rate=self._opts.sample_rate,
                        num_channels=1,
                    )

                    # Create a task to push data to the decoder
                    async def push_data():
                        try:
                            async with response["AudioStream"] as resp:
                                async for data, _ in resp.content.iter_chunks():
                                    decoder.push(data)
                        finally:
                            decoder.end_input()

                    # Start pushing data to the decoder
                    push_task = asyncio.create_task(push_data())

                    try:
                        # Create emitter and process decoded frames
                        emitter = tts.SynthesizedAudioEmitter(
                            event_ch=self._event_ch,
                            request_id=request_id,
                            segment_id=self._segment_id,
                        )
                        async for frame in decoder:
                            emitter.push(frame)
                        emitter.flush()
                        await push_task
                    finally:
                        await utils.aio.gracefully_cancel(push_task)

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
