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
from dataclasses import dataclass, replace


from aiobotocore.config import AioConfig
import botocore
import aioboto3
import aiohttp

import botocore.exceptions
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

from .models import TTSLanguages, TTSSpeechEngine
from .utils import _strip_nones, get_aws_async_session

NUM_CHANNELS: int = 1
DEFAULT_SPEECH_ENGINE: TTSSpeechEngine = "generative"
DEFAULT_VOICE = "Ruth"


@dataclass
class _TTSOptions:
    # https://docs.aws.amazon.com/polly/latest/dg/API_SynthesizeSpeech.html
    voice: str
    speech_engine: TTSSpeechEngine
    region: str | None
    sample_rate: int
    language: TTSLanguages | str | None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "Ruth",
        language: NotGivenOr[TTSLanguages | str] = NOT_GIVEN,
        speech_engine: TTSSpeechEngine = "generative",
        sample_rate: int = 16000,
        region: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        session: aioboto3.Session | None = None,
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
            region(str, optional): The region to use for the synthesis. Defaults to "us-east-1".
            api_key(str, optional): AWS access key id.
            api_secret(str, optional): AWS secret access key.
            session(aioboto3.Session, optional): Optional aioboto3 session to use.
        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        self._session = session or get_aws_async_session(
            api_key=api_key or None, api_secret=api_secret or None, region=region or None
        )
        self._opts = _TTSOptions(
            voice=voice,
            speech_engine=speech_engine,
            region=region or None,
            language=language or None,
            sample_rate=sample_rate,
        )

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, *, tts: TTS, text: str, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> None:
        super().__init__(tts=tts, input_text=text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.SynthesizedAudioEmitter):
        try:
            config = AioConfig(
                connect_timeout=self._conn_options.timeout,
                read_timeout=10,
                retries={"mode": "standard", "total_max_attempts": 1},
            )
            async with self._tts._session.client("polly", config=config) as client:  # type: ignore
                response = await client.synthesize_speech(
                    **_strip_nones(
                        {
                            "Text": self._input_text,
                            "OutputFormat": "mp3",
                            "Engine": self._opts.speech_engine,
                            "VoiceId": self._opts.voice,
                            "TextType": "text",
                            "SampleRate": str(self._opts.sample_rate),
                            "LanguageCode": self._opts.language,
                        }
                    )
                )

                if "AudioStream" in response:
                    output_emitter.start(
                        request_id=response["ResponseMetadata"]["RequestId"],
                        sample_rate=self._opts.sample_rate,
                        num_channels=NUM_CHANNELS,
                    )

                    async with response["AudioStream"] as resp:
                        async for data, _ in resp.content.iter_chunks():
                            output_emitter.push(data)

                    output_emitter.flush()
        except botocore.exceptions.ConnectTimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError() from e
