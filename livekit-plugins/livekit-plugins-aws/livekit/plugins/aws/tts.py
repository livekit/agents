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

from dataclasses import dataclass, replace
from typing import cast

import aioboto3  # type: ignore
import botocore  # type: ignore
import botocore.exceptions  # type: ignore
from aiobotocore.config import AioConfig  # type: ignore

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
    tts,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .models import TTSLanguages, TTSSpeechEngine, TTSTextType
from .utils import _strip_nones

DEFAULT_SPEECH_ENGINE: TTSSpeechEngine = "generative"
DEFAULT_VOICE = "Ruth"
DEFAULT_TEXT_TYPE: TTSTextType = "text"


@dataclass
class _TTSOptions:
    # https://docs.aws.amazon.com/polly/latest/dg/API_SynthesizeSpeech.html
    voice: str
    speech_engine: TTSSpeechEngine
    region: str | None
    sample_rate: int
    language: TTSLanguages | str | None
    text_type: TTSTextType


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "Ruth",
        language: NotGivenOr[TTSLanguages | str] = NOT_GIVEN,
        speech_engine: TTSSpeechEngine = "generative",
        text_type: TTSTextType = "text",
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

        See https://docs.aws.amazon.com/polly/latest/dg/API_SynthesizeSpeech.html for more details on the AWS Polly TTS.

        Args:
            voice (TTSModels, optional): Voice ID to use for the synthesis. Defaults to "Ruth".
            language (TTSLanguages, optional): language code for the Synthesize Speech request. This is only necessary if using a bilingual voice, such as Aditi, which can be used for either Indian English (en-IN) or Hindi (hi-IN).
            speech_engine(TTSSpeechEngine, optional): The engine to use for the synthesis. Defaults to "generative".
            text_type(TTSTextType, optional): Type of text to synthesize. Use "ssml" for SSML-enhanced text. Defaults to "text".
            sample_rate(int, optional): The audio frequency specified in Hz. Defaults to 16000.
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
            num_channels=1,
        )
        self._session = session or aioboto3.Session(
            aws_access_key_id=api_key if is_given(api_key) else None,
            aws_secret_access_key=api_secret if is_given(api_secret) else None,
            region_name=region if is_given(region) else None,
        )

        self._opts = _TTSOptions(
            voice=voice,
            speech_engine=speech_engine,
            text_type=text_type,
            region=region or None,
            language=language or None,
            sample_rate=sample_rate,
        )

    @property
    def model(self) -> str:
        return self._opts.speech_engine

    @property
    def provider(self) -> str:
        return "Amazon Polly"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, text=text, conn_options=conn_options)

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        speech_engine: NotGivenOr[TTSSpeechEngine] = NOT_GIVEN,
        text_type: NotGivenOr[TTSTextType] = NOT_GIVEN,
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language
        if is_given(speech_engine):
            self._opts.speech_engine = cast(TTSSpeechEngine, speech_engine)
        if is_given(text_type):
            self._opts.text_type = cast(TTSTextType, text_type)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, *, tts: TTS, text: str, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> None:
        super().__init__(tts=tts, input_text=text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
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
                            "TextType": self._opts.text_type,
                            "SampleRate": str(self._opts.sample_rate),
                            "LanguageCode": self._opts.language,
                        }
                    )
                )

                if "AudioStream" in response:
                    output_emitter.initialize(
                        request_id=response["ResponseMetadata"]["RequestId"],
                        sample_rate=self._opts.sample_rate,
                        num_channels=1,
                        mime_type="audio/mp3",
                    )

                    async with response["AudioStream"] as resp:
                        async for data, _ in resp.content.iter_chunks():
                            output_emitter.push(data)
        except botocore.exceptions.ConnectTimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError() from e
