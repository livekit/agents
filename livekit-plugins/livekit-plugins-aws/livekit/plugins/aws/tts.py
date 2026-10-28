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
from typing import TYPE_CHECKING, cast

import botocore
import botocore.exceptions
from aiobotocore.config import AioConfig
from aiobotocore.session import AioSession

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
    LanguageCode,
    tts,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .models import TTSLanguages, TTSSpeechEngine, TTSTextType
from .utils import resolve_session

if TYPE_CHECKING:
    from types_aiobotocore_polly.literals import (
        LanguageCodeType,
        VoiceIdType,
    )
    from types_aiobotocore_polly.type_defs import SynthesizeSpeechInputTypeDef

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
    language: LanguageCode | None
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
        session: AioSession | None = None,
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
            session(AioSession, optional): Optional aiobotocore session to use.
        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._session = resolve_session(session)
        if session is None:
            if is_given(api_key) and api_key and is_given(api_secret) and api_secret:
                self._session.set_credentials(api_key, api_secret)
            if is_given(region) and region:
                self._session.set_config_variable("region", region)

        self._opts = _TTSOptions(
            voice=voice,
            speech_engine=speech_engine,
            text_type=text_type,
            region=region or None,
            language=LanguageCode(language) if is_given(language) and language else None,
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
        return ChunkedStream(
            tts=self,
            text=text,
            options=self._opts,
            session=self._session,
            conn_options=conn_options,
        )

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
            self._opts.language = LanguageCode(language)
        if is_given(speech_engine):
            self._opts.speech_engine = speech_engine
        if is_given(text_type):
            self._opts.text_type = text_type


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        text: str,
        options: _TTSOptions,
        session: AioSession,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(tts=tts, input_text=text, conn_options=conn_options)
        self._opts = replace(options)
        self._session = session

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            config = AioConfig(
                connect_timeout=self._conn_options.connect_timeout,
                read_timeout=self._conn_options.read_timeout,
                retries={"mode": "standard", "total_max_attempts": 1},
            )
            # Other services in this SDK overload set have no generated return stub.
            async with self._session.create_client("polly", config=config) as client:  # pyright: ignore[reportUnknownMemberType]
                # The service accepts new identifiers before its generated literal stubs catch up.
                request: SynthesizeSpeechInputTypeDef = {
                    "Text": self.input_text,
                    "OutputFormat": "mp3",
                    "Engine": self._opts.speech_engine,
                    "VoiceId": cast("VoiceIdType", self._opts.voice),
                    "TextType": self._opts.text_type,
                    "SampleRate": str(self._opts.sample_rate),
                }
                if self._opts.language is not None:
                    request["LanguageCode"] = cast("LanguageCodeType", self._opts.language)
                response = await client.synthesize_speech(**request)

                if "AudioStream" in response:
                    output_emitter.initialize(
                        request_id=response["ResponseMetadata"]["RequestId"],
                        sample_rate=self._opts.sample_rate,
                        num_channels=1,
                        mime_type="audio/mp3",
                    )

                    async with response["AudioStream"] as resp:
                        async for data in resp.iter_chunks():
                            output_emitter.push(data)
        except botocore.exceptions.ConnectTimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError() from e
