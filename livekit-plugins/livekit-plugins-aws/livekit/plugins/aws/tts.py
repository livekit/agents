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
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

from ._utils import (
    TTS_LANGUAGE,
    TTS_OUTPUT_FORMAT,
    TTS_SPEECH_ENGINE,
    _get_aws_credentials,
)

TTS_NUM_CHANNELS: int = 1


@dataclass
class _TTSOptions:
    # https://docs.aws.amazon.com/polly/latest/dg/API_SynthesizeSpeech.html
    voice: str | None
    output_format: TTS_OUTPUT_FORMAT
    speech_engine: TTS_SPEECH_ENGINE
    speech_region: str | None
    sample_rate: int
    language: TTS_LANGUAGE | None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str | None = "Ruth",
        language: TTS_LANGUAGE | None = None,
        output_format: TTS_OUTPUT_FORMAT = "pcm",
        speech_engine: TTS_SPEECH_ENGINE = "generative",
        sample_rate: int = 16000,
        speech_region: str = "us-east-1",
        api_key: str | None = None,
        api_secret: str | None = None,
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
            output_format(TTS_OUTPUT_FORMAT, optional): The format in which the returned output will be encoded. Defaults to "pcm".
            sample_rate(int, optional): The audio frequency specified in Hz. Defaults to 16000.
            speech_engine(TTS_SPEECH_ENGINE, optional): The engine to use for the synthesis. Defaults to "generative".
            speech_region(str, optional): The region to use for the synthesis. Defaults to "us-east-1".
            api_key(str, optional): AWS access key id.
            api_secret(str, optional): AWS secret access key.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=sample_rate,
            num_channels=TTS_NUM_CHANNELS,
        )

        self._api_key, self._api_secret = _get_aws_credentials(
            api_key, api_secret, speech_region
        )

        self._opts = _TTSOptions(
            voice=voice,
            output_format=output_format,
            speech_engine=speech_engine,
            speech_region=speech_region,
            language=language,
            sample_rate=sample_rate,
        )
        self._session = session or get_session()

    def get_client(self):
        """Returns a client creator context."""
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
        segment_id: str | None = None,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            text=text,
            conn_options=conn_options,
            opts=self._opts,
            get_client=self.get_client,
            segment_id=segment_id,
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        text: str,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        get_client: Callable[[], Any],
        segment_id: str | None = None,
    ) -> None:
        super().__init__(tts=tts, input_text=text, conn_options=conn_options)
        self._opts = opts
        self._get_client = get_client
        self._segment_id = segment_id or utils.shortuuid()

    async def _run(self):
        request_id = utils.shortuuid()

        try:
            async with self._get_client() as client:
                params = {
                    "Text": self._input_text,
                    "OutputFormat": self._opts.output_format,
                    "Engine": self._opts.speech_engine,
                    "VoiceId": self._opts.voice,
                    "TextType": "text",
                    "SampleRate": str(self._opts.sample_rate),
                    "LanguageCode": self._opts.language,
                }

                response = await client.synthesize_speech(**_strip_nones(params))

                if "AudioStream" in response:
                    decoder = utils.codecs.Mp3StreamDecoder()
                    async with response["AudioStream"] as resp:
                        async for data, _ in resp.content.iter_chunks():
                            if self._opts.output_format == "mp3":
                                frames = decoder.decode_chunk(data)
                                for frame in frames:
                                    self._event_ch.send_nowait(
                                        tts.SynthesizedAudio(
                                            request_id=request_id,
                                            segment_id=self._segment_id,
                                            frame=frame,
                                        )
                                    )
                            else:
                                self._event_ch.send_nowait(
                                    tts.SynthesizedAudio(
                                        request_id=request_id,
                                        segment_id=self._segment_id,
                                        frame=rtc.AudioFrame(
                                            data=data,
                                            sample_rate=self._opts.sample_rate,
                                            num_channels=1,
                                            samples_per_channel=len(data) // 2,
                                        ),
                                    )
                                )

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


def _strip_nones(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}
