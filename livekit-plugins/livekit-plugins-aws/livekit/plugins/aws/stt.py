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

import aioboto3
from amazon_transcribe.auth import StaticCredentialResolver
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.model import Result, TranscriptEvent

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, stt, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .utils import get_aws_async_session, validate_aws_credentials

DEFAULT_REGION = "us-east-1"
REFRESH_INTERVAL = 1800


@dataclass
class STTOptions:
    region: str
    sample_rate: int
    language: str
    encoding: str
    vocabulary_name: NotGivenOr[str]
    session_id: NotGivenOr[str]
    vocab_filter_method: NotGivenOr[str]
    vocab_filter_name: NotGivenOr[str]
    show_speaker_label: NotGivenOr[bool]
    enable_channel_identification: NotGivenOr[bool]
    number_of_channels: NotGivenOr[int]
    enable_partial_results_stabilization: NotGivenOr[bool]
    partial_results_stability: NotGivenOr[str]
    language_model_name: NotGivenOr[str]


class STT(stt.STT):
    def __init__(
        self,
        *,
        region: str = "us-east-1",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = 48000,
        language: str = "en-US",
        encoding: str = "pcm",
        vocabulary_name: NotGivenOr[str] = NOT_GIVEN,
        session_id: NotGivenOr[str] = NOT_GIVEN,
        vocab_filter_method: NotGivenOr[str] = NOT_GIVEN,
        vocab_filter_name: NotGivenOr[str] = NOT_GIVEN,
        show_speaker_label: NotGivenOr[bool] = NOT_GIVEN,
        enable_channel_identification: NotGivenOr[bool] = NOT_GIVEN,
        number_of_channels: NotGivenOr[int] = NOT_GIVEN,
        enable_partial_results_stabilization: NotGivenOr[bool] = NOT_GIVEN,
        partial_results_stability: NotGivenOr[str] = NOT_GIVEN,
        language_model_name: NotGivenOr[str] = NOT_GIVEN,
        session: aioboto3.Session | None = None,
        refresh_interval: NotGivenOr[int] = NOT_GIVEN,
    ):
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self._session = session
        self._api_key = api_key if is_given(api_key) else None
        self._api_secret = api_secret if is_given(api_secret) else None
        self._region = region if is_given(region) else None
        if not self._session:
            validate_aws_credentials(api_key=self._api_key, api_secret=self._api_secret)

        self._config = STTOptions(
            region=self._region,
            language=language,
            sample_rate=sample_rate,
            encoding=encoding,
            vocabulary_name=vocabulary_name,
            session_id=session_id,
            vocab_filter_method=vocab_filter_method,
            vocab_filter_name=vocab_filter_name,
            show_speaker_label=show_speaker_label,
            enable_channel_identification=enable_channel_identification,
            number_of_channels=number_of_channels,
            enable_partial_results_stabilization=enable_partial_results_stabilization,
            partial_results_stability=partial_results_stability,
            language_model_name=language_model_name,
        )
        self._pool = utils.ConnectionPool[TranscribeStreamingClient](
            connect_cb=self._create_client,
            max_session_duration=refresh_interval
            if is_given(refresh_interval)
            else REFRESH_INTERVAL,
        )

    async def _create_client(self) -> TranscribeStreamingClient:
        session = self._session or await get_aws_async_session(
            region=self._region,
            api_key=self._api_key,
            api_secret=self._api_secret,
        )
        creds = await session.get_credentials()
        frozen_credentials = await creds.get_frozen_credentials()
        return TranscribeStreamingClient(
            region=self._config.region,
            credential_resolver=StaticCredentialResolver(
                access_key_id=frozen_credentials.access_key,
                secret_access_key=frozen_credentials.secret_key,
                session_token=frozen_credentials.token,
            ),
        )

    async def aclose(self) -> None:
        await self._pool.aclose()
        await super().aclose()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Amazon Transcribe does not support single frame recognition")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        return SpeechStream(
            stt=self,
            pool=self._pool,
            conn_options=conn_options,
            opts=self._config,
        )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        stt: STT,
        opts: STTOptions,
        pool: utils.ConnectionPool[TranscribeStreamingClient],
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._pool = pool

    async def _run(self) -> None:
        async with self._pool.connection() as client:
            live_config = {
                "language_code": self._opts.language,
                "media_sample_rate_hz": self._opts.sample_rate,
                "media_encoding": self._opts.encoding,
                "vocabulary_name": self._opts.vocabulary_name,
                "session_id": self._opts.session_id,
                "vocab_filter_method": self._opts.vocab_filter_method,
                "vocab_filter_name": self._opts.vocab_filter_name,
                "show_speaker_label": self._opts.show_speaker_label,
                "enable_channel_identification": self._opts.enable_channel_identification,
                "number_of_channels": self._opts.number_of_channels,
                "enable_partial_results_stabilization": self._opts.enable_partial_results_stabilization,  # noqa: E501
                "partial_results_stability": self._opts.partial_results_stability,
                "language_model_name": self._opts.language_model_name,
            }
            filtered_config = {k: v for k, v in live_config.items() if v and is_given(v)}
            stream = await client.start_stream_transcription(**filtered_config)

            @utils.log_exceptions(logger=logger)
            async def input_generator():
                async for frame in self._input_ch:
                    if isinstance(frame, rtc.AudioFrame):
                        await stream.input_stream.send_audio_event(audio_chunk=frame.data.tobytes())
                await stream.input_stream.end_stream()

            @utils.log_exceptions(logger=logger)
            async def handle_transcript_events():
                async for event in stream.output_stream:
                    if isinstance(event, TranscriptEvent):
                        self._process_transcript_event(event)

            tasks = [
                asyncio.create_task(input_generator()),
                asyncio.create_task(handle_transcript_events()),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)

    def _process_transcript_event(self, transcript_event: TranscriptEvent):
        stream = transcript_event.transcript.results
        for resp in stream:
            if resp.start_time and resp.start_time == 0.0:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )

            if resp.end_time and resp.end_time > 0.0:
                if resp.is_partial:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            alternatives=[_streaming_recognize_response_to_speech_data(resp)],
                        )
                    )

                else:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[_streaming_recognize_response_to_speech_data(resp)],
                        )
                    )

            if not resp.is_partial:
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))


def _streaming_recognize_response_to_speech_data(resp: Result) -> stt.SpeechData:
    data = stt.SpeechData(
        language="en-US",
        start_time=resp.start_time if resp.start_time else 0.0,
        end_time=resp.end_time if resp.end_time else 0.0,
        text=resp.alternatives[0].transcript if resp.alternatives else "",
    )

    return data
