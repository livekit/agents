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
import concurrent.futures
import contextlib
import os
from dataclasses import dataclass
from typing import Any

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    Language,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from .log import logger
from .utils import DEFAULT_REGION

try:
    from aws_sdk_transcribe_streaming.client import TranscribeStreamingClient
    from aws_sdk_transcribe_streaming.config import Config
    from aws_sdk_transcribe_streaming.models import (
        AudioEvent,
        AudioStream,
        AudioStreamAudioEvent,
        BadRequestException,
        Result,
        StartStreamTranscriptionInput,
        TranscriptEvent,
        TranscriptResultStream,
    )
    from smithy_aws_core.identity import (
        AWSCredentialsIdentity,
        ContainerCredentialsResolver,
        EnvironmentCredentialsResolver,
        IMDSCredentialsResolver,
        StaticCredentialsResolver,
    )
    from smithy_core.aio.identity import ChainedIdentityResolver
    from smithy_core.aio.interfaces.eventstream import EventPublisher, EventReceiver
    from smithy_http.aio.crt import AWSCRTHTTPClient

    _AWS_SDK_AVAILABLE = True
except ImportError:
    _AWS_SDK_AVAILABLE = False


@dataclass
class Credentials:
    access_key_id: str
    secret_access_key: str
    session_token: str | None = None


@dataclass
class STTOptions:
    sample_rate: int
    language: Language
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
    region: str


class STT(stt.STT):
    def __init__(
        self,
        *,
        region: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = 24000,
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
        credentials: NotGivenOr[Credentials] = NOT_GIVEN,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="word",
                offline_recognize=False,
            )
        )

        if not _AWS_SDK_AVAILABLE:
            raise ImportError(
                "The 'aws_sdk_transcribe_streaming' package is not installed. "
                "This implementation requires Python 3.12+ and the 'aws_sdk_transcribe_streaming' dependency."
            )

        if not is_given(region):
            region = os.getenv("AWS_REGION") or DEFAULT_REGION

        self._config = STTOptions(
            language=Language(language),
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
            region=region,
        )

        self._credentials = credentials if is_given(credentials) else None

    @property
    def model(self) -> str:
        return (
            self._config.language_model_name
            if is_given(self._config.language_model_name)
            else "unknown"
        )

    @property
    def provider(self) -> str:
        return "Amazon Transcribe"

    async def aclose(self) -> None:
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
            conn_options=conn_options,
            opts=self._config,
            credentials=self._credentials,
        )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        credentials: Credentials | None = None,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._credentials = credentials
        self._http_client = AWSCRTHTTPClient()

    async def _run(self) -> None:
        while True:
            config_kwargs: dict[str, Any] = {"region": self._opts.region}
            if self._credentials:
                # Use a credentials resolver for explicit credentials
                # for some reason, Config with direct values doesn't work
                class StaticCredsResolver:
                    def __init__(self, creds: Credentials):
                        self._identity = AWSCredentialsIdentity(
                            access_key_id=creds.access_key_id,
                            secret_access_key=creds.secret_access_key,
                            session_token=creds.session_token,
                        )

                    async def get_identity(self, **kwargs: Any) -> AWSCredentialsIdentity:
                        return self._identity

                config_kwargs["aws_credentials_identity_resolver"] = StaticCredsResolver(
                    self._credentials
                )
            else:
                config_kwargs["aws_credentials_identity_resolver"] = ChainedIdentityResolver(
                    resolvers=(
                        StaticCredentialsResolver(),
                        EnvironmentCredentialsResolver(),
                        ContainerCredentialsResolver(http_client=self._http_client),
                        IMDSCredentialsResolver(http_client=self._http_client),
                    )
                )

            client: TranscribeStreamingClient = TranscribeStreamingClient(
                config=Config(**config_kwargs)
            )

            live_config = {
                "language_code": self._opts.language,
                "media_sample_rate_hertz": self._opts.sample_rate,
                "media_encoding": self._opts.encoding,
                "vocabulary_name": self._opts.vocabulary_name,
                "session_id": self._opts.session_id,
                "vocab_filter_method": self._opts.vocab_filter_method,
                "vocab_filter_name": self._opts.vocab_filter_name,
                "show_speaker_label": self._opts.show_speaker_label,
                "enable_channel_identification": self._opts.enable_channel_identification,
                "number_of_channels": self._opts.number_of_channels,
                "enable_partial_results_stabilization": self._opts.enable_partial_results_stabilization,
                "partial_results_stability": self._opts.partial_results_stability,
                "language_model_name": self._opts.language_model_name,
            }
            filtered_config = {k: v for k, v in live_config.items() if v and is_given(v)}

            tasks: list[asyncio.Task[Any]] = []

            try:
                stream = await client.start_stream_transcription(
                    input=StartStreamTranscriptionInput(**filtered_config)
                )

                # Get the output stream
                _, output_stream = await stream.await_output()

                async def input_generator(
                    audio_stream: EventPublisher[AudioStream],
                ) -> None:
                    try:
                        async for frame in self._input_ch:
                            if isinstance(frame, rtc.AudioFrame):
                                await audio_stream.send(
                                    AudioStreamAudioEvent(
                                        value=AudioEvent(audio_chunk=frame.data.tobytes())
                                    )
                                )
                    finally:
                        # Send empty frame to close (required by AWS Transcribe)
                        try:
                            await audio_stream.send(
                                AudioStreamAudioEvent(value=AudioEvent(audio_chunk=b""))
                            )
                        except Exception:
                            pass
                        finally:
                            with contextlib.suppress(Exception):
                                await audio_stream.close()

                async def handle_transcript_events(
                    output_stream: EventReceiver[TranscriptResultStream],
                ) -> None:
                    try:
                        async for event in output_stream:
                            if isinstance(event.value, TranscriptEvent):
                                self._process_transcript_event(event.value)
                    except BadRequestException as e:
                        if (
                            e.message
                            and "complete signal was sent without the preceding empty frame"
                            in e.message
                        ):
                            # This can happen during cancellation if the empty frame wasn't sent in time
                            logger.warning(
                                "AWS Transcribe stream closed with empty frame error (this is usually harmless)"
                            )
                        else:
                            raise
                    except concurrent.futures.InvalidStateError:
                        logger.warning(
                            "AWS Transcribe stream closed unexpectedly (InvalidStateError)"
                        )
                        pass

                tasks = [
                    asyncio.create_task(input_generator(stream.input_stream)),
                    asyncio.create_task(handle_transcript_events(output_stream)),
                ]
                gather_future = asyncio.gather(*tasks)

                await asyncio.shield(gather_future)
            except BadRequestException as e:
                if e.message and e.message.startswith("Your request timed out"):
                    # AWS times out after 15s of inactivity, this tends to happen
                    # at the end of the session, when the input is gone, we'll ignore it and
                    # just treat it as a silent retry
                    logger.info("restarting transcribe session")
                    continue
                else:
                    raise e
            finally:
                if tasks:
                    # Close input stream first
                    await utils.aio.gracefully_cancel(tasks[0])

                    # Wait for output stream to close cleanly
                    try:
                        await asyncio.wait_for(tasks[1], timeout=3.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        await utils.aio.gracefully_cancel(tasks[1])

                # Ensure gather future is retrieved to avoid "exception never retrieved"
                with contextlib.suppress(Exception):
                    await gather_future

    def _process_transcript_event(self, transcript_event: TranscriptEvent) -> None:
        if not transcript_event.transcript or not transcript_event.transcript.results:
            return

        stream = transcript_event.transcript.results
        for resp in stream:
            if resp.start_time is not None and resp.start_time == 0.0:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )

            if resp.end_time is not None and resp.end_time > 0.0:
                if resp.is_partial:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            alternatives=[self._streaming_recognize_response_to_speech_data(resp)],
                        )
                    )

                else:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[self._streaming_recognize_response_to_speech_data(resp)],
                        )
                    )

            if not resp.is_partial:
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    def _streaming_recognize_response_to_speech_data(self, resp: Result) -> stt.SpeechData:
        confidence = 0.0
        if resp.alternatives and (items := resp.alternatives[0].items):
            confidence = items[0].confidence or 0.0

        return stt.SpeechData(
            language=Language(resp.language_code or self._opts.language),
            start_time=(resp.start_time or 0.0) + self.start_time_offset,
            end_time=(resp.end_time or 0.0) + self.start_time_offset,
            text=resp.alternatives[0].transcript if resp.alternatives else "",
            confidence=confidence,
            words=[
                TimedString(
                    text=item.content,
                    start_time=item.start_time + self.start_time_offset,
                    end_time=item.end_time + self.start_time_offset,
                    start_time_offset=self.start_time_offset,
                    confidence=item.confidence or 0.0,
                )
                for item in resp.alternatives[0].items
            ]
            if resp.alternatives and resp.alternatives[0].items
            else None,
        )
