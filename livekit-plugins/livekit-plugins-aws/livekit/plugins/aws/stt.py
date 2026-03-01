# ruff: noqa: F821 "aws_sdk_transcribe_streaming" already requires Python 3.12+

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
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

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
    TranscriptResultStreamUnknown,
)
from smithy_aws_core.identity import (
    AWSCredentialsIdentity,
    ContainerCredentialsResolver,
    EnvironmentCredentialsResolver,
    IMDSCredentialsResolver,
    StaticCredentialsResolver,
)
from smithy_core.aio.identity import ChainedIdentityResolver
from smithy_http.aio.crt import AWSCRTHTTPClient

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
from livekit.plugins.aws.log import logger
from livekit.plugins.aws.utils import DEFAULT_REGION

if TYPE_CHECKING:
    from smithy_core.aio.interfaces.eventstream import EventPublisher, EventReceiver
_RecognitionIterator = AsyncIterator[rtc.AudioFrame | stt.RecognizeStream._FlushSentinel]


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


class _StaticCredsResolver:
    def __init__(self, creds: Credentials):
        self._identity = AWSCredentialsIdentity(
            access_key_id=creds.access_key_id,
            secret_access_key=creds.secret_access_key,
            session_token=creds.session_token,
        )

    async def get_identity(self, **kwargs: Any) -> AWSCredentialsIdentity:
        return self._identity


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
            ),
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
        raise NotImplementedError(
            "Amazon Transcribe does not support single frame recognition",
        )

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
    _KEEPALIVE_TIMEOUT_S = timedelta(seconds=10).total_seconds()

    def __init__(
        self,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        credentials: Credentials | None = None,
    ) -> None:
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=opts.sample_rate,
        )
        self._opts = opts
        self._http_client = AWSCRTHTTPClient()
        self._credentials = (
            ChainedIdentityResolver(
                resolvers=(
                    StaticCredentialsResolver(),
                    EnvironmentCredentialsResolver(),
                    ContainerCredentialsResolver(http_client=self._http_client),
                    IMDSCredentialsResolver(http_client=self._http_client),
                ),
            )
            if credentials is None
            else _StaticCredsResolver(credentials)
        )
        self._input_closed = False  # Track if input is genuinely done

    async def _run(self) -> None:
        """Main loop: run one streaming session, retry on transient failures.

        Workflow:
        - Reuse one `input_iterator` across reconnects (no dropped frames).
        - Run `_run_single_session()`; on error call `_should_retry()`.
        - Retry or raise; exit on clean completion.

        Raises:
            BaseException: Non-retryable session error.
        """
        input_iterator = aiter(self._input_ch)

        while not self._input_closed:
            logger.info(
                "session start: lang=%s vocab=%s",
                self._opts.language,
                self._opts.vocabulary_name,
            )
            # 1. Spawn the session as a standalone Task
            # This allows us to "join" it without a try/except block
            session_task = asyncio.create_task(self._run_single_session(input_iterator))

            # 2. Wait for completion (Does not raise exception)
            await asyncio.wait([session_task])

            # 3. Inspect the outcome
            exc = session_task.exception()
            if exc is None:
                logger.info("session end")
                break

            # 4. Decision Logic: Retry or Crash?
            if _should_retry(exc):
                logger.warning("retry: %s", exc)
                continue

            logger.error("session fail: %s", exc)
            raise exc

    async def _run_single_session(self, input_iterator: _RecognitionIterator) -> None:
        """Runs one AWS Transcribe streaming session (audio up, transcripts
        down).

        Starts Transcribe, and run `_send_audio_loop()` and `_receive_transcript_loop()`
        concurrently in a TaskGroup.

        Args:
            input_iterator: Stream of LiveKit audio frames (shared across retries).
        """
        config = Config(
            aws_credentials_identity_resolver=self._credentials,
            region=self._opts.region,
        )

        client = TranscribeStreamingClient(config=config)

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
            "enable_partial_results_stabilization": (
                self._opts.enable_partial_results_stabilization
            ),
            "partial_results_stability": self._opts.partial_results_stability,
            "language_model_name": self._opts.language_model_name,
        }
        stream = await client.start_stream_transcription(
            input=StartStreamTranscriptionInput(
                **{
                    key: val
                    for key, val in live_config.items()
                    if is_given(val) and val is not None
                },
            ),
        )
        _, output_stream = await stream.await_output()

        # TaskGroup raises ExceptionGroup if any child fails
        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(self._send_audio_loop(input_iterator, stream.input_stream))
            task_group.create_task(self._receive_transcript_loop(output_stream))

    async def _send_audio_loop(
        self,
        input_iterator: _RecognitionIterator,
        audio_stream: EventPublisher[AudioStream],
    ) -> None:
        """Sends audio frames to AWS; emits keep-alives; closes on end-of-
        input.

        Args:
            input_iterator: Source of frames (rtc.AudioFrame; None means input ended).
            audio_stream: AWS publisher for `AudioEvent` chunks.
        """
        # Prime the first read
        read_task = asyncio.create_task(anext(input_iterator, None))
        pending = {read_task}

        try:
            while True:
                # Efficient wait: returns immediately on data, or after timeout
                done, pending = await asyncio.wait(
                    pending,
                    timeout=self._KEEPALIVE_TIMEOUT_S,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if read_task in done:
                    frame = read_task.result()

                    if frame is None:
                        # if None (from `anext(input_iterator, None)` above)
                        # `input_iterator` has done i.e. audio stream gracefully stopped
                        logger.info("audio end-of-input")
                        self._input_closed = True
                        break

                    if isinstance(frame, rtc.AudioFrame):
                        await audio_stream.send(
                            AudioStreamAudioEvent(
                                value=AudioEvent(audio_chunk=frame.data.tobytes()),
                            ),
                        )

                    # Schedule next read immediately
                    read_task = asyncio.create_task(anext(input_iterator, None))
                    pending = {read_task}
                else:
                    # read_task not in done -> timeout reached
                    # send PCM16 little-endian silent to keep the audio stream alive
                    logger.debug("audio keep-alive")
                    await audio_stream.send(
                        AudioStreamAudioEvent(
                            value=AudioEvent(audio_chunk=b"\x00\x00"),
                        ),
                    )

        finally:
            read_task.cancel()

            # Only signal clean close if we actually finished inputs
            if self._input_closed:
                await audio_stream.send(
                    AudioStreamAudioEvent(value=AudioEvent(audio_chunk=b"")),
                )

    async def _receive_transcript_loop(
        self,
        output_stream: EventReceiver[TranscriptResultStream],
    ) -> None:
        """Reads AWS transcript events and forwards them for emission.

        Args:
            output_stream: AWS receiver for transcript result events.
        """
        logger.debug("recv start")

        async for event in output_stream:
            if isinstance(event, TranscriptResultStreamUnknown):
                continue
            if isinstance(event.value, TranscriptEvent):
                self._process_transcript_event(event.value)

        logger.info("recv end")

    def _process_transcript_event(self, transcript_event: TranscriptEvent) -> None:
        if not transcript_event.transcript or not transcript_event.transcript.results:
            return

        stream = transcript_event.transcript.results
        for resp in stream:
            if resp.start_time is not None and resp.start_time == 0.0:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH),
                )

            if resp.end_time is not None and resp.end_time > 0.0:
                if resp.is_partial:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            alternatives=[
                                self._streaming_recognize_response_to_speech_data(resp),
                            ],
                        ),
                    )

                else:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[
                                self._streaming_recognize_response_to_speech_data(resp),
                            ],
                        ),
                    )

            if not resp.is_partial:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH),
                )

    def _streaming_recognize_response_to_speech_data(
        self,
        resp: Result,
    ) -> stt.SpeechData:
        confidence = 0.0
        if resp.alternatives and (items := resp.alternatives[0].items):
            confidence = items[0].confidence or 0.0

        return stt.SpeechData(
            language=Language(resp.language_code or self._opts.language),
            start_time=(resp.start_time or 0.0) + self.start_time_offset,
            end_time=(resp.end_time or 0.0) + self.start_time_offset,
            text=(resp.alternatives[0].transcript or "") if resp.alternatives else "",
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
                if item.content
            ]
            if resp.alternatives and resp.alternatives[0].items
            else None,
        )


def _should_retry(exc: BaseException) -> bool:
    """Returns True if the error is retryable (handles ExceptionGroup from
    TaskGroup).

    Args:
      exc: Session exception (may be an ExceptionGroup).

    Returns:
      bool: Whether to reconnect and retry.
    """
    # Errors we expect from AWS timeouts or network drops
    retry_types = (BadRequestException,)

    if isinstance(exc, ExceptionGroup):
        # Check if the group ONLY contains retryable errors
        matched, remainder = exc.split(retry_types)
        return matched is not None and remainder is None

    return isinstance(exc, retry_types)
