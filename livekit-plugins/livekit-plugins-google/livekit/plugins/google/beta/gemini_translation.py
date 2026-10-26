"""Gemini Live translation (``gemini-3.5-live-translate-preview``) as a :class:`TranslationModel`.

A translation session on the Live API takes no instructions, tools or chat history: it only
translates the audio streamed into it, continuously, into the language its ``translation_config``
names. The setup is fixed for a connection's life, so retargeting reconnects with the new
language.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import AsyncIterable, Callable, Iterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from typing import Literal

import google.auth.credentials
import google.auth.exceptions
from google.auth._default_async import default_async
from google.genai import Client as GenAIClient, types
from google.genai.errors import APIError, ClientError
from google.genai.live import AsyncSession
from livekit import rtc
from livekit.agents import APIConnectionError, utils
from livekit.agents.beta.translation import (
    InputTranscript,
    OutputSpeechStopped,
    OutputTranscript,
    TranslatedAudioFrame,
    TranslationCapabilities,
    TranslationModel,
    TranslationModelError,
    TranslationModelSession,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import audio as audio_utils, is_given

from ..log import logger
from ..version import __version__

GeminiTranslationModels = Literal["gemini-3.5-live-translate-preview"]

DEFAULT_TRANSLATION_MODEL = "gemini-3.5-live-translate-preview"
# the model is served from the global endpoint only on Vertex AI
DEFAULT_VERTEX_LOCATION = "global"
# a transcript the server does not mark finished ends once its pieces have been quiet this long
DEFAULT_TRANSCRIPT_SEGMENT_TIMEOUT = 1.0

INPUT_AUDIO_SAMPLE_RATE = 16000
INPUT_AUDIO_CHANNELS = 1
OUTPUT_AUDIO_SAMPLE_RATE = 24000
OUTPUT_AUDIO_CHANNELS = 1

# websocket close codes for a setup the server refuses: invalid argument, policy violation
_SETUP_REJECTED_CLOSE_CODES = (1007, 1008)


# client errors a retry can get past: rate limiting, and a request cancelled in transit
_RETRYABLE_CLIENT_CODES = frozenset({429, 499})
_TRANSIENT_AUTH_ERRORS = (
    google.auth.exceptions.TransportError,
    google.auth.exceptions.TimeoutError,
)


def _is_setup_error(e: Exception) -> bool:
    """Whether a connection that never opened failed on its setup, which a retry repeats."""
    if isinstance(e, ClientError):
        return e.code not in _RETRYABLE_CLIENT_CODES
    if isinstance(e, APIError):
        return e.code in _SETUP_REJECTED_CLOSE_CODES
    if isinstance(e, google.auth.exceptions.GoogleAuthError):
        # bad credentials stay bad, but a refresh lost on the network is worth another try
        transient = (
            e.retryable
            or isinstance(e, _TRANSIENT_AUTH_ERRORS)
            or isinstance(e.__cause__, _TRANSIENT_AUTH_ERRORS)
        )
        return not transient
    return isinstance(e, (ValueError, TypeError))


@dataclass
class _TranslationOptions:
    model: str
    target_language: str
    echo_target_language: bool
    input_audio_transcription: types.AudioTranscriptionConfig | None
    output_audio_transcription: types.AudioTranscriptionConfig | None
    api_key: str | None
    vertexai: bool
    project: str | None
    location: str | None
    credentials: google.auth.credentials.Credentials | None
    http_options: types.HttpOptions | None
    transcript_segment_timeout: float
    conn_options: APIConnectOptions


class GeminiTranslationModel(TranslationModel):
    """Speech-to-speech translation with Gemini 3.5 Live Translate.

    The source language is detected by the model, so only the language to translate into is
    set. Run it with :class:`~livekit.agents.beta.translation.TranslationSession`.

    Example:
        ```python
        session = TranslationSession(
            model=google.beta.GeminiTranslationModel(target_language="es")
        )
        await session.start(room=ctx.room)
        ```
    """

    def __init__(
        self,
        *,
        target_language: str,
        model: GeminiTranslationModels | str = DEFAULT_TRANSLATION_MODEL,
        echo_target_language: bool = False,
        input_audio_transcription: NotGivenOr[types.AudioTranscriptionConfig | None] = NOT_GIVEN,
        output_audio_transcription: NotGivenOr[types.AudioTranscriptionConfig | None] = NOT_GIVEN,
        api_key: str | None = None,
        vertexai: bool | None = None,
        project: str | None = None,
        location: str | None = None,
        credentials: google.auth.credentials.Credentials | None = None,
        http_options: types.HttpOptions | None = None,
        transcript_segment_timeout: float = DEFAULT_TRANSCRIPT_SEGMENT_TIMEOUT,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            target_language: The language to translate into, as a BCP-47 code such as ``"es"``.
            model: The translation model.
            echo_target_language: Whether speech already in the target language is repeated.
                When off, the model stays silent for it.
            input_audio_transcription: Transcription of the speech heard, in the language it
                was spoken. On by default; ``None`` turns it off.
            output_audio_transcription: Transcription of the translation spoken. On by
                default; ``None`` turns it off.
            api_key: Gemini API key; read from ``GOOGLE_API_KEY`` when omitted.
            vertexai: Whether to use Vertex AI; read from ``GOOGLE_GENAI_USE_VERTEXAI`` when
                omitted.
            project: Google Cloud project for Vertex AI; read from ``GOOGLE_CLOUD_PROJECT``
                or the default credentials when omitted.
            location: Vertex AI location; defaults to ``global``, the only one serving the
                model.
            credentials: Google Cloud credentials for Vertex AI.
            http_options: HTTP options for the genai client.
            transcript_segment_timeout: Seconds without a transcript piece that end a
                transcript the server did not mark finished.
            conn_options: Retry and timeout settings for the connection.

        Raises:
            ValueError: If no API key or Vertex AI project can be found.
        """
        if not is_given(input_audio_transcription):
            input_audio_transcription = types.AudioTranscriptionConfig()
        if not is_given(output_audio_transcription):
            output_audio_transcription = types.AudioTranscriptionConfig()

        super().__init__(
            capabilities=TranslationCapabilities(
                input_transcription=input_audio_transcription is not None,
                output_transcription=output_audio_transcription is not None,
                turn_reporting=False,
                output_segmentation=False,
                retargetable=True,
            ),
            target_language=target_language,
            source_language=None,
        )

        use_vertexai = (
            vertexai
            if vertexai is not None
            else os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "0").lower() in ["true", "1"]
        )
        gemini_api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        gcp_project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        gcp_location: str | None = location or DEFAULT_VERTEX_LOCATION

        if use_vertexai:
            if not gcp_project:
                _, gcp_project = default_async(  # type: ignore
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            if not gcp_project:
                raise ValueError(
                    "Project is required for VertexAI via project kwarg or "
                    "GOOGLE_CLOUD_PROJECT environment variable"
                )
            gemini_api_key = None  # VertexAI does not require an API key
        else:
            gcp_project = None
            gcp_location = None
            if credentials is not None:
                logger.warning(
                    "'credentials' is only applicable to VertexAI and will be ignored for the "
                    "Gemini API"
                )
                credentials = None
            if not gemini_api_key:
                raise ValueError(
                    "API key is required for Google API either via api_key or GOOGLE_API_KEY "
                    "environment variable"
                )

        self._opts = _TranslationOptions(
            model=model,
            target_language=target_language,
            echo_target_language=echo_target_language,
            input_audio_transcription=input_audio_transcription,
            output_audio_transcription=output_audio_transcription,
            api_key=gemini_api_key,
            vertexai=use_vertexai,
            project=gcp_project,
            location=gcp_location,
            credentials=credentials,
            http_options=http_options,
            transcript_segment_timeout=transcript_segment_timeout,
            conn_options=conn_options,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Vertex AI" if self._opts.vertexai else "Gemini"

    def update_options(self, *, target_language: str) -> None:
        super().update_options(target_language=target_language)
        self._opts.target_language = target_language

    def session(self) -> GeminiTranslationSession:
        return GeminiTranslationSession(self)

    async def aclose(self) -> None:
        pass


class _TranscriptSegmenter:
    """Ends a transcript when the server marks it finished, or once its pieces go quiet.

    Each callback gets the piece and the transcript so far, so a consumer can take either.
    """

    def __init__(
        self,
        timeout: float,
        on_text: Callable[[str, str, str | None], None],
        on_final: Callable[[str, str | None], None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._timeout = timeout
        self._on_text = on_text
        self._on_final = on_final
        self._loop = loop
        self._timer: asyncio.TimerHandle | None = None
        self._language: str | None = None
        self._text = ""

    def push(self, text: str, *, language: str | None, finished: bool) -> None:
        if self._timer is None:
            # gemini opens a transcript with a space, which does not belong at its start
            text = text.lstrip()

        if language:
            self._language = language

        if text:
            self._text += text
            self._on_text(text, self._text, self._language)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = self._loop.call_later(self._timeout, self.flush)

        if finished:
            self.flush()

    def flush(self) -> None:
        """End the open transcript, if one is open."""
        if self._timer is None:
            return

        self._timer.cancel()
        self._timer = None
        text, self._text = self._text, ""
        language, self._language = self._language, None
        self._on_final(text, language)


class GeminiTranslationSession(TranslationModelSession[None]):
    """One run of Gemini Live translation, reconnecting when its setup has to change.

    An ``input_transcript`` carries the transcript so far, as an STT interim does, and the
    final one the whole of it. An ``output_transcript`` carries the text added since the last,
    as the translation's transcript is a delta stream, and the one that ends it carries none,
    only ``is_final``.
    """

    def __init__(self, translation_model: GeminiTranslationModel) -> None:
        super().__init__(translation_model)
        self._model: GeminiTranslationModel = translation_model
        self._opts = replace(translation_model._opts)
        self._loop = asyncio.get_event_loop()

        self._msg_ch = utils.aio.Chan[types.Blob]()
        self._audio_ch = utils.aio.Chan[TranslatedAudioFrame | OutputSpeechStopped]()
        self._input_resampler: rtc.AudioResampler | None = None
        # 100ms chunks, as the model recommends
        self._bstream = audio_utils.AudioByteStream(
            INPUT_AUDIO_SAMPLE_RATE,
            INPUT_AUDIO_CHANNELS,
            samples_per_channel=INPUT_AUDIO_SAMPLE_RATE // 10,
        )
        # whether audio went out since the model last marked the end of a translation
        self._speaking = False
        # set to replace the connection, e.g. once the target language changed
        self._restart = asyncio.Event()
        self._num_retries = 0
        # the language the current connection was set up with; audio still arriving from an
        # old connection after a retarget is in that one, not the new target
        self._output_language = self._opts.target_language
        # audio whose send failed, sent first on the next connection so no speech is lost
        self._unsent_audio: types.Blob | None = None

        self._input_segmenter = _TranscriptSegmenter(
            self._opts.transcript_segment_timeout,
            on_text=lambda _, text, language: self.emit(
                "input_transcript", InputTranscript(text=text, is_final=False, language=language)
            ),
            on_final=lambda text, language: self.emit(
                "input_transcript", InputTranscript(text=text, is_final=True, language=language)
            ),
            loop=self._loop,
        )
        self._output_segmenter = _TranscriptSegmenter(
            self._opts.transcript_segment_timeout,
            on_text=lambda delta, _, language: self.emit(
                "output_transcript",
                OutputTranscript(
                    text=delta, is_final=False, language=language or self._output_language
                ),
            ),
            on_final=lambda _, language: self.emit(
                "output_transcript",
                OutputTranscript(
                    text="", is_final=True, language=language or self._output_language
                ),
            ),
            loop=self._loop,
        )

        http_options = self._opts.http_options or types.HttpOptions(
            timeout=int(self._opts.conn_options.timeout * 1000)
        )
        if not http_options.headers:
            http_options.headers = {}
        http_options.headers["x-goog-api-client"] = f"livekit-agents/{__version__}"

        self._client = GenAIClient(
            api_key=self._opts.api_key,
            vertexai=self._opts.vertexai,
            project=self._opts.project,
            location=self._opts.location,
            credentials=self._opts.credentials,
            http_options=http_options,
        )

        self._main_atask = asyncio.create_task(
            self._main_task(), name="GeminiTranslationSession._main_task"
        )

    @property
    def audio_stream(self) -> AsyncIterable[TranslatedAudioFrame | OutputSpeechStopped]:
        return self._audio_ch

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        for f in self._resample_audio(frame):
            self._send_audio(self._bstream.write(f.data.tobytes()))

    def _send_audio(self, frames: list[rtc.AudioFrame]) -> None:
        for nf in frames:
            with contextlib.suppress(utils.aio.channel.ChanClosed):
                self._msg_ch.send_nowait(
                    types.Blob(
                        data=nf.data.tobytes(),
                        mime_type=f"audio/pcm;rate={INPUT_AUDIO_SAMPLE_RATE}",
                    )
                )

    def _flush_input_audio(self) -> None:
        """Send the tail of the input still held short of a full chunk."""
        if self._input_resampler is not None:
            for f in self._input_resampler.flush():
                self._send_audio(self._bstream.write(f.data.tobytes()))
            self._input_resampler = None

        self._send_audio(self._bstream.flush())

    def clear_audio(self) -> None:
        # what the server sent is already here; drop what the consumer has not read yet
        with contextlib.suppress(utils.aio.channel.ChanEmpty, utils.aio.channel.ChanClosed):
            while True:
                self._audio_ch.recv_nowait()

    def update_options(self, *, target_language: str) -> None:
        if target_language == self._opts.target_language:
            return

        self._opts.target_language = target_language
        # the setup is fixed for a connection's life, so the new language needs a new one
        self._restart.set()

    async def aclose(self) -> None:
        if not self._closing:
            self._closing = True
            self._flush_input_audio()
            # the send task drains what is queued, then ends the connection
            self._msg_ch.close()

        # a failure that ended the session was already emitted as an error
        with contextlib.suppress(Exception):
            await self._main_atask

        try:
            await self._client.aio.aclose()
        except Exception:
            logger.warning("failed to close the genai client", exc_info=True)

    def _build_connect_config(self) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            input_audio_transcription=self._opts.input_audio_transcription,
            output_audio_transcription=self._opts.output_audio_transcription,
            translation_config=types.TranslationConfig(
                target_language_code=self._opts.target_language,
                echo_target_language=self._opts.echo_target_language,
            ),
        )

    def _connect(
        self, config: types.LiveConnectConfig
    ) -> AbstractAsyncContextManager[AsyncSession]:
        return self._client.aio.live.connect(model=self._opts.model, config=config)

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                # input audio changed to a different sample rate
                yield from self._input_resampler.flush()
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != INPUT_AUDIO_SAMPLE_RATE
            or frame.num_channels != INPUT_AUDIO_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=INPUT_AUDIO_SAMPLE_RATE,
                num_channels=INPUT_AUDIO_CHANNELS,
            )

        if self._input_resampler:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        max_retries = self._opts.conn_options.max_retry
        try:
            while not self._msg_ch.closed:
                self._restart.clear()
                connected = False
                try:
                    language = self._opts.target_language
                    async with self._connect(self._build_connect_config()) as session:
                        connected = True
                        self._output_language = language
                        await self._run_session(session)

                except Exception as e:
                    if self._msg_ch.closed:
                        break

                    if (
                        (not connected and _is_setup_error(e))
                        or max_retries == 0
                        or self._num_retries == max_retries
                    ):
                        self._emit_error(e, recoverable=False)
                        raise APIConnectionError(
                            message="Gemini Live translation connection failed"
                        ) from e

                    self._emit_error(e, recoverable=True)
                    retry_interval = self._opts.conn_options._interval_for_retry(self._num_retries)
                    logger.warning(
                        f"Gemini Live translation connection failed, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={"attempt": self._num_retries, "max_retries": max_retries},
                    )
                    await asyncio.sleep(retry_interval)
                    self._num_retries += 1
                finally:
                    # what the old connection was saying will not be finished on the next one
                    self._input_segmenter.flush()
                    self._output_segmenter.flush()
                    self._end_output_speech()
        finally:
            self._audio_ch.close()

    async def _run_session(self, session: AsyncSession) -> None:
        tasks = [
            asyncio.create_task(self._send_task(session), name="gemini-translation-send"),
            asyncio.create_task(self._recv_task(session), name="gemini-translation-recv"),
            asyncio.create_task(self._restart.wait(), name="gemini-translation-restart"),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            await utils.aio.cancel_and_wait(*tasks)

    async def _send_task(self, session: AsyncSession) -> None:
        if (blob := self._unsent_audio) is not None:
            self._unsent_audio = None
            await self._send_or_keep(session, blob)

        async for blob in self._msg_ch:
            await self._send_or_keep(session, blob)

    async def _send_or_keep(self, session: AsyncSession, blob: types.Blob) -> None:
        try:
            await session.send_realtime_input(audio=blob)
        except (Exception, asyncio.CancelledError):
            # a failed or interrupted send means the connection is going; keep the audio for
            # the next one
            self._unsent_audio = blob
            raise

    async def _recv_task(self, session: AsyncSession) -> None:
        while True:
            # receive() ends at each turn_complete; the session goes on past it
            async for response in session.receive():
                # reset only once the server answers, so a connection that drops straight
                # after opening still runs out of retries
                self._num_retries = 0
                if response.server_content:
                    self._handle_server_content(response.server_content)
                if response.go_away:
                    logger.debug(
                        "Gemini Live translation is closing the connection, reconnecting",
                        extra={"time_left": response.go_away.time_left},
                    )
                    return

    def _handle_server_content(self, server_content: types.LiveServerContent) -> None:
        if model_turn := server_content.model_turn:
            for part in model_turn.parts or []:
                if part.inline_data and isinstance(data := part.inline_data.data, bytes):
                    self._handle_output_audio(data)

        if (transcription := server_content.input_transcription) is not None:
            self._input_segmenter.push(
                transcription.text or "",
                language=transcription.language_code,
                finished=bool(transcription.finished),
            )

        if (transcription := server_content.output_transcription) is not None:
            self._output_segmenter.push(
                transcription.text or "",
                language=transcription.language_code,
                finished=bool(transcription.finished),
            )

        if server_content.generation_complete or server_content.turn_complete:
            self._end_output_speech()

    def _handle_output_audio(self, data: bytes) -> None:
        samples_per_channel = len(data) // (2 * OUTPUT_AUDIO_CHANNELS)
        if samples_per_channel == 0:
            return

        frame = rtc.AudioFrame(
            data=data[: samples_per_channel * 2 * OUTPUT_AUDIO_CHANNELS],
            sample_rate=OUTPUT_AUDIO_SAMPLE_RATE,
            num_channels=OUTPUT_AUDIO_CHANNELS,
            samples_per_channel=samples_per_channel,
        )
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._audio_ch.send_nowait(
                TranslatedAudioFrame(frame=frame, language=self._output_language)
            )
            self._speaking = True

    def _end_output_speech(self) -> None:
        """Mark the end of the translation the model has been speaking, if it spoke one."""
        if not self._speaking:
            return

        self._speaking = False
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._audio_ch.send_nowait(OutputSpeechStopped())

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            TranslationModelError(
                timestamp=time.time(),
                label=self._model.label,
                error=error,
                recoverable=recoverable,
            ),
        )
