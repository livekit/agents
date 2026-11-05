from __future__ import annotations

import asyncio
import os
import weakref
from dataclasses import dataclass

import httpx  # mistralai raises httpx.TimeoutException from its httpx v1 client

from livekit import rtc
from livekit.agents import (
    VAD,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    TokenUsage,
    VADEventType,
    VADStream,
    stt,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TranscriptChunk
from mistralai.client import Mistral
from mistralai.client.errors import HTTPValidationError, SDKError
from mistralai.client.models import (
    RealtimeTranscriptionError,
    RealtimeTranscriptionSessionCreated,
    TranscriptionStreamDone,
    TranscriptionStreamLanguage,
    TranscriptionStreamTextDelta,
)
from mistralai.extra.realtime import RealtimeConnection, RealtimeTranscription

from ._http import close_client, request_timeout_ms
from .log import logger
from .models import STTModels


def _is_realtime(model: str) -> bool:
    return "realtime" in model


DEFAULT_MODEL: STTModels = "voxtral-mini-latest"

SAMPLE_RATE: int = 16000
NUM_CHANNELS: int = 1


@dataclass
class _STTOptions:
    model: STTModels | str
    language: LanguageCode | None
    context_bias: list[str] | None
    target_streaming_delay_ms: int | None


class STT(stt.STT):
    def __init__(
        self,
        client: Mistral | None = None,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        context_bias: NotGivenOr[list[str]] = NOT_GIVEN,
        target_streaming_delay_ms: NotGivenOr[int] = NOT_GIVEN,
        vad: VAD | None = None,
    ):
        """
        Create a new instance of MistralAI STT.

        Args:
            client: Optional pre-configured MistralAI client instance.
            api_key: Your Mistral AI API key. If not provided, will use the MISTRAL_API_KEY environment variable.
            model: The Mistral AI model to use for transcription, default is batch "voxtral-mini-latest".
            language: The optional language code to use for better transcription accuracy if language is already known (e.g., "fr" for French).
                Only used with batch models.
            context_bias: Up to 100 words or phrases to guide the model toward good spelling or names or domain-specific vocabulary.
                Only used with batch models.
            target_streaming_delay_ms: Target streaming delay in milliseconds for realtime mode. Only used with realtime models.
            vad: Voice Activity Detector used to trigger audio flush for realtime models (which lack server-side endpointing).
                When not provided, Silero VAD is auto-loaded with default settings. Only used with realtime models.
        """
        resolved_model = model if is_given(model) else DEFAULT_MODEL
        is_realtime = _is_realtime(resolved_model)
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=is_realtime,
                interim_results=is_realtime,
                aligned_transcript=False,
                offline_recognize=not is_realtime,
            )
        )
        self._opts = _STTOptions(
            model=resolved_model,
            language=LanguageCode(language) if is_given(language) else None,
            context_bias=context_bias if is_given(context_bias) else None,
            target_streaming_delay_ms=target_streaming_delay_ms
            if is_given(target_streaming_delay_ms)
            else None,
        )

        if is_realtime and vad is None:
            try:
                from livekit.plugins.silero import VAD as SileroVAD

                vad = SileroVAD.load()
            except ImportError as e:
                raise ImportError(
                    "livekit-plugins-silero is required for Voxtral realtime models (no server-side endpointing)."
                ) from e
        self._vad = vad

        mistral_api_key = api_key if is_given(api_key) else os.environ.get("MISTRAL_API_KEY")
        if not client and not mistral_api_key:
            raise ValueError("Mistral AI API key is required. Set MISTRAL_API_KEY or pass api_key")
        self._owns_client = client is None
        self._client = client or Mistral(api_key=mistral_api_key)
        self._reconnect_events: weakref.WeakSet[asyncio.Event] = weakref.WeakSet()
        self._pool = utils.ConnectionPool[RealtimeConnection](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )

    async def _connect_ws(self, conn_options: APIConnectOptions) -> RealtimeConnection:
        rt = RealtimeTranscription(self._client.sdk_configuration)
        http_headers = None
        cfg = self._client.sdk_configuration
        client_headers = getattr(cfg.async_client, "headers", None) or getattr(
            cfg.client, "headers", None
        )
        if client_headers:
            http_headers = dict(client_headers)
        return await asyncio.wait_for(
            rt.connect(
                model=self._opts.model,
                target_streaming_delay_ms=self._opts.target_streaming_delay_ms,
                http_headers=http_headers,
            ),
            timeout=conn_options.connect_timeout,
        )

    async def _close_ws(self, conn: RealtimeConnection) -> None:
        await conn.close()

    def prewarm(self) -> None:
        if _is_realtime(self._opts.model):
            self._pool.prewarm()

    async def aclose(self) -> None:
        try:
            await self._pool.aclose()
        finally:
            if self._owns_client:
                await close_client(self._client)

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "MistralAI"

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        context_bias: NotGivenOr[list[str]] = NOT_GIVEN,
        target_streaming_delay_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Update the STT options.

        Args:
            language: The optional language code to use for better transcription accuracy if language is already known (e.g., "fr" for French).
                Only used with batch models.
            context_bias: Up to 100 words or phrases to guide the model toward good spelling or names or domain-specific vocabulary.
                Only used with batch models.
            target_streaming_delay_ms: Target streaming delay in milliseconds for realtime mode. Only used with realtime models.
        """
        if is_given(language):
            self._opts.language = LanguageCode(language)
        if is_given(context_bias):
            self._opts.context_bias = context_bias
        if is_given(target_streaming_delay_ms):
            self._opts.target_streaming_delay_ms = target_streaming_delay_ms
            self._pool.invalidate()
            for event in self._reconnect_events:
                event.set()

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        timeout_ms = request_timeout_ms(conn_options)
        try:
            if is_given(language):
                self._opts.language = LanguageCode(language)
            data = rtc.combine_audio_frames(buffer).to_wav_bytes()

            resp = await self._client.audio.transcriptions.complete_async(
                model=self._opts.model,
                file={"content": data, "file_name": "audio.wav"},
                language=self._opts.language if self._opts.language else None,
                context_bias=self._opts.context_bias if self._opts.context_bias else None,
                timestamp_granularities=["segment"] if not self._opts.language else None,
                timeout_ms=timeout_ms,
                retries=None,
            )

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=resp.text,
                        language=LanguageCode(resp.language)
                        if resp.language
                        else self._opts.language or LanguageCode(""),
                        start_time=(resp.segments[0].start or 0.0) if resp.segments else 0.0,
                        end_time=(resp.segments[-1].end or 0.0) if resp.segments else 0.0,
                        words=[
                            TranscriptChunk(
                                text=segment.text,
                                start_time=segment.start if segment.start is not None else None,
                                end_time=segment.end if segment.end is not None else None,
                            )
                            for segment in resp.segments
                        ]
                        if resp.segments
                        else None,
                    ),
                ],
            )

        except (TimeoutError, httpx.TimeoutException) as e:
            raise APITimeoutError() from e
        except (SDKError, HTTPValidationError) as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.headers.get("x-request-id"),
                body=e.body,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        reconnect_event = asyncio.Event()
        stream = SpeechStream(
            stt=self,
            pool=self._pool,
            conn_options=conn_options,
            language=language,
            vad_instance=self._vad,
            reconnect_event=reconnect_event,
        )
        self._reconnect_events.add(reconnect_event)
        return stream


class SpeechStream(stt.RecognizeStream):
    """Realtime speech recognition stream for Voxtral Realtime."""

    def __init__(
        self,
        *,
        stt: STT,
        pool: utils.ConnectionPool[RealtimeConnection],
        reconnect_event: asyncio.Event,
        conn_options: APIConnectOptions,
        language: NotGivenOr[str] = NOT_GIVEN,
        vad_instance: VAD | None = None,
    ) -> None:
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=SAMPLE_RATE,
        )
        self._pool = pool
        self._vad = vad_instance
        self._reconnect_event = reconnect_event
        self._request_id = ""
        self._speaking = False
        self._detected_language = LanguageCode(language) if is_given(language) else LanguageCode("")

    async def _send_task(
        self,
        connection: RealtimeConnection,
        vad_stream: VADStream | None,
    ) -> None:
        samples_per_chunk = SAMPLE_RATE // 20  # 50ms chunks
        audio_bstream = utils.media.AudioByteStream(
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            samples_per_channel=samples_per_chunk,
        )

        async for data in self.input:
            if isinstance(data, self.Flush):
                for frame in audio_bstream.flush():
                    await connection.send_audio(frame.data.tobytes())
                await connection.flush_audio()
            else:
                if vad_stream is not None:
                    vad_stream.push_frame(data)

                for frame in audio_bstream.push(data.data.tobytes()):
                    await connection.send_audio(frame.data.tobytes())

        if vad_stream is not None:
            vad_stream.end_input()
        await connection.end_audio()

    async def _vad_task(
        self,
        vad_stream: VADStream,
        connection: RealtimeConnection,
    ) -> None:
        async for ev in vad_stream:
            if ev.type == VADEventType.START_OF_SPEECH:
                if not self._speaking:
                    self._speaking = True
                    self.send(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
            elif ev.type == VADEventType.END_OF_SPEECH:
                # force Mistral to finalize the current speech segment
                await connection.flush_audio()
                if self._speaking:
                    self._speaking = False
                    self.send(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    async def _recv_task(self, connection: RealtimeConnection) -> None:
        current_text = ""

        async for event in connection.events():
            if isinstance(event, RealtimeTranscriptionSessionCreated):
                if event.session:
                    self._request_id = event.session.request_id

            elif isinstance(event, TranscriptionStreamLanguage):
                if event.audio_language:
                    self._detected_language = LanguageCode(event.audio_language)

            elif isinstance(event, TranscriptionStreamTextDelta):
                current_text += event.text
                self.send(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[
                            stt.SpeechData(
                                text=current_text,
                                language=self._detected_language,
                            )
                        ],
                    )
                )

            elif isinstance(event, TranscriptionStreamDone):
                current_text = ""
                words = (
                    [
                        TranscriptChunk(
                            text=seg.text,
                            start_time=(
                                seg.start + self.start_time_offset
                                if seg.start is not None
                                else None
                            ),
                            end_time=(
                                seg.end + self.start_time_offset if seg.end is not None else None
                            ),
                        )
                        for seg in event.segments
                    ]
                    if event.segments
                    else None
                )

                self.send(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[
                            stt.SpeechData(
                                text=event.text,
                                language=LanguageCode(event.language)
                                if event.language
                                else self._detected_language,
                                words=words,
                            )
                        ],
                    )
                )

                audio_seconds = event.usage.prompt_audio_seconds  # unset when not measured
                self.send(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.RECOGNITION_USAGE,
                        request_id=self._request_id,
                        recognition_usage=stt.RecognitionUsage(
                            audio_duration=audio_seconds
                            if isinstance(audio_seconds, int)
                            else None,
                            tokens=TokenUsage(
                                input_tokens=event.usage.prompt_tokens or 0,
                                output_tokens=event.usage.completion_tokens or 0,
                            ),
                        ),
                    )
                )

            elif isinstance(event, RealtimeTranscriptionError):
                raise APIStatusError(
                    message=str(event.error.message), status_code=event.error.code, body=event.error
                )

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        try:
            while True:
                async with self._pool.connection(conn_options=self._conn_options) as connection:
                    vad_stream = self._vad.stream() if self._vad else None

                    tasks = [
                        asyncio.create_task(self._send_task(connection, vad_stream)),
                        asyncio.create_task(self._recv_task(connection)),
                    ]
                    if vad_stream is not None:
                        tasks.append(asyncio.create_task(self._vad_task(vad_stream, connection)))

                    tasks_group = asyncio.gather(*tasks)
                    wait_reconnect = asyncio.create_task(self._reconnect_event.wait())

                    try:
                        done, _ = await asyncio.wait(
                            (tasks_group, wait_reconnect),
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        for task in done:
                            if task != wait_reconnect:
                                task.result()

                        if wait_reconnect not in done:
                            break

                        self._reconnect_event.clear()
                    finally:
                        await utils.aio.gracefully_cancel(*tasks, wait_reconnect)
                        tasks_group.cancel()
                        tasks_group.exception()
                        if vad_stream is not None:
                            await vad_stream.aclose()

        except (APIStatusError, APITimeoutError, APIConnectionError):
            raise
        except (TimeoutError, httpx.TimeoutException) as e:
            raise APITimeoutError() from e
        except (SDKError, HTTPValidationError) as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.headers.get("x-request-id"),
                body=e.body,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e
