from __future__ import annotations

import asyncio
import os
import weakref
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
    vad,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString
from mistralai.client import Mistral
from mistralai.client.errors import SDKError
from mistralai.client.models import (
    RealtimeTranscriptionError,
    RealtimeTranscriptionSessionCreated,
    TranscriptionStreamDone,
    TranscriptionStreamLanguage,
    TranscriptionStreamTextDelta,
)
from mistralai.extra.realtime import RealtimeConnection, RealtimeTranscription

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
        vad: vad.VAD | None = None,
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
        self._client = client or Mistral(api_key=mistral_api_key)
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()
        self._pool = utils.ConnectionPool[RealtimeConnection](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )

    async def _connect_ws(self, timeout: float) -> RealtimeConnection:
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
            timeout=timeout,
        )

    async def _close_ws(self, conn: RealtimeConnection) -> None:
        await conn.close()
        ws = conn._websocket
        if ws.keepalive_task is not None:
            ws.keepalive_task.cancel()

    def prewarm(self) -> None:
        if _is_realtime(self._opts.model):
            self._pool.prewarm()

    async def aclose(self) -> None:
        await self._pool.aclose()

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
            for stream in self._streams:
                stream._reconnect_event.set()

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
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
            )

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=resp.text,
                        language=LanguageCode(resp.language)
                        if resp.language
                        else self._opts.language or LanguageCode(""),
                        start_time=resp.segments[0].start if resp.segments else 0,
                        end_time=resp.segments[-1].end if resp.segments else 0,
                        words=[
                            TimedString(
                                text=segment.text,
                                start_time=segment.start,
                                end_time=segment.end,
                            )
                            for segment in resp.segments
                        ]
                        if resp.segments
                        else None,
                    ),
                ],
            )

        except SDKError as e:
            if e.status_code in (408, 504):
                raise APITimeoutError() from e
            raise APIStatusError(e.message, status_code=e.status_code, body=e.body) from e
        except Exception as e:
            raise APIConnectionError() from e

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        stream = SpeechStream(
            stt=self,
            pool=self._pool,
            conn_options=conn_options,
            language=language,
            vad_instance=self._vad,
        )
        self._streams.add(stream)
        return stream


class SpeechStream(stt.RecognizeStream):
    """Realtime speech recognition stream for Voxtral Realtime."""

    def __init__(
        self,
        *,
        stt: STT,
        pool: utils.ConnectionPool[RealtimeConnection],
        conn_options: APIConnectOptions,
        language: NotGivenOr[str] = NOT_GIVEN,
        vad_instance: vad.VAD | None = None,
    ) -> None:
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=SAMPLE_RATE,
        )
        self._pool = pool
        self._opts = stt._opts
        self._vad = vad_instance
        self._reconnect_event = asyncio.Event()
        self._request_id = ""
        self._audio_duration = 0.0
        self._speaking = False
        self._detected_language = LanguageCode(language) if is_given(language) else LanguageCode("")

    async def _send_task(
        self,
        connection: RealtimeConnection,
        vad_stream: vad.VADStream | None,
    ) -> None:
        samples_per_chunk = SAMPLE_RATE // 20  # 50ms chunks
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            samples_per_channel=samples_per_chunk,
        )

        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                for frame in audio_bstream.flush():
                    await connection.send_audio(frame.data.tobytes())
                await connection.flush_audio()
            else:
                if vad_stream is not None:
                    vad_stream.push_frame(data)

                data_bytes = data.data.tobytes()
                bytes_per_second = SAMPLE_RATE * NUM_CHANNELS * 2
                self._audio_duration += len(data_bytes) / bytes_per_second
                for frame in audio_bstream.write(data.data.tobytes()):
                    await connection.send_audio(frame.data.tobytes())

        if vad_stream is not None:
            vad_stream.end_input()
        await connection.end_audio()

    async def _vad_task(
        self,
        vad_stream: vad.VADStream,
        connection: RealtimeConnection,
    ) -> None:
        async for ev in vad_stream:
            if ev.type == vad.VADEventType.START_OF_SPEECH:
                if not self._speaking:
                    self._speaking = True
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )
            elif ev.type == vad.VADEventType.END_OF_SPEECH:
                # force Mistral to finalize the current speech segment
                await connection.flush_audio()
                if self._speaking:
                    self._speaking = False
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    )

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
                self._event_ch.send_nowait(
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
                        TimedString(
                            text=seg.text,
                            start_time=seg.start + self.start_time_offset,
                            end_time=seg.end + self.start_time_offset,
                        )
                        for seg in event.segments
                    ]
                    if event.segments
                    else None
                )

                self._event_ch.send_nowait(
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

                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.RECOGNITION_USAGE,
                        request_id=self._request_id,
                        recognition_usage=stt.RecognitionUsage(
                            audio_duration=event.usage.prompt_audio_seconds or 0,
                            input_tokens=event.usage.prompt_tokens or 0,
                            output_tokens=event.usage.completion_tokens or 0,
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
                async with self._pool.connection(timeout=self._conn_options.timeout) as connection:
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
        except SDKError as e:
            if e.status_code in (408, 504):
                raise APITimeoutError() from e
            raise APIStatusError(e.message, status_code=e.status_code, body=e.body) from e
        except Exception as e:
            raise APIConnectionError() from e
