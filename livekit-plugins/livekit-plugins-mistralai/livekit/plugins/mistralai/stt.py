from __future__ import annotations

import asyncio
import json
import os
import re
import weakref
from dataclasses import dataclass
from typing import Any, TypedDict

import httpx2  # mistralai raises httpx2.TimeoutException from its http client
from websockets.exceptions import ConnectionClosed

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
from mistralai.client.errors import HTTPValidationError, SDKError
from mistralai.extra.exceptions import RealtimeTranscriptionException
from mistralai.extra.realtime import RealtimeConnection, RealtimeTranscription

from .log import logger
from .models import STTModels

DEFAULT_MODEL: STTModels = "voxtral-mini-transcribe-realtime-3"

SAMPLE_RATE: int = 16000
NUM_CHANNELS: int = 1


def _handshake_status_error(e: RealtimeTranscriptionException) -> APIStatusError | None:
    """Map a websocket handshake rejection (e.g. HTTP 401/403/404) to an APIStatusError.

    The SDK wraps `websockets.exceptions.InvalidStatus` in a generic exception, which would
    otherwise be treated as a retryable connection error.
    """
    response = getattr(e.__cause__, "response", None)
    status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, int):
        return None

    headers = getattr(response, "headers", None)
    body = getattr(response, "body", None)
    return APIStatusError(
        str(e),
        status_code=status_code,
        request_id=headers.get("x-request-id") if headers is not None else None,
        body=body.decode(errors="replace") if isinstance(body, bytes) else body,
        retryable=status_code != 429,
    )


class TurnDetection(TypedDict, total=False):
    """Server-side turn detection settings of the Voxtral 3 realtime models.

    All fields are optional, the server applies its defaults to the ones left out.
    The thresholds must satisfy ``start_threshold >= eager_end_threshold > end_threshold``
    and ``end_timeout_ms`` must be at least ``min_silence_ms``.
    """

    start_threshold: float
    eager_end_threshold: float
    end_threshold: float
    min_silence_ms: int
    end_timeout_ms: int | None


def _is_realtime(model: str) -> bool:
    return "realtime" in model


def _is_voxtral_3(model: str) -> bool:
    return re.search(r"-transcribe(-realtime)?-3(-|$)", model) is not None


@dataclass
class _STTOptions:
    model: STTModels | str
    language: LanguageCode | None
    context_bias: list[str] | None
    target_streaming_delay_ms: int | None
    diarize: bool = False
    turn_detection: TurnDetection | None = None


class STT(stt.STT):
    def __init__(
        self,
        client: Mistral | None = None,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        context_bias: NotGivenOr[list[str]] = NOT_GIVEN,
        target_streaming_delay_ms: NotGivenOr[int] = NOT_GIVEN,
        diarize: NotGivenOr[bool] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        vad: vad.VAD | None = None,
    ):
        """
        Create a new instance of MistralAI STT.

        Args:
            client: Optional pre-configured MistralAI client instance.
            api_key: Your Mistral AI API key. If not provided, will use the MISTRAL_API_KEY environment variable.
            model: The Mistral AI model to use for transcription, default is the realtime "voxtral-mini-transcribe-realtime-3".
            language: The optional language code to use for better transcription accuracy if language is already known (e.g., "fr" for French).
                Used with batch models and the Voxtral 3 realtime model.
            context_bias: Up to 100 words or phrases to guide the model toward good spelling or names or domain-specific vocabulary.
                Used with batch models and the Voxtral 3 realtime model.
            target_streaming_delay_ms: Target streaming delay in milliseconds for realtime mode. Only used with realtime models.
            diarize: Label each piece of text with its speaker (``speaker_id``). Supported by the batch models and
                the Voxtral 3 realtime model, but not by earlier realtime models.
                Voxtral 3 doesn't support diarization together with its server-side turn detection, so enabling
                ``diarize`` disables ``turn_detection`` by default, and passing both raises a ``ValueError``.
                The plugin then emits no start and end of speech events, and the voice pipeline can't use
                ``turn_detection="stt"``, unless a ``vad`` is given to provide them.
                The realtime model numbers the speakers within one transcript only: the numbering restarts after
                each flush, so the speakers of different transcripts can't be compared.
            turn_detection: Server-side turn detection of the Voxtral 3 realtime model, which provides the start and
                end of speech events and finalizes the transcript of each turn. Enabled with the server defaults
                when not provided, unless ``diarize`` is enabled or a ``vad`` is given, which it can't be used with.
                Pass a :class:`TurnDetection` dict to tune it, or ``None`` to disable it. Only supported by the
                Voxtral 3 realtime model.
            vad: Optional Voice Activity Detector run by the plugin on the realtime models: its end of speech
                flushes the audio to get the final transcript, and its start and end of speech are emitted as speech
                events, which the voice pipeline needs for ``turn_detection="stt"`` when the server doesn't provide
                them (earlier realtime models, or Voxtral 3 with ``diarize``). It takes precedence over the
                server-side turn detection of Voxtral 3, which is disabled by default, and can't be used with an
                explicit ``turn_detection``. Only used with realtime models.

        Note:
            Realtime models without server-side turn detection or a ``vad`` (earlier models, or Voxtral 3 with
            ``turn_detection=None`` or ``diarize``) have no endpointing. The voice pipeline (`AgentSession`)
            requests a flush from its own VAD and turn detection to get the final transcripts. A stream used on its
            own needs to be flushed with `stream.flush()`.
        """
        resolved_model = model if is_given(model) else DEFAULT_MODEL
        is_realtime = _is_realtime(resolved_model)

        diarize_enabled = is_given(diarize) and diarize
        if diarize_enabled and is_realtime and not _is_voxtral_3(resolved_model):
            raise ValueError("diarization is not supported by this realtime model")

        server_turn_detection: TurnDetection | None = None
        if is_realtime and _is_voxtral_3(resolved_model):
            if is_given(turn_detection):
                server_turn_detection = (
                    turn_detection.copy() if turn_detection is not None else None
                )
            elif not diarize_enabled and vad is None:
                server_turn_detection = {}
        elif is_given(turn_detection) and turn_detection is not None:
            raise ValueError("turn detection is only supported by the latest realtime model")

        if diarize_enabled and server_turn_detection is not None:
            raise ValueError("diarization and turn detection can't be used together")
        if vad is not None and server_turn_detection is not None:
            raise ValueError("a vad and turn detection can't be used together")

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=is_realtime,
                interim_results=is_realtime,
                diarization=diarize_enabled,
                aligned_transcript=False,
                offline_recognize=not is_realtime,
                manual_flush=is_realtime and server_turn_detection is None and vad is None,
            )
        )
        self._opts = _STTOptions(
            model=resolved_model,
            language=LanguageCode(language) if is_given(language) else None,
            context_bias=list(context_bias) if is_given(context_bias) and context_bias else None,
            target_streaming_delay_ms=target_streaming_delay_ms
            if is_given(target_streaming_delay_ms)
            else None,
            diarize=diarize_enabled,
            turn_detection=server_turn_detection,
        )

        mistral_api_key = api_key if is_given(api_key) else os.environ.get("MISTRAL_API_KEY")
        if not client and not mistral_api_key:
            raise ValueError("Mistral AI API key is required. Set MISTRAL_API_KEY or pass api_key")
        self._client = client or Mistral(api_key=mistral_api_key)
        self._vad = vad if is_realtime else None
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

        async def _connect() -> RealtimeConnection:
            connection = await rt.connect(model=self._opts.model, http_headers=http_headers)
            if session := self._session_fields():
                try:
                    await connection._websocket.send(
                        json.dumps({"type": "session.update", "session": session})
                    )
                except BaseException:
                    await connection.close()
                    raise
            return connection

        return await asyncio.wait_for(_connect(), timeout=timeout)

    def _session_fields(self) -> dict[str, Any]:
        """The realtime session settings, sent with a ``session.update`` message."""
        fields: dict[str, Any] = {}
        if self._opts.target_streaming_delay_ms is not None:
            fields["target_streaming_delay_ms"] = self._opts.target_streaming_delay_ms

        if _is_realtime(self._opts.model) and _is_voxtral_3(self._opts.model):
            if self._opts.language:
                fields["language"] = str(self._opts.language)
            if self._opts.context_bias:
                fields["context_bias"] = self._opts.context_bias
            if self._opts.diarize:
                fields["diarize"] = True
            if self._opts.turn_detection is not None:
                fields["turn_detection"] = dict(self._opts.turn_detection)
        return fields

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
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        context_bias: NotGivenOr[list[str]] = NOT_GIVEN,
        target_streaming_delay_ms: NotGivenOr[int] = NOT_GIVEN,
        diarize: NotGivenOr[bool] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
    ) -> None:
        """
        Update the STT options.

        Options that don't apply to the model are accepted and ignored, like in the constructor.
        Batch models apply the new options to the next `recognize()` call.
        The realtime session can't be reconfigured once audio has started, so changing a setting of
        the session reconnects the active streams: a transcript that was not finalized yet is lost.

        Args:
            model: The Mistral AI model to use for transcription. It must be of the same kind
                (batch or realtime) as the current one, since the kind is fixed at construction.
            language: The optional language code to use for better transcription accuracy if language is already known (e.g., "fr" for French).
                Used with batch models and the Voxtral 3 realtime model.
            context_bias: Up to 100 words or phrases to guide the model toward good spelling or names or domain-specific vocabulary.
                Used with batch models and the Voxtral 3 realtime model.
            target_streaming_delay_ms: Target streaming delay in milliseconds for realtime mode. Only used with realtime models.
            diarize: Label each piece of text with its speaker. Not supported by earlier realtime models.
                Voxtral 3 doesn't support diarization together with its server-side turn detection: changing
                ``diarize`` on the Voxtral 3 realtime model also resets the turn detection to its default, which is
                disabled with ``diarize`` (or a ``vad``) and enabled otherwise, unless ``turn_detection`` is given.
                Without turn detection, the plugin emits no start and end of speech events, so the voice pipeline
                can't use ``turn_detection="stt"`` unless the STT was created with a ``vad``.
            turn_detection: Server-side turn detection of the Voxtral 3 realtime model. Only supported by that
                model, and not together with ``diarize`` or a ``vad``.

        Raises:
            ValueError: If the model is not of the same kind as the current one, or if the options
                are not supported by the model or can't be used together.
        """
        resolved_model = model if is_given(model) else self._opts.model
        resolved_diarize = diarize if is_given(diarize) else self._opts.diarize
        was_voxtral_3_realtime = _is_realtime(self._opts.model) and _is_voxtral_3(self._opts.model)
        is_voxtral_3_realtime = _is_realtime(resolved_model) and _is_voxtral_3(resolved_model)

        new_turn_detection: TurnDetection | None = None
        if is_given(turn_detection):
            new_turn_detection = turn_detection.copy() if turn_detection is not None else None
        elif is_voxtral_3_realtime:
            if was_voxtral_3_realtime and resolved_diarize == self._opts.diarize:
                new_turn_detection = self._opts.turn_detection
            elif not resolved_diarize and self._vad is None:
                new_turn_detection = {}

        if _is_realtime(resolved_model) != self.capabilities.streaming:
            raise ValueError(
                f"can't switch from {self._opts.model} to {resolved_model}: this STT was created "
                f"with a {'realtime' if self.capabilities.streaming else 'batch'} model, "
                "create a new STT to use the other kind"
            )
        if new_turn_detection is not None and not is_voxtral_3_realtime:
            raise ValueError("turn detection is only supported by latest realtime model")
        if resolved_diarize and _is_realtime(resolved_model) and not is_voxtral_3_realtime:
            raise ValueError("diarization is not supported by this realtime model")
        if resolved_diarize and new_turn_detection is not None:
            raise ValueError("diarization and turn detection can't be used together")
        if self._vad is not None and new_turn_detection is not None:
            raise ValueError("a vad and turn detection can't be used together")

        new_context_bias = (
            (list(context_bias) if context_bias else None)
            if is_given(context_bias)
            else self._opts.context_bias
        )

        session_changed = (
            resolved_model != self._opts.model
            or new_turn_detection != self._opts.turn_detection
            or resolved_diarize != self._opts.diarize
            or (
                is_given(target_streaming_delay_ms)
                and target_streaming_delay_ms != self._opts.target_streaming_delay_ms
            )
            or (
                is_voxtral_3_realtime
                and (
                    (
                        is_given(language)
                        and (LanguageCode(language) if language else None) != self._opts.language
                    )
                    or new_context_bias != self._opts.context_bias
                )
            )
        )

        self._opts.model = resolved_model
        self._opts.diarize = resolved_diarize
        self._opts.turn_detection = new_turn_detection
        if is_given(language):
            self._opts.language = LanguageCode(language) if language else None
        self._opts.context_bias = new_context_bias
        if is_given(target_streaming_delay_ms):
            self._opts.target_streaming_delay_ms = target_streaming_delay_ms
        self._capabilities.diarization = resolved_diarize
        self._capabilities.manual_flush = (
            self.capabilities.streaming and new_turn_detection is None and self._vad is None
        )

        if session_changed and self.capabilities.streaming:
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
            # a per-call language applies to this request only
            language_code = LanguageCode(language) if is_given(language) else self._opts.language
            data = rtc.combine_audio_frames(buffer).to_wav_bytes()

            resp = await self._client.audio.transcriptions.complete_async(
                model=self._opts.model,
                file={"content": data, "file_name": "audio.wav"},
                language=language_code if language_code else None,
                context_bias=self._opts.context_bias if self._opts.context_bias else None,
                diarize=self._opts.diarize,
                timestamp_granularities=["segment"],
            )

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=resp.text,
                        language=LanguageCode(resp.language)
                        if resp.language
                        else language_code or LanguageCode(""),
                        start_time=resp.segments[0].start if resp.segments else 0,
                        end_time=resp.segments[-1].end if resp.segments else 0,
                        speaker_id=next(
                            (seg.speaker_id for seg in resp.segments or [] if seg.speaker_id), None
                        ),
                        words=[
                            TimedString(
                                text=segment.text,
                                start_time=segment.start,
                                end_time=segment.end,
                                speaker_id=segment.speaker_id or None,
                            )
                            for segment in resp.segments
                        ]
                        if resp.segments
                        else None,
                    ),
                ],
            )

        except (asyncio.TimeoutError, httpx2.TimeoutException) as e:
            raise APITimeoutError() from e
        except (SDKError, HTTPValidationError) as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.headers.get("x-request-id"),
                body=e.body,
                retryable=e.status_code != 429,
            ) from e
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
    ) -> None:
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=SAMPLE_RATE,
        )
        self._pool = pool
        self._opts = stt._opts
        self._vad = stt._vad
        self._reconnect_event = asyncio.Event()
        self._request_id = ""
        self._audio_duration = 0.0
        self._speaking = False
        self._audio_pending = False
        self._audio_ended = False
        self._current_text = ""
        self._preflight_text = ""
        self._language = LanguageCode(language) if is_given(language) else None
        self._detected_language = LanguageCode(language) if is_given(language) else LanguageCode("")

    async def _apply_stream_language(self, connection: RealtimeConnection) -> None:
        """Apply this stream's language override to a Voxtral 3 realtime connection."""
        # the language of the stream overrides the one of the session, which can still be
        # changed as no audio was sent on the connection yet
        if (
            self._language
            and self._language != self._opts.language
            and _is_realtime(self._opts.model)
            and _is_voxtral_3(self._opts.model)
        ):
            await connection._websocket.send(
                json.dumps({"type": "session.update", "session": {"language": str(self._language)}})
            )

    async def _send_audio(self, connection: RealtimeConnection, data: bytes) -> None:
        await connection.send_audio(data)
        self._audio_pending = True

    async def _flush_audio(self, connection: RealtimeConnection) -> None:
        if not self._audio_pending:
            return
        self._audio_pending = False
        await connection.flush_audio()

    async def _send_task(
        self, connection: RealtimeConnection, vad_stream: vad.VADStream | None
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
                    await self._send_audio(connection, frame.data.tobytes())
                await self._flush_audio(connection)
            else:
                if vad_stream is not None:
                    vad_stream.push_frame(data)

                data_bytes = data.data.tobytes()
                bytes_per_second = SAMPLE_RATE * NUM_CHANNELS * 2
                self._audio_duration += len(data_bytes) / bytes_per_second
                for frame in audio_bstream.write(data.data.tobytes()):
                    await self._send_audio(connection, frame.data.tobytes())

        if vad_stream is not None:
            vad_stream.end_input()
        await connection.end_audio()
        self._audio_ended = True

    async def _vad_task(self, vad_stream: vad.VADStream, connection: RealtimeConnection) -> None:
        async for ev in vad_stream:
            if ev.type == vad.VADEventType.START_OF_SPEECH:
                if not self._speaking:
                    self._speaking = True
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )
            elif ev.type == vad.VADEventType.END_OF_SPEECH:
                # force Mistral to finalize the current speech segment
                await self._flush_audio(connection)
                if self._speaking:
                    self._speaking = False
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    )

    async def _recv_task(self, connection: RealtimeConnection) -> None:
        try:
            await self._recv_events(connection)
        except ConnectionClosed:
            if not self._audio_ended:
                raise

    async def _recv_events(self, connection: RealtimeConnection) -> None:
        self._request_id = connection.request_id
        self._current_text = ""
        self._preflight_text = ""

        async for message in connection._websocket:
            try:
                event = json.loads(message)
            except ValueError:
                logger.warning("ignoring a message that is not valid json")
                continue

            if isinstance(event, dict):
                await self._on_event(connection, event)

    async def _on_event(self, connection: RealtimeConnection, event: dict[str, Any]) -> None:
        event_type = event.get("type")

        if event_type in ("session.created", "session.updated"):
            if request_id := (event.get("session") or {}).get("request_id"):
                self._request_id = request_id

        elif event_type == "transcription.language":
            if language := event.get("audio_language"):
                self._detected_language = LanguageCode(language)
            self._mark_provider_progress()

        elif event_type == "transcription.text.delta":
            self._mark_provider_progress()
            self._on_text(event.get("text") or "")

        elif event_type == "transcription.segment":
            self._mark_provider_progress()
            self._on_text(event.get("text") or "", speaker_id=event.get("speaker_id"))

        elif event_type == "transcription.done":
            self._mark_provider_progress()
            self._on_done(event)

        elif event_type == "error":
            error = event.get("error") or {}
            code = error.get("code")
            # a retry would restart the session, so report it like the LLM does: not retryable
            raise APIStatusError(
                message=str(error.get("message")),
                status_code=code if isinstance(code, int) else -1,
                request_id=self._request_id or None,
                body=error,
                retryable=False,
            )

        elif isinstance(event_type, str) and event_type.startswith("turn."):
            await self._on_turn_event(connection, event_type)

    def _on_text(self, text: str, *, speaker_id: str | None = None) -> None:
        self._current_text += text
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=[
                    stt.SpeechData(
                        text=self._current_text,
                        language=self._detected_language,
                        speaker_id=speaker_id,
                    )
                ],
            )
        )

    def _on_done(self, event: dict[str, Any]) -> None:
        self._current_text = ""
        self._preflight_text = ""
        segments = [
            (seg.get("text") or "", seg.get("start"), seg.get("end"), seg.get("speaker_id"))
            for seg in event.get("segments") or []
        ]
        words = (
            [
                TimedString(
                    text=seg_text,
                    start_time=start + self.start_time_offset if start is not None else NOT_GIVEN,
                    end_time=end + self.start_time_offset if end is not None else NOT_GIVEN,
                    speaker_id=speaker_id or None,
                )
                for seg_text, start, end, speaker_id in segments
            ]
            if segments
            else None
        )

        if event.get("text") or segments or not self._audio_ended:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[
                        stt.SpeechData(
                            text=event.get("text") or "",
                            language=LanguageCode(language)
                            if (language := event.get("language"))
                            else self._detected_language,
                            speaker_id=next((spk for _, _, _, spk in segments if spk), None),
                            words=words,
                        )
                    ],
                )
            )

        usage = event.get("usage") or {}
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                recognition_usage=stt.RecognitionUsage(
                    audio_duration=usage.get("prompt_audio_seconds") or 0,
                    input_tokens=usage.get("prompt_tokens") or 0,
                    output_tokens=usage.get("completion_tokens") or 0,
                ),
            )
        )

    async def _on_turn_event(self, connection: RealtimeConnection, event_type: str | None) -> None:
        """Server-side turn detection (Voxtral 3 realtime)."""
        if event_type == "turn.start":
            if not self._speaking:
                self._speaking = True
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )

        elif event_type == "turn.eager_end":
            if self._speaking and self._current_text and self._current_text != self._preflight_text:
                self._preflight_text = self._current_text
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[
                            stt.SpeechData(
                                text=self._current_text,
                                language=self._detected_language,
                            )
                        ],
                    )
                )

        elif event_type == "turn.resume":
            # the user kept talking after the eager end: re-emit the transcript as an interim.
            # it matches the preflight, so the pipeline keeps the preemptive generation until
            # the transcript moves past it
            if self._speaking and self._current_text:
                self._on_text("")

        elif event_type == "turn.end":
            await self._flush_audio(connection)
            if self._speaking:
                self._speaking = False
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        try:
            while True:
                async with self._pool.connection(timeout=self._conn_options.timeout) as connection:
                    self._audio_pending = False
                    await self._apply_stream_language(connection)
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
                            # the audio ended, so the server closes this connection: don't reuse it
                            self._pool.remove(connection)
                            break

                        self._reconnect_event.clear()
                        if self._speaking:
                            # the turn is lost with the session
                            self._speaking = False
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                            )
                    finally:
                        await utils.aio.gracefully_cancel(*tasks, wait_reconnect)
                        tasks_group.cancel()
                        tasks_group.exception()
                        if vad_stream is not None:
                            await vad_stream.aclose()

        except (APIStatusError, APITimeoutError, APIConnectionError):
            raise
        except (asyncio.TimeoutError, httpx2.TimeoutException) as e:
            raise APITimeoutError() from e
        except (SDKError, HTTPValidationError) as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.headers.get("x-request-id"),
                body=e.body,
                retryable=e.status_code != 429,
            ) from e
        except RealtimeTranscriptionException as e:
            if (status_error := _handshake_status_error(e)) is not None:
                raise status_error from e
            raise APIConnectionError() from e
        except Exception as e:
            raise APIConnectionError() from e
