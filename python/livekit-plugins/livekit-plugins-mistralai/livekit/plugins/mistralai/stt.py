from __future__ import annotations

import base64
import os
import weakref
from dataclasses import dataclass
from typing import Annotated, Literal, NotRequired, TypedDict, assert_never
from urllib.parse import urlencode

import httpx  # mistralai raises httpx.TimeoutException from its httpx v1 client
from pydantic import Field, JsonValue

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
from mistralai.client import Mistral, models
from mistralai.client.errors import HTTPValidationError, SDKError
from mistralai.client.utils import get_security, get_security_from_env

from ._http import close_client, request_timeout_ms
from .models import STTModels


def _is_realtime(model: str) -> bool:
    return "realtime" in model


_HTTP_CLIENT_DEFAULTS = frozenset({"accept", "accept-encoding", "connection", "user-agent"})

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
                # usage events carry Mistral's billed prompt audio seconds
                reports_usage=True,
                # the VAD's end flushes the audio, which is transcribed after it
                transcribes_after_speech_ends=True,
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
        self._streams = weakref.WeakSet[SpeechStream]()

    def _realtime_target(self) -> stt.WebSocketTarget:
        """The realtime socket, authorized and routed as the client's own requests are."""
        cfg = self._client.sdk_configuration
        base_url, _ = cfg.get_server_details()
        security = cfg.security() if callable(cfg.security) else cfg.security
        resolved = get_security_from_env(security, models.Security)
        headers, query = get_security(resolved) if resolved is not None else ({}, {})
        client_headers = getattr(cfg.async_client, "headers", None) or getattr(
            cfg.client, "headers", None
        )
        # the HTTP client's defaults (connection, accept-encoding, ...) do not belong on a
        # websocket handshake: only the application's own headers are carried over
        custom = {
            name: value
            for name, value in dict(client_headers or {}).items()
            if name.lower() not in _HTTP_CLIENT_DEFAULTS
        }
        headers = {"User-Agent": cfg.user_agent, **headers, **custom}
        params = {key: values[-1] for key, values in query.items() if values}
        params["model"] = self._opts.model
        url = f"{base_url.rstrip('/')}/v1/audio/transcriptions/realtime?{urlencode(params)}"
        if url.startswith("http"):
            url = url.replace("http", "ws", 1)
        return stt.WebSocketTarget(url, headers=headers)

    async def aclose(self) -> None:
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
            for stream in list(self._streams):
                stream.reconnect("options changed")

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
        stream = SpeechStream(
            stt=self, conn_options=conn_options, language=language, vad_instance=self._vad
        )
        self._streams.add(stream)
        return stream


class _Session(TypedDict):
    request_id: NotRequired[str]


class _SessionCreated(TypedDict):
    type: Literal["session.created"]
    session: NotRequired[_Session | None]


class _Language(TypedDict):
    type: Literal["transcription.language"]
    audio_language: NotRequired[str | None]


class _TextDelta(TypedDict):
    type: Literal["transcription.text.delta"]
    text: str


class _Segment(TypedDict):
    text: str
    start: NotRequired[float | None]
    end: NotRequired[float | None]


class _Usage(TypedDict):
    prompt_tokens: NotRequired[int | None]
    completion_tokens: NotRequired[int | None]
    prompt_audio_seconds: NotRequired[int | None]


class _Done(TypedDict):
    type: Literal["transcription.done"]
    text: str
    language: NotRequired[str | None]
    segments: NotRequired[list[_Segment] | None]
    usage: NotRequired[_Usage | None]


class _ErrorDetail(TypedDict):
    message: NotRequired[JsonValue]
    code: NotRequired[int]


class _Error(TypedDict):
    type: Literal["error"]
    error: _ErrorDetail


_Message = Annotated[
    _SessionCreated | _Language | _TextDelta | _Done | _Error, Field(discriminator="type")
]


class _AudioAppend(TypedDict):
    type: Literal["input_audio.append"]
    audio: str


class _AudioControl(TypedDict):
    type: Literal["input_audio.flush", "input_audio.end"]


class _SessionUpdate(TypedDict):
    type: Literal["session.update"]
    session: dict[str, int]


_Command = _AudioAppend | _AudioControl | _SessionUpdate


class SpeechStream(stt.WebSocketStream[_Message, _Command]):
    """Realtime speech recognition stream for Voxtral Realtime."""

    frame_ms = 50
    # audio waits for session.created
    awaits_ready = True
    # the local VAD bounds the speech: Voxtral has no server-side endpointing
    provider_vad = True

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        language: NotGivenOr[str] = NOT_GIVEN,
        vad_instance: VAD | None = None,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._mistral = stt
        self._vad = vad_instance
        self._vad_stream: VADStream | None = None
        self._request_id = ""
        self._text = ""
        self._detected_language = LanguageCode(language) if is_given(language) else LanguageCode("")

    async def _run(self) -> None:
        if self._vad is None:
            await super()._run()
            return
        vad_stream = self._vad_stream = self._vad.stream()
        try:
            async with utils.aio.TaskSet() as tasks:
                await tasks.first_completed(super()._run(), self._listen(vad_stream))
        finally:
            self._vad_stream = None
            await vad_stream.aclose()

    async def _listen(self, vad_stream: VADStream) -> None:
        async for ev in vad_stream:
            if ev.type == VADEventType.START_OF_SPEECH:
                self.report(stt.SpeechStarted())
            elif ev.type == VADEventType.END_OF_SPEECH:
                self.report(stt.SpeechEnded())
                # force Mistral to finalize the current speech segment
                self.send_now(_AudioControl(type="input_audio.flush"))

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        self._text = ""
        return self._mistral._realtime_target()

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> _Command:
        match item:
            case rtc.AudioFrame():
                if self._vad_stream is not None:
                    self._vad_stream.push_frame(item)
                return _AudioAppend(
                    type="input_audio.append", audio=base64.b64encode(item.data).decode("ascii")
                )
            case stt.Flush():
                return _AudioControl(type="input_audio.flush")
            case stt.EndOfAudio():
                return _AudioControl(type="input_audio.end")
            case _:
                assert_never(item)

    def on_message(
        self, message: _SessionCreated | _Language | _TextDelta | _Done | _Error
    ) -> None:
        if message["type"] == "session.created":
            self._request_id = (message.get("session") or _Session()).get("request_id", "")
            if (delay := self._mistral._opts.target_streaming_delay_ms) is not None:
                self.send_now(
                    _SessionUpdate(
                        type="session.update", session={"target_streaming_delay_ms": delay}
                    )
                )
            self.report(stt.Ready())
        elif message["type"] == "transcription.language":
            if language := message.get("audio_language"):
                self._detected_language = LanguageCode(language)
        elif message["type"] == "transcription.text.delta":
            self._text += message["text"]
            self.report(
                stt.Transcript(self._text, self._detected_language, request_id=self._request_id)
            )
        elif message["type"] == "error":
            error = message["error"]
            raise APIStatusError(
                "Mistral transcription error",
                status_code=error.get("code", -1),
                body=error,
                detail=str(error.get("message")),
            )
        elif message["type"] == "transcription.done":
            self._text = ""
            self._report_usage(message.get("usage") or _Usage())
            self.report(self._final(message))
        else:
            self._skip_unknown_message(message)

    def _final(self, message: _Done) -> stt.Transcript:
        offset = self.start_time_offset
        segments = message.get("segments") or []
        language = message.get("language")
        return stt.Transcript(
            message["text"],
            LanguageCode(language) if language else self._detected_language,
            final=True,
            request_id=self._request_id,
            words=[
                TranscriptChunk(
                    text=seg["text"],
                    start_time=start + offset if (start := seg.get("start")) is not None else None,
                    end_time=end + offset if (end := seg.get("end")) is not None else None,
                )
                for seg in segments
            ]
            or None,
        )

    def _report_usage(self, usage: _Usage) -> None:
        self.send(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                recognition_usage=stt.RecognitionUsage(
                    audio_duration=usage.get("prompt_audio_seconds"),
                    tokens=TokenUsage(
                        input_tokens=usage.get("prompt_tokens") or 0,
                        output_tokens=usage.get("completion_tokens") or 0,
                    ),
                ),
            )
        )
