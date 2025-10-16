from __future__ import annotations

import asyncio
import base64
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Any, Literal, TypedDict, Union, overload

import aiohttp

from livekit import rtc

from .. import stt, utils
from .._exceptions import APIConnectionError, APIError, APIStatusError
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import is_given
from ._utils import create_access_token

DeepgramModels = Literal[
    "deepgram",
    "deepgram/nova-3",
    "deepgram/nova-3-general",
    "deepgram/nova-3-medical",
    "deepgram/nova-2",
    "deepgram/nova-2-general",
    "deepgram/nova-2-medical",
    "deepgram/nova-2-conversationalai",
    "deepgram/nova-2-phonecall",
]
CartesiaModels = Literal[
    "cartesia",
    "cartesia/ink-whisper",
]
AssemblyAIModels = Literal[
    "assemblyai",
    "assemblyai/universal-streaming",
]


class CartesiaOptions(TypedDict, total=False):
    min_volume: float  # default: not specified
    max_silence_duration_secs: float  # default: not specified


class DeepgramOptions(TypedDict, total=False):
    filler_words: bool  # default: True
    interim_results: bool  # default: True
    endpointing: int  # default: 25 (ms)
    punctuate: bool  # default: False
    smart_format: bool
    keywords: list[tuple[str, float]]
    keyterms: list[str]
    profanity_filter: bool
    numerals: bool
    mip_opt_out: bool


class AssemblyaiOptions(TypedDict, total=False):
    format_turns: bool  # default: False
    end_of_turn_confidence_threshold: float  # default: 0.01
    min_end_of_turn_silence_when_confident: int  # default: 0
    max_turn_silence: int  # default: not specified
    keyterms_prompt: list[str]  # default: not specified


STTLanguages = Literal["multi", "en", "de", "es", "fr", "ja", "pt", "zh", "hi"]


STTModels = Union[
    DeepgramModels,
    CartesiaModels,
    AssemblyAIModels,
    Literal["auto"],  # automatically select a provider based on the language
]
STTEncoding = Literal["pcm_s16le"]

DEFAULT_ENCODING: STTEncoding = "pcm_s16le"
DEFAULT_SAMPLE_RATE: int = 16000
DEFAULT_BASE_URL = "https://agent-gateway.livekit.cloud/v1"


@dataclass
class STTOptions:
    model: NotGivenOr[STTModels | str]
    language: NotGivenOr[str]
    encoding: STTEncoding
    sample_rate: int
    base_url: str
    api_key: str
    api_secret: str
    extra_kwargs: dict[str, Any]


class STT(stt.STT):
    @overload
    def __init__(
        self,
        model: CartesiaModels,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[STTEncoding] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        extra_kwargs: NotGivenOr[CartesiaOptions] = NOT_GIVEN,
    ) -> None: ...

    @overload
    def __init__(
        self,
        model: DeepgramModels,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[STTEncoding] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        extra_kwargs: NotGivenOr[DeepgramOptions] = NOT_GIVEN,
    ) -> None: ...

    @overload
    def __init__(
        self,
        model: AssemblyAIModels,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[STTEncoding] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        extra_kwargs: NotGivenOr[AssemblyaiOptions] = NOT_GIVEN,
    ) -> None: ...

    @overload
    def __init__(
        self,
        model: str,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[STTEncoding] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> None: ...

    def __init__(
        self,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[STTEncoding] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        extra_kwargs: NotGivenOr[
            dict[str, Any] | CartesiaOptions | DeepgramOptions | AssemblyaiOptions
        ] = NOT_GIVEN,
    ) -> None:
        """Livekit Cloud Inference STT

        Args:
            model (STTModels | str, optional): STT model to use.
            language (str, optional): Language of the STT model.
            encoding (STTEncoding, optional): Encoding of the STT model.
            sample_rate (int, optional): Sample rate of the STT model.
            base_url (str, optional): LIVEKIT_URL, if not provided, read from environment variable.
            api_key (str, optional): LIVEKIT_API_KEY, if not provided, read from environment variable.
            api_secret (str, optional): LIVEKIT_API_SECRET, if not provided, read from environment variable.
            http_session (aiohttp.ClientSession, optional): HTTP session to use.
            extra_kwargs (dict, optional): Extra kwargs to pass to the STT model.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True),
        )

        lk_base_url = (
            base_url
            if is_given(base_url)
            else os.environ.get("LIVEKIT_INFERENCE_URL", DEFAULT_BASE_URL)
        )

        lk_api_key = (
            api_key
            if is_given(api_key)
            else os.getenv("LIVEKIT_INFERENCE_API_KEY", os.getenv("LIVEKIT_API_KEY", ""))
        )
        if not lk_api_key:
            raise ValueError(
                "api_key is required, either as argument or set LIVEKIT_API_KEY environmental variable"
            )

        lk_api_secret = (
            api_secret
            if is_given(api_secret)
            else os.getenv("LIVEKIT_INFERENCE_API_SECRET", os.getenv("LIVEKIT_API_SECRET", ""))
        )
        if not lk_api_secret:
            raise ValueError(
                "api_secret is required, either as argument or set LIVEKIT_API_SECRET environmental variable"
            )

        self._opts = STTOptions(
            model=model,
            language=language,
            encoding=encoding if is_given(encoding) else DEFAULT_ENCODING,
            sample_rate=sample_rate if is_given(sample_rate) else DEFAULT_SAMPLE_RATE,
            base_url=lk_base_url,
            api_key=lk_api_key,
            api_secret=lk_api_secret,
            extra_kwargs=dict(extra_kwargs) if is_given(extra_kwargs) else {},
        )

        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @classmethod
    def from_model_string(cls, model: str) -> STT:
        """Create a STT instance from a model string

        Args:
            model (str): STT model to use, in "provider/model[:language]" format

        Returns:
            STT: STT instance
        """

        language: NotGivenOr[str] = NOT_GIVEN
        if (idx := model.rfind(":")) != -1:
            language = model[idx + 1 :]
            model = model[:idx]
        return cls(model, language=language)

    @property
    def model(self) -> str:
        return self._opts.model if is_given(self._opts.model) else "unknown"

    @property
    def provider(self) -> str:
        return "livekit"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "LiveKit STT does not support batch recognition, use stream() instead"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Create a streaming transcription session."""
        options = self._sanitize_options(language=language)
        stream = SpeechStream(stt=self, opts=options, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
    ) -> None:
        """Update STT configuration options."""
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language

        for stream in self._streams:
            stream.update_options(model=model, language=language)

    def _sanitize_options(
        self, *, language: NotGivenOr[STTLanguages | str] = NOT_GIVEN
    ) -> STTOptions:
        """Create a sanitized copy of options with language override if provided."""
        options = replace(self._opts)

        if is_given(language):
            options.language = language

        return options


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._session = stt._ensure_session()
        self._request_id = str(utils.shortuuid("stt_request_"))

        self._reconnect_event = asyncio.Event()
        self._speaking = False
        self._speech_duration: float = 0

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
    ) -> None:
        """Update streaming transcription options."""
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        self._reconnect_event.set()

    async def _run(self) -> None:
        """Main loop for streaming transcription."""
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=self._opts.sample_rate // 20,  # 50ms
            )

            async for ev in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(ev, rtc.AudioFrame):
                    frames.extend(audio_bstream.push(ev.data))
                elif isinstance(ev, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())

                for frame in frames:
                    self._speech_duration += frame.duration
                    audio_bytes = frame.data.tobytes()
                    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
                    audio_msg = {
                        "type": "input_audio",
                        "audio": base64_audio,
                    }
                    await ws.send_str(json.dumps(audio_msg))

            closing_ws = True
            finalize_msg = {
                "type": "session.finalize",
            }
            await ws.send_str(json.dumps(finalize_msg))

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        return
                    raise APIStatusError(message="LiveKit STT connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected LiveKit STT message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                msg_type = data.get("type")
                if msg_type == "session.created":
                    pass
                elif msg_type == "interim_transcript":
                    self._process_transcript(data, is_final=False)
                elif msg_type == "final_transcript":
                    self._process_transcript(data, is_final=True)
                elif msg_type == "session.finalized":
                    pass
                elif msg_type == "session.closed":
                    pass
                elif msg_type == "error":
                    raise APIError(f"LiveKit STT returned error: {msg.data}")
                else:
                    logger.warning("received unexpected message from LiveKit STT: %s", data)

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    tasks_group.exception()  # retrieve the exception
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Connect to the LiveKit STT WebSocket."""
        params: dict[str, Any] = {
            "settings": {
                "sample_rate": str(self._opts.sample_rate),
                "encoding": self._opts.encoding,
                "extra": self._opts.extra_kwargs,
            },
        }

        if self._opts.model and self._opts.model != "auto":
            params["model"] = self._opts.model

        if self._opts.language:
            params["settings"]["language"] = self._opts.language

        base_url = self._opts.base_url
        if base_url.startswith(("http://", "https://")):
            base_url = base_url.replace("http", "ws", 1)
        headers = {
            "Authorization": f"Bearer {create_access_token(self._opts.api_key, self._opts.api_secret)}"
        }
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(f"{base_url}/stt", headers=headers),
                self._conn_options.timeout,
            )
            params["type"] = "session.create"
            await ws.send_str(json.dumps(params))
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            if isinstance(e, aiohttp.ClientResponseError) and e.status == 429:
                raise APIStatusError("LiveKit STT quota exceeded", status_code=e.status) from e
            raise APIConnectionError("failed to connect to LiveKit STT") from e
        return ws

    def _process_transcript(self, data: dict, is_final: bool) -> None:
        request_id = data.get("request_id", self._request_id)
        text = data.get("transcript", "")
        language = data.get("language", self._opts.language or "en")

        if not text and not is_final:
            return
        # We'll have a more accurate way of detecting when speech started when we have VAD
        if not self._speaking:
            self._speaking = True
            start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
            self._event_ch.send_nowait(start_event)

        speech_data = stt.SpeechData(
            language=language,
            start_time=data.get("start", 0),
            end_time=data.get("duration", 0),  # This is the duration transcribed so far
            confidence=data.get("confidence", 1.0),
            text=text,
        )

        if is_final:
            if self._speech_duration > 0:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.RECOGNITION_USAGE,
                        request_id=request_id,
                        recognition_usage=stt.RecognitionUsage(
                            audio_duration=self._speech_duration,
                        ),
                    )
                )
                self._speech_duration = 0

            event = stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=request_id,
                alternatives=[speech_data],
            )
            self._event_ch.send_nowait(event)

            if self._speaking:
                self._speaking = False
                end_event = stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                self._event_ch.send_nowait(end_event)
        else:
            event = stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                request_id=request_id,
                alternatives=[speech_data],
            )
            self._event_ch.send_nowait(event)
