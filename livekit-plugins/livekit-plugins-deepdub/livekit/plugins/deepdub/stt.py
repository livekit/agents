from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import aiohttp
import numpy as np

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.language import LanguageCode
from livekit.agents.stt import SpeechEventType
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import (
    DEFAULT_STT_BASE_URL,
    DEFAULT_STT_MODEL,
    DEFAULT_STT_SAMPLE_RATE,
    STT_LANGUAGES,
    STT_PATH,
    STTLanguages,
    STTModels,
)

_KEEPALIVE_MESSAGE = json.dumps({"type": "keepalive"})
_FINALIZE_MESSAGE = json.dumps({"type": "finalize"})
_END_TOKEN = "<end>"
_FINALIZED_TOKEN = "<fin>"
_CHUNK_DURATION_MS = 100
_KEEPALIVE_INTERVAL = 5.0


@dataclass
class _STTOptions:
    api_key: str
    base_url: str
    model: STTModels | str
    language_hints: list[str] | None
    language_hints_strict: bool
    context_terms: list[str] | None
    context_text: str | None
    max_endpoint_delay_ms: int | None
    sample_rate: int
    client_reference_id: str | None

    def get_ws_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{STT_PATH}"


class STT(stt.STT):
    """Deepdub real-time speech-to-text for Hebrew and English.

    Streams audio over the Deepdub WebSocket API and returns interim transcripts
    while the user speaks, then a final transcript when the server detects the end
    of the turn. Only streaming recognition is supported.

    Args:
        api_key: Deepdub API key. Defaults to the ``DEEPDUB_API_KEY`` environment
            variable.
        model: Deepdub speech-to-text model. Defaults to ``"stt-egf"``.
        language: ``"en"`` or ``"he"`` to lock recognition to one language. When
            not set, the language is detected per utterance and can switch between
            Hebrew and English within a session.
        language_hints_strict: Restrict detection to the hinted language.
        context_terms: Words to boost, such as names and product terms. Up to 200.
        context_text: Earlier conversation as plain text, used as recognition
            history.
        max_endpoint_delay_ms: How long a pause must last before a turn ends, from
            500 to 3000. Uses the server default when not set.
        sample_rate: Sample rate of the audio sent to Deepdub. Defaults to 16000.
        client_reference_id: Your own identifier, echoed in Deepdub logs.
        base_url: WebSocket host. Defaults to ``"wss://wsgw.deepdub.ai"``.
        http_session: Optional shared ``aiohttp.ClientSession``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: STTModels | str = DEFAULT_STT_MODEL,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        language_hints_strict: bool = False,
        context_terms: list[str] | None = None,
        context_text: str | None = None,
        max_endpoint_delay_ms: int | None = None,
        sample_rate: int = DEFAULT_STT_SAMPLE_RATE,
        client_reference_id: str | None = None,
        base_url: str = DEFAULT_STT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="chunk",
                offline_recognize=False,
                manual_flush=True,
            )
        )
        deepdub_api_key = api_key or os.environ.get("DEEPDUB_API_KEY")
        if not deepdub_api_key:
            raise ValueError(
                "Deepdub API key is required, either as argument or set"
                " DEEPDUB_API_KEY environment variable"
            )
        if max_endpoint_delay_ms is not None and not (500 <= max_endpoint_delay_ms <= 3000):
            raise ValueError("max_endpoint_delay_ms must be between 500 and 3000")

        self._opts = _STTOptions(
            api_key=deepdub_api_key,
            base_url=base_url,
            model=model,
            language_hints=_language_hints(language),
            language_hints_strict=language_hints_strict,
            context_terms=context_terms,
            context_text=context_text,
            max_endpoint_delay_ms=max_endpoint_delay_ms,
            sample_rate=sample_rate,
            client_reference_id=client_reference_id,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        """The Deepdub model name."""
        return self._opts.model

    @property
    def provider(self) -> str:
        """The provider name, ``"Deepdub"``."""
        return "Deepdub"

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
        raise NotImplementedError("Deepdub speech-to-text supports streaming recognition only")

    def stream(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Open a streaming recognition session.

        Args:
            language: Overrides the language set on the ``STT`` for this session.
            conn_options: Connection options for the session.

        Returns:
            A ``SpeechStream``.
        """
        return SpeechStream(stt=self, conn_options=conn_options, language=language)


class SpeechStream(stt.SpeechStream):
    """One streaming recognition session; one WebSocket per connection attempt."""

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._opts.sample_rate)
        self._stt: STT = stt
        self._language_hints = (
            _language_hints(language) if is_given(language) else stt._opts.language_hints
        )
        self._reported_audio_ms = 0

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        # Deepdub takes mono audio, and the base stream resamples without mixing channels.
        super().push_frame(_downmix_to_mono(frame))

    def _start_message(self) -> dict[str, Any]:
        opts = self._stt._opts
        config: dict[str, Any] = {
            "model": opts.model,
            "audio_format": "pcm_s16le",
            "sample_rate": opts.sample_rate,
            "num_channels": 1,
            "enable_endpoint_detection": True,
        }
        if self._language_hints:
            config["language_hints"] = self._language_hints
            config["language_hints_strict"] = opts.language_hints_strict
        if opts.max_endpoint_delay_ms is not None:
            config["max_endpoint_delay_ms"] = opts.max_endpoint_delay_ms
        context: dict[str, Any] = {}
        if opts.context_terms:
            context["terms"] = opts.context_terms
        if opts.context_text:
            context["text"] = opts.context_text
        if context:
            config["context"] = context
        if opts.client_reference_id:
            config["client_reference_id"] = opts.client_reference_id
        return config

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        ws = await asyncio.wait_for(
            self._stt._ensure_session().ws_connect(
                self._stt._opts.get_ws_url(),
                headers={"Authorization": f"Bearer {self._stt._opts.api_key}"},
                autoping=False,
            ),
            self._conn_options.timeout,
        )
        await ws.send_str(json.dumps(self._start_message()))
        return ws

    async def _run(self) -> None:
        try:
            ws = await self._connect_ws()
        except asyncio.TimeoutError:
            raise APITimeoutError("timed out connecting to Deepdub speech-to-text") from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                f"Deepdub speech-to-text handshake failed: {e.message}",
                status_code=e.status,
                request_id=None,
                body=None,
            ) from None
        except Exception as e:
            # Keep only the exception type: an aiohttp handshake error can carry the
            # request headers, which include the API key.
            raise APIConnectionError(type(e).__name__) from None

        self._input_ch.restart_provider_clock()
        self._reported_audio_ms = 0
        input_done = asyncio.Event()

        async def _send_task() -> None:
            samples_per_chunk = self._stt._opts.sample_rate * _CHUNK_DURATION_MS // 1000
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._stt._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_chunk,
            )
            unfinalized_audio = False
            async for data in self._input_ch:
                if isinstance(data, rtc.AudioFrame):
                    frames = audio_bstream.write(data.data.tobytes())
                else:
                    frames = audio_bstream.flush()
                for frame in frames:
                    await ws.send_bytes(frame.data.tobytes())
                    unfinalized_audio = True
                if isinstance(data, self._FlushSentinel) and unfinalized_audio:
                    await ws.send_str(_FINALIZE_MESSAGE)
                    unfinalized_audio = False

            for frame in audio_bstream.flush():
                await ws.send_bytes(frame.data.tobytes())
            input_done.set()
            # An empty text frame ends the audio; the server answers with `finished`.
            await ws.send_str("")

        async def _keepalive_task() -> None:
            while True:
                await asyncio.sleep(_KEEPALIVE_INTERVAL)
                if input_done.is_set():
                    return
                await ws.send_str(_KEEPALIVE_MESSAGE)

        tasks = [
            asyncio.create_task(_send_task()),
            asyncio.create_task(_keepalive_task()),
        ]
        recv = asyncio.create_task(self._receive(ws, input_done))
        try:
            # The session ends when the receiver sees `finished`; a sender that fails
            # first ends it early.
            done: set[asyncio.Task[None]] = set()
            while recv not in done:
                pending = [t for t in (recv, *tasks) if not t.done()]
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task is not recv:
                        task.result()
            recv.result()
        except APIError:
            raise
        except (aiohttp.ClientError, ConnectionError) as e:
            raise APIConnectionError(type(e).__name__) from None
        finally:
            await utils.aio.gracefully_cancel(recv, *tasks)
            await ws.close()

    async def _receive(
        self, ws: aiohttp.ClientWebSocketResponse, input_done: asyncio.Event
    ) -> None:
        final = _TokenAccumulator()
        speaking = False

        def end_utterance() -> None:
            nonlocal speaking
            if final.text:
                if not speaking:
                    # The first words of this utterance arrived together with its endpoint.
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                    )
                data = final.to_speech_data(self.start_time_offset)
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[data])
                )
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=SpeechEventType.END_OF_SPEECH, alternatives=[data])
                )
            elif speaking:
                # Only provisional words were heard (a cough, noise) and none became
                # final; the utterance must still be closed.
                self._event_ch.send_nowait(stt.SpeechEvent(type=SpeechEventType.END_OF_SPEECH))
            final.reset()
            speaking = False

        monitor = utils.aio.WebSocketMonitor(
            ws, liveness_timeout=self._conn_options.liveness_timeout
        )
        while True:
            # Iterating the monitor would end silently on a close frame; receiving
            # directly lets a mid-session disconnect raise and be retried.
            msg = await monitor.receive()
            if msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.ERROR,
            ):
                end_utterance()
                if input_done.is_set():
                    return
                raise APIConnectionError("Deepdub speech-to-text connection closed unexpectedly")
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            try:
                content = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.debug("ignoring non-JSON text frame from Deepdub")
                continue

            if "error_code" in content or "error_message" in content:
                end_utterance()
                raise _api_error(content)

            non_final = _TokenAccumulator()
            for token in content.get("tokens") or []:
                text = token.get("text", "")
                if token.get("is_final"):
                    if text in (_END_TOKEN, _FINALIZED_TOKEN):
                        end_utterance()
                    else:
                        final.update(token)
                elif text not in (_END_TOKEN, _FINALIZED_TOKEN):
                    non_final.update(token)

            if final.text or non_final.text:
                if not speaking:
                    speaking = True
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                    )
                # Everything heard so far is final, so the text is stable enough for
                # preemptive generation even before the endpoint arrives.
                event_type = (
                    SpeechEventType.PREFLIGHT_TRANSCRIPT
                    if final.text and not non_final.text
                    else SpeechEventType.INTERIM_TRANSCRIPT
                )
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=event_type,
                        alternatives=[final.merged_speech_data(non_final, self.start_time_offset)],
                    )
                )

            self._report_usage(content.get("total_audio_proc_ms"))
            if content.get("finished"):
                end_utterance()
                return
            self._mark_provider_progress()

    def _report_usage(self, total_audio_proc_ms: Any) -> None:
        if not isinstance(total_audio_proc_ms, (int, float)):
            return
        delta_ms = int(total_audio_proc_ms) - self._reported_audio_ms
        if delta_ms <= 0:
            return
        self._reported_audio_ms = int(total_audio_proc_ms)
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=SpeechEventType.RECOGNITION_USAGE,
                recognition_usage=stt.RecognitionUsage(audio_duration=delta_ms / 1000),
            )
        )


class _TokenAccumulator:
    """Joins tokens into one transcript with its time span, language and confidence."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.text = ""
        self.start_ms: float | None = None
        self.end_ms: float = 0.0
        self._confidences: list[float] = []
        self._chars_by_language: dict[str, int] = {}

    def update(self, token: dict[str, Any]) -> None:
        text = token.get("text", "")
        self.text += text
        if (start := token.get("start_ms")) is not None:
            self.start_ms = start if self.start_ms is None else min(self.start_ms, start)
        if (end := token.get("end_ms")) is not None:
            self.end_ms = max(self.end_ms, end)
        if (confidence := token.get("confidence")) is not None:
            self._confidences.append(float(confidence))
        if language := token.get("language"):
            self._chars_by_language[language] = self._chars_by_language.get(language, 0) + len(text)

    @property
    def language(self) -> str:
        if not self._chars_by_language:
            return ""
        return max(self._chars_by_language, key=lambda lang: self._chars_by_language[lang])

    def to_speech_data(self, start_time_offset: float) -> stt.SpeechData:
        return self.merged_speech_data(_TokenAccumulator(), start_time_offset)

    def merged_speech_data(
        self, other: _TokenAccumulator, start_time_offset: float
    ) -> stt.SpeechData:
        starts = [acc.start_ms for acc in (self, other) if acc.start_ms is not None]
        confidences = self._confidences + other._confidences
        return stt.SpeechData(
            text=self.text + other.text,
            language=LanguageCode(self.language or other.language),
            start_time=(min(starts) if starts else 0.0) / 1000 + start_time_offset,
            end_time=max(self.end_ms, other.end_ms) / 1000 + start_time_offset,
            confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        )


def _downmix_to_mono(frame: rtc.AudioFrame) -> rtc.AudioFrame:
    if frame.num_channels == 1:
        return frame
    samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
    # Average in floating point to avoid overflowing when mixing loud channels.
    mono = samples.mean(axis=1).astype(np.int16)
    return rtc.AudioFrame(
        data=mono.tobytes(),
        sample_rate=frame.sample_rate,
        num_channels=1,
        samples_per_channel=frame.samples_per_channel,
    )


def _api_error(content: dict[str, Any]) -> APIStatusError:
    code = content.get("error_code")
    status_code = code if isinstance(code, int) else -1
    return APIStatusError(
        f"Deepdub speech-to-text error ({content.get('error_type')}):"
        f" {content.get('error_message')}",
        status_code=status_code,
        request_id=None,
        body=None,
    )


def _language_hints(language: NotGivenOr[str]) -> list[str] | None:
    if not is_given(language):
        return None
    code = LanguageCode(language).language
    if code not in STT_LANGUAGES:
        raise ValueError(f"Deepdub speech-to-text supports {sorted(STT_LANGUAGES)}, got {code!r}")
    return [code]
