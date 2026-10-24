"""OpenAI Realtime translation (``gpt-realtime-translate``) as a :class:`TranslationModel`.

A translation session is not a conversation: it has its own endpoint
(``/v1/realtime/translations``), takes no instructions, and translates the audio streamed into it
continuously, with no response lifecycle. The server reports no turns and no end of a
translation, only deltas: the session marks the end of a transcript itself once its deltas go
quiet for ``transcript_segment_timeout``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
from collections.abc import AsyncIterable, Callable, Iterator
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Literal
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, utils
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
from livekit.agents.llm._realtime.openai import (
    NUM_CHANNELS,
    OPENAI_BASE_URL,
    SAMPLE_RATE,
    _is_fatal_error,
    lk_oai_debug,
)
from livekit.agents.llm._realtime.openai_utils import DEFAULT_MAX_SESSION_DURATION
from livekit.agents.log import logger
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from openai.types.realtime import NoiseReductionType

TranslationModels = Literal["gpt-realtime-translate"]

DEFAULT_TRANSLATION_MODEL = "gpt-realtime-translate"
DEFAULT_INPUT_TRANSCRIPTION_MODEL = "gpt-realtime-whisper"
# the server sends deltas only; a transcript ends once they have been quiet this long
DEFAULT_TRANSCRIPT_SEGMENT_TIMEOUT = 1.0

TranslationEvents = Literal["openai_server_event_received", "openai_client_event_queued"]


@dataclass
class _TranslationOptions:
    model: str
    target_language: str
    input_transcription_model: str | None
    noise_reduction: NoiseReductionType | None
    api_key: str
    base_url: str
    transcript_segment_timeout: float
    max_session_duration: float | None
    conn_options: APIConnectOptions


class RealtimeTranslationModel(TranslationModel):
    """Speech-to-speech translation with OpenAI's ``gpt-realtime-translate``.

    The source language is detected by the model, so only the language to translate into is
    set. Run it with :class:`~livekit.agents.beta.translation.TranslationSession`.

    Example:
        ```python
        session = TranslationSession(
            model=openai.beta.RealtimeTranslationModel(target_language="es")
        )
        await session.start(room=ctx.room)
        ```
    """

    def __init__(
        self,
        *,
        target_language: str,
        model: TranslationModels | str = DEFAULT_TRANSLATION_MODEL,
        input_transcription_model: str | None = DEFAULT_INPUT_TRANSCRIPTION_MODEL,
        noise_reduction: NoiseReductionType | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        transcript_segment_timeout: float = DEFAULT_TRANSCRIPT_SEGMENT_TIMEOUT,
        max_session_duration: float | None = DEFAULT_MAX_SESSION_DURATION,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            target_language: The language to translate into, as a code such as ``"es"``.
            model: The translation model.
            input_transcription_model: The model that transcribes the speech heard, in the
                language it was spoken, or ``None`` to transcribe only the translation.
            noise_reduction: ``"near_field"`` for close-talking microphones, ``"far_field"``
                for laptop or room microphones, or ``None`` for none.
            api_key: OpenAI API key; read from ``OPENAI_API_KEY`` when omitted.
            base_url: Base URL of the OpenAI API.
            transcript_segment_timeout: Seconds without a transcript delta that end a
                transcript, since the server marks no end itself.
            max_session_duration: Seconds after which the connection is replaced, ahead of
                the server's own session limit. ``None`` keeps it until it drops.
            http_session: Shared HTTP session for the websocket.
            conn_options: Retry and timeout settings for the connection.

        Raises:
            ValueError: If no API key is given and ``OPENAI_API_KEY`` is unset.
        """
        super().__init__(
            capabilities=TranslationCapabilities(
                input_transcription=input_transcription_model is not None,
                output_transcription=True,
                turn_reporting=False,
                output_segmentation=False,
                retargetable=True,
            ),
            target_language=target_language,
            source_language=None,
        )

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "The api_key client option must be set either by passing api_key "
                "to the client or by setting the OPENAI_API_KEY environment variable"
            )

        self._opts = _TranslationOptions(
            model=model,
            target_language=target_language,
            input_transcription_model=input_transcription_model,
            noise_reduction=noise_reduction,
            api_key=api_key,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL,
            transcript_segment_timeout=transcript_segment_timeout,
            max_session_duration=max_session_duration,
            conn_options=conn_options,
        )
        self._http_session = http_session
        self._http_session_owned = False

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return urlparse(self._opts.base_url).netloc

    def update_options(self, *, target_language: str) -> None:
        super().update_options(target_language=target_language)
        self._opts.target_language = target_language

    def session(self) -> RealtimeTranslationSession:
        return RealtimeTranslationSession(self)

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            try:
                self._http_session = utils.http_context.http_session()
            except RuntimeError:
                self._http_session = aiohttp.ClientSession()
                self._http_session_owned = True

        return self._http_session

    async def aclose(self) -> None:
        if self._http_session_owned and self._http_session:
            await self._http_session.close()


def _translation_url(base_url: str, model: str) -> str:
    if base_url.startswith("http"):
        base_url = base_url.replace("http", "ws", 1)

    parsed_url = urlparse(base_url)
    path_stripped = parsed_url.path.rstrip("/")
    if path_stripped in ["", "/v1", "/openai", "/openai/v1"]:
        path = path_stripped + "/realtime/translations"
    else:
        path = parsed_url.path

    query_params = parse_qs(parsed_url.query)
    query_params.setdefault("model", [model])
    query = urlencode(query_params, doseq=True)
    return urlunparse((parsed_url.scheme, parsed_url.netloc, path, "", query, ""))


class _TranscriptSegmenter:
    """Ends a transcript once its deltas go quiet, since the server sends no end itself."""

    def __init__(
        self,
        timeout: float,
        on_text: Callable[[str], None],
        on_final: Callable[[], None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._timeout = timeout
        self._on_text = on_text
        self._on_final = on_final
        self._loop = loop
        self._timer: asyncio.TimerHandle | None = None

    def push(self, delta: str) -> None:
        if not delta:
            return

        self._on_text(delta)
        if self._timer is not None:
            self._timer.cancel()
        self._timer = self._loop.call_later(self._timeout, self.flush)

    def flush(self) -> None:
        """End the open transcript, if one is open."""
        if self._timer is None:
            return

        self._timer.cancel()
        self._timer = None
        self._on_final()


class RealtimeTranslationSession(TranslationModelSession[TranslationEvents]):
    """One connection to the OpenAI Realtime translation endpoint.

    Transcripts arrive as deltas: each ``input_transcript`` and ``output_transcript`` carries
    the text added since the last, and the one that ends a transcript carries none, only
    ``is_final``.

    It exposes two more events:
    - openai_server_event_received: the raw server events
    - openai_client_event_queued: the raw client events sent
    """

    def __init__(self, translation_model: RealtimeTranslationModel) -> None:
        super().__init__(translation_model)
        self._model: RealtimeTranslationModel = translation_model
        self._opts = replace(translation_model._opts)
        self._loop = asyncio.get_event_loop()

        self._msg_ch = utils.aio.Chan[dict[str, Any]]()
        self._audio_ch = utils.aio.Chan[TranslatedAudioFrame | OutputSpeechStopped]()
        self._input_resampler: rtc.AudioResampler | None = None
        # 100ms chunks
        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )

        self._input_segmenter = _TranscriptSegmenter(
            self._opts.transcript_segment_timeout,
            on_text=lambda text: self.emit(
                "input_transcript", InputTranscript(text=text, is_final=False)
            ),
            on_final=lambda: self.emit("input_transcript", InputTranscript(text="", is_final=True)),
            loop=self._loop,
        )
        self._output_segmenter = _TranscriptSegmenter(
            self._opts.transcript_segment_timeout,
            on_text=lambda text: self.emit(
                "output_transcript",
                OutputTranscript(text=text, is_final=False, language=self._opts.target_language),
            ),
            on_final=lambda: self.emit(
                "output_transcript",
                OutputTranscript(text="", is_final=True, language=self._opts.target_language),
            ),
            loop=self._loop,
        )

        self.send_event(self._create_session_update_event())
        self._main_atask = asyncio.create_task(
            self._main_task(), name="RealtimeTranslationSession._main_task"
        )

    @property
    def audio_stream(self) -> AsyncIterable[TranslatedAudioFrame | OutputSpeechStopped]:
        return self._audio_ch

    def send_event(self, event: dict[str, Any]) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        for f in self._resample_audio(frame):
            for nf in self._bstream.write(f.data.tobytes()):
                self.send_event(
                    {
                        "type": "session.input_audio_buffer.append",
                        "audio": base64.b64encode(nf.data).decode("utf-8"),
                    }
                )

    def clear_audio(self) -> None:
        # what the server sent is already here; drop what the consumer has not read yet
        with contextlib.suppress(utils.aio.channel.ChanEmpty, utils.aio.channel.ChanClosed):
            while True:
                self._audio_ch.recv_nowait()

    def update_options(self, *, target_language: str) -> None:
        if target_language == self._opts.target_language:
            return

        # the transcript so far was in the old language
        self._output_segmenter.flush()
        self._opts.target_language = target_language
        self.send_event(
            {
                "type": "session.update",
                "event_id": utils.shortuuid("session_update_"),
                "session": {"audio": {"output": {"language": target_language}}},
            }
        )

    async def aclose(self) -> None:
        if not self._closing:
            self._closing = True
            self._input_segmenter.flush()
            self._output_segmenter.flush()
            # sent last, so the server ends the session after the audio already queued
            self.send_event({"type": "session.close"})
            self._msg_ch.close()

        # a failure that ended the session was already emitted as an error
        with contextlib.suppress(Exception):
            await self._main_atask

    def _create_session_update_event(self) -> dict[str, Any]:
        audio_input: dict[str, Any] = {}
        if self._opts.input_transcription_model is not None:
            audio_input["transcription"] = {"model": self._opts.input_transcription_model}
        if self._opts.noise_reduction is not None:
            audio_input["noise_reduction"] = {"type": self._opts.noise_reduction}

        audio: dict[str, Any] = {"output": {"language": self._opts.target_language}}
        if audio_input:
            audio["input"] = audio_input

        return {
            "type": "session.update",
            "event_id": utils.shortuuid("session_update_"),
            "session": {"audio": audio},
        }

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                # input audio changed to a different sample rate
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        if self._input_resampler:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        num_retries = 0
        max_retries = self._opts.conn_options.max_retry
        reconnecting = False
        try:
            while not self._msg_ch.closed:
                try:
                    ws_conn = await self._create_ws_conn()
                    if reconnecting:
                        # a new connection starts from defaults; the options go first, before
                        # any audio still queued behind them
                        await self._send(ws_conn, self._create_session_update_event())
                        self._input_segmenter.flush()
                        self._output_segmenter.flush()
                        logger.debug("reconnected to OpenAI Realtime translation")
                    num_retries = 0
                    # a failed first handshake leaves the startup update queued, with nothing
                    # to replay
                    reconnecting = True
                    await self._run_ws(ws_conn)

                except APIError as e:
                    if max_retries == 0 or not e.retryable:
                        self._emit_error(e, recoverable=False)
                        raise
                    elif num_retries == max_retries:
                        self._emit_error(e, recoverable=False)
                        raise APIConnectionError(
                            f"OpenAI Realtime translation connection failed after "
                            f"{num_retries} attempts",
                        ) from e
                    else:
                        self._emit_error(e, recoverable=True)

                        retry_interval = self._opts.conn_options._interval_for_retry(num_retries)
                        logger.warning(
                            "OpenAI Realtime translation connection failed, "
                            f"retrying in {retry_interval}s",
                            exc_info=e,
                            extra={"attempt": num_retries, "max_retries": max_retries},
                        )
                        await asyncio.sleep(retry_interval)
                    num_retries += 1

                except Exception as e:
                    self._emit_error(e, recoverable=False)
                    raise
        finally:
            self._audio_ch.close()

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        url = _translation_url(self._opts.base_url, self._opts.model)
        headers = {
            "User-Agent": "LiveKit Agents",
            "Authorization": f"Bearer {self._opts.api_key}",
        }

        if lk_oai_debug:
            logger.debug(f"connecting to Realtime translation API: {url}")

        try:
            return await asyncio.wait_for(
                self._model._ensure_http_session().ws_connect(url=url, headers=headers),
                self._opts.conn_options.timeout,
            )
        except aiohttp.ClientError as e:
            raise APIConnectionError("OpenAI Realtime translation client connection error") from e
        except asyncio.TimeoutError as e:
            raise APIConnectionError(
                message="OpenAI Realtime translation connection timed out"
            ) from e

    async def _send(self, ws_conn: aiohttp.ClientWebSocketResponse, msg: dict[str, Any]) -> None:
        self.emit("openai_client_event_queued", msg)
        await ws_conn.send_str(json.dumps(msg))
        if lk_oai_debug and msg["type"] != "session.input_audio_buffer.append":
            logger.debug(">>>", extra={"lk.pii.event": msg})

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        closing = False

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal closing
            async for msg in self._msg_ch:
                try:
                    await self._send(ws_conn, msg)
                except Exception:
                    logger.exception("failed to send event")

            closing = True
            await ws_conn.close()

        @utils.log_exceptions(logger=logger)
        async def _recv_task() -> None:
            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:  # closing is expected, see _send_task
                        return

                    # this will trigger a reconnection
                    raise APIConnectionError(
                        message="OpenAI Realtime translation connection closed unexpectedly"
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                event = json.loads(msg.data)
                self.emit("openai_server_event_received", event)

                if lk_oai_debug:
                    event_copy = event
                    if event["type"] == "session.output_audio.delta":
                        event_copy = {**event, "delta": "..."}
                    logger.debug("<<<", extra={"lk.pii.event": event_copy})

                try:
                    self._handle_event(event)
                except Exception as e:
                    # terminal server errors must break the recv loop so _main_task stops
                    # reconnecting; every other handler failure is logged and skipped
                    if isinstance(e, APIError) and not e.retryable:
                        raise
                    if event["type"] == "session.output_audio.delta":
                        event["delta"] = event["delta"][:10] + "..."
                    logger.exception("failed to handle event", extra={"lk.pii.event": event})

        tasks = [
            asyncio.create_task(_recv_task(), name="_recv_task"),
            asyncio.create_task(_send_task(), name="_send_task"),
        ]
        wait_reconnect_task: asyncio.Task[None] | None = None
        if self._opts.max_session_duration is not None:
            wait_reconnect_task = asyncio.create_task(
                asyncio.sleep(self._opts.max_session_duration), name="_timeout_task"
            )
            tasks.append(wait_reconnect_task)
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # propagate exceptions from completed tasks
            for task in done:
                if task != wait_reconnect_task:
                    task.result()
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws_conn.close()

    def _handle_event(self, event: dict[str, Any]) -> None:
        if self._closing:
            # draining after aclose; nothing reads what is left
            return

        event_type = event["type"]
        if event_type == "session.output_audio.delta":
            self._handle_output_audio_delta(event)
        elif event_type == "session.output_transcript.delta":
            self._output_segmenter.push(event.get("delta") or "")
        elif event_type == "session.input_transcript.delta":
            self._input_segmenter.push(event.get("delta") or "")
        elif event_type == "error":
            self._handle_error(event)
        elif lk_oai_debug:
            logger.debug(f"unhandled event: {event_type}", extra={"lk.pii.event": event})

    def _handle_output_audio_delta(self, event: dict[str, Any]) -> None:
        data = base64.b64decode(event["delta"])
        sample_rate = event.get("sample_rate") or SAMPLE_RATE
        num_channels = event.get("channels") or NUM_CHANNELS
        samples_per_channel = len(data) // (2 * num_channels)
        if samples_per_channel == 0:
            return

        frame = rtc.AudioFrame(
            data=data[: samples_per_channel * 2 * num_channels],
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=samples_per_channel,
        )
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._audio_ch.send_nowait(
                TranslatedAudioFrame(frame=frame, language=self._opts.target_language)
            )

    def _handle_error(self, event: dict[str, Any]) -> None:
        error_body = event.get("error") or {}
        logger.error(
            f"OpenAI Realtime translation returned an error: {error_body}",
            extra={"error": error_body},
        )
        recoverable = not _is_fatal_error(
            SimpleNamespace(code=error_body.get("code"), type=error_body.get("type"))
        )
        error = APIError(
            message="OpenAI Realtime translation returned an error",
            body=error_body,
            retryable=recoverable,
        )
        if not recoverable:
            # the recv loop re-raises it so _main_task emits it with recoverable=False and
            # stops reconnecting
            raise error
        self._emit_error(error, recoverable=True)

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
