from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any, Literal
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, llm, utils
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ..log import logger

# OpenAI Realtime Translation is a *continuous* speech-to-speech translation API.
# Unlike the conversational Realtime API it has no response/turn lifecycle: there
# are no `response.*`, `conversation.*`, `.done`, or server-VAD events. The server
# only streams three kinds of deltas while the speaker talks:
#   - session.input_transcript.delta   (source-language transcript)
#   - session.output_transcript.delta  (translated text)
#   - session.output_audio.delta       (translated audio, base64 PCM16)
# plus session.created / session.updated / session.closed / error.
#
# To map this onto LiveKit's turn-based AgentSession (which is driven by
# `generation_created`), we synthesize turn boundaries from idle gaps in the
# output stream: the first output delta opens a "segment" (one generation), and a
# short idle with no further output deltas closes it. See RealtimeTranslationSession.

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TRANSLATION_MODEL = "gpt-realtime-translate"

# how long the output stream must be idle before the current segment is finalized
DEFAULT_OUTPUT_SEGMENT_IDLE = 0.8
# how long the input transcript must be idle before it is committed to the chat ctx
DEFAULT_INPUT_SEGMENT_IDLE = 1.0

lk_oai_debug = int(os.getenv("LK_OPENAI_DEBUG", 0))


@dataclass
class _TranslationOptions:
    model: str
    target_language: str
    api_key: str
    base_url: str
    transcription_model: str | None
    input_audio_noise_reduction: str | None
    output_segment_idle: float
    input_segment_idle: float
    safety_identifier: str | None
    max_session_duration: float | None
    """recycle the connection after this many seconds if provided"""
    conn_options: APIConnectOptions


@dataclass
class _TranslationSegment:
    """A synthesized "turn" of translated output (one llm.GenerationCreatedEvent)."""

    segment_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    created_timestamp: float
    first_token_timestamp: float | None = None


def process_translation_url(url: str, model: str) -> str:
    """Build the realtime translations websocket URL from a base url + model."""
    if url.startswith("http"):
        url = url.replace("http", "ws", 1)

    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    path_stripped = parsed_url.path.rstrip("/")
    if not parsed_url.path or path_stripped in ("", "/v1", "/openai", "/openai/v1"):
        path = path_stripped + "/realtime/translations"
    else:
        path = parsed_url.path

    if "model" not in query_params:
        query_params["model"] = [model]

    new_query = urlencode(query_params, doseq=True)
    return urlunparse(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            path,
            parsed_url.params,
            new_query,
            parsed_url.fragment,
        )
    )


class RealtimeTranslationModel(llm.RealtimeModel):
    """OpenAI Realtime Translation model (``gpt-realtime-translate``).

    Wraps the dedicated ``/v1/realtime/translations`` endpoint, which performs live
    simultaneous speech-to-speech translation: audio streamed in one language is
    translated to ``target_language`` and returned as audio + transcript deltas
    while the speaker is still talking.

    Use it as a drop-in realtime model, typically with no STT/TTS/VAD::

        session = AgentSession(llm=RealtimeTranslationModel(target_language="es"))

    or drive a :meth:`session` directly (e.g. to translate many tracks into many
    languages at once — see the multi-user translator example).

    Notes:
        - One model instance translates into a single ``target_language`` (one-way).
        - ``allow_interruptions=False`` is incompatible with this model: it reports
          ``turn_detection=True`` (the server segments autonomously), and the voice
          agent forbids disabling interruptions in that mode.
    """

    def __init__(
        self,
        *,
        target_language: str,
        model: str = DEFAULT_TRANSLATION_MODEL,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        transcription_model: NotGivenOr[str | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[str | None] = NOT_GIVEN,
        output_segment_idle: float = DEFAULT_OUTPUT_SEGMENT_IDLE,
        input_segment_idle: float = DEFAULT_INPUT_SEGMENT_IDLE,
        safety_identifier: str | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            target_language: Target language code for the translation, e.g. ``"es"``.
                Sent as ``session.audio.output.language``.
            model: Translation model name. Defaults to ``gpt-realtime-translate``.
            api_key: OpenAI API key. If None, read from ``OPENAI_API_KEY``.
            base_url: Override the API base url (defaults to ``OPENAI_BASE_URL`` env
                or the public OpenAI endpoint).
            transcription_model: Optional model for the source-language transcript,
                sent as ``session.audio.input.transcription.model``.
            input_audio_noise_reduction: Optional noise reduction type (e.g.
                ``"near_field"`` / ``"far_field"``), sent as
                ``session.audio.input.noise_reduction.type``.
            output_segment_idle: Seconds of output-stream silence after which the
                current translated "turn" is finalized. Lower = snappier turn
                boundaries, higher = fewer splits mid-utterance.
            input_segment_idle: Seconds of input-transcript silence after which the
                source transcript is committed to the chat context.
            safety_identifier: Optional value for the ``OpenAI-Safety-Identifier``
                header.
            max_session_duration: Recycle the websocket connection after this many
                seconds. Defaults to None (no proactive reconnect) since recycling
                interrupts the continuous stream.
            http_session: Optional shared aiohttp session.
            conn_options: Retry/backoff and connection settings.
        """
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                # the server segments the stream autonomously: the framework must
                # not run its own VAD/turn-detection nor call commit_audio/generate_reply
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
                mutable_chat_context=False,
                mutable_instructions=False,
                mutable_tools=False,
                per_response_tool_choice=False,
                supports_say=False,
            )
        )

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key "
                "to the client or by setting the OPENAI_API_KEY environment variable"
            )

        base_url_val = (
            base_url if is_given(base_url) else os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)
        )

        self._opts = _TranslationOptions(
            model=model,
            target_language=target_language,
            api_key=api_key,
            base_url=base_url_val,
            transcription_model=transcription_model if is_given(transcription_model) else None,
            input_audio_noise_reduction=(
                input_audio_noise_reduction if is_given(input_audio_noise_reduction) else None
            ),
            output_segment_idle=output_segment_idle,
            input_segment_idle=input_segment_idle,
            safety_identifier=safety_identifier,
            max_session_duration=max_session_duration if is_given(max_session_duration) else None,
            conn_options=conn_options,
        )
        self._http_session = http_session
        self._http_session_owned = False
        self._sessions = weakref.WeakSet[RealtimeTranslationSession]()
        self._provider_label = "OpenAI Realtime Translation API"

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return urlparse(self._opts.base_url).netloc or "openai"

    @property
    def target_language(self) -> str:
        return self._opts.target_language

    def update_options(self, *, target_language: NotGivenOr[str] = NOT_GIVEN) -> None:
        """Update the target language for new and existing sessions."""
        if is_given(target_language):
            self._opts.target_language = target_language

        for sess in self._sessions:
            sess.update_options(target_language=target_language)

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            try:
                self._http_session = utils.http_context.http_session()
            except RuntimeError:
                self._http_session = aiohttp.ClientSession()
                self._http_session_owned = True

        return self._http_session

    def session(self) -> RealtimeTranslationSession:
        sess = RealtimeTranslationSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        if self._http_session_owned and self._http_session:
            await self._http_session.close()


class RealtimeTranslationSession(
    llm.RealtimeSession[Literal["openai_server_event_received", "openai_client_event_queued"]]
):
    """A session for the OpenAI Realtime Translation API.

    It exposes two extra events alongside the standard realtime events:
    - ``openai_server_event_received``: raw server events from the translation API
    - ``openai_client_event_queued``: raw client events sent to the translation API
    """

    def __init__(self, realtime_model: RealtimeTranslationModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeTranslationModel = realtime_model
        # per-session copy of opts so update_options can diff against session state
        self._opts = replace(realtime_model._opts)
        self._msg_ch = utils.aio.Chan[dict[str, Any]]()
        self._input_resampler: rtc.AudioResampler | None = None

        # 100ms chunks
        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )

        # the currently open output segment (None while idle between segments)
        self._current_segment: _TranslationSegment | None = None
        self._output_idle_handle: asyncio.TimerHandle | None = None

        # partial source-language transcript accumulated across deltas
        self._input_item_id: str | None = None
        self._input_accumulator: str = ""
        self._input_idle_handle: asyncio.TimerHandle | None = None

        self._session_id: str | None = None

        self._main_atask = asyncio.create_task(
            self._main_task(), name="RealtimeTranslationSession._main_task"
        )
        self.send_event(self._create_session_update_event())

    # -- properties (no chat context / tools for translation) ---------------------

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return llm.ChatContext.empty()

    @property
    def tools(self) -> llm.ToolContext:
        return llm.ToolContext.empty()

    # -- client event helpers -----------------------------------------------------

    def send_event(self, event: dict[str, Any]) -> None:
        with contextlib.suppress(utils.aio.ChanClosed):
            self._msg_ch.send_nowait(event)

    def _create_session_update_event(self) -> dict[str, Any]:
        audio_input: dict[str, Any] = {}
        if self._opts.transcription_model is not None:
            audio_input["transcription"] = {"model": self._opts.transcription_model}
        if self._opts.input_audio_noise_reduction is not None:
            audio_input["noise_reduction"] = {"type": self._opts.input_audio_noise_reduction}

        audio: dict[str, Any] = {"output": {"language": self._opts.target_language}}
        if audio_input:
            audio["input"] = audio_input

        return {"type": "session.update", "session": {"audio": audio}}

    # -- connection / main loop ---------------------------------------------------

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        num_retries: int = 0
        max_retries = self._opts.conn_options.max_retry

        async def _reconnect(ws_conn: aiohttp.ClientWebSocketResponse) -> None:
            logger.debug(f"reconnecting to {self._realtime_model._provider_label}")
            ev = self._create_session_update_event()
            try:
                self.emit("openai_client_event_queued", ev)
                await ws_conn.send_str(json.dumps(ev))
            except Exception as e:
                raise APIConnectionError(
                    message=(
                        f"Failed to send session update to {self._realtime_model._provider_label} "
                        "during session re-connection"
                    ),
                ) from e

            # discard any in-flight output/input from the dropped connection
            self._finalize_input_transcript()
            self._close_segment()
            self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

        reconnecting = False
        while not self._msg_ch.closed:
            try:
                ws_conn = await self._create_ws_conn()
                if reconnecting:
                    await _reconnect(ws_conn)
                    num_retries = 0
                await self._run_ws(ws_conn)

            except APIError as e:
                if max_retries == 0 or not e.retryable:
                    self._emit_error(e, recoverable=False)
                    raise
                elif num_retries == max_retries:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"{self._realtime_model._provider_label} connection failed after "
                        f"{num_retries} attempts",
                    ) from e
                else:
                    self._emit_error(e, recoverable=True)
                    retry_interval = self._opts.conn_options._interval_for_retry(num_retries)
                    logger.warning(
                        f"{self._realtime_model._provider_label} connection failed, "
                        f"retrying in {retry_interval}s",
                        exc_info=e,
                        extra={"attempt": num_retries, "max_retries": max_retries},
                    )
                    await asyncio.sleep(retry_interval)
                num_retries += 1

            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

            reconnecting = True

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        headers = {
            "User-Agent": "LiveKit Agents",
            "Authorization": f"Bearer {self._opts.api_key}",
        }
        if self._opts.safety_identifier:
            headers["OpenAI-Safety-Identifier"] = self._opts.safety_identifier

        url = process_translation_url(self._opts.base_url, self._opts.model)
        if lk_oai_debug:
            logger.debug(f"connecting to Realtime Translation API: {url}")

        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._realtime_model._ensure_http_session().ws_connect(url=url, headers=headers),
                self._opts.conn_options.timeout,
            )
            self._report_connection_acquired(time.perf_counter() - t0)
            return ws
        except aiohttp.ClientError as e:
            raise APIConnectionError(
                f"{self._realtime_model._provider_label} client connection error"
            ) from e
        except asyncio.TimeoutError as e:
            raise APIConnectionError(
                message=f"{self._realtime_model._provider_label} connection timed out",
            ) from e

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        closing = False

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal closing
            async for msg in self._msg_ch:
                try:
                    self.emit("openai_client_event_queued", msg)
                    await ws_conn.send_str(json.dumps(msg))

                    if lk_oai_debug:
                        msg_copy = msg
                        if msg.get("type") == "session.input_audio_buffer.append":
                            msg_copy = {**msg, "audio": "..."}
                        logger.debug(f">>> {msg_copy}")
                except Exception:
                    logger.exception("failed to send translation event")

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
                    if closing:
                        return
                    raise APIConnectionError(
                        message=f"{self._realtime_model._provider_label} connection closed unexpectedly"  # noqa: E501
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                event = json.loads(msg.data)
                # emit the raw dict so consumers can observe provider-specific events
                self.emit("openai_server_event_received", event)

                try:
                    if lk_oai_debug:
                        event_copy = event
                        if event.get("type") == "session.output_audio.delta":
                            event_copy = {**event, "delta": "..."}
                        logger.debug(f"<<< {event_copy}")

                    self._dispatch_event(event)
                except Exception:
                    if event.get("type") == "session.output_audio.delta" and event.get("delta"):
                        event["delta"] = event["delta"][:10] + "..."
                    logger.exception("failed to handle translation event", extra={"event": event})

        tasks = [
            asyncio.create_task(_recv_task(), name="_recv_task"),
            asyncio.create_task(_send_task(), name="_send_task"),
        ]
        wait_reconnect_task: asyncio.Task | None = None
        if self._opts.max_session_duration is not None:
            wait_reconnect_task = asyncio.create_task(
                asyncio.sleep(self._opts.max_session_duration), name="_timeout_task"
            )
            tasks.append(wait_reconnect_task)

        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task != wait_reconnect_task:
                    task.result()

            if wait_reconnect_task and wait_reconnect_task in done:
                # recycle the connection: finalize anything in flight first
                self._finalize_input_transcript()
                self._close_segment()
                closing = True
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws_conn.close()

    # -- server event dispatch ----------------------------------------------------

    def _dispatch_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "session.created":
            session = event.get("session") or {}
            self._session_id = session.get("id")
        elif etype == "session.updated":
            pass
        elif etype == "session.input_transcript.delta":
            self._on_input_transcript_delta(event.get("delta") or "")
        elif etype == "session.output_transcript.delta":
            self._on_output_transcript_delta(event.get("delta") or "")
        elif etype == "session.output_audio.delta":
            self._on_output_audio_delta(event)
        elif etype == "session.closed":
            self._finalize_input_transcript()
            self._close_segment()
        elif etype == "error":
            self._handle_error(event)
        elif lk_oai_debug:
            logger.debug(f"unhandled translation event: {etype}", extra={"event": event})

    # -- output segmentation (synthesized generations) ----------------------------

    def _ensure_segment(self) -> _TranslationSegment:
        if self._current_segment is not None:
            return self._current_segment

        segment_id = utils.shortuuid("translation_")
        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()

        modalities: asyncio.Future[list[Literal["text", "audio"]]] = asyncio.Future()
        # resolve eagerly: the consumer (_process_one_message) awaits modalities
        # BEFORE reading any text/audio, so resolving it lazily would gate playout
        # on segment end and add full-segment latency to every turn.
        modalities.set_result(["audio", "text"])

        message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=segment_id,
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        # exactly one message per segment; closing these streams is what ends the turn
        message_ch.close()
        function_ch.close()

        segment = _TranslationSegment(
            segment_id=segment_id,
            text_ch=text_ch,
            audio_ch=audio_ch,
            created_timestamp=time.time(),
        )
        self._current_segment = segment

        self.emit(
            "generation_created",
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=False,
                response_id=segment_id,
            ),
        )
        return segment

    def _on_output_transcript_delta(self, delta: str) -> None:
        if not delta:
            return
        segment = self._ensure_segment()
        try:
            segment.text_ch.send_nowait(delta)
        except utils.aio.ChanClosed:
            # belt-and-suspenders: a late delta after the idle timer closed the
            # segment. Open a fresh segment and retry.
            self._current_segment = None
            segment = self._ensure_segment()
            segment.text_ch.send_nowait(delta)
        if segment.first_token_timestamp is None:
            segment.first_token_timestamp = time.time()
        self._reset_output_idle_timer()

    def _on_output_audio_delta(self, event: dict[str, Any]) -> None:
        b64 = event.get("delta")
        if not b64:
            return
        data = base64.b64decode(b64)
        sample_rate = int(event.get("sample_rate") or SAMPLE_RATE)
        num_channels = int(event.get("channels") or NUM_CHANNELS)
        frame = rtc.AudioFrame(
            data=data,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=len(data) // (2 * num_channels),
        )

        segment = self._ensure_segment()
        try:
            segment.audio_ch.send_nowait(frame)
        except utils.aio.ChanClosed:
            self._current_segment = None
            segment = self._ensure_segment()
            segment.audio_ch.send_nowait(frame)
        if segment.first_token_timestamp is None:
            segment.first_token_timestamp = time.time()
        self._reset_output_idle_timer()

    def _reset_output_idle_timer(self) -> None:
        if self._output_idle_handle is not None:
            self._output_idle_handle.cancel()
        self._output_idle_handle = asyncio.get_event_loop().call_later(
            self._opts.output_segment_idle, self._close_segment
        )

    def _close_segment(self) -> None:
        if self._output_idle_handle is not None:
            self._output_idle_handle.cancel()
            self._output_idle_handle = None

        segment = self._current_segment
        if segment is None:
            return
        # clear the reference BEFORE closing the channels so a delta that arrives
        # in the same loop tick opens a fresh segment instead of writing to a
        # closed channel.
        self._current_segment = None

        if not segment.text_ch.closed:
            segment.text_ch.close()
        if not segment.audio_ch.closed:
            segment.audio_ch.close()

        self._emit_segment_metrics(segment)

    def _emit_segment_metrics(self, segment: _TranslationSegment) -> None:
        created = segment.created_timestamp
        ttft = segment.first_token_timestamp - created if segment.first_token_timestamp else -1
        duration = time.time() - created
        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                timestamp=created,
                request_id=segment.segment_id,
                ttft=ttft,
                duration=duration,
                cancelled=False,
                label=self._realtime_model.label,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                tokens_per_second=0.0,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(
                    audio_tokens=0,
                    cached_tokens=0,
                    text_tokens=0,
                    cached_tokens_details=RealtimeModelMetrics.CachedTokenDetails(
                        text_tokens=0, audio_tokens=0, image_tokens=0
                    ),
                    image_tokens=0,
                ),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                    text_tokens=0, audio_tokens=0, image_tokens=0
                ),
                metadata=Metadata(
                    model_name=self._realtime_model.model,
                    model_provider=self._realtime_model.provider,
                ),
            ),
        )

    # -- input (source-language) transcript ---------------------------------------

    def _on_input_transcript_delta(self, delta: str) -> None:
        if not delta:
            return
        if self._input_item_id is None:
            self._input_item_id = utils.shortuuid("item_")
            self._input_accumulator = ""
        self._input_accumulator += delta

        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=self._input_item_id,
                transcript=self._input_accumulator,
                is_final=False,
            ),
        )
        self._reset_input_idle_timer()

    def _reset_input_idle_timer(self) -> None:
        if self._input_idle_handle is not None:
            self._input_idle_handle.cancel()
        self._input_idle_handle = asyncio.get_event_loop().call_later(
            self._opts.input_segment_idle, self._finalize_input_transcript
        )

    def _finalize_input_transcript(self) -> None:
        if self._input_idle_handle is not None:
            self._input_idle_handle.cancel()
            self._input_idle_handle = None

        item_id, transcript = self._input_item_id, self._input_accumulator
        self._input_item_id = None
        self._input_accumulator = ""
        if item_id is None or not transcript:
            return

        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(item_id=item_id, transcript=transcript, is_final=True),
        )

    # -- audio input --------------------------------------------------------------

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        for f in self._resample_audio(frame):
            data = f.data.tobytes()
            for nf in self._bstream.write(data):
                self.send_event(
                    {
                        "type": "session.input_audio_buffer.append",
                        "audio": base64.b64encode(nf.data).decode("utf-8"),
                    }
                )

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
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

    # -- options ------------------------------------------------------------------

    def update_options(
        self,
        *,
        target_language: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,  # unused; no tools
    ) -> None:
        if is_given(target_language) and target_language != self._opts.target_language:
            self._opts.target_language = target_language
            self.send_event(self._create_session_update_event())

    # -- unsupported / no-op realtime methods -------------------------------------

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        # the translation model has no response.create: it translates input audio
        # continuously and autonomously. Return a failed future to match the
        # `-> Future` contract instead of raising synchronously.
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
        fut.set_exception(
            llm.RealtimeError(
                "generate_reply() is not supported by the realtime translation model; "
                "it translates input audio continuously and autonomously"
            )
        )
        return fut

    def interrupt(self) -> None:
        # stop the current translated playout (rarely used for translation)
        self._close_segment()

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        pass

    def commit_audio(self) -> None:
        pass

    def clear_audio(self) -> None:
        pass

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    async def update_instructions(self, instructions: str) -> None:
        pass

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        pass

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        pass

    # -- teardown -----------------------------------------------------------------

    async def aclose(self) -> None:
        self._finalize_input_transcript()
        self._close_segment()
        # ask the server to flush + close gracefully, then drain the send loop
        self.send_event({"type": "session.close"})
        self._msg_ch.close()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_atask

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model._label,
                error=error,
                recoverable=recoverable,
            ),
        )

    def _handle_error(self, event: dict[str, Any]) -> None:
        err = event.get("error")
        message = err.get("message") if isinstance(err, dict) else str(err)
        provider_label = self._realtime_model._provider_label
        logger.error(f"{provider_label} returned an error: {message}", extra={"error": err})
        self._emit_error(
            APIError(message=f"{provider_label} returned an error", body=err, retryable=True),
            recoverable=True,
        )
