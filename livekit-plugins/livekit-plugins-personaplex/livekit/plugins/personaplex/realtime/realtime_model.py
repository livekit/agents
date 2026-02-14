"""PersonaPlex real-time model implementation for LiveKit agents.

This module provides a real-time language model using NVIDIA PersonaPlex's
WebSocket API for full-duplex conversational AI with audio I/O.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Literal
from urllib.parse import quote, urlencode

import aiohttp
import numpy as np
import sphn  # type: ignore[import-untyped]

from livekit import rtc
from livekit.agents import APIConnectionError, llm, utils
from livekit.agents.metrics.base import Metadata, RealtimeModelMetrics
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ..log import logger
from ..models import PersonaplexVoice

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

# Message type prefixes for the PersonaPlex binary WebSocket protocol
MSG_HANDSHAKE = 0x00
MSG_AUDIO = 0x01
MSG_TEXT = 0x02

# Special text tokens to ignore (padding/EOS markers)
_SPECIAL_TOKENS = {0, 3}

DEFAULT_SILENCE_THRESHOLD_MS = 500
MAX_RETRY_DELAY = 30.0
INITIAL_RETRY_DELAY = 1.0


@dataclass
class _PersonaplexOptions:
    base_url: str
    voice: str
    text_prompt: str
    seed: int | None
    silence_threshold_ms: int
    use_ssl: bool = False


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]

    response_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]

    _created_timestamp: float = field(default_factory=time.time)
    _first_token_timestamp: float | None = None
    _completed_timestamp: float | None = None
    _done: bool = False
    output_text: str = ""


class RealtimeModel(llm.RealtimeModel):
    """Real-time language model using NVIDIA PersonaPlex.

    Connects to a PersonaPlex WebSocket server for full-duplex
    audio-in/audio-out conversational AI. The model handles speech
    recognition, language understanding, and speech synthesis in a
    single end-to-end model.

    The server must be running separately (e.g., via `moshi-server`).
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        voice: PersonaplexVoice | str = "NATF2",
        text_prompt: str = "You are a helpful assistant.",
        seed: int | None = None,
        silence_threshold_ms: int = DEFAULT_SILENCE_THRESHOLD_MS,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Initialize the PersonaPlex RealtimeModel.

        Args:
            base_url: WebSocket URL of the PersonaPlex server
                (e.g. "ws://localhost:8998"). If not set, reads from
                PERSONAPLEX_URL env var. Defaults to "ws://localhost:8998".
            voice: Voice prompt to use. One of the 18 available voices
                (e.g. "NATF2", "NATM0", "VARF1").
            text_prompt: System instruction / persona description for
                the model. Set at connection time.
            seed: Optional seed for reproducible generation.
            silence_threshold_ms: Duration of silence (no audio from server)
                before finalizing a generation. Default 500ms.
            http_session: Optional aiohttp session to reuse.
        """
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=False,
                user_transcription=False,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
            )
        )

        resolved_url: str = base_url or os.environ.get("PERSONAPLEX_URL") or "localhost:8998"
        # Detect SSL from the scheme before stripping it
        use_ssl = resolved_url.startswith(("wss://", "https://"))
        for prefix in ("ws://", "wss://", "http://", "https://"):
            if resolved_url.startswith(prefix):
                resolved_url = resolved_url[len(prefix) :]
                break

        self._opts = _PersonaplexOptions(
            base_url=resolved_url,
            voice=voice,
            text_prompt=text_prompt,
            seed=seed,
            silence_threshold_ms=silence_threshold_ms,
            use_ssl=use_ssl,
        )

        self._http_session_owned = False
        self._http_session = http_session
        self._label = f"personaplex-{voice}"
        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return "personaplex-7b"

    @property
    def provider(self) -> str:
        return "nvidia"

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session_owned = True
            self._http_session = utils.http_context.http_session()
        return self._http_session

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(realtime_model=self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        if self._http_session_owned and self._http_session:
            await self._http_session.close()


class RealtimeSession(llm.RealtimeSession[Literal["personaplex_server_event"]]):
    """Manages a WebSocket connection to a PersonaPlex server.

    Handles bidirectional binary audio streaming with Opus encoding,
    generation lifecycle management, and text token handling.
    """

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._opts = replace(realtime_model._opts)

        self._tools = llm.ToolContext.empty()
        self._chat_ctx = llm.ChatContext.empty()
        self._msg_ch = utils.aio.Chan[bytes]()

        self._input_resampler: rtc.AudioResampler | None = None
        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE,
            NUM_CHANNELS,
            samples_per_channel=SAMPLE_RATE // 10,  # 100ms frames
        )

        self._opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
        self._opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)

        self._current_generation: _ResponseGeneration | None = None
        self._pending_generation_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        self._silence_timer_handle: asyncio.TimerHandle | None = None

        self._handshake_event = asyncio.Event()
        self._session_should_close = asyncio.Event()
        self._closed = False
        self._closing = False

        self._main_atask = asyncio.create_task(
            self._main_task(), name="PersonaplexSession._main_task"
        )

    # -- Properties --

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools

    # -- Public API: audio input --

    @utils.log_exceptions(logger=logger)
    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return

        for resampled_frame in self._resample_audio(frame):
            for audio_frame in self._bstream.push(resampled_frame.data):
                self._encode_and_send(audio_frame)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass  # PersonaPlex is audio-only

    # -- Public API: generation control --

    @utils.log_exceptions(logger=logger)
    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        if self._pending_generation_fut and not self._pending_generation_fut.done():
            self._pending_generation_fut.cancel("Superseded by new generate_reply")

        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._pending_generation_fut = fut

        if is_given(instructions):
            logger.warning(
                "PersonaPlex does not support dynamic instructions. "
                "Instruction changes require reconnection via update_instructions()."
            )

        def _on_timeout() -> None:
            if not fut.done():
                fut.set_exception(
                    llm.RealtimeError(
                        "generate_reply timed out waiting for generation_created event."
                    )
                )
                if self._pending_generation_fut is fut:
                    self._pending_generation_fut = None

        timeout_handle = asyncio.get_running_loop().call_later(10.0, _on_timeout)
        fut.add_done_callback(lambda _: timeout_handle.cancel())
        return fut

    def interrupt(self) -> None:
        if self._current_generation and not self._current_generation._done:
            self._finalize_generation(interrupted=True)

    def commit_audio(self) -> None:
        pass  # Full-duplex, continuous streaming

    def commit_user_turn(self) -> None:
        logger.warning("commit_user_turn is not supported by PersonaPlex.")

    def clear_audio(self) -> None:
        pass  # No server-side audio buffer

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        logger.debug("truncate is not supported by PersonaPlex.")

    # -- Public API: updates --

    async def update_instructions(self, instructions: str) -> None:
        if self._opts.text_prompt != instructions:
            self._opts.text_prompt = instructions
            self._mark_restart_needed()

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self._chat_ctx = chat_ctx.copy()
        logger.debug("PersonaPlex does not support dynamic chat context updates.")

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        logger.debug("PersonaPlex does not support tools.")

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        pass

    # -- Lifecycle --

    async def aclose(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._closing = True
        self._msg_ch.close()
        self._session_should_close.set()

        await utils.aio.cancel_and_wait(self._main_atask)

        if self._pending_generation_fut and not self._pending_generation_fut.done():
            self._pending_generation_fut.cancel("Session closed")

        if self._current_generation and not self._current_generation._done:
            self._finalize_generation(interrupted=True)

    # -- Internal: connection management --

    def _mark_restart_needed(self) -> None:
        if not self._session_should_close.is_set():
            self._session_should_close.set()
            old_ch = self._msg_ch
            old_ch.close()
            self._msg_ch = utils.aio.Chan[bytes]()

            if self._pending_generation_fut and not self._pending_generation_fut.done():
                self._pending_generation_fut.cancel("Session restart")
            self._pending_generation_fut = None

    def _build_ws_url(self) -> str:
        params: dict[str, str] = {
            "voice_prompt": f"{self._opts.voice}.pt",
            "text_prompt": self._opts.text_prompt,
        }
        if self._opts.seed is not None:
            params["seed"] = str(self._opts.seed)

        query = urlencode(params, quote_via=quote)
        scheme = "wss" if self._opts.use_ssl else "ws"
        return f"{scheme}://{self._opts.base_url}/api/chat?{query}"

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        retry_delay = INITIAL_RETRY_DELAY

        while not self._msg_ch.closed:
            self._session_should_close.clear()
            self._handshake_event.clear()

            # Reset codec state for new connection
            self._opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
            self._opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)

            try:
                ws_url = self._build_ws_url()
                http_session = self._realtime_model._ensure_http_session()

                ws_conn = await http_session.ws_connect(ws_url)
                self._closing = False
                retry_delay = INITIAL_RETRY_DELAY  # reset on successful connect

                logger.info(f"Connected to PersonaPlex server at {self._opts.base_url}")

                send_task = asyncio.create_task(self._send_task(ws_conn), name="_send_task")
                recv_task = asyncio.create_task(self._recv_task(ws_conn), name="_recv_task")
                restart_wait_task = asyncio.create_task(
                    self._session_should_close.wait(), name="_restart_wait"
                )

                try:
                    done, _ = await asyncio.wait(
                        [send_task, recv_task, restart_wait_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task != restart_wait_task:
                            task.result()
                finally:
                    await ws_conn.close()
                    await utils.aio.cancel_and_wait(send_task, recv_task, restart_wait_task)

                if restart_wait_task not in done and self._msg_ch.closed:
                    break

                if restart_wait_task in done:
                    self.emit(
                        "session_reconnected",
                        llm.RealtimeSessionReconnectedEvent(),
                    )

            except Exception as e:
                logger.error(f"PersonaPlex WebSocket error: {e}", exc_info=True)

                is_recoverable = isinstance(
                    e, (aiohttp.ClientConnectionError, asyncio.TimeoutError, APIConnectionError)
                )

                if isinstance(e, APIConnectionError):
                    error = e
                else:
                    error = APIConnectionError(f"Connection failed: {e}")

                self.emit(
                    "error",
                    llm.RealtimeModelError(
                        timestamp=time.time(),
                        label=self._realtime_model._label,
                        error=error,
                        recoverable=is_recoverable,
                    ),
                )

                if not is_recoverable or self._msg_ch.closed:
                    break

                logger.debug(f"Retrying in {retry_delay:.1f}s")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)

    @utils.log_exceptions(logger=logger)
    async def _send_task(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        async for msg in self._msg_ch:
            if self._session_should_close.is_set():
                break

            try:
                await ws_conn.send_bytes(msg)
            except Exception as e:
                logger.error(f"Error sending message: {e}", exc_info=True)
                break

        self._closing = True

    @utils.log_exceptions(logger=logger)
    async def _recv_task(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            if self._session_should_close.is_set():
                break

            msg = await ws_conn.receive()

            if msg.type == aiohttp.WSMsgType.BINARY:
                data = msg.data
                if len(data) == 0:
                    continue

                msg_type = data[0]
                payload = data[1:]

                try:
                    if msg_type == MSG_HANDSHAKE:
                        logger.debug("PersonaPlex handshake received")
                        self._handshake_event.set()

                    elif msg_type == MSG_AUDIO:
                        self._handle_audio_data(payload)

                    elif msg_type == MSG_TEXT:
                        self._handle_text_token(payload)

                    else:
                        logger.warning(f"Unknown PersonaPlex message type: 0x{msg_type:02x}")
                except Exception:
                    logger.exception("Error handling PersonaPlex message")

            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                if self._closing:
                    return
                raise APIConnectionError(message="PersonaPlex connection closed unexpectedly")

            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError(
                    message=f"PersonaPlex WebSocket error: {ws_conn.exception()}"
                )

    # -- Internal: audio encode/decode --

    def _encode_and_send(self, audio_frame: rtc.AudioFrame) -> None:
        """Encode a PCM audio frame to Opus and queue for sending."""
        if not audio_frame.data or len(audio_frame.data) == 0:
            return

        try:
            # Convert int16 PCM to float32 for sphn
            pcm_int16 = np.frombuffer(audio_frame.data, dtype=np.int16)
            if pcm_int16.size == 0:
                return

            pcm_float = pcm_int16.astype(np.float32) / 32768.0

            # sphn >=0.2: append_pcm returns opus bytes directly
            opus_bytes = self._opus_writer.append_pcm(pcm_float)

            if opus_bytes:
                # Prepend audio message type
                message = bytes([MSG_AUDIO]) + opus_bytes
                with contextlib.suppress(utils.aio.channel.ChanClosed):
                    self._msg_ch.send_nowait(message)
        except (TypeError, ValueError) as e:
            logger.warning(f"Skipping invalid audio frame in _encode_and_send: {e}")

    def _handle_audio_data(self, opus_payload: bytes) -> None:
        """Decode Opus audio from server and push to generation."""
        try:
            # sphn >=0.2: append_bytes returns pcm directly
            pcm_float = self._opus_reader.append_bytes(opus_payload)

            if pcm_float is None or len(pcm_float) == 0:
                return

            # Convert float32 to int16 PCM
            pcm_int16 = np.clip(pcm_float * 32768.0, -32768, 32767).astype(np.int16)
            pcm_bytes = pcm_int16.tobytes()

            # Ensure generation exists
            if not self._current_generation or self._current_generation._done:
                self._start_new_generation()

            gen = self._current_generation
            assert gen is not None

            if gen._first_token_timestamp is None and len(pcm_bytes) > 0:
                gen._first_token_timestamp = time.time()
                self._resolve_pending_generation(gen)

            frame = rtc.AudioFrame(
                data=pcm_bytes,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(pcm_int16),
            )
            with contextlib.suppress(utils.aio.channel.ChanClosed):
                gen.audio_ch.send_nowait(frame)

            # Reset silence timer on every audio frame
            self._reset_silence_timer()

        except Exception as e:
            logger.error(f"Error processing audio data: {e}")

    def _handle_text_token(self, payload: bytes) -> None:
        """Handle text token from server."""
        try:
            # Filter special tokens by raw byte value (padding/EOS markers)
            if len(payload) == 1 and payload[0] in _SPECIAL_TOKENS:
                return

            text = payload.decode("utf-8")

            if not text:
                return

            # Ensure generation exists
            if not self._current_generation or self._current_generation._done:
                self._start_new_generation()

            gen = self._current_generation
            assert gen is not None

            with contextlib.suppress(utils.aio.channel.ChanClosed):
                gen.text_ch.send_nowait(text)
            gen.output_text += text

        except Exception as e:
            logger.error(f"Error processing text token: {e}")

    # -- Internal: generation lifecycle --

    def _start_new_generation(self) -> None:
        if self._current_generation and not self._current_generation._done:
            logger.debug("Starting new generation while another is active. Finalizing previous.")
            self._finalize_generation(interrupted=True)

        response_id = utils.shortuuid("personaplex-turn-")
        self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            response_id=response_id,
            text_ch=utils.aio.Chan[str](),
            audio_ch=utils.aio.Chan[rtc.AudioFrame](),
        )

        msg_modalities = asyncio.Future[list[Literal["text", "audio"]]]()
        msg_modalities.set_result(["audio", "text"])

        self._current_generation.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=response_id,
                text_stream=self._current_generation.text_ch,
                audio_stream=self._current_generation.audio_ch,
                modalities=msg_modalities,
            )
        )

        generation_ev = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
            response_id=response_id,
        )
        self.emit("generation_created", generation_ev)

        logger.debug(f"Started generation {response_id}")

    def _resolve_pending_generation(self, gen: _ResponseGeneration) -> None:
        """Resolve pending generate_reply future if one exists."""
        if self._pending_generation_fut and not self._pending_generation_fut.done():
            ev = llm.GenerationCreatedEvent(
                message_stream=gen.message_ch,
                function_stream=gen.function_ch,
                user_initiated=True,
                response_id=gen.response_id,
            )
            self._pending_generation_fut.set_result(ev)
            self._pending_generation_fut = None

    def _finalize_generation(self, *, interrupted: bool = False) -> None:
        if not self._current_generation or self._current_generation._done:
            return

        gen = self._current_generation
        gen._completed_timestamp = time.time()
        gen._done = True

        if not gen.text_ch.closed:
            gen.text_ch.close()
        if not gen.audio_ch.closed:
            gen.audio_ch.close()

        gen.function_ch.close()
        gen.message_ch.close()

        self._cancel_silence_timer()

        # Append assistant message to local chat context
        if gen.output_text:
            self._chat_ctx.add_message(
                role="assistant",
                content=gen.output_text,
                id=gen.response_id,
            )

        self._emit_generation_metrics(interrupted=interrupted)

    def _emit_generation_metrics(self, *, interrupted: bool) -> None:
        if self._current_generation is None:
            return

        gen = self._current_generation
        if gen._first_token_timestamp is None and not gen.output_text:
            self._current_generation = None
            return

        current_time = time.time()
        completed_ts = gen._completed_timestamp or current_time
        created_ts = gen._created_timestamp
        first_token_ts = gen._first_token_timestamp

        ttft = first_token_ts - created_ts if first_token_ts else -1
        duration = completed_ts - created_ts

        metrics = RealtimeModelMetrics(
            timestamp=created_ts,
            request_id=gen.response_id,
            ttft=ttft,
            duration=duration,
            cancelled=interrupted,
            label=self._realtime_model.label,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            tokens_per_second=0,
            input_token_details=RealtimeModelMetrics.InputTokenDetails(
                audio_tokens=0,
                cached_tokens=0,
                text_tokens=0,
                cached_tokens_details=None,
                image_tokens=0,
            ),
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                text_tokens=0,
                audio_tokens=0,
                image_tokens=0,
            ),
            metadata=Metadata(
                model_name=self._realtime_model.model,
                model_provider=self._realtime_model.provider,
            ),
        )

        self.emit("metrics_collected", metrics)
        self._current_generation = None

    # -- Internal: silence detection --

    def _reset_silence_timer(self) -> None:
        self._cancel_silence_timer()
        loop = asyncio.get_running_loop()
        threshold_s = self._opts.silence_threshold_ms / 1000.0
        self._silence_timer_handle = loop.call_later(threshold_s, self._on_silence_timeout)

    def _cancel_silence_timer(self) -> None:
        if self._silence_timer_handle:
            self._silence_timer_handle.cancel()
            self._silence_timer_handle = None

    def _on_silence_timeout(self) -> None:
        if self._current_generation and not self._current_generation._done:
            logger.debug("Silence detected, finalizing generation")
            self._finalize_generation(interrupted=False)

    # -- Internal: audio resampling --

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
