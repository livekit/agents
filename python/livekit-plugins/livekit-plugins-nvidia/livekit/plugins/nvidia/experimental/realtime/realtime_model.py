"""PersonaPlex real-time model implementation for LiveKit agents.

This module provides a real-time language model using NVIDIA PersonaPlex's
WebSocket API for full-duplex conversational AI with audio I/O.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field, replace
from typing import Literal, TypedDict, assert_never

import numpy as np
import sphn

from livekit import rtc
from livekit.agents import APIConnectOptions, llm, utils
from livekit.agents.llm import realtime as llm_realtime
from livekit.agents.metrics import Metadata, RealtimeModelMetrics
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr

from .log import logger
from .models import PersonaplexVoice

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

# Message type prefixes for the PersonaPlex binary WebSocket protocol
MSG_HANDSHAKE = 0x00
MSG_AUDIO = 0x01
MSG_TEXT = 0x02

# Special text tokens to ignore (padding/EOS markers)
_SPECIAL_TOKENS = {0, 3}

DEFAULT_SILENCE_THRESHOLD_MS = 500
OPUS_FRAME_MS = 80  # a valid Opus frame size


@dataclass
class _PersonaplexOptions:
    base_url: str
    voice: str
    text_prompt: str
    seed: int | None
    silence_threshold_ms: int
    conn_options: APIConnectOptions
    use_ssl: bool = False


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]

    response_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]

    created_timestamp: float = field(default_factory=time.time)
    first_token_timestamp: float | None = None
    completed_timestamp: float | None = None
    done: bool = False
    output_text: str = ""


class RealtimeModel(llm.RealtimeSessionModel):
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
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
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
            conn_options: Connection timeouts, and the retries after a lost connection.
        """
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=False,
                user_transcription=False,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
                per_response_tool_choice=False,
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
            conn_options=conn_options,
            use_ssl=use_ssl,
        )
        self._label = f"personaplex-{voice}"

    @property
    def model(self) -> str:
        return "personaplex-7b"

    @property
    def provider(self) -> str:
        return "nvidia"

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        # disabling server-side turn detection is unsupported (can_disable_turn_detection=False)
        return RealtimeSession(realtime_model=self, options=self._opts)


# the server's binary frames: a type byte, then its payload


class Handshake(TypedDict):
    type: Literal["handshake"]


class Audio(TypedDict):
    type: Literal["audio"]
    opus: bytes


class Text(TypedDict):
    type: Literal["text"]
    text: str


ServerMessage = Handshake | Audio | Text


class RealtimeSession(llm.RealtimeWebSocketSession[ServerMessage, bytes]):
    """A full-duplex PersonaPlex conversation: Opus audio both ways, text tokens from the server.
    The persona is fixed per connection, so new instructions open a new one."""

    frame_ms = OPUS_FRAME_MS
    # the server reads audio only once it sent its handshake
    awaits_ready = True

    def __init__(self, realtime_model: RealtimeModel, *, options: _PersonaplexOptions) -> None:
        super().__init__(
            realtime_model,
            conn_options=options.conn_options,
            input_sample_rate=SAMPLE_RATE,
            input_num_channels=NUM_CHANNELS,
        )
        self._realtime_model: RealtimeModel = realtime_model
        self._opts = replace(options)
        self._tools = llm.ToolContext.empty()
        self._chat_ctx = llm.ChatContext.empty()
        self._current_generation: _ResponseGeneration | None = None
        self._silence_timer_handle: asyncio.TimerHandle | None = None
        self._opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
        self._opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
        self._connect_started = 0.0

    async def connect(self, conn_options: APIConnectOptions) -> llm_realtime.WebSocketTarget:
        self._connect_started = time.perf_counter()
        self._opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
        self._opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
        self._finalize_generation(interrupted=True)
        params = {"voice_prompt": f"{self._opts.voice}.pt", "text_prompt": self._opts.text_prompt}
        if self._opts.seed is not None:
            params["seed"] = str(self._opts.seed)
        scheme = "wss" if self._opts.use_ssl else "ws"
        return llm_realtime.WebSocketTarget(
            f"{scheme}://{self._opts.base_url}/api/chat", params=params
        )

    def decode(self, frame: bytes | str) -> ServerMessage | None:
        if isinstance(frame, str):
            raise llm_realtime.protocol_error(self, f"a text frame: {frame}")
        if not frame:
            return None
        kind, payload = frame[0], frame[1:]
        if kind == MSG_HANDSHAKE:
            return Handshake(type="handshake")
        if kind == MSG_AUDIO:
            return Audio(type="audio", opus=payload)
        if kind == MSG_TEXT:
            # padding and end-of-stream markers
            if len(payload) == 1 and payload[0] in _SPECIAL_TOKENS:
                return None
            try:
                return Text(type="text", text=payload.decode("utf-8"))
            except UnicodeDecodeError as e:
                raise llm_realtime.protocol_error(
                    self, f"a text token that is not UTF-8: {e}"
                ) from e
        return None

    def encode(self, item: rtc.AudioFrame) -> bytes | None:
        if not item.data:
            return None
        pcm = np.frombuffer(item.data, dtype=np.int16).astype(np.float32) / 32768.0
        opus = self._opus_writer.append_pcm(pcm)
        if not opus:
            return None
        return bytes([MSG_AUDIO]) + opus

    def on_message(self, message: ServerMessage) -> None:
        if message["type"] == "handshake":
            self._report_connection_acquired(time.perf_counter() - self._connect_started)
            self.report(llm_realtime.Ready())
        elif message["type"] == "audio":
            self._handle_audio_data(message["opus"])
        elif message["type"] == "text":
            if message["text"]:
                gen = self._active_generation()
                gen.text_ch.send_nowait(message["text"])
                gen.output_text += message["text"]
        else:
            assert_never(message)

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass  # PersonaPlex is audio-only

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        raise NotImplementedError(
            "generate_reply is not yet supported by the PersonaPlex realtime model."
        )

    def interrupt(self) -> None:
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

    async def update_instructions(self, instructions: str) -> None:
        if self._opts.text_prompt != instructions:
            self._opts.text_prompt = instructions
            self.reconnect("instructions changed")

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self._chat_ctx = chat_ctx.copy()
        logger.debug("PersonaPlex does not support dynamic chat context updates.")

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        logger.debug("PersonaPlex does not support tools.")

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        pass

    async def aclose(self) -> None:
        await super().aclose()
        self._finalize_generation(interrupted=True)

    def _handle_audio_data(self, opus: bytes) -> None:
        pcm_float = self._opus_reader.append_bytes(opus)
        if len(pcm_float) == 0:
            return
        pcm = np.clip(pcm_float * 32768.0, -32768, 32767).astype(np.int16)
        gen = self._active_generation()
        if gen.first_token_timestamp is None:
            gen.first_token_timestamp = time.time()
        gen.audio_ch.send_nowait(
            rtc.AudioFrame(
                data=pcm.tobytes(),
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(pcm),
            )
        )
        # the model has no turn events: a generation ends after a silence
        self._cancel_silence_timer()
        self._silence_timer_handle = asyncio.get_running_loop().call_later(
            self._opts.silence_threshold_ms / 1000.0, self._finalize_generation
        )

    def _active_generation(self) -> _ResponseGeneration:
        gen = self._current_generation
        if gen is not None and not gen.done:
            return gen
        response_id = utils.shortuuid("personaplex-turn-")
        gen = self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            response_id=response_id,
            text_ch=utils.aio.Chan[str](),
            audio_ch=utils.aio.Chan[rtc.AudioFrame](),
        )
        modalities = asyncio.get_running_loop().create_future()
        modalities.set_result(["audio", "text"])
        gen.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=response_id,
                text_stream=gen.text_ch,
                audio_stream=gen.audio_ch,
                modalities=modalities,
            )
        )
        self.emit(
            "generation_created",
            llm.GenerationCreatedEvent(
                message_stream=gen.message_ch,
                function_stream=gen.function_ch,
                user_initiated=False,
                response_id=response_id,
            ),
        )
        return gen

    def _finalize_generation(self, *, interrupted: bool = False) -> None:
        self._cancel_silence_timer()
        gen = self._current_generation
        if gen is None or gen.done:
            return
        gen.completed_timestamp = time.time()
        gen.done = True
        gen.text_ch.close()
        gen.audio_ch.close()
        gen.function_ch.close()
        gen.message_ch.close()
        if gen.output_text:
            self._chat_ctx.add_message(
                role="assistant", content=gen.output_text, id=gen.response_id
            )
        self._current_generation = None
        # a generation that produced nothing has no metrics
        if gen.first_token_timestamp is None and not gen.output_text:
            return
        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                timestamp=gen.created_timestamp,
                request_id=gen.response_id,
                ttft=(
                    gen.first_token_timestamp - gen.created_timestamp
                    if gen.first_token_timestamp
                    else None
                ),
                duration=gen.completed_timestamp - gen.created_timestamp,
                cancelled=interrupted,
                label=self._realtime_model.label,
                tokens_per_second=0,
                metadata=Metadata(
                    model_name=self._realtime_model.model,
                    model_provider=self._realtime_model.provider,
                ),
            ),
        )

    def _cancel_silence_timer(self) -> None:
        if self._silence_timer_handle:
            self._silence_timer_handle.cancel()
            self._silence_timer_handle = None
