"""Ultravox realtime model: a call created over HTTP, then its WebSocket for audio and data
messages."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from pydantic import JsonValue, TypeAdapter, ValidationError

from livekit import rtc
from livekit.agents import APIConnectOptions, http, llm, utils
from livekit.agents.llm.realtime import InputSpeechStartedEvent, InputSpeechStoppedEvent
from livekit.agents.llm.utils import compute_chat_ctx_diff
from livekit.agents.metrics import Metadata, RealtimeModelMetrics
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGiven, NotGivenOr
from livekit.agents.utils import is_given

from ..log import logger
from ..models import UltravoxModel, UltravoxVoice
from ..utils import parse_tools
from .events import (
    ClientMessage,
    ClientToolInvocation,
    ClientToolResult,
    ForcedAgentMessage,
    ServerMessage,
    SetOutputMedium,
    State,
    Transcript,
    UserTextMessage,
)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
NUM_CHANNELS = 1

ULTRAVOX_BASE_URL = "https://api.ultravox.ai/api"
DEFAULT_MODEL = "fixie-ai/ultravox"
DEFAULT_VOICE = "Mark"
REPLY_TIMEOUT = 5.0


class _Call(TypedDict):
    joinUrl: str


_CALL = TypeAdapter(_Call)


@dataclass
class _UltravoxOptions:
    """Configuration options for Ultravox model."""

    model_id: str
    voice: str
    external_voice: NotGivenOr[Mapping[str, JsonValue]]
    api_key: str
    base_url: str
    system_prompt: str
    input_sample_rate: int
    output_sample_rate: int
    temperature: NotGivenOr[float]
    language_hint: NotGivenOr[str]
    max_duration: NotGivenOr[str]
    time_exceeded_message: NotGivenOr[str]
    enable_greeting_prompt: NotGivenOr[bool]
    first_speaker: NotGivenOr[str]
    output_medium: Literal["text", "voice"]
    conn_options: APIConnectOptions


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]

    response_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]

    created_timestamp: float = field(default_factory=time.time)
    """The timestamp when the generation is created"""
    first_token_timestamp: float | None = None
    """The timestamp when the first audio token is received"""
    completed_timestamp: float | None = None
    """The timestamp when the generation is completed"""
    done: bool = False
    """Whether the generation is done (set when the turn is complete)"""
    output_text: str = ""
    """Accumulated output text from agent responses"""


class RealtimeModel(llm.RealtimeModel):
    """Real-time language model using Ultravox.

    Connects to Ultravox's WebSocket API for streaming STT, LLM, and TTS.

    Supports dynamic context injection via deferred messages:
    - System messages are injected as <instruction> tags without triggering responses
    - User messages are sent as regular text messages
    - Enables RAG integration and mid-conversation context updates
    """

    def __init__(
        self,
        *,
        model: UltravoxModel | str = DEFAULT_MODEL,
        voice: UltravoxVoice | str = DEFAULT_VOICE,
        external_voice: NotGivenOr[Mapping[str, JsonValue]] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        output_medium: NotGivenOr[Literal["text", "voice"]] = NOT_GIVEN,
        input_sample_rate: int = INPUT_SAMPLE_RATE,
        output_sample_rate: int = OUTPUT_SAMPLE_RATE,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        language_hint: NotGivenOr[str] = NOT_GIVEN,
        max_duration: NotGivenOr[str] = NOT_GIVEN,
        time_exceeded_message: NotGivenOr[str] = NOT_GIVEN,
        enable_greeting_prompt: NotGivenOr[bool] = NOT_GIVEN,
        first_speaker: NotGivenOr[str] = "FIRST_SPEAKER_USER",
        http_session: http.HttpSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Initialize the Ultravox RealtimeModel.

        Parameters
        ----------
        model_id : str | UltravoxModel
            The Ultravox model to use.
        voice : str | UltravoxVoice
            The voice to use for TTS.
        external_voice : dict[str, JsonValue], optional
            The Ultravox configuration for an external TTS provider. Mutually exclusive with ``voice``.
        api_key : str, optional
            The Ultravox API key. If None, will try to use environment variables.
        base_url : str, optional
            The base URL for the Ultravox API.
        system_prompt : str
            The system prompt for the model.
        output_medium : Literal["text", "voice"], optional
            The output medium to use for the model.
        input_sample_rate : int
            Input audio sample rate.
        output_sample_rate : int
            Output audio sample rate.
        temperature : float, optional
            Controls response randomness (0.0-1.0). Lower values are more deterministic.
        language_hint : str, optional
            LanguageCode hint for better multilingual support (e.g., 'en', 'es', 'fr').
        max_duration : str, optional
            Maximum call duration (e.g., '30m', '1h'). Call ends when exceeded.
        time_exceeded_message : str, optional
            Message to play when max duration is reached.
        enable_greeting_prompt : bool
            Whether to enable greeting prompt if no initial message. Default True.
        first_speaker : str, optional
            Who speaks first ('FIRST_SPEAKER_AGENT' or 'FIRST_SPEAKER_UNSPECIFIED'). If not set, model decides.
        http_session : http.HttpSession, optional
            HTTP session to use for requests.
        conn_options : APIConnectOptions
            Connection timeouts, and the retries after a failed connection (each creates a call).
        """
        output_medium = output_medium if is_given(output_medium) else "voice"

        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=True,
                audio_output=output_medium == "voice",
                manual_function_calls=False,
                per_response_tool_choice=False,
                supports_say=True,
            )
        )

        ultravox_api_key = api_key or os.environ.get("ULTRAVOX_API_KEY")
        if not ultravox_api_key:
            raise ValueError(
                "Ultravox API key is required. "
                "Provide it via api_key parameter or ULTRAVOX_API_KEY environment variable."
            )

        if is_given(external_voice):
            if not external_voice:
                raise ValueError(
                    "external_voice must be a non-empty dict. "
                    "Omit it to use the built-in `voice` instead."
                )
            if voice != DEFAULT_VOICE:
                logger.warning(
                    "both `voice` and `external_voice` were provided; "
                    "`external_voice` takes precedence and `voice` will be ignored."
                )

        self._opts = _UltravoxOptions(
            model_id=model,
            voice=voice,
            external_voice=external_voice,
            api_key=ultravox_api_key,
            base_url=base_url or ULTRAVOX_BASE_URL,
            system_prompt=system_prompt,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            temperature=temperature,
            language_hint=language_hint,
            max_duration=max_duration,
            time_exceeded_message=time_exceeded_message,
            enable_greeting_prompt=enable_greeting_prompt,
            first_speaker=first_speaker,
            output_medium=output_medium,
            conn_options=conn_options,
        )

        self._http_session_owned = False
        self._http_session = http_session
        self._label = f"ultravox-{model}"

    @property
    def model(self) -> str:
        return self._opts.model_id

    @property
    def provider(self) -> str:
        return "Ultravox"

    def _ensure_http_session(self) -> http.HttpSession:
        """Ensure HTTP session is available."""
        if self._http_session is None:
            self._http_session_owned = True
            self._http_session = http.http_session()
        return self._http_session

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        """Create a new Ultravox real-time session.

        Returns
        -------
        RealtimeSession
            An instance of the Ultravox real-time session.
        """
        # disabling server-side turn detection is unsupported (can_disable_turn_detection=False)
        return RealtimeSession(
            realtime_model=self, options=self._opts, http_session=self._ensure_http_session
        )

    def update_options(
        self, *, output_medium: NotGivenOr[Literal["text", "voice"]] = NOT_GIVEN
    ) -> None:
        """Update model options."""

        # capabilities are fixed at construction: the output medium decides `audio_output`
        if (
            is_given(output_medium)
            and (output_medium == "voice") != self._capabilities.audio_output
        ):
            raise ValueError(
                "output_medium is fixed at construction: it sets the audio_output capability; "
                "create a new RealtimeModel to change it"
            )

    async def aclose(self) -> None:
        if self._http_session_owned and self._http_session:
            await self._http_session.aclose()


_BARGE_IN = UserTextMessage(
    type="user_text_message", text="", urgency="immediate", deferResponse=True
)


class RealtimeSession(llm.RealtimeSocketSession[ServerMessage, ClientMessage]):
    """One Ultravox call per connection: a change Ultravox applies only to a new call (the
    instructions, the tools) creates one."""

    frame_ms = 100

    def __init__(
        self,
        realtime_model: RealtimeModel,
        *,
        options: _UltravoxOptions,
        http_session: Callable[[], http.HttpSession],
    ) -> None:
        super().__init__(
            realtime_model,
            conn_options=options.conn_options,
            input_sample_rate=options.input_sample_rate,
            input_num_channels=NUM_CHANNELS,
        )
        self._realtime_model: RealtimeModel = realtime_model
        self._get_http_session = http_session
        self._opts = options
        self._tools = llm.ToolContext.empty()
        self._chat_ctx = llm.ChatContext.empty()
        self._current_generation: _ResponseGeneration | None = None
        self._pending_generation_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        # whether the pending reply's request went out: only then does a reply resolve it
        self._pending_requested = False
        self._say_tasks = utils.aio.TaskSet()
        self._last_user_final_ts: float | None = None
        self._connect_started = 0.0

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        self._connect_started = time.perf_counter()
        payload: dict[str, JsonValue] = {
            "systemPrompt": self._opts.system_prompt,
            "model": self._opts.model_id,
            "medium": {
                "serverWebSocket": {
                    "inputSampleRate": self._opts.input_sample_rate,
                    "outputSampleRate": self._opts.output_sample_rate,
                    "clientBufferSizeMs": 30000,
                }
            },
            "selectedTools": list(parse_tools(list(self._tools.function_tools.values()))),
        }
        if is_given(self._opts.external_voice):
            payload["externalVoice"] = dict(self._opts.external_voice)
        else:
            payload["voice"] = self._opts.voice
        optional: dict[str, NotGivenOr[JsonValue]] = {
            "temperature": self._opts.temperature,
            "languageHint": self._opts.language_hint,
            "maxDuration": self._opts.max_duration,
            "timeExceededMessage": self._opts.time_exceeded_message,
            "firstSpeaker": self._opts.first_speaker,
        }
        payload.update({k: v for k, v in optional.items() if is_given(v)})
        params = {} if self._opts.enable_greeting_prompt else {"enableGreetingPrompt": "false"}

        async with self._get_http_session().post(
            f"{self._opts.base_url.rstrip('/')}/calls",
            json=payload,
            params=params,
            headers={"User-Agent": "LiveKit Agents", "X-API-Key": self._opts.api_key},
            conn_options=conn_options,
        ) as resp:
            await resp.raise_for_status()
            body = await resp.json()
        try:
            join_url = _CALL.validate_python(body)["joinUrl"]
        except ValidationError as e:
            raise http.protocol_error(self, f"the created call has no joinUrl: {e}") from e

        if self._opts.output_medium == "text":
            self.send_command(SetOutputMedium(type="set_output_medium", medium="text"))
        return http.SocketTarget(join_url)

    def encode(self, item: rtc.AudioFrame | http.EndOfInput) -> http.Encoded[ClientMessage]:
        return bytes(item.data) if isinstance(item, rtc.AudioFrame) else None

    def on_message(self, message: ServerMessage) -> None:
        if isinstance(message, bytes):
            self._handle_audio_data(message)
        elif message["type"] == "transcript":
            self._handle_transcript(message)
        elif message["type"] == "state":
            self._handle_state(message)
        elif message["type"] == "client_tool_invocation":
            self._handle_tool_invocation(message)
        elif message["type"] == "playback_clear_buffer":
            self.emit("input_speech_started", InputSpeechStartedEvent())
        elif message["type"] == "call_started":
            self._report_connection_acquired(time.perf_counter() - self._connect_started)

    # Helper function to fix TTFT issue : TTFT was showing -1.0 seconds during function calls
    def _pick_created_timestamp(self) -> float:
        """The last user-final timestamp when recent, else now: avoids a tiny TTFT (created too
        late) and a stale one (created too early)."""
        now = time.time()
        if self._last_user_final_ts is not None and 0 <= now - self._last_user_final_ts <= 10.0:
            return self._last_user_final_ts
        return now

    def update_options(
        self,
        *,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        output_medium: NotGivenOr[Literal["text", "voice"]] = NOT_GIVEN,
    ) -> None:
        if is_given(output_medium):
            self.send_command(SetOutputMedium(type="set_output_medium", medium=output_medium))
        if is_given(tool_choice):
            logger.warning("tool choice updates are not supported by Ultravox.")

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """Sends the new items as Ultravox deferred messages: system and developer messages as
        ``<instruction>`` text, user messages as text, tool outputs as tool results. Assistant
        messages are Ultravox's own, and deletions are not supported."""
        diff_ops = compute_chat_ctx_diff(self._chat_ctx, chat_ctx)
        if not diff_ops.to_create:
            return
        if diff_ops.to_remove:
            logger.warning(
                f"ignoring {len(diff_ops.to_remove)} message deletions (not supported by Ultravox)"
            )

        for created in diff_ops.to_create:
            item = chat_ctx.get_by_id(created.item_id)
            if item is None:
                continue
            if item.type == "message" and item.role in ("system", "developer"):
                if item.raw_text_content:
                    self._send_text(f"<instruction>{item.raw_text_content}</instruction>")
            elif item.type == "message" and item.role == "user":
                if item.raw_text_content:
                    self._send_text(item.raw_text_content)
            elif item.type == "function_call_output":
                result = ClientToolResult(
                    type="client_tool_result",
                    invocationId=item.call_id,
                    # a result that wants no reply is recorded without speech
                    agentReaction="speaks" if item.reply_required else "listens",
                    responseType="tool-response",
                )
                if item.is_error:
                    result["errorType"] = "implementation-error"
                    result["errorMessage"] = item.output
                else:
                    result["result"] = item.output
                self.send_command(result)

        self._chat_ctx = chat_ctx.copy()

    def _send_text(self, text: str, *, defer_response: bool = True) -> None:
        self.send_command(
            UserTextMessage(type="user_text_message", text=text, deferResponse=defer_response)
        )

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        current = set(self._tools.function_tools)
        self._tools.update_tools(tools)
        if current != set(self._tools.function_tools):
            self.reconnect("tools changed")

    async def update_instructions(self, instructions: str | NotGiven = NOT_GIVEN) -> None:
        if is_given(instructions) and self._opts.system_prompt != instructions:
            self._opts.system_prompt = instructions
            self.reconnect("instructions changed")

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools

    def push_video(self, frame: rtc.VideoFrame) -> None:
        logger.warning("push_video is not supported by Ultravox.")

    def _supersede_pending(self, reason: str) -> None:
        # the slot is cleared first, so the done callback reads it as superseded, not cancelled
        if (old := self._pending_generation_fut) is not None and not old.done():
            self._pending_generation_fut = None
            old.cancel(reason)

    def _new_pending(self) -> asyncio.Future[llm.GenerationCreatedEvent]:
        fut = asyncio.get_running_loop().create_future()
        self._pending_generation_fut = fut
        self._pending_requested = False

        def _on_done(f: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            if self._pending_generation_fut is f:
                self._pending_generation_fut = None
                if f.cancelled() and self._pending_requested:
                    # cancelled by its caller after the request went out: barge in so Ultravox
                    # does not answer
                    self.send_command(_BARGE_IN)

        fut.add_done_callback(_on_done)
        return fut

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        if is_given(tools):
            logger.warning("per-response tools is not supported by Ultravox Realtime API, ignoring")
        self._supersede_pending("Superseded by new generate_reply call")
        fut = self._new_pending()
        # Ultravox echoes the text back as a user transcript
        text = f"<instruction>{instructions}</instruction>" if is_given(instructions) else ""
        self._send_text(text, defer_response=False)
        self._pending_requested = True
        self._say_tasks.spawn(self._expire(fut, "generate_reply"), name="ultravox-reply-timeout")
        return fut

    def say(self, text: str | AsyncIterable[str]) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """Speaks ``text`` verbatim (``ForcedAgentMessage``), uninterruptible."""
        self._supersede_pending("Superseded by say()")
        fut = self._new_pending()
        self._say_tasks.spawn(self._say_task(text, fut), name="ultravox-say")
        return fut

    async def _say_task(
        self, text: str | AsyncIterable[str], fut: asyncio.Future[llm.GenerationCreatedEvent]
    ) -> None:
        try:
            collected = text if isinstance(text, str) else "".join([c async for c in text])
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
            return
        # superseded or cancelled while collecting: never speak stale, uninterruptible text
        if self._pending_generation_fut is not fut:
            return
        self.send_command(
            ForcedAgentMessage(type="forced_agent_message", content=collected, uninterruptible=True)
        )
        self._pending_requested = True
        await self._expire(fut, "say()")

    async def _expire(self, fut: asyncio.Future[llm.GenerationCreatedEvent], what: str) -> None:
        """Fails ``fut`` once Ultravox did not start the reply in time."""
        try:
            await asyncio.wait_for(asyncio.shield(fut), REPLY_TIMEOUT)
        except TimeoutError:
            if not fut.done():
                fut.set_exception(
                    llm.RealtimeError(f"{what} timed out waiting for generation_created event.")
                )
        except (asyncio.CancelledError, Exception):
            if not fut.done():
                raise

    def _resolve_pending(self, gen: _ResponseGeneration) -> None:
        fut = self._pending_generation_fut
        if fut is not None and not fut.done() and self._pending_requested:
            fut.set_result(
                llm.GenerationCreatedEvent(
                    message_stream=gen.message_ch,
                    function_stream=gen.function_ch,
                    user_initiated=True,
                )
            )

    def interrupt(self) -> None:
        if self._current_generation and not self._current_generation.done:
            # deferResponse keeps Ultravox from answering the barge-in
            self.send_command(_BARGE_IN)
            self._interrupt_current_generation()

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        logger.warning("truncate is not supported by Ultravox.")

    def commit_audio(self) -> None:
        logger.warning("commit audio is not supported by Ultravox.")

    def clear_audio(self) -> None:
        logger.warning("clear audio is not supported by Ultravox.")

    async def aclose(self) -> None:
        await self._say_tasks.aclose()
        await super().aclose()
        self._supersede_pending("Session closed")
        self._interrupt_current_generation()

    def _start_new_generation(self, *, created_ts: float | None = None) -> _ResponseGeneration:
        if self._current_generation and not self._current_generation.done:
            logger.warning("starting new generation while another is active. Finalizing previous.")
            self._interrupt_current_generation()

        response_id = utils.shortuuid("ultravox-turn-")
        gen = self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            response_id=response_id,
            text_ch=utils.aio.Chan[str](),
            audio_ch=utils.aio.Chan[rtc.AudioFrame](),
            created_timestamp=created_ts or time.time(),
        )
        modalities = asyncio.get_running_loop().create_future()
        modalities.set_result(
            ["audio", "text"] if self._realtime_model.capabilities.audio_output else ["text"]
        )
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
                message_stream=gen.message_ch, function_stream=gen.function_ch, user_initiated=False
            ),
        )
        return gen

    def _active_generation(self) -> _ResponseGeneration:
        gen = self._current_generation
        if gen is None or gen.done:
            gen = self._start_new_generation(created_ts=self._pick_created_timestamp())
        return gen

    def _close_generation(self, *, interrupted: bool) -> None:
        gen = self._current_generation
        if gen is None:
            return
        if not interrupted and gen.done:
            return
        if not interrupted:
            gen.completed_timestamp = time.time()
        gen.text_ch.close()
        gen.audio_ch.close()
        gen.function_ch.close()
        gen.message_ch.close()
        gen.done = True
        if interrupted and gen.output_text:
            self._chat_ctx.add_message(
                role="assistant", content=gen.output_text, id=gen.response_id
            )
        self._emit_generation_metrics(gen, interrupted=interrupted)

    def _interrupt_current_generation(self) -> None:
        self._close_generation(interrupted=True)

    def _emit_generation_metrics(self, gen: _ResponseGeneration, *, interrupted: bool) -> None:
        self._current_generation = None
        # a turn that produced nothing (a tool-only turn) has no metrics
        if gen.first_token_timestamp is None and not gen.output_text:
            return
        completed = gen.completed_timestamp or time.time()
        first_token = gen.first_token_timestamp
        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                timestamp=gen.created_timestamp,
                request_id=gen.response_id,
                ttft=first_token - gen.created_timestamp if first_token else None,
                duration=completed - gen.created_timestamp,
                cancelled=interrupted,
                label=self._realtime_model.label,
                tokens_per_second=0,
                metadata=Metadata(
                    model_name=self._realtime_model.model,
                    model_provider=self._realtime_model.provider,
                ),
            ),
        )

    def _handle_transcript(self, event: Transcript) -> None:
        text = event.get("text")
        final = event["final"]
        if event["role"] == "user":
            item_id = f"msg_user_{event['ordinal']}"
            if final and text and text.strip():
                self._chat_ctx.add_message(role="user", content=text, id=item_id)
            if text:
                self.emit(
                    "input_audio_transcription_completed",
                    llm.InputTranscriptionCompleted(
                        item_id=item_id, transcript=text, is_final=final
                    ),
                )
                if final:
                    self._last_user_final_ts = time.time()
        elif event["role"] == "agent":
            gen = self._active_generation()
            incremental = event.get("delta") or (text if not final else None)
            if incremental:
                if gen.first_token_timestamp is None:
                    gen.first_token_timestamp = time.time()
                    self._resolve_pending(gen)
                gen.text_ch.send_nowait(incremental)
                gen.output_text += incremental
            if final:
                self._close_generation(interrupted=False)

    def _handle_state(self, event: State) -> None:
        if event["state"] == "listening":
            self._interrupt_current_generation()
        elif event["state"] == "thinking":
            # the user finished speaking: TTFT counts from here
            self._active_generation()
        elif event["state"] == "speaking":
            self._resolve_pending(self._active_generation())
            self.emit(
                "input_speech_stopped", InputSpeechStoppedEvent(user_transcription_enabled=False)
            )

    def _handle_tool_invocation(self, event: ClientToolInvocation) -> None:
        if self._current_generation is None:
            self._start_new_generation(created_ts=self._pick_created_timestamp())
        assert self._current_generation is not None
        self._current_generation.function_ch.send_nowait(
            llm.FunctionCall(
                call_id=event["invocationId"],
                name=event["toolName"],
                arguments=json.dumps(event["parameters"]),
            )
        )
        # a tool call ends its turn
        self._interrupt_current_generation()

    def _handle_audio_data(self, audio: bytes) -> None:
        gen = self._current_generation
        if gen is None or gen.done:
            gen = self._start_new_generation()
        if gen.first_token_timestamp is None and any(audio):
            gen.first_token_timestamp = time.time()
        gen.audio_ch.send_nowait(
            rtc.AudioFrame(
                data=audio,
                sample_rate=self._opts.output_sample_rate,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(audio) // (2 * NUM_CHANNELS),
            )
        )
