"""Ultravox real-time model implementation for LiveKit agents.

This module provides a real-time language model using Ultravox's WebSocket API
for streaming speech-to-text, language model, and text-to-speech capabilities.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Union, cast

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, llm, utils
from livekit.agents.llm.tool_context import (
    get_function_info,
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)
from livekit.agents.llm.utils import compute_chat_ctx_diff, function_arguments_to_pydantic_model
from livekit.agents.metrics.base import RealtimeModelMetrics
from livekit.agents.types import NOT_GIVEN, NotGiven, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins.ultravox.log import logger
from livekit.plugins.ultravox.models import (
    AVAILABLE_MODELS,
    AVAILABLE_VOICES,
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    Models,
    Voices,
)

from .events import (
    CallStartedEvent,
    ClientToolInvocationEvent,
    ClientToolResultEvent,
    DebugEvent,
    InputTextMessageEvent,
    PingEvent,
    PlaybackClearBufferEvent,
    PongEvent,
    SetOutputMediumEvent,
    StateEvent,
    TranscriptEvent,
    UltravoxEvent,
    parse_ultravox_event,
    serialize_ultravox_event,
)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
NUM_CHANNELS = 1
ULTRAVOX_BASE_URL = "https://api.ultravox.ai/api"
lk_ultravox_debug = os.getenv("LK_ULTRAVOX_DEBUG", "false").lower() == "true"


def _normalize_model(value: str | None) -> str:
    """Normalize model value with warnings for unknown models."""
    if not value:
        return DEFAULT_MODEL
    if value not in AVAILABLE_MODELS:
        logger.warning(
            "[ultravox] Unrecognized model '%s'; sending as-is (server validates).", value
        )
    return value


def _normalize_voice(value: str | None) -> str:
    """Normalize voice value with warnings for unknown voices."""
    if not value:
        return DEFAULT_VOICE
    if value not in AVAILABLE_VOICES:
        logger.warning(
            "[ultravox] Unrecognized voice '%s'; sending as-is (server validates).", value
        )
    return value


@dataclass
class _UltravoxOptions:
    """Configuration options for Ultravox model."""

    model_id: str
    voice: str
    api_key: str
    base_url: str
    system_prompt: str
    input_sample_rate: int
    output_sample_rate: int
    client_buffer_size_ms: int
    temperature: float | None
    language_hint: str | None
    max_duration: str | None
    time_exceeded_message: str | None
    enable_greeting_prompt: bool
    first_speaker: str | None


@dataclass
class _ResponseGeneration:
    """
    Reference implementation: livekit-plugins/livekit-plugins-google/livekit/plugins/google/beta/realtime/realtime_api.py
    """

    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]

    response_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]

    _created_timestamp: float = field(default_factory=time.time)
    """The timestamp when the generation is created"""
    _first_token_timestamp: float | None = None
    """The timestamp when the first audio token is received"""
    _completed_timestamp: float | None = None
    """The timestamp when the generation is completed"""
    _done: bool = False
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
        model_id: str | Models = DEFAULT_MODEL,
        voice: str | Voices = DEFAULT_VOICE,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        input_sample_rate: int = INPUT_SAMPLE_RATE,
        output_sample_rate: int = OUTPUT_SAMPLE_RATE,
        client_buffer_size_ms: int = 60,
        temperature: float | None = None,
        language_hint: str | None = None,
        max_duration: str | None = None,
        time_exceeded_message: str | None = None,
        enable_greeting_prompt: bool = True,
        first_speaker: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Initialize the Ultravox RealtimeModel.

        Parameters
        ----------
        model_id : str | Models
            The Ultravox model to use.
        voice : str | Voices
            The voice to use for TTS.
        api_key : str, optional
            The Ultravox API key. If None, will try to use environment variables.
        base_url : str, optional
            The base URL for the Ultravox API.
        system_prompt : str
            The system prompt for the model.
        input_sample_rate : int
            Input audio sample rate.
        output_sample_rate : int
            Output audio sample rate.
        client_buffer_size_ms : int
            Size of the client-side audio buffer in milliseconds. For WebSocket integration,
            larger buffer sizes (>200ms) are recommended to prevent audio underflow.
            The implementation will automatically use a minimum of 200ms.
        temperature : float, optional
            Controls response randomness (0.0-1.0). Lower values are more deterministic.
        language_hint : str, optional
            Language hint for better multilingual support (e.g., 'en', 'es', 'fr').
        max_duration : str, optional
            Maximum call duration (e.g., '30m', '1h'). Call ends when exceeded.
        time_exceeded_message : str, optional
            Message to play when max duration is reached.
        enable_greeting_prompt : bool
            Whether to enable greeting prompt if no initial message. Default True.
        first_speaker : str, optional
            Who speaks first ('FIRST_SPEAKER_AGENT' or 'FIRST_SPEAKER_UNSPECIFIED'). If not set, model decides.
        http_session : aiohttp.ClientSession, optional
            HTTP session to use for requests.
        """
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                # Ultravox manages the turn detection internally
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=True,
                audio_output=True,
            )
        )

        ultravox_api_key = api_key or os.environ.get("ULTRAVOX_API_KEY")
        if not ultravox_api_key:
            raise ValueError(
                "Ultravox API key is required. "
                "Provide it via api_key parameter or ULTRAVOX_API_KEY environment variable."
            )

        # Normalize model and voice values
        normalized_model = _normalize_model(model_id)
        normalized_voice = _normalize_voice(voice)

        self._opts = _UltravoxOptions(
            model_id=normalized_model,
            voice=normalized_voice,
            api_key=ultravox_api_key,
            base_url=base_url or ULTRAVOX_BASE_URL,
            system_prompt=system_prompt,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            client_buffer_size_ms=client_buffer_size_ms,
            temperature=temperature,
            language_hint=language_hint,
            max_duration=max_duration,
            time_exceeded_message=time_exceeded_message,
            enable_greeting_prompt=enable_greeting_prompt,
            first_speaker=first_speaker,
        )

        self._http_session = http_session
        self._label = f"ultravox-{normalized_model}"
        self._sessions = weakref.WeakSet[RealtimeSession]()

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session is available."""
        if self._http_session is None:
            self._http_session = utils.http_context.http_session()
        return self._http_session

    def session(self) -> RealtimeSession:
        """Create a new Ultravox real-time session.

        Returns
        -------
        RealtimeSession
            An instance of the Ultravox real-time session.
        """
        sess = RealtimeSession(realtime_model=self)
        self._sessions.add(sess)
        return sess

    def update_options(
        self, *, output_medium: NotGivenOr[Literal["text", "voice"]] = NOT_GIVEN
    ) -> None:
        """Update model options."""

        for sess in self._sessions:
            sess.update_options(output_medium=output_medium)

    async def aclose(self) -> None:
        pass


class RealtimeSession(
    llm.RealtimeSession[Literal["ultravox_server_event_received", "ultravox_client_event_queued"]]
):
    """
    Manages a WebSocket connection and bidirectional communication with Ultravox's Realtime API.
    """

    def __init__(self, realtime_model: RealtimeModel) -> None:
        """Initialize the Ultravox RealtimeSession.

        Parameters
        ----------
        realtime_model : RealtimeModel
            The RealtimeModel instance providing configuration.
        """
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._opts = realtime_model._opts
        self._tools = llm.ToolContext.empty()
        self._msg_ch = utils.aio.Chan[Union[UltravoxEvent, dict[str, Any], bytes]]()
        self._input_resampler: rtc.AudioResampler | None = None

        self._main_atask = asyncio.create_task(self._main_task(), name="UltravoxSession._main_task")

        self._pending_generation_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        self._current_generation: _ResponseGeneration | None = None
        self._chat_ctx = llm.ChatContext.empty()

        # Server-event gating for generate_reply race condition fix
        self._pending_generation_epoch: float | None = None
        # Track last seen ordinals per role to avoid cross-role drops
        self._last_seen_user_ord: int = -1
        self._last_seen_agent_ord: int = -1

        self._bstream = utils.audio.AudioByteStream(
            self._opts.input_sample_rate,
            NUM_CHANNELS,
            samples_per_channel=self._opts.input_sample_rate // 10,
        )

        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._closed = False
        self._closing = False
        self._last_user_final_ts: float | None = None
        # indicates if the underlying session should end
        self._session_should_close = asyncio.Event()

    # Helper function to fix TTFT issue : TTFT was showing -1.0 seconds during function calls
    def _pick_created_timestamp(self) -> float:
        """Pick a creation timestamp anchored to the most recent user-final if fresh.

        Returns the last user-final timestamp if it exists and is recent; otherwise now.
        This avoids tiny TTFT (creation too late) and stale TTFT (creation too early).
        """
        now = time.time()
        if self._last_user_final_ts is not None:
            dt = now - self._last_user_final_ts
            if 0 <= dt <= 10.0:
                return self._last_user_final_ts
        return now

    def _mark_restart_needed(self) -> None:
        if not self._session_should_close.is_set():
            self._session_should_close.set()
            # Close old channel before creating new one
            old_ch = self._msg_ch
            old_ch.close()
            self._msg_ch = utils.aio.Chan[Union[UltravoxEvent, dict[str, Any], bytes]]()

            # Clear pending generation state on restart
            if self._pending_generation_fut and not self._pending_generation_fut.done():
                self._pending_generation_fut.cancel("Session restart")
            self._pending_generation_fut = None
            self._pending_generation_epoch = None

    def update_options(
        self,
        *,
        output_medium: NotGivenOr[Literal["text", "voice"]] = NOT_GIVEN,
        tool_choice: Literal["auto", "required", "none"]
        | llm.ToolChoice
        | NotGiven
        | None = NOT_GIVEN,
    ) -> None:
        """Update session options."""
        if is_given(output_medium):
            self._send_client_event(
                SetOutputMediumEvent(medium=cast(Literal["text", "voice"], output_medium))
            )

        if is_given(tool_choice):
            logger.warning("Ultravox does not support dynamic tool choice updates")

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """Update chat context using Ultravox deferred messages.

        Only sends NEW messages that haven't been sent to Ultravox yet.
        System and developer messages are sent as deferred instructions using the <instruction> pattern.
        User messages are sent as regular text messages.
        Assistant messages are skipped (managed by Ultravox internally).
        Function calls/results are handled via the existing tool mechanism.

        Args:
            chat_ctx: The updated chat context to inject
        """
        # Compute the diff - only process new/changed items
        diff_ops = compute_chat_ctx_diff(self._chat_ctx, chat_ctx)

        #  debug: count of created items
        if lk_ultravox_debug:
            logger.debug(f"[ultravox] update_chat_ctx: to_create={len(diff_ops.to_create)}")

        if not diff_ops.to_create:
            if lk_ultravox_debug:
                logger.debug("[ultravox] No new context items to inject")
            return

        if diff_ops.to_remove:
            logger.warning(
                f"[ultravox] Ignoring {len(diff_ops.to_remove)} message deletions (not supported by Ultravox)"
            )

        logger.debug(f"[ultravox] Processing {len(diff_ops.to_create)} new context items")

        # Process new items only (Ultravox doesn't support deletions)
        for _, msg_id in diff_ops.to_create:
            item = chat_ctx.get_by_id(msg_id)
            if not item:
                continue

            if item.type == "message" and item.role in ("system", "developer"):
                if item.text_content:
                    logger.debug(
                        f"[ultravox] Injecting {item.role} instruction: {item.text_content[:100]}..."
                    )
                    self._send_client_event(
                        InputTextMessageEvent(
                            text=f"<instruction>{item.text_content}</instruction>",
                            defer_response=True,
                        )
                    )

            elif item.type == "message" and item.role == "user":
                # Inject user message as context; do not trigger an immediate response
                if item.text_content:
                    logger.debug(f"[ultravox] Injecting user message: {item.text_content[:100]}...")
                    self._send_client_event(
                        InputTextMessageEvent(text=item.text_content, defer_response=True)
                    )
            elif item.type == "function_call_output":
                # Bridge tool result back to Ultravox using the original invocationId
                if lk_ultravox_debug:
                    logger.debug(
                        f"[ultravox] bridging tool result: invocationId={item.call_id} "
                        f"is_error={getattr(item, 'is_error', False)} "
                        f"result_len={len(str(getattr(item, 'output', '') or ''))}"
                    )

                tool_result = ClientToolResultEvent(
                    invocationId=item.call_id,
                    agent_reaction="speaks",
                )

                if getattr(item, "is_error", False):
                    tool_result.error_type = "implementation-error"
                    tool_result.error_message = getattr(item, "error_message", None) or str(
                        getattr(item, "output", "")
                    )
                else:
                    tool_result.result = str(getattr(item, "output", ""))

                self._send_client_event(tool_result)

                #  debug: tool result bridged
                if lk_ultravox_debug:
                    logger.debug(f"[ultravox] tool_result_bridged: id={item.call_id}")

        # Update local chat context
        self._chat_ctx = chat_ctx.copy()

    async def update_tools(self, tools: list[llm.FunctionTool | llm.RawFunctionTool]) -> None:
        """Update the available tools."""
        # Get current and new tool names for comparison
        current_tool_names = set(self._tools.function_tools.keys())

        # Always update the tools
        self._tools.update_tools(tools)
        new_tool_names = set(self._tools.function_tools.keys())

        # Restart session only if tool set actually changed
        if current_tool_names != new_tool_names:
            self._mark_restart_needed()

    async def update_instructions(self, instructions: str | NotGiven = NOT_GIVEN) -> None:
        """Update the system instructions."""
        # This means we need to restart the whole conversation
        if is_given(instructions) and self._opts.system_prompt != instructions:
            self._opts.system_prompt = instructions
            self._mark_restart_needed()

    @property
    def chat_ctx(self) -> llm.ChatContext:
        """Get the current chat context."""
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        """Get the current tool context."""
        return self._tools

    @utils.log_exceptions(logger=logger)
    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Push audio frames to the session for transcription by Ultravox."""
        if self._closed:
            return

        for resampled_frame in self._resample_audio(frame):
            for audio_frame in self._bstream.write(resampled_frame.data.tobytes()):
                # Send audio data through message channel for proper ordering
                self._send_audio_bytes(audio_frame.data.tobytes())

    def push_video(self, frame: rtc.VideoFrame) -> None:
        """Push video frames (not supported by Ultravox)."""
        pass

    def _send_client_event(self, event: UltravoxEvent | dict[str, Any]) -> None:
        """Send an event to the Ultravox WebSocket."""
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    def _send_audio_bytes(self, audio_data: bytes) -> None:
        """Send audio bytes to the Ultravox WebSocket via message channel."""
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(audio_data)

    @utils.log_exceptions(logger=logger)
    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """Generate a reply from the LLM based on the instructions."""
        # Cancel prior pending generation if exists
        if self._pending_generation_fut and not self._pending_generation_fut.done():
            logger.warning(
                "generate_reply called while another generation is pending, cancelling previous."
            )
            self._pending_generation_fut.cancel("Superseded by new generate_reply call")

        # Record epoch for server-event gating
        self._pending_generation_epoch = time.perf_counter()

        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._pending_generation_fut = fut

        if is_given(instructions):
            self._send_client_event(InputTextMessageEvent(text=instructions, defer_response=False))

        def _on_timeout() -> None:
            if not fut.done():
                fut.set_exception(
                    llm.RealtimeError(
                        "generate_reply timed out waiting for generation_created event."
                    )
                )
                if self._pending_generation_fut is fut:
                    self._pending_generation_fut = None
                    self._pending_generation_epoch = None

        timeout_handle = asyncio.get_event_loop().call_later(5.0, _on_timeout)
        fut.add_done_callback(lambda _: timeout_handle.cancel())

        return fut

    def interrupt(self) -> None:
        """Interrupt the current generation."""
        logger.debug("Interrupted current generation")

        # Only send barge-in if there's an active generation to interrupt
        if self._current_generation and not self._current_generation._done:
            # Send programmatic interruption to server via text barge-in

            # Use text barge-in with immediate urgency to interrupt
            # deferResponse=true prevents Ultravox from generating a response
            interrupt_msg = {
                "type": "user_text_message",
                "text": "(barge-in)",
                "urgency": "immediate",
                "deferResponse": True,
            }
            self._send_client_event(interrupt_msg)  # Direct call, already non-blocking

            # Finalize the active generation
            self._mark_current_generation_done()

    def truncate(
        self,
        *,
        message_id: str,
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Ultravox has no server-side truncate; we simply ignore the request."""
        pass

    async def aclose(self) -> None:
        """Close the session and clean up resources."""
        if self._closed:
            return

        self._closed = True
        self._msg_ch.close()
        self._session_should_close.set()

        await utils.aio.cancel_and_wait(self._main_atask)

        if self._pending_generation_fut and not self._pending_generation_fut.done():
            self._pending_generation_fut.cancel("Session closed")

        if self._current_generation:
            self._mark_current_generation_done()

        self._closed = True

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        """Main task with restart loop for managing WebSocket sessions."""
        while not self._msg_ch.closed:
            # Clear restart signal for new session
            self._session_should_close.clear()
            # Reset ordinal tracking on reconnect to avoid stale event issues
            self._last_seen_user_ord = -1
            self._last_seen_agent_ord = -1

            try:
                # Create new Ultravox session
                headers = {
                    "User-Agent": "LiveKit Agents",
                    "X-API-Key": self._realtime_model._opts.api_key,
                    "Content-Type": "application/json",
                }

                # Build query parameters
                query_params = {}
                if not self._realtime_model._opts.enable_greeting_prompt:
                    query_params["enableGreetingPrompt"] = "false"

                # Construct URL with query parameters
                create_call_url = f"{self._realtime_model._opts.base_url.rstrip('/')}/calls"
                if query_params:
                    query_string = "&".join(f"{k}={v}" for k, v in query_params.items())
                    create_call_url += f"?{query_string}"

                # Build payload with core parameters
                recommended_buffer_size = max(self._realtime_model._opts.client_buffer_size_ms, 200)

                payload: dict[str, Any] = {
                    "systemPrompt": self._realtime_model._opts.system_prompt,
                    "model": self._realtime_model._opts.model_id,
                    "voice": self._realtime_model._opts.voice,
                    "medium": {
                        "serverWebSocket": {
                            "inputSampleRate": self._realtime_model._opts.input_sample_rate,
                            "outputSampleRate": self._realtime_model._opts.output_sample_rate,
                            "clientBufferSizeMs": recommended_buffer_size,
                        }
                    },
                    "selectedTools": self._prepare_tools(list(self._tools.function_tools.values())),
                }

                # Add optional parameters only if specified
                if self._realtime_model._opts.temperature is not None:
                    payload["temperature"] = self._realtime_model._opts.temperature
                if self._realtime_model._opts.language_hint is not None:
                    payload["languageHint"] = self._realtime_model._opts.language_hint
                if self._realtime_model._opts.max_duration is not None:
                    payload["maxDuration"] = self._realtime_model._opts.max_duration
                if self._realtime_model._opts.time_exceeded_message is not None:
                    payload["timeExceededMessage"] = (
                        self._realtime_model._opts.time_exceeded_message
                    )
                if self._realtime_model._opts.first_speaker is not None:
                    payload["firstSpeaker"] = self._realtime_model._opts.first_speaker

                # Create call and connect to WebSocket
                http_session = self._realtime_model._ensure_http_session()
                async with http_session.post(
                    create_call_url, json=payload, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    response_json = await resp.json()
                    join_url = response_json.get("joinUrl")
                    if not join_url:
                        raise APIConnectionError("Ultravox call created, but no joinUrl received.")

                ws_conn = await http_session.ws_connect(join_url)
                self._closing = False

                # Create tasks for send/recv and restart monitoring
                send_task = asyncio.create_task(self._send_task(ws_conn), name="_send_task")
                recv_task = asyncio.create_task(self._recv_task(ws_conn), name="_recv_task")
                restart_wait_task = asyncio.create_task(
                    self._session_should_close.wait(), name="_restart_wait"
                )

                try:
                    # Wait for any task to complete
                    done, _ = await asyncio.wait(
                        [send_task, recv_task, restart_wait_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task != restart_wait_task:
                            # propagate exception if any
                            task.result()
                finally:
                    # Close current WebSocket
                    await ws_conn.close()
                    await utils.aio.cancel_and_wait(send_task, recv_task, restart_wait_task)

                # If restart triggered, loop continues
                # If msg_ch closed, exit loop
                if restart_wait_task not in done and self._msg_ch.closed:
                    break

            except Exception as e:
                logger.error(f"Ultravox WebSocket error: {e}", exc_info=True)

                # Determine if error is recoverable based on type
                is_recoverable = False
                if isinstance(e, (aiohttp.ClientConnectionError, asyncio.TimeoutError)):
                    is_recoverable = True

                # Convert to appropriate API error type
                if isinstance(e, (APIConnectionError, APIError)):
                    error = e
                elif isinstance(e, aiohttp.ClientResponseError):
                    error = APIError(f"HTTP {e.status}: {e.message}", retryable=is_recoverable)
                else:
                    error = APIConnectionError(f"Connection failed: {str(e)}")

                self.emit(
                    "error",
                    llm.RealtimeModelError(
                        timestamp=time.time(),
                        label=self._realtime_model._label,
                        error=error,
                        recoverable=is_recoverable,
                    ),
                )

                # Break loop on non-recoverable errors or if channel is closed
                if not is_recoverable or self._msg_ch.closed:
                    break

                # Wait before retrying on recoverable errors
                await asyncio.sleep(1.0)

    @utils.log_exceptions(logger=logger)
    async def _send_task(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        """Task for sending messages to Ultravox WebSocket."""
        async for msg in self._msg_ch:
            # Check if restart is needed
            if self._session_should_close.is_set():
                break

            try:
                if isinstance(msg, bytes):
                    # Handle binary audio data
                    self.emit(
                        "ultravox_client_event_queued", {"type": "audio_bytes", "len": len(msg)}
                    )
                    await ws_conn.send_bytes(msg)
                    # You will want to comment these logs when in debugging mode as they are noisy
                    # if lk_ultravox_debug:
                    #     logger.info(f">>> [audio bytes: {len(msg)} bytes]")
                elif isinstance(msg, dict):
                    msg_dict = msg
                    self.emit("ultravox_client_event_queued", msg_dict)
                    await ws_conn.send_str(json.dumps(msg_dict))
                    if lk_ultravox_debug:
                        logger.debug(f">>> {msg_dict}")
                else:
                    msg_dict = serialize_ultravox_event(msg)
                    self.emit("ultravox_client_event_queued", msg_dict)
                    await ws_conn.send_str(json.dumps(msg_dict))
                    if lk_ultravox_debug:
                        logger.debug(f">>> {msg_dict}")
            except Exception as e:
                logger.error(f"Error sending message: {e}", exc_info=True)
                break

        self._closing = True

    @utils.log_exceptions(logger=logger)
    async def _recv_task(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        """Task for receiving messages from Ultravox WebSocket."""
        while True:
            # Check if restart is needed
            if self._session_should_close.is_set():
                break

            msg = await ws_conn.receive()
            # Generation will be started when we receive state change to "speaking" or first transcript

            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    self.emit("ultravox_server_event_received", data)
                    if lk_ultravox_debug:
                        logger.debug(f"<<< {data}")
                    event = parse_ultravox_event(data)
                    self._handle_ultravox_event(event)

                except Exception as e:
                    logger.error(f"Error handling message: {e}", exc_info=True)

            elif msg.type == aiohttp.WSMsgType.BINARY:
                self._handle_audio_data(msg.data)
            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                # If we're already closing due to send loop shutdown, just return
                if self._closing:
                    return
                # Unexpected close
                raise APIConnectionError(message="Ultravox S2S connection closed unexpectedly")
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"Ultravox WebSocket error: {ws_conn.exception()}")
                break

    @utils.log_exceptions(logger=logger)
    def _start_new_generation(self, *, created_ts: float | None = None) -> None:
        """Start a new response generation."""
        if self._current_generation and not self._current_generation._done:
            logger.warning("starting new generation while another is active. Finalizing previous.")
            self._mark_current_generation_done()

        response_id = utils.shortuuid("ultravox-turn-")
        self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            response_id=response_id,
            text_ch=utils.aio.Chan[str](),
            audio_ch=utils.aio.Chan[rtc.AudioFrame](),
            _created_timestamp=created_ts or time.time(),
        )
        self._current_generation.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=response_id,
                text_stream=self._current_generation.text_ch,
                audio_stream=self._current_generation.audio_ch,
            )
        )
        generation_ev = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
        )
        self.emit("generation_created", generation_ev)

        if lk_ultravox_debug:
            logger.debug(f"[ultravox] start_generation id={response_id}")

    @utils.log_exceptions(logger=logger)
    def _mark_current_generation_done(self) -> None:
        if not self._current_generation:
            return

        gen = self._current_generation
        if not gen.text_ch.closed:
            gen.text_ch.close()
        if not gen.audio_ch.closed:
            gen.audio_ch.close()

        gen.function_ch.close()
        gen.message_ch.close()
        gen._done = True

        # Append assistant message to local chat context
        if gen.output_text:
            self._chat_ctx.add_message(
                role="assistant",
                content=gen.output_text,
                id=gen.response_id,
            )

        # Emit metrics for interrupted/completed generation
        self._emit_generation_metrics(interrupted=True)

    def _handle_ultravox_event(self, event: UltravoxEvent) -> None:
        """Handle incoming Ultravox events and map them to LiveKit events."""
        if isinstance(event, TranscriptEvent):
            self._handle_transcript_event(event)
        elif isinstance(event, StateEvent):
            self._handle_state_event(event)
        elif isinstance(event, ClientToolInvocationEvent):
            self._handle_tool_invocation_event(event)
        elif isinstance(event, PongEvent):
            self._handle_pong_event(event)
        elif isinstance(event, PlaybackClearBufferEvent):
            self._handle_playback_clear_buffer_event(event)
        elif isinstance(event, CallStartedEvent):
            pass
        elif isinstance(event, DebugEvent):
            self._handle_debug_event(event)
        else:
            logger.warning(f"Unhandled Ultravox event: {event}")

    @utils.log_exceptions(logger=logger)
    def _handle_transcript_event(self, event: TranscriptEvent) -> None:
        """Handle transcript events from Ultravox."""
        if lk_ultravox_debug:
            kind = "delta" if event.delta else ("text" if event.text else "empty")
            logger.debug(
                f"[ultravox] transcript role={event.role} medium={event.medium} ord={event.ordinal} final={event.final} kind={kind} text_len={len(event.text or '')} delta_len={len(event.delta or '')}"
            )

        # Gate by ordinal to avoid late/stale events (scoped per role to avoid dropping cross-role finals)
        if event.role == "user":
            if event.ordinal < self._last_seen_user_ord:
                if lk_ultravox_debug:
                    logger.debug(
                        f"[ultravox] skip_stale_user ord={event.ordinal} < last_user={self._last_seen_user_ord}"
                    )
                return
            self._last_seen_user_ord = event.ordinal
        elif event.role == "agent":
            if event.ordinal < self._last_seen_agent_ord:
                if lk_ultravox_debug:
                    logger.debug(
                        f"[ultravox] skip_stale_agent ord={event.ordinal} < last_agent={self._last_seen_agent_ord}"
                    )
                return
            self._last_seen_agent_ord = event.ordinal

        if event.role == "user":
            # Keep local chat history in sync (append-only) only if transcript is non-empty
            if event.final and (event.text and event.text.strip()):
                self._chat_ctx.add_message(
                    role="user",
                    content=event.text,
                    id=f"msg_{event.ordinal}",
                )
            # Emit final transcription only if non-empty to avoid blank user messages
            if event.final and (event.text and event.text.strip()):
                self._last_user_final_ts = time.time()
                if lk_ultravox_debug:
                    logger.debug(
                        f"[ultravox] user_final ord={event.ordinal} text_len={len(event.text or '')}"
                    )
                self.emit(
                    "input_audio_transcription_completed",
                    llm.InputTranscriptionCompleted(
                        item_id=f"msg_{event.ordinal}",
                        transcript=event.text,
                        is_final=True,
                    ),
                )
        elif event.role == "agent":
            # ! The transcript for the first message isn't always complete for some reason
            if self._current_generation is None:
                # Start a generation anchored to recent user-final or now
                if lk_ultravox_debug:
                    logger.debug("[ultravox] starting generation on first agent transcript")
                self._start_new_generation(created_ts=self._pick_created_timestamp())

            msg_gen = self._current_generation
            assert msg_gen is not None

            # Handle incremental transcript updates (delta or non-final text)
            incremental_text = event.delta or (event.text if not event.final else None)
            if incremental_text:
                # Set first token timestamp on first text delta (TTFT measurement)
                if msg_gen._first_token_timestamp is None:
                    msg_gen._first_token_timestamp = time.time()

                    # Resolve pending generation on first agent TranscriptEvent as backup
                    if (
                        self._pending_generation_fut
                        and not self._pending_generation_fut.done()
                        and self._pending_generation_epoch is not None
                        and time.perf_counter() > self._pending_generation_epoch
                    ):
                        logger.info(
                            "Resolving pending generate_reply via first agent TranscriptEvent"
                        )
                        generation_created = llm.GenerationCreatedEvent(
                            message_stream=msg_gen.message_ch,
                            function_stream=msg_gen.function_ch,
                            user_initiated=True,
                        )
                        self._pending_generation_fut.set_result(generation_created)
                        self._pending_generation_fut = None
                        self._pending_generation_epoch = None

                msg_gen.text_ch.send_nowait(incremental_text)
                msg_gen.output_text += incremental_text
                if lk_ultravox_debug:
                    preview = incremental_text[:40].replace("\n", " ") if incremental_text else ""
                    logger.debug(
                        f"[ultravox] agent_text ord={event.ordinal} len={len(incremental_text)} preview={preview!r}"
                    )

            # Handle final transcript
            if event.final:
                msg_gen.text_ch.close()
                msg_gen.audio_ch.close()
                self._handle_response_done()

    @utils.log_exceptions(logger=logger)
    def _handle_response_done(self) -> None:
        if self._current_generation is None:
            return

        self._current_generation._completed_timestamp = time.time()

        if not self._current_generation.text_ch.closed:
            self._current_generation.text_ch.close()
        if not self._current_generation.audio_ch.closed:
            self._current_generation.audio_ch.close()

        self._current_generation.function_ch.close()
        self._current_generation.message_ch.close()

        # Emit metrics for completed generation
        self._emit_generation_metrics(interrupted=False)

    @utils.log_exceptions(logger=logger)
    def _emit_generation_metrics(self, interrupted: bool = False) -> None:
        """Emit RealtimeModelMetrics for the current generation."""
        if self._current_generation is None:
            return

        gen = self._current_generation
        # Skip metrics if no output tokens/text were produced (e.g., tool-only placeholder turns)
        if gen._first_token_timestamp is None and not gen.output_text:
            self._current_generation = None
            return
        current_time = time.time()
        completed_timestamp = gen._completed_timestamp or current_time
        created_timestamp = gen._created_timestamp
        first_token_timestamp = gen._first_token_timestamp

        # Calculate timing metrics
        # TTFT should be from when user stops speaking (generation created) to first response token
        ttft = first_token_timestamp - created_timestamp if first_token_timestamp else -1
        duration = completed_timestamp - created_timestamp

        metrics = RealtimeModelMetrics(
            timestamp=created_timestamp,
            request_id=gen.response_id,
            ttft=ttft,
            duration=duration,
            cancelled=interrupted,
            label=self._realtime_model._label,
            input_tokens=0,  # Ultravox doesn't provide token counts
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
        )

        self.emit("metrics_collected", metrics)

        # Clear the current generation after emitting metrics
        self._current_generation = None

    def _handle_state_event(self, event: StateEvent) -> None:
        """Handle state events from Ultravox."""
        if lk_ultravox_debug:
            logger.debug(f"Ultravox state: {event.state}")
        if event.state == "listening":
            if lk_ultravox_debug:
                logger.debug("[ultravox] state=listening")
            # Do not emit input_speech_started here; rely on actual user transcripts to infer turns
        elif event.state == "thinking":
            # Start generation when Ultravox begins processing (user finished speaking)
            # This is the proper TTFT start time: when user stops speaking and agent starts processing
            if not self._current_generation or self._current_generation._done:
                logger.debug(
                    "Starting new generation (Ultravox thinking state - user finished speaking)"
                )
                # Anchor creation to recent user-final or now
                self._start_new_generation(created_ts=self._pick_created_timestamp())
        elif event.state == "speaking":
            # Ensure a generation exists so early audio frames are captured
            if not self._current_generation or self._current_generation._done:
                # Ensure a generation exists; anchor creation to recent user-final or now
                self._start_new_generation(created_ts=self._pick_created_timestamp())

            assert self._current_generation is not None
            # Resolve pending generation with server confirmation via "speaking" event
            if (
                self._pending_generation_fut
                and not self._pending_generation_fut.done()
                and self._pending_generation_epoch is not None
                and time.perf_counter() > self._pending_generation_epoch
            ):
                logger.debug("Resolving pending generate_reply via speaking state event")
                generation_created = llm.GenerationCreatedEvent(
                    message_stream=self._current_generation.message_ch,
                    function_stream=self._current_generation.function_ch,
                    user_initiated=True,
                )
                self._pending_generation_fut.set_result(generation_created)
                self._pending_generation_fut = None
                self._pending_generation_epoch = None

            # Do not emit input_speech_stopped here; rely on user final transcript events.
            if lk_ultravox_debug:
                logger.debug("[ultravox] state=speaking (no input_speech_stopped)")

    def _handle_tool_invocation_event(self, event: ClientToolInvocationEvent) -> None:
        """Handle tool invocation events from Ultravox."""
        if lk_ultravox_debug:
            logger.debug(
                f"[ultravox] tool_invocation received: tool={event.tool_name} "
                f"invocationId={event.invocation_id} params_keys={list(event.parameters.keys())}"
            )

        # Emit FunctionCall to maintain framework compatibility
        function_call = llm.FunctionCall(
            call_id=event.invocation_id,
            name=event.tool_name,
            arguments=json.dumps(event.parameters),
        )

        if self._current_generation is None:
            # Tool invocations do not represent model output yet; anchor to recent user-final or now
            self._start_new_generation(created_ts=self._pick_created_timestamp())

        assert self._current_generation is not None
        self._current_generation.function_ch.send_nowait(function_call)

        if lk_ultravox_debug:
            logger.debug(f"[ultravox] emitted FunctionCall id={event.invocation_id}")

        if lk_ultravox_debug and self._current_generation is not None:
            gen_id = self._current_generation.response_id
            logger.debug(
                f"[ultravox] tool_invocation trace: id={event.invocation_id} gen_id={gen_id}"
            )

        # Always close tool turn immediately upon invocation
        if lk_ultravox_debug:
            logger.debug(
                f"[ultravox] close_on_invocation: closing generation for tool id={event.invocation_id}"
            )
        self._mark_current_generation_done()

    def _handle_pong_event(self, event: PongEvent) -> None:
        """Handle pong events from Ultravox."""
        current_time = time.perf_counter()
        latency = current_time - event.timestamp
        self._send_client_event(
            PingEvent(
                timestamp=current_time,
            )
        )
        if lk_ultravox_debug:
            logger.debug(f"Ultravox latency: {latency:.3f}s")

    def _handle_playback_clear_buffer_event(self, event: PlaybackClearBufferEvent) -> None:
        """Handle playback clear buffer events from Ultravox.

        This event is WebSocket-specific and indicates that the client should
        clear any buffered audio output to prevent audio lag or overlapping.
        """
        logger.debug("[ultravox] Received PlaybackClearBuffer - clearing audio buffer")

        # Best-effort: clear local input buffer (output playout buffer is managed
        # downstream in the agent audio pipeline).
        self.clear_audio()

    def _handle_debug_event(self, event: DebugEvent) -> None:
        """Handle debug events from Ultravox."""
        if lk_ultravox_debug:
            logger.debug(f"[ultravox] Debug: {event.message}")

    @utils.log_exceptions(logger=logger)
    def _handle_audio_data(self, audio_data: bytes) -> None:
        """Handle binary audio data from Ultravox."""
        try:
            # Check if we have a current generation before processing audio
            if not self._current_generation or self._current_generation._done:
                if lk_ultravox_debug:
                    logger.debug(
                        "[ultravox] Received audio data but no current generation, ignoring"
                    )
                return

            # Set first token timestamp when we receive first audio from Ultravox (TTFT measurement)
            # Only set if this is actually the first audio token (non-zero data)
            if (
                self._current_generation._first_token_timestamp is None
                and len(audio_data) > 0
                and any(audio_data)
            ):  # Check for non-zero audio data
                self._current_generation._first_token_timestamp = time.time()

            frame = rtc.AudioFrame(
                data=audio_data,
                sample_rate=self._opts.output_sample_rate,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(audio_data) // (2 * NUM_CHANNELS),
            )
            self._current_generation.audio_ch.send_nowait(frame)
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")

    def commit_audio(self) -> None:
        """Commit the current audio input segment."""
        pass

    def clear_audio(self) -> None:
        """Best-effort local input buffer clear.

        Ultravox does not provide a client-side event to clear server-held input buffers.
        This clears any locally buffered input audio that hasn't been framed/sent.
        Output audio buffering is cleared via server 'playback_clear_buffer' events.
        """
        # Drop any locally buffered input bytes
        self._bstream._buf.clear()

        # Drop any pending samples queued in the resampler
        if self._input_resampler is not None:
            self._input_resampler = None

        logger.warning(
            "[ultravox] clear_audio: best-effort local input buffer clear; "
            "server-side input clearing is not supported. Output buffering is "
            "cleared via PlaybackClearBuffer."
        )

    @utils.log_exceptions(logger=logger)
    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        """Resample audio frame to the required sample rate."""
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                # Input audio changed to a different sample rate
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != self._realtime_model._opts.input_sample_rate
            or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=self._realtime_model._opts.input_sample_rate,
                num_channels=NUM_CHANNELS,
            )

        if self._input_resampler:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    async def send_tool_result(self, call_id: str, result: str) -> None:
        """Send tool execution result back to Ultravox."""
        if lk_ultravox_debug:
            preview = (
                (result[:200] + "...") if isinstance(result, str) and len(result) > 200 else result
            )
            logger.debug(f"[ultravox] send_tool_result: call_id={call_id} preview={preview!r}")

        event = ClientToolResultEvent(
            invocationId=call_id,
            result=result,
            agent_reaction="speaks",
        )
        self._send_client_event(event)

    def _prepare_tools(
        self, tools: list[llm.FunctionTool | llm.RawFunctionTool]
    ) -> list[dict[str, Any]]:
        """Prepare tools for sending to Ultravox. https://docs.ultravox.ai/essentials/tools#creating-your-first-custom-tool"""

        results: list[dict[str, Any]] = []
        for tool in tools:
            if is_raw_function_tool(tool):
                raw_fnc_info = get_raw_function_info(tool)
                name = raw_fnc_info.name
                description = raw_fnc_info.raw_schema.get("description", None)
                parameters = raw_fnc_info.raw_schema.get("parameters", {})
            elif is_function_tool(tool):
                fnc_info = get_function_info(tool)
                model = function_arguments_to_pydantic_model(tool)
                name = fnc_info.name
                description = fnc_info.description
                parameters = model.model_json_schema()

            def _extract_type(prop: dict[str, Any]) -> str:
                """Best-effort guess of a parameter's primitive type."""
                if "type" in prop:
                    assert isinstance(prop["type"], str)
                    return prop["type"]
                if "enum" in prop:
                    return "string"
                if "items" in prop:
                    return "array"
                for key in ("anyOf", "oneOf"):
                    if key in prop:
                        for variant in prop[key]:
                            if isinstance(variant, dict) and "type" in variant:
                                assert isinstance(variant["type"], str)
                                return variant["type"]
                # Fallback to string
                return "string"

            results.append(
                {
                    "temporaryTool": {
                        "modelToolName": name,
                        "description": description,
                        "dynamicParameters": [
                            {
                                "name": pn,
                                "location": "PARAMETER_LOCATION_BODY",
                                "schema": (
                                    p
                                    if "type" in p
                                    else {  # fallback minimal schema for enum/anyOf etc.
                                        "type": _extract_type(p),
                                        **(
                                            {"description": p.get("description")}
                                            if p.get("description")
                                            else {}
                                        ),
                                    }
                                ),
                                "required": pn in parameters.get("required", []),
                            }
                            for pn, p in parameters["properties"].items()
                        ],
                        "client": {},
                    }
                }
            )
        return results
