"""Ultravox real-time model implementation for LiveKit agents.

This module provides a real-time language model using Ultravox's WebSocket API
for streaming speech-to-text, language model, and text-to-speech capabilities.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Union

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, llm, utils
from livekit.agents.metrics.base import RealtimeModelMetrics
from livekit.agents.types import NOT_GIVEN, NotGiven, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins.ultravox.log import logger
from livekit.plugins.ultravox.models import Models, Voices

from .events import (
    ClientToolInvocationEvent,
    ClientToolResultEvent,
    InputTextMessageEvent,
    PingEvent,
    PlaybackClearBufferEvent,
    PongEvent,
    # SetOutputMediumEvent,
    StateEvent,
    TranscriptEvent,
    UltravoxEventType,
    parse_ultravox_event,
    serialize_ultravox_event,
)

SAMPLE_RATE = 16000  # Ultravox input sample rate
OUTPUT_SAMPLE_RATE = 24000  # Ultravox output sample rate
NUM_CHANNELS = 1
ULTRAVOX_BASE_URL = "https://api.ultravox.ai/api"
lk_ultravox_debug = True


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


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]

    response_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    input_transcription: str = ""

    _created_timestamp: float = field(default_factory=time.time)
    """The timestamp when the generation is created"""
    _first_token_timestamp: float | None = None
    """The timestamp when the first audio token is received"""
    _completed_timestamp: float | None = None
    """The timestamp when the generation is completed"""
    _done: bool = False
    """Whether the generation is done (set when the turn is complete)"""


class RealtimeModel(llm.RealtimeModel):
    """Real-time language model using Ultravox.

    Connects to Ultravox's WebSocket API for streaming STT, LLM, and TTS.
    """

    def __init__(
        self,
        *,
        model_id: str | Models = "fixie-ai/ultravox",
        voice: str | Voices = "Mark",
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        input_sample_rate: int = SAMPLE_RATE,
        output_sample_rate: int = OUTPUT_SAMPLE_RATE,
        client_buffer_size_ms: int = 60,
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
            Size of the client-side audio buffer in milliseconds.
        http_session : aiohttp.ClientSession, optional
            HTTP session to use for requests.
        """
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
            )
        )

        ultravox_api_key = api_key or os.environ.get("ULTRAVOX_API_KEY")
        if not ultravox_api_key:
            raise ValueError(
                "Ultravox API key is required. "
                "Provide it via api_key parameter or ULTRAVOX_API_KEY environment variable."
            )

        self._opts = _UltravoxOptions(
            model_id=model_id,
            voice=voice,
            api_key=ultravox_api_key,
            base_url=base_url or ULTRAVOX_BASE_URL,
            system_prompt=system_prompt,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            client_buffer_size_ms=client_buffer_size_ms,
        )

        self._http_session = http_session
        self._label = f"ultravox-{model_id}"
        self._sessions = weakref.WeakSet[RealtimeSession]()

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session is available."""
        if self._http_session is None:
            self._http_session = utils.http_context.http_session()
        return self._http_session

    # def update_options(
    #     self,
    #     *,
    #     voice: NotGivenOr[str] = NOT_GIVEN,
    #     model_id: NotGivenOr[str] = NOT_GIVEN,
    #     system_prompt: NotGivenOr[str] = NOT_GIVEN,
    # ) -> None:
    #     """Update model options."""
    #     if is_given(voice):
    #         self._opts.voice = voice
    #     if is_given(model_id):
    #         self._opts.model_id = model_id
    #     if is_given(system_prompt):
    #         self._opts.system_prompt = system_prompt

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
        self._tools = llm.ToolContext.empty()
        self._msg_ch = utils.aio.Chan[Union[UltravoxEventType, dict[str, Any]]]()
        self._input_resampler: rtc.AudioResampler | None = None

        self._main_atask = asyncio.create_task(self._main_task(), name="UltravoxSession._main_task")

        self._response_created_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}
        self._pending_generation_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        self._current_generation: _ResponseGeneration | None = None
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()

        self._update_chat_ctx_lock = asyncio.Lock()
        self._update_fnc_ctx_lock = asyncio.Lock()
        self._current_message_id: str | None = None

        # Audio buffer for streaming
        self._bstream = utils.audio.AudioByteStream(
            self._realtime_model._opts.input_sample_rate,
            NUM_CHANNELS,
            samples_per_channel=self._realtime_model._opts.input_sample_rate // 10,
        )
        self._pushed_duration_s = 0.0

        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._closed = False
        self._closing = False

        self._pending_tool_calls: dict[str, llm.FunctionCall] = {}

    def send_event(self, event: UltravoxEventType | dict[str, Any]) -> None:
        """Send an event to the Ultravox WebSocket."""
        try:
            self._msg_ch.send_nowait(event)
        except utils.aio.ChanClosed:
            logger.warning("Message channel closed, cannot send event")

    def update_options(
        self,
        tool_choice: Literal["auto", "required", "none"]
        | llm.ToolChoice
        | NotGiven
        | None = NOT_GIVEN,
    ) -> None:
        """Update session options."""
        # Ultravox doesn't support dynamic tool choice updates
        pass

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        """Main task for managing WebSocket connection and message handling."""
        headers = {
            "User-Agent": "LiveKit Agents",
            "X-API-Key": self._realtime_model._opts.api_key,
            "Content-Type": "application/json",
        }

        create_call_url = f"{self._realtime_model._opts.base_url.rstrip('/')}/calls"
        payload = {
            "systemPrompt": self._realtime_model._opts.system_prompt,
            "model": self._realtime_model._opts.model_id,
            "voice": self._realtime_model._opts.voice,
            "medium": {
                "serverWebSocket": {
                    "inputSampleRate": self._realtime_model._opts.input_sample_rate,
                    "outputSampleRate": self._realtime_model._opts.output_sample_rate,
                    "clientBufferSizeMs": self._realtime_model._opts.client_buffer_size_ms,
                }
            },
            # TODO: add tools
        }

        try:
            http_session = self._realtime_model._ensure_http_session()
            async with http_session.post(create_call_url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                response_json = await resp.json()
                join_url = response_json.get("joinUrl")
                if not join_url:
                    raise APIConnectionError("Ultravox call created, but no joinUrl received.")

            ws_conn = await http_session.ws_connect(join_url)
            self._ws = ws_conn

            tasks = [
                asyncio.create_task(self._recv_task(), name="_recv_task"),
                asyncio.create_task(self._send_task(), name="_send_task"),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.cancel_and_wait(*tasks)
                await self._ws.close()

        except Exception as e:
            logger.error(f"Ultravox WebSocket error: {e}", exc_info=True)
            self.emit(
                "error",
                llm.RealtimeModelError(
                    timestamp=time.time(),
                    label=self._realtime_model._label,
                    error=e
                    if isinstance(e, (APIConnectionError, APIError))
                    else APIConnectionError(str(e)),
                    recoverable=False,
                ),
            )
        finally:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None
        self._closed = True

    @utils.log_exceptions(logger=logger)
    async def _send_task(self) -> None:
        """Task for sending messages to Ultravox WebSocket."""

        # system message should be ignored
        should_ignore_first_message = True
        async for msg in self._msg_ch:
            if should_ignore_first_message:
                should_ignore_first_message = False
                continue

            try:
                if isinstance(msg, dict):
                    msg_dict = msg
                else:
                    msg_dict = serialize_ultravox_event(msg)

                self.emit("ultravox_client_event_queued", msg_dict)
                await self._ws.send_str(json.dumps(msg_dict))
                if lk_ultravox_debug:
                    logger.info(f">>> {msg_dict}")
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                break

        self._closing = True

    @utils.log_exceptions(logger=logger)
    async def _recv_task(self) -> None:
        """Task for receiving messages from Ultravox WebSocket."""
        while True:
            msg = await self._ws.receive()
            if (not self._current_generation or self._current_generation._done) and (
                msg.type in {aiohttp.WSMsgType.BINARY}
            ):
                logger.info("Starting new generation")
                self._start_new_generation()

            if msg.type == aiohttp.WSMsgType.TEXT:
                logger.info("Received text message")
                try:
                    data = json.loads(msg.data)
                    self.emit("ultravox_server_event_received", data)
                    if lk_ultravox_debug:
                        logger.info(f"<<< {data}")
                    event = parse_ultravox_event(data)
                    self._handle_ultravox_event(event)

                except Exception as e:
                    logger.error(f"Error handling message: {e}", exc_info=True)

            elif msg.type == aiohttp.WSMsgType.BINARY:
                logger.info("Received audio data")
                self._handle_audio_data(msg.data)
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                if not self._closing:
                    logger.error(f"Ultravox WebSocket connection closed unexpectedly: {msg}")
                break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"Ultravox WebSocket error: {self._ws.exception()}")
                break

    @utils.log_exceptions(logger=logger)
    def _handle_ultravox_event(self, event: UltravoxEventType) -> None:
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
        else:
            logger.info(f"Unhandled Ultravox event: {event}")

    @utils.log_exceptions(logger=logger)
    def _handle_transcript_event(self, event: TranscriptEvent) -> None:
        """Handle transcript events from Ultravox."""
        if event.role == "user":
            # User transcription - emit input_audio_transcription_completed when final
            if event.final:
                self.emit(
                    "input_audio_transcription_completed",
                    llm.InputTranscriptionCompleted(
                        item_id=f"msg_{event.ordinal}",
                        transcript=event.text or event.delta or "",
                        is_final=True,
                    ),
                )
        elif event.role == "agent":
            text = event.text or event.delta or ""
            if not text:
                return

            if self._current_generation is None:
                self._start_new_generation()

            msg_gen = self._current_generation
            msg_gen.text_ch.send_nowait(text)

            if event.final:
                msg_gen.text_ch.close()
                msg_gen.audio_ch.close()
                self._handle_response_done()

    @utils.log_exceptions(logger=logger)
    def _handle_response_done(self) -> None:
        if self._current_generation is None:
            return

        created_timestamp = self._current_generation._created_timestamp
        first_token_timestamp = self._current_generation._first_token_timestamp

        if not self._current_generation.text_ch.closed:
            self._current_generation.text_ch.close()
        if not self._current_generation.audio_ch.closed:
            self._current_generation.audio_ch.close()

        self._current_generation.function_ch.close()
        self._current_generation.message_ch.close()
        self._current_generation = None

        ttft = first_token_timestamp - created_timestamp if first_token_timestamp else -1
        duration = time.time() - created_timestamp
        metrics = RealtimeModelMetrics(
            timestamp=created_timestamp,
            request_id="",
            ttft=ttft,
            duration=duration,
            cancelled=False,
            label=self._realtime_model._label,
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
        )
        self.emit("metrics_collected", metrics)

    def _handle_state_event(self, event: StateEvent) -> None:
        """Handle state events from Ultravox."""
        logger.info(f"Ultravox state: {event.state}")
        if event.state == "listening":
            self.emit("input_speech_started", llm.InputSpeechStartedEvent())
        elif event.state == "speaking":
            self.emit(
                "input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=True)
            )

    def _handle_tool_invocation_event(self, event: ClientToolInvocationEvent) -> None:
        """Handle tool invocation events from Ultravox."""
        function_call = llm.FunctionCall(
            call_id=event.invocation_id,
            name=event.tool_name,
            arguments=json.dumps(event.parameters),
        )

        self._pending_tool_calls[event.invocation_id] = function_call

        if self._current_generation is None:
            self._start_new_generation()
        self._current_generation.function_ch.send_nowait(function_call)

    def _handle_pong_event(self, event: PongEvent) -> None:
        """Handle pong events from Ultravox."""
        current_time = time.time()
        latency = current_time - event.timestamp
        self.send_event(
            PingEvent(
                timestamp=current_time,
            )
        )
        logger.info(f"Ultravox latency: {latency:.3f}s")

    def _handle_playback_clear_buffer_event(self, event: PlaybackClearBufferEvent) -> None:
        """Handle playback clear buffer events from Ultravox."""
        # logger.info("Ultravox playback buffer cleared")
        # This indicates interruption - clear audio buffers if needed
        pass

    @utils.log_exceptions(logger=logger)
    def _handle_audio_data(self, audio_data: bytes) -> None:
        """Handle binary audio data from Ultravox."""
        try:
            frame = rtc.AudioFrame(
                data=audio_data,
                sample_rate=self._realtime_model._opts.output_sample_rate,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(audio_data) // (2 * NUM_CHANNELS),
            )
            self._current_generation.audio_ch.send_nowait(frame)
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")

    @utils.log_exceptions(logger=logger)
    def _start_new_generation(self) -> None:
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
            _created_timestamp=time.time(),
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
        if self._pending_generation_fut and not self._pending_generation_fut.done():
            generation_ev.user_initiated = True
            self._pending_generation_fut.set_result(generation_ev)
            self._pending_generation_fut = None

        self.emit("generation_created", generation_ev)

    @property
    def chat_ctx(self) -> llm.ChatContext:
        """Get the current chat context."""
        return self._remote_chat_ctx.to_chat_ctx()

    @property
    def tools(self) -> llm.ToolContext:
        """Get the current tool context."""
        return self._tools

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """Update the chat context."""
        pass

    async def update_tools(self, tools: list[llm.FunctionTool | llm.RawFunctionTool]) -> None:
        """Update the available tools."""
        async with self._update_fnc_ctx_lock:
            self._tools = llm.ToolContext(tools)
            # TODO: Send tools to Ultravox if they support dynamic tool updates

    async def update_instructions(self, instructions: str) -> None:
        """Update the system instructions."""
        # Send text message to update instructions
        event = InputTextMessageEvent(text=instructions, defer_response=True)
        self.send_event(event)

    @utils.log_exceptions(logger=logger)
    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Push audio frames to the session for transcription by Ultravox."""
        if self._closed or not self._ws:
            return

        for resampled_frame in self._resample_audio(frame):
            for audio_frame in self._bstream.write(resampled_frame.data.tobytes()):
                if self._ws and not self._ws.closed:
                    asyncio.create_task(self._ws.send_bytes(audio_frame.data.tobytes()))

        self._pushed_duration_s += frame.duration

    def push_video(self, frame: rtc.VideoFrame) -> None:
        """Push video frames (not supported by Ultravox)."""
        pass

    def commit_audio(self) -> None:
        """Commit the current audio input segment."""
        pass

    def clear_audio(self) -> None:
        """Clear the audio buffer."""
        self._bstream._buf.clear()
        self._pushed_duration_s = 0.0

    @utils.log_exceptions(logger=logger)
    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """Generate a reply from the LLM."""
        if self._pending_generation_fut and not self._pending_generation_fut.done():
            logger.warning(
                "generate_reply called while another generation is pending, cancelling previous."
            )
            self._pending_generation_fut.cancel("Superseded by new generate_reply call")

        fut = asyncio.Future()
        self._pending_generation_fut = fut

        if is_given(instructions):
            event = InputTextMessageEvent(text=instructions, defer_response=False)
            self.send_event(event)

        def _on_timeout() -> None:
            if not fut.done():
                fut.set_exception(
                    llm.RealtimeError(
                        "generate_reply timed out waiting for generation_created event."
                    )
                )
                if self._pending_generation_fut is fut:
                    self._pending_generation_fut = None

        timeout_handle = asyncio.get_event_loop().call_later(5.0, _on_timeout)
        fut.add_done_callback(lambda _: timeout_handle.cancel())

        return fut

    def interrupt(self) -> None:
        """Interrupt the current generation."""
        # Ultravox doesn't have a specific interrupt message, but we can clear the buffer
        logger.info("Interrupted current generation")
        self.clear_audio()

    def truncate(self, *, message_id: str, audio_end_ms: int) -> None:
        """Truncate a message (not directly supported by Ultravox)."""
        logger.warning("Message truncation not supported by Ultravox")

    async def aclose(self) -> None:
        """Close the session and clean up resources."""
        if self._closed:
            return

        self._closed = True
        self._msg_ch.close()

        if self._main_atask:
            await utils.aio.cancel_and_wait(self._main_atask)

        for fut in self._response_created_futures.values():
            if not fut.done():
                fut.set_exception(llm.RealtimeError("Session closed before response created"))
        self._response_created_futures.clear()

        if self._current_generation:
            self._mark_current_generation_done()

        await self._cleanup()

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
        if call_id in self._pending_tool_calls:
            event = ClientToolResultEvent(
                invocationId=call_id,
                result=result,
                agent_reaction="speaks",
            )
            self.send_event(event)
            del self._pending_tool_calls[call_id]
        else:
            logger.warning(f"Tool call {call_id} not found in pending calls")
