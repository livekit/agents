from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import typing
import weakref
from dataclasses import dataclass, field
from typing import Literal

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import audio as audio_utils, is_given
from phonic import AsyncPhonic
from phonic.conversations.socket_client import (
    AsyncConversationsSocketClient,
)
from phonic.types import (
    AddSystemMessagePayload,
    AudioChunkPayload,
    AudioChunkResponsePayload,
    ConfigPayload,
    GenerateReplyPayload,
    InputTextPayload,
    ToolCallInterruptedPayload,
    ToolCallOutputPayload,
    ToolCallPayload,
)

from ..log import logger

PHONIC_INPUT_SAMPLE_RATE = 44100
PHONIC_OUTPUT_SAMPLE_RATE = 44100
PHONIC_NUM_CHANNELS = 1
PHONIC_INPUT_FRAME_MS = 20
WS_CLOSE_NORMAL = 1000
TOOL_CALL_OUTPUT_TIMEOUT_MS = 60000


@dataclass
class _RealtimeOptions:
    api_key: str
    phonic_agent: NotGivenOr[str]
    voice: NotGivenOr[str]
    welcome_message: NotGivenOr[str | None]
    generate_welcome_message: NotGivenOr[bool | None]
    project: NotGivenOr[str | None]
    languages: NotGivenOr[list[str]]
    audio_speed: NotGivenOr[float]
    phonic_tools: NotGivenOr[list[str]]
    boosted_keywords: NotGivenOr[list[str]]
    generate_no_input_poke_text: NotGivenOr[bool]
    no_input_poke_sec: NotGivenOr[float]
    no_input_poke_text: NotGivenOr[str]
    no_input_end_conversation_sec: NotGivenOr[float]
    conn_options: APIConnectOptions
    instructions: NotGivenOr[str] = NOT_GIVEN


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]

    response_id: str
    input_id: str

    input_transcription: str = ""
    output_text: str = ""

    _created_timestamp: float = field(default_factory=time.time)
    _done: bool = False

    def push_text(self, text: str) -> None:
        if self.output_text:
            self.output_text += text
        else:
            self.output_text = text

        self.text_ch.send_nowait(text)


class RealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        phonic_agent: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        welcome_message: NotGivenOr[str | None] = NOT_GIVEN,
        generate_welcome_message: NotGivenOr[bool] = NOT_GIVEN,
        project: NotGivenOr[str | None] = NOT_GIVEN,
        languages: NotGivenOr[list[str]] = NOT_GIVEN,
        audio_speed: NotGivenOr[float] = NOT_GIVEN,
        phonic_tools: NotGivenOr[list[str]] = NOT_GIVEN,
        boosted_keywords: NotGivenOr[list[str]] = NOT_GIVEN,
        generate_no_input_poke_text: NotGivenOr[bool] = NOT_GIVEN,
        no_input_poke_sec: NotGivenOr[float] = NOT_GIVEN,
        no_input_poke_text: NotGivenOr[str] = NOT_GIVEN,
        no_input_end_conversation_sec: NotGivenOr[float] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Initialize a RealtimeModel for Phonic's Realtime API.

        Args:
            api_key: Phonic API key. If not provided, reads from PHONIC_API_KEY environment variable.
            phonic_agent: Phonic agent to use for the conversation. Options explicitly set
                here will override the agent's default settings.
            voice: Voice ID for agent audio output.
            welcome_message: Message for the agent to say when the conversation starts.
                Ignored when ``generate_welcome_message`` is True.
            generate_welcome_message: When True, the welcome message is automatically generated
                and ``welcome_message`` is ignored.
            project: Project name to use for the conversation.
            languages: ISO 639-1 language codes the agent should recognize and speak.
            audio_speed: Audio playback speed multiplier.
            phonic_tools: Phonic tool names available to the assistant.
            boosted_keywords: Keywords to boost in speech recognition.
            generate_no_input_poke_text: When True, auto-generate poke text when the user is silent.
            no_input_poke_sec: Seconds of silence before sending a poke message.
            no_input_poke_text: Custom poke message text. Ignored when
                ``generate_no_input_poke_text`` is True.
            no_input_end_conversation_sec: Seconds of silence before ending the conversation.
            conn_options: Retry/backoff and connection settings.
        """
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=True,
                audio_output=True,
                manual_function_calls=False,
            )
        )

        api_key = api_key or os.environ.get("PHONIC_API_KEY", NOT_GIVEN)
        if not is_given(api_key):
            raise ValueError(
                "Phonic API key is required. Provide `api_key` or "
                "set PHONIC_API_KEY environment variable."
            )

        self._opts = _RealtimeOptions(
            api_key=api_key,
            phonic_agent=phonic_agent,
            voice=voice,
            welcome_message=welcome_message,
            generate_welcome_message=generate_welcome_message,
            project=project,
            languages=languages,
            audio_speed=audio_speed,
            phonic_tools=phonic_tools,
            boosted_keywords=boosted_keywords,
            generate_no_input_poke_text=generate_no_input_poke_text,
            no_input_poke_sec=no_input_poke_sec,
            no_input_poke_text=no_input_poke_text,
            no_input_end_conversation_sec=no_input_end_conversation_sec,
            conn_options=conn_options,
        )

        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return "phonic"

    @property
    def provider(self) -> str:
        return "phonic"

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    def update_options(
        self,
    ) -> None:
        logger.warning("update_options is not supported by the Phonic realtime model.")

    async def aclose(self) -> None:
        pass


class RealtimeSession(llm.RealtimeSession):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._opts = realtime_model._opts
        self._tools = llm.ToolContext.empty()
        self._chat_ctx = llm.ChatContext.empty()

        self._bstream = audio_utils.AudioByteStream(
            sample_rate=PHONIC_INPUT_SAMPLE_RATE,
            num_channels=PHONIC_NUM_CHANNELS,
            samples_per_channel=PHONIC_INPUT_SAMPLE_RATE * PHONIC_INPUT_FRAME_MS // 1000,
        )
        self._input_resampler: rtc.AudioResampler | None = None
        self._input_resampler_rate: int | None = None

        self._client = AsyncPhonic(
            api_key=self._opts.api_key,
        )

        self._socket: AsyncConversationsSocketClient | None = None
        self._socket_ctx: typing.AsyncContextManager[AsyncConversationsSocketClient] | None = None
        self._send_ch = utils.aio.Chan[AudioChunkPayload]()
        self._main_atask = asyncio.create_task(self._main_task(), name="phonic-realtime-session")

        self._current_generation: _ResponseGeneration | None = None
        self._conversation_id: str | None = None

        self._session_should_close = asyncio.Event()
        self._session_lock = asyncio.Lock()

        self._generate_reply_task: asyncio.Task[None] | None = None
        self._instructions_ready = asyncio.Event()
        self._tools_ready = asyncio.Event()
        self._ready_to_start = asyncio.Event()
        self._config_sent = False
        self._pending_tool_call_ids: set[str] = set()
        self._tool_definitions: list[dict] = []
        self._system_prompt_postfix: str = ""

    async def _close_active_session(self) -> None:
        async with self._session_lock:
            if self._socket_ctx:
                try:
                    await self._socket_ctx.__aexit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Error closing Phonic socket: {e}")
                finally:
                    self._socket = None
                    self._socket_ctx = None

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    async def update_instructions(self, instructions: str) -> None:
        if self._config_sent:
            logger.warning(
                "update_instructions called after config was already sent. "
                "Phonic does not support updating instructions mid-session."
            )
            return
        self._opts.instructions = instructions
        self._instructions_ready.set()

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        if not self._config_sent:
            messages = [
                item
                for item in chat_ctx.items
                if isinstance(item, llm.ChatMessage)
                and item.text_content
                and item.text_content.strip()
            ]
            if messages:
                turn_history = "\n".join(f"{m.role}: {m.text_content}" for m in messages)
                if turn_history.strip():
                    logger.debug(
                        "update_chat_ctx called with messages prior to config being sent to "
                        "Phonic. Including conversation state in system instructions."
                    )
                    self._system_prompt_postfix = (
                        "\n\nThis conversation is being continued from an existing "
                        "conversation. You are the assistant speaking to the user. "
                        "The following is the conversation history:\n" + turn_history
                    )
                self._chat_ctx = chat_ctx.copy()
            return

        diff_ops = llm.utils.compute_chat_ctx_diff(self._chat_ctx, chat_ctx)
        sent_tool_call_output = False
        sent_system_message = False

        for _, item_id in diff_ops.to_create:
            item = chat_ctx.get_by_id(item_id)
            if item is None:
                continue

            if (
                isinstance(item, llm.FunctionCallOutput)
                and item.call_id in self._pending_tool_call_ids
            ):
                self._pending_tool_call_ids.remove(item.call_id)
                logger.info(f"Sending tool call output for {item.name} (call_id: {item.call_id})")
                if self._socket:
                    await self._socket.send_tool_call_output(
                        ToolCallOutputPayload(
                            tool_call_id=item.call_id,
                            output=str(item.output),
                        )
                    )
                    sent_tool_call_output = True

            if isinstance(item, llm.ChatMessage) and item.role in ("system", "developer"):
                text = item.text_content
                if text:
                    logger.debug(f"Sending add system message: {text}")
                    if self._socket:
                        await self._socket.send_add_system_message(
                            AddSystemMessagePayload(system_message=text)
                        )
                        sent_system_message = True

        self._chat_ctx = chat_ctx.copy()

        if not sent_tool_call_output and not sent_system_message:
            logger.warning(
                "update_chat_ctx called but no new tool call outputs to send. "
                "Phonic does not support general chat context updates."
            )
        if sent_tool_call_output:
            self._start_new_assistant_turn()

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        if self._config_sent:
            logger.warning(
                "update_tools called after config was already sent. "
                "Phonic does not support updating tools mid-session."
            )
            return

        self._tools = llm.ToolContext(tools)
        self._tool_definitions = []
        for tool_schema in self._tools.parse_function_tools("openai", strict=True):
            # We disallow tool chaining and tool calls during agent speech to reduce complexity
            # of managing state while operating within the LiveKit Realtime generations framework
            self._tool_definitions.append(
                {
                    "type": "custom_websocket",
                    "tool_schema": tool_schema,
                    "tool_call_output_timeout_ms": TOOL_CALL_OUTPUT_TIMEOUT_MS,
                    "wait_for_speech_before_tool_call": True,
                    "allow_tool_chaining": False,
                }
            )

        self._tools_ready.set()

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        logger.warning("update_options is not supported by the Phonic realtime model.")

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if (
            self._session_should_close.is_set()
            or not self._ready_to_start.is_set()
            or not self._socket
        ):
            return

        for f in self._resample_audio(frame):
            for nf in self._bstream.write(f.data.tobytes()):
                b64_audio = base64.b64encode(nf.data.tobytes()).decode("utf-8")
                self._send_ch.send_nowait(AudioChunkPayload(audio=b64_audio))

    def push_video(self, frame: rtc.VideoFrame) -> None:
        logger.warning("push_video is not supported by the Phonic realtime model.")

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        payload = GenerateReplyPayload(
            system_message=instructions if is_given(instructions) else None,
        )
        if self._generate_reply_task and not self._generate_reply_task.done():
            self._generate_reply_task.cancel()
        self._generate_reply_task = asyncio.create_task(self._send_generate_reply(payload))

        self._close_current_generation(interrupted=False)
        generation_ev = self._start_new_assistant_turn(user_initiated=True)
        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        fut.set_result(generation_ev)
        return fut

    async def _send_generate_reply(self, payload: GenerateReplyPayload) -> None:
        await self._ready_to_start.wait()
        if self._session_should_close.is_set():
            return
        if self._socket:
            await self._socket.send_generate_reply(payload)

    def commit_audio(self) -> None:
        logger.warning("commit_audio is not supported by the Phonic realtime model.")

    def clear_audio(self) -> None:
        logger.warning("clear_audio is not supported by the Phonic realtime model.")

    def interrupt(self) -> None:
        if self._current_generation:
            logger.warning(
                "interrupt() is not supported by Phonic realtime model. "
                "User interruptions are automatically handled by Phonic."
            )

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        logger.warning(
            "truncate is not supported by the Phonic realtime model. "
            "User interruptions are automatically handled by Phonic."
        )

    async def aclose(self) -> None:
        self._session_should_close.set()
        self._send_ch.close()
        self._instructions_ready.set()
        self._tools_ready.set()
        self._ready_to_start.set()

        self._close_current_generation(interrupted=False)

        if self._generate_reply_task and not self._generate_reply_task.done():
            await utils.aio.cancel_and_wait(self._generate_reply_task)

        if self._main_atask:
            await utils.aio.cancel_and_wait(self._main_atask)

        await self._close_active_session()

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        try:
            logger.debug("Connecting to Phonic Realtime API...")
            # The Phonic Python SDK uses an async context manager for connect()
            self._socket_ctx = self._client.conversations.connect()
            self._socket = await self._socket_ctx.__aenter__()

            # Need to wait for instructions and tools before sending config
            await self._instructions_ready.wait()
            await self._tools_ready.wait()

            if self._session_should_close.is_set():
                return

            self._config_sent = True

            tools_payload: list[dict | str] = []
            if self._opts.phonic_tools is not NOT_GIVEN and self._opts.phonic_tools:
                tools_payload.extend(self._opts.phonic_tools)
            tools_payload.extend(self._tool_definitions)

            if not is_given(self._opts.instructions):
                logger.warning("Instructions are not set. Phonic will not start a conversation.")
                return

            config = {
                "type": "config",
                "agent": self._opts.phonic_agent,
                "project": self._opts.project,
                "welcome_message": self._opts.welcome_message,
                "generate_welcome_message": self._opts.generate_welcome_message,
                "system_prompt": self._opts.instructions + self._system_prompt_postfix,
                "voice_id": self._opts.voice,
                "input_format": "pcm_44100",
                "output_format": "pcm_44100",
                "recognized_languages": self._opts.languages,
                "audio_speed": self._opts.audio_speed,
                "tools": tools_payload if len(tools_payload) > 0 else NOT_GIVEN,
                "boosted_keywords": self._opts.boosted_keywords,
                "generate_no_input_poke_text": self._opts.generate_no_input_poke_text,
                "no_input_poke_sec": self._opts.no_input_poke_sec,
                "no_input_poke_text": self._opts.no_input_poke_text,
                "no_input_end_conversation_sec": self._opts.no_input_end_conversation_sec,
            }
            # Filter out NOT_GIVEN values
            config_filtered = typing.cast(
                dict[str, typing.Any],
                {k: v for k, v in config.items() if v is not NOT_GIVEN},
            )
            await self._socket.send_config(ConfigPayload(**config_filtered))

            recv_task = asyncio.create_task(self._recv_task(self._socket), name="phonic-recv")
            send_task = asyncio.create_task(self._send_task(self._socket), name="phonic-send")
            shutdown_wait_task = asyncio.create_task(
                self._session_should_close.wait(), name="phonic-shutdown-wait"
            )

            done, pending = await asyncio.wait(
                [recv_task, send_task, shutdown_wait_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                exception = task.exception()
                if task is not shutdown_wait_task and exception:
                    logger.error(f"Error in Phonic task: {exception}")
                    raise exception

            for task in pending:
                await utils.aio.cancel_and_wait(task)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Phonic Realtime API error: {e}", exc_info=e)
            self._emit_error(e, recoverable=False)
        finally:
            await self._close_active_session()
            self._close_current_generation(interrupted=False)

    @utils.log_exceptions(logger=logger)
    async def _send_task(self, socket: AsyncConversationsSocketClient) -> None:
        async for payload in self._send_ch:
            await socket.send_audio_chunk(payload)

    @utils.log_exceptions(logger=logger)
    async def _recv_task(self, socket: AsyncConversationsSocketClient) -> None:
        try:
            async for message in socket:
                if self._session_should_close.is_set():
                    break

                msg_type = message.type

                if msg_type == "assistant_started_speaking":
                    self._start_new_assistant_turn()
                elif msg_type == "assistant_finished_speaking":
                    self._close_current_generation(interrupted=False)
                elif msg_type == "audio_chunk":
                    self._handle_audio_chunk(message)
                elif msg_type == "input_text":
                    self._handle_input_text(message)
                elif msg_type == "user_started_speaking":
                    self._handle_input_speech_started()
                elif msg_type == "user_finished_speaking":
                    self._handle_input_speech_stopped()
                elif msg_type == "tool_call":
                    self._handle_tool_call(message)
                elif msg_type == "warning":
                    logger.warning(f"Phonic warning: {message.warning.message}")
                elif msg_type == "error":
                    self._emit_error(Exception(message.error.message), recoverable=False)
                elif msg_type == "assistant_ended_conversation":
                    self._emit_error(
                        Exception(
                            "assistant_ended_conversation is not supported by "
                            "the Phonic realtime model with LiveKit Agents."
                        ),
                        recoverable=False,
                    )
                elif msg_type == "conversation_created":
                    self._conversation_id = message.conversation_id
                    logger.info(f"Phonic Conversation began with ID: {self._conversation_id}")
                elif msg_type == "tool_call_interrupted":
                    self._handle_tool_call_interrupted(message)
                elif msg_type == "ready_to_start_conversation":
                    self._ready_to_start.set()
        except Exception as e:
            if not self._session_should_close.is_set():
                logger.error(f"Error in Phonic receive loop: {e}", exc_info=e)
                self._emit_error(e, recoverable=True)
                raise e

    def _start_new_assistant_turn(self, user_initiated: bool = False) -> llm.GenerationCreatedEvent:
        if self._current_generation:
            self._close_current_generation(interrupted=True)

        response_id = utils.shortuuid("PS_")
        self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            text_ch=utils.aio.Chan[str](),
            audio_ch=utils.aio.Chan[rtc.AudioFrame](),
            response_id=response_id,
            input_id=utils.shortuuid("PI_"),
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
            user_initiated=user_initiated,
            response_id=response_id,
        )
        self.emit("generation_created", generation_ev)
        return generation_ev

    def _close_current_generation(self, interrupted: bool) -> None:
        gen = self._current_generation
        if not gen or gen._done:
            return

        if gen.output_text:
            self._chat_ctx.add_message(
                role="assistant",
                content=gen.output_text,
                id=gen.response_id,
                interrupted=interrupted,
            )

        if not gen.text_ch.closed:
            gen.text_ch.send_nowait("")
            gen.text_ch.close()
        if not gen.audio_ch.closed:
            gen.audio_ch.close()

        gen.function_ch.close()
        gen.message_ch.close()
        gen._done = True
        self._current_generation = None

    def _handle_audio_chunk(self, message: AudioChunkResponsePayload) -> None:
        # In Phonic, audio chunks can come in when assistant isn't explicitly active.
        # We start a generation if text is present to align with the framework pattern.
        if self._current_generation is None and message.text:
            logger.debug("Starting new generation due to text in audio chunk")
            self._start_new_assistant_turn()

        gen = self._current_generation
        if gen is None:
            return

        if message.text:
            gen.push_text(message.text)

        if message.audio:
            try:
                audio_bytes = base64.b64decode(message.audio)
                sample_count = len(audio_bytes) // 2  # 16-bit PCM = 2 bytes per sample
                if sample_count > 0:
                    frame = rtc.AudioFrame(
                        data=audio_bytes,
                        sample_rate=PHONIC_OUTPUT_SAMPLE_RATE,
                        num_channels=PHONIC_NUM_CHANNELS,
                        samples_per_channel=sample_count // PHONIC_NUM_CHANNELS,
                    )
                    gen.audio_ch.send_nowait(frame)
            except Exception as e:
                logger.error(f"Failed to decode Phonic audio chunk: {e}")

    def _handle_input_text(self, message: InputTextPayload) -> None:
        item_id = utils.shortuuid("PI_")
        transcript = message.text

        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=item_id,
                transcript=transcript,
                is_final=True,
            ),
        )

        self._chat_ctx.add_message(
            role="user",
            content=transcript,
            id=item_id,
        )

    def _handle_tool_call(self, message: ToolCallPayload) -> None:
        tool_call_id = message.tool_call_id
        tool_name = message.tool_name
        parameters = message.parameters

        self._pending_tool_call_ids.add(tool_call_id)

        if self._current_generation is None:
            logger.warning("Encountered tool call but no active generation. Starting new turn.")
            self._start_new_assistant_turn()

        assert self._current_generation is not None, (
            "current_generation should not be None when handling tool call"
        )

        self._current_generation.function_ch.send_nowait(
            llm.FunctionCall(
                call_id=tool_call_id,
                name=tool_name,
                arguments=json.dumps(parameters),
            )
        )

        # At most 1 tool call is supported per turn due to `allow_tool_chaining: False`,
        # allowing us to close the generation.
        self._close_current_generation(interrupted=False)

    def _handle_tool_call_interrupted(self, message: ToolCallInterruptedPayload) -> None:
        tool_call_id = message.tool_call_id
        tool_name = message.tool_name

        if tool_call_id in self._pending_tool_call_ids:
            self._pending_tool_call_ids.remove(tool_call_id)

        logger.warning(
            f"Tool call for {tool_name} (call_id: {tool_call_id}) "
            "was cancelled due to user interruption."
        )

    def _handle_input_speech_started(self) -> None:
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())
        self._close_current_generation(interrupted=True)

    def _handle_input_speech_stopped(self) -> None:
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(user_transcription_enabled=True),
        )

    def _resample_audio(self, frame: rtc.AudioFrame) -> typing.Iterator[rtc.AudioFrame]:
        if self._input_resampler is not None:
            if frame.sample_rate != self._input_resampler_rate:
                self._input_resampler = None
                self._input_resampler_rate = None

        if self._input_resampler is None and (
            frame.sample_rate != PHONIC_INPUT_SAMPLE_RATE
            or frame.num_channels != PHONIC_NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=PHONIC_INPUT_SAMPLE_RATE,
                num_channels=PHONIC_NUM_CHANNELS,
            )
            self._input_resampler_rate = frame.sample_rate

        if self._input_resampler is not None:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

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
