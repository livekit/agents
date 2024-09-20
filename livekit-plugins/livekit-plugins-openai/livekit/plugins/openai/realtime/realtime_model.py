from __future__ import annotations

import asyncio
import os
import base64
import aiohttp
from typing import AsyncIterable, Literal, Protocol, Union

from dataclasses import dataclass
from livekit.agents import llm, utils
from livekit.agents.llm import _oai_api
from livekit import rtc
from . import api_proto
from .log import logger

ConversationEventTypes = Literal[
    "add_message",
    "message_added",
    "generation_finished",
    "generation_canceled",
]

EventTypes = Union[
    Literal[
        "start_session",
        "error",
        "vad_speech_started",
        "vad_speech_stopped",
        "input_transcribed",
    ],
    ConversationEventTypes,
]


@dataclass
class VadConfig:
    threshold: float
    prefix_padding_ms: float
    silence_duration_ms: float


@dataclass
class StartSessionEvent:
    session_id: str
    model: str
    system_fingerprint: str


@dataclass
class PendingToolCall:
    name: str
    """name of the function"""
    arguments: str
    """accumulated arguments"""
    tool_call_id: str
    """id of the tool call"""


@dataclass
class PendingMessage:
    previous_id: str
    """id of the previous message, usefull for ordering"""
    id: str
    """id of the message"""
    role: api_proto.Role
    """role of the message"""
    text: str
    """accumulated text content"""
    audio: list[rtc.AudioFrame]
    """accumulated audio content"""
    text_stream: AsyncIterable[str]
    """stream of text content"""
    audio_stream: AsyncIterable[rtc.AudioFrame]
    """stream of audio content"""
    tool_calls: list[PendingToolCall]
    """pending tool calls"""


@dataclass
class _ConvConfig:
    voice: api_proto.Voices
    temperature: float
    subscribe_to_user_audio: bool
    max_tokens: int
    disable_audio: bool
    tool_choice: api_proto.ToolChoice


@dataclass
class _ModelOptions:
    base_url: str | None
    transcribe_input: bool
    turn_detection: api_proto.TurnDetectionType
    vad: VadConfig | None
    api_key: str
    conversation_config: _ConvConfig


class _ConversationProtocol(Protocol):
    def update_conversation_config(
        self,
        *,
        voice: api_proto.Voices | None = None,
        subscribe_to_user_audio: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        disable_audio: bool | None = None,
        tool_choice: api_proto.ToolChoice | None = None,
    ) -> None: ...

    @property
    def chat_ctx(self) -> llm.ChatContext: ...

    @property
    def fnc_ctx(self) -> llm.FunctionContext | None: ...

    @fnc_ctx.setter
    def fnc_ctx(self, fnc_ctx: llm.FunctionContext | None) -> None: ...


class RealtimeModel:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__()

        self._base_url = base_url
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "OpenAI API key is required, either using the argument or by setting the OPENAI_API_KEY environmental variable"
            )

        if not base_url:
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        self._api_key = api_key
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def session(
        self,
        *,
        chat_ctx: llm.ChatContext | None = None,
        fnc_ctx: llm.FunctionContext | None = None,
        voice: api_proto.Voices = "alloy",
        turn_detection: api_proto.TurnDetectionType = "server_vad",
        transcribe_input: bool = True,
        vad: VadConfig | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.6,
        disable_audio: bool = False,
        tool_choice: api_proto.ToolChoice = "auto",
        subscribe_to_user_audio: bool = True,
    ) -> RealtimeSession:
        return RealtimeSession(
            chat_ctx=chat_ctx or llm.ChatContext(),
            fnc_ctx=fnc_ctx,
            opts=_ModelOptions(
                api_key=self._api_key,
                transcribe_input=transcribe_input,
                base_url=self._base_url,
                vad=vad,
                turn_detection=turn_detection,
                conversation_config=_ConvConfig(
                    voice=voice,
                    temperature=temperature,
                    subscribe_to_user_audio=subscribe_to_user_audio,
                    max_tokens=max_tokens,
                    disable_audio=disable_audio,
                    tool_choice=tool_choice,
                ),
            ),
            http_session=self._ensure_session(),
        )

    async def aclose(self) -> None:
        pass


class RealtimeSession(utils.EventEmitter[EventTypes], _ConversationProtocol):
    def __init__(
        self,
        *,
        opts: _ModelOptions,
        http_session: aiohttp.ClientSession,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
    ) -> None:
        super().__init__()
        self._main_atask = asyncio.create_task(
            self._main_task(), name="openai-realtime-session"
        )

        self._opts = opts
        self._send_ch = utils.aio.Chan[api_proto.ClientEvents]()
        self._default_conversation = RealtimeConversation(
            "default",
            self._opts.conversation_config,
            chat_ctx,
            fnc_ctx,
            self._send_ch,
        )
        self._conversations = dict[str, RealtimeConversation]()
        self._conversations["default"] = self._default_conversation
        self._default_conversation.update_conversation_config()
        self._http_session = http_session

        self._session_id = "not-connected"
        self._pending_messages = dict[str, PendingMessage]()

    async def aclose(self) -> None:
        if self._send_ch.closed:
            return

        self._send_ch.close()
        await self._main_atask

    def update_conversation_config(
        self,
        *,
        voice: api_proto.Voices | None = None,
        subscribe_to_user_audio: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        disable_audio: bool | None = None,
        tool_choice: api_proto.ToolChoice | None = None,
    ) -> None:
        self._default_conversation.update_conversation_config(
            voice=voice,
            subscribe_to_user_audio=subscribe_to_user_audio,
            temperature=temperature,
            max_tokens=max_tokens,
            disable_audio=disable_audio,
            tool_choice=tool_choice,
        )

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._default_conversation.chat_ctx

    @property
    def fnc_ctx(self) -> llm.FunctionContext | None:
        return self._default_conversation.fnc_ctx

    @fnc_ctx.setter
    def fnc_ctx(self, fnc_ctx: llm.FunctionContext | None) -> None:
        self._default_conversation.fnc_ctx = fnc_ctx

    def truncate_content(self, *, message_id: str, text_chars: int, audio_samples: int):
        self._send_ch.send_nowait(
            {
                "event": "truncate_content",
                "message_id": message_id,
                "index": 0,  # unused for now
                "text_chars": text_chars,
                "audio_samples": audio_samples,
            }
        )

    def add_user_audio(self, frame: rtc.AudioFrame) -> None:
        self._send_ch.send_nowait(
            {
                "event": "add_user_audio",
                "data": base64.b64encode(frame.data).decode("utf-8"),
            }
        )

    def commit_user_audio(self) -> None:
        self._send_ch.send_nowait({"event": "commit_user_audio"})

    def create_conversation(
        self,
        *,
        label: str,
        chat_ctx: llm.ChatContext | None = None,
        fnc_ctx: llm.FunctionContext | None = None,
        voice: api_proto.Voices = "alloy",
        max_tokens: int = 2048,
        temperature: float = 0.6,
        disable_audio: bool = False,
        subscribe_to_user_audio: bool = True,
        tool_choice: api_proto.ToolChoice = "auto",
    ) -> RealtimeConversation:
        if label in self._conversations:
            raise ValueError(f"conversation with label '{label}' already exists")

        self._send_ch.send_nowait(
            {
                "event": "create_conversation",
                "label": label,
            }
        )

        conv_cfg = _ConvConfig(
            voice=voice,
            temperature=temperature,
            subscribe_to_user_audio=subscribe_to_user_audio,
            max_tokens=max_tokens,
            disable_audio=disable_audio,
            tool_choice=tool_choice,
        )

        conv = RealtimeConversation(
            label,
            conv_cfg,
            chat_ctx or llm.ChatContext(),
            fnc_ctx,
            self._send_ch,
        )
        conv.update_conversation_config()
        self._conversations[label] = conv
        return conv

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        try:
            headers = {"Authorization": "Bearer " + self._opts.api_key}
            ws_conn = await self._http_session.ws_connect(
                api_proto.API_URL,  # TODO(theomonnom): Use exposed base_url
                headers=headers,
            )
        except Exception:
            logger.exception("failed to connect to OpenAI API S2S")
            return

        initial_session_cfg: api_proto.ClientEvent.UpdateSessionConfig = {
            "event": "update_session_config",
            "turn_detection": self._opts.turn_detection,
            "input_audio_format": "pcm16",
            "transcribe_input": True,
        }
        await ws_conn.send_json(initial_session_cfg)

        closing = False

        @utils.log_exceptions(logger=logger)
        async def _send_task():
            nonlocal closing
            async for msg in self._send_ch:
                await ws_conn.send_json(msg)

            closing = True
            await ws_conn.close()

        @utils.log_exceptions(logger=logger)
        async def _recv_task():
            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:
                        return

                    raise Exception("OpenAI S2S connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning(
                        "unexpected OpenAI S2S message type %s",
                        msg.type,
                        extra=self.logging_extra(),
                    )
                    continue

                try:
                    data = msg.json()
                    event: api_proto.ServerEventType = data["event"]

                    if event == "start_session":
                        self._handle_start_session(data)
                    elif event == "error":
                        self._handle_error(data)
                    elif event == "vad_speech_started":
                        self._handle_vad_speech_started(data)
                    elif event == "vad_speech_stopped":
                        self._handle_vad_speech_stopped(data)
                    elif event == "input_transcribed":
                        self._handle_input_transcribed(data)
                    elif event == "add_message":
                        self._handle_add_message(data)
                    elif event == "add_content":
                        self._handle_add_content(data)
                    elif event == "message_added":
                        self._handle_message_added(data)
                    elif event == "generation_finished":
                        self._handle_generation_finished(data)
                    elif event == "generation_canceled":
                        self._handle_generation_finished(data)

                except Exception:
                    logger.exception(
                        "failed to handle OpenAI S2S message",
                        extra={"websocket_message": msg, **self.logging_extra()},
                    )

        tasks = [
            asyncio.create_task(_send_task(), name="openai-realtime-send"),
            asyncio.create_task(_recv_task(), name="openai-realtime-recv"),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    def _handle_start_session(self, data: api_proto.ServerEvent.StartSession) -> None:
        logger.debug(
            "session started",
            extra={
                "session_id": data["session_id"],
                "model": data["model"],
                "system_fingerprint": data["system_fingerprint"],
            },
        )
        self._session_id = data["session_id"]
        self.emit(
            "start_session",
            StartSessionEvent(
                session_id=data["session_id"],
                model=data["model"],
                system_fingerprint=data["system_fingerprint"],
            ),
        )

    def _handle_error(self, data: api_proto.ServerEvent.Error) -> None:
        logger.error(
            "error from the OpenAI API",
            extra={"message": data["message"], **self.logging_extra()},
        )
        self.emit("error", data["message"])

    def _handle_vad_speech_started(
        self, data: api_proto.ServerEvent.VadSpeechStarted
    ) -> None:
        self.emit("vad_speech_started")

    def _handle_vad_speech_stopped(
        self, data: api_proto.ServerEvent.VadSpeechStopped
    ) -> None:
        self.emit("vad_speech_stopped")

    def _handle_input_transcribed(self, data: api_proto.ServerEvent.InputTranscribed):
        self.emit("input_transcribed", transcript=data["transcript"])

    def _handle_add_message(self, data: api_proto.ServerEvent.AddMessage):
        conv = self._conversations[data["conversation_label"]]
        previous_id = data["previous_id"]
        message = data["message"]
        id = message["id"]  # type: ignore
        role = message["role"]
        content = message["content"]

        if previous_id in self._pending_messages:
            logger.warning(
                "received message with pending previous_id %s, discarding",
                previous_id,
            )
            return

        tool_calls = []
        for cnt in content:
            if cnt["type"] == "tool_call":
                tool_calls.append(
                    PendingToolCall(
                        name=cnt["name"],
                        arguments="",
                        tool_call_id=cnt["tool_call_id"],
                    )
                )

        msg = PendingMessage(
            previous_id=previous_id,
            id=id,
            role=role,
            text="",
            audio=[],
            tool_calls=tool_calls,
            text_stream=utils.aio.Chan[str](),
            audio_stream=utils.aio.Chan[rtc.AudioFrame](),
        )
        self._pending_messages[id] = msg
        conv._handle_add_message(msg)

    def _handle_add_content(self, data: api_proto.ServerEvent.AddContent):
        msg = self._pending_messages.get(data["message_id"])
        if msg is None:
            logger.warning(
                "received content for unknown message %s, discarding",
                data["message_id"],
            )
            return

        assert isinstance(msg.text_stream, utils.aio.Chan)
        assert isinstance(msg.audio_stream, utils.aio.Chan)

        if data["type"] == "text":
            msg.text += data["data"]
            msg.text_stream.send_nowait(data["data"])
        elif data["type"] == "audio":
            pcm_data = base64.b64decode(data["data"])
            frame = rtc.AudioFrame(
                pcm_data,
                api_proto.SAMPLE_RATE,
                api_proto.NUM_CHANNELS,
                len(pcm_data) // 2,
            )

            msg.audio.append(frame)
            msg.audio_stream.send_nowait(frame)
        elif data["type"] == "tool_call":
            tool_call = msg.tool_calls[-1]
            tool_call.arguments += data["data"]

    def _handle_message_added(self, data: api_proto.ServerEvent.MessageAdded):
        conv = self._conversations[data["conversation_label"]]
        pending_msg = self._pending_messages.pop(data["id"])

        assert isinstance(pending_msg.text_stream, utils.aio.Chan)
        assert isinstance(pending_msg.audio_stream, utils.aio.Chan)

        pending_msg.text_stream.close()
        pending_msg.audio_stream.close()

        conv._handle_message_added(pending_msg)

    def _handle_generation_finished(
        self, data: api_proto.ServerEvent.GenerationFinished
    ):
        if data["reason"] not in ("stop", "interrupt"):
            logger.warning(
                "generation finished with reason %s",
                extra={
                    "conversation_label": data["conversation_label"],
                    "reason": data["reason"],
                    "message_ids": data["message_ids"],
                    **self.logging_extra(),
                },
            )

        conv = self._conversations[data["conversation_label"]]
        conv._handle_generation_finished(data)

    def _handle_generation_canceled(
        self, data: api_proto.ServerEvent.GenerationCanceled
    ):
        conv = self._conversations[data["conversation_label"]]
        conv._handle_generation_canceled(data)

    def logging_extra(self) -> dict:
        return {"session_id": self._session_id}


class RealtimeConversation(
    utils.EventEmitter[ConversationEventTypes], _ConversationProtocol
):
    def __init__(
        self,
        label: str,
        config: _ConvConfig,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
        send_ch: utils.aio.Chan[api_proto.ClientEvents],
    ) -> None:
        super().__init__()
        self._label = label
        self._chat_ctx = chat_ctx
        self._fnc_ctx = fnc_ctx
        self._config = config
        self._send_ch = send_ch

    def update_conversation_config(
        self,
        *,
        voice: api_proto.Voices | None = None,
        subscribe_to_user_audio: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        disable_audio: bool | None = None,
        tool_choice: api_proto.ToolChoice | None = None,
    ) -> None:
        self._config.voice = voice if voice is not None else self._config.voice
        self._config.subscribe_to_user_audio = (
            subscribe_to_user_audio
            if subscribe_to_user_audio is not None
            else self._config.subscribe_to_user_audio
        )
        self._config.temperature = (
            temperature if temperature is not None else self._config.temperature
        )
        self._config.max_tokens = (
            max_tokens if max_tokens is not None else self._config.max_tokens
        )
        self._config.disable_audio = (
            disable_audio if disable_audio is not None else self._config.disable_audio
        )
        self._config.tool_choice = (
            tool_choice if tool_choice is not None else self._config.tool_choice
        )

        tools = []
        if self._fnc_ctx is not None:
            for fnc in self._fnc_ctx.ai_functions.values():
                tools.append(llm._oai_api.build_oai_function_description(fnc))

        self._send_ch.send_nowait(
            {
                "event": "update_conversation_config",
                "conversation_label": self._label,
                "disable_audio": self._config.disable_audio,
                "max_tokens": self._config.max_tokens,
                "output_audio_format": "pcm16",
                "subscribe_to_user_audio": self._config.subscribe_to_user_audio,
                "system_message": _find_first_system_message(self._chat_ctx) or "",
                "temperature": self._config.temperature,
                "voice": self._config.voice,
                "tools": tools,
                "tool_choice": self._config.tool_choice,
            }
        )

    def delete(self) -> None:
        if self._label == "default":
            raise ValueError("cannot delete the default conversation")

        self._send_ch.send_nowait(
            {
                "event": "delete_conversation",
                "label": self._label,
            }
        )

    def generate(self) -> None:
        self._send_ch.send_nowait(
            {
                "event": "generate",
                "conversation_label": self._label,
            }
        )

    def cancel_generation(self) -> None:
        self._send_ch.send_nowait(
            {
                "event": "cancel_generation",
                "conversation_label": self._label,
            }
        )

    def add_message(
        self,
        *,
        message: llm.ChatMessage | list[llm.ChatMessage],
        previous_id: str | None = None,
    ) -> None:
        if isinstance(message, llm.ChatMessage):
            message = [message]

        oai_messages: list[api_proto.Message] = []
        for msg in message:
            oai_content: list[api_proto.MessageContent] = []

            if isinstance(msg.content, str):
                oai_content.append({"type": "text", "text": msg.content})
            elif isinstance(msg.content, llm.ChatAudio):
                frames = msg.content.frame
                if isinstance(frames, rtc.AudioFrame):
                    frames = [frames]

                frame = utils.merge_frames(frames)
                b64_data = base64.b64encode(frame.data).decode("utf-8")
                oai_content.append({"type": "audio", "audio": b64_data})

            oai_messages.append(
                {
                    "id": msg.id or "",
                    "role": msg.role,
                    "tool_call_id": msg.tool_call_id or "",
                    "content": oai_content,
                }
            )

        self._send_ch.send_nowait(
            {
                "event": "add_message",
                "previous_id": previous_id or "",
                "conversation_label": self._label,
                "message": oai_messages,
            }
        )

    def delete_message(self, *, message_id: str) -> None:
        self._send_ch.send_nowait(
            {
                "event": "delete_message",
                "id": message_id,
                "conversation_label": self._label,
            }
        )

    def _handle_add_message(self, msg: PendingMessage):
        self.emit("add_message", msg)

    def _handle_message_added(self, msg: PendingMessage):
        idx = len(self._chat_ctx.messages)
        for i, lk_msg in enumerate(self._chat_ctx.messages):
            if lk_msg.id == msg.previous_id:
                idx = i + 1
                break

        lk_msg = llm.ChatMessage(id=msg.id, role=msg.role)
        lk_msg.content = []

        if msg.text:
            lk_msg.content.append(msg.text)

        if msg.audio:
            lk_msg.content.append(llm.ChatAudio(frame=msg.audio))

        if msg.tool_calls:
            lk_msg.tool_calls = []

            for tool_call in msg.tool_calls:
                if (
                    self._fnc_ctx is None
                    or tool_call.name not in self._fnc_ctx.ai_functions
                ):
                    # TODO(theomonnom): this could happens in a bad timing after updating the
                    # conversation config. (we should be able to recover from this bad timing)
                    logger.warning(
                        "received tool call for unknown function %s, discarding",
                        tool_call.name,
                    )
                    continue

                lk_msg.tool_calls.append(
                    _oai_api.create_ai_function_info(
                        self._fnc_ctx,
                        tool_call.tool_call_id,
                        tool_call.name,
                        tool_call.arguments,
                    )
                )

        self._chat_ctx.messages.insert(idx, lk_msg)
        self.emit("message_added", lk_msg)

    def _handle_generation_finished(
        self, data: api_proto.ServerEvent.GenerationFinished
    ):
        self.emit("generation_finished", data["reason"])

    def _handle_generation_canceled(
        self, data: api_proto.ServerEvent.GenerationCanceled
    ):
        self.emit("generation_canceled")

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def fnc_ctx(self) -> llm.FunctionContext | None:
        return self._fnc_ctx

    @fnc_ctx.setter
    def fnc_ctx(self, fnc_ctx: llm.FunctionContext | None) -> None:
        self._fnc_ctx = fnc_ctx


def _find_first_system_message(chat_ctx: llm.ChatContext) -> str | None:
    for msg in chat_ctx.messages:
        if msg.role == "system":
            assert isinstance(msg.content, str)
            return msg.content
