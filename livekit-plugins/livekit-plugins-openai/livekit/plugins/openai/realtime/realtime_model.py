from __future__ import annotations

import asyncio
import base64
import functools
import os
from dataclasses import dataclass
from typing import AsyncIterable, Literal, Protocol, Union

import aiohttp
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.llm import _oai_api

from . import api_proto
from .log import logger


EventTypes = Literal[
    "start_session",
    "error",
    "vad_speech_started",
    "vad_speech_stopped",
    "input_transcribed",
    "add_message",
    "message_added",
    "generation_finished",
    "generation_canceled",
]


@dataclass
class _ModelOptions:
    modalities: list[api_proto.Modality]
    instructions: str | None
    voice: api_proto.Voice
    input_audio_format: api_proto.AudioFormat
    output_audio_format: api_proto.AudioFormat
    input_audio_transcription: api_proto.InputAudioTranscription | None
    turn_detection: api_proto.TurnDetectionType
    temparature: float
    max_output_tokens: int
    api_key: str
    base_url: str


class RealtimeModel:
    def __init__(
        self,
        *,
        modalities: list[api_proto.Modality] = ["text", "audio"],
        instructions: str | None = None,
        voice: api_proto.Voice = "alloy",
        input_audio_format: api_proto.AudioFormat = "pcm16",
        output_audio_format: api_proto.AudioFormat = "pcm16",
        input_audio_transcription: api_proto.InputAudioTranscription = {
            "model": "whisper-1"
        },
        turn_detection: api_proto.TurnDetectionType = {"type": "server_vad"},
        temparature: float = 0.6,
        max_output_tokens: int = 2048,
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

        self._default_opts = _ModelOptions(
            modalities=modalities,
            instructions=instructions,
            voice=voice,
            input_audio_format=input_audio_format,
            output_audio_format=output_audio_format,
            input_audio_transcription=input_audio_transcription,
            turn_detection=turn_detection,
            temparature=temparature,
            max_output_tokens=max_output_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        self._http_session = http_session
        self._rt_sessions = []

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            self._http_session = utils.http_context.http_session()

        return self._http_session

    @property
    def sessions(self) -> list[RealtimeSession]:
        return self._rt_sessions

    def session(
        self,
        *,
        chat_ctx: llm.ChatContext | None = None,
        fnc_ctx: llm.FunctionContext | None = None,
        modalities: list[api_proto.Modality] | None = None,
        instructions: str | None = None,
        voice: api_proto.Voice | None = None,
        input_audio_format: api_proto.AudioFormat | None = None,
        output_audio_format: api_proto.AudioFormat | None = None,
        input_audio_transcription: api_proto.InputAudioTranscription | None = None,
        turn_detection: api_proto.TurnDetectionType | None = None,
        temparature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> RealtimeSession:
        opts = _ModelOptions(
            modalities=modalities or self._default_opts.modalities,
            instructions=instructions or self._default_opts.instructions,
            voice=voice or self._default_opts.voice,
            input_audio_format=input_audio_format
            or self._default_opts.input_audio_format,
            output_audio_format=output_audio_format
            or self._default_opts.output_audio_format,
            input_audio_transcription=(
                input_audio_transcription
                or self._default_opts.input_audio_transcription
            ),
            turn_detection=turn_detection or self._default_opts.turn_detection,
            temparature=temparature or self._default_opts.temparature,
            max_output_tokens=max_output_tokens or self._default_opts.max_output_tokens,
            api_key=self._default_opts.api_key,
            base_url=self._default_opts.base_url,
        )

        new_session = RealtimeSession(
            chat_ctx=chat_ctx or llm.ChatContext(),
            fnc_ctx=fnc_ctx,
            opts=opts,
            http_session=self._ensure_session(),
        )
        self._rt_sessions.append(new_session)
        return new_session

    async def aclose(self) -> None:
        for session in self._rt_sessions:
            await session.aclose()


class RealtimeSession(utils.EventEmitter[EventTypes]):
    class InputAudioBuffer:
        def __init__(self, sess: RealtimeSession) -> None:
            self._sess = sess

        def append(self, frame: rtc.AudioFrame) -> None:
            self._sess._queue_msg(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(frame.data).decode("utf-8"),
                }
            )

        def clear(self) -> None:
            self._sess._queue_msg({"type": "input_audio_buffer.clear"})

        def commit(self) -> None:
            self._sess._queue_msg({"type": "input_audio_buffer.commit"})

    class ConvsersationItem:
        def __init__(self, sess: RealtimeSession) -> None:
            self._sess = sess

        def create(self, message: llm.ChatMessage) -> None:
            message_content = message.content
            if not isinstance(message_content, list):
                message_content = [message.content]

            event: api_proto.ClientEvent.ConversationItemCreate | None = None
            if message.role == "user":
                user_contents: list[
                    api_proto.InputTextContent | api_proto.InputAudioContent
                ] = []
                for cnt in message_content:
                    if isinstance(cnt, str):
                        user_contents.append(
                            {
                                "type": "input_text",
                                "text": cnt,
                            }
                        )
                    elif isinstance(cnt, llm.ChatAudio):
                        user_contents.append(
                            {
                                "type": "input_audio",
                                "audio": base64.b64encode(
                                    utils.merge_frames(cnt.frame).data
                                ).decode("utf-8"),
                            }
                        )

                event = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": user_contents,
                    },
                }

            elif message.role == "assistant":
                assistant_contents: list[api_proto.TextContent] = []
                for cnt in message_content:
                    if isinstance(cnt, str):
                        assistant_contents.append(
                            {
                                "type": "text",
                                "text": cnt,
                            }
                        )
                    elif isinstance(cnt, llm.ChatAudio):
                        logger.warning(
                            "audio content in assistant message is not supported"
                        )

                event = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "content": assistant_contents,
                    },
                }
            elif message.role == "system":
                system_contents: list[api_proto.InputTextContent] = []
                for cnt in message_content:
                    if isinstance(cnt, str):
                        system_contents.append(
                            {
                                "type": "input_text",
                                "text": cnt,
                            }
                        )
                    elif isinstance(cnt, llm.ChatAudio):
                        logger.warning(
                            "audio content in system message is not supported"
                        )

            tool_call_id = message.tool_call_id
            if tool_call_id:
                assert isinstance(message.content, str)
                event = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": message.content,
                    },
                }

            if event is None:
                logger.warning(
                    "chat message is not supported inside the realtime API %s",
                    message,
                    extra=self._sess.logging_extra(),
                )
                return

            self._sess._queue_msg(event)

        def truncate(
            self, *, item_id: str, content_index: int, audio_end_ms: int
        ) -> None:
            self._sess._queue_msg(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": content_index,
                    "audio_end_ms": audio_end_ms,
                }
            )

        def delete(self, *, item_id: str) -> None:
            self._sess._queue_msg(
                {"type": "conversation.item.delete", "item_id": item_id}
            )

    class Conversation:
        def __init__(self, sess: RealtimeSession) -> None:
            self._sess = sess

        @property
        def item(self) -> RealtimeSession.ConvsersationItem:
            return RealtimeSession.ConvsersationItem(self._sess)

    class Response:
        def __init__(self, sess: RealtimeSession) -> None:
            self._sess = sess

        def create(self) -> None:
            self._sess._queue_msg({"type": "response.create"})

        def cancel(self) -> None:
            self._sess._queue_msg({"type": "response.cancel"})

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

        self._chat_ctx = chat_ctx
        self._fnc_ctx = fnc_ctx

        self._opts = opts
        self._send_ch = utils.aio.Chan[api_proto.ClientEvents]()
        self._http_session = http_session

        self._session_id = "not-connected"
        self.session_update()  # initial session init

    async def aclose(self) -> None:
        if self._send_ch.closed:
            return

        self._send_ch.close()
        await self._main_atask

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def fnc_ctx(self) -> llm.FunctionContext | None:
        return self._fnc_ctx

    @fnc_ctx.setter
    def fnc_ctx(self, fnc_ctx: llm.FunctionContext | None) -> None:
        self._fnc_ctx = fnc_ctx

    @property
    def default_conversation(self) -> Conversation:
        return RealtimeSession.Conversation(self)

    @property
    def input_audio_buffer(self) -> InputAudioBuffer:
        return RealtimeSession.InputAudioBuffer(self)

    @property
    def response(self) -> Response:
        return RealtimeSession.Response(self)

    def session_update(
        self,
        *,
        modalities: list[api_proto.Modality] | None = None,
        instructions: str | None = None,
        voice: api_proto.Voice | None = None,
        input_audio_format: api_proto.AudioFormat | None = None,
        output_audio_format: api_proto.AudioFormat | None = None,
        input_audio_transcription: api_proto.InputAudioTranscription | None = None,
        turn_detection: api_proto.TurnDetectionType | None = None,
        tool_choice: api_proto.ToolChoice | None = None,
        temparature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self._opts = _ModelOptions(
            modalities=modalities or self._opts.modalities,
            instructions=instructions or self._opts.instructions,
            voice=voice or self._opts.voice,
            input_audio_format=input_audio_format or self._opts.input_audio_format,
            output_audio_format=output_audio_format or self._opts.output_audio_format,
            input_audio_transcription=(
                input_audio_transcription or self._opts.input_audio_transcription
            ),
            turn_detection=turn_detection or self._opts.turn_detection,
            temparature=temparature or self._opts.temparature,
            max_output_tokens=max_output_tokens or self._opts.max_output_tokens,
            api_key=self._opts.api_key,
            base_url=self._opts.base_url,
        )

        tools = []
        if self._fnc_ctx is not None:
            for fnc in self._fnc_ctx.ai_functions.values():
                tools.append(llm._oai_api.build_oai_function_description(fnc))

        self._queue_msg(
            {
                "type": "session.update",
                "session": {
                    "modalities": self._opts.modalities,
                    "instructions": self._opts.instructions,
                    "voice": self._opts.voice,
                    "input_audio_format": self._opts.input_audio_format,
                    "output_audio_format": self._opts.output_audio_format,
                    "input_audio_transcription": self._opts.input_audio_transcription,
                    "turn_detection": self._opts.turn_detection,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "temperature": self._opts.temparature,
                    "max_output_tokens": self._opts.max_output_tokens,
                },
            }
        )

    def _queue_msg(self, msg: api_proto.ClientEvents) -> None:
        self._send_ch.send_nowait(msg)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        try:
            headers = {
                "Authorization": "Bearer " + self._opts.api_key,
                "OpenAI-Beta": "realtime=v1",
            }
            ws_conn = await self._http_session.ws_connect(
                api_proto.API_URL,
                headers=headers,
            )
        except Exception:
            logger.exception("failed to connect to OpenAI API S2S")
            return

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
                    print(data)
                    event: api_proto.ServerEventType = data["event"]

                    if event == "session.created":
                        self._handle_session_created(data)
                    elif event == "error":
                        self._handle_error(data)

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

    def _handle_session_created(
        self, session_created: api_proto.ServerEvent.SessionCreated
    ):
        self._session_id = session_created["session"]["id"]

    def _handle_error(self, error: api_proto.ServerEvent.Error):
        logger.error(
            "OpenAI S2S error %s",
            error,
            extra=self.logging_extra(),
        )

    def logging_extra(self) -> dict:
        return {"session_id": self._session_id}
