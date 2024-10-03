from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from typing import AsyncIterable, Literal
from urllib.parse import urlencode, urljoin

import aiohttp
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.llm import _oai_api
from typing_extensions import TypedDict

from . import api_proto
from .log import logger

EventTypes = Literal[
    "start_session",
    "error",
    "input_speech_started",
    "input_speech_stopped",
    "input_speech_committed",
    "input_speech_transcription_completed",
    "input_speech_transcription_failed",
    "response_created",
    "response_output_added",  # message & assistant
    "response_content_added",  # message type (audio/text)
    "response_content_done",
    "response_output_done",
    "response_done",
]


@dataclass
class InputTranscriptionCompleted:
    item_id: str
    """id of the item"""
    transcript: str
    """transcript of the input audio"""


@dataclass
class InputTranscriptionFailed:
    item_id: str
    """id of the item"""
    message: str
    """error message"""


@dataclass
class RealtimeResponse:
    id: str
    """id of the message"""
    status: api_proto.ResponseStatus
    """status of the response"""
    status_details: api_proto.ResponseStatusDetails | None
    """details of the status (only with "incomplete, cancelled and failed")"""
    output: list[RealtimeOutput]
    """list of outputs"""
    done_fut: asyncio.Future[None]
    """future that will be set when the response is completed"""


@dataclass
class RealtimeOutput:
    response_id: str
    """id of the response"""
    item_id: str
    """id of the item"""
    output_index: int
    """index of the output"""
    role: api_proto.Role
    """role of the message"""
    type: Literal["message", "function_call"]
    """type of the output"""
    content: list[RealtimeContent]
    """list of content"""
    done_fut: asyncio.Future[None]
    """future that will be set when the output is completed"""


@dataclass
class RealtimeToolCall:
    name: str
    """name of the function"""
    arguments: str
    """accumulated arguments"""
    tool_call_id: str
    """id of the tool call"""


# TODO(theomonnom): add the content type directly inside RealtimeContent?
# text/audio/transcript?
@dataclass
class RealtimeContent:
    response_id: str
    """id of the response"""
    item_id: str
    """id of the item"""
    output_index: int
    """index of the output"""
    content_index: int
    """index of the content"""
    text: str
    """accumulated text content"""
    audio: list[rtc.AudioFrame]
    """accumulated audio content"""
    text_stream: AsyncIterable[str]
    """stream of text content"""
    audio_stream: AsyncIterable[rtc.AudioFrame]
    """stream of audio content"""
    tool_calls: list[RealtimeToolCall]
    """pending tool calls"""


@dataclass
class ServerVadOptions:
    threshold: float
    prefix_padding_ms: int
    silence_duration_ms: int


@dataclass
class InputTranscriptionOptions:
    model: api_proto.InputTranscriptionModel | str


@dataclass
class _ModelOptions:
    model: str
    modalities: list[api_proto.Modality]
    instructions: str
    voice: api_proto.Voice
    input_audio_format: api_proto.AudioFormat
    output_audio_format: api_proto.AudioFormat
    input_audio_transcription: InputTranscriptionOptions
    turn_detection: ServerVadOptions
    tool_choice: api_proto.ToolChoice
    temperature: float
    max_response_output_tokens: int | Literal["inf"]
    api_key: str
    base_url: str


class _ContentPtr(TypedDict):
    response_id: str
    output_index: int
    content_index: int


DEFAULT_SERVER_VAD_OPTIONS = ServerVadOptions(
    threshold=0.5,
    prefix_padding_ms=300,
    silence_duration_ms=500,
)
DEFAULT_INPUT_AUDIO_TRANSCRIPTION = InputTranscriptionOptions(model="whisper-1")


class RealtimeModel:
    def __init__(
        self,
        *,
        instructions: str = "",
        modalities: list[api_proto.Modality] = ["text", "audio"],
        model: str = "gpt-4o-realtime-preview-2024-10-01",
        voice: api_proto.Voice = "alloy",
        input_audio_format: api_proto.AudioFormat = "pcm16",
        output_audio_format: api_proto.AudioFormat = "pcm16",
        input_audio_transcription: InputTranscriptionOptions = DEFAULT_INPUT_AUDIO_TRANSCRIPTION,
        turn_detection: ServerVadOptions = DEFAULT_SERVER_VAD_OPTIONS,
        tool_choice: api_proto.ToolChoice = "auto",
        temperature: float = 0.8,
        max_response_output_tokens: int | Literal["inf"] = "inf",
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
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
            model=model,
            modalities=modalities,
            instructions=instructions,
            voice=voice,
            input_audio_format=input_audio_format,
            output_audio_format=output_audio_format,
            input_audio_transcription=input_audio_transcription,
            turn_detection=turn_detection,
            temperature=temperature,
            tool_choice=tool_choice,
            max_response_output_tokens=max_response_output_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        self._loop = loop or asyncio.get_event_loop()
        self._rt_sessions: list[RealtimeSession] = []
        self._http_session = http_session

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
        tool_choice: api_proto.ToolChoice | None = None,
        input_audio_transcription: InputTranscriptionOptions | None = None,
        turn_detection: ServerVadOptions | None = None,
        temperature: float | None = None,
        max_response_output_tokens: int | Literal["inf"] | None = None,
    ) -> RealtimeSession:
        opts = _ModelOptions(
            model=self._default_opts.model,
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
            tool_choice=tool_choice or self._default_opts.tool_choice,
            turn_detection=turn_detection or self._default_opts.turn_detection,
            temperature=temperature or self._default_opts.temperature,
            max_response_output_tokens=max_response_output_tokens
            or self._default_opts.max_response_output_tokens,
            api_key=self._default_opts.api_key,
            base_url=self._default_opts.base_url,
        )

        new_session = RealtimeSession(
            chat_ctx=chat_ctx or llm.ChatContext(),
            fnc_ctx=fnc_ctx,
            opts=opts,
            http_session=self._ensure_session(),
            loop=self._loop,
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

    class ConversationItem:
        def __init__(self, sess: RealtimeSession) -> None:
            self._sess = sess

        def create(
            self, message: llm.ChatMessage, previous_item_id: str | None = None
        ) -> None:
            message_content = message.content
            if message_content is None:
                return

            tool_call_id = message.tool_call_id
            if tool_call_id:
                assert isinstance(message_content, str)
                event = {
                    "type": "conversation.item.create",
                    "previous_item_id": previous_item_id,
                    "item": {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": message_content,
                    },
                }
            else:
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
                        "previous_item_id": previous_item_id,
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
                        "previous_item_id": previous_item_id,
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
                {
                    "type": "conversation.item.delete",
                    "item_id": item_id,
                }
            )

    class Conversation:
        def __init__(self, sess: RealtimeSession) -> None:
            self._sess = sess

        @property
        def item(self) -> RealtimeSession.ConversationItem:
            return RealtimeSession.ConversationItem(self._sess)

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
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._main_atask = asyncio.create_task(
            self._main_task(), name="openai-realtime-session"
        )

        self._chat_ctx = chat_ctx
        self._fnc_ctx = fnc_ctx
        self._loop = loop

        self._opts = opts
        self._send_ch = utils.aio.Chan[api_proto.ClientEvents]()
        self._http_session = http_session

        self._pending_responses: dict[str, RealtimeResponse] = {}

        self._session_id = "not-connected"
        self.session_update()  # initial session init

        self._fnc_tasks = utils.aio.TaskSet()

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
    def conversation(self) -> Conversation:
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
        input_audio_transcription: InputTranscriptionOptions | None = None,
        turn_detection: ServerVadOptions | None = None,
        tool_choice: api_proto.ToolChoice | None = None,
        temperature: float | None = None,
        max_response_output_tokens: int | Literal["inf"] | None = None,
    ) -> None:
        self._opts = _ModelOptions(
            model=self._opts.model,
            modalities=modalities or self._opts.modalities,
            instructions=instructions or self._opts.instructions,
            voice=voice or self._opts.voice,
            input_audio_format=input_audio_format or self._opts.input_audio_format,
            output_audio_format=output_audio_format or self._opts.output_audio_format,
            input_audio_transcription=(
                input_audio_transcription or self._opts.input_audio_transcription
            ),
            tool_choice=tool_choice or self._opts.tool_choice,
            turn_detection=turn_detection or self._opts.turn_detection,
            temperature=temperature or self._opts.temperature,
            max_response_output_tokens=max_response_output_tokens
            or self._opts.max_response_output_tokens,
            api_key=self._opts.api_key,
            base_url=self._opts.base_url,
        )

        tools = []
        if self._fnc_ctx is not None:
            for fnc in self._fnc_ctx.ai_functions.values():
                # the realtime API is using internally-tagged polymorphism.
                # build_oai_function_description was built for the ChatCompletion API
                function_data = llm._oai_api.build_oai_function_description(fnc)[
                    "function"
                ]
                function_data["type"] = "function"
                tools.append(function_data)

        server_vad_opts: api_proto.ServerVad = {
            "type": "server_vad",
            "threshold": self._opts.turn_detection.threshold,
            "prefix_padding_ms": self._opts.turn_detection.prefix_padding_ms,
            "silence_duration_ms": self._opts.turn_detection.silence_duration_ms,
        }

        self._queue_msg(
            {
                "type": "session.update",
                "session": {
                    "modalities": self._opts.modalities,
                    "instructions": self._opts.instructions,
                    "voice": self._opts.voice,
                    "input_audio_format": self._opts.input_audio_format,
                    "output_audio_format": self._opts.output_audio_format,
                    "input_audio_transcription": {
                        "model": self._opts.input_audio_transcription.model,
                    },
                    "turn_detection": server_vad_opts,
                    "tools": tools,
                    "tool_choice": self._opts.tool_choice,
                    "temperature": self._opts.temperature,
                    "max_response_output_tokens": self._opts.max_response_output_tokens,
                },
            }
        )

    def _queue_msg(self, msg: api_proto.ClientEvents) -> None:
        self._send_ch.send_nowait(msg)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        try:
            params = {"model": self._opts.model}
            base_url = self._opts.base_url.rstrip("/")
            url = urljoin(base_url + "/", "realtime") + f"?{urlencode(params)}"
            headers = {
                "Authorization": "Bearer " + self._opts.api_key,
                "OpenAI-Beta": "realtime=v1",
            }

            ws_conn = await self._http_session.ws_connect(
                url,
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
                    event: api_proto.ServerEventType = data["type"]

                    if event == "session.created":
                        self._handle_session_created(data)
                    elif event == "error":
                        self._handle_error(data)
                    elif event == "input_audio_buffer.speech_started":
                        self._handle_input_audio_buffer_speech_started(data)
                    elif event == "input_audio_buffer.speech_stopped":
                        self._handle_input_audio_buffer_speech_stopped(data)
                    elif event == "input_audio_buffer.committed":
                        self._handle_input_audio_buffer_speech_committed(data)
                    elif (
                        event == "conversation.item.input_audio_transcription.completed"
                    ):
                        self._handle_conversation_item_input_audio_transcription_completed(
                            data
                        )
                    elif event == "conversation.item.input_audio_transcription.failed":
                        self._handle_conversation_item_input_audio_transcription_failed(
                            data
                        )
                    elif event == "response.created":
                        self._handle_response_created(data)
                    elif event == "response.output_item.added":
                        self._handle_response_output_item_added(data)
                    elif event == "response.content_part.added":
                        self._handle_response_content_part_added(data)
                    elif event == "response.audio.delta":
                        self._handle_response_audio_delta(data)
                    elif event == "response.audio_transcript.delta":
                        self._handle_response_audio_transcript_delta(data)
                    elif event == "response.audio.done":
                        self._handle_response_audio_done(data)
                    elif event == "response.audio_transcript.done":
                        self._handle_response_audio_transcript_done(data)
                    elif event == "response.content_part.done":
                        self._handle_response_content_part_done(data)
                    elif event == "response.output_item.done":
                        self._handle_response_output_item_done(data)
                    elif event == "response.done":
                        self._handle_response_done(data)

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

    def _handle_input_audio_buffer_speech_started(
        self, speech_started: api_proto.ServerEvent.InputAudioBufferSpeechStarted
    ):
        self.emit("input_speech_started")

    def _handle_input_audio_buffer_speech_stopped(
        self, speech_stopped: api_proto.ServerEvent.InputAudioBufferSpeechStopped
    ):
        self.emit("input_speech_stopped")

    def _handle_input_audio_buffer_speech_committed(
        self, speech_committed: api_proto.ServerEvent.InputAudioBufferCommitted
    ):
        self.emit("input_speech_committed")

    def _handle_conversation_item_input_audio_transcription_completed(
        self,
        transcription_completed: api_proto.ServerEvent.ConversationItemInputAudioTranscriptionCompleted,
    ):
        transcript = transcription_completed["transcript"]
        self.emit(
            "input_speech_transcription_completed",
            InputTranscriptionCompleted(
                item_id=transcription_completed["item_id"],
                transcript=transcript,
            ),
        )

    def _handle_conversation_item_input_audio_transcription_failed(
        self,
        transcription_failed: api_proto.ServerEvent.ConversationItemInputAudioTranscriptionFailed,
    ):
        error = transcription_failed["error"]
        logger.error(
            "OAI S2S failed to transcribe input audio: %s",
            error["message"],
            extra=self.logging_extra(),
        )
        self.emit(
            "input_speech_transcription_failed",
            InputTranscriptionFailed(
                item_id=transcription_failed["item_id"],
                message=error["message"],
            ),
        )

    def _handle_response_created(
        self, response_created: api_proto.ServerEvent.ResponseCreated
    ):
        response = response_created["response"]
        done_fut = self._loop.create_future()
        status_details = response.get("status_details")
        new_response = RealtimeResponse(
            id=response["id"],
            status=response["status"],
            status_details=status_details,
            output=[],
            done_fut=done_fut,
        )
        self._pending_responses[new_response.id] = new_response
        self.emit("response_created", new_response)

    def _handle_response_output_item_added(
        self, response_output_added: api_proto.ServerEvent.ResponseOutputItemAdded
    ):
        response_id = response_output_added["response_id"]
        response = self._pending_responses[response_id]
        done_fut = self._loop.create_future()
        item_data = response_output_added["item"]

        assert item_data["type"] in ("message", "function_call")

        new_output = RealtimeOutput(
            response_id=response_id,
            item_id=item_data["id"],
            output_index=response_output_added["output_index"],
            type=item_data["type"],
            role=item_data.get(
                "role", "assistant"
            ),  # function_call doesn't have a role field, defaulting it to assistant
            content=[],
            done_fut=done_fut,
        )
        response.output.append(new_output)
        self.emit("response_output_added", new_output)

    def _handle_response_content_part_added(
        self, response_content_added: api_proto.ServerEvent.ResponseContentPartAdded
    ):
        response_id = response_content_added["response_id"]
        response = self._pending_responses[response_id]
        output_index = response_content_added["output_index"]
        output = response.output[output_index]

        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()

        new_content = RealtimeContent(
            response_id=response_id,
            item_id=response_content_added["item_id"],
            output_index=output_index,
            content_index=response_content_added["content_index"],
            text="",
            audio=[],
            text_stream=text_ch,
            audio_stream=audio_ch,
            tool_calls=[],
        )
        output.content.append(new_content)
        self.emit("response_content_added", new_content)

    def _handle_response_audio_delta(
        self, response_audio_delta: api_proto.ServerEvent.ResponseAudioDelta
    ):
        content = self._get_content(response_audio_delta)
        data = base64.b64decode(response_audio_delta["delta"])
        audio = rtc.AudioFrame(
            data=data,
            sample_rate=api_proto.SAMPLE_RATE,
            num_channels=api_proto.NUM_CHANNELS,
            samples_per_channel=len(data) // 2,
        )
        content.audio.append(audio)

        assert isinstance(content.audio_stream, utils.aio.Chan)
        content.audio_stream.send_nowait(audio)

    def _handle_response_audio_transcript_delta(
        self,
        response_audio_transcript_delta: api_proto.ServerEvent.ResponseAudioTranscriptDelta,
    ):
        content = self._get_content(response_audio_transcript_delta)
        transcript = response_audio_transcript_delta["delta"]
        content.text += transcript

        assert isinstance(content.text_stream, utils.aio.Chan)
        content.text_stream.send_nowait(transcript)

    def _handle_response_audio_done(
        self, response_audio_done: api_proto.ServerEvent.ResponseAudioDone
    ):
        content = self._get_content(response_audio_done)
        assert isinstance(content.audio_stream, utils.aio.Chan)
        content.audio_stream.close()

    def _handle_response_audio_transcript_done(
        self,
        response_audio_transcript_done: api_proto.ServerEvent.ResponseAudioTranscriptDone,
    ):
        content = self._get_content(response_audio_transcript_done)
        assert isinstance(content.text_stream, utils.aio.Chan)
        content.text_stream.close()

    def _handle_response_content_part_done(
        self, response_content_done: api_proto.ServerEvent.ResponseContentPartDone
    ):
        content = self._get_content(response_content_done)
        self.emit("response_content_done", content)

    def _handle_response_output_item_done(
        self, response_output_done: api_proto.ServerEvent.ResponseOutputItemDone
    ):
        response_id = response_output_done["response_id"]
        response = self._pending_responses[response_id]
        output_index = response_output_done["output_index"]
        output = response.output[output_index]

        if output.type == "function_call":
            if self._fnc_ctx is None:
                logger.error(
                    "function call received but no fnc_ctx is available",
                    extra=self.logging_extra(),
                )
                return

            # parse the arguments and call the function inside the fnc_ctx
            item = response_output_done["item"]
            assert item["type"] == "function_call"

            fnc_call_info = _oai_api.create_ai_function_info(
                self._fnc_ctx,
                item["call_id"],
                item["name"],
                item["arguments"],
            )

            self._fnc_tasks.create_task(
                self._run_fnc_task(fnc_call_info, output.item_id)
            )

        output.done_fut.set_result(None)
        self.emit("response_output_done", output)

    def _handle_response_done(self, response_done: api_proto.ServerEvent.ResponseDone):
        response_data = response_done["response"]
        response_id = response_data["id"]
        response = self._pending_responses[response_id]
        response.done_fut.set_result(None)

        response.status = response_data["status"]
        response.status_details = response_data.get("status_details")

        if response.status == "failed":
            assert response.status_details is not None

            error = response.status_details.get("error")
            code: str | None = None
            message: str | None = None
            if error is not None:
                code = error.get("code")
                message = error.get("message")

            logger.error(
                "response generation failed",
                extra={"code": code, "error": message, **self.logging_extra()},
            )
        elif response.status == "incomplete":
            assert response.status_details is not None
            reason = response.status_details.get("reason")

            logger.warning(
                "response generation incomplete",
                extra={"reason": reason, **self.logging_extra()},
            )

        self.emit("response_done", response)

    def _get_content(self, ptr: _ContentPtr) -> RealtimeContent:
        response = self._pending_responses[ptr["response_id"]]
        output = response.output[ptr["output_index"]]
        content = output.content[ptr["content_index"]]
        return content

    @utils.log_exceptions(logger=logger)
    async def _run_fnc_task(self, fnc_call_info: llm.FunctionCallInfo, item_id: str):
        logger.debug(
            "executing ai function",
            extra={
                "function": fnc_call_info.function_info.name,
            },
        )

        called_fnc = fnc_call_info.execute()
        await called_fnc.task

        tool_call = llm.ChatMessage.create_tool_from_called_function(called_fnc)

        if called_fnc.result is not None:
            self.conversation.item.create(tool_call, item_id)
            self.response.create()

    def logging_extra(self) -> dict:
        return {"session_id": self._session_id}
