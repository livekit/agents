from __future__ import annotations

import asyncio
import base64
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import AsyncIterable, Literal, Union, cast, overload
from urllib.parse import urlencode

import aiohttp
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.llm import _oai_api
from typing_extensions import TypedDict

from . import api_proto, remote_items
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
    "function_calls_collected",
    "function_calls_finished",
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
    usage: api_proto.Usage | None
    """usage of the response"""
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
    content_type: api_proto.Modality
    """type of the content"""


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
    model: str | None
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
    api_key: str | None
    base_url: str
    entra_token: str | None
    azure_deployment: str | None
    is_azure: bool
    api_version: str | None


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
    @overload
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
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        azure_deployment: str | None = None,
        entra_token: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        base_url: str | None = None,
        instructions: str = "",
        modalities: list[api_proto.Modality] = ["text", "audio"],
        voice: api_proto.Voice = "alloy",
        input_audio_format: api_proto.AudioFormat = "pcm16",
        output_audio_format: api_proto.AudioFormat = "pcm16",
        input_audio_transcription: InputTranscriptionOptions = DEFAULT_INPUT_AUDIO_TRANSCRIPTION,
        turn_detection: ServerVadOptions = DEFAULT_SERVER_VAD_OPTIONS,
        tool_choice: api_proto.ToolChoice = "auto",
        temperature: float = 0.8,
        max_response_output_tokens: int | Literal["inf"] = "inf",
        http_session: aiohttp.ClientSession | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        instructions: str = "",
        modalities: list[api_proto.Modality] = ["text", "audio"],
        model: str | None = "gpt-4o-realtime-preview-2024-10-01",
        voice: api_proto.Voice = "alloy",
        input_audio_format: api_proto.AudioFormat = "pcm16",
        output_audio_format: api_proto.AudioFormat = "pcm16",
        input_audio_transcription: InputTranscriptionOptions = DEFAULT_INPUT_AUDIO_TRANSCRIPTION,
        turn_detection: ServerVadOptions = DEFAULT_SERVER_VAD_OPTIONS,
        tool_choice: api_proto.ToolChoice = "auto",
        temperature: float = 0.8,
        max_response_output_tokens: int | Literal["inf"] = "inf",
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        # azure specific parameters
        azure_deployment: str | None = None,
        entra_token: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
    ) -> None:
        """
        Initializes a RealtimeClient instance for interacting with OpenAI's Realtime API.

        Args:
            instructions (str, optional): Initial system instructions for the model. Defaults to "".
            api_key (str or None, optional): OpenAI API key. If None, will attempt to read from the environment variable OPENAI_API_KEY
            modalities (list[api_proto.Modality], optional): Modalities to use, such as ["text", "audio"]. Defaults to ["text", "audio"].
            model (str or None, optional): The name of the model to use. Defaults to "gpt-4o-realtime-preview-2024-10-01".
            voice (api_proto.Voice, optional): Voice setting for audio outputs. Defaults to "alloy".
            input_audio_format (api_proto.AudioFormat, optional): Format of input audio data. Defaults to "pcm16".
            output_audio_format (api_proto.AudioFormat, optional): Format of output audio data. Defaults to "pcm16".
            input_audio_transcription (InputTranscriptionOptions, optional): Options for transcribing input audio. Defaults to DEFAULT_INPUT_AUDIO_TRANSCRIPTION.
            turn_detection (ServerVadOptions, optional): Options for server-based voice activity detection (VAD). Defaults to DEFAULT_SERVER_VAD_OPTIONS.
            tool_choice (api_proto.ToolChoice, optional): Tool choice for the model, such as "auto". Defaults to "auto".
            temperature (float, optional): Sampling temperature for response generation. Defaults to 0.8.
            max_response_output_tokens (int or Literal["inf"], optional): Maximum number of tokens in the response. Defaults to "inf".
            base_url (str or None, optional): Base URL for the API endpoint. If None, defaults to OpenAI's default API URL.
            http_session (aiohttp.ClientSession or None, optional): Async HTTP session to use for requests. If None, a new session will be created.
            loop (asyncio.AbstractEventLoop or None, optional): Event loop to use for async operations. If None, the current event loop is used.

        Raises:
            ValueError: If the API key is not provided and cannot be found in environment variables.
        """
        super().__init__()
        self._base_url = base_url

        is_azure = (
            api_version is not None
            or entra_token is not None
            or azure_deployment is not None
        )

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if api_key is None and not is_azure:
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
            azure_deployment=azure_deployment,
            entra_token=entra_token,
            is_azure=is_azure,
            api_version=api_version,
        )

        self._loop = loop or asyncio.get_event_loop()
        self._rt_sessions: list[RealtimeSession] = []
        self._http_session = http_session

    @classmethod
    def with_azure(
        cls,
        *,
        azure_deployment: str,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        entra_token: str | None = None,
        base_url: str | None = None,
        instructions: str = "",
        modalities: list[api_proto.Modality] = ["text", "audio"],
        voice: api_proto.Voice = "alloy",
        input_audio_format: api_proto.AudioFormat = "pcm16",
        output_audio_format: api_proto.AudioFormat = "pcm16",
        input_audio_transcription: InputTranscriptionOptions = DEFAULT_INPUT_AUDIO_TRANSCRIPTION,
        turn_detection: ServerVadOptions = DEFAULT_SERVER_VAD_OPTIONS,
        tool_choice: api_proto.ToolChoice = "auto",
        temperature: float = 0.8,
        max_response_output_tokens: int | Literal["inf"] = "inf",
        http_session: aiohttp.ClientSession | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        """
        Create a RealtimeClient instance configured for Azure OpenAI Service.

        Args:
            azure_deployment (str): The name of your Azure OpenAI deployment.
            azure_endpoint (str or None, optional): The endpoint URL for your Azure OpenAI resource. If None, will attempt to read from the environment variable AZURE_OPENAI_ENDPOINT.
            api_version (str or None, optional): API version to use with Azure OpenAI Service. If None, will attempt to read from the environment variable OPENAI_API_VERSION.
            api_key (str or None, optional): Azure OpenAI API key. If None, will attempt to read from the environment variable AZURE_OPENAI_API_KEY.
            entra_token (str or None, optional): Azure Entra authentication token. Required if not using API key authentication.
            base_url (str or None, optional): Base URL for the API endpoint. If None, constructed from the azure_endpoint.
            instructions (str, optional): Initial system instructions for the model. Defaults to "".
            modalities (list[api_proto.Modality], optional): Modalities to use, such as ["text", "audio"]. Defaults to ["text", "audio"].
            voice (api_proto.Voice, optional): Voice setting for audio outputs. Defaults to "alloy".
            input_audio_format (api_proto.AudioFormat, optional): Format of input audio data. Defaults to "pcm16".
            output_audio_format (api_proto.AudioFormat, optional): Format of output audio data. Defaults to "pcm16".
            input_audio_transcription (InputTranscriptionOptions, optional): Options for transcribing input audio. Defaults to DEFAULT_INPUT_AUDIO_TRANSCRIPTION.
            turn_detection (ServerVadOptions, optional): Options for server-based voice activity detection (VAD). Defaults to DEFAULT_SERVER_VAD_OPTIONS.
            tool_choice (api_proto.ToolChoice, optional): Tool choice for the model, such as "auto". Defaults to "auto".
            temperature (float, optional): Sampling temperature for response generation. Defaults to 0.8.
            max_response_output_tokens (int or Literal["inf"], optional): Maximum number of tokens in the response. Defaults to "inf".
            http_session (aiohttp.ClientSession or None, optional): Async HTTP session to use for requests. If None, a new session will be created.
            loop (asyncio.AbstractEventLoop or None, optional): Event loop to use for async operations. If None, the current event loop is used.

        Returns:
            RealtimeClient: An instance of RealtimeClient configured for Azure OpenAI Service.

        Raises:
            ValueError: If required Azure parameters are missing or invalid.
        """
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None and entra_token is None:
            raise ValueError(
                "Missing credentials. Please pass one of `api_key`, `entra_token`, or the `AZURE_OPENAI_API_KEY` environment variable."
            )

        api_version = api_version or os.getenv("OPENAI_API_VERSION")
        if api_version is None:
            raise ValueError(
                "Must provide either the `api_version` argument or the `OPENAI_API_VERSION` environment variable"
            )

        if base_url is None:
            azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            if azure_endpoint is None:
                raise ValueError(
                    "Missing Azure endpoint. Please pass the `azure_endpoint` parameter or set the `AZURE_OPENAI_ENDPOINT` environment variable."
                )

            base_url = f"{azure_endpoint.rstrip('/')}/openai"
        elif azure_endpoint is not None:
            raise ValueError("base_url and azure_endpoint are mutually exclusive")

        return cls(
            instructions=instructions,
            modalities=modalities,
            voice=voice,
            input_audio_format=input_audio_format,
            output_audio_format=output_audio_format,
            input_audio_transcription=input_audio_transcription,
            turn_detection=turn_detection,
            tool_choice=tool_choice,
            temperature=temperature,
            max_response_output_tokens=max_response_output_tokens,
            api_key=api_key,
            http_session=http_session,
            loop=loop,
            azure_deployment=azure_deployment,
            api_version=api_version,
            entra_token=entra_token,
            base_url=base_url,
        )

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
        opts = deepcopy(self._default_opts)
        if modalities is not None:
            opts.modalities = modalities
        if instructions is not None:
            opts.instructions = instructions
        if voice is not None:
            opts.voice = voice
        if input_audio_format is not None:
            opts.input_audio_format = input_audio_format
        if output_audio_format is not None:
            opts.output_audio_format = output_audio_format
        if tool_choice is not None:
            opts.tool_choice = tool_choice
        if input_audio_transcription is not None:
            opts.input_audio_transcription
        if turn_detection is not None:
            opts.turn_detection = turn_detection
        if temperature is not None:
            opts.temperature = temperature
        if max_response_output_tokens is not None:
            opts.max_response_output_tokens = max_response_output_tokens

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
        ) -> asyncio.Future[bool]:
            fut = asyncio.Future[bool]()

            message_content = message.content
            tool_call_id = message.tool_call_id
            if not tool_call_id and message_content is None:
                # not a function call while the message content is None
                fut.set_result(False)
                return fut
            event: api_proto.ClientEvent.ConversationItemCreate | None = None
            if tool_call_id:
                if message.role == "tool":
                    # function_call_output
                    assert isinstance(message_content, str)
                    event = {
                        "type": "conversation.item.create",
                        "previous_item_id": previous_item_id,
                        "item": {
                            "id": message.id,
                            "type": "function_call_output",
                            "call_id": tool_call_id,
                            "output": message_content,
                        },
                    }
                else:
                    # function_call
                    if not message.tool_calls or message.name is None:
                        logger.warning(
                            "function call message has no name or tool calls: %s",
                            message,
                            extra=self._sess.logging_extra(),
                        )
                        fut.set_result(False)
                        return fut
                    if len(message.tool_calls) > 1:
                        logger.warning(
                            "function call message has multiple tool calls, "
                            "only the first one will be used",
                            extra=self._sess.logging_extra(),
                        )

                    event = {
                        "type": "conversation.item.create",
                        "previous_item_id": previous_item_id,
                        "item": {
                            "id": message.id,
                            "type": "function_call",
                            "call_id": tool_call_id,
                            "name": message.name,
                            "arguments": message.tool_calls[0].raw_arguments,
                        },
                    }
            else:
                if message_content is None:
                    logger.warning(
                        "message content is None, skipping: %s",
                        message,
                        extra=self._sess.logging_extra(),
                    )
                    fut.set_result(False)
                    return fut
                if not isinstance(message_content, list):
                    message_content = [message_content]

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
                            "id": message.id,
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
                            "id": message.id,
                            "type": "message",
                            "role": "assistant",
                            "content": assistant_contents,
                        },
                    }
                elif message.role == "system":
                    system_contents: list[api_proto.InputTextContent] = []
                    for cnt in message_content:
                        if isinstance(cnt, str):
                            system_contents.append({"type": "input_text", "text": cnt})
                        elif isinstance(cnt, llm.ChatAudio):
                            logger.warning(
                                "audio content in system message is not supported"
                            )

                    event = {
                        "type": "conversation.item.create",
                        "previous_item_id": previous_item_id,
                        "item": {
                            "id": message.id,
                            "type": "message",
                            "role": "system",
                            "content": system_contents,
                        },
                    }

            if event is None:
                logger.warning(
                    "chat message is not supported inside the realtime API %s",
                    message,
                    extra=self._sess.logging_extra(),
                )
                fut.set_result(False)
                return fut

            self._sess._item_created_futs[message.id] = fut
            self._sess._queue_msg(event)
            return fut

        def truncate(
            self, *, item_id: str, content_index: int, audio_end_ms: int
        ) -> asyncio.Future[bool]:
            fut = asyncio.Future[bool]()
            self._sess._item_truncated_futs[item_id] = fut
            self._sess._queue_msg(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": content_index,
                    "audio_end_ms": audio_end_ms,
                }
            )
            return fut

        def delete(self, *, item_id: str) -> asyncio.Future[bool]:
            fut = asyncio.Future[bool]()
            self._sess._item_deleted_futs[item_id] = fut
            self._sess._queue_msg(
                {
                    "type": "conversation.item.delete",
                    "item_id": item_id,
                }
            )
            return fut

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
        # manage conversation items internally
        self._remote_converstation_items = remote_items._RemoteConversationItems()

        # wait for the item to be created or deleted
        self._item_created_futs: dict[str, asyncio.Future[bool]] = {}
        self._item_deleted_futs: dict[str, asyncio.Future[bool]] = {}
        self._item_truncated_futs: dict[str, asyncio.Future[bool]] = {}

        self._fnc_ctx = fnc_ctx
        self._loop = loop

        self._opts = opts
        self._send_ch = utils.aio.Chan[api_proto.ClientEvents]()
        self._http_session = http_session

        self._pending_responses: dict[str, RealtimeResponse] = {}

        self._session_id = "not-connected"
        self.session_update()  # initial session init

        # sync the chat context to the session
        self._init_sync_task = asyncio.create_task(self.set_chat_ctx(chat_ctx))

        self._fnc_tasks = utils.aio.TaskSet()

    async def aclose(self) -> None:
        if self._send_ch.closed:
            return

        self._send_ch.close()
        await self._main_atask

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
        self._opts = deepcopy(self._opts)
        if modalities is not None:
            self._opts.modalities = modalities
        if instructions is not None:
            self._opts.instructions = instructions
        if voice is not None:
            self._opts.voice = voice
        if input_audio_format is not None:
            self._opts.input_audio_format = input_audio_format
        if output_audio_format is not None:
            self._opts.output_audio_format = output_audio_format
        if input_audio_transcription is not None:
            self._opts.input_audio_transcription = input_audio_transcription
        if turn_detection is not None:
            self._opts.turn_detection = turn_detection
        if tool_choice is not None:
            self._opts.tool_choice = tool_choice
        if temperature is not None:
            self._opts.temperature = temperature
        if max_response_output_tokens is not None:
            self._opts.max_response_output_tokens = max_response_output_tokens

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

        session_data: api_proto.ClientEvent.SessionUpdateData = {
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
            "max_response_output_tokens": None,
        }

        # azure doesn't support inf for max_response_output_tokens
        if not self._opts.is_azure or isinstance(
            self._opts.max_response_output_tokens, int
        ):
            session_data["max_response_output_tokens"] = (
                self._opts.max_response_output_tokens
            )
        else:
            del session_data["max_response_output_tokens"]  # type: ignore

        self._queue_msg(
            {
                "type": "session.update",
                "session": session_data,
            }
        )

    def chat_ctx_copy(self) -> llm.ChatContext:
        return self._remote_converstation_items.to_chat_context()

    async def set_chat_ctx(self, new_ctx: llm.ChatContext) -> None:
        """Sync the chat context with the agent's chat context.

        Compute the minimum number of insertions and deletions to transform the old
        chat context messages to the new chat context messages.
        """
        original_ctx = self._remote_converstation_items.to_chat_context()

        changes = utils._compute_changes(
            original_ctx.messages, new_ctx.messages, key_fnc=lambda x: x.id
        )
        logger.debug(
            "sync chat context",
            extra={
                "to_delete": [msg.id for msg in changes.to_delete],
                "to_add": [
                    (prev.id if prev else None, msg.id) for prev, msg in changes.to_add
                ],
            },
        )

        # append an empty audio message if all new messages are text
        if changes.to_add and not any(
            isinstance(msg.content, llm.ChatAudio) for _, msg in changes.to_add
        ):
            # Patch: add an empty audio message to the chat context
            # to set the API in audio mode
            data = b"\x00\x00" * api_proto.SAMPLE_RATE
            _empty_audio = rtc.AudioFrame(
                data=data,
                sample_rate=api_proto.SAMPLE_RATE,
                num_channels=api_proto.NUM_CHANNELS,
                samples_per_channel=len(data) // 2,
            )
            changes.to_add.append(
                (
                    None,
                    llm.ChatMessage(
                        role="user", content=llm.ChatAudio(frame=_empty_audio)
                    ),
                )
            )
            logger.debug("added empty audio message to the chat context")

        _futs = [
            self.conversation.item.delete(item_id=msg.id) for msg in changes.to_delete
        ] + [
            self.conversation.item.create(msg, prev.id if prev else None)
            for prev, msg in changes.to_add
        ]

        # wait for all the futures to complete
        await asyncio.gather(*_futs)

    def _update_converstation_item_content(
        self, item_id: str, content: llm.ChatContent | list[llm.ChatContent] | None
    ) -> None:
        item = self._remote_converstation_items.get(item_id)
        if item is None:
            logger.warning(
                "conversation item not found, skipping update",
                extra={"item_id": item_id},
            )
            return
        item.content = content

    def _queue_msg(self, msg: api_proto.ClientEvents) -> None:
        self._send_ch.send_nowait(msg)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        try:
            headers = {"User-Agent": "LiveKit Agents"}
            query_params: dict[str, str] = {}

            base_url = self._opts.base_url
            if self._opts.is_azure:
                if self._opts.entra_token:
                    headers["Authorization"] = f"Bearer {self._opts.entra_token}"

                if self._opts.api_key:
                    headers["api-key"] = self._opts.api_key

                if self._opts.api_version:
                    query_params["api-version"] = self._opts.api_version

                if self._opts.azure_deployment:
                    query_params["deployment"] = self._opts.azure_deployment
            else:
                # OAI endpoint
                headers["Authorization"] = f"Bearer {self._opts.api_key}"
                headers["OpenAI-Beta"] = "realtime=v1"

                if self._opts.model:
                    query_params["model"] = self._opts.model

            url = f"{base_url.rstrip('/')}/realtime?{urlencode(query_params)}"
            if url.startswith("http"):
                url = url.replace("http", "ws", 1)

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
                    elif event == "conversation.item.created":
                        self._handle_conversation_item_created(data)
                    elif event == "conversation.item.deleted":
                        self._handle_conversation_item_deleted(data)
                    elif event == "conversation.item.truncated":
                        self._handle_conversation_item_truncated(data)
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

    def _handle_conversation_item_created(
        self, item_created: api_proto.ServerEvent.ConversationItemCreated
    ):
        previous_item_id = item_created["previous_item_id"]
        item = item_created["item"]
        item_type = item["type"]
        item_id = item["id"]

        # Create message based on item type
        # Leave the content empty and fill it in later from the content parts
        if item_type == "message":
            # Handle message items (system/user/assistant)
            item = cast(Union[api_proto.SystemItem, api_proto.UserItem], item)
            role = item["role"]
            message = llm.ChatMessage(id=item_id, role=role)
            if item.get("content"):
                content = item["content"][0]
                if content["type"] in ("text", "input_text"):
                    content = cast(api_proto.InputTextContent, content)
                    message.content = content["text"]
                elif content["type"] == "input_audio" and content.get("audio"):
                    audio_data = base64.b64decode(content["audio"])
                    message.content = llm.ChatAudio(
                        frame=rtc.AudioFrame(
                            data=audio_data,
                            sample_rate=api_proto.SAMPLE_RATE,
                            num_channels=api_proto.NUM_CHANNELS,
                            samples_per_channel=len(audio_data) // 2,
                        )
                    )

        elif item_type == "function_call":
            # Handle function call items
            item = cast(api_proto.FunctionCallItem, item)
            message = llm.ChatMessage(
                id=item_id,
                role="assistant",
                name=item["name"],
                tool_call_id=item["call_id"],
            )

        elif item_type == "function_call_output":
            # Handle function call output items
            item = cast(api_proto.FunctionCallOutputItem, item)
            message = llm.ChatMessage(
                id=item_id,
                role="tool",
                tool_call_id=item["call_id"],
                content=item["output"],
            )

        else:
            logger.error(
                f"unknown conversation item type {item_type}",
                extra=self.logging_extra(),
            )
            return

        # Insert into conversation items
        self._remote_converstation_items.insert_after(previous_item_id, message)
        if item_id in self._item_created_futs:
            self._item_created_futs[item_id].set_result(True)
            del self._item_created_futs[item_id]
        logger.debug("conversation item created", extra=item_created)

    def _handle_conversation_item_deleted(
        self, item_deleted: api_proto.ServerEvent.ConversationItemDeleted
    ):
        # Delete from conversation items
        item_id = item_deleted["item_id"]
        self._remote_converstation_items.delete(item_id)
        if item_id in self._item_deleted_futs:
            self._item_deleted_futs[item_id].set_result(True)
            del self._item_deleted_futs[item_id]
        logger.debug("conversation item deleted", extra=item_deleted)

    def _handle_conversation_item_truncated(
        self, item_truncated: api_proto.ServerEvent.ConversationItemTruncated
    ):
        item_id = item_truncated["item_id"]
        if item_id in self._item_truncated_futs:
            self._item_truncated_futs[item_id].set_result(True)
            del self._item_truncated_futs[item_id]

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
            usage=response.get("usage"),
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

        item_type: Literal["message", "function_call"] = item_data["type"]  # type: ignore
        assert item_type in ("message", "function_call")
        # function_call doesn't have a role field, defaulting it to assistant
        item_role: api_proto.Role = item_data.get("role") or "assistant"  # type: ignore

        new_output = RealtimeOutput(
            response_id=response_id,
            item_id=item_data["id"],
            output_index=response_output_added["output_index"],
            type=item_type,
            role=item_role,
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
        content_type = response_content_added["part"]["type"]

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
            content_type=content_type,
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

            msg = self._remote_converstation_items.get(output.item_id)
            if msg is not None:
                # update the content of the message
                assert msg.tool_call_id == item["call_id"]
                assert msg.role == "assistant"
                msg.name = item["name"]
                msg.tool_calls = [fnc_call_info]

            self.emit("function_calls_collected", [fnc_call_info])

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
        response.usage = response_data.get("usage")

        if response.status == "failed":
            assert response.status_details is not None

            error = response.status_details.get("error")
            code: str | None = None
            message: str | None = None
            if error is not None:
                code = error.get("code")  # type: ignore
                message = error.get("message")  # type: ignore

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
            create_fut = self.conversation.item.create(
                tool_call,
                previous_item_id=item_id,
            )
            self.response.create()
            await create_fut

        # update the message with the tool call result
        msg = self._remote_converstation_items.get(tool_call.id)
        if msg is not None:
            assert msg.tool_call_id == tool_call.tool_call_id
            assert msg.role == "tool"
            msg.name = tool_call.name
            msg.content = tool_call.content
            msg.tool_exception = tool_call.tool_exception

        self.emit("function_calls_finished", [called_fnc])

    def logging_extra(self) -> dict:
        return {"session_id": self._session_id}
