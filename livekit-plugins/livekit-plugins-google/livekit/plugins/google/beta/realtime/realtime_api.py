from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from typing import AsyncIterable, Literal

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.llm.function_context import _create_ai_function_info
from livekit.agents.multimodal import (
    _RealtimeAPI,
    _RealtimeAPISession,
)

from google import genai  # type: ignore
from google.genai.types import (  # type: ignore
    FunctionResponse,
    GenerationConfigDict,
    LiveClientToolResponse,
    LiveConnectConfigDict,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
)

from ...log import logger
from .api_proto import (
    ClientEvents,
    LiveAPIModels,
    ResponseModality,
    Voice,
    _build_tools,
)

EventTypes = Literal[
    "start_session",
    "input_speech_started",
    "response_content_added",
    "response_content_done",
    "function_calls_collected",
    "function_calls_finished",
    "function_calls_cancelled",
]


@dataclass
class GeminiContent:
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    text: str
    audio: list[rtc.AudioFrame]
    text_stream: AsyncIterable[str]
    audio_stream: AsyncIterable[rtc.AudioFrame]
    content_type: Literal["text", "audio"]


@dataclass
class Capabilities:
    supports_chat_ctx_manipulation: bool


@dataclass
class ModelOptions:
    model: LiveAPIModels | str
    api_key: str | None
    voice: Voice | str
    response_modalities: ResponseModality
    vertexai: bool
    project: str | None
    location: str | None
    candidate_count: int
    temperature: float | None
    max_output_tokens: int | None
    top_p: float | None
    top_k: int | None
    presence_penalty: float | None
    frequency_penalty: float | None
    instructions: str


class RealtimeModel(_RealtimeAPI):
    def __init__(
        self,
        *,
        instructions: str = "",
        model: LiveAPIModels | str = "gemini-2.0-flash-exp",
        api_key: str | None = None,
        voice: Voice | str = "Puck",
        modalities: ResponseModality = "AUDIO",
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        candidate_count: int = 1,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        """
        Initializes a RealtimeModel instance for interacting with Google's Realtime API.

        Args:
            instructions (str, optional): Initial system instructions for the model. Defaults to "".
            api_key (str or None, optional): OpenAI API key. If None, will attempt to read from the environment variable OPENAI_API_KEY
            modalities (ResponseModality): Modalities to use, such as ["TEXT", "AUDIO"]. Defaults to ["AUDIO"].
            model (str or None, optional): The name of the model to use. Defaults to "gemini-2.0-flash-exp".
            voice (api_proto.Voice, optional): Voice setting for audio outputs. Defaults to "Puck".
            temperature (float, optional): Sampling temperature for response generation. Defaults to 0.8.
            vertexai (bool, optional): Whether to use VertexAI for the API. Defaults to False.
                project (str or None, optional): The project to use for the API. Defaults to None. (for vertexai)
                location (str or None, optional): The location to use for the API. Defaults to None. (for vertexai)
            candidate_count (int, optional): The number of candidate responses to generate. Defaults to 1.
            top_p (float, optional): The top-p value for response generation
            top_k (int, optional): The top-k value for response generation
            presence_penalty (float, optional): The presence penalty for response generation
            frequency_penalty (float, optional): The frequency penalty for response generation
            loop (asyncio.AbstractEventLoop or None, optional): Event loop to use for async operations. If None, the current event loop is used.

        Raises:
            ValueError: If the API key is not provided and cannot be found in environment variables.
        """
        super().__init__()
        self._capabilities = Capabilities(
            supports_chat_ctx_manipulation=False,
        )
        self._model = model
        self._loop = loop or asyncio.get_event_loop()
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._vertexai = vertexai
        self._project_id = project or os.environ.get("GOOGLE_PROJECT")
        self._location = location or os.environ.get("GOOGLE_LOCATION")
        if self._api_key is None and not self._vertexai:
            raise ValueError("GOOGLE_API_KEY is not set")

        self._rt_sessions: list[GeminiRealtimeSession] = []
        self._opts = ModelOptions(
            model=model,
            api_key=api_key,
            voice=voice,
            response_modalities=modalities,
            vertexai=vertexai,
            project=project,
            location=location,
            candidate_count=candidate_count,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            instructions=instructions,
        )

    @property
    def sessions(self) -> list[GeminiRealtimeSession]:
        return self._rt_sessions

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    def session(
        self,
        *,
        chat_ctx: llm.ChatContext | None = None,
        fnc_ctx: llm.FunctionContext | None = None,
    ) -> _RealtimeAPISession:
        session = GeminiRealtimeSession(
            opts=self._opts,
            chat_ctx=chat_ctx or llm.ChatContext(),
            fnc_ctx=fnc_ctx,
            loop=self._loop,
        )
        self._rt_sessions.append(session)

        return session

    async def aclose(self) -> None:
        for session in self._rt_sessions:
            await session.aclose()


class GeminiRealtimeSession(utils.EventEmitter[EventTypes], _RealtimeAPISession):
    def __init__(
        self,
        *,
        opts: ModelOptions,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
        loop: asyncio.AbstractEventLoop,
    ):
        """
        Initializes a GeminiRealtimeSession instance for interacting with Google's Realtime API.

        Args:
            opts (ModelOptions): The model options for the session.
            chat_ctx (llm.ChatContext): The chat context for the session.
            fnc_ctx (llm.FunctionContext or None): The function context for the session.
            loop (asyncio.AbstractEventLoop): The event loop for the session.
        """
        super().__init__()
        self._loop = loop
        self._opts = opts
        self._chat_ctx = chat_ctx
        self._fnc_ctx = fnc_ctx
        self._fnc_tasks = utils.aio.TaskSet()

        tools = []
        if self._fnc_ctx is not None:
            functions = _build_tools(self._fnc_ctx)
            tools.append({"function_declarations": functions})

        self._config = LiveConnectConfigDict(
            model=self._opts.model,
            response_modalities=self._opts.response_modalities,
            generation_config=GenerationConfigDict(
                candidate_count=self._opts.candidate_count,
                temperature=self._opts.temperature,
                max_output_tokens=self._opts.max_output_tokens,
                top_p=self._opts.top_p,
                top_k=self._opts.top_k,
                presence_penalty=self._opts.presence_penalty,
                frequency_penalty=self._opts.frequency_penalty,
            ),
            system_instruction=self._opts.instructions,
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(
                        voice_name=self._opts.voice
                    )
                )
            ),
            tools=tools,
        )
        self._client = genai.Client(
            http_options={"api_version": "v1alpha"},
            api_key=self._opts.api_key,
            vertexai=self._opts.vertexai,
            project=self._opts.project,
            location=self._opts.location,
        )
        self._main_atask = asyncio.create_task(
            self._main_task(), name="gemini-realtime-session"
        )
        # dummy task to wait for the session to be initialized # TODO: sync chat ctx
        self._init_sync_task = asyncio.create_task(
            asyncio.sleep(0), name="gemini-realtime-session-init"
        )
        self._send_ch = utils.aio.Chan[ClientEvents]()
        self._active_response_id = None

    async def aclose(self) -> None:
        if self._send_ch.closed:
            return

        self._send_ch.close()
        await self._main_atask

    @property
    def fnc_ctx(self) -> llm.FunctionContext | None:
        return self._fnc_ctx

    @fnc_ctx.setter
    def fnc_ctx(self, value: llm.FunctionContext | None) -> None:
        self._fnc_ctx = value

    def _push_audio(self, frame: rtc.AudioFrame) -> None:
        data = base64.b64encode(frame.data).decode("utf-8")
        self._queue_msg({"mime_type": "audio/pcm", "data": data})

    def _queue_msg(self, msg: dict) -> None:
        self._send_ch.send_nowait(msg)

    def chat_ctx_copy(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    async def set_chat_ctx(self, ctx: llm.ChatContext) -> None:
        self._chat_ctx = ctx.copy()

    @utils.log_exceptions(logger=logger)
    async def _main_task(self):
        @utils.log_exceptions(logger=logger)
        async def _send_task():
            async for msg in self._send_ch:
                await self._session.send(msg)

            await self._session.send(".", end_of_turn=True)

        @utils.log_exceptions(logger=logger)
        async def _recv_task():
            while True:
                async for response in self._session.receive():
                    if self._active_response_id is None:
                        self._active_response_id = utils.shortuuid()
                        text_stream = utils.aio.Chan[str]()
                        audio_stream = utils.aio.Chan[rtc.AudioFrame]()
                        content = GeminiContent(
                            response_id=self._active_response_id,
                            item_id=self._active_response_id,
                            output_index=0,
                            content_index=0,
                            text="",
                            audio=[],
                            text_stream=text_stream,
                            audio_stream=audio_stream,
                            content_type=self._opts.response_modalities,
                        )
                        self.emit("response_content_added", content)

                    server_content = response.server_content
                    if server_content:
                        model_turn = server_content.model_turn
                        if model_turn:
                            for part in model_turn.parts:
                                if part.text:
                                    content.text_stream.send_nowait(part.text)
                                if part.inline_data:
                                    frame = rtc.AudioFrame(
                                        data=part.inline_data.data,
                                        sample_rate=24000,
                                        num_channels=1,
                                        samples_per_channel=len(part.inline_data.data)
                                        // 2,
                                    )
                                    content.audio_stream.send_nowait(frame)

                        if server_content.interrupted or server_content.turn_complete:
                            for stream in (content.text_stream, content.audio_stream):
                                if isinstance(stream, utils.aio.Chan):
                                    stream.close()

                            if server_content.interrupted:
                                self.emit("input_speech_started")
                            elif server_content.turn_complete:
                                self.emit("response_content_done", content)

                            self._active_response_id = None

                    if response.tool_call:
                        if self._fnc_ctx is None:
                            raise ValueError("Function context is not set")
                        fnc_calls = []
                        for fnc_call in response.tool_call.function_calls:
                            fnc_call_info = _create_ai_function_info(
                                self._fnc_ctx,
                                fnc_call.id,
                                fnc_call.name,
                                json.dumps(fnc_call.args),
                            )
                            fnc_calls.append(fnc_call_info)

                        self.emit("function_calls_collected", fnc_calls)

                        for fnc_call_info in fnc_calls:
                            self._fnc_tasks.create_task(
                                self._run_fnc_task(fnc_call_info, content.item_id)
                            )

                    # Handle function call cancellations
                    if response.tool_call_cancellation:
                        logger.warning(
                            "function call cancelled",
                            extra={
                                "function_call_ids": response.tool_call_cancellation.function_call_ids,
                            },
                        )
                        self.emit(
                            "function_calls_cancelled",
                            response.tool_call_cancellation.function_call_ids,
                        )

        async with self._client.aio.live.connect(
            model=self._opts.model, config=self._config
        ) as session:
            self._session = session
            tasks = [
                asyncio.create_task(_send_task(), name="gemini-realtime-send"),
                asyncio.create_task(_recv_task(), name="gemini-realtime-recv"),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
                await self._session.close()

    @utils.log_exceptions(logger=logger)
    async def _run_fnc_task(self, fnc_call_info: llm.FunctionCallInfo, item_id: str):
        logger.debug(
            "executing ai function",
            extra={
                "function": fnc_call_info.function_info.name,
            },
        )

        called_fnc = fnc_call_info.execute()
        try:
            await called_fnc.task
        except Exception as e:
            logger.exception(
                "error executing ai function",
                extra={
                    "function": fnc_call_info.function_info.name,
                },
                exc_info=e,
            )
        tool_call = llm.ChatMessage.create_tool_from_called_function(called_fnc)
        if tool_call.content is not None:
            tool_response = LiveClientToolResponse(
                function_responses=[
                    FunctionResponse(
                        name=tool_call.name,
                        id=tool_call.tool_call_id,
                        response={"result": tool_call.content},
                    )
                ]
            )
            await self._session.send(tool_response)

            self.emit("function_calls_finished", [called_fnc])