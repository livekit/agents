from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Literal

from google import genai
from google.genai._api_client import HttpOptions
from google.genai.types import (
    Blob,
    Content,
    FunctionResponse,
    GenerationConfig,
    LiveClientContent,
    LiveClientRealtimeInput,
    LiveClientToolResponse,
    LiveConnectConfig,
    Modality,
    Part,
    PrebuiltVoiceConfig,
    SpeechConfig,
    Tool,
    VoiceConfig,
)
from livekit import rtc
from livekit.agents import llm, utils

# from livekit.agents.llm.function_context import _create_ai_function_info
from livekit.agents.utils import images

from ...log import logger
from .api_proto import (
    ClientEvents,
    LiveAPIModels,
    Voice,
)
from .transcriber import ModelTranscriber, TranscriberSession, TranscriptionContent

EventTypes = Literal[
    "start_session",
    "input_speech_started",
    "response_content_added",
    "response_content_done",
    "function_calls_collected",
    "function_calls_finished",
    "function_calls_cancelled",
    "input_speech_transcription_completed",
    "agent_speech_transcription_completed",
    "agent_speech_stopped",
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
class InputTranscription:
    item_id: str
    transcript: str


@dataclass
class Capabilities:
    supports_truncate: bool
    input_audio_sample_rate: int | None = None


@dataclass
class ModelOptions:
    model: LiveAPIModels | str
    api_key: str | None
    voice: Voice | str
    response_modalities: list[Modality] | None
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
    instructions: Content | None
    enable_user_audio_transcription: bool
    enable_agent_audio_transcription: bool


class RealtimeModel:
    def __init__(
        self,
        *,
        instructions: str | None = None,
        model: LiveAPIModels | str = "gemini-2.0-flash-exp",
        api_key: str | None = None,
        voice: Voice | str = "Puck",
        modalities: list[Modality] = None,
        enable_user_audio_transcription: bool = True,
        enable_agent_audio_transcription: bool = True,
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

        Environment Requirements:
        - For VertexAI: Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the service account key file.
        The Google Cloud project and location can be set via `project` and `location` arguments or the environment variables
        `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`. By default, the project is inferred from the service account key file,
        and the location defaults to "us-central1".
        - For Google Gemini API: Set the `api_key` argument or the `GOOGLE_API_KEY` environment variable.

        Args:
            instructions (str, optional): Initial system instructions for the model. Defaults to "".
            api_key (str or None, optional): Google Gemini API key. If None, will attempt to read from the environment variable GOOGLE_API_KEY.
            modalities (list[Modality], optional): Modalities to use, such as ["TEXT", "AUDIO"]. Defaults to ["AUDIO"].
            model (str or None, optional): The name of the model to use. Defaults to "gemini-2.0-flash-exp".
            voice (api_proto.Voice, optional): Voice setting for audio outputs. Defaults to "Puck".
            enable_user_audio_transcription (bool, optional): Whether to enable user audio transcription. Defaults to True
            enable_agent_audio_transcription (bool, optional): Whether to enable agent audio transcription. Defaults to True
            temperature (float, optional): Sampling temperature for response generation. Defaults to 0.8.
            vertexai (bool, optional): Whether to use VertexAI for the API. Defaults to False.
                project (str or None, optional): The project id to use for the API. Defaults to None. (for vertexai)
                location (str or None, optional): The location to use for the API. Defaults to None. (for vertexai)
            candidate_count (int, optional): The number of candidate responses to generate. Defaults to 1.
            top_p (float, optional): The top-p value for response generation
            top_k (int, optional): The top-k value for response generation
            presence_penalty (float, optional): The presence penalty for response generation
            frequency_penalty (float, optional): The frequency penalty for response generation
            loop (asyncio.AbstractEventLoop or None, optional): Event loop to use for async operations. If None, the current event loop is used.

        Raises:
            ValueError: If the API key is not provided and cannot be found in environment variables.
        """  # noqa: E501
        if modalities is None:
            modalities = ["AUDIO"]
        super().__init__()
        self._capabilities = Capabilities(
            supports_truncate=False,
            input_audio_sample_rate=16000,
        )
        self._model = model
        self._loop = loop or asyncio.get_event_loop()
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
        if vertexai:
            if not self._project or not self._location:
                raise ValueError(
                    "Project and location are required for VertexAI either via project and location or GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables"  # noqa: E501
                )
            self._api_key = None  # VertexAI does not require an API key

        else:
            self._project = None
            self._location = None
            if not self._api_key:
                raise ValueError(
                    "API key is required for Google API either via api_key or GOOGLE_API_KEY environment variable"  # noqa: E501
                )

        instructions_content = Content(parts=[Part(text=instructions)]) if instructions else None

        self._rt_sessions: list[GeminiRealtimeSession] = []
        self._opts = ModelOptions(
            model=model,
            api_key=self._api_key,
            voice=voice,
            enable_user_audio_transcription=enable_user_audio_transcription,
            enable_agent_audio_transcription=enable_agent_audio_transcription,
            response_modalities=modalities,
            vertexai=vertexai,
            project=self._project,
            location=self._location,
            candidate_count=candidate_count,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            instructions=instructions_content,
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
    ) -> GeminiRealtimeSession:
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


class GeminiRealtimeSession(utils.EventEmitter[EventTypes]):
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
        self._is_interrupted = False

        tools = []
        if self._fnc_ctx is not None:
            functions = _build_tools(self._fnc_ctx)
            tools.append(Tool(function_declarations=functions))

        self._config = LiveConnectConfig(
            response_modalities=self._opts.response_modalities,
            generation_config=GenerationConfig(
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
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=self._opts.voice)
                )
            ),
            tools=tools,
        )
        self._client = genai.Client(
            http_options=HttpOptions(api_version="v1alpha"),
            api_key=self._opts.api_key,
            vertexai=self._opts.vertexai,
            project=self._opts.project,
            location=self._opts.location,
        )
        self._main_atask = asyncio.create_task(self._main_task(), name="gemini-realtime-session")
        if self._opts.enable_user_audio_transcription:
            self._transcriber = TranscriberSession(client=self._client, model=self._opts.model)
            self._transcriber.on("input_speech_done", self._on_input_speech_done)
        if self._opts.enable_agent_audio_transcription:
            self._agent_transcriber = ModelTranscriber(client=self._client, model=self._opts.model)
            self._agent_transcriber.on("input_speech_done", self._on_agent_speech_done)
        # init dummy task
        self._init_sync_task = asyncio.create_task(asyncio.sleep(0))
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

    def _push_media_chunk(self, data: bytes, mime_type: str) -> None:
        realtime_input = LiveClientRealtimeInput(
            media_chunks=[Blob(data=data, mime_type=mime_type)],
        )
        self._queue_msg(realtime_input)

    DEFAULT_ENCODE_OPTIONS = images.EncodeOptions(
        format="JPEG",
        quality=75,
        resize_options=images.ResizeOptions(width=1024, height=1024, strategy="scale_aspect_fit"),
    )

    def push_video(
        self,
        frame: rtc.VideoFrame,
        encode_options: images.EncodeOptions = DEFAULT_ENCODE_OPTIONS,
    ) -> None:
        """Push a video frame to the Gemini Multimodal Live session.

        Args:
            frame (rtc.VideoFrame): The video frame to push.
            encode_options (images.EncodeOptions, optional): The encode options for the video frame. Defaults to 1024x1024 JPEG.

        Notes:
        - This will be sent immediately so you should use a sampling frame rate that makes sense for your application and Gemini's constraints. 1 FPS is a good starting point.
        """  # noqa: E501
        encoded_data = images.encode(
            frame,
            encode_options,
        )
        mime_type = (
            "image/jpeg"
            if encode_options.format == "JPEG"
            else "image/png"
            if encode_options.format == "PNG"
            else "image/jpeg"
        )
        self._push_media_chunk(encoded_data, mime_type)

    def _push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._opts.enable_user_audio_transcription:
            self._transcriber._push_audio(frame)

        self._push_media_chunk(frame.data.tobytes(), "audio/pcm")

    def _queue_msg(self, msg: ClientEvents) -> None:
        self._send_ch.send_nowait(msg)

    def chat_ctx_copy(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    async def set_chat_ctx(self, ctx: llm.ChatContext) -> None:
        self._chat_ctx = ctx.copy()

    def cancel_response(self) -> None:
        raise NotImplementedError("cancel_response is not supported yet")

    def create_response(
        self,
        on_duplicate: Literal["cancel_existing", "cancel_new", "keep_both"] = "keep_both",
    ) -> None:
        turns, _ = _build_gemini_ctx(self._chat_ctx, id(self))
        ctx = [self._opts.instructions] + turns if self._opts.instructions else turns

        if not ctx:
            logger.warning(
                "gemini-realtime-session: No chat context to send, sending dummy content."
            )
            ctx = [Content(parts=[Part(text=".")])]

        self._queue_msg(LiveClientContent(turns=ctx, turn_complete=True))

    def commit_audio_buffer(self) -> None:
        raise NotImplementedError("commit_audio_buffer is not supported yet")

    def server_vad_enabled(self) -> bool:
        return True

    def _on_input_speech_done(self, content: TranscriptionContent) -> None:
        if content.response_id and content.text:
            self.emit(
                "input_speech_transcription_completed",
                InputTranscription(
                    item_id=content.response_id,
                    transcript=content.text,
                ),
            )

        # self._chat_ctx.append(text=content.text, role="user")
        # TODO: implement sync mechanism to make sure the transcribed user speech is inside the chat_ctx and always before the generated agent speech  # noqa: E501

    def _on_agent_speech_done(self, content: TranscriptionContent) -> None:
        if content.response_id and content.text:
            self.emit(
                "agent_speech_transcription_completed",
                InputTranscription(
                    item_id=content.response_id,
                    transcript=content.text,
                ),
            )
            # self._chat_ctx.append(text=content.text, role="assistant")

    @utils.log_exceptions(logger=logger)
    async def _main_task(self):
        @utils.log_exceptions(logger=logger)
        async def _send_task():
            async for msg in self._send_ch:
                await self._session.send(input=msg)

            await self._session.send(input=".", end_of_turn=True)

        @utils.log_exceptions(logger=logger)
        async def _recv_task():
            while True:
                async for response in self._session.receive():
                    if self._active_response_id is None:
                        self._is_interrupted = False
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
                            content_type="audio",
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
                                        samples_per_channel=len(part.inline_data.data) // 2,
                                    )
                                    if self._opts.enable_agent_audio_transcription:
                                        content.audio.append(frame)
                                    content.audio_stream.send_nowait(frame)

                        if server_content.interrupted or server_content.turn_complete:
                            if self._opts.enable_agent_audio_transcription:
                                self._agent_transcriber._push_audio(content.audio)
                            for stream in (content.text_stream, content.audio_stream):
                                if isinstance(stream, utils.aio.Chan):
                                    stream.close()

                            self.emit("agent_speech_stopped")
                            self._is_interrupted = True

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
                                "function_call_ids": response.tool_call_cancellation.function_call_ids,  # noqa: E501
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
                if self._opts.enable_user_audio_transcription:
                    await self._transcriber.aclose()
                if self._opts.enable_agent_audio_transcription:
                    await self._agent_transcriber.aclose()

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
            await self._session.send(input=tool_response)

            self.emit("function_calls_finished", [called_fnc])
