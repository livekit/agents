from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Optional

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from google import genai
from google.genai._api_client import HttpOptions
from google.genai.types import (
    Blob,
    Content,
    FunctionDeclaration,
    GenerationConfig,
    LiveClientContent,
    LiveClientRealtimeInput,
    LiveConnectConfig,
    Modality,
    Part,
    PrebuiltVoiceConfig,
    SpeechConfig,
    Tool,
    VoiceConfig,
)

from ...log import logger
from ...utils import _build_gemini_fnc, to_chat_ctx
from .api_proto import (
    ClientEvents,
    LiveAPIModels,
    Voice,
)

# from .transcriber import TranscriberSession, TranscriptionContent


INPUT_AUDIO_SAMPLE_RATE = 16000
OUTPUT_AUDIO_SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class InputTranscription:
    item_id: str
    transcript: str


@dataclass
class _RealtimeOptions:
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


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]

    messages: dict[str, _MessageGeneration]


class RealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        *,
        instructions: str | None = None,
        model: LiveAPIModels | str = "gemini-2.0-flash-exp",
        api_key: str | None = None,
        voice: Voice | str = "Puck",
        modalities: list[Modality] = [Modality.AUDIO],
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
    ) -> None:
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
            ValueError: If the API key is required but not found.
        """
        super().__init__(capabilities=llm.RealtimeCapabilities(message_truncation=False))
        self._loop = loop or asyncio.get_event_loop()
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
        if vertexai:
            if not self._project or not self._location:
                raise ValueError(
                    "Project and location are required for VertexAI either via project and location or GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables"
                )
            self._api_key = None  # VertexAI does not require an API key

        else:
            self._project = None
            self._location = None
            if not self._api_key:
                raise ValueError(
                    "API key is required for Google API either via api_key or GOOGLE_API_KEY environment variable"
                )

        instructions_content = Content(parts=[Part(text=instructions)]) if instructions else None
        self._opts = _RealtimeOptions(
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

    def session(self) -> "RealtimeSession":
        return RealtimeSession(self)

    async def aclose(self) -> None: ...


class RealtimeSession(llm.RealtimeSession):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._opts = realtime_model._opts
        self._fnc_ctx = llm.FunctionContext.empty()
        self._chat_ctx = llm.ChatContext.empty()
        self._msg_ch = utils.aio.Chan[ClientEvents]()
        self._tools: list[FunctionDeclaration] = []
        self._client = genai.Client(
            http_options=HttpOptions(api_version="v1alpha"),
            api_key=self._opts.api_key,
            vertexai=self._opts.vertexai,
            project=self._opts.project,
            location=self._opts.location,
        )
        self._main_atask = asyncio.create_task(self._main_task(), name="gemini-realtime-session")

        self._current_generation: Optional[_ResponseGeneration] = None
        # self._transcriber: Optional[TranscriberSession] = None
        # self._agent_transcriber: Optional[TranscriberSession] = None

        self._is_interrupted = False
        self._active_response_id = None
        self._session = None
        self._update_chat_ctx_lock = asyncio.Lock()
        self._update_fnc_ctx_lock = asyncio.Lock()

        # if self._opts.enable_user_audio_transcription:
        #     self._transcriber = TranscriberSession(
        #         client=self._client, model=self._opts.model
        #     )
        #     self._transcriber.on("input_speech_done", self._on_input_speech_done)

        # if self._opts.enable_agent_audio_transcription:
        #     self._agent_transcriber = TranscriberSession(
        #         client=self._client, model=self._opts.model
        #     )
        #     self._agent_transcriber.on("input_speech_done", self._on_agent_speech_done)

    async def update_instructions(self, instructions: str) -> None:
        # No-op for Gemini
        pass

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        async with self._update_chat_ctx_lock:
            turns, _ = to_chat_ctx(chat_ctx, id(self))
            self._chat_ctx = chat_ctx
            if turns:
                self._msg_ch.send_nowait(LiveClientContent(turns=turns, turn_complete=True))

    async def update_fnc_ctx(self, fnc_ctx: llm.FunctionContext | list[llm.AIFunction]) -> None:
        async with self._update_fnc_ctx_lock:
            if isinstance(fnc_ctx, list):
                fnc_ctx = llm.FunctionContext(fnc_ctx)

            retained_functions: list[llm.AIFunction] = []
            tools: list[FunctionDeclaration] = []

            for ai_fnc in fnc_ctx.ai_functions.values():
                tool_desc = _build_gemini_fnc(ai_fnc)
                tools.append(tool_desc)
                retained_functions.append(ai_fnc)

            self._fnc_ctx = llm.FunctionContext(retained_functions)
            self._tools = [Tool(function_declarations=tools)]

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def fnc_ctx(self) -> llm.FunctionContext:
        return self._fnc_ctx.copy()

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        # if self._opts.enable_user_audio_transcription and self._transcriber:
        #     self._transcriber._push_audio(frame)

        realtime_input = LiveClientRealtimeInput(
            media_chunks=[Blob(data=frame.data.tobytes(), mime_type="audio/pcm")],
        )
        self._msg_ch.send_nowait(realtime_input)

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        fut = asyncio.Future()

        ctx = [Content(parts=[Part(text=instructions or ".")])]
        self._msg_ch.send_nowait(LiveClientContent(turns=ctx, turn_complete=True))

        return fut

    def interrupt(self) -> None:
        logger.warning("interrupt() - no direct cancellation in Gemini")
        self._is_interrupted = True

    def truncate(self, *, message_id: str, audio_end_ms: int) -> None:
        logger.warning(f"truncate(...) called for {message_id}, ignoring for Gemini")

    async def aclose(self) -> None:
        self._msg_ch.close()
        if self._session:
            await self._session.close()
        if self._transcriber:
            await self._transcriber.aclose()
        if self._agent_transcriber:
            await self._agent_transcriber.aclose()
        if self._main_atask:
            await utils.aio.cancel_and_wait(self._main_atask)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self):
        print("self tools", self._tools)
        config = LiveConnectConfig(
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
            tools=self._tools,
        )
        print(config)

        async with self._client.aio.live.connect(model=self._opts.model, config=config) as session:
            self._session = session

            @utils.log_exceptions(logger=logger)
            async def _send_task():
                async for msg in self._msg_ch:
                    await session.send(input=msg)

                await session.send(input=".", end_of_turn=True)

            @utils.log_exceptions(logger=logger)
            async def _recv_task():
                while True:
                    async for response in session.receive():
                        if self._active_response_id is None:
                            self._start_new_generation()

                        if response.server_content:
                            self._handle_server_content(response.server_content)

                        if response.tool_call:
                            self._handle_tool_calls(response.tool_call)

                        if response.tool_call_cancellation:
                            self._handle_tool_call_cancellation(response.tool_call_cancellation)

            send_task = asyncio.create_task(_send_task(), name="gemini-realtime-send")
            recv_task = asyncio.create_task(_recv_task(), name="gemini-realtime-recv")
            try:
                await asyncio.gather(send_task, recv_task)
            finally:
                await utils.aio.cancel_and_wait(send_task, recv_task)

    def _start_new_generation(self):
        self._is_interrupted = False
        self._active_response_id = utils.shortuuid("gemini-turn-")
        self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            messages={},
        )

        # We'll assume each chunk belongs to a single message ID self._active_response_id
        item_generation = _MessageGeneration(
            message_id=self._active_response_id,
            text_ch=utils.aio.Chan[str](),
            audio_ch=utils.aio.Chan[rtc.AudioFrame](),
        )

        self._current_generation.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=self._active_response_id,
                text_stream=item_generation.text_ch,
                audio_stream=item_generation.audio_ch,
            )
        )

        generation_event = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
        )
        self.emit("generation_created", generation_event)

        self._current_generation.messages[self._active_response_id] = item_generation

    def _handle_server_content(self, server_content):
        if not self._current_generation or not self._active_response_id:
            logger.warning(
                "gemini-realtime-session: No active response ID, skipping server content"
            )
            return

        item_generation = self._current_generation.messages[self._active_response_id]

        model_turn = server_content.model_turn
        if model_turn:
            for part in model_turn.parts:
                if part.text:
                    item_generation.text_ch.send_nowait(part.text)
                if part.inline_data:
                    frame_data = part.inline_data.data
                    frame = rtc.AudioFrame(
                        data=frame_data,
                        sample_rate=OUTPUT_AUDIO_SAMPLE_RATE,
                        num_channels=NUM_CHANNELS,
                        samples_per_channel=len(frame_data) // 2,
                    )
                    # if self._opts.enable_agent_audio_transcription:
                    #     self._agent_transcriber._push_audio(frame)
                    item_generation.audio_ch.send_nowait(frame)

        if server_content.interrupted or server_content.turn_complete:
            self._finalize_response(item_generation)

    def _finalize_response(self, item_generation: _MessageGeneration):
        item_generation.text_ch.close()
        item_generation.audio_ch.close()

        if self._current_generation:
            self._current_generation.message_ch.close()
            self._current_generation.function_ch.close()
            self._current_generation = None

        self._is_interrupted = True
        self._active_response_id = None
        self.emit("agent_speech_stopped")

    def _handle_tool_calls(self, tool_call):
        if not self._current_generation:
            return
        for fnc_call in tool_call.function_calls:
            print("fnc_call", fnc_call)
            self._current_generation.function_ch.send_nowait(
                llm.FunctionCall(
                    call_id=fnc_call.id,
                    name=fnc_call.name,
                    arguments=json.dumps(fnc_call.args),
                )
            )

    def _handle_tool_call_cancellation(self, tool_call_cancellation):
        logger.warning(
            "function call cancelled",
            extra={
                "function_call_ids": tool_call_cancellation.ids,
            },
        )
        self.emit("function_calls_cancelled", tool_call_cancellation.ids)

    def commit_audio_buffer(self) -> None:
        raise NotImplementedError("commit_audio_buffer is not supported yet")

    def server_vad_enabled(self) -> bool:
        return True

    # def _on_input_speech_done(self, content: TranscriptionContent) -> None:
    #     if content.response_id and content.text:
    #         self.emit(
    #             "input_speech_transcription_completed",
    #             multimodal.InputTranscriptionCompleted(
    #                 item_id=content.response_id,
    #                 transcript=content.text,
    #             ),
    #         )
    #         # self._chat_ctx.append(text=content.text, role="user")
    #         # TODO: implement sync mechanism to make sure the transcribed user speech is inside the chat_ctx and always before the generated agent speech

    # def _on_agent_speech_done(self, content: TranscriptionContent) -> None:
    #     if not self._is_interrupted and content.response_id and content.text:
    #         self.emit(
    #             "agent_speech_transcription_completed",
    #             multimodal.InputTranscriptionCompleted(
    #                 item_id=content.response_id,
    #                 transcript=content.text,
    #             ),
    #         )
    #         # self._chat_ctx.append(text=content.text, role="assistant")
