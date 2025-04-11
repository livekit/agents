from __future__ import annotations

import asyncio
import json
import os
import weakref
from dataclasses import dataclass

from google import genai
from google.genai.types import (
    AudioTranscriptionConfig,
    Blob,
    Content,
    FunctionDeclaration,
    GenerationConfig,
    LiveClientContent,
    LiveClientRealtimeInput,
    LiveConnectConfig,
    LiveServerContent,
    LiveServerGoAway,
    LiveServerToolCall,
    LiveServerToolCallCancellation,
    Modality,
    Part,
    PrebuiltVoiceConfig,
    SpeechConfig,
    Tool,
    UsageMetadata,
    VoiceConfig,
)
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import images, is_given

from ...log import logger
from ...utils import _build_gemini_fnc, get_tool_results_for_realtime, to_chat_ctx
from .api_proto import ClientEvents, LiveAPIModels, Voice

INPUT_AUDIO_SAMPLE_RATE = 16000
OUTPUT_AUDIO_SAMPLE_RATE = 24000
NUM_CHANNELS = 1

DEFAULT_ENCODE_OPTIONS = images.EncodeOptions(
    format="JPEG",
    quality=75,
    resize_options=images.ResizeOptions(width=1024, height=1024, strategy="scale_aspect_fit"),
)


@dataclass
class InputTranscription:
    item_id: str
    transcript: str


@dataclass
class _RealtimeOptions:
    model: LiveAPIModels | str
    api_key: str | None
    voice: Voice | str
    response_modalities: NotGivenOr[list[Modality]]
    vertexai: bool
    project: str | None
    location: str | None
    candidate_count: int
    temperature: NotGivenOr[float]
    max_output_tokens: NotGivenOr[int]
    top_p: NotGivenOr[float]
    top_k: NotGivenOr[int]
    presence_penalty: NotGivenOr[float]
    frequency_penalty: NotGivenOr[float]
    instructions: NotGivenOr[str]
    input_audio_transcription: AudioTranscriptionConfig | None
    output_audio_transcription: AudioTranscriptionConfig | None


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
        instructions: NotGivenOr[str] = NOT_GIVEN,
        model: LiveAPIModels | str = "gemini-2.0-flash-exp",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice: Voice | str = "Puck",
        modalities: NotGivenOr[list[Modality]] = NOT_GIVEN,
        vertexai: bool = False,
        project: NotGivenOr[str] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        candidate_count: int = 1,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        top_k: NotGivenOr[int] = NOT_GIVEN,
        presence_penalty: NotGivenOr[float] = NOT_GIVEN,
        frequency_penalty: NotGivenOr[float] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[AudioTranscriptionConfig | None] = NOT_GIVEN,
        output_audio_transcription: NotGivenOr[AudioTranscriptionConfig | None] = NOT_GIVEN,
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
            api_key (str, optional): Google Gemini API key. If None, will attempt to read from the environment variable GOOGLE_API_KEY.
            modalities (list[Modality], optional): Modalities to use, such as ["TEXT", "AUDIO"]. Defaults to ["AUDIO"].
            model (str, optional): The name of the model to use. Defaults to "gemini-2.0-flash-exp".
            voice (api_proto.Voice, optional): Voice setting for audio outputs. Defaults to "Puck".
            temperature (float, optional): Sampling temperature for response generation. Defaults to 0.8.
            vertexai (bool, optional): Whether to use VertexAI for the API. Defaults to False.
                project (str, optional): The project id to use for the API. Defaults to None. (for vertexai)
                location (str, optional): The location to use for the API. Defaults to None. (for vertexai)
            candidate_count (int, optional): The number of candidate responses to generate. Defaults to 1.
            top_p (float, optional): The top-p value for response generation
            top_k (int, optional): The top-k value for response generation
            presence_penalty (float, optional): The presence penalty for response generation
            frequency_penalty (float, optional): The frequency penalty for response generation
            input_audio_transcription (AudioTranscriptionConfig | None, optional): The configuration for input audio transcription. Defaults to None.)
            output_audio_transcription (AudioTranscriptionConfig | None, optional): The configuration for output audio transcription. Defaults to AudioTranscriptionConfig().

        Raises:
            ValueError: If the API key is required but not found.
        """  # noqa: E501
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=False,
            )
        )

        gemini_api_key = api_key if is_given(api_key) else os.environ.get("GOOGLE_API_KEY")
        gcp_project = project if is_given(project) else os.environ.get("GOOGLE_CLOUD_PROJECT")
        gcp_location = location if is_given(location) else os.environ.get("GOOGLE_CLOUD_LOCATION")
        if vertexai:
            if not gcp_project or not gcp_location:
                raise ValueError(
                    "Project and location are required for VertexAI either via project and location or GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables"  # noqa: E501
                )
            gemini_api_key = None  # VertexAI does not require an API key

        else:
            gcp_project = None
            gcp_location = None
            if not gemini_api_key:
                raise ValueError(
                    "API key is required for Google API either via api_key or GOOGLE_API_KEY environment variable"  # noqa: E501
                )

        if not is_given(input_audio_transcription):
            input_audio_transcription = None
        if not is_given(output_audio_transcription):
            output_audio_transcription = AudioTranscriptionConfig()

        self._opts = _RealtimeOptions(
            model=model,
            api_key=gemini_api_key,
            voice=voice,
            response_modalities=modalities,
            vertexai=vertexai,
            project=gcp_project,
            location=gcp_location,
            candidate_count=candidate_count,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            instructions=instructions,
            input_audio_transcription=input_audio_transcription,
            output_audio_transcription=output_audio_transcription,
        )

        self._sessions = weakref.WeakSet[RealtimeSession]()

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    def update_options(
        self, *, voice: NotGivenOr[str] = NOT_GIVEN, temperature: NotGivenOr[float] = NOT_GIVEN
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice

        if is_given(temperature):
            self._opts.temperature = temperature

        for sess in self._sessions:
            sess.update_options(voice=self._opts.voice, temperature=self._opts.temperature)

    async def aclose(self) -> None: ...


class RealtimeSession(llm.RealtimeSession):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._opts = realtime_model._opts
        self._tools = llm.ToolContext.empty()
        self._chat_ctx = llm.ChatContext.empty()
        self._msg_ch = utils.aio.Chan[ClientEvents]()
        self._gemini_tools: list[Tool] = []
        self._client = genai.Client(
            api_key=self._opts.api_key,
            vertexai=self._opts.vertexai,
            project=self._opts.project,
            location=self._opts.location,
        )
        self._main_atask = asyncio.create_task(self._main_task(), name="gemini-realtime-session")

        self._current_generation: _ResponseGeneration | None = None

        self._is_interrupted = False
        self._active_response_id = None
        self._session = None
        self._update_chat_ctx_lock = asyncio.Lock()
        self._update_fnc_ctx_lock = asyncio.Lock()
        self._response_created_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}
        self._pending_generation_event_id = None

        self._reconnect_event = asyncio.Event()
        self._session_lock = asyncio.Lock()
        self._gemini_close_task: asyncio.Task | None = None

    def _schedule_gemini_session_close(self) -> None:
        if self._session is not None:
            self._gemini_close_task = asyncio.create_task(self._close_gemini_session())

    async def _close_gemini_session(self) -> None:
        async with self._session_lock:
            if self._session:
                try:
                    await self._session.close()
                finally:
                    self._session = None

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice

        if is_given(temperature):
            self._opts.temperature = temperature

        if self._session:
            logger.warning("Updating options; triggering Gemini session reconnect.")
            self._reconnect_event.set()
            self._schedule_gemini_session_close()

    async def update_instructions(self, instructions: str) -> None:
        self._opts.instructions = instructions
        if self._session:
            logger.warning("Updating instructions; triggering Gemini session reconnect.")
            self._reconnect_event.set()
            self._schedule_gemini_session_close()

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        async with self._update_chat_ctx_lock:
            self._chat_ctx = chat_ctx
            turns, _ = to_chat_ctx(self._chat_ctx, id(self), ignore_functions=True)
            tool_results = get_tool_results_for_realtime(self._chat_ctx)
            if turns:
                self._msg_ch.send_nowait(LiveClientContent(turns=turns, turn_complete=False))
            if tool_results:
                self._msg_ch.send_nowait(tool_results)

    async def update_tools(self, tools: list[llm.FunctionTool]) -> None:
        async with self._update_fnc_ctx_lock:
            retained_tools: list[llm.FunctionTool] = []
            gemini_function_declarations: list[FunctionDeclaration] = []

            for tool in tools:
                gemini_function = _build_gemini_fnc(tool)
                gemini_function_declarations.append(gemini_function)
                retained_tools.append(tool)

            self._tools = llm.ToolContext(retained_tools)
            self._gemini_tools = [Tool(function_declarations=gemini_function_declarations)]
            if self._session and gemini_function_declarations:
                logger.warning("Updating tools; triggering Gemini session reconnect.")
                self._reconnect_event.set()
                self._schedule_gemini_session_close()

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self.push_media(frame.data.tobytes(), "audio/pcm")

    def push_video(self, frame: rtc.VideoFrame) -> None:
        encoded_data = images.encode(frame, DEFAULT_ENCODE_OPTIONS)
        self.push_media(encoded_data, "image/jpeg")

    def push_media(self, bytes: bytes, mime_type: str) -> None:
        realtime_input = LiveClientRealtimeInput(
            media_chunks=[Blob(data=bytes, mime_type=mime_type)]
        )
        self._msg_ch.send_nowait(realtime_input)

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        fut = asyncio.Future()

        event_id = utils.shortuuid("gemini-response-")
        self._response_created_futures[event_id] = fut
        self._pending_generation_event_id = event_id

        instructions_content = instructions if is_given(instructions) else "."
        ctx = [Content(parts=[Part(text=instructions_content)], role="user")]
        self._msg_ch.send_nowait(LiveClientContent(turns=ctx, turn_complete=True))

        def _on_timeout() -> None:
            if event_id in self._response_created_futures and not fut.done():
                fut.set_exception(llm.RealtimeError("generate_reply timed out."))
                self._response_created_futures.pop(event_id, None)
                if self._pending_generation_event_id == event_id:
                    self._pending_generation_event_id = None

        handle = asyncio.get_event_loop().call_later(5.0, _on_timeout)
        fut.add_done_callback(lambda _: handle.cancel())

        return fut

    def interrupt(self) -> None:
        logger.warning("interrupt() - no direct cancellation in Gemini")

    def truncate(self, *, message_id: str, audio_end_ms: int) -> None:
        logger.warning(f"truncate(...) called for {message_id}, ignoring for Gemini")

    async def aclose(self) -> None:
        self._msg_ch.close()

        for fut in self._response_created_futures.values():
            if not fut.done():
                fut.set_exception(llm.RealtimeError("Session closed"))

        if self._main_atask:
            await utils.aio.cancel_and_wait(self._main_atask)

        if self._gemini_close_task:
            await utils.aio.cancel_and_wait(self._gemini_close_task)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self):
        while True:
            config = LiveConnectConfig(
                response_modalities=self._opts.response_modalities
                if is_given(self._opts.response_modalities)
                else [Modality.AUDIO],
                generation_config=GenerationConfig(
                    candidate_count=self._opts.candidate_count,
                    temperature=self._opts.temperature
                    if is_given(self._opts.temperature)
                    else None,
                    max_output_tokens=self._opts.max_output_tokens
                    if is_given(self._opts.max_output_tokens)
                    else None,
                    top_p=self._opts.top_p if is_given(self._opts.top_p) else None,
                    top_k=self._opts.top_k if is_given(self._opts.top_k) else None,
                    presence_penalty=self._opts.presence_penalty
                    if is_given(self._opts.presence_penalty)
                    else None,
                    frequency_penalty=self._opts.frequency_penalty
                    if is_given(self._opts.frequency_penalty)
                    else None,
                ),
                system_instruction=Content(parts=[Part(text=self._opts.instructions)])
                if is_given(self._opts.instructions)
                else None,
                speech_config=SpeechConfig(
                    voice_config=VoiceConfig(
                        prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=self._opts.voice)
                    )
                ),
                tools=self._gemini_tools,
                input_audio_transcription=self._opts.input_audio_transcription,
                output_audio_transcription=self._opts.output_audio_transcription,
            )

            async with self._client.aio.live.connect(
                model=self._opts.model, config=config
            ) as session:
                async with self._session_lock:
                    self._session = session

                @utils.log_exceptions(logger=logger)
                async def _send_task():
                    async for msg in self._msg_ch:
                        if isinstance(msg, LiveClientContent):
                            await session.send(input=msg, end_of_turn=True)

                        await session.send(input=msg)
                    await session.send(input=".", end_of_turn=True)

                @utils.log_exceptions(logger=logger)
                async def _recv_task():
                    while True:
                        async for response in session.receive():
                            if self._active_response_id is None:
                                self._start_new_generation()
                            if response.setup_complete:
                                logger.info("connection established with gemini live api server")
                            if response.server_content:
                                self._handle_server_content(response.server_content)
                            if response.tool_call:
                                self._handle_tool_calls(response.tool_call)
                            if response.tool_call_cancellation:
                                self._handle_tool_call_cancellation(response.tool_call_cancellation)
                            if response.usage_metadata:
                                self._handle_usage_metadata(response.usage_metadata)
                            if response.go_away:
                                self._handle_go_away(response.go_away)

                send_task = asyncio.create_task(_send_task(), name="gemini-realtime-send")
                recv_task = asyncio.create_task(_recv_task(), name="gemini-realtime-recv")
                reconnect_task = asyncio.create_task(
                    self._reconnect_event.wait(), name="reconnect-wait"
                )

                try:
                    done, _ = await asyncio.wait(
                        [send_task, recv_task, reconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        if task != reconnect_task:
                            task.result()

                    if reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.cancel_and_wait(send_task, recv_task, reconnect_task)

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

        # Resolve any pending future from generate_reply()
        if self._pending_generation_event_id and (
            fut := self._response_created_futures.pop(self._pending_generation_event_id, None)
        ):
            fut.set_result(generation_event)

        self._pending_generation_event_id = None
        self.emit("generation_created", generation_event)

        self._current_generation.messages[self._active_response_id] = item_generation

    def _handle_server_content(self, server_content: LiveServerContent):
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
                    item_generation.audio_ch.send_nowait(frame)
        input_transcription = server_content.input_transcription
        if input_transcription and input_transcription.text:
            self.emit(
                "input_audio_transcription_completed",
                llm.InputTranscriptionCompleted(
                    item_id=self._active_response_id, transcript=input_transcription.text
                ),
            )
        output_transcription = server_content.output_transcription
        if output_transcription and output_transcription.text:
            item_generation.text_ch.send_nowait(output_transcription.text)

        if server_content.interrupted or server_content.turn_complete:
            self._finalize_response()

    def _finalize_response(self) -> None:
        if not self._current_generation:
            return

        for item_generation in self._current_generation.messages.values():
            item_generation.text_ch.close()
            item_generation.audio_ch.close()

        self._current_generation.function_ch.close()
        self._current_generation.message_ch.close()
        self._current_generation = None
        self._is_interrupted = True
        self._active_response_id = None
        self.emit("agent_speech_stopped")

    def _handle_tool_calls(self, tool_call: LiveServerToolCall):
        if not self._current_generation:
            return
        for fnc_call in tool_call.function_calls:
            self._current_generation.function_ch.send_nowait(
                llm.FunctionCall(
                    call_id=fnc_call.id,
                    name=fnc_call.name,
                    arguments=json.dumps(fnc_call.args),
                )
            )
        self._finalize_response()

    def _handle_tool_call_cancellation(
        self, tool_call_cancellation: LiveServerToolCallCancellation
    ):
        logger.warning(
            "function call cancelled",
            extra={
                "function_call_ids": tool_call_cancellation.ids,
            },
        )
        self.emit("function_calls_cancelled", tool_call_cancellation.ids)

    def _handle_usage_metadata(self, usage_metadata: UsageMetadata):
        # todo: handle metrics
        logger.info("Usage metadata", extra={"usage_metadata": usage_metadata})

    def _handle_go_away(self, go_away: LiveServerGoAway):
        # should we reconnect?
        logger.warning(
            f"gemini live api server will soon disconnect. time left: {go_away.time_left}"
        )

    def commit_audio(self) -> None:
        raise NotImplementedError("commit_audio_buffer is not supported yet")

    def clear_audio(self) -> None:
        raise NotImplementedError("clear_audio is not supported yet")

    def server_vad_enabled(self) -> bool:
        return True
