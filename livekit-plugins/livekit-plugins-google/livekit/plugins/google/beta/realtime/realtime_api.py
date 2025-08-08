from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from google import genai
from google.genai import types
from google.genai.live import AsyncSession
from livekit import rtc
from livekit.agents import APIConnectionError, llm, utils
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import audio as audio_utils, images, is_given
from livekit.plugins.google.beta.realtime.api_proto import ClientEvents, LiveAPIModels, Voice

from ...log import logger
from ...tools import _LLMTool
from ...utils import create_tools_config, get_tool_results_for_realtime, to_fnc_ctx

INPUT_AUDIO_SAMPLE_RATE = 16000
INPUT_AUDIO_CHANNELS = 1
OUTPUT_AUDIO_SAMPLE_RATE = 24000
OUTPUT_AUDIO_CHANNELS = 1

DEFAULT_IMAGE_ENCODE_OPTIONS = images.EncodeOptions(
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
    language: NotGivenOr[str]
    response_modalities: list[types.Modality]
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
    input_audio_transcription: types.AudioTranscriptionConfig | None
    output_audio_transcription: types.AudioTranscriptionConfig | None
    image_encode_options: NotGivenOr[images.EncodeOptions]
    conn_options: APIConnectOptions
    http_options: NotGivenOr[types.HttpOptions]
    enable_affective_dialog: NotGivenOr[bool] = NOT_GIVEN
    proactivity: NotGivenOr[bool] = NOT_GIVEN
    realtime_input_config: NotGivenOr[types.RealtimeInputConfig] = NOT_GIVEN
    context_window_compression: NotGivenOr[types.ContextWindowCompressionConfig] = NOT_GIVEN
    api_version: NotGivenOr[str] = NOT_GIVEN
    gemini_tools: NotGivenOr[list[_LLMTool]] = NOT_GIVEN


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]

    input_id: str
    response_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]

    input_transcription: str = ""
    output_text: str = ""

    _created_timestamp: float = field(default_factory=time.time)
    """The timestamp when the generation is created"""
    _first_token_timestamp: float | None = None
    """The timestamp when the first audio token is received"""
    _completed_timestamp: float | None = None
    """The timestamp when the generation is completed"""
    _done: bool = False
    """Whether the generation is done (set when the turn is complete)"""

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
        instructions: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[LiveAPIModels | str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice: Voice | str = "Puck",
        language: NotGivenOr[str] = NOT_GIVEN,
        modalities: NotGivenOr[list[types.Modality]] = NOT_GIVEN,
        vertexai: NotGivenOr[bool] = NOT_GIVEN,
        project: NotGivenOr[str] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        candidate_count: int = 1,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        top_k: NotGivenOr[int] = NOT_GIVEN,
        presence_penalty: NotGivenOr[float] = NOT_GIVEN,
        frequency_penalty: NotGivenOr[float] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[types.AudioTranscriptionConfig | None] = NOT_GIVEN,
        output_audio_transcription: NotGivenOr[types.AudioTranscriptionConfig | None] = NOT_GIVEN,
        image_encode_options: NotGivenOr[images.EncodeOptions] = NOT_GIVEN,
        enable_affective_dialog: NotGivenOr[bool] = NOT_GIVEN,
        proactivity: NotGivenOr[bool] = NOT_GIVEN,
        realtime_input_config: NotGivenOr[types.RealtimeInputConfig] = NOT_GIVEN,
        context_window_compression: NotGivenOr[types.ContextWindowCompressionConfig] = NOT_GIVEN,
        api_version: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        http_options: NotGivenOr[types.HttpOptions] = NOT_GIVEN,
        _gemini_tools: NotGivenOr[list[_LLMTool]] = NOT_GIVEN,
    ) -> None:
        """
        Initializes a RealtimeModel instance for interacting with Google's Realtime API.

        Environment Requirements:
        - For VertexAI: Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the service account key file or use any of the other Google Cloud auth methods.
        The Google Cloud project and location can be set via `project` and `location` arguments or the environment variables
        `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`. By default, the project is inferred from the service account key file,
        and the location defaults to "us-central1".
        - For Google Gemini API: Set the `api_key` argument or the `GOOGLE_API_KEY` environment variable.

        Args:
            instructions (str, optional): Initial system instructions for the model. Defaults to "".
            api_key (str, optional): Google Gemini API key. If None, will attempt to read from the environment variable GOOGLE_API_KEY.
            modalities (list[Modality], optional): Modalities to use, such as ["TEXT", "AUDIO"]. Defaults to ["AUDIO"].
            model (str, optional): The name of the model to use. Defaults to "gemini-2.0-flash-live-001" or "gemini-2.0-flash-exp" (vertexai).
            voice (api_proto.Voice, optional): Voice setting for audio outputs. Defaults to "Puck".
            language (str, optional): The language(BCP-47 Code) to use for the API. supported languages - https://ai.google.dev/gemini-api/docs/live#supported-languages
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
            image_encode_options (images.EncodeOptions, optional): The configuration for image encoding. Defaults to DEFAULT_ENCODE_OPTIONS.
            enable_affective_dialog (bool, optional): Whether to enable affective dialog. Defaults to False.
            proactivity (bool, optional): Whether to enable proactive audio. Defaults to False.
            realtime_input_config (RealtimeInputConfig, optional): The configuration for realtime input. Defaults to None.
            context_window_compression (ContextWindowCompressionConfig, optional): The configuration for context window compression. Defaults to None.
            conn_options (APIConnectOptions, optional): The configuration for the API connection. Defaults to DEFAULT_API_CONNECT_OPTIONS.
            _gemini_tools (list[LLMTool], optional): Gemini-specific tools to use for the session. This parameter is experimental and may change.

        Raises:
            ValueError: If the API key is required but not found.
        """  # noqa: E501
        if not is_given(input_audio_transcription):
            input_audio_transcription = types.AudioTranscriptionConfig()
        if not is_given(output_audio_transcription):
            output_audio_transcription = types.AudioTranscriptionConfig()

        server_turn_detection = True
        if (
            is_given(realtime_input_config)
            and realtime_input_config.automatic_activity_detection
            and realtime_input_config.automatic_activity_detection.disabled
        ):
            server_turn_detection = False

        modalities = modalities if is_given(modalities) else [types.Modality.AUDIO]

        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=server_turn_detection,
                user_transcription=input_audio_transcription is not None,
                auto_tool_reply_generation=True,
                audio_output=types.Modality.AUDIO in modalities,
            )
        )

        if not is_given(model):
            if vertexai:
                model = "gemini-2.0-flash-exp"
            else:
                model = "gemini-2.0-flash-live-001"

        gemini_api_key = api_key if is_given(api_key) else os.environ.get("GOOGLE_API_KEY")
        gcp_project = project if is_given(project) else os.environ.get("GOOGLE_CLOUD_PROJECT")
        gcp_location: str | None = (
            location
            if is_given(location)
            else os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
        )
        use_vertexai = (
            vertexai
            if is_given(vertexai)
            else os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "0").lower() in ["true", "1"]
        )

        if use_vertexai:
            if not gcp_project or not gcp_location:
                raise ValueError(
                    "Project is required for VertexAI via project kwarg or GOOGLE_CLOUD_PROJECT environment variable"  # noqa: E501
                )
            gemini_api_key = None  # VertexAI does not require an API key
        else:
            gcp_project = None
            gcp_location = None
            if not gemini_api_key:
                raise ValueError(
                    "API key is required for Google API either via api_key or GOOGLE_API_KEY environment variable"  # noqa: E501
                )

        self._opts = _RealtimeOptions(
            model=model,
            api_key=gemini_api_key,
            voice=voice,
            response_modalities=modalities,
            vertexai=use_vertexai,
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
            language=language,
            image_encode_options=image_encode_options,
            enable_affective_dialog=enable_affective_dialog,
            proactivity=proactivity,
            realtime_input_config=realtime_input_config,
            context_window_compression=context_window_compression,
            api_version=api_version,
            gemini_tools=_gemini_tools,
            conn_options=conn_options,
            http_options=http_options,
        )

        self._sessions = weakref.WeakSet[RealtimeSession]()

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Update the options for the RealtimeModel.

        Args:
            voice (str, optional): The voice to use for the session.
            temperature (float, optional): The temperature to use for the session.
            tools (list[LLMTool], optional): The tools to use for the session.
        """
        if is_given(voice):
            self._opts.voice = voice

        if is_given(temperature):
            self._opts.temperature = temperature

        for sess in self._sessions:
            sess.update_options(
                voice=self._opts.voice,
                temperature=self._opts.temperature,
            )

    async def aclose(self) -> None:
        pass


class RealtimeSession(llm.RealtimeSession):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._opts = realtime_model._opts
        self._tools = llm.ToolContext.empty()
        self._gemini_declarations: list[types.FunctionDeclaration] = []
        self._chat_ctx = llm.ChatContext.empty()
        self._msg_ch = utils.aio.Chan[ClientEvents]()
        self._input_resampler: rtc.AudioResampler | None = None

        # 50ms chunks
        self._bstream = audio_utils.AudioByteStream(
            INPUT_AUDIO_SAMPLE_RATE,
            INPUT_AUDIO_CHANNELS,
            samples_per_channel=INPUT_AUDIO_SAMPLE_RATE // 20,
        )

        api_version = self._opts.api_version
        if not api_version and (self._opts.enable_affective_dialog or self._opts.proactivity):
            api_version = "v1alpha"

        http_options = self._opts.http_options or types.HttpOptions(
            timeout=int(self._opts.conn_options.timeout * 1000)
        )
        if api_version:
            http_options.api_version = api_version

        self._client = genai.Client(
            api_key=self._opts.api_key,
            vertexai=self._opts.vertexai,
            project=self._opts.project,
            location=self._opts.location,
            http_options=http_options,
        )

        self._main_atask = asyncio.create_task(self._main_task(), name="gemini-realtime-session")

        self._current_generation: _ResponseGeneration | None = None
        self._active_session: AsyncSession | None = None
        # indicates if the underlying session should end
        self._session_should_close = asyncio.Event()
        self._response_created_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}
        self._pending_generation_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None

        self._session_resumption_handle: str | None = None
        self._in_user_activity = False
        self._session_lock = asyncio.Lock()
        self._num_retries = 0

    async def _close_active_session(self) -> None:
        async with self._session_lock:
            if self._active_session:
                try:
                    await self._active_session.close()
                except Exception as e:
                    logger.warning(f"error closing Gemini session: {e}")
                finally:
                    self._active_session = None

    def _mark_restart_needed(self) -> None:
        if not self._session_should_close.is_set():
            self._session_should_close.set()
            # reset the msg_ch, do not send messages from previous session
            self._msg_ch = utils.aio.Chan[ClientEvents]()

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
    ) -> None:
        should_restart = False
        if is_given(voice) and self._opts.voice != voice:
            self._opts.voice = voice
            should_restart = True

        if is_given(temperature) and self._opts.temperature != temperature:
            self._opts.temperature = temperature if is_given(temperature) else NOT_GIVEN
            should_restart = True

        if should_restart:
            self._mark_restart_needed()

    async def update_instructions(self, instructions: str) -> None:
        if not is_given(self._opts.instructions) or self._opts.instructions != instructions:
            self._opts.instructions = instructions
            self._mark_restart_needed()

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        async with self._session_lock:
            if not self._active_session:
                self._chat_ctx = chat_ctx.copy()
                return

        diff_ops = llm.utils.compute_chat_ctx_diff(self._chat_ctx, chat_ctx)

        if diff_ops.to_remove:
            logger.warning("Gemini Live does not support removing messages")

        append_ctx = llm.ChatContext.empty()
        for _, item_id in diff_ops.to_create:
            item = chat_ctx.get_by_id(item_id)
            if item:
                append_ctx.items.append(item)

        if append_ctx.items:
            turns_dict, _ = append_ctx.copy(
                exclude_function_call=True,
            ).to_provider_format(format="google", inject_dummy_user_message=False)
            # we are not generating, and do not need to inject
            turns = [types.Content.model_validate(turn) for turn in turns_dict]
            tool_results = get_tool_results_for_realtime(append_ctx, vertexai=self._opts.vertexai)
            if turns:
                self._send_client_event(types.LiveClientContent(turns=turns, turn_complete=False))
            if tool_results:
                self._send_client_event(tool_results)

        # since we don't have a view of the history on the server side, we'll assume
        # the current state is accurate. this isn't perfect because removals aren't done.
        self._chat_ctx = chat_ctx.copy()

    async def update_tools(self, tools: list[llm.FunctionTool | llm.RawFunctionTool]) -> None:
        new_declarations: list[types.FunctionDeclaration] = to_fnc_ctx(tools)
        current_tool_names = {f.name for f in self._gemini_declarations}
        new_tool_names = {f.name for f in new_declarations}

        if current_tool_names != new_tool_names:
            self._gemini_declarations = new_declarations
            self._tools = llm.ToolContext(tools)
            self._mark_restart_needed()

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    @property
    def _manual_activity_detection(self) -> bool:
        if (
            is_given(self._opts.realtime_input_config)
            and self._opts.realtime_input_config.automatic_activity_detection is not None
            and self._opts.realtime_input_config.automatic_activity_detection.disabled
        ):
            return True
        return False

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        for f in self._resample_audio(frame):
            for nf in self._bstream.write(f.data.tobytes()):
                realtime_input = types.LiveClientRealtimeInput(
                    media_chunks=[types.Blob(data=nf.data.tobytes(), mime_type="audio/pcm")]
                )
                self._send_client_event(realtime_input)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        encoded_data = images.encode(
            frame, self._opts.image_encode_options or DEFAULT_IMAGE_ENCODE_OPTIONS
        )
        realtime_input = types.LiveClientRealtimeInput(
            media_chunks=[types.Blob(data=encoded_data, mime_type="image/jpeg")]
        )
        self._send_client_event(realtime_input)

    def _send_client_event(self, event: ClientEvents) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        if self._pending_generation_fut and not self._pending_generation_fut.done():
            logger.warning(
                "generate_reply called while another generation is pending, cancelling previous."
            )
            self._pending_generation_fut.cancel("Superseded by new generate_reply call")

        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._pending_generation_fut = fut

        if self._in_user_activity:
            self._send_client_event(
                types.LiveClientRealtimeInput(
                    activity_end=types.ActivityEnd(),
                )
            )
            self._in_user_activity = False

        # Gemini requires the last message to end with user's turn
        # so we need to add a placeholder user turn in order to trigger a new generation
        turns = []
        if is_given(instructions):
            turns.append(types.Content(parts=[types.Part(text=instructions)], role="model"))
        turns.append(types.Content(parts=[types.Part(text=".")], role="user"))
        self._send_client_event(types.LiveClientContent(turns=turns, turn_complete=True))

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

    def start_user_activity(self) -> None:
        if not self._manual_activity_detection:
            return

        if not self._in_user_activity:
            self._in_user_activity = True
            self._send_client_event(
                types.LiveClientRealtimeInput(
                    activity_start=types.ActivityStart(),
                )
            )

    def interrupt(self) -> None:
        # Gemini Live treats activity start as interruption, so we rely on start_user_activity
        # notifications to handle it
        if (
            self._opts.realtime_input_config
            and self._opts.realtime_input_config.activity_handling
            == types.ActivityHandling.NO_INTERRUPTION
        ):
            return
        self.start_user_activity()

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        logger.warning("truncate is not supported by the Google Realtime API.")
        pass

    async def aclose(self) -> None:
        self._msg_ch.close()
        self._session_should_close.set()

        if self._main_atask:
            await utils.aio.cancel_and_wait(self._main_atask)

        await self._close_active_session()

        if self._pending_generation_fut and not self._pending_generation_fut.done():
            self._pending_generation_fut.cancel("Session closed")

        for fut in self._response_created_futures.values():
            if not fut.done():
                fut.set_exception(llm.RealtimeError("Session closed before response created"))
        self._response_created_futures.clear()

        if self._current_generation:
            self._mark_current_generation_done()

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        max_retries = self._opts.conn_options.max_retry

        while not self._msg_ch.closed:
            # previous session might not be closed yet, we'll do it here.
            await self._close_active_session()

            self._session_should_close.clear()
            config = self._build_connect_config()
            session = None
            try:
                logger.debug("connecting to Gemini Realtime API...")
                async with self._client.aio.live.connect(
                    model=self._opts.model, config=config
                ) as session:
                    async with self._session_lock:
                        self._active_session = session
                        turns_dict, _ = self._chat_ctx.copy(
                            exclude_function_call=True,
                        ).to_provider_format(format="google", inject_dummy_user_message=False)
                        if turns_dict:
                            turns = [types.Content.model_validate(turn) for turn in turns_dict]
                            await session.send_client_content(
                                turns=turns,  # type: ignore
                                turn_complete=False,
                            )
                    # queue up existing chat context
                    send_task = asyncio.create_task(
                        self._send_task(session), name="gemini-realtime-send"
                    )
                    recv_task = asyncio.create_task(
                        self._recv_task(session), name="gemini-realtime-recv"
                    )
                    restart_wait_task = asyncio.create_task(
                        self._session_should_close.wait(), name="gemini-restart-wait"
                    )

                    done, pending = await asyncio.wait(
                        [send_task, recv_task, restart_wait_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task is not restart_wait_task and task.exception():
                            logger.error(f"error in task {task.get_name()}: {task.exception()}")
                            raise task.exception() or Exception(f"{task.get_name()} failed")

                    if restart_wait_task not in done and self._msg_ch.closed:
                        break

                    for task in pending:
                        await utils.aio.cancel_and_wait(task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Gemini Realtime API error: {e}", exc_info=e)
                if not self._msg_ch.closed:
                    # we shouldn't retry when it's not connected, usually this means incorrect
                    # parameters or setup
                    if not session or max_retries == 0:
                        self._emit_error(e, recoverable=False)
                        raise APIConnectionError(message="Failed to connect to Gemini Live") from e

                    if self._num_retries == max_retries:
                        self._emit_error(e, recoverable=False)
                        raise APIConnectionError(
                            message=f"Failed to connect to Gemini Live after {max_retries} attempts"
                        ) from e

                    retry_interval = self._opts.conn_options._interval_for_retry(self._num_retries)
                    logger.warning(
                        f"Gemini Realtime API connection failed, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={"attempt": self._num_retries, "max_retries": max_retries},
                    )
                    await asyncio.sleep(retry_interval)
                    self._num_retries += 1
            finally:
                await self._close_active_session()

    async def _send_task(self, session: AsyncSession) -> None:
        try:
            async for msg in self._msg_ch:
                async with self._session_lock:
                    if self._session_should_close.is_set() or (
                        not self._active_session or self._active_session != session
                    ):
                        break
                if isinstance(msg, types.LiveClientContent):
                    await session.send_client_content(
                        turns=msg.turns,  # type: ignore
                        turn_complete=msg.turn_complete or True,
                    )
                elif isinstance(msg, types.LiveClientToolResponse) and msg.function_responses:
                    await session.send_tool_response(function_responses=msg.function_responses)
                elif isinstance(msg, types.LiveClientRealtimeInput):
                    if msg.media_chunks:
                        for media_chunk in msg.media_chunks:
                            await session.send_realtime_input(media=media_chunk)
                    elif msg.activity_start:
                        await session.send_realtime_input(activity_start=msg.activity_start)
                    elif msg.activity_end:
                        await session.send_realtime_input(activity_end=msg.activity_end)
                else:
                    logger.warning(f"Warning: Received unhandled message type: {type(msg)}")

        except Exception as e:
            if not self._session_should_close.is_set():
                logger.error(f"error in send task: {e}", exc_info=e)
                self._mark_restart_needed()
        finally:
            logger.debug("send task finished.")

    async def _recv_task(self, session: AsyncSession) -> None:
        try:
            while True:
                async with self._session_lock:
                    if self._session_should_close.is_set() or (
                        not self._active_session or self._active_session != session
                    ):
                        logger.debug("receive task: Session changed or closed, stopping receive.")
                        break

                async for response in session.receive():
                    if (not self._current_generation or self._current_generation._done) and (
                        response.server_content or response.tool_call
                    ):
                        self._start_new_generation()

                    if response.session_resumption_update:
                        if (
                            response.session_resumption_update.resumable
                            and response.session_resumption_update.new_handle
                        ):
                            self._session_resumption_handle = (
                                response.session_resumption_update.new_handle
                            )

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

                    if self._num_retries > 0:
                        self._num_retries = 0  # reset the retry counter

                # TODO(dz): a server-side turn is complete
        except Exception as e:
            if not self._session_should_close.is_set():
                logger.error(f"error in receive task: {e}", exc_info=e)
                self._mark_restart_needed()
        finally:
            self._mark_current_generation_done()

    def _build_connect_config(self) -> types.LiveConnectConfig:
        temp = self._opts.temperature if is_given(self._opts.temperature) else None

        tools_config = create_tools_config(
            function_tools=self._gemini_declarations,
            gemini_tools=self._opts.gemini_tools if is_given(self._opts.gemini_tools) else None,
        )
        conf = types.LiveConnectConfig(
            response_modalities=self._opts.response_modalities,
            generation_config=types.GenerationConfig(
                candidate_count=self._opts.candidate_count,
                temperature=temp,
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
            system_instruction=types.Content(parts=[types.Part(text=self._opts.instructions)])
            if is_given(self._opts.instructions)
            else None,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self._opts.voice)
                ),
                language_code=self._opts.language if is_given(self._opts.language) else None,
            ),
            tools=tools_config,  # type: ignore
            input_audio_transcription=self._opts.input_audio_transcription,
            output_audio_transcription=self._opts.output_audio_transcription,
            session_resumption=types.SessionResumptionConfig(
                handle=self._session_resumption_handle
            ),
        )

        if is_given(self._opts.proactivity):
            conf.proactivity = types.ProactivityConfig(proactive_audio=self._opts.proactivity)
        if is_given(self._opts.enable_affective_dialog):
            conf.enable_affective_dialog = self._opts.enable_affective_dialog
        if is_given(self._opts.realtime_input_config):
            conf.realtime_input_config = self._opts.realtime_input_config
        if is_given(self._opts.context_window_compression):
            conf.context_window_compression = self._opts.context_window_compression

        return conf

    def _start_new_generation(self) -> None:
        if self._current_generation and not self._current_generation._done:
            logger.warning("starting new generation while another is active. Finalizing previous.")
            self._mark_current_generation_done()

        response_id = utils.shortuuid("GR_")
        self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            response_id=response_id,
            input_id=utils.shortuuid("GI_"),
            text_ch=utils.aio.Chan[str](),
            audio_ch=utils.aio.Chan[rtc.AudioFrame](),
            _created_timestamp=time.time(),
        )
        if not self._realtime_model.capabilities.audio_output:
            self._current_generation.audio_ch.close()

        msg_modalities = asyncio.Future[list[Literal["text", "audio"]]]()
        msg_modalities.set_result(
            ["audio", "text"] if self._realtime_model.capabilities.audio_output else ["text"]
        )
        self._current_generation.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=response_id,
                text_stream=self._current_generation.text_ch,
                audio_stream=self._current_generation.audio_ch,
                modalities=msg_modalities,
            )
        )

        generation_event = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
        )

        if self._pending_generation_fut and not self._pending_generation_fut.done():
            generation_event.user_initiated = True
            self._pending_generation_fut.set_result(generation_event)
            self._pending_generation_fut = None
        else:
            # emit input_speech_started event before starting an agent initiated generation
            # to interrupt the previous audio playout if any
            self._handle_input_speech_started()

        self.emit("generation_created", generation_event)

    def _handle_server_content(self, server_content: types.LiveServerContent) -> None:
        current_gen = self._current_generation
        if not current_gen:
            logger.warning("received server content but no active generation.")
            return

        if model_turn := server_content.model_turn:
            for part in model_turn.parts or []:
                if part.text:
                    current_gen.push_text(part.text)
                if part.inline_data:
                    if not current_gen._first_token_timestamp:
                        current_gen._first_token_timestamp = time.time()
                    frame_data = part.inline_data.data
                    try:
                        if not isinstance(frame_data, bytes):
                            raise ValueError("frame_data is not bytes")
                        frame = rtc.AudioFrame(
                            data=frame_data,
                            sample_rate=OUTPUT_AUDIO_SAMPLE_RATE,
                            num_channels=OUTPUT_AUDIO_CHANNELS,
                            samples_per_channel=len(frame_data) // (2 * OUTPUT_AUDIO_CHANNELS),
                        )
                        current_gen.audio_ch.send_nowait(frame)
                    except ValueError as e:
                        logger.error(f"Error creating audio frame from Gemini data: {e}")

        if input_transcription := server_content.input_transcription:
            text = input_transcription.text
            if text:
                if current_gen.input_transcription == "":
                    # gemini would start with a space, which doesn't make sense
                    # at beginning of the transcript
                    text = text.lstrip()
                current_gen.input_transcription += text
                self.emit(
                    "input_audio_transcription_completed",
                    llm.InputTranscriptionCompleted(
                        item_id=current_gen.input_id,
                        transcript=current_gen.input_transcription,
                        is_final=False,
                    ),
                )

        if output_transcription := server_content.output_transcription:
            text = output_transcription.text
            if text:
                current_gen.push_text(text)

        if server_content.generation_complete or server_content.turn_complete:
            current_gen._completed_timestamp = time.time()

        if server_content.interrupted:
            self._handle_input_speech_started()

        if server_content.turn_complete:
            self._mark_current_generation_done()

    def _mark_current_generation_done(self) -> None:
        if not self._current_generation or self._current_generation._done:
            return

        # emit input_speech_stopped event after the generation is done
        self._handle_input_speech_stopped()

        gen = self._current_generation

        # The only way we'd know that the transcription is complete is by when they are
        # done with generation
        if gen.input_transcription:
            self.emit(
                "input_audio_transcription_completed",
                llm.InputTranscriptionCompleted(
                    item_id=gen.input_id,
                    transcript=gen.input_transcription,
                    is_final=True,
                ),
            )

            # since gemini doesn't give us a view of the chat history on the server side,
            # we would handle it manually here
            self._chat_ctx.add_message(
                role="user",
                content=gen.input_transcription,
                id=gen.input_id,
            )

        if gen.output_text:
            self._chat_ctx.add_message(
                role="assistant",
                content=gen.output_text,
                id=gen.response_id,
            )

        if not gen.text_ch.closed:
            if self._opts.output_audio_transcription is None:
                # close the text data of transcription synchronizer
                gen.text_ch.send_nowait("")
            gen.text_ch.close()
        if not gen.audio_ch.closed:
            gen.audio_ch.close()

        gen.function_ch.close()
        gen.message_ch.close()
        gen._done = True

    def _handle_input_speech_started(self) -> None:
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _handle_input_speech_stopped(self) -> None:
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(user_transcription_enabled=False),
        )

    def _handle_tool_calls(self, tool_call: types.LiveServerToolCall) -> None:
        if not self._current_generation:
            logger.warning("received tool call but no active generation.")
            return

        gen = self._current_generation
        for fnc_call in tool_call.function_calls or []:
            arguments = json.dumps(fnc_call.args)

            gen.function_ch.send_nowait(
                llm.FunctionCall(
                    call_id=fnc_call.id or utils.shortuuid("fnc-call-"),
                    name=fnc_call.name,  # type: ignore
                    arguments=arguments,
                )
            )
        self._mark_current_generation_done()

    def _handle_tool_call_cancellation(
        self, tool_call_cancellation: types.LiveServerToolCallCancellation
    ) -> None:
        logger.warning(
            "server cancelled tool calls",
            extra={"function_call_ids": tool_call_cancellation.ids},
        )

    def _handle_usage_metadata(self, usage_metadata: types.UsageMetadata) -> None:
        current_gen = self._current_generation
        if not current_gen:
            logger.warning("no active generation to report metrics for")
            return

        ttft = (
            current_gen._first_token_timestamp - current_gen._created_timestamp
            if current_gen._first_token_timestamp
            else -1
        )
        duration = (
            current_gen._completed_timestamp or time.time()
        ) - current_gen._created_timestamp

        def _token_details_map(
            token_details: list[types.ModalityTokenCount] | None,
        ) -> dict[str, int]:
            token_details_map = {"audio_tokens": 0, "text_tokens": 0, "image_tokens": 0}
            if not token_details:
                return token_details_map

            for token_detail in token_details:
                if not token_detail.token_count:
                    continue

                if token_detail.modality == types.MediaModality.AUDIO:
                    token_details_map["audio_tokens"] += token_detail.token_count
                elif token_detail.modality == types.MediaModality.TEXT:
                    token_details_map["text_tokens"] += token_detail.token_count
                elif token_detail.modality == types.MediaModality.IMAGE:
                    token_details_map["image_tokens"] += token_detail.token_count
            return token_details_map

        metrics = RealtimeModelMetrics(
            label=self._realtime_model._label,
            request_id=current_gen.response_id,
            timestamp=current_gen._created_timestamp,
            duration=duration,
            ttft=ttft,
            cancelled=False,
            input_tokens=usage_metadata.prompt_token_count or 0,
            output_tokens=usage_metadata.response_token_count or 0,
            total_tokens=usage_metadata.total_token_count or 0,
            tokens_per_second=(usage_metadata.response_token_count or 0) / duration
            if duration > 0
            else 0,
            input_token_details=RealtimeModelMetrics.InputTokenDetails(
                **_token_details_map(usage_metadata.prompt_tokens_details),
                cached_tokens=sum(
                    token_detail.token_count or 0
                    for token_detail in usage_metadata.cache_tokens_details or []
                ),
                cached_tokens_details=RealtimeModelMetrics.CachedTokenDetails(
                    **_token_details_map(usage_metadata.cache_tokens_details),
                ),
            ),
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                **_token_details_map(usage_metadata.response_tokens_details),
            ),
        )
        self.emit("metrics_collected", metrics)

    def _handle_go_away(self, go_away: types.LiveServerGoAway) -> None:
        logger.warning(
            f"Gemini server indicates disconnection soon. Time left: {go_away.time_left}"
        )
        # TODO(dz): this isn't a seamless reconnection just yet
        self._session_should_close.set()

    def commit_audio(self) -> None:
        pass

    def clear_audio(self) -> None:
        pass

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                # input audio changed to a different sample rate
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != INPUT_AUDIO_SAMPLE_RATE
            or frame.num_channels != INPUT_AUDIO_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=INPUT_AUDIO_SAMPLE_RATE,
                num_channels=INPUT_AUDIO_CHANNELS,
            )

        if self._input_resampler:
            # TODO(long): flush the resampler when the input source is changed
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
