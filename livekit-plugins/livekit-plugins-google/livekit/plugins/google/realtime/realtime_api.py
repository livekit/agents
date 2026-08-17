from __future__ import annotations

import asyncio
import json
import os
import time
import weakref
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

import google.auth.credentials
from google.auth._default_async import default_async
from google.genai import Client as GenAIClient, types
from google.genai.live import AsyncSession
from livekit import rtc
from livekit.agents import LanguageCode, llm, utils
from livekit.agents.llm.realtime import (
    _UserMessageSyncResult,
    _UserMessageSyncStatus,
)
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import audio as audio_utils, images, is_given
from livekit.plugins.google.realtime.api_proto import ClientEvents, LiveAPIModels, Voice

from ..log import logger
from ..utils import create_function_response, create_tools_config, get_tool_results_for_realtime
from ..version import __version__

INPUT_AUDIO_SAMPLE_RATE = 16000
INPUT_AUDIO_CHANNELS = 1
OUTPUT_AUDIO_SAMPLE_RATE = 24000
OUTPUT_AUDIO_CHANNELS = 1

# Bound audio retained after a manual-input discard while the contaminated provider turn
# finishes. Truncation is logged because Gemini cannot clear its input buffer in-session.
MANUAL_AUDIO_QUARANTINE_MAX_DURATION = 1.0

DEFAULT_IMAGE_ENCODE_OPTIONS = images.EncodeOptions(
    format="JPEG",
    quality=75,
    resize_options=images.ResizeOptions(width=1024, height=1024, strategy="scale_aspect_fit"),
)

lk_google_debug = int(os.getenv("LK_GOOGLE_DEBUG", 0))

# stop rejecting tool calls after this many in a row to avoid a loop (tool_choice="none")
MAX_TOOL_CALL_REJECTIONS = 3

# Known VertexAI models for the Live API
# See: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api
KNOWN_VERTEXAI_MODELS: frozenset[str] = frozenset(
    {
        "gemini-live-2.5-flash-native-audio",
    }
)

# Known Gemini API models for the Live API
# See: https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-live
KNOWN_GEMINI_API_MODELS: frozenset[str] = frozenset(
    {
        "gemini-3.1-flash-live-preview",
        "gemini-2.5-flash-native-audio-preview-12-2025",
    }
)


def _validate_model_api_match(model: str, use_vertexai: bool) -> None:
    """
    Validate that the model name matches the API being used.

    Raises ValueError if a known model is used with the wrong API configuration.

    Args:
        model: The model name being used
        use_vertexai: Whether VertexAI is enabled
    """
    if use_vertexai and model in KNOWN_GEMINI_API_MODELS:
        raise ValueError(
            f"Model '{model}' is a Gemini API model, but vertexai=True. "
            f"Use a VertexAI model (e.g., 'gemini-live-2.5-flash-native-audio') "
            f"or set vertexai=False."
        )

    if not use_vertexai and model in KNOWN_VERTEXAI_MODELS:
        raise ValueError(
            f"Model '{model}' is a VertexAI model, but vertexai=False. "
            f"Use a Gemini API model (e.g., 'gemini-2.5-flash-native-audio-preview-12-2025') "
            f"or set vertexai=True."
        )


def _warn_vertex_scheduling_unsupported() -> None:
    logger.warning(
        "tool_response_scheduling is not supported by Vertex AI and will be ignored; "
        "tool responses use the default scheduling there."
    )


def _get_1008_error_hint(error_message: str) -> str | None:
    """
    Generate a hint for WebSocket 1008 policy violation errors.

    This provides a generic hint when the connection fails with a 1008 error,
    which often indicates the model name doesn't match the API being used.

    Args:
        error_message: The error message from the WebSocket exception

    Returns:
        A helpful hint string, or None if not a 1008 error
    """
    if "1008" not in error_message and "policy violation" not in error_message.lower():
        return None

    return (
        "\n\nHint: A 1008 policy violation error often indicates that the model name "
        "doesn't match the API being used. VertexAI models typically start with "
        "'gemini-live-', while Gemini API models start with 'gemini-2.' or similar. "
        "Please verify your model name matches your API configuration."
    )


@dataclass
class InputTranscription:
    item_id: str
    transcript: str


class _InputState(Enum):
    IDLE = auto()
    AUDIO_ACTIVE = auto()
    INTERRUPT_ONLY = auto()
    TEXT_PENDING = auto()
    TEXT_TRIGGER_SENT = auto()
    AUDIO_TRIGGER_SENT = auto()
    LEGACY_TRIGGER_SENT = auto()
    ABORTED = auto()


@dataclass
class _ToolResponseOutboxEntry:
    event: types.LiveClientToolResponse
    call_ids: tuple[str, ...]
    queued_epoch: int | None = None
    in_flight_epoch: int | None = None


@dataclass
class _DeferredManualInput:
    realtime_inputs: deque[types.LiveClientRealtimeInput] = field(default_factory=deque)
    has_realtime_input: bool = False
    sealed: bool = False
    generation_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
    instructions: NotGivenOr[str] = NOT_GIVEN


@dataclass
class _QuarantinedManualInput:
    realtime_input: types.LiveClientRealtimeInput
    audio_duration: float = 0.0


@dataclass
class _InputSendInFlight:
    event: ClientEvents
    sequence: int
    epoch: int
    migrated_epoch: int | None = None


@dataclass
class _RealtimeOptions:
    model: LiveAPIModels | str
    api_key: str | None
    voice: Voice | str
    language: NotGivenOr[LanguageCode]
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
    media_resolution: NotGivenOr[types.MediaResolution] = NOT_GIVEN
    enable_affective_dialog: NotGivenOr[bool] = NOT_GIVEN
    proactivity: NotGivenOr[bool] = NOT_GIVEN
    realtime_input_config: NotGivenOr[types.RealtimeInputConfig] = NOT_GIVEN
    context_window_compression: NotGivenOr[types.ContextWindowCompressionConfig] = NOT_GIVEN
    api_version: NotGivenOr[str] = NOT_GIVEN
    tool_behavior: NotGivenOr[types.Behavior] = NOT_GIVEN
    tool_response_scheduling: NotGivenOr[types.FunctionResponseScheduling] = NOT_GIVEN
    tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN
    thinking_config: NotGivenOr[types.ThinkingConfig] = NOT_GIVEN
    session_resumption: NotGivenOr[types.SessionResumptionConfig] = NOT_GIVEN
    credentials: google.auth.credentials.Credentials | None = None


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
    _extra_content_warned: bool = False
    """Whether we've warned about audio/text arriving after generation completed"""

    def push_text(self, text: str) -> None:
        if self.text_ch.closed:
            # generation_complete already finalized the output; a turn should not emit
            # more text, so drop it (see _handle_server_content)
            if not self._extra_content_warned:
                self._extra_content_warned = True
                logger.warning("Gemini sent text after generation completed; dropping it")
            return

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
        tool_behavior: NotGivenOr[types.Behavior] = NOT_GIVEN,
        tool_response_scheduling: NotGivenOr[types.FunctionResponseScheduling] = NOT_GIVEN,
        session_resumption: NotGivenOr[types.SessionResumptionConfig] = NOT_GIVEN,
        api_version: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        http_options: NotGivenOr[types.HttpOptions] = NOT_GIVEN,
        media_resolution: NotGivenOr[types.MediaResolution] = NOT_GIVEN,
        thinking_config: NotGivenOr[types.ThinkingConfig] = NOT_GIVEN,
        credentials: google.auth.credentials.Credentials | None = None,
    ) -> None:
        """
        Initializes a RealtimeModel instance for interacting with Google's Realtime API.

        Environment Requirements:
        - For VertexAI: Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the service account key file or use any of the other Google Cloud auth methods.
        The Google Cloud project and location can be set via `project` and `location` arguments or the environment variables
        `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`. By default, the project is inferred from the service account key file,
        and the location defaults to "us-central1".
        - For Google Gemini API: Set the `api_key` argument or the `GOOGLE_API_KEY` environment variable.

        To use an external STT as the model's text input, explicitly disable Gemini's automatic
        activity detection and select ``realtime_input_mode="text"`` on ``AgentSession``::

            model = RealtimeModel(
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
                )
            )
            session = AgentSession(
                llm=model,
                stt=external_stt,
                vad=external_vad,
                turn_handling={"realtime_input_mode": "text"},
            )

        Args:
            instructions (str, optional): Initial system instructions for the model. Defaults to "".
            api_key (str, optional): Google Gemini API key. If None, will attempt to read from the environment variable GOOGLE_API_KEY.
            modalities (list[Modality], optional): Modalities to use, such as ["TEXT", "AUDIO"]. Defaults to ["AUDIO"].
            model (str, optional): The name of the model to use. Defaults to "gemini-2.5-flash-native-audio-preview-12-2025" or "gemini-live-2.5-flash-native-audio" (vertexai).
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
            media_resolution (MediaResolution, optional): The media resolution for the session. Defaults to None.
            enable_affective_dialog (bool, optional): Whether to enable affective dialog. Defaults to False.
            proactivity (bool, optional): Whether to enable proactive audio. Defaults to False.
            realtime_input_config (RealtimeInputConfig, optional): The configuration for realtime input. Defaults to None.
            context_window_compression (ContextWindowCompressionConfig, optional): The configuration for context window compression. Defaults to None.
            tool_behavior (Behavior, optional): The behavior for tool call. Default behavior is BLOCK in Gemini Realtime API.
            tool_response_scheduling (FunctionResponseScheduling, optional): The scheduling for tool response. Default scheduling is WHEN_IDLE.
            session_resumption (SessionResumptionConfig, optional): The configuration for session resumption. Defaults to None.
            thinking_config (ThinkingConfig, optional): Native audio thinking configuration.
            conn_options (APIConnectOptions, optional): The configuration for the API connection. Defaults to DEFAULT_API_CONNECT_OPTIONS.

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
        use_vertexai = (
            vertexai
            if is_given(vertexai)
            else os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "0").lower() in ["true", "1"]
        )
        if use_vertexai and is_given(tool_response_scheduling):
            _warn_vertex_scheduling_unsupported()
        if not is_given(model):
            model = (
                "gemini-live-2.5-flash-native-audio"
                if use_vertexai
                else "gemini-2.5-flash-native-audio-preview-12-2025"
            )

        mutable = "3.1" not in model
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=server_turn_detection,
                user_transcription=input_audio_transcription is not None,
                auto_tool_reply_generation=True,
                audio_output=types.Modality.AUDIO in modalities,
                manual_function_calls=False,
                mutable_chat_context=mutable,
                mutable_instructions=mutable,
                mutable_tools=False,
                per_response_tool_choice=False,
            )
        )

        gemini_api_key = api_key if is_given(api_key) else os.environ.get("GOOGLE_API_KEY")
        gcp_project = project if is_given(project) else os.environ.get("GOOGLE_CLOUD_PROJECT")
        gcp_location: str | None = (
            location
            if is_given(location)
            else os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
        )

        if use_vertexai:
            if not gcp_project:
                _, gcp_project = default_async(  # type: ignore
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            if not gcp_project or not gcp_location:
                raise ValueError(
                    "Project is required for VertexAI via project kwarg or GOOGLE_CLOUD_PROJECT environment variable"  # noqa: E501
                )
            gemini_api_key = None  # VertexAI does not require an API key
        else:
            gcp_project = None
            gcp_location = None
            if credentials is not None:
                logger.warning(
                    "'credentials' is only applicable to VertexAI and will be ignored for the Gemini API"
                )
                credentials = None
            if not gemini_api_key:
                raise ValueError(
                    "API key is required for Google API either via api_key or GOOGLE_API_KEY environment variable"  # noqa: E501
                )

        # Validate model/API compatibility for known models
        _validate_model_api_match(model, use_vertexai)

        if "3.1" in model:
            logger.warning(
                f"'{model}' has limited mid-session update support. instructions, chat "
                "context, and tool updates will not be applied until the next session."
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
            language=LanguageCode(language) if isinstance(language, str) else language,
            image_encode_options=image_encode_options,
            enable_affective_dialog=enable_affective_dialog,
            proactivity=proactivity,
            realtime_input_config=realtime_input_config,
            context_window_compression=context_window_compression,
            api_version=api_version,
            tool_behavior=tool_behavior,
            tool_response_scheduling=tool_response_scheduling,
            conn_options=conn_options,
            http_options=http_options,
            media_resolution=media_resolution,
            thinking_config=thinking_config,
            session_resumption=session_resumption,
            credentials=credentials,
        )

        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        if self._opts.vertexai:
            return "Vertex AI"
        else:
            return "Gemini"

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        # Gemini drives manual turns via activity_start/activity_end, not commit_audio/clear_audio,
        # so the pipeline can't gatekeep turns yet; keep can_disable_turn_detection=False for now
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        tool_behavior: NotGivenOr[types.Behavior] = NOT_GIVEN,
        tool_response_scheduling: NotGivenOr[types.FunctionResponseScheduling] = NOT_GIVEN,
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

        if is_given(tool_behavior):
            self._opts.tool_behavior = tool_behavior

        if is_given(tool_response_scheduling):
            self._opts.tool_response_scheduling = tool_response_scheduling

        for sess in self._sessions:
            sess.update_options(
                voice=self._opts.voice,
                temperature=self._opts.temperature,
                tool_behavior=self._opts.tool_behavior,
                tool_response_scheduling=self._opts.tool_response_scheduling,
            )

    async def aclose(self) -> None:
        pass


class RealtimeSession(llm.RealtimeSession):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._opts = realtime_model._opts
        self._tools = llm.ToolContext.empty()
        self._chat_ctx = llm.ChatContext.empty()
        self._msg_ch = utils.aio.Chan[ClientEvents]()
        self._input_resampler: rtc.AudioResampler | None = None

        # 50ms chunks
        self._bstream = audio_utils.AudioByteStream(
            INPUT_AUDIO_SAMPLE_RATE,
            INPUT_AUDIO_CHANNELS,
            samples_per_channel=INPUT_AUDIO_SAMPLE_RATE // 20,
        )
        self._quarantined_manual_inputs: deque[_QuarantinedManualInput] = deque()
        self._quarantined_manual_audio_duration = 0.0
        self._manual_audio_quarantine_active = False
        self._manual_audio_quarantine_truncation_warned = False
        self._input_event_sequence = 0
        self._input_event_sequences: dict[int, int] = {}
        self._invalid_input_event_sequences: set[int] = set()
        self._input_send_in_flight_sequence: int | None = None
        self._input_send_in_flight: _InputSendInFlight | None = None
        self._delivered_input_event_ids: set[int] = set()
        self._provider_visible_input_sequence: int | None = None
        self._pending_text_input_item_id: str | None = None
        self._provider_turn_active = False
        self._restart_after_provider_turn_epoch: int | None = None
        self._go_away_restart_epoch: int | None = None
        self._go_away_deadline_handle: asyncio.TimerHandle | None = None
        self._deferred_manual_inputs: deque[_DeferredManualInput] = deque()
        self._deferred_manual_input_pipeline_active = False

        api_version = self._opts.api_version
        if (
            not api_version
            and (self._opts.enable_affective_dialog or self._opts.proactivity)
            and not self._opts.vertexai
        ):
            api_version = "v1alpha"

        http_options = self._opts.http_options or types.HttpOptions(
            timeout=int(self._opts.conn_options.timeout * 1000)
        )
        if api_version:
            http_options.api_version = api_version
        if not http_options.headers:
            http_options.headers = {}
        http_options.headers["x-goog-api-client"] = f"livekit-agents/{__version__}"

        self._client = GenAIClient(
            api_key=self._opts.api_key,
            vertexai=self._opts.vertexai,
            project=self._opts.project,
            location=self._opts.location,
            credentials=self._opts.credentials,
            http_options=http_options,
        )

        self._closed = False
        self._terminal_error: llm.RealtimeError | None = None
        self._main_atask = asyncio.create_task(self._main_task(), name="gemini-realtime-session")

        self._current_generation: _ResponseGeneration | None = None
        self._active_session: AsyncSession | None = None
        # indicates if the underlying session should end
        self._session_should_close = asyncio.Event()
        self._response_created_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}
        self._pending_generation_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        self._pending_generation_epoch: int | None = None
        self._input_state = _InputState.IDLE
        self._activity_has_realtime_input = False
        self._session_epoch = 0
        self._provider_session_established = False
        self._tool_response_outbox: dict[int, _ToolResponseOutboxEntry] = {}
        self._delivered_tool_response_event_ids: set[int] = set()

        # number of tool calls rejected in the current tool_choice="none" turn; non-zero also
        # means we're draining that turn's trailing events (which have no generation to attach
        # to). reset when the next generation starts.
        self._rejected_tool_calls = 0

        self._session_resumption_handle: str | None = (
            self._opts.session_resumption.handle
            if is_given(self._opts.session_resumption)
            else None
        )
        self._provider_tool_call_ids: set[str] = set()
        self._settled_provider_tool_call_ids: set[str] = set()

        self._session_lock = asyncio.Lock()
        self._num_retries = 0
        # error recorded by the recv/send tasks so _main_task can bound retries
        # and surface it through the "error" event
        self._session_error: Exception | None = None

    async def _close_active_session(self) -> None:
        async with self._session_lock:
            if self._active_session:
                try:
                    await self._active_session.close()
                except Exception as e:
                    logger.warning(f"error closing Gemini session: {e}")
                finally:
                    self._active_session = None

    @property
    def _in_user_activity(self) -> bool:
        return self._input_state in (_InputState.AUDIO_ACTIVE, _InputState.INTERRUPT_ONLY)

    @property
    def _client_content_user_turn_pending(self) -> bool:
        return self._input_state == _InputState.TEXT_PENDING

    def _clear_local_audio_input(self) -> None:
        self._quarantined_manual_inputs.clear()
        self._quarantined_manual_audio_duration = 0.0
        self._manual_audio_quarantine_truncation_warned = False
        self._bstream.clear()
        self._input_resampler = None

    def _buffer_quarantined_manual_input(
        self,
        realtime_input: types.LiveClientRealtimeInput,
        *,
        audio_duration: float = 0.0,
    ) -> None:
        if realtime_input.video is not None:
            # Gemini consumes at most one video frame per second. Keep only the latest
            # unowned frame without disturbing the relative order of retained audio.
            self._quarantined_manual_inputs = deque(
                item
                for item in self._quarantined_manual_inputs
                if item.realtime_input.video is None
            )

        self._quarantined_manual_inputs.append(
            _QuarantinedManualInput(
                realtime_input=realtime_input,
                audio_duration=audio_duration,
            )
        )
        self._quarantined_manual_audio_duration += audio_duration
        truncated = False
        quarantined_audio_count = sum(
            item.audio_duration > 0.0 for item in self._quarantined_manual_inputs
        )
        while (
            self._quarantined_manual_audio_duration > MANUAL_AUDIO_QUARANTINE_MAX_DURATION
            and quarantined_audio_count > 1
        ):
            for index, item in enumerate(self._quarantined_manual_inputs):
                if item.audio_duration > 0.0:
                    del self._quarantined_manual_inputs[index]
                    self._quarantined_manual_audio_duration -= item.audio_duration
                    break
            quarantined_audio_count -= 1
            truncated = True

        if truncated and not self._manual_audio_quarantine_truncation_warned:
            self._manual_audio_quarantine_truncation_warned = True
            logger.warning(
                "manual audio quarantine exceeded %.1fs; retaining only the most recent media",
                MANUAL_AUDIO_QUARANTINE_MAX_DURATION,
            )

    def _replay_quarantined_manual_inputs(self) -> None:
        if self._quarantined_manual_inputs:
            self._activity_has_realtime_input = True
        while self._quarantined_manual_inputs:
            self._send_input_event(self._quarantined_manual_inputs.popleft().realtime_input)
        self._quarantined_manual_audio_duration = 0.0
        self._manual_audio_quarantine_truncation_warned = False

    def _move_quarantined_manual_inputs(
        self, destination: deque[types.LiveClientRealtimeInput]
    ) -> None:
        while self._quarantined_manual_inputs:
            destination.append(self._quarantined_manual_inputs.popleft().realtime_input)
        self._quarantined_manual_audio_duration = 0.0
        self._manual_audio_quarantine_truncation_warned = False

    def _send_input_event(self, event: ClientEvents) -> bool:
        self._input_event_sequences[id(event)] = self._input_event_sequence
        accepted = self._send_client_event(event)
        # Historically this private seam was void-returning, and existing integrations/tests
        # commonly replace it with a recorder whose successful return is None. Only the
        # provider queue's explicit False means that insertion was rejected.
        if accepted is False:
            self._input_event_sequences.pop(id(event), None)
        return accepted is not False

    def _advance_input_sequence(self, *, invalidate: bool = False) -> int:
        sequence = self._input_event_sequence
        if invalidate:
            self._invalid_input_event_sequences.add(sequence)
        self._input_event_sequence += 1
        return sequence

    def _release_invalid_input_sequence(self, sequence: int) -> None:
        if sequence not in self._invalid_input_event_sequences:
            return
        if (
            sequence != self._input_send_in_flight_sequence
            and sequence not in self._input_event_sequences.values()
        ):
            self._invalid_input_event_sequences.discard(sequence)

    def _input_may_be_provider_visible(self, sequence: int) -> bool:
        return sequence in (
            self._provider_visible_input_sequence,
            self._input_send_in_flight_sequence,
        )

    def _discard_pending_text_input(self) -> None:
        item_id = self._pending_text_input_item_id
        if item_id is not None:
            self._chat_ctx.items[:] = [item for item in self._chat_ctx.items if item.id != item_id]
        self._pending_text_input_item_id = None

    def _force_pending_discard_restart(self) -> None:
        if self._restart_after_provider_turn_epoch is None:
            return
        self._restart_after_provider_turn_epoch = None
        self._mark_restart_needed(preserve_manual_audio=True)

    def _cancel_go_away_deadline(self) -> None:
        handle = self._go_away_deadline_handle
        self._go_away_deadline_handle = None
        if handle is not None:
            handle.cancel()

    @staticmethod
    def _go_away_restart_delay(time_left: str | None) -> float:
        if not time_left:
            return 0.1
        value = time_left.strip()
        if value.endswith("s"):
            value = value[:-1]
        try:
            seconds = float(value)
        except ValueError:
            logger.warning(
                "Gemini returned an unrecognized GoAway deadline; restarting promptly",
                extra={"time_left": time_left},
            )
            return 0.1
        return max(min(seconds * 0.9, seconds - 0.05), 0.0)

    def _schedule_go_away_restart(self, time_left: str | None) -> None:
        self._cancel_go_away_deadline()
        restart_epoch = self._session_epoch
        self._go_away_restart_epoch = restart_epoch

        def _on_deadline() -> None:
            if self._go_away_restart_epoch != restart_epoch:
                return
            self._go_away_deadline_handle = None
            self._force_go_away_restart()

        self._go_away_deadline_handle = asyncio.get_running_loop().call_later(
            self._go_away_restart_delay(time_left), _on_deadline
        )

    def _force_go_away_restart(self) -> None:
        if self._go_away_restart_epoch != self._session_epoch:
            return

        self._go_away_restart_epoch = None
        self._cancel_go_away_deadline()
        previous_input_state = self._input_state

        if previous_input_state == _InputState.TEXT_TRIGGER_SENT:
            pending_fut = self._pending_generation_fut
            if pending_fut is not None and not pending_fut.done():
                # Preserve the exact logical request while moving its provider acknowledgement
                # to the fresh epoch. The old timeout callback is epoch-guarded.
                self._pending_generation_fut = None
                self._pending_generation_epoch = None
                self._input_state = _InputState.TEXT_PENDING
                self._mark_restart_needed()
                self._track_pending_generation(pending_fut)
                completion_accepted = self._send_input_event(
                    types.LiveClientContent(turn_complete=True)
                )
                self._advance_input_sequence()
                self._input_state = _InputState.TEXT_TRIGGER_SENT
                if not completion_accepted:
                    self._input_state = _InputState.ABORTED
                    self._settle_pending_generation(
                        llm.RealtimeError(
                            "Gemini Realtime could not replay the pending text completion"
                        )
                    )
                return

        if previous_input_state in (_InputState.AUDIO_ACTIVE, _InputState.AUDIO_TRIGGER_SENT):
            self._settle_pending_generation(
                llm.RealtimeError(
                    "Gemini Realtime raw audio input cannot be replayed after the GoAway deadline"
                )
            )
            self._input_state = _InputState.ABORTED

        # A forced deadline cannot trust the active provider checkpoint. Rebuild from the
        # authoritative text history; _mark_restart_needed preserves TEXT_PENDING exactly.
        self._mark_restart_needed()

    def _settle_pending_generation(self, error: llm.RealtimeError) -> None:
        if self._pending_generation_fut and not self._pending_generation_fut.done():
            # Detach before completing the future. Its done callback otherwise interprets the
            # completion as an external cancellation of the current provider request.
            pending_fut = self._pending_generation_fut
            self._pending_generation_fut = None
            self._pending_generation_epoch = None
            pending_fut.set_exception(error)

    def _track_pending_generation(
        self,
        fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        if fut is None:
            fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._pending_generation_fut = fut
        self._pending_generation_epoch = self._session_epoch
        generation_epoch = self._session_epoch

        def _on_timeout() -> None:
            if (
                not fut.done()
                and self._pending_generation_fut is fut
                and self._pending_generation_epoch == generation_epoch
            ):
                self._pending_generation_fut = None
                self._pending_generation_epoch = None
                self._input_state = _InputState.IDLE
                fut.set_exception(
                    llm.RealtimeError(
                        "generate_reply timed out waiting for generation_created event."
                    )
                )
                self._mark_restart_needed()

        timeout_handle = asyncio.get_event_loop().call_later(5.0, _on_timeout)

        def _on_fut_done(f: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            timeout_handle.cancel()
            is_current = (
                self._pending_generation_fut is fut
                and self._pending_generation_epoch == generation_epoch
            )
            if is_current:
                self._pending_generation_fut = None
                self._pending_generation_epoch = None
            if f.cancelled() and is_current:
                self._input_state = _InputState.IDLE
                # Gemini provides no request ID or cancel event. A fresh session epoch is the
                # only way to prevent the abandoned response from satisfying a later request.
                self._mark_restart_needed()

        fut.add_done_callback(_on_fut_done)
        return fut

    def _discard_deferred_manual_input(
        self,
        *,
        error: llm.RealtimeError | None = None,
        cancel_reason: str = "Deferred Gemini input discarded",
        all_inputs: bool = False,
    ) -> None:
        if all_inputs:
            self._deferred_manual_input_pipeline_active = False
        if not self._deferred_manual_inputs:
            return

        if all_inputs:
            deferred_inputs = list(self._deferred_manual_inputs)
            self._deferred_manual_inputs.clear()
        else:
            deferred_inputs = [self._deferred_manual_inputs.pop()]

        self._clear_local_audio_input()
        self._activity_has_realtime_input = False
        for deferred in deferred_inputs:
            deferred.realtime_inputs.clear()
            fut = deferred.generation_fut
            if fut is None or fut.done():
                continue
            if error is not None:
                fut.set_exception(error)
            else:
                fut.cancel(cancel_reason)

    def _reap_cancelled_deferred_manual_input(self) -> None:
        if not self._deferred_manual_inputs:
            return

        retained: deque[_DeferredManualInput] = deque()
        for deferred in self._deferred_manual_inputs:
            fut = deferred.generation_fut
            if fut is not None and fut.cancelled():
                deferred.realtime_inputs.clear()
                continue
            retained.append(deferred)
        self._deferred_manual_inputs = retained

    def _defer_manual_generation(
        self,
        *,
        instructions: NotGivenOr[str],
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        assert self._deferred_manual_inputs
        deferred = self._deferred_manual_inputs[-1]

        if deferred.generation_fut is not None and not deferred.generation_fut.done():
            logger.warning("superseding a deferred Gemini generation request")
            previous_fut = deferred.generation_fut
            deferred.generation_fut = None
            previous_fut.cancel("Superseded by a newer generate_reply request")

        if not deferred.sealed:
            self._flush_audio_input()
            self._move_quarantined_manual_inputs(deferred.realtime_inputs)
            deferred.has_realtime_input = bool(deferred.realtime_inputs)
            deferred.sealed = True

        if deferred.has_realtime_input:
            if is_given(instructions):
                logger.warning(
                    "per-response instructions are not supported for an active Gemini audio "
                    "activity; ignoring instructions"
                )
            deferred.instructions = NOT_GIVEN
        else:
            deferred.instructions = instructions

        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        deferred.generation_fut = fut

        def _on_deferred_done(f: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            if f.cancelled() and deferred.generation_fut is f:
                self._reap_cancelled_deferred_manual_input()

        fut.add_done_callback(_on_deferred_done)
        return fut

    def _activate_next_deferred_manual_input(self) -> None:
        self._reap_cancelled_deferred_manual_input()
        while self._deferred_manual_inputs:
            deferred = self._deferred_manual_inputs.popleft()
            fut = deferred.generation_fut
            if fut is None or not fut.done():
                break
            deferred.realtime_inputs.clear()
        else:
            self._deferred_manual_input_pipeline_active = False
            return

        if fut is None:
            self._deferred_manual_input_pipeline_active = False
            self._manual_audio_quarantine_active = False
            self._input_state = _InputState.AUDIO_ACTIVE
            self._activity_has_realtime_input = False
            start_accepted = self._send_input_event(
                types.LiveClientRealtimeInput(activity_start=types.ActivityStart())
            )
            if deferred.realtime_inputs:
                self._activity_has_realtime_input = True
            while deferred.realtime_inputs:
                self._send_input_event(deferred.realtime_inputs.popleft())
            self._replay_quarantined_manual_inputs()
            if not start_accepted:
                self._input_state = _InputState.ABORTED
                self._clear_local_audio_input()
            return

        self._deferred_manual_input_pipeline_active = True
        # The transaction was sealed at EOU. Keep later audio quarantined as pre-roll for the
        # next ActivityStart; only the frozen transaction frames belong before this ActivityEnd.
        if not deferred.has_realtime_input:
            self._input_state = _InputState.IDLE
            turns: list[types.Content] = []
            if is_given(deferred.instructions):
                turns.append(
                    types.Content(parts=[types.Part(text=deferred.instructions)], role="model")
                )
            turns.append(types.Content(parts=[types.Part(text=".")], role="user"))
            accepted = self._send_client_event(
                types.LiveClientContent(turns=turns, turn_complete=True)
            )
            if accepted:
                self._input_state = _InputState.LEGACY_TRIGGER_SENT
                self._track_pending_generation(fut)
            else:
                fut.set_exception(llm.RealtimeError("Gemini Realtime input trigger was not queued"))
            return

        self._input_state = _InputState.AUDIO_ACTIVE
        self._activity_has_realtime_input = False
        start_accepted = self._send_input_event(
            types.LiveClientRealtimeInput(activity_start=types.ActivityStart())
        )
        while deferred.realtime_inputs:
            self._send_input_event(deferred.realtime_inputs.popleft())

        end_accepted = self._send_input_event(
            types.LiveClientRealtimeInput(activity_end=types.ActivityEnd())
        )
        if start_accepted and end_accepted:
            self._advance_input_sequence()
            self._input_state = _InputState.AUDIO_TRIGGER_SENT
            self._activity_has_realtime_input = False
            self._track_pending_generation(fut)
        else:
            self._input_state = _InputState.ABORTED
            self._clear_local_audio_input()
            fut.set_exception(llm.RealtimeError("Gemini Realtime input trigger was not queued"))

    def _clear_provider_tool_state(self) -> None:
        for entry in self._tool_response_outbox.values():
            entry.queued_epoch = None
            entry.in_flight_epoch = None
        self._tool_response_outbox.clear()
        self._delivered_tool_response_event_ids.clear()
        self._provider_tool_call_ids.clear()
        self._settled_provider_tool_call_ids.clear()

    def _set_terminal_error(self, error: llm.RealtimeError) -> None:
        if self._terminal_error is None:
            self._terminal_error = error

        # A terminal transport failure has no reconnect loop left to consume queued work.
        # Settle every public surface immediately while leaving aclose() responsible for
        # releasing the active session and the owned GenAI client.
        self._discard_deferred_manual_input(error=self._terminal_error, all_inputs=True)
        self._cancel_go_away_deadline()
        self._go_away_restart_epoch = None
        self._input_state = _InputState.IDLE
        self._restart_after_provider_turn_epoch = None
        self._provider_turn_active = False
        self._provider_visible_input_sequence = None
        self._input_send_in_flight_sequence = None
        self._input_send_in_flight = None
        self._delivered_input_event_ids.clear()
        self._pending_text_input_item_id = None
        self._manual_audio_quarantine_active = False
        self._activity_has_realtime_input = False
        self._clear_local_audio_input()
        self._input_event_sequences.clear()
        self._invalid_input_event_sequences.clear()
        self._clear_provider_tool_state()
        self._mark_current_generation_done()
        self._settle_pending_generation(self._terminal_error)
        self._session_should_close.set()
        self._msg_ch.close()

    def _mark_restart_needed(
        self,
        *,
        on_error: bool = False,
        resume_session: bool = False,
        preserve_manual_audio: bool = False,
    ) -> bool:
        if self._closed or self._terminal_error is not None:
            return False

        self._cancel_go_away_deadline()
        self._go_away_restart_epoch = None
        self._restart_after_provider_turn_epoch = None
        preserve_manual_audio = preserve_manual_audio or self._manual_audio_quarantine_active
        previous_input_state = self._input_state
        # Session resumption is safe only between input turns. A fresh epoch must fail or replay
        # active input deliberately instead of silently omitting ActivityStart, audio, or text.
        resume_session = (
            resume_session
            and self._session_resumption_handle is not None
            and previous_input_state == _InputState.IDLE
        )
        if previous_input_state == _InputState.TEXT_TRIGGER_SENT:
            resume_session = False
            # Keep the exact user message as ordinary history, but clear its completion state.
            # A later user turn owns the next (and only) generation trigger.
            self._input_state = _InputState.IDLE
        elif previous_input_state == _InputState.AUDIO_TRIGGER_SENT:
            resume_session = False
            # Raw audio cannot be replayed or correlated across Gemini connections.
            self._input_state = _InputState.ABORTED
        elif previous_input_state == _InputState.LEGACY_TRIGGER_SENT:
            resume_session = False
            self._input_state = _InputState.IDLE
        elif resume_session:
            # Resumption is allowed only while idle; provider and local input state remain idle.
            self._input_state = _InputState.IDLE
        elif previous_input_state == _InputState.TEXT_PENDING:
            # No completion was requested yet; replay the exact user append after reconnect.
            self._input_state = _InputState.TEXT_PENDING
        elif previous_input_state == _InputState.AUDIO_ACTIVE:
            # Raw audio cannot be replayed on a fresh connection. Remember that this turn was
            # abandoned so a later EOT fails rather than manufacturing a placeholder turn.
            self._input_state = _InputState.ABORTED
        elif previous_input_state == _InputState.ABORTED:
            # Preserve the one-shot rejection marker until the abandoned turn's generate_reply
            # arrives or a new explicit activity supersedes it.
            self._input_state = _InputState.ABORTED
        else:
            self._input_state = _InputState.IDLE

        if previous_input_state != _InputState.TEXT_PENDING:
            self._pending_text_input_item_id = None

        if not resume_session:
            # Session resumption restores provider-side activity/audio. Intentional restarts
            # (clear, cancellation, config/protocol changes) must start a genuinely fresh epoch.
            self._session_resumption_handle = None
            self._advance_input_sequence(invalidate=True)
            if not preserve_manual_audio:
                self._clear_local_audio_input()
                self._manual_audio_quarantine_active = False
            self._provider_visible_input_sequence = None
            self._input_send_in_flight_sequence = None
            self._input_send_in_flight = None
            self._delivered_input_event_ids.clear()
            self._provider_turn_active = False
            self._activity_has_realtime_input = False
            # A fresh epoch cannot continue channels belonging to the abandoned response.
            self._mark_current_generation_done()
        self._settle_pending_generation(
            llm.RealtimeError("Gemini Realtime session restarted before generation started")
        )

        self._session_epoch += 1
        self._session_should_close.set()
        # Detach the old transport queue so a stale sender cannot consume new-epoch events.
        # A genuine session resumption keeps provider-side input state, so migrate queued events
        # in order. A fresh epoch reconstructs replayable text from authoritative chat context.
        # Tool responses belong to the provider call that requested them and can cross this
        # boundary only when that exact provider session is resumed.
        old_msg_ch = self._msg_ch
        new_msg_ch = utils.aio.Chan[ClientEvents]()
        if resume_session:
            input_in_flight = self._input_send_in_flight
            if (
                input_in_flight is not None
                and input_in_flight.epoch == self._session_epoch - 1
                and input_in_flight.migrated_epoch is None
            ):
                input_in_flight.migrated_epoch = self._session_epoch
                self._input_event_sequences[id(input_in_flight.event)] = input_in_flight.sequence
                new_msg_ch.send_nowait(input_in_flight.event)

            # The sender may already have removed one response from the old queue. Restore its
            # original position ahead of later queued input. If the old SDK send subsequently
            # succeeds, its delivery tombstone makes this placeholder a no-op.
            for entry in self._tool_response_outbox.values():
                if entry.in_flight_epoch is None:
                    continue
                entry.in_flight_epoch = None
                entry.queued_epoch = self._session_epoch
                new_msg_ch.send_nowait(entry.event)

        while not old_msg_ch.empty():
            msg = old_msg_ch.recv_nowait()
            if id(msg) in self._delivered_input_event_ids:
                self._delivered_input_event_ids.discard(id(msg))
                self._input_event_sequences.pop(id(msg), None)
                continue

            if id(msg) in self._delivered_tool_response_event_ids:
                self._delivered_tool_response_event_ids.discard(id(msg))
                continue

            if queued_entry := self._tool_response_outbox.get(id(msg)):
                if resume_session:
                    queued_entry.queued_epoch = self._session_epoch
                    new_msg_ch.send_nowait(msg)
                continue

            input_sequence = self._input_event_sequences.pop(id(msg), None)
            if resume_session:
                new_msg_ch.send_nowait(msg)
                if input_sequence is not None:
                    self._input_event_sequences[id(msg)] = input_sequence
            elif not on_error:
                if isinstance(msg, types.LiveClientContent) and msg.turn_complete is True:
                    logger.debug(
                        "discarding client content completion during Gemini session restart",
                        extra={"content": str(msg)},
                    )
        old_msg_ch.close()
        self._msg_ch = new_msg_ch
        if not resume_session:
            if self._tool_response_outbox:
                logger.warning(
                    "discarding pending Gemini tool response because the provider session "
                    "cannot be resumed",
                    extra={"count": len(self._tool_response_outbox)},
                )
            self._clear_provider_tool_state()
            self._provider_session_established = False
            self._invalid_input_event_sequences.clear()
            self._activate_next_deferred_manual_input()

        return resume_session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        tool_behavior: NotGivenOr[types.Behavior] = NOT_GIVEN,
        tool_response_scheduling: NotGivenOr[types.FunctionResponseScheduling] = NOT_GIVEN,
    ) -> None:
        should_restart = False
        if is_given(voice) and self._opts.voice != voice:
            self._opts.voice = voice
            should_restart = True

        if is_given(temperature) and self._opts.temperature != temperature:
            self._opts.temperature = temperature if is_given(temperature) else NOT_GIVEN
            should_restart = True

        if is_given(tool_behavior) and self._opts.tool_behavior != tool_behavior:
            self._opts.tool_behavior = tool_behavior
            should_restart = True

        if (
            is_given(tool_response_scheduling)
            and self._opts.tool_response_scheduling != tool_response_scheduling
        ):
            self._opts.tool_response_scheduling = tool_response_scheduling
            if self._opts.vertexai:
                _warn_vertex_scheduling_unsupported()
            # no need to restart

        if is_given(tool_choice):
            # no per-response tool_choice on Gemini; "none" is emulated by rejecting any tool
            # call emitted during the turn (see _reject_tool_calls).
            self._opts.tool_choice = tool_choice
            if tool_choice == "none":
                logger.warning(
                    "the Google Realtime API has no tool_choice='none'; tool calls emitted "
                    "this turn will be rejected so the model replies directly."
                )
            elif tool_choice not in (None, "auto"):
                logger.warning(
                    f"tool_choice='{tool_choice}' is not supported by the Google Realtime API, "
                    "falling back to 'auto'."
                )

        if should_restart:
            self._mark_restart_needed()

    async def update_instructions(self, instructions: str) -> None:
        if not is_given(self._opts.instructions) or self._opts.instructions != instructions:
            self._opts.instructions = instructions

            async with self._session_lock:
                if not self._active_session:
                    # No active session yet — restart will pick up new instructions via _build_connect_config
                    self._mark_restart_needed()
                    return

            if not self._realtime_model.capabilities.mutable_instructions:
                return

            # Active session exists — send mid-session system instruction update (no reconnect needed)
            logger.debug("Updating instructions mid-session")
            self._send_client_event(
                types.LiveClientContent(
                    turns=[
                        types.Content(
                            parts=[types.Part(text=instructions)],
                            # Vertex AI ignores role=None or role="system" and only works with role="model".
                            # Gemini Live API (non-Vertex) errors on role="system"; role=None works as system role.
                            role="model" if self._opts.vertexai else None,
                        )
                    ],
                    turn_complete=False,
                )
            )

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        await self._update_chat_ctx(chat_ctx)

    async def _sync_user_message(
        self, chat_ctx: llm.ChatContext, message_id: str
    ) -> _UserMessageSyncResult:
        if self._terminal_error is not None:
            error = llm.RealtimeError(
                f"Gemini Realtime session is unavailable: {self._terminal_error}"
            )
            return _UserMessageSyncResult(_UserMessageSyncStatus.REJECTED, error)
        if self._closed:
            error = llm.RealtimeError("Gemini Realtime session is closed")
            return _UserMessageSyncResult(_UserMessageSyncStatus.REJECTED, error)

        try:
            accepted = await self._update_chat_ctx(chat_ctx, target_message_id=message_id)
        except llm.RealtimeError as error:
            return _UserMessageSyncResult(_UserMessageSyncStatus.REJECTED, error)

        if accepted is False:
            rejected_error = llm.RealtimeError(
                "Gemini Realtime rejected the finalized user message before queueing it"
            )
            return _UserMessageSyncResult(_UserMessageSyncStatus.REJECTED, rejected_error)
        return _UserMessageSyncResult(_UserMessageSyncStatus.ACCEPTED)

    async def _update_chat_ctx(
        self,
        chat_ctx: llm.ChatContext,
        *,
        target_message_id: str | None = None,
    ) -> bool | None:
        if self._terminal_error is not None:
            raise llm.RealtimeError(
                f"Gemini Realtime session is unavailable: {self._terminal_error}"
            )
        if self._closed:
            return False if target_message_id is not None else None

        # Check for system/developer messages that will be dropped
        system_msg_count = sum(
            1 for msg in chat_ctx.messages() if msg.role in ("system", "developer")
        )
        if system_msg_count > 0:
            logger.warning(
                f"Gemini Realtime model '{self._opts.model}' does not support 'system' or "
                f"'developer' roles in chat history. Dropping {system_msg_count} system "
                f"message(s) from chat context. Gemini Realtime only supports 'user' and "
                f"'model' roles. Use update_instructions() to set system-level context instead."
            )

        chat_ctx = chat_ctx.copy(
            exclude_handoff=True,
            exclude_instructions=True,
            exclude_empty_message=True,
            exclude_config_update=True,
        )
        diff_ops = llm.utils.compute_chat_ctx_diff(self._chat_ctx, chat_ctx)
        created_ids = {item_id for _, item_id in diff_ops.to_create}
        target_already_present = (
            target_message_id is not None
            and target_message_id not in created_ids
            and self._chat_ctx.get_by_id(target_message_id) is not None
        )

        append_ctx = llm.ChatContext.empty()
        for _, item_id in diff_ops.to_create:
            item = chat_ctx.get_by_id(item_id)
            if item:
                append_ctx.items.append(item)

        turns: list[types.Content] = []
        if append_ctx.items and self._realtime_model.capabilities.mutable_chat_context:
            turns_dict, _ = append_ctx.copy(exclude_function_call=True).to_provider_format(
                format="google", inject_dummy_user_message=False
            )
            turns = [types.Content.model_validate(turn) for turn in turns_dict]

        pending_user = (
            next(
                (
                    item
                    for item in reversed(append_ctx.items)
                    if isinstance(item, llm.ChatMessage) and item.role == "user"
                ),
                None,
            )
            if turns and turns[-1].role == "user"
            else None
        )
        next_pending_text_input_item_id = pending_user.id if pending_user is not None else None
        next_input_state = _InputState.TEXT_PENDING if pending_user is not None else None
        target_is_pending_user = (
            target_message_id is not None and next_pending_text_input_item_id == target_message_id
        )

        if next_input_state == _InputState.TEXT_PENDING:
            self._discard_deferred_manual_input(
                error=llm.RealtimeError(
                    "Gemini Realtime deferred audio input was replaced by text input"
                ),
                all_inputs=True,
            )
            self._force_pending_discard_restart()
            if self._manual_audio_quarantine_active:
                self._clear_local_audio_input()
                self._manual_audio_quarantine_active = False

        async with self._session_lock:
            if self._terminal_error is not None:
                raise llm.RealtimeError(
                    f"Gemini Realtime session is unavailable: {self._terminal_error}"
                )
            if self._closed:
                return False if target_message_id is not None else None

            if tool_response := self._build_owned_tool_response(append_ctx):
                event, call_ids = tool_response
                self._register_tool_response(event, call_ids)

            if not self._active_session or self._session_should_close.is_set():
                if turns and self._session_resumption_handle is not None:
                    self._mark_restart_needed()
                self._chat_ctx = chat_ctx
                if next_input_state is not None:
                    self._input_state = next_input_state
                    self._pending_text_input_item_id = next_pending_text_input_item_id
                if target_message_id is not None:
                    return target_already_present or target_is_pending_user
                return None

        if diff_ops.to_remove:
            logger.warning("Gemini Live does not support removing messages")

        if turns and self._input_state in (
            _InputState.AUDIO_ACTIVE,
            _InputState.INTERRUPT_ONLY,
        ):
            # Client content and realtime activity are separate Gemini input protocols. Close
            # the old transport without resumption and replay the authoritative context once.
            self._mark_restart_needed()
            self._chat_ctx = chat_ctx
            if next_input_state is not None:
                self._input_state = next_input_state
                self._pending_text_input_item_id = next_pending_text_input_item_id
            if target_message_id is not None:
                return target_already_present or target_is_pending_user
            return None

        target_accepted: bool | None = True if target_already_present else None
        if append_ctx.items and self._realtime_model.capabilities.mutable_chat_context and turns:
            content_event = types.LiveClientContent(turns=turns, turn_complete=False)
            if next_input_state == _InputState.TEXT_PENDING:
                input_accepted = self._send_input_event(content_event)
                if target_is_pending_user:
                    target_accepted = input_accepted
                if not input_accepted and target_is_pending_user:
                    return False
            else:
                self._send_client_event(content_event)

            if next_input_state is not None:
                self._input_state = next_input_state
                self._pending_text_input_item_id = next_pending_text_input_item_id

        # Since Gemini does not expose its server-side history, accepted queue insertion or
        # authoritative reconnect replay is the provider synchronization boundary.
        self._chat_ctx = chat_ctx
        if target_message_id is not None:
            return target_accepted is True
        return None

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        tool_ctx = llm.ToolContext(tools)
        if self._tools == tool_ctx:
            return

        self._tools = tool_ctx
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

    @property
    def session_resumption_handle(self) -> str | None:
        return self._session_resumption_handle

    def _queue_audio_frame(self, frame: rtc.AudioFrame) -> None:
        if self._manual_activity_detection:
            if self._deferred_manual_inputs and not self._deferred_manual_inputs[-1].sealed:
                deferred = self._deferred_manual_inputs[-1]
                deferred.realtime_inputs.append(self._audio_frame_to_realtime_input(frame))
                deferred.has_realtime_input = True
                return
            if self._manual_audio_quarantine_active:
                self._buffer_quarantined_manual_input(
                    self._audio_frame_to_realtime_input(frame),
                    audio_duration=frame.duration,
                )
                return
        self._send_audio_frame(frame)

    def _audio_frame_to_realtime_input(
        self, frame: rtc.AudioFrame
    ) -> types.LiveClientRealtimeInput:
        return types.LiveClientRealtimeInput(
            audio=types.Blob(
                data=frame.data.tobytes(),
                mime_type=f"audio/pcm;rate={INPUT_AUDIO_SAMPLE_RATE}",
            )
        )

    def _send_audio_frame(self, frame: rtc.AudioFrame) -> None:
        event = self._audio_frame_to_realtime_input(frame)
        if self._manual_activity_detection:
            self._send_input_event(event)
        else:
            self._send_client_event(event)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed or self._terminal_error is not None:
            return

        self._reap_cancelled_deferred_manual_input()
        if frame.samples_per_channel > 0 and (
            not self._manual_activity_detection
            or self._input_state in (_InputState.AUDIO_ACTIVE, _InputState.INTERRUPT_ONLY)
        ):
            self._activity_has_realtime_input = True
        for f in self._resample_audio(frame):
            for nf in self._bstream.write(f.data.tobytes()):
                self._queue_audio_frame(nf)

    def _flush_audio_input(self) -> None:
        if self._input_resampler is not None:
            for frame in self._input_resampler.flush():
                for nf in self._bstream.write(frame.data.tobytes()):
                    self._queue_audio_frame(nf)
            self._input_resampler = None

        # ActivityEnd is a hard input boundary. Flush the final partial chunk before it so
        # legitimate tail audio is delivered and can never be combined with the next turn.
        for frame in self._bstream.flush():
            self._queue_audio_frame(frame)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        if self._closed or self._terminal_error is not None:
            return

        if not self._manual_activity_detection or self._input_state in (
            _InputState.AUDIO_ACTIVE,
            _InputState.INTERRUPT_ONLY,
        ):
            self._activity_has_realtime_input = True
        encoded_data = images.encode(
            frame, self._opts.image_encode_options or DEFAULT_IMAGE_ENCODE_OPTIONS
        )
        realtime_input = types.LiveClientRealtimeInput(
            video=types.Blob(data=encoded_data, mime_type="image/jpeg")
        )
        if self._manual_activity_detection:
            if self._deferred_manual_inputs and not self._deferred_manual_inputs[-1].sealed:
                deferred = self._deferred_manual_inputs[-1]
                deferred.realtime_inputs.append(realtime_input)
                deferred.has_realtime_input = True
                return
            if self._manual_audio_quarantine_active:
                self._buffer_quarantined_manual_input(realtime_input)
                return
            self._send_input_event(realtime_input)
        else:
            self._send_client_event(realtime_input)

    def _send_client_event(self, event: ClientEvents) -> bool:
        try:
            self._msg_ch.send_nowait(event)
        except utils.aio.channel.ChanClosed:
            return False
        return True

    def _build_owned_tool_response(
        self, append_ctx: llm.ChatContext
    ) -> tuple[types.LiveClientToolResponse, tuple[str, ...]] | None:
        pending_call_ids = {
            call_id for entry in self._tool_response_outbox.values() for call_id in entry.call_ids
        }
        owned_outputs: list[llm.FunctionCallOutput] = []
        abandoned_call_ids: list[str] = []
        for item in append_ctx.items:
            if not isinstance(item, llm.FunctionCallOutput):
                continue

            call_id = item.call_id
            if call_id in pending_call_ids or call_id in self._settled_provider_tool_call_ids:
                continue
            if call_id in self._provider_tool_call_ids:
                owned_outputs.append(item)
            else:
                abandoned_call_ids.append(call_id)

        if abandoned_call_ids and self._provider_session_established:
            logger.warning(
                "discarding Gemini tool response for calls not owned by the current provider "
                f"session: {', '.join(abandoned_call_ids)}"
            )
        if not owned_outputs:
            return None

        # Vertex drops scheduling, and Gemini reads it only on NON_BLOCKING tools.
        supports_silent_scheduling = (
            not self._opts.vertexai and self._opts.tool_behavior == types.Behavior.NON_BLOCKING
        )
        if not supports_silent_scheduling and (
            silenced := [item.name for item in owned_outputs if not item.reply_required]
        ):
            logger.warning(
                "a tool result wants no reply, but Gemini will answer it anyway; declare "
                "the tools NON_BLOCKING on the Gemini API to keep it silent. Sending it "
                "regardless, since an unanswered call blocks the session.",
                extra={"functions": silenced},
            )

        owned_ctx = llm.ChatContext.empty()
        owned_ctx.items.extend(owned_outputs)
        event = get_tool_results_for_realtime(
            owned_ctx,
            vertexai=self._opts.vertexai,
            tool_response_scheduling=self._opts.tool_response_scheduling,
            supports_silent_scheduling=supports_silent_scheduling,
        )
        if not event:
            return None
        return event, tuple(item.call_id for item in owned_outputs)

    def _register_tool_response(
        self, event: types.LiveClientToolResponse, call_ids: tuple[str, ...]
    ) -> None:
        entry = _ToolResponseOutboxEntry(event=event, call_ids=call_ids)
        self._tool_response_outbox[id(event)] = entry
        self._queue_pending_tool_responses()

    def _queue_pending_tool_responses(self) -> None:
        if self._closed or self._terminal_error is not None or self._msg_ch.closed:
            return

        for entry in self._tool_response_outbox.values():
            if entry.queued_epoch is not None or entry.in_flight_epoch is not None:
                continue

            # Publish ownership before the channel event. A sender can only consume the event
            # after this synchronous method returns to the event loop.
            entry.queued_epoch = self._session_epoch
            if not self._send_client_event(entry.event):
                entry.queued_epoch = None

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        if self._closed:
            fut = asyncio.Future[llm.GenerationCreatedEvent]()
            fut.set_exception(llm.RealtimeError("Gemini Realtime session is closed"))
            return fut
        if self._terminal_error is not None:
            fut = asyncio.Future[llm.GenerationCreatedEvent]()
            fut.set_exception(
                llm.RealtimeError(f"Gemini Realtime session is unavailable: {self._terminal_error}")
            )
            return fut

        if is_given(tools):
            logger.warning("per-response tools is not supported by Google Realtime API, ignoring")
        if not self._realtime_model.capabilities.mutable_chat_context:
            logger.warning(
                f"generate_reply is not compatible with '{self._opts.model}' and will be ignored."
            )
            fut = asyncio.Future[llm.GenerationCreatedEvent]()
            fut.set_exception(
                llm.RealtimeError(f"generate_reply is not compatible with '{self._opts.model}'")
            )
            return fut
        self._reap_cancelled_deferred_manual_input()
        if self._pending_generation_fut and not self._pending_generation_fut.done():
            logger.warning("superseding a pending Gemini generation request")
            pending_fut = self._pending_generation_fut
            self._pending_generation_fut = None
            self._pending_generation_epoch = None
            pending_fut.cancel("Superseded by a newer generate_reply request")
            # Gemini has no request IDs, so the old response must be isolated on an abandoned
            # epoch before a new public future can be registered.
            self._mark_restart_needed()

        if self._deferred_manual_inputs:
            return self._defer_manual_generation(instructions=instructions)

        if self._input_state == _InputState.ABORTED:
            self._input_state = _InputState.IDLE
            self._activity_has_realtime_input = False
            fut = asyncio.Future[llm.GenerationCreatedEvent]()
            fut.set_exception(
                llm.RealtimeError(
                    "Gemini Realtime user input was discarded during a session restart"
                )
            )
            return fut

        self._force_pending_discard_restart()

        if self._input_state == _InputState.INTERRUPT_ONLY:
            # Client content and realtime activities are separate Gemini input protocols. A text
            # turn cannot be completed by ActivityEnd, and an interruption-only activity has no
            # input to generate from. Restart closes that activity before using the legacy
            # application-generation trigger below.
            self._mark_restart_needed()

        if self._input_state == _InputState.AUDIO_ACTIVE and not self._activity_has_realtime_input:
            # An empty/false-positive activity cannot generate from ActivityEnd alone. Discard
            # the unmatched ActivityStart on a fresh epoch, then preserve the legacy application
            # generation behavior (including the placeholder) below.
            self._mark_restart_needed()
            self._input_state = _InputState.IDLE

        fut = self._track_pending_generation()

        # ActivityEnd and client-content turn completion each trigger generation. Use
        # exactly one trigger for the input that is already pending on the provider.
        if self._input_state == _InputState.TEXT_PENDING:
            if is_given(instructions):
                logger.warning(
                    "per-response instructions are not supported when completing pending "
                    "Gemini client content; ignoring instructions"
                )
            # update_chat_ctx sent the finalized user text with turn_complete=False. Complete
            # that same turn without manufacturing another model-visible user message.
            completion_accepted = self._send_input_event(
                types.LiveClientContent(turn_complete=True)
            )
            self._advance_input_sequence()
            if not completion_accepted:
                self._input_state = _InputState.ABORTED
                self._settle_pending_generation(
                    llm.RealtimeError("Gemini Realtime user message completion was not queued")
                )
                return fut
            self._input_state = _InputState.TEXT_TRIGGER_SENT
        elif self._input_state == _InputState.AUDIO_ACTIVE:
            if is_given(instructions):
                logger.warning(
                    "per-response instructions are not supported for an active Gemini audio "
                    "activity; ignoring instructions"
                )
            self._flush_audio_input()
            self._send_input_event(
                types.LiveClientRealtimeInput(
                    activity_end=types.ActivityEnd(),
                )
            )
            self._advance_input_sequence()
            self._input_state = _InputState.AUDIO_TRIGGER_SENT
        else:
            # Gemini requires the last message to end with a user's turn. Keep the placeholder
            # for application/tool/instruction-only generations that have no user input pending.
            turns = []
            if is_given(instructions):
                turns.append(types.Content(parts=[types.Part(text=instructions)], role="model"))
            turns.append(types.Content(parts=[types.Part(text=".")], role="user"))
            self._send_client_event(types.LiveClientContent(turns=turns, turn_complete=True))
            self._input_state = _InputState.LEGACY_TRIGGER_SENT

        self._activity_has_realtime_input = False

        return fut

    def _start_user_activity(self, *, expects_generation: bool) -> None:
        if self._closed or self._terminal_error is not None or not self._manual_activity_detection:
            return
        self._reap_cancelled_deferred_manual_input()
        if (
            expects_generation
            and (
                self._restart_after_provider_turn_epoch == self._session_epoch
                or self._deferred_manual_input_pipeline_active
                or bool(self._deferred_manual_inputs)
            )
            and self._opts.realtime_input_config
            and self._opts.realtime_input_config.activity_handling
            == types.ActivityHandling.NO_INTERRUPTION
        ):
            if not self._deferred_manual_inputs or self._deferred_manual_inputs[-1].sealed:
                self._deferred_manual_inputs.append(_DeferredManualInput())
            deferred = self._deferred_manual_inputs[-1]
            # ActivityStart turns bounded post-discard residue into input owned by this turn.
            # Later frames bypass the residue cap and accumulate on the same transaction.
            self._move_quarantined_manual_inputs(deferred.realtime_inputs)
            deferred.has_realtime_input = bool(deferred.realtime_inputs)
            self._deferred_manual_input_pipeline_active = True
            return
        self._force_pending_discard_restart()

        if expects_generation and self._input_state in (
            _InputState.TEXT_TRIGGER_SENT,
            _InputState.AUDIO_TRIGGER_SENT,
            _InputState.LEGACY_TRIGGER_SENT,
        ):
            # A new manual activity supersedes a generation that has not started yet.
            self._mark_restart_needed(preserve_manual_audio=True)

        if self._input_state == _InputState.ABORTED and expects_generation:
            self._input_state = _InputState.IDLE

        if self._input_state == _InputState.IDLE:
            self._activity_has_realtime_input = False
            self._input_state = (
                _InputState.AUDIO_ACTIVE if expects_generation else _InputState.INTERRUPT_ONLY
            )
            self._send_input_event(
                types.LiveClientRealtimeInput(
                    activity_start=types.ActivityStart(),
                )
            )
            self._manual_audio_quarantine_active = False
            self._replay_quarantined_manual_inputs()
        elif self._input_state == _InputState.INTERRUPT_ONLY and expects_generation:
            # An activity opened only to interrupt output becomes a real user-input activity once
            # the framework reports an external activity boundary.
            self._input_state = _InputState.AUDIO_ACTIVE
        elif self._input_state == _InputState.TEXT_PENDING:
            logger.warning(
                "cannot open a Gemini realtime audio activity while client content is pending"
            )

    def start_user_activity(self) -> None:
        self._start_user_activity(expects_generation=True)

    def interrupt(self) -> None:
        # Gemini Live treats activity start as interruption, so we rely on start_user_activity
        # notifications to handle it
        if (
            self._opts.realtime_input_config
            and self._opts.realtime_input_config.activity_handling
            == types.ActivityHandling.NO_INTERRUPTION
        ):
            return
        self._start_user_activity(expects_generation=False)

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
        if self._closed:
            return
        self._closed = True
        self._discard_deferred_manual_input(cancel_reason="Session closed", all_inputs=True)

        if self._pending_generation_fut is not None:
            # Settle public state before awaiting network/task shutdown, which may block.
            pending_fut = self._pending_generation_fut
            self._pending_generation_fut = None
            self._pending_generation_epoch = None
            if not pending_fut.done():
                pending_fut.cancel("Session closed")

        self._msg_ch.close()
        self._session_should_close.set()
        self._cancel_go_away_deadline()
        self._go_away_restart_epoch = None
        self._session_epoch += 1
        self._input_state = _InputState.IDLE
        self._restart_after_provider_turn_epoch = None
        self._provider_turn_active = False
        self._provider_visible_input_sequence = None
        self._input_send_in_flight_sequence = None
        self._input_send_in_flight = None
        self._delivered_input_event_ids.clear()
        self._pending_text_input_item_id = None
        self._manual_audio_quarantine_active = False
        self._activity_has_realtime_input = False
        self._clear_local_audio_input()
        self._input_event_sequences.clear()
        self._invalid_input_event_sequences.clear()
        self._clear_provider_tool_state()
        self._provider_session_established = False

        if self._main_atask:
            await utils.aio.cancel_and_wait(self._main_atask)

        await self._close_active_session()

        for fut in self._response_created_futures.values():
            if not fut.done():
                fut.set_exception(llm.RealtimeError("Session closed before response created"))
        self._response_created_futures.clear()

        if self._current_generation:
            self._mark_current_generation_done()

        # release the genai http clients owned by this session. Without this
        # they stay open until the garbage collector runs `AsyncClient.__del__`,
        # which schedules `aclose()` on whatever event loop happens to be
        # running at that moment.
        try:
            await self._client.aio.aclose()
        except Exception:
            logger.warning("failed to close the genai client", exc_info=True)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        max_retries = self._opts.conn_options.max_retry

        while not self._msg_ch.closed:
            # previous session might not be closed yet, we'll do it here.
            await self._close_active_session()
            if self._closed:
                break

            self._session_should_close.clear()
            connect_epoch = self._session_epoch
            connect_msg_ch = self._msg_ch
            config = self._build_connect_config()
            resuming_session = self._session_resumption_handle is not None
            session = None
            try:
                logger.debug("connecting to Gemini Realtime API...")
                t0 = time.perf_counter()
                async with self._client.aio.live.connect(
                    model=self._opts.model, config=config
                ) as session:
                    self._report_connection_acquired(time.perf_counter() - t0)
                    async with self._session_lock:
                        if (
                            self._closed
                            or self._session_should_close.is_set()
                            or connect_epoch != self._session_epoch
                            or connect_msg_ch is not self._msg_ch
                        ):
                            continue
                        self._active_session = session
                        session_epoch = connect_epoch

                        # Check for system/developer messages in initial chat context
                        system_msg_count = sum(
                            1
                            for msg in self._chat_ctx.messages()
                            if msg.role in ("system", "developer")
                        )
                        if system_msg_count > 0:
                            logger.warning(
                                f"Gemini Realtime model '{self._opts.model}' does not support 'system' or "
                                f"'developer' roles in chat history. Dropping {system_msg_count} system "
                                f"message(s) from initial chat context during session initialization. "
                                f"Gemini Realtime only supports 'user' and 'model' roles. Use "
                                f"update_instructions() to set system-level context instead."
                            )

                        turns_dict, _ = self._chat_ctx.copy(
                            exclude_function_call=True,
                            exclude_handoff=True,
                            exclude_instructions=True,
                            exclude_empty_message=True,
                            exclude_config_update=True,
                        ).to_provider_format(format="google", inject_dummy_user_message=False)
                        turns = [types.Content.model_validate(turn) for turn in turns_dict]
                        if turns and not resuming_session:
                            input_sequence = (
                                self._input_event_sequence
                                if self._input_state == _InputState.TEXT_PENDING
                                else None
                            )
                            if input_sequence is not None:
                                self._input_send_in_flight_sequence = input_sequence
                            try:
                                await session.send_client_content(
                                    turns=turns,  # type: ignore
                                    turn_complete=False,
                                )
                            finally:
                                if self._input_send_in_flight_sequence == input_sequence:
                                    self._input_send_in_flight_sequence = None
                            if (
                                input_sequence is not None
                                and input_sequence not in self._invalid_input_event_sequences
                                and connect_epoch == self._session_epoch
                            ):
                                self._provider_visible_input_sequence = input_sequence

                        # The provider is ready to accept outputs for tool calls this session
                        # has observed; historical or otherwise unowned outputs remain local.
                        self._provider_session_established = True
                        self._queue_pending_tool_responses()

                    connection_tasks: list[asyncio.Task[object]] = []
                    try:
                        send_task = asyncio.create_task(
                            self._send_task(session, session_epoch, connect_msg_ch),
                            name="gemini-realtime-send",
                        )
                        recv_task = asyncio.create_task(
                            self._recv_task(session, session_epoch), name="gemini-realtime-recv"
                        )
                        restart_wait_task = asyncio.create_task(
                            self._session_should_close.wait(), name="gemini-restart-wait"
                        )
                        connection_tasks.extend((send_task, recv_task, restart_wait_task))

                        done, _ = await asyncio.wait(
                            connection_tasks,
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        for task in done:
                            if task is not restart_wait_task and task.exception():
                                logger.error(f"error in task {task.get_name()}: {task.exception()}")
                                raise task.exception() or Exception(f"{task.get_name()} failed")

                        if restart_wait_task not in done and self._msg_ch.closed:
                            break
                    finally:
                        await utils.aio.cancel_and_wait(*connection_tasks)

                    # the recv/send tasks signal restart by setting _session_should_close
                    # rather than raising. propagate any error they recorded so the handler
                    # below can bound retries and surface it through the "error" event.
                    if self._session_error is not None:
                        err = self._session_error
                        self._session_error = None
                        raise err

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Provide a hint for 1008 errors (often model/API mismatch for unknown models)
                hint = _get_1008_error_hint(str(e))
                if hint:
                    logger.error(f"Gemini Realtime API error: {e}{hint}", exc_info=e)
                else:
                    logger.error(f"Gemini Realtime API error: {e}", exc_info=e)

                if not self._msg_ch.closed:
                    # Gemini Live closes with 1007 ("Request contains an invalid argument")
                    # when the session context is exhausted. Reconnecting replays the same
                    # oversized chat context and fails identically, producing a tight retry
                    # loop, so treat it as fatal to the session instead of retrying.
                    if getattr(e, "code", None) == 1007 or "1007" in str(e):
                        logger.error(
                            "Gemini Live closed the session: context exhausted (1007). "
                            "Reconnecting would replay the same context and fail again; "
                            "terminating the session.",
                            exc_info=e,
                        )
                        self._emit_error(e, recoverable=False)
                        self._set_terminal_error(
                            llm.RealtimeError("Gemini Live session context exhausted (1007)")
                        )
                        return

                    # we shouldn't retry when it's not connected, usually this means incorrect
                    # parameters or setup
                    if not session or max_retries == 0:
                        self._emit_error(e, recoverable=False)
                        error_msg = "Failed to connect to Gemini Live"
                        if hint:
                            error_msg += hint
                        self._set_terminal_error(llm.RealtimeError(error_msg))
                        return

                    if self._num_retries == max_retries:
                        self._emit_error(e, recoverable=False)
                        error_msg = f"Failed to connect to Gemini Live after {max_retries} attempts"
                        if hint:
                            error_msg += hint
                        self._set_terminal_error(llm.RealtimeError(error_msg))
                        return

                    self._emit_error(e, recoverable=True)
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

    async def _send_message(self, session: AsyncSession, msg: ClientEvents) -> None:
        if isinstance(msg, types.LiveClientContent):
            await session.send_client_content(
                turns=msg.turns,  # type: ignore
                turn_complete=msg.turn_complete if msg.turn_complete is not None else True,
            )
        elif isinstance(msg, types.LiveClientToolResponse) and msg.function_responses:
            await session.send_tool_response(function_responses=msg.function_responses)
        elif isinstance(msg, types.LiveClientRealtimeInput):
            if msg.audio:
                await session.send_realtime_input(audio=msg.audio)
            elif msg.video:
                await session.send_realtime_input(video=msg.video)
            elif msg.text:
                await session.send_realtime_input(text=msg.text)
            elif msg.activity_start:
                await session.send_realtime_input(activity_start=msg.activity_start)
            elif msg.activity_end:
                await session.send_realtime_input(activity_end=msg.activity_end)
        else:
            logger.warning(f"Warning: Received unhandled message type: {type(msg)}")

    async def _send_task(
        self,
        session: AsyncSession,
        session_epoch: int,
        msg_ch: utils.aio.Chan[ClientEvents] | None = None,
    ) -> None:
        msg_ch = self._msg_ch if msg_ch is None else msg_ch
        try:
            while True:
                async with self._session_lock:
                    if (
                        self._session_should_close.is_set()
                        or not self._active_session
                        or self._active_session != session
                        or session_epoch != self._session_epoch
                        or msg_ch is not self._msg_ch
                    ):
                        break
                try:
                    msg = await msg_ch.recv()
                except utils.aio.ChanClosed:
                    break

                if id(msg) in self._delivered_input_event_ids:
                    self._delivered_input_event_ids.discard(id(msg))
                    self._input_event_sequences.pop(id(msg), None)
                    continue

                if id(msg) in self._delivered_tool_response_event_ids:
                    self._delivered_tool_response_event_ids.discard(id(msg))
                    continue

                tool_response_entry = self._tool_response_outbox.get(id(msg))
                if tool_response_entry is not None:
                    if tool_response_entry.queued_epoch != session_epoch:
                        # A stale sender found an event whose logical response is owned by a
                        # different epoch. Its authoritative queue entry will deliver it.
                        continue
                    assert isinstance(msg, types.LiveClientToolResponse)
                    retained_responses = [
                        (call_id, response)
                        for call_id, response in zip(
                            tool_response_entry.call_ids,
                            msg.function_responses or [],
                            strict=True,
                        )
                        if call_id in self._provider_tool_call_ids
                    ]
                    if len(retained_responses) != len(tool_response_entry.call_ids):
                        if not retained_responses:
                            self._tool_response_outbox.pop(id(msg), None)
                            tool_response_entry.queued_epoch = None
                            continue
                        tool_response_entry.call_ids = tuple(
                            call_id for call_id, _ in retained_responses
                        )
                        msg.function_responses = [response for _, response in retained_responses]
                    tool_response_entry.queued_epoch = None
                    tool_response_entry.in_flight_epoch = session_epoch

                input_send_in_flight: _InputSendInFlight | None = None
                input_sequence = self._input_event_sequences.pop(id(msg), None)
                if input_sequence is not None:
                    input_send_in_flight = _InputSendInFlight(
                        event=msg, sequence=input_sequence, epoch=session_epoch
                    )
                    self._input_send_in_flight = input_send_in_flight
                    self._input_send_in_flight_sequence = input_sequence
                try:
                    if input_sequence in self._invalid_input_event_sequences:
                        continue

                    async with self._session_lock:
                        if (
                            self._session_should_close.is_set()
                            or not self._active_session
                            or self._active_session != session
                            or session_epoch != self._session_epoch
                            or msg_ch is not self._msg_ch
                        ):
                            break
                        if input_sequence in self._invalid_input_event_sequences:
                            continue

                    await self._send_message(session, msg)
                    if (
                        input_send_in_flight is not None
                        and self._input_send_in_flight is input_send_in_flight
                        and input_send_in_flight.migrated_epoch is not None
                    ):
                        self._delivered_input_event_ids.add(id(msg))

                    if tool_response_entry is not None:
                        # Awaiting the SDK call only confirms the websocket send, but it is the
                        # strongest available commit point. Commit only if this provider lineage
                        # still owns the entry; a fresh restart may have abandoned it meanwhile.
                        if self._tool_response_outbox.get(id(msg)) is tool_response_entry:
                            if tool_response_entry.queued_epoch is not None:
                                self._delivered_tool_response_event_ids.add(id(msg))
                            self._tool_response_outbox.pop(id(msg), None)
                            self._provider_tool_call_ids.difference_update(
                                tool_response_entry.call_ids
                            )
                            self._settled_provider_tool_call_ids.update(
                                tool_response_entry.call_ids
                            )
                        tool_response_entry.in_flight_epoch = None

                    async with self._session_lock:
                        authoritative = (
                            not self._session_should_close.is_set()
                            and self._active_session == session
                            and session_epoch == self._session_epoch
                            and msg_ch is self._msg_ch
                        )
                        if (
                            authoritative
                            and input_sequence is not None
                            and input_sequence not in self._invalid_input_event_sequences
                        ):
                            self._provider_visible_input_sequence = input_sequence
                except asyncio.CancelledError:
                    if (
                        tool_response_entry is not None
                        and self._tool_response_outbox.get(id(msg)) is tool_response_entry
                        and tool_response_entry.in_flight_epoch == session_epoch
                    ):
                        tool_response_entry.in_flight_epoch = None
                        self._queue_pending_tool_responses()
                    raise
                finally:
                    if self._input_send_in_flight is input_send_in_flight:
                        self._input_send_in_flight = None

                    if self._input_send_in_flight_sequence == input_sequence:
                        self._input_send_in_flight_sequence = None
                    if input_sequence is not None:
                        self._release_invalid_input_sequence(input_sequence)

                if lk_google_debug and isinstance(
                    msg,
                    (
                        types.LiveClientContent,
                        types.LiveClientToolResponse,
                        types.LiveClientRealtimeInput,
                    ),
                ):
                    if not isinstance(msg, types.LiveClientRealtimeInput) or not (
                        msg.audio or msg.video or msg.text
                    ):
                        logger.debug(
                            f">>> sent {type(msg).__name__}",
                            extra={"content": msg.model_dump(exclude_defaults=True)},
                        )

        except Exception as e:
            if not self._session_should_close.is_set():
                logger.error(f"error in send task: {e}", exc_info=e)
                self._session_error = e
                self._mark_restart_needed(
                    on_error=True, resume_session=bool(self._provider_tool_call_ids)
                )
        finally:
            logger.debug("send task finished.")

    async def _recv_task(self, session: AsyncSession, session_epoch: int) -> None:
        try:
            while True:
                async with self._session_lock:
                    if (
                        self._session_should_close.is_set()
                        or not self._active_session
                        or self._active_session != session
                        or session_epoch != self._session_epoch
                    ):
                        logger.debug("receive task: Session changed or closed, stopping receive.")
                        break

                async for response in session.receive():
                    async with self._session_lock:
                        if (
                            self._session_should_close.is_set()
                            or not self._active_session
                            or self._active_session != session
                            or session_epoch != self._session_epoch
                        ):
                            logger.debug(
                                "receive task: ignoring response from an abandoned session epoch"
                            )
                            break

                    if lk_google_debug:
                        resp_copy = response.model_dump(exclude_defaults=True)
                        # remove audio from debugging logs
                        if (
                            (sc := resp_copy.get("server_content"))
                            and (mt := sc.get("model_turn"))
                            and (parts := mt.get("parts"))
                        ):
                            for part in parts:
                                if part and part.get("inline_data"):
                                    part["inline_data"] = "<audio>"
                        logger.debug("<<< received response", extra={"response": resp_copy})

                    if response.tool_call and self._opts.tool_choice == "none":
                        # reject without opening a generation, so the pending generate_reply
                        # stays bound to the model's eventual reply and tools stay suppressed
                        # for the whole turn.
                        self._reject_tool_calls(response.tool_call.function_calls or [])
                        continue

                    if not self._current_generation or self._current_generation._done:
                        if (sc := response.server_content) and sc.interrupted:
                            # two cases an interrupted event is sent without an active generation
                            # 1) the generation is done but playout is not finished (turn_complete -> interrupted)
                            # 2) the generation is not started (interrupted -> turn_complete)
                            # for both cases, we interrupt the agent if there is no pending generation from `generate_reply`
                            # for the second case, the pending generation will be stopped by `turn_complete` event coming later
                            if not self._pending_generation_fut:
                                self._handle_input_speech_started()

                            sc.interrupted = None
                            sc_copy = sc.model_dump(exclude_none=True)
                            if not sc_copy:
                                # ignore empty server content
                                response.server_content = None
                                if lk_google_debug:
                                    logger.debug("ignoring empty server content")

                        if self._is_new_generation(response):
                            self._start_new_generation(session_epoch)
                            if lk_google_debug:
                                logger.debug(f"new generation started: {self._current_generation}")

                    if update := response.session_resumption_update:
                        if update.resumable is False:
                            self._session_resumption_handle = None
                        elif update.resumable and update.new_handle:
                            self._session_resumption_handle = update.new_handle

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

        except Exception as e:
            if not self._session_should_close.is_set():
                logger.error(f"error in receive task: {e}", exc_info=e)
                self._session_error = e
                self._mark_restart_needed(on_error=True, resume_session=True)
        finally:
            if session_epoch == self._session_epoch:
                self._mark_current_generation_done()
                self._finish_provider_turn(session_epoch)

    def _build_connect_config(self) -> types.LiveConnectConfig:
        temp = self._opts.temperature if is_given(self._opts.temperature) else None

        tools_config, _ = create_tools_config(
            self._tools,
            tool_behavior=self._opts.tool_behavior,
            use_parameters_json_schema=False,
        )
        conf = types.LiveConnectConfig(
            response_modalities=self._opts.response_modalities,
            history_config=types.HistoryConfig(initial_history_in_client_content=True)
            if not self._realtime_model.capabilities.mutable_chat_context
            else None,
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
                thinking_config=self._opts.thinking_config
                if is_given(self._opts.thinking_config)
                else None,
                media_resolution=self._opts.media_resolution
                if is_given(self._opts.media_resolution)
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
            tools=tools_config,
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

    def _acknowledge_provider_input(self) -> None:
        input_state = self._input_state
        # Provider input ownership is sequence-scoped, independently of the generation's logical
        # turn. Committing audio/text advances the sequence first, so an exact current marker
        # belongs to later input that this generation cannot consume.
        preserve_provider_input = (
            self._provider_visible_input_sequence == self._input_event_sequence
        )
        preserve_logical_input = input_state in (
            _InputState.AUDIO_ACTIVE,
            _InputState.INTERRUPT_ONLY,
            _InputState.TEXT_PENDING,
            _InputState.ABORTED,
        )
        if not preserve_logical_input:
            self._input_state = _InputState.IDLE
            self._activity_has_realtime_input = False
            self._pending_text_input_item_id = None
        if not preserve_provider_input:
            self._provider_visible_input_sequence = None
        self._provider_turn_active = True

    def _start_new_generation(self, session_epoch: int | None = None) -> None:
        session_epoch = self._session_epoch if session_epoch is None else session_epoch
        if session_epoch != self._session_epoch:
            return
        self._acknowledge_provider_input()
        self._rejected_tool_calls = 0
        if self._current_generation and not self._current_generation._done:
            logger.warning("starting new generation while another is active. Finalizing previous.")
            self._mark_current_generation_done()
            if session_epoch != self._session_epoch:
                return

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
            response_id=self._current_generation.response_id,
        )

        if (
            self._pending_generation_fut
            and not self._pending_generation_fut.done()
            and self._pending_generation_epoch == session_epoch
        ):
            generation_event.user_initiated = True
            self._pending_generation_fut.set_result(generation_event)
            self._pending_generation_fut = None
            self._pending_generation_epoch = None
        else:
            # emit input_speech_started event before starting an agent initiated generation
            # to interrupt the previous audio playout if any
            self._handle_input_speech_started()

        self.emit("generation_created", generation_event)

    def _handle_server_content(self, server_content: types.LiveServerContent) -> None:
        current_gen = self._current_generation
        if not current_gen:
            if self._rejected_tool_calls:
                logger.debug(
                    "ignoring server content from a rejected tool call turn",
                    extra={"server_content": server_content.model_dump_json(exclude_none=True)},
                )
            else:
                logger.warning("received server content but no active generation.")
            if server_content.turn_complete:
                self._finish_provider_turn()
            return

        if model_turn := server_content.model_turn:
            for part in model_turn.parts or []:
                if part.thought:
                    # bypass reasoning output
                    continue
                if part.text:
                    current_gen.push_text(part.text)
                if part.inline_data:
                    if current_gen.audio_ch.closed:
                        # generation_complete already closed the audio stream; a turn
                        # should not emit more audio, so drop any late frame
                        if not current_gen._extra_content_warned:
                            current_gen._extra_content_warned = True
                            logger.warning(
                                "Gemini sent audio after generation completed; dropping it"
                            )
                        continue
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

        # gemini delays turn_complete until it thinks client-side playback finished, so end
        # the output streams on generation_complete instead
        if server_content.generation_complete:
            self._close_output_streams(current_gen)

        if server_content.interrupted and not self._pending_generation_fut:
            # interrupt agent if there is no pending user initiated generation
            self._handle_input_speech_started()

        if server_content.turn_complete:
            self._mark_current_generation_done()
            self._finish_provider_turn()

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

        self._close_output_streams(gen)

        gen.function_ch.close()
        gen.message_ch.close()
        gen._done = True
        if lk_google_debug:
            logger.debug(f"generation done {gen}")

    def _finish_provider_turn(self, session_epoch: int | None = None) -> None:
        session_epoch = self._session_epoch if session_epoch is None else session_epoch
        if session_epoch != self._session_epoch:
            return

        self._provider_turn_active = False
        if self._restart_after_provider_turn_epoch == session_epoch:
            self._restart_after_provider_turn_epoch = None
            self._mark_restart_needed(preserve_manual_audio=True)
            return

        if self._go_away_restart_epoch == session_epoch and self._input_state == _InputState.IDLE:
            # generation_created acknowledged the input and the provider has now completed every
            # output/tool continuation it owns. This is Gemini's safe resumption boundary.
            if not self._mark_restart_needed(resume_session=True):
                return

        if self._deferred_manual_input_pipeline_active or self._deferred_manual_inputs:
            self._activate_next_deferred_manual_input()

    def _close_output_streams(self, gen: _ResponseGeneration) -> None:
        # ends the audio segment and finalizes the output transcript. called on
        # generation_complete (audio/text are done by then) and again at final teardown.
        if not gen.text_ch.closed:
            if self._opts.output_audio_transcription is None:
                # close the text data of transcription synchronizer
                gen.text_ch.send_nowait("")
            gen.text_ch.close()
        if not gen.audio_ch.closed:
            gen.audio_ch.close()

    def _handle_input_speech_started(self) -> None:
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _handle_input_speech_stopped(self) -> None:
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(user_transcription_enabled=False),
        )

    def _reject_tool_calls(self, function_calls: list[types.FunctionCall]) -> None:
        if not function_calls:
            return

        self._acknowledge_provider_input()
        self._rejected_tool_calls += 1
        extra = {"functions": [fnc_call.name for fnc_call in function_calls]}
        if self._rejected_tool_calls > MAX_TOOL_CALL_REJECTIONS:
            # stop responding to break the loop; the user can still interrupt by voice
            if self._rejected_tool_calls == MAX_TOOL_CALL_REJECTIONS + 1:
                logger.error(
                    "model keeps calling tools despite tool_choice='none'; "
                    f"stopping after {MAX_TOOL_CALL_REJECTIONS} rejections to avoid a loop",
                    extra=extra,
                )
            return

        logger.warning("rejecting tool call requested while tool_choice='none'", extra=extra)
        outputs = [
            llm.FunctionCallOutput(
                name=fnc_call.name or "",
                call_id=fnc_call.id or "",
                output="Tool calls are disabled for this turn, respond to the user directly.",
                is_error=True,
            )
            for fnc_call in function_calls
        ]
        call_ids = tuple(output.call_id for output in outputs)
        self._provider_tool_call_ids.update(call_ids)
        self._settled_provider_tool_call_ids.difference_update(call_ids)
        responses = [
            create_function_response(
                output,
                vertexai=self._opts.vertexai,
                tool_response_scheduling=self._opts.tool_response_scheduling,
            )
            for output in outputs
        ]
        self._register_tool_response(
            types.LiveClientToolResponse(function_responses=responses), call_ids
        )

    def _handle_tool_calls(self, tool_call: types.LiveServerToolCall) -> None:
        if not self._current_generation:
            logger.warning("received tool call but no active generation.")
            return

        gen = self._current_generation
        function_calls = tool_call.function_calls or []

        for fnc_call in function_calls:
            call_id = fnc_call.id or utils.shortuuid("fnc-call-")
            self._provider_tool_call_ids.add(call_id)
            self._settled_provider_tool_call_ids.discard(call_id)
            arguments = json.dumps(fnc_call.args)

            gen.function_ch.send_nowait(
                llm.FunctionCall(
                    call_id=call_id,
                    name=fnc_call.name,
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
        cancelled_call_ids = set(tool_call_cancellation.ids or [])
        self._provider_tool_call_ids.difference_update(cancelled_call_ids)
        self._settled_provider_tool_call_ids.update(cancelled_call_ids)

    def _handle_usage_metadata(self, usage_metadata: types.UsageMetadata) -> None:
        current_gen = self._current_generation
        if not current_gen:
            if self._rejected_tool_calls:
                logger.debug("ignoring usage metadata from a rejected tool call turn")
            else:
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
            label=self._realtime_model.label,
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
            metadata=Metadata(
                model_name=self._realtime_model.model, model_provider=self._realtime_model.provider
            ),
        )
        self.emit("metrics_collected", metrics)

    def _handle_go_away(self, go_away: types.LiveServerGoAway) -> None:
        logger.warning(
            f"Gemini server indicates disconnection soon. Time left: {go_away.time_left}"
        )
        if self._closed or self._terminal_error is not None:
            return

        if not self._provider_turn_active and self._input_state == _InputState.IDLE:
            resumed = self._mark_restart_needed(resume_session=True)
            if resumed and (
                self._deferred_manual_input_pipeline_active or self._deferred_manual_inputs
            ):
                self._activate_next_deferred_manual_input()
            return

        self._schedule_go_away_restart(go_away.time_left)

    def commit_audio(self) -> None:
        logger.warning("commit_audio is not supported by Gemini Realtime API.")

    def clear_audio(self) -> None:
        if self._closed or self._terminal_error is not None:
            return

        if self._manual_activity_detection:
            self._reap_cancelled_deferred_manual_input()
            self._discard_deferred_manual_input(
                error=llm.RealtimeError(
                    "Gemini Realtime user input was discarded before generation started"
                )
            )
            discarded_sequence = self._advance_input_sequence(invalidate=True)
            provider_input_pending = self._input_may_be_provider_visible(discarded_sequence)
            if self._input_state == _InputState.TEXT_PENDING:
                self._discard_pending_text_input()
            self._clear_local_audio_input()
            self._activity_has_realtime_input = False
            if self._provider_visible_input_sequence == discarded_sequence:
                self._provider_visible_input_sequence = None
            self._release_invalid_input_sequence(discarded_sequence)

            if not provider_input_pending:
                if self._input_state in (
                    _InputState.AUDIO_ACTIVE,
                    _InputState.INTERRUPT_ONLY,
                    _InputState.TEXT_PENDING,
                ):
                    self._input_state = _InputState.IDLE
                return

            self._manual_audio_quarantine_active = True
            # Gemini has no in-session buffer-clear/cancel-activity event. Restarting is the
            # only way to guarantee an abandoned manual turn cannot leak into the next one.
            self._input_state = _InputState.ABORTED
            if self._provider_turn_active:
                # The abandoned input belongs to this provider epoch, but its active output does
                # not. Wait for the provider turn, including any tool calls, to finish.
                self._restart_after_provider_turn_epoch = self._session_epoch
                return

            self._mark_restart_needed(preserve_manual_audio=True)
        else:
            logger.warning("clear_audio is not supported by Gemini Realtime API.")

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

    def _is_new_generation(self, resp: types.LiveServerMessage) -> bool:
        if resp.tool_call:
            return True

        if (sc := resp.server_content) and (
            sc.model_turn
            or (
                sc.output_transcription and sc.output_transcription and sc.output_transcription.text
            )
            or (sc.input_transcription and sc.input_transcription and sc.input_transcription.text)
            # or (sc.generation_complete is not None)
            # or (sc.turn_complete is not None)
        ):
            # Some Gemini models send a `generation_complete` event after tool calls, but others do not.
            # We mark the generation as done after a tool call and need to ignore any empty transcriptions or generation_complete events.
            # This prevents new empty generations from starting and interrupting tool execution.
            return True

        return False
