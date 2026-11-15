from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import google.auth.credentials
from google.genai import types
from livekit import rtc
from livekit.agents import APIError, LanguageCode, TokenUsage, llm, utils
from livekit.agents.llm import realtime as llm_realtime
from livekit.agents.metrics import Metadata, RealtimeModelMetrics
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given, media as images

from ..log import logger
from ..utils import (
    GeminiBackend,
    create_function_response,
    create_tools_config,
    gemini_backend,
    get_tool_results_for_realtime,
)
from .api_proto import (
    ClientMessage,
    LiveAPIModels,
    LiveEndpoint,
    ModalityTokenCount,
    ServerContent,
    ServerMessage,
    ToolCall,
    UsageMetadata,
    Voice,
    dump,
)

INPUT_AUDIO_SAMPLE_RATE = 16000
INPUT_AUDIO_CHANNELS = 1
OUTPUT_AUDIO_SAMPLE_RATE = 24000
OUTPUT_AUDIO_CHANNELS = 1

DEFAULT_IMAGE_ENCODE_OPTIONS = images.EncodeOptions(
    format="JPEG",
    quality=75,
    resize_options=images.ResizeOptions(width=1024, height=1024, strategy="scale_aspect_fit"),
)

# stop rejecting tool calls after this many in a row to avoid a loop (tool_choice="none")
MAX_TOOL_CALL_REJECTIONS = 3

# Known VertexAI models for the Live API
# See: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api
KNOWN_VERTEXAI_MODELS: frozenset[str] = frozenset(
    {
        "gemini-3.8-live",
        "gemini-live-2.5-flash-native-audio",
    }
)

# Known Gemini API models for the Live API
# See: https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-live
KNOWN_GEMINI_API_MODELS: frozenset[str] = frozenset(
    {
        "gemini-3.8-live",
        "gemini-3.8-live-extended-thinking",
        "gemini-3.1-flash-live-preview",
        "gemini-2.5-flash-native-audio-preview-12-2025",
    }
)


# generate_reply() appends a "." user turn so Gemini sees a completed turn. These models
# answer that placeholder with an empty turn instead, so they must not get it.
MODELS_WITHOUT_REPLY_PLACEHOLDER: tuple[str, ...] = ("3.1", "3.8")


def _needs_reply_placeholder(model: str) -> bool:
    return not any(tag in model for tag in MODELS_WITHOUT_REPLY_PLACEHOLDER)


# These models declare tools NON_BLOCKING unless the client says otherwise. Sending nothing
# would leave the server async while we still treat the tools as blocking.
MODELS_DEFAULT_NON_BLOCKING: tuple[str, ...] = ("3.8",)


def _default_tool_behavior(model: str) -> NotGivenOr[types.Behavior]:
    if any(tag in model for tag in MODELS_DEFAULT_NON_BLOCKING):
        return types.Behavior.NON_BLOCKING
    return NOT_GIVEN


def _warn_model_api_mismatch(model: str, use_vertexai: bool) -> None:
    """Warn when a known model may not be available on the selected API."""
    if use_vertexai and model in KNOWN_GEMINI_API_MODELS and model not in KNOWN_VERTEXAI_MODELS:
        logger.warning(
            f"Model '{model}' may not be available on VertexAI (vertexai=True). "
            "If the connection fails, use a VertexAI model "
            "(e.g., 'gemini-live-2.5-flash-native-audio') or set vertexai=False."
        )

    if not use_vertexai and model in KNOWN_VERTEXAI_MODELS and model not in KNOWN_GEMINI_API_MODELS:
        logger.warning(
            f"Model '{model}' may not be available on the Gemini API (vertexai=False). "
            "If the connection fails, use a Gemini API model "
            "(e.g., 'gemini-2.5-flash-native-audio-preview-12-2025') or set vertexai=True."
        )


def _warn_vertex_scheduling_unsupported() -> None:
    logger.warning(
        "tool_response_scheduling is not supported by Vertex AI and will be ignored; "
        "tool responses use the default scheduling there."
    )


@dataclass
class _RealtimeOptions:
    model: LiveAPIModels | str
    backend: GeminiBackend
    voice: Voice | str
    language: NotGivenOr[LanguageCode]
    response_modalities: list[types.Modality]
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

    created_timestamp: float = field(default_factory=time.time)
    """The timestamp when the generation is created"""
    completed_timestamp: float | None = None
    """The timestamp when the generation is completed"""
    done: bool = False
    """Whether the generation is done (set when the turn is complete)"""
    extra_content_warned: bool = False
    """Whether we've warned about audio/text arriving after generation completed"""

    def push_text(self, text: str) -> None:
        if self.text_ch.closed:
            # generation_complete already finalized the output; a turn should not emit
            # more text, so drop it (see _handle_server_content)
            if not self.extra_content_warned:
                self.extra_content_warned = True
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
        credentials: NotGivenOr[google.auth.credentials.Credentials] = NOT_GIVEN,
        credentials_info: NotGivenOr[Mapping[str, object]] = NOT_GIVEN,
        credentials_file: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Initializes a RealtimeModel instance for interacting with Google's Realtime API.

        Args:
            instructions (str, optional): Initial system instructions for the model. Defaults to "".
            api_key (str, optional): Gemini API key. Defaults to the `GOOGLE_API_KEY` environment variable.
            modalities (list[Modality], optional): Modalities to use, such as ["TEXT", "AUDIO"]. Defaults to ["AUDIO"].
            model (str, optional): The name of the model to use. Defaults to "gemini-2.5-flash-native-audio-preview-12-2025" or "gemini-live-2.5-flash-native-audio" (vertexai).
            voice (api_proto.Voice, optional): Voice setting for audio outputs. Defaults to "Puck".
            language (str, optional): The language(BCP-47 Code) to use for the API. supported languages - https://ai.google.dev/gemini-api/docs/live#supported-languages
            temperature (float, optional): Sampling temperature for response generation. Defaults to 0.8.
            vertexai (bool, optional): Whether to use Vertex AI. Defaults to `GOOGLE_GENAI_USE_VERTEXAI`, else to Vertex AI when credentials, a project or a location are given.
            project (str, optional): Google Cloud project for Vertex AI. Defaults to `GOOGLE_CLOUD_PROJECT`, then the credentials' project.
            location (str, optional): Google Cloud region for Vertex AI. Defaults to `GOOGLE_CLOUD_LOCATION`, then "us-central1".
            credentials (Credentials, optional): Credentials for Vertex AI.
            credentials_info (dict, optional): Service account info for Vertex AI.
            credentials_file (str, optional): Path to a service account JSON file for Vertex AI.
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
            tool_behavior (Behavior, optional): The behavior for tool call. Defaults to NON_BLOCKING on models that declare it by default (Gemini 3.8 Live), and to the server default (BLOCKING) elsewhere.
            tool_response_scheduling (FunctionResponseScheduling, optional): The scheduling for tool response. Default scheduling is WHEN_IDLE.
            session_resumption (SessionResumptionConfig, optional): The configuration for session resumption. Defaults to None.
            thinking_config (ThinkingConfig, optional): Native audio thinking configuration.
                thinking_level is not supported by gemini-3.8-live on the Gemini API.
            conn_options (APIConnectOptions, optional): The configuration for the API connection. Defaults to DEFAULT_API_CONNECT_OPTIONS.

        Raises:
            ValueError: If the API key or VertexAI project is missing, or thinking_level is unsupported.
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
        backend = gemini_backend(
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
            credentials=credentials,
            credentials_info=credentials_info,
            credentials_file=credentials_file,
            default_location="us-central1",
        )
        use_vertexai = backend.vertexai
        if use_vertexai and is_given(tool_response_scheduling):
            _warn_vertex_scheduling_unsupported()
        if not is_given(model):
            model = (
                "gemini-live-2.5-flash-native-audio"
                if use_vertexai
                else "gemini-2.5-flash-native-audio-preview-12-2025"
            )

        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=server_turn_detection,
                user_transcription=input_audio_transcription is not None,
                auto_tool_reply_generation=True,
                audio_output=types.Modality.AUDIO in modalities,
                manual_function_calls=False,
                mutable_chat_context=True,
                mutable_instructions=True,
                mutable_tools=False,
                per_response_tool_choice=False,
            )
        )

        _warn_model_api_mismatch(model, use_vertexai)

        if (
            not use_vertexai
            and model.removeprefix("models/") == "gemini-3.8-live"
            and is_given(thinking_config)
            and thinking_config.thinking_level is not None
        ):
            raise ValueError(
                f"Model '{model}' does not support thinking_level on the Gemini API. "
                "Omit thinking_level or use 'gemini-3.8-live-extended-thinking'."
            )

        self._opts = _RealtimeOptions(
            model=model,
            backend=backend,
            voice=voice,
            response_modalities=modalities,
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
            tool_behavior=tool_behavior
            if is_given(tool_behavior)
            else _default_tool_behavior(model),
            tool_response_scheduling=tool_response_scheduling,
            conn_options=conn_options,
            http_options=http_options,
            media_resolution=media_resolution,
            thinking_config=thinking_config,
            session_resumption=session_resumption,
        )

        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        if self._opts.backend.vertexai:
            return "Vertex AI"
        else:
            return "Gemini"

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        # Gemini drives manual turns via activity_start/activity_end, not commit_audio/clear_audio,
        # so the pipeline can't gatekeep turns yet; keep can_disable_turn_detection=False for now
        sess = RealtimeSession(self, options=self._opts)
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


def _by_modality(details: list[ModalityTokenCount] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for detail in details or []:
        if (modality := detail.get("modality")) and (count := detail.get("tokenCount")) is not None:
            counts[modality.lower()] = counts.get(modality.lower(), 0) + count
    return counts


class RealtimeSession(llm.RealtimeWebSocketSession[ServerMessage, ClientMessage]):
    """A Gemini Live session. Each connection opens with its setup; the conversation follows
    once Gemini completed it, resumed from the last resumption handle when there is one, else
    replayed from the chat context."""

    frame_ms = 50

    def __init__(self, realtime_model: RealtimeModel, *, options: _RealtimeOptions) -> None:
        super().__init__(
            realtime_model,
            conn_options=options.conn_options,
            input_sample_rate=INPUT_AUDIO_SAMPLE_RATE,
            input_num_channels=INPUT_AUDIO_CHANNELS,
        )
        self._opts = options
        self._tools = llm.ToolContext.empty()
        self._chat_ctx = llm.ChatContext.empty()

        api_version = self._opts.api_version if is_given(self._opts.api_version) else None
        if (
            not api_version
            and (self._opts.enable_affective_dialog or self._opts.proactivity)
            and not self._opts.backend.vertexai
        ):
            api_version = "v1alpha"
        self._endpoint = LiveEndpoint(
            vertexai=self._opts.backend.vertexai,
            api_key=self._opts.backend.api_key,
            project=self._opts.backend.project,
            location=self._opts.backend.location,
            credentials=self._opts.backend.credentials,
            http_options=self._opts.http_options if is_given(self._opts.http_options) else None,
            api_version=api_version,
        )

        self._current_generation: _ResponseGeneration | None = None
        self._pending_generation_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        # number of tool calls rejected in the current tool_choice="none" turn; non-zero also
        # means we're draining that turn's trailing events (which have no generation to attach
        # to). reset when the next generation starts.
        self._rejected_tool_calls = 0
        # call ids we made up for tool calls the server sent without one; their responses must
        # not carry an id, since the server never issued it. kept for the session's lifetime: a
        # resumption can replay a response long after it was first queued
        self._synthetic_call_ids: set[str] = set()

        self._session_resumption_handle: str | None = (
            self._opts.session_resumption.handle
            if is_given(self._opts.session_resumption)
            else None
        )
        # chat ctx the handle stands for; None until the first handle arrives
        self._resumption_chat_ctx: llm.ChatContext | None = None
        self._chat_ctx_lock = asyncio.Lock()
        self._in_user_activity = False

        self._setup: dict[str, Any] = {}
        self._setup_sent = False
        # the next connection sends the conversation as it then is: updates until then only
        # change the chat ctx
        self._history_due = True
        # chat items queued but not yet written, by their message: a handle does not claim them
        self._unsent: dict[int, tuple[ClientMessage, frozenset[str]]] = {}
        # the calls Gemini made in this session, and the tool outputs written to it: a fresh
        # session still answers a call whose output it was not given yet
        self._calls: set[str] = set()
        self._delivered_outputs: set[str] = set()
        self._connect_started = 0.0

        self._video_frame: rtc.VideoFrame | None = None
        self._video_encode_warned = False
        self._video_tasks = utils.aio.TaskSet()

    # connection

    awaits_ready = True

    async def connect(self, conn_options: APIConnectOptions) -> llm_realtime.WebSocketTarget:
        self._history_due = True
        self._connect_started = time.perf_counter()
        target, model = await self._endpoint.target(self._opts.model)
        self._setup = self._build_setup(model)
        async with self._chat_ctx_lock:
            if self._session_resumption_handle is not None:
                # the handle restores the conversation; send only what came after it
                history = (
                    await self._diff(self._resumption_chat_ctx, self._chat_ctx)
                    if self._resumption_chat_ctx is not None
                    else []
                )
            else:
                history = await self._history()
            self._unsent = {}
            # the conversation goes ahead of a request queued before the connection
            for message, item_ids in reversed(history):
                self._send(message, item_ids, first=True)
            self._history_due = False
        self._setup_sent = True
        return target

    def setup(self) -> ClientMessage:
        return {"setup": self._setup}

    def encode(self, item: rtc.AudioFrame) -> ClientMessage:
        blob = types.Blob(data=bytes(item.data), mime_type=f"audio/pcm;rate={item.sample_rate}")
        return {"realtimeInput": {"audio": dump(blob)}}

    def _send(
        self, message: ClientMessage, item_ids: frozenset[str] = frozenset(), *, first: bool = False
    ) -> None:
        """Sends ``message``, which carries the chat items ``item_ids``."""
        if item_ids:
            self._unsent[id(message)] = (message, item_ids)
        self.send_now(message, first=first)

    def sent_now(self, message: bytes | ClientMessage) -> None:
        entry = self._unsent.pop(id(message), None)
        if entry is not None and isinstance(message, dict) and "toolResponse" in message:
            self._delivered_outputs |= entry[1]

    def _rotate(self, reason: str) -> None:
        # what changes until the new connection rides in its history
        self._history_due = True
        self.reconnect(reason)

    def _build_setup(self, model: str) -> dict[str, Any]:
        try:
            tools, _ = create_tools_config(
                self._tools,
                tool_behavior=self._opts.tool_behavior,
                use_parameters_json_schema=False,
            )
        except ValueError as e:
            raise APIError(str(e), kind="invalid_request") from e

        def given(value: NotGivenOr[Any]) -> Any:
            return value if is_given(value) else None

        generation = dump(
            types.GenerationConfig(
                candidate_count=self._opts.candidate_count,
                temperature=given(self._opts.temperature),
                max_output_tokens=given(self._opts.max_output_tokens),
                top_p=given(self._opts.top_p),
                top_k=given(self._opts.top_k),
                presence_penalty=given(self._opts.presence_penalty),
                frequency_penalty=given(self._opts.frequency_penalty),
                thinking_config=given(self._opts.thinking_config),
                media_resolution=given(self._opts.media_resolution),
            )
        )
        generation["responseModalities"] = [m.value for m in self._opts.response_modalities]
        if types.Modality.AUDIO in self._opts.response_modalities:
            generation["speechConfig"] = dump(
                types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self._opts.voice)
                    ),
                    language_code=given(self._opts.language),
                )
            )
        if is_given(self._opts.enable_affective_dialog):
            generation["enableAffectiveDialog"] = self._opts.enable_affective_dialog

        setup: dict[str, Any] = {
            "model": model,
            "generationConfig": generation,
            "tools": [dump(tool) for tool in tools],
        }
        # a config without a handle is refused on some Gemini API models (1008): it is sent
        # only to resume, or on Vertex AI when the caller configured resumption
        configured = self._opts.session_resumption
        handle = self._session_resumption_handle
        if handle is not None or (self._opts.backend.vertexai and is_given(configured)):
            transparent = configured.transparent if is_given(configured) else None
            setup["sessionResumption"] = dump(
                types.SessionResumptionConfig(
                    handle=handle, transparent=transparent if self._opts.backend.vertexai else None
                )
            )
        if is_given(self._opts.instructions):
            setup["systemInstruction"] = dump(
                types.Content(parts=[types.Part(text=self._opts.instructions)])
            )
        if self._opts.input_audio_transcription is not None:
            setup["inputAudioTranscription"] = dump(self._opts.input_audio_transcription)
        if self._opts.output_audio_transcription is not None:
            setup["outputAudioTranscription"] = dump(self._opts.output_audio_transcription)
        if is_given(self._opts.proactivity):
            setup["proactivity"] = {"proactiveAudio": self._opts.proactivity}
        if is_given(self._opts.realtime_input_config):
            setup["realtimeInputConfig"] = dump(self._opts.realtime_input_config)
        if is_given(self._opts.context_window_compression):
            setup["contextWindowCompression"] = dump(self._opts.context_window_compression)
        return setup

    # conversation state

    def _warn_system_messages(self, chat_ctx: llm.ChatContext) -> None:
        count = sum(1 for msg in chat_ctx.messages() if msg.role in ("system", "developer"))
        if count > 0:
            logger.warning(
                f"Gemini Realtime model '{self._opts.model}' does not support 'system' or "
                f"'developer' roles in chat history. Dropping {count} system message(s) from "
                "chat context. Gemini Realtime only supports 'user' and 'model' roles. Use "
                "update_instructions() to set system-level context instead."
            )

    async def _history(self) -> list[tuple[ClientMessage, frozenset[str]]]:
        """The whole conversation, for a session that starts without a resumption handle, and
        the tool outputs Gemini was not given yet: their calls still wait for them."""
        self._warn_system_messages(self._chat_ctx)
        messages: list[tuple[ClientMessage, frozenset[str]]] = []
        conversation = self._chat_ctx.copy(
            exclude_function_call=True,
            exclude_handoff=True,
            exclude_instructions=True,
            exclude_empty_message=True,
            exclude_config_update=True,
        )
        turns_dict, _ = await conversation.to_provider_format(
            format="google", inject_dummy_user_message=False
        )
        if turns_dict:
            turns = [types.Content.model_validate(turn) for turn in turns_dict]
            content = types.LiveClientContent(turns=turns, turn_complete=False)
            ids = frozenset(item.id for item in conversation.items)
            messages.append(({"clientContent": dump(content)}, ids))
        outputs = llm.ChatContext(
            [
                item
                for item in self._chat_ctx.items
                if item.type == "function_call_output"
                and item.call_id in self._calls
                and item.id not in self._delivered_outputs
            ]
        )
        if tool_results := self._tool_results(outputs):
            ids = frozenset(item.id for item in outputs.items)
            messages.append(({"toolResponse": dump(tool_results)}, ids))
        return messages

    async def _diff(
        self, known: llm.ChatContext, chat_ctx: llm.ChatContext
    ) -> list[tuple[ClientMessage, frozenset[str]]]:
        """What brings a session that holds ``known`` to ``chat_ctx``."""
        diff_ops = llm.utils.compute_chat_ctx_diff(known, chat_ctx)
        if diff_ops.to_remove:
            logger.warning("Gemini Live does not support removing messages")

        append_ctx = llm.ChatContext.empty()
        for created in diff_ops.to_create:
            if item := chat_ctx.get_by_id(created.item_id):
                append_ctx.items.append(item)
        if not append_ctx.items:
            return []

        # vertex drops `scheduling`, and Gemini reads it only on NON_BLOCKING tools
        supports_silent_scheduling = (
            not self._opts.backend.vertexai
            and self._opts.tool_behavior == types.Behavior.NON_BLOCKING
        )
        if not supports_silent_scheduling and (
            silenced := [
                item.name
                for item in append_ctx.items
                if item.type == "function_call_output" and not item.reply_required
            ]
        ):
            logger.warning(
                "a tool result wants no reply, but Gemini will answer it anyway; pass "
                "tool_behavior=NON_BLOCKING to keep it silent. Sending it regardless, "
                "since an unanswered call blocks the session.",
                extra={"functions": silenced},
            )

        messages: list[tuple[ClientMessage, frozenset[str]]] = []
        turns_dict, _ = await append_ctx.copy(exclude_function_call=True).to_provider_format(
            format="google", inject_dummy_user_message=False
        )
        if turns_dict:
            turns = [types.Content.model_validate(turn) for turn in turns_dict]
            content = types.LiveClientContent(turns=turns, turn_complete=False)
            ids = frozenset(i.id for i in append_ctx.items if i.type != "function_call_output")
            messages.append(({"clientContent": dump(content)}, ids))
        if tool_results := self._tool_results(append_ctx):
            ids = frozenset(i.id for i in append_ctx.items if i.type == "function_call_output")
            messages.append(({"toolResponse": dump(tool_results)}, ids))
        return messages

    def _tool_results(self, chat_ctx: llm.ChatContext) -> types.LiveClientToolResponse | None:
        return get_tool_results_for_realtime(
            chat_ctx,
            vertexai=self._opts.backend.vertexai,
            tool_response_scheduling=self._opts.tool_response_scheduling,
            # vertex drops `scheduling`, and Gemini reads it only on NON_BLOCKING tools
            supports_silent_scheduling=(
                not self._opts.backend.vertexai
                and self._opts.tool_behavior == types.Behavior.NON_BLOCKING
            ),
            synthetic_call_ids=self._synthetic_call_ids,
        )

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
            self._opts.temperature = temperature
            should_restart = True

        if is_given(tool_behavior) and self._opts.tool_behavior != tool_behavior:
            self._opts.tool_behavior = tool_behavior
            should_restart = True

        if (
            is_given(tool_response_scheduling)
            and self._opts.tool_response_scheduling != tool_response_scheduling
        ):
            self._opts.tool_response_scheduling = tool_response_scheduling
            if self._opts.backend.vertexai:
                _warn_vertex_scheduling_unsupported()

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
            self._rotate("options changed")

    async def update_instructions(self, instructions: str) -> None:
        if is_given(self._opts.instructions) and self._opts.instructions == instructions:
            return
        self._opts.instructions = instructions
        if not self._setup_sent:
            return  # the first setup carries them
        content = types.LiveClientContent(
            turns=[
                types.Content(
                    parts=[types.Part(text=instructions)],
                    # Vertex AI reads a mid-session "system" turn as the new instructions. The
                    # Gemini API refuses that role, and 3.1 and 3.8 close with 1007 on a turn
                    # without one: "model" is what it accepts.
                    role="system" if self._opts.backend.vertexai else "model",
                )
            ],
            turn_complete=False,
        )
        self._send({"clientContent": dump(content)})

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self._warn_system_messages(chat_ctx)
        chat_ctx = chat_ctx.copy(
            exclude_handoff=True,
            exclude_instructions=True,
            exclude_empty_message=True,
            exclude_config_update=True,
        )
        async with self._chat_ctx_lock:
            if not self._history_due:
                for message, item_ids in await self._diff(self._chat_ctx, chat_ctx):
                    self._send(message, item_ids)
            # Gemini gives no view of its history: the current state is assumed accurate, though
            # removals are not applied
            self._chat_ctx = chat_ctx

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        tool_ctx = llm.ToolContext(tools)
        if self._tools == tool_ctx:
            return
        self._tools = tool_ctx
        # the handle holds the old tool config: the new tools start a fresh session
        self._session_resumption_handle = None
        self._resumption_chat_ctx = None
        self._rotate("tools changed")

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    @property
    def _manual_activity_detection(self) -> bool:
        config = self._opts.realtime_input_config
        return bool(
            is_given(config)
            and config.automatic_activity_detection is not None
            and config.automatic_activity_detection.disabled
        )

    @property
    def session_resumption_handle(self) -> str | None:
        return self._session_resumption_handle

    # input

    def push_video(self, frame: rtc.VideoFrame) -> None:
        # only the latest frame is sent: one is encoded at a time
        idle = self._video_frame is None
        self._video_frame = frame
        if idle:
            self._video_tasks.spawn(self._send_video(), name="gemini-video")

    async def _send_video(self) -> None:
        options = self._opts.image_encode_options or DEFAULT_IMAGE_ENCODE_OPTIONS
        while (frame := self._video_frame) is not None:
            try:
                encoded = await images.encode(frame, options)
            except Exception as error:
                if not self._video_encode_warned:
                    self._video_encode_warned = True
                    logger.warning("dropping video frame after encoding failure: %s", error)
                encoded = None
            if self._video_frame is frame:
                self._video_frame = None
            if encoded is not None:
                blob = types.Blob(data=encoded, mime_type=f"image/{options.format.lower()}")
                self.send_now({"realtimeInput": {"video": dump(blob)}})

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        if is_given(tools):
            logger.warning("per-response tools is not supported by Google Realtime API, ignoring")
        if self._pending_generation_fut and not self._pending_generation_fut.done():
            logger.warning(
                "generate_reply called while another generation is pending, cancelling previous."
            )
            # clear the slot before cancelling so the done callback doesn't treat it
            # as an external cancellation and signal the server.
            old_fut = self._pending_generation_fut
            self._pending_generation_fut = None
            old_fut.cancel("Superseded by new generate_reply call")

        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.get_running_loop().create_future()
        self._pending_generation_fut = fut

        if self._in_user_activity:
            self._send({"realtimeInput": {"activityEnd": {}}})
            self._in_user_activity = False

        turns: list[types.Content] = []
        if not _needs_reply_placeholder(self._opts.model):
            # a model turn would be taken as the start of the reply: the instructions go in
            # wrapped, as the user's
            if is_given(instructions):
                text = f"<instructions>\n{instructions}\n</instructions>"
                turns.append(types.Content(parts=[types.Part(text=text)], role="user"))
        else:
            if is_given(instructions):
                turns.append(types.Content(parts=[types.Part(text=instructions)], role="model"))
            turns.append(types.Content(parts=[types.Part(text=".")], role="user"))
        content = types.LiveClientContent(turns=turns, turn_complete=True)
        self._send({"clientContent": dump(content)})

        def _on_timeout() -> None:
            if not fut.done():
                fut.set_exception(
                    llm.RealtimeError(
                        "generate_reply timed out waiting for generation_created event."
                    )
                )
                if self._pending_generation_fut is fut:
                    self._pending_generation_fut = None

        timeout_handle = asyncio.get_running_loop().call_later(5.0, _on_timeout)

        def _on_fut_done(f: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            timeout_handle.cancel()
            is_current = self._pending_generation_fut is fut
            if is_current:
                self._pending_generation_fut = None
            if f.cancelled() and is_current:
                # external cancel: signal interrupt to Gemini via activity_start
                self.interrupt()

        fut.add_done_callback(_on_fut_done)
        return fut

    def start_user_activity(self) -> None:
        if self._manual_activity_detection and not self._in_user_activity:
            self._in_user_activity = True
            self._send({"realtimeInput": {"activityStart": {}}})

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

    def commit_audio(self) -> None:
        logger.warning("commit_audio is not supported by Gemini Realtime API.")

    def clear_audio(self) -> None:
        logger.warning("clear_audio is not supported by Gemini Realtime API.")

    async def aclose(self) -> None:
        await self._video_tasks.aclose()
        await super().aclose()
        if (fut := self._pending_generation_fut) is not None:
            self._pending_generation_fut = None
            fut.cancel("Session closed")
        self._mark_current_generation_done()

    # server messages

    def on_message(self, message: ServerMessage) -> None:
        if "setupComplete" in message:
            self._report_connection_acquired(time.perf_counter() - self._connect_started)
            self.report(llm_realtime.Ready())
            return

        tool_call = message.get("toolCall")
        if tool_call and self._opts.tool_choice == "none":
            # reject without opening a generation, so the pending generate_reply stays bound to
            # the model's eventual reply and tools stay suppressed for the whole turn.
            self._reject_tool_calls(tool_call)
            return

        server_content = message.get("serverContent")
        if not self._current_generation or self._current_generation.done:
            if server_content and server_content.get("interrupted"):
                # two cases an interrupted event is sent without an active generation
                # 1) the generation is done but playout is not finished (turn_complete -> interrupted)
                # 2) the generation is not started (interrupted -> turn_complete)
                # for both cases, we interrupt the agent if there is no pending generation from
                # `generate_reply`; for the second case, the pending generation will be stopped by
                # the `turn_complete` event coming later
                if not self._pending_generation_fut:
                    self._handle_input_speech_started()
                server_content = ServerContent(
                    **{k: v for k, v in server_content.items() if k != "interrupted"}  # type: ignore[typeddict-item]
                )
            if self._is_new_generation(tool_call, server_content):
                self._start_new_generation()

        update = message.get("sessionResumptionUpdate")
        if update and update.get("resumable") and (handle := update.get("newHandle")):
            self._session_resumption_handle = handle
            unsent = frozenset().union(*(ids for _, ids in self._unsent.values()))
            self._resumption_chat_ctx = llm.ChatContext(
                [item for item in self._chat_ctx.items if item.id not in unsent]
            )

        if server_content:
            self._handle_server_content(server_content)
        if tool_call:
            self._handle_tool_calls(tool_call)
        if cancellation := message.get("toolCallCancellation"):
            logger.warning(
                "server cancelled tool calls", extra={"function_call_ids": cancellation.get("ids")}
            )
        if usage := message.get("usageMetadata"):
            self._handle_usage_metadata(usage)
        if (go_away := message.get("goAway")) is not None:
            logger.warning(
                f"Gemini server indicates disconnection soon. Time left: {go_away.get('timeLeft')}"
            )
            self._rotate("go_away")

    def _is_new_generation(self, tool_call: ToolCall | None, sc: ServerContent | None) -> bool:
        if tool_call:
            return True
        # Some Gemini models send a `generation_complete` event after tool calls, but others do
        # not. We mark the generation as done after a tool call and need to ignore any empty
        # transcriptions or generation_complete events, so new empty generations do not start
        # and interrupt tool execution.
        return bool(
            sc
            and (
                sc.get("modelTurn")
                or sc.get("outputTranscription", {}).get("text")
                or sc.get("inputTranscription", {}).get("text")
            )
        )

    def _start_new_generation(self) -> None:
        self._rejected_tool_calls = 0
        if self._current_generation and not self._current_generation.done:
            logger.warning("starting new generation while another is active. Finalizing previous.")
            self._mark_current_generation_done()

        response_id = utils.shortuuid("GR_")
        gen = self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            response_id=response_id,
            input_id=utils.shortuuid("GI_"),
            text_ch=utils.aio.Chan[str](),
            audio_ch=utils.aio.Chan[rtc.AudioFrame](),
            created_timestamp=time.time(),
        )
        audio_output = self._realtime_model.capabilities.audio_output
        if not audio_output:
            gen.audio_ch.close()

        modalities = asyncio.get_running_loop().create_future()
        modalities.set_result(["audio", "text"] if audio_output else ["text"])
        gen.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=response_id,
                text_stream=gen.text_ch,
                audio_stream=gen.audio_ch,
                modalities=modalities,
            )
        )
        generation_event = llm.GenerationCreatedEvent(
            message_stream=gen.message_ch,
            function_stream=gen.function_ch,
            user_initiated=False,
            response_id=response_id,
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

    def _handle_server_content(self, server_content: ServerContent) -> None:
        current_gen = self._current_generation
        if not current_gen:
            if self._rejected_tool_calls:
                logger.debug(
                    "ignoring server content from a rejected tool call turn",
                    extra={"lk.pii.server_content": json.dumps(server_content)},
                )
            else:
                logger.warning("received server content but no active generation.")
            return

        # With audio output and output transcription on, spoken words arrive through
        # output_transcription. Model-turn text is not spoken and must not leak into captions.
        forward_model_text = (
            not self._realtime_model.capabilities.audio_output
            or self._opts.output_audio_transcription is None
        )

        for part in server_content.get("modelTurn", {}).get("parts", []):
            if part.get("thought"):
                continue  # reasoning output
            if (text := part.get("text")) and forward_model_text:
                current_gen.push_text(text)
            if inline_data := part.get("inlineData"):
                if not inline_data.get("mimeType", "audio/").startswith("audio/"):
                    continue  # a modality the session does not play
                if current_gen.audio_ch.closed:
                    # generation_complete already closed the audio stream; a turn should not
                    # emit more audio, so drop any late frame
                    if not current_gen.extra_content_warned:
                        current_gen.extra_content_warned = True
                        logger.warning("Gemini sent audio after generation completed; dropping it")
                    continue
                try:
                    data = base64.b64decode(inline_data["data"], validate=True)
                except binascii.Error as e:
                    raise llm_realtime.protocol_error(self, f"inlineData is not base64: {e}") from e
                current_gen.audio_ch.send_nowait(
                    rtc.AudioFrame(
                        data=data,
                        sample_rate=OUTPUT_AUDIO_SAMPLE_RATE,
                        num_channels=OUTPUT_AUDIO_CHANNELS,
                        samples_per_channel=len(data) // (2 * OUTPUT_AUDIO_CHANNELS),
                    )
                )

        if text := server_content.get("inputTranscription", {}).get("text"):
            if current_gen.input_transcription == "":
                # gemini starts with a space, which makes no sense at the beginning
                text = text.lstrip()
            current_gen.input_transcription += text
            self.emit(
                "input_audio_transcription_completed",
                llm.InputTranscriptionCompleted(
                    item_id=current_gen.input_id,
                    transcript=current_gen.input_transcription,
                    is_final=False,
                    turn_started_at=current_gen.created_timestamp,
                ),
            )

        if text := server_content.get("outputTranscription", {}).get("text"):
            current_gen.push_text(text)

        if server_content.get("generationComplete") or server_content.get("turnComplete"):
            current_gen.completed_timestamp = time.time()

        # gemini delays turn_complete until it thinks client-side playback finished, so end
        # the output streams on generation_complete instead
        if server_content.get("generationComplete"):
            self._close_output_streams(current_gen)

        if server_content.get("interrupted") and not self._pending_generation_fut:
            # interrupt agent if there is no pending user initiated generation
            self._handle_input_speech_started()

        if server_content.get("turnComplete"):
            self._mark_current_generation_done()

    def _mark_current_generation_done(self) -> None:
        if not self._current_generation or self._current_generation.done:
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
                    turn_started_at=gen.created_timestamp,
                ),
            )
            # gemini gives no view of its history: the transcript is added here
            self._chat_ctx.add_message(
                role="user", content=gen.input_transcription, id=gen.input_id
            )

        if gen.output_text:
            self._chat_ctx.add_message(
                role="assistant", content=gen.output_text, id=gen.response_id
            )

        self._close_output_streams(gen)
        gen.function_ch.close()
        gen.message_ch.close()
        gen.done = True

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
            "input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=False)
        )

    def _reject_tool_calls(self, tool_call: ToolCall) -> None:
        function_calls = tool_call.get("functionCalls", [])
        if not function_calls:
            return

        self._rejected_tool_calls += 1
        extra = {"functions": [call.get("name") for call in function_calls]}
        if self._rejected_tool_calls > MAX_TOOL_CALL_REJECTIONS:
            # stop responding to break the loop; the user can still interrupt by voice
            if self._rejected_tool_calls == MAX_TOOL_CALL_REJECTIONS + 1:
                logger.warning(
                    "model keeps calling tools despite tool_choice='none'; "
                    f"stopping after {MAX_TOOL_CALL_REJECTIONS} rejections to avoid a loop",
                    extra=extra,
                )
            return

        logger.warning("rejecting tool call requested while tool_choice='none'", extra=extra)
        responses = [
            create_function_response(
                llm.FunctionCallOutput(
                    name=call.get("name", ""),
                    call_id=call.get("id", ""),
                    output="Tool calls are disabled for this turn, respond to the user directly.",
                    is_error=True,
                ),
                vertexai=self._opts.backend.vertexai,
                tool_response_scheduling=self._opts.tool_response_scheduling,
                send_id=bool(call.get("id")),
            )
            for call in function_calls
        ]
        self._send(
            {"toolResponse": dump(types.LiveClientToolResponse(function_responses=responses))}
        )

    def _handle_tool_calls(self, tool_call: ToolCall) -> None:
        if not self._current_generation:
            logger.warning("received tool call but no active generation.")
            return

        gen = self._current_generation
        for call in tool_call.get("functionCalls", []):
            if not (name := call.get("name")):
                raise llm_realtime.protocol_error(self, f"a function call without a name: {call}")
            call_id = call.get("id")
            if not call_id:
                call_id = utils.shortuuid("fnc-call-")
                self._synthetic_call_ids.add(call_id)
            self._calls.add(call_id)
            gen.function_ch.send_nowait(
                llm.FunctionCall(call_id=call_id, name=name, arguments=json.dumps(call.get("args")))
            )
        self._mark_current_generation_done()

    def _handle_usage_metadata(self, usage: UsageMetadata) -> None:
        current_gen = self._current_generation
        if not current_gen:
            if self._rejected_tool_calls:
                logger.debug("ignoring usage metadata from a rejected tool call turn")
            else:
                logger.warning("no active generation to report metrics for")
            return

        duration = (current_gen.completed_timestamp or time.time()) - current_gen.created_timestamp
        prompt = _by_modality(
            [*usage.get("promptTokensDetails", []), *usage.get("toolUsePromptTokensDetails", [])]
        )
        cached = _by_modality(usage.get("cacheTokensDetails"))
        # Vertex AI names the response after its candidates
        response = _by_modality(
            usage.get("responseTokensDetails", usage.get("candidatesTokensDetails"))
        )
        response_count = usage.get("responseTokenCount", usage.get("candidatesTokenCount", 0))
        thoughts = usage.get("thoughtsTokenCount")
        # Gemini counts built-in tool prompts outside the prompt and thinking outside the
        # response; our totals include them.
        input_tokens = usage.get("promptTokenCount", 0) + usage.get("toolUsePromptTokenCount", 0)
        output_tokens = response_count + (thoughts or 0)
        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                label=self._realtime_model.label,
                request_id=current_gen.response_id,
                timestamp=current_gen.created_timestamp,
                duration=duration,
                # Gemini has no response-start event; generation creation can be triggered by
                # an input transcript or by the first output packet.
                ttft=None,
                cancelled=False,
                tokens=TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=usage.get("cachedContentTokenCount"),
                    output_reasoning_tokens=thoughts,
                    input_text_tokens=prompt.get("text"),
                    input_audio_tokens=prompt.get("audio"),
                    input_image_tokens=prompt.get("image"),
                    output_text_tokens=response.get("text"),
                    output_audio_tokens=response.get("audio"),
                    output_image_tokens=response.get("image"),
                    cache_read_text_tokens=cached.get("text"),
                    cache_read_audio_tokens=cached.get("audio"),
                    cache_read_image_tokens=cached.get("image"),
                ),
                tokens_per_second=output_tokens / duration if duration > 0 else 0,
                metadata=Metadata(
                    model_name=self._realtime_model.model,
                    model_provider=self._realtime_model.provider,
                ),
            ),
        )
