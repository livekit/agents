"""Phonic speech-to-speech over its conversation WebSocket (``/v1/sts/ws``)."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import time
import weakref
from collections.abc import AsyncIterable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, NotRequired, Required, TypedDict

from pydantic import Field, JsonValue

from livekit import rtc
from livekit.agents import APIError, llm, utils
from livekit.agents.llm import realtime as llm_realtime
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    TranscriptChunk,
)
from livekit.agents.utils import is_given

from ..log import logger

PHONIC_INPUT_SAMPLE_RATE = 24000
PHONIC_OUTPUT_SAMPLE_RATE = 24000
PHONIC_NUM_CHANNELS = 1

CONVERSATION_HISTORY_PREFIX = (
    "\n\nThis conversation is being continued from an existing conversation. "
    "You are the assistant speaking to the user. "
    "The following is the conversation history:\n"
)
PHONIC_INPUT_FRAME_MS = 20
PHONIC_BASE_URL = "wss://api.phonic.ai"
REPLY_TIMEOUT = 10.0
TOOL_CALL_OUTPUT_TIMEOUT_MS = 60000


# Phonic's built-in tools, referenced by name in ``phonic_tools``. A configs_for_tools entry for one of
# these carries its built-in config (below) so it is sent to Phonic as an inline object rather than a name.
_BUILT_IN_TOOL_NAMES = frozenset(
    {"choose_not_to_respond", "keypad_input", "natural_conversation_ending"}
)


class PhonicToolConfig(TypedDict, total=False):
    """Per-tool behavior overrides for ``configs_for_tools`` (see README). ``name`` is required;
    every other field is optional and falls back to the plugin default when omitted."""

    name: Required[str]
    require_speech_before_tool_call: bool
    forbid_speech_after_tool_call: bool
    forbid_tool_call_after_speech: bool
    allow_tool_chaining: bool
    # Built-in tools only (set on the matching ``phonic_tools`` entry):
    respond_after_sec: float  # choose_not_to_respond: seconds to wait before a follow-up (or omit)
    speech_before_tool_call: (
        str  # keypad_input / natural_conversation_ending: required|optional|suppressed
    )


IntelligenceLevel = Literal["standard", "high"]

ObservabilityIntegration = Literal["braintrust"]

PhonicModel = Literal["phonic_v0_5", "phonic_v1", "phonic_v1_1"]


class PronunciationEntry(TypedDict):
    """A single ``{ word, pronunciation }`` entry of ``pronunciation_dictionary``."""

    word: str
    pronunciation: str


class ConfigurationEndpoint(TypedDict, total=False):
    """Endpoint the agent calls to fetch per-conversation configuration. ``url`` is required;
    ``headers`` and ``timeout_ms`` are optional."""

    url: str
    headers: dict[str, str]
    timeout_ms: int


class ToolDefinition(TypedDict):
    """A Phonic Responses API tool definition."""

    name: str
    description: str
    parameters: dict[str, Any]


def _to_phonic_tool_definition(tool_schema: dict[str, Any]) -> ToolDefinition:
    function = tool_schema["function"]
    return {
        "name": function["name"],
        "description": function.get("description") or "",
        "parameters": function["parameters"],
    }


def to_phonic_tool_definitions(tool_context: llm.ToolContext) -> list[ToolDefinition]:
    """Convert LiveKit function tools to Phonic Responses API definitions.

    The returned values contain schemas only; the executable callables remain in
    ``tool_context`` for the caller to invoke when Phonic returns a tool call.
    """

    return [
        _to_phonic_tool_definition(tool_schema)
        for tool_schema in tool_context.parse_function_tools("openai", strict=True)
    ]


@dataclass
class _RealtimeOptions:
    api_key: str
    phonic_agent: NotGivenOr[str]
    voice: NotGivenOr[str]
    welcome_message: NotGivenOr[str | None]
    generate_welcome_message: NotGivenOr[bool | None]
    project: NotGivenOr[str | None]
    default_language: NotGivenOr[str]
    additional_languages: NotGivenOr[list[str]]
    multilingual_mode: NotGivenOr[Literal["auto", "request"]]
    audio_speed: NotGivenOr[float]
    phonic_tools: NotGivenOr[list[str]]
    boosted_keywords: NotGivenOr[list[str]]
    min_words_to_interrupt: NotGivenOr[int]
    generate_no_input_poke_text: NotGivenOr[bool]
    no_input_poke_sec: NotGivenOr[float]
    no_input_poke_text: NotGivenOr[str]
    no_input_end_conversation_sec: NotGivenOr[float]
    websocket_timeout_sec: NotGivenOr[int]
    intelligence_level: NotGivenOr[IntelligenceLevel]
    phonic_model: NotGivenOr[PhonicModel]
    is_welcome_message_interruptible: NotGivenOr[bool]
    vad_prebuffer_duration_ms: NotGivenOr[int]
    vad_min_speech_duration_ms: NotGivenOr[int]
    vad_min_silence_duration_ms: NotGivenOr[int]
    vad_threshold: NotGivenOr[float]
    enable_assistant_backchannel: NotGivenOr[bool]
    assistant_backchannel_aggressiveness: NotGivenOr[float]
    pronunciation_dictionary: NotGivenOr[list[PronunciationEntry]]
    template_variables: NotGivenOr[dict[str, str]]
    enable_redaction: NotGivenOr[bool]
    enable_watermarking: NotGivenOr[bool]
    mcp_servers: NotGivenOr[list[str]]
    observability_integrations: NotGivenOr[list[ObservabilityIntegration]]
    configuration_endpoint: NotGivenOr[ConfigurationEndpoint | None]
    additional_params: NotGivenOr[Mapping[str, JsonValue]]
    configs_for_tools: NotGivenOr[list[PhonicToolConfig]]
    conn_options: APIConnectOptions
    base_url: str = PHONIC_BASE_URL
    instructions: NotGivenOr[str] = NOT_GIVEN


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    text_ch: utils.aio.Chan[str | TranscriptChunk]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]

    response_id: str
    input_id: str

    input_transcription: str = ""
    output_text: str = ""

    # Running offset (seconds) into the assistant audio stream, used to stamp each
    # text chunk with the time span of the audio it was delivered alongside.
    audio_cursor_sec: float = 0.0

    _created_timestamp: float = field(default_factory=time.time)
    done: bool = False

    def push_text(self, text: str | TranscriptChunk) -> None:
        self.output_text += text.text if isinstance(text, TranscriptChunk) else text
        self.text_ch.send_nowait(text)


class RealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        phonic_agent: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        welcome_message: NotGivenOr[str | None] = NOT_GIVEN,
        generate_welcome_message: NotGivenOr[bool] = NOT_GIVEN,
        project: NotGivenOr[str | None] = NOT_GIVEN,
        default_language: NotGivenOr[str] = NOT_GIVEN,
        additional_languages: NotGivenOr[list[str]] = NOT_GIVEN,
        multilingual_mode: NotGivenOr[Literal["auto", "request"]] = NOT_GIVEN,
        audio_speed: NotGivenOr[float] = NOT_GIVEN,
        phonic_tools: NotGivenOr[list[str]] = NOT_GIVEN,
        boosted_keywords: NotGivenOr[list[str]] = NOT_GIVEN,
        min_words_to_interrupt: NotGivenOr[int] = NOT_GIVEN,
        generate_no_input_poke_text: NotGivenOr[bool] = NOT_GIVEN,
        no_input_poke_sec: NotGivenOr[float] = NOT_GIVEN,
        no_input_poke_text: NotGivenOr[str] = NOT_GIVEN,
        no_input_end_conversation_sec: NotGivenOr[float] = NOT_GIVEN,
        websocket_timeout_sec: NotGivenOr[int] = NOT_GIVEN,
        intelligence_level: NotGivenOr[IntelligenceLevel] = NOT_GIVEN,
        phonic_model: NotGivenOr[PhonicModel] = NOT_GIVEN,
        is_welcome_message_interruptible: NotGivenOr[bool] = NOT_GIVEN,
        vad_prebuffer_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_min_speech_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        enable_assistant_backchannel: NotGivenOr[bool] = NOT_GIVEN,
        assistant_backchannel_aggressiveness: NotGivenOr[float] = NOT_GIVEN,
        pronunciation_dictionary: NotGivenOr[list[PronunciationEntry]] = NOT_GIVEN,
        template_variables: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        enable_redaction: NotGivenOr[bool] = NOT_GIVEN,
        enable_watermarking: NotGivenOr[bool] = NOT_GIVEN,
        mcp_servers: NotGivenOr[list[str]] = NOT_GIVEN,
        observability_integrations: NotGivenOr[list[ObservabilityIntegration]] = NOT_GIVEN,
        configuration_endpoint: NotGivenOr[ConfigurationEndpoint | None] = NOT_GIVEN,
        additional_params: NotGivenOr[Mapping[str, JsonValue]] = NOT_GIVEN,
        configs_for_tools: NotGivenOr[list[PhonicToolConfig]] = NOT_GIVEN,
        base_url: str = PHONIC_BASE_URL,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Initialize a RealtimeModel for Phonic's Realtime API.

        Args:
            api_key: Phonic API key. If not provided, reads from PHONIC_API_KEY environment variable.
            phonic_agent: Phonic agent to use for the conversation. Options explicitly set
                here will override the agent's default settings.
            voice: Voice ID for agent audio output.
            welcome_message: Message for the agent to say when the conversation starts.
                Ignored when ``generate_welcome_message`` is True.
            generate_welcome_message: When True, the welcome message is automatically generated
                and ``welcome_message`` is ignored.
            project: Project name to use for the conversation.
            default_language: ISO 639-1 default language for recognition and speech.
            additional_languages: Further ISO 639-1 codes the agent may use (must not include
                ``default_language``).
            multilingual_mode: ``\"auto\"`` to detect language per utterance, ``\"request\"`` to
                switch only when the user asks (recommended).
            audio_speed: Audio playback speed multiplier.
            phonic_tools: Phonic tool names available to the assistant.
            boosted_keywords: Keywords to boost in speech recognition.
            min_words_to_interrupt: Minimum number of user words required to interrupt the assistant.
            generate_no_input_poke_text: When True, auto-generate poke text when the user is silent.
            no_input_poke_sec: Seconds of silence before sending a poke message.
            no_input_poke_text: Custom poke message text. Ignored when
                ``generate_no_input_poke_text`` is True.
            no_input_end_conversation_sec: Seconds of silence before ending the conversation.
            websocket_timeout_sec: Seconds of inactivity before the Phonic websocket is closed.
            intelligence_level: Model intelligence level, ``"standard"`` or ``"high"``.
            phonic_model: Phonic model version to use, one of ``"phonic_v0_5"``,
                ``"phonic_v1"`` or ``"phonic_v1_1"``.
            is_welcome_message_interruptible: When False, the welcome message cannot be
                interrupted by the user.
            vad_prebuffer_duration_ms: Voice-activity-detection prebuffer duration, in milliseconds.
            vad_min_speech_duration_ms: Minimum speech duration for VAD, in milliseconds.
            vad_min_silence_duration_ms: Minimum silence duration for VAD, in milliseconds.
            vad_threshold: Voice-activity-detection threshold.
            enable_assistant_backchannel: When True, the assistant produces backchannel responses
                (e.g. "mm-hmm", "yeah") while the user is speaking.
            assistant_backchannel_aggressiveness: How aggressively the assistant backchannels.
                Only applies when ``enable_assistant_backchannel`` is True.
            pronunciation_dictionary: List of ``{ word, pronunciation }`` entries; words must be unique.
            template_variables: Variables substituted into the system prompt and welcome message.
            enable_redaction: When True, PII/PHI is redacted from transcripts and bleeped from audio
                after the conversation ends.
            enable_watermarking: When True, embeds an inaudible provenance watermark in the agent's
                generated audio. Adds a very small amount of latency.
            mcp_servers: Names of pre-configured MCP servers to make available to the assistant.
                Names must be unique.
            observability_integrations: Names of observability integrations to forward traces to
                (currently ``"braintrust"``).
            configuration_endpoint: When set, the agent calls this endpoint to fetch per-conversation
                configuration options. Pass None to disable.
            additional_params: Additional runtime parameters forwarded to Phonic.
            configs_for_tools: Per-tool behavior overrides, one ``PhonicToolConfig`` per tool
                (keyed by ``name``); omitted fields fall back to the plugin defaults. See the
                README for the available fields.
            base_url: Phonic's WebSocket base URL.
            conn_options: Retry/backoff and connection settings.
        """
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=True,
                audio_output=True,
                manual_function_calls=False,
                mutable_chat_context=True,
                mutable_instructions=True,
                mutable_tools=True,
                per_response_tool_choice=False,
                supports_say=True,
            )
        )

        api_key = api_key or os.environ.get("PHONIC_API_KEY", NOT_GIVEN)
        if not is_given(api_key):
            raise ValueError(
                "Phonic API key is required. Provide `api_key` or "
                "set PHONIC_API_KEY environment variable."
            )

        self._opts = _RealtimeOptions(
            api_key=api_key,
            phonic_agent=phonic_agent,
            voice=voice,
            welcome_message=welcome_message,
            generate_welcome_message=generate_welcome_message,
            project=project,
            default_language=default_language,
            additional_languages=additional_languages,
            multilingual_mode=multilingual_mode,
            audio_speed=audio_speed,
            phonic_tools=phonic_tools,
            boosted_keywords=boosted_keywords,
            min_words_to_interrupt=min_words_to_interrupt,
            generate_no_input_poke_text=generate_no_input_poke_text,
            no_input_poke_sec=no_input_poke_sec,
            no_input_poke_text=no_input_poke_text,
            no_input_end_conversation_sec=no_input_end_conversation_sec,
            websocket_timeout_sec=websocket_timeout_sec,
            intelligence_level=intelligence_level,
            phonic_model=phonic_model,
            is_welcome_message_interruptible=is_welcome_message_interruptible,
            vad_prebuffer_duration_ms=vad_prebuffer_duration_ms,
            vad_min_speech_duration_ms=vad_min_speech_duration_ms,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
            vad_threshold=vad_threshold,
            enable_assistant_backchannel=enable_assistant_backchannel,
            assistant_backchannel_aggressiveness=assistant_backchannel_aggressiveness,
            pronunciation_dictionary=pronunciation_dictionary,
            template_variables=template_variables,
            enable_redaction=enable_redaction,
            enable_watermarking=enable_watermarking,
            mcp_servers=mcp_servers,
            observability_integrations=observability_integrations,
            configuration_endpoint=configuration_endpoint,
            additional_params=additional_params,
            configs_for_tools=configs_for_tools,
            conn_options=conn_options,
            base_url=base_url,
        )

        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return "phonic"

    @property
    def provider(self) -> str:
        return "phonic"

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        # disabling server-side turn detection is unsupported (can_disable_turn_detection=False)
        sess = RealtimeSession(self, options=self._opts)
        self._sessions.add(sess)
        return sess

    def update_options(
        self,
        *,
        phonic_agent: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        welcome_message: NotGivenOr[str | None] = NOT_GIVEN,
        generate_welcome_message: NotGivenOr[bool | None] = NOT_GIVEN,
        project: NotGivenOr[str | None] = NOT_GIVEN,
        default_language: NotGivenOr[str] = NOT_GIVEN,
        additional_languages: NotGivenOr[list[str]] = NOT_GIVEN,
        multilingual_mode: NotGivenOr[Literal["auto", "request"]] = NOT_GIVEN,
        audio_speed: NotGivenOr[float] = NOT_GIVEN,
        phonic_tools: NotGivenOr[list[str]] = NOT_GIVEN,
        boosted_keywords: NotGivenOr[list[str]] = NOT_GIVEN,
        min_words_to_interrupt: NotGivenOr[int] = NOT_GIVEN,
        generate_no_input_poke_text: NotGivenOr[bool] = NOT_GIVEN,
        no_input_poke_sec: NotGivenOr[float] = NOT_GIVEN,
        no_input_poke_text: NotGivenOr[str] = NOT_GIVEN,
        no_input_end_conversation_sec: NotGivenOr[float] = NOT_GIVEN,
        websocket_timeout_sec: NotGivenOr[int] = NOT_GIVEN,
        intelligence_level: NotGivenOr[IntelligenceLevel] = NOT_GIVEN,
        phonic_model: NotGivenOr[PhonicModel] = NOT_GIVEN,
        is_welcome_message_interruptible: NotGivenOr[bool] = NOT_GIVEN,
        vad_prebuffer_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_min_speech_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        enable_assistant_backchannel: NotGivenOr[bool] = NOT_GIVEN,
        assistant_backchannel_aggressiveness: NotGivenOr[float] = NOT_GIVEN,
        pronunciation_dictionary: NotGivenOr[list[PronunciationEntry]] = NOT_GIVEN,
        template_variables: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        enable_redaction: NotGivenOr[bool] = NOT_GIVEN,
        enable_watermarking: NotGivenOr[bool] = NOT_GIVEN,
        mcp_servers: NotGivenOr[list[str]] = NOT_GIVEN,
        observability_integrations: NotGivenOr[list[ObservabilityIntegration]] = NOT_GIVEN,
        configuration_endpoint: NotGivenOr[ConfigurationEndpoint | None] = NOT_GIVEN,
        additional_params: NotGivenOr[Mapping[str, JsonValue]] = NOT_GIVEN,
        configs_for_tools: NotGivenOr[list[PhonicToolConfig]] = NOT_GIVEN,
    ) -> None:
        """Change Phonic config fields on the active session(s) mid-conversation (e.g. switch
        ``default_language`` when advancing to the next task). Only the fields you pass are changed;
        each is applied immediately via a Phonic ``reset``.

        When ``default_language`` changes and ``additional_languages`` isn't passed, the previous
        default is rotated into ``additional_languages`` (and the new default removed) so the
        language set stays intact — the API rejects a default that also appears there."""
        for sess in self._sessions:
            sess.update_options(
                phonic_agent=phonic_agent,
                voice=voice,
                welcome_message=welcome_message,
                generate_welcome_message=generate_welcome_message,
                project=project,
                default_language=default_language,
                additional_languages=additional_languages,
                multilingual_mode=multilingual_mode,
                audio_speed=audio_speed,
                phonic_tools=phonic_tools,
                boosted_keywords=boosted_keywords,
                min_words_to_interrupt=min_words_to_interrupt,
                generate_no_input_poke_text=generate_no_input_poke_text,
                no_input_poke_sec=no_input_poke_sec,
                no_input_poke_text=no_input_poke_text,
                no_input_end_conversation_sec=no_input_end_conversation_sec,
                websocket_timeout_sec=websocket_timeout_sec,
                intelligence_level=intelligence_level,
                phonic_model=phonic_model,
                is_welcome_message_interruptible=is_welcome_message_interruptible,
                vad_prebuffer_duration_ms=vad_prebuffer_duration_ms,
                vad_min_speech_duration_ms=vad_min_speech_duration_ms,
                vad_min_silence_duration_ms=vad_min_silence_duration_ms,
                vad_threshold=vad_threshold,
                enable_assistant_backchannel=enable_assistant_backchannel,
                assistant_backchannel_aggressiveness=assistant_backchannel_aggressiveness,
                pronunciation_dictionary=pronunciation_dictionary,
                template_variables=template_variables,
                enable_redaction=enable_redaction,
                enable_watermarking=enable_watermarking,
                mcp_servers=mcp_servers,
                observability_integrations=observability_integrations,
                configuration_endpoint=configuration_endpoint,
                additional_params=additional_params,
                configs_for_tools=configs_for_tools,
            )

    async def aclose(self) -> None:
        pass


# Phonic's server messages: only the fields the plugin reads


class _Ready(TypedDict):
    type: Literal["ready_to_start_conversation"]


class _ConversationCreated(TypedDict):
    type: Literal["conversation_created"]
    conversation_id: str


class _AudioChunk(TypedDict):
    type: Literal["audio_chunk"]
    audio: str
    text: NotRequired[str]


class _InputText(TypedDict):
    type: Literal["input_text"]
    text: str


class _UserStartedSpeaking(TypedDict):
    type: Literal["user_started_speaking"]


class _UserFinishedSpeaking(TypedDict):
    type: Literal["user_finished_speaking"]


class _AssistantStartedSpeaking(TypedDict):
    type: Literal["assistant_started_speaking"]


class _AssistantFinishedSpeaking(TypedDict):
    type: Literal["assistant_finished_speaking"]


class _ToolCall(TypedDict):
    type: Literal["tool_call"]
    tool_call_id: str
    tool_name: str
    parameters: dict[str, JsonValue]


class _ToolCallInterrupted(TypedDict):
    type: Literal["tool_call_interrupted"]
    tool_call_id: str
    tool_name: str


class _Detail(TypedDict):
    message: str
    code: NotRequired[str]


class _Error(TypedDict):
    type: Literal["error"]
    error: _Detail
    param_errors: NotRequired[dict[str, str] | None]


class _Warning(TypedDict):
    type: Literal["warning"]
    warning: _Detail


class _AssistantEndedConversation(TypedDict):
    type: Literal["assistant_ended_conversation"]


ServerMessage = Annotated[
    _Ready
    | _ConversationCreated
    | _AudioChunk
    | _InputText
    | _UserStartedSpeaking
    | _UserFinishedSpeaking
    | _AssistantStartedSpeaking
    | _AssistantFinishedSpeaking
    | _ToolCall
    | _ToolCallInterrupted
    | _Error
    | _Warning
    | _AssistantEndedConversation,
    Field(discriminator="type"),
]

# the client's messages


class _SendAudio(TypedDict):
    type: Literal["audio_chunk"]
    audio: str


class _GenerateReply(TypedDict):
    type: Literal["generate_reply"]
    system_message: str | None


class _Say(TypedDict):
    type: Literal["say"]
    text: str


class _ToolCallOutput(TypedDict):
    type: Literal["tool_call_output"]
    tool_call_id: str
    output: str


class _AddSystemMessage(TypedDict):
    type: Literal["add_system_message"]
    system_message: str


class _Reset(TypedDict):
    type: Literal["reset"]
    config: dict[str, object]


# ``config`` is its options at the top level: ``{"type": "config", **options}``
ClientMessage = (
    _SendAudio
    | _GenerateReply
    | _Say
    | _ToolCallOutput
    | _AddSystemMessage
    | _Reset
    | dict[str, object]
)


class RealtimeSession(llm.RealtimeWebSocketSession[ServerMessage, ClientMessage]):
    """One Phonic conversation per connection. Its config goes out once the instructions and the
    tools are known; every other command waits for ``ready_to_start_conversation``."""

    frame_ms = PHONIC_INPUT_FRAME_MS

    def __init__(self, realtime_model: RealtimeModel, *, options: _RealtimeOptions) -> None:
        super().__init__(
            realtime_model,
            conn_options=options.conn_options,
            input_sample_rate=PHONIC_INPUT_SAMPLE_RATE,
            input_num_channels=PHONIC_NUM_CHANNELS,
        )
        self._opts = options
        self._tools = llm.ToolContext.empty()
        self._chat_ctx = llm.ChatContext.empty()
        self._current_generation: _ResponseGeneration | None = None
        self._conversation_id: str | None = None

        self._tools_set = False
        # the instructions and the tools are known: a conversation can be configured
        self._configured = asyncio.Event()
        self._config_sent = False  # the first config went out: instructions and tools are fixed
        self._pending_reply: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        self._pending_tool_call_ids: set[str] = set()
        self._tool_definitions: list[dict[str, object]] = []
        self._configs_for_tools: dict[str, PhonicToolConfig] = {}
        self._system_prompt_postfix = ""
        self._pending_user_text: str | None = None
        self._say_tasks = utils.aio.TaskSet()

    awaits_ready = True

    async def connect(self, conn_options: APIConnectOptions) -> llm_realtime.WebSocketTarget:
        await self._configured.wait()
        return llm_realtime.WebSocketTarget(
            f"{self._opts.base_url.rstrip('/')}/v1/sts/ws",
            headers={
                "Authorization": f"Bearer {self._opts.api_key}",
                "x-phonic-client": "livekit-agents-py",
            },
        )

    def encode(self, item: rtc.AudioFrame) -> ClientMessage:
        return _SendAudio(type="audio_chunk", audio=base64.b64encode(item.data).decode())

    def on_message(self, message: ServerMessage) -> None:
        if message["type"] == "ready_to_start_conversation":
            self.report(llm_realtime.Ready())
        elif message["type"] == "conversation_created":
            self._conversation_id = message["conversation_id"]
            logger.info(f"Phonic conversation began with ID: {self._conversation_id}")
        elif message["type"] == "assistant_started_speaking":
            self._start_new_assistant_turn()
        elif message["type"] == "assistant_finished_speaking":
            self._close_current_generation(interrupted=False)
        elif message["type"] == "audio_chunk":
            self._handle_audio_chunk(message)
        elif message["type"] == "input_text":
            self._handle_input_text(message["text"])
        elif message["type"] == "user_started_speaking":
            self.emit("input_speech_started", llm.InputSpeechStartedEvent())
            self._close_current_generation(interrupted=True)
        elif message["type"] == "user_finished_speaking":
            self.emit(
                "input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=True)
            )
        elif message["type"] == "tool_call":
            self._handle_tool_call(message)
        elif message["type"] == "tool_call_interrupted":
            self._pending_tool_call_ids.discard(message["tool_call_id"])
            logger.warning(
                f"Tool call for {message['tool_name']} (call_id: {message['tool_call_id']}) "
                "was cancelled due to user interruption."
            )
        elif message["type"] == "warning":
            logger.warning(f"Phonic warning: {message['warning']['message']}")
        elif message["type"] == "error":
            # parameter errors are the request's; anything else is Phonic's own failure
            raise APIError(
                "Phonic reported an error",
                kind="invalid_request" if message.get("param_errors") else "server",
                detail=json.dumps(message),
            )
        elif message["type"] == "assistant_ended_conversation":
            raise APIError(
                "Phonic ended the conversation, which LiveKit Agents does not support",
                kind="invalid_request",
            )
        else:
            self._skip_unknown_message(message)

    def _update_configured(self) -> None:
        if is_given(self._opts.instructions) and self._tools_set:
            self._configured.set()

    def setup(self) -> ClientMessage:
        assert is_given(self._opts.instructions)
        system_prompt = self._opts.instructions + self._system_prompt_postfix
        options = self._build_config_options(
            system_prompt=system_prompt, tools_payload=self._build_tools_payload()
        )
        if self._config_sent:
            # a new conversation after a lost one: it carries the history, and no welcome
            system_prompt = self._opts.instructions
            if turn_history := self._build_turn_history(self._chat_ctx):
                system_prompt += CONVERSATION_HISTORY_PREFIX + turn_history
            options |= {
                "system_prompt": system_prompt,
                "welcome_message": None,
                "generate_welcome_message": False,
            }
        self._config_sent = True
        return {"type": "config", **options}

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    async def update_instructions(self, instructions: str) -> None:
        if self._config_sent:
            logger.warning(
                "update_instructions called after config was already sent. "
                "Phonic does not support updating instructions mid-session."
            )
            return
        self._opts.instructions = instructions
        self._update_configured()

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        if self._config_sent:
            logger.warning(
                "update_tools called after config was already sent. "
                "Phonic does not support updating tools mid-session."
            )
            return
        self._tools = llm.ToolContext(tools)
        self._rebuild_tool_definitions()
        self._tools_set = True
        self._update_configured()

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        if not self._config_sent:
            if turn_history := self._build_turn_history(chat_ctx):
                # the conversation so far rides in the first config's system prompt
                self._system_prompt_postfix = CONVERSATION_HISTORY_PREFIX + turn_history
                self._chat_ctx = chat_ctx.copy()
            return

        diff_ops = llm.utils.compute_chat_ctx_diff(self._chat_ctx, chat_ctx)
        sent_tool_call_output = False
        sent_system_message = False
        forbid_speech = False
        buffered_user_text = False
        last_item_id = chat_ctx.items[-1].id if chat_ctx.items else None

        for created in diff_ops.to_create:
            item = chat_ctx.get_by_id(created.item_id)
            if item is None:
                continue

            if (
                isinstance(item, llm.FunctionCallOutput)
                and item.call_id in self._pending_tool_call_ids
            ):
                self._pending_tool_call_ids.remove(item.call_id)
                self.send_now(
                    _ToolCallOutput(
                        type="tool_call_output", tool_call_id=item.call_id, output=item.output
                    )
                )
                sent_tool_call_output = True
                # the tool forbids speech after its call, or the result wants no reply
                cfg = self._configs_for_tools.get(item.name or "")
                if (
                    cfg is not None and cfg.get("forbid_speech_after_tool_call", False)
                ) or not item.reply_required:
                    forbid_speech = True

            if isinstance(item, llm.ChatMessage) and item.role in ("system", "developer"):
                if text := item.raw_text_content:
                    self.send_now(_AddSystemMessage(type="add_system_message", system_message=text))
                    sent_system_message = True

            # only a user message appended at the tail of the context is text input
            if (
                isinstance(item, llm.ChatMessage)
                and item.role == "user"
                and item.id == last_item_id
                and (text := item.raw_text_content)
            ):
                self._pending_user_text = text
                buffered_user_text = True

        self._chat_ctx = chat_ctx.copy()

        if not sent_tool_call_output and not sent_system_message and not buffered_user_text:
            logger.warning(
                "update_chat_ctx called but no new tool call outputs to send. "
                "Phonic does not support general chat context updates."
            )
        # a tool that forbids speech after its call gets no reply: a turn opened for it would
        # never receive audio nor a finished-speaking event
        if sent_tool_call_output and not forbid_speech:
            self._start_new_assistant_turn()

    async def _update_session(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> None:
        # before the first config, each field is set the usual way and rides in that config
        if not self._config_sent:
            await super()._update_session(instructions=instructions, chat_ctx=chat_ctx, tools=tools)
            return

        # a reset starts a fresh turn: close the outgoing agent's response and drop its
        # buffered user text, so neither leaks into the new agent's config
        self._close_current_generation(interrupted=True)
        self._pending_user_text = None
        if is_given(instructions):
            self._opts.instructions = instructions
        if is_given(tools):
            self._tools = llm.ToolContext(tools)
            self._tool_definitions = self._serialize_tools()
        if is_given(chat_ctx):
            self._chat_ctx = chat_ctx.copy()
        self._send_reset()

    def _send_reset(self) -> None:
        """Applies the current options, instructions, tools and history to the conversation."""
        system_prompt = self._opts.instructions if is_given(self._opts.instructions) else ""
        if turn_history := self._build_turn_history(self._chat_ctx):
            system_prompt += CONVERSATION_HISTORY_PREFIX + turn_history
        config = self._build_config_options(
            system_prompt=system_prompt, tools_payload=self._build_tools_payload()
        )
        self.send_now(_Reset(type="reset", config=config))

    def _serialize_tools(self) -> list[dict[str, object]]:
        tool_definitions: list[dict[str, object]] = []
        for tool_schema in self._tools.parse_function_tools("openai", strict=True):
            cfg = self._configs_for_tools.get(_to_phonic_tool_definition(tool_schema)["name"])
            tool_definitions.append(
                {
                    "type": "custom_websocket",
                    "tool_schema": tool_schema,
                    "tool_call_output_timeout_ms": TOOL_CALL_OUTPUT_TIMEOUT_MS,
                    # fixed, not configurable: the plugin does not support tool calls during
                    # agent speech
                    "wait_for_speech_before_tool_call": True,
                    "require_speech_before_tool_call": bool(
                        cfg and cfg.get("require_speech_before_tool_call", False)
                    ),
                    "allow_tool_chaining": bool(cfg and cfg.get("allow_tool_chaining", False)),
                    "forbid_speech_after_tool_call": bool(
                        cfg and cfg.get("forbid_speech_after_tool_call", False)
                    ),
                    "forbid_tool_call_after_speech": bool(
                        cfg and cfg.get("forbid_tool_call_after_speech", False)
                    ),
                }
            )
        return tool_definitions

    def _rebuild_tool_definitions(self) -> None:
        """Rebuilds the per-tool configs and the tool definitions after the tools or a
        tool-related option (configs_for_tools, phonic_tools) changed."""
        self._configs_for_tools = {
            c["name"]: c
            for c in (
                self._opts.configs_for_tools if is_given(self._opts.configs_for_tools) else []
            )
        }
        self._tool_definitions = self._serialize_tools()

    def _serialize_phonic_tool(self, name: str) -> dict[str, object] | str:
        """A phonic_tools entry: an inline built-in object when it's a built-in with a config in
        configs_for_tools (so respond_after_sec / speech_before_tool_call reach Phonic), else the
        bare name (which uses the tool's default config)."""
        if name not in _BUILT_IN_TOOL_NAMES:
            return name
        cfg = self._configs_for_tools.get(name)
        if cfg is None:
            return name
        if name == "choose_not_to_respond":
            if "respond_after_sec" not in cfg:
                return name
            tool_config: dict[str, object] = {"respond_after_sec": cfg["respond_after_sec"]}
        else:  # keypad_input, natural_conversation_ending
            if "speech_before_tool_call" not in cfg:
                return name
            tool_config = {"speech_before_tool_call": cfg["speech_before_tool_call"]}
        return {"type": "built_in", "name": name, "tool_config": tool_config}

    def _build_tools_payload(self) -> list[dict[str, object] | str]:
        tools_payload: list[dict[str, object] | str] = []
        if is_given(self._opts.phonic_tools) and self._opts.phonic_tools:
            tools_payload.extend(
                self._serialize_phonic_tool(name) for name in self._opts.phonic_tools
            )
        tools_payload.extend(self._tool_definitions)
        return tools_payload

    def _build_turn_history(self, chat_ctx: llm.ChatContext) -> str:
        return "\n".join(
            f"{item.role}: {item.raw_text_content}"
            for item in chat_ctx.items
            if isinstance(item, llm.ChatMessage)
            and item.raw_text_content
            and item.raw_text_content.strip()
        )

    def _build_config_options(
        self, *, system_prompt: str, tools_payload: list[dict[str, object] | str]
    ) -> dict[str, object]:
        options: dict[str, object] = {
            "agent": self._opts.phonic_agent,
            "project": self._opts.project,
            "welcome_message": self._opts.welcome_message,
            "generate_welcome_message": self._opts.generate_welcome_message,
            "system_prompt": system_prompt,
            "voice_id": self._opts.voice,
            "input_format": "pcm_24000",
            "output_format": "pcm_24000",
            "stream_ahead_of_real_time": True,
            "default_language": self._opts.default_language,
            "additional_languages": self._opts.additional_languages,
            "multilingual_mode": self._opts.multilingual_mode,
            "audio_speed": self._opts.audio_speed,
            "tools": tools_payload if len(tools_payload) > 0 else NOT_GIVEN,
            "boosted_keywords": self._opts.boosted_keywords,
            "min_words_to_interrupt": self._opts.min_words_to_interrupt,
            "generate_no_input_poke_text": self._opts.generate_no_input_poke_text,
            "no_input_poke_sec": self._opts.no_input_poke_sec,
            "no_input_poke_text": self._opts.no_input_poke_text,
            "no_input_end_conversation_sec": self._opts.no_input_end_conversation_sec,
            "websocket_timeout_sec": self._opts.websocket_timeout_sec,
            "intelligence_level": self._opts.intelligence_level,
            "phonic_model": self._opts.phonic_model,
            "is_welcome_message_interruptible": self._opts.is_welcome_message_interruptible,
            "vad_prebuffer_duration_ms": self._opts.vad_prebuffer_duration_ms,
            "vad_min_speech_duration_ms": self._opts.vad_min_speech_duration_ms,
            "vad_min_silence_duration_ms": self._opts.vad_min_silence_duration_ms,
            "vad_threshold": self._opts.vad_threshold,
            "enable_assistant_backchannel": self._opts.enable_assistant_backchannel,
            "assistant_backchannel_aggressiveness": self._opts.assistant_backchannel_aggressiveness,
            "pronunciation_dictionary": self._opts.pronunciation_dictionary,
            "template_variables": self._opts.template_variables,
            "enable_redaction": self._opts.enable_redaction,
            "enable_watermarking": self._opts.enable_watermarking,
            "mcp_servers": self._opts.mcp_servers,
            "observability_integrations": self._opts.observability_integrations,
            "configuration_endpoint": self._opts.configuration_endpoint,
            "additional_params": self._opts.additional_params,
        }
        return {k: v for k, v in options.items() if v is not NOT_GIVEN}

    def update_options(
        self,
        *,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        phonic_agent: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        welcome_message: NotGivenOr[str | None] = NOT_GIVEN,
        generate_welcome_message: NotGivenOr[bool | None] = NOT_GIVEN,
        project: NotGivenOr[str | None] = NOT_GIVEN,
        default_language: NotGivenOr[str] = NOT_GIVEN,
        additional_languages: NotGivenOr[list[str]] = NOT_GIVEN,
        multilingual_mode: NotGivenOr[Literal["auto", "request"]] = NOT_GIVEN,
        audio_speed: NotGivenOr[float] = NOT_GIVEN,
        phonic_tools: NotGivenOr[list[str]] = NOT_GIVEN,
        boosted_keywords: NotGivenOr[list[str]] = NOT_GIVEN,
        min_words_to_interrupt: NotGivenOr[int] = NOT_GIVEN,
        generate_no_input_poke_text: NotGivenOr[bool] = NOT_GIVEN,
        no_input_poke_sec: NotGivenOr[float] = NOT_GIVEN,
        no_input_poke_text: NotGivenOr[str] = NOT_GIVEN,
        no_input_end_conversation_sec: NotGivenOr[float] = NOT_GIVEN,
        websocket_timeout_sec: NotGivenOr[int] = NOT_GIVEN,
        intelligence_level: NotGivenOr[IntelligenceLevel] = NOT_GIVEN,
        phonic_model: NotGivenOr[PhonicModel] = NOT_GIVEN,
        is_welcome_message_interruptible: NotGivenOr[bool] = NOT_GIVEN,
        vad_prebuffer_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_min_speech_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        enable_assistant_backchannel: NotGivenOr[bool] = NOT_GIVEN,
        assistant_backchannel_aggressiveness: NotGivenOr[float] = NOT_GIVEN,
        pronunciation_dictionary: NotGivenOr[list[PronunciationEntry]] = NOT_GIVEN,
        template_variables: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        enable_redaction: NotGivenOr[bool] = NOT_GIVEN,
        enable_watermarking: NotGivenOr[bool] = NOT_GIVEN,
        mcp_servers: NotGivenOr[list[str]] = NOT_GIVEN,
        observability_integrations: NotGivenOr[list[ObservabilityIntegration]] = NOT_GIVEN,
        configuration_endpoint: NotGivenOr[ConfigurationEndpoint | None] = NOT_GIVEN,
        additional_params: NotGivenOr[Mapping[str, JsonValue]] = NOT_GIVEN,
        configs_for_tools: NotGivenOr[list[PhonicToolConfig]] = NOT_GIVEN,
    ) -> None:
        # tool_choice is the base update_options param (the framework sends it every turn); Phonic
        # does not support it and ignores it. Every other field is an optional config change.
        changes: dict[str, object] = {
            name: value
            for name, value in (
                ("phonic_agent", phonic_agent),
                ("voice", voice),
                ("welcome_message", welcome_message),
                ("generate_welcome_message", generate_welcome_message),
                ("project", project),
                ("default_language", default_language),
                ("additional_languages", additional_languages),
                ("multilingual_mode", multilingual_mode),
                ("audio_speed", audio_speed),
                ("phonic_tools", phonic_tools),
                ("boosted_keywords", boosted_keywords),
                ("min_words_to_interrupt", min_words_to_interrupt),
                ("generate_no_input_poke_text", generate_no_input_poke_text),
                ("no_input_poke_sec", no_input_poke_sec),
                ("no_input_poke_text", no_input_poke_text),
                ("no_input_end_conversation_sec", no_input_end_conversation_sec),
                ("websocket_timeout_sec", websocket_timeout_sec),
                ("intelligence_level", intelligence_level),
                ("phonic_model", phonic_model),
                ("is_welcome_message_interruptible", is_welcome_message_interruptible),
                ("vad_prebuffer_duration_ms", vad_prebuffer_duration_ms),
                ("vad_min_speech_duration_ms", vad_min_speech_duration_ms),
                ("vad_min_silence_duration_ms", vad_min_silence_duration_ms),
                ("vad_threshold", vad_threshold),
                ("enable_assistant_backchannel", enable_assistant_backchannel),
                ("assistant_backchannel_aggressiveness", assistant_backchannel_aggressiveness),
                ("pronunciation_dictionary", pronunciation_dictionary),
                ("template_variables", template_variables),
                ("enable_redaction", enable_redaction),
                ("enable_watermarking", enable_watermarking),
                ("mcp_servers", mcp_servers),
                ("observability_integrations", observability_integrations),
                ("configuration_endpoint", configuration_endpoint),
                ("additional_params", additional_params),
                ("configs_for_tools", configs_for_tools),
            )
            if is_given(value)
        }
        if not changes:
            return

        # Rotate the previous default into additional_languages when switching default_language so
        # it stays usable (and drop the new default, which the API forbids there), unless the caller
        # set additional_languages explicitly.
        new_default_language = changes.get("default_language")
        if (
            new_default_language is not None
            and new_default_language != self._opts.default_language
            and "additional_languages" not in changes
        ):
            previous_default_language = self._opts.default_language
            merged = (
                [previous_default_language] if is_given(previous_default_language) else []
            ) + (
                list(self._opts.additional_languages)
                if is_given(self._opts.additional_languages)
                else []
            )
            deduped: list[str] = []
            for lang in merged:
                if lang != new_default_language and lang not in deduped:
                    deduped.append(lang)
            changes["additional_languages"] = deduped

        changed = False
        for name, value in changes.items():
            if getattr(self._opts, name) != value:
                setattr(self._opts, name, value)
                changed = True
        if not changed:
            return

        # the tool definitions are built from the tool-related options: rebuild them so the reset
        # carries the new tool behavior
        if changes.keys() & {"configs_for_tools", "phonic_tools"}:
            self._rebuild_tool_definitions()

        if self._config_sent:
            self._close_current_generation(interrupted=True)
            self._pending_user_text = None
            self._send_reset()

    def push_video(self, frame: rtc.VideoFrame) -> None:
        logger.warning("push_video is not supported by the Phonic realtime model.")

    def _request_reply(
        self, command: _GenerateReply | _Say | None, what: str
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """Sends a reply request; the future resolves with the turn it opens."""
        if (old := self._pending_reply) is not None and not old.done():
            old.cancel()
        self._close_current_generation(interrupted=False)
        if command is not None:
            self.send_now(command)

        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.get_running_loop().create_future()
        self._pending_reply = fut

        def _expire() -> None:
            if not fut.done():
                fut.set_exception(llm.RealtimeError(f"{what} timed out."))

        handle = asyncio.get_running_loop().call_later(REPLY_TIMEOUT, _expire)

        def _on_done(f: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            handle.cancel()
            if self._pending_reply is f:
                self._pending_reply = None

        fut.add_done_callback(_on_done)
        return fut

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        if is_given(tools):
            logger.warning("per-response tools is not supported by Phonic Realtime API, ignoring")
        system_message = instructions if is_given(instructions) else None
        if self._pending_user_text:
            user_text = (
                f'The user sent the following text message: "{self._pending_user_text}". '
                "Please respond to their message."
            )
            system_message = f"{system_message}\n\n{user_text}" if system_message else user_text
            self._pending_user_text = None
        return self._request_reply(
            _GenerateReply(type="generate_reply", system_message=system_message), "generate_reply"
        )

    def say(self, text: str | AsyncIterable[str]) -> asyncio.Future[llm.GenerationCreatedEvent]:
        # say() never consumes buffered user text: drop it so it does not leak into a later reply
        self._pending_user_text = None
        if isinstance(text, str):
            return self._request_reply(_Say(type="say", text=text), "say()")
        fut = self._request_reply(None, "say()")
        self._say_tasks.spawn(self._say_collected(text, fut), name="phonic-say")
        return fut

    async def _say_collected(
        self, text: AsyncIterable[str], fut: asyncio.Future[llm.GenerationCreatedEvent]
    ) -> None:
        try:
            collected = "".join([chunk async for chunk in text])
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
            return
        if self._pending_reply is fut:
            self.send_now(_Say(type="say", text=collected))

    def commit_audio(self) -> None:
        logger.warning("commit_audio is not supported by the Phonic realtime model.")

    def clear_audio(self) -> None:
        logger.warning("clear_audio is not supported by the Phonic realtime model.")

    def interrupt(self) -> None:
        if self._current_generation:
            logger.warning(
                "interrupt() is not supported by Phonic realtime model. "
                "User interruptions are automatically handled by Phonic."
            )

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        logger.warning(
            "truncate is not supported by the Phonic realtime model. "
            "User interruptions are automatically handled by Phonic."
        )

    def _emit_socket_error(self, error: Exception, *, recoverable: bool) -> None:
        # A reply sent on the failed connection is not replayed on reconnect. Settle it now,
        # rather than leaving the caller waiting for a turn that can no longer arrive.
        pending = self._pending_reply
        self._pending_reply = None
        if pending is not None and not pending.done():
            pending.set_exception(llm.RealtimeError(str(error)))
        self._close_current_generation(interrupted=True)
        super()._emit_socket_error(error, recoverable=recoverable)

    async def aclose(self) -> None:
        await self._say_tasks.aclose()
        await super().aclose()
        self._close_current_generation(interrupted=False)
        if (fut := self._pending_reply) is not None:
            self._pending_reply = None
            fut.cancel()

    def _start_new_assistant_turn(self) -> None:
        if self._current_generation:
            self._close_current_generation(interrupted=True)

        response_id = utils.shortuuid("PS_")
        gen = self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            text_ch=utils.aio.Chan[str | TranscriptChunk](),
            audio_ch=utils.aio.Chan[rtc.AudioFrame](),
            response_id=response_id,
            input_id=utils.shortuuid("PI_"),
        )
        modalities = asyncio.get_running_loop().create_future()
        modalities.set_result(["audio", "text"])
        gen.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=response_id,
                text_stream=gen.text_ch,
                audio_stream=gen.audio_ch,
                modalities=modalities,
            )
        )
        generation_ev = llm.GenerationCreatedEvent(
            message_stream=gen.message_ch,
            function_stream=gen.function_ch,
            user_initiated=False,
            response_id=response_id,
        )
        if (fut := self._pending_reply) is not None and not fut.done():
            generation_ev.user_initiated = True
            fut.set_result(generation_ev)
        self.emit("generation_created", generation_ev)

    def _close_current_generation(self, interrupted: bool) -> None:
        gen = self._current_generation
        if not gen or gen.done:
            return
        if gen.output_text:
            self._chat_ctx.add_message(
                role="assistant",
                content=gen.output_text,
                id=gen.response_id,
                interrupted=interrupted,
            )
        gen.text_ch.send_nowait("")
        gen.text_ch.close()
        gen.audio_ch.close()
        gen.function_ch.close()
        gen.message_ch.close()
        gen.done = True
        self._current_generation = None

    def _handle_audio_chunk(self, message: _AudioChunk) -> None:
        text = message.get("text", "")
        # audio can arrive outside an assistant turn: text opens one
        if self._current_generation is None and text:
            self._start_new_assistant_turn()
        gen = self._current_generation
        if gen is None:
            return

        try:
            audio = base64.b64decode(message["audio"], validate=True)
        except binascii.Error as e:
            raise llm_realtime.protocol_error(self, f"audio_chunk audio is not base64: {e}") from e
        # the text and its audio come in one chunk: the text is stamped with the audio's span
        samples = len(audio) // (2 * PHONIC_NUM_CHANNELS)
        duration = samples / PHONIC_OUTPUT_SAMPLE_RATE
        if text:
            gen.push_text(
                TranscriptChunk(
                    text, start_time=gen.audio_cursor_sec, end_time=gen.audio_cursor_sec + duration
                )
            )
        if samples > 0:
            gen.audio_ch.send_nowait(
                rtc.AudioFrame(
                    data=audio[: samples * 2 * PHONIC_NUM_CHANNELS],
                    sample_rate=PHONIC_OUTPUT_SAMPLE_RATE,
                    num_channels=PHONIC_NUM_CHANNELS,
                    samples_per_channel=samples,
                )
            )
            gen.audio_cursor_sec += duration

    def _handle_input_text(self, transcript: str) -> None:
        item_id = utils.shortuuid("PI_")
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(item_id=item_id, transcript=transcript, is_final=True),
        )
        self._chat_ctx.add_message(role="user", content=transcript, id=item_id)

    def _handle_tool_call(self, message: _ToolCall) -> None:
        self._pending_tool_call_ids.add(message["tool_call_id"])
        if self._current_generation is None:
            self._start_new_assistant_turn()
        assert self._current_generation is not None
        self._current_generation.function_ch.send_nowait(
            llm.FunctionCall(
                call_id=message["tool_call_id"],
                name=message["tool_name"],
                arguments=json.dumps(message["parameters"]),
            )
        )
        # a tool call ends its turn; with allow_tool_chaining a chained call opens a new one
        self._close_current_generation(interrupted=False)
