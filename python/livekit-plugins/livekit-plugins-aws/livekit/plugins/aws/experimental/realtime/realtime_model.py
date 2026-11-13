from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import os
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import boto3

# The SDK ships annotated namespace modules without a recognized typing marker.
from aws_sdk_bedrock_runtime.client import (  # pyright: ignore[reportMissingTypeStubs]
    AsyncBedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)

# The SDK ships annotated namespace modules without a recognized typing marker.
from aws_sdk_bedrock_runtime.config import (  # pyright: ignore[reportMissingTypeStubs]
    AsyncBedrockRuntimeConfig,  # pyright: ignore[reportMissingTypeStubs]
)

# The SDK ships annotated namespace modules without a recognized typing marker.
from aws_sdk_bedrock_runtime.models import (  # pyright: ignore[reportMissingTypeStubs]
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
    InvokeModelWithBidirectionalStreamOutputChunk,
    ModelErrorException,
    ModelNotReadyException,
    ModelStreamErrorException,
    ModelTimeoutException,
    ThrottlingException,
    ValidationException,
)
from botocore.credentials import Credentials
from botocore.exceptions import BotoCoreError
from pydantic import ValidationError as PayloadValidationError
from smithy_aws_core.identity import AWSCredentialsIdentity
from smithy_aws_event_stream.exceptions import InvalidEventBytes
from smithy_core.aio.interfaces.identity import IdentityResolver
from smithy_core.aio.retries import StandardRetryStrategy
from smithy_http.aio.crt import AWSCRTHTTPClient

from livekit import rtc
from livekit.agents import (
    APIError,
    APIStatusError,
    TokenUsage,
    llm,
    utils,
)
from livekit.agents.metrics import Metadata, RealtimeModelMetrics
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ...log import logger
from .events import (
    SonicEventBuilder as seb,
    SonicOutputEvent,
    Tool,
    ToolConfiguration,
    ToolInputSchema,
    ToolSpec,
)
from .pretty_printer import AnsiColors, log_event_data, log_message
from .turn_tracker import TurnTracker
from .types import MODALITIES, REALTIME_MODELS, SONIC1_VOICES, SONIC2_VOICES, TURN_DETECTION

DEFAULT_INPUT_SAMPLE_RATE = 16000
DEFAULT_OUTPUT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1
DEFAULT_CHUNK_SIZE = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 1024
MAX_MESSAGE_SIZE = 1024
MAX_MESSAGES = 40
DEFAULT_MAX_SESSION_RESTART_ATTEMPTS = 3
DEFAULT_MAX_SESSION_RESTART_DELAY = 10
RECOVERABLE_VALIDATION_ERROR_MESSAGES = (
    "InternalErrorCode=531::RST_STREAM closed stream. HTTP/2 error code: NO_ERROR",
    "System instability detected. Please retry your request.",
)
# Session recycling: restart before 8-min AWS limit or credential expiry
# Override with LK_SESSION_MAX_DURATION env var for testing (e.g., "60" for 1 minute)
MAX_SESSION_DURATION_SECONDS = int(os.getenv("LK_SESSION_MAX_DURATION", 6 * 60))
# A recycle waits for the assistant's audio to stay quiet this long before closing the stream.
RECYCLE_QUIET_PERIOD_SECONDS = 1.0
# How long a recycle waits for the open generation's completionEnd after the prompt ends.
RECYCLE_COMPLETION_END_TIMEOUT_SECONDS = 2.0
# How long a closing stream's receive loop gets to end on its own before it is cancelled.
RESPONSE_DRAIN_TIMEOUT_SECONDS = 1.0
# How long a recycle for changed tools waits for the tool calls in flight to be answered, so
# their results reach the stream that asked for them.
TOOL_RESULTS_RECYCLE_TIMEOUT_SECONDS = 5.0
BARGE_IN_SIGNAL = '{ "interrupted" : true }'  # Nova Sonic's barge-in detection signal


def _is_recoverable_validation_error(exc: object) -> bool:
    message = getattr(exc, "message", str(exc))
    return any(text in message for text in RECOVERABLE_VALIDATION_ERROR_MESSAGES)


DEFAULT_SYSTEM_PROMPT = (
    "Your name is Sonic, and you are a friendly and enthusiastic voice assistant. "
    "You love helping people and having natural conversations. "
    "Be warm, conversational, and engaging. "
    "Keep your responses natural and concise for voice interaction. "
    "Do not repeat yourself. "
    "If you are not sure what the user means, ask them to confirm or clarify. "
    "If after asking for clarification you still do not understand, be honest and tell them you do not understand. "
    "Do not make up information or make assumptions. If you do not know the answer, say so. "
    "When making tool calls, inform the user that you are using a tool to generate the response. "
    "Avoid formatted lists or numbering and keep your output as a spoken transcript. "
    "\n\n"
    "CRITICAL LANGUAGE MIRRORING RULES:\n"
    "- Always reply in the language the user speaks. DO NOT mix with English unless the user does.\n"
    "- If the user talks in English, reply in English.\n"
    "- Please respond in the language the user is talking to you in. If you have a question or suggestion, ask it in the language the user is talking in.\n"
    "- Ensure that our communication remains in the same language as the user."
)

lk_bedrock_debug = int(os.getenv("LK_BEDROCK_DEBUG", 0))


@dataclass
class _RealtimeOptions:
    """Configuration container for a Sonic realtime session.

    Attributes:
        voice (str): Voice identifier used for TTS output.
        temperature (float): Sampling temperature controlling randomness; 1.0 is most deterministic.
        top_p (float): Nucleus sampling parameter; 0.0 considers all tokens.
        max_tokens (int): Maximum number of tokens the model may generate in a single response.
        tool_choice (llm.ToolChoice | None): Strategy that dictates how the model should invoke tools.
        region (str): AWS region hosting the Bedrock Sonic model endpoint.
        turn_detection (TURN_DETECTION): Turn-taking sensitivity - "HIGH", "MEDIUM" (default), or "LOW".
        modalities (MODALITIES): Input/output mode - "audio" for audio-only, "mixed" for audio + text input.
    """  # noqa: E501

    voice: str
    temperature: float
    top_p: float
    max_tokens: int
    tool_choice: llm.ToolChoice | None
    region: str
    turn_detection: TURN_DETECTION
    modalities: MODALITIES


@dataclass
class _MessageGeneration:
    """Grouping of streams that together represent one assistant message.

    Attributes:
        message_id (str): Unique identifier that ties together text and audio for a single assistant turn.
        text_ch (utils.aio.Chan[str]): Channel that yields partial text tokens as they arrive.
        audio_ch (utils.aio.Chan[rtc.AudioFrame]): Channel that yields audio frames for the same assistant turn.
    """  # noqa: E501

    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]


@dataclass
class _ResponseGeneration:
    """Book-keeping dataclass tracking the lifecycle of a Nova Sonic completion.

    Nova Sonic uses a completion model where one completionStart event begins a cycle
    that may contain multiple content blocks (USER ASR, TOOL, ASSISTANT text/audio).
    This generation stays open for the entire completion cycle.

    Attributes:
        completion_id (str): Nova Sonic's completionId that ties all events together.
        message_ch (utils.aio.Chan[llm.MessageGeneration]): Stream for assistant messages.
        function_ch (utils.aio.Chan[llm.FunctionCall]): Stream that emits function tool calls.
        response_id (str): LiveKit response_id for the assistant's response.
        message_gen (_MessageGeneration | None): Current message generation for assistant output.
        content_id_map (dict[str, str]): Map Nova Sonic contentId -> type (USER/ASSISTANT/TOOL).
        created_timestamp (float): Wall-clock time when the generation record was created.
        first_token_timestamp (float | None): Wall-clock time of first token emission.
        completed_timestamp (float | None): Wall-clock time when the turn fully completed.
        restart_attempts (int): Number of restart attempts for this specific completion.
    """  # noqa: E501

    completion_id: str
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    response_id: str
    message_gen: _MessageGeneration | None = None
    content_id_map: dict[str, str] = field(default_factory=dict[str, str])
    created_timestamp: float = field(default_factory=time.time)
    first_token_timestamp: float | None = None
    completed_timestamp: float | None = None
    restart_attempts: int = 0
    done_fut: asyncio.Future[None] | None = None  # Resolved when generation completes
    emitted: bool = False  # Track if generation_created event was emitted


class Boto3CredentialsResolver(IdentityResolver[AWSCredentialsIdentity, Mapping[str, object]]):
    """IdentityResolver that signs Bedrock requests with the boto3 credential chain.

    Temporary credentials are botocore's refreshable ones, which renew themselves while 10-15
    minutes of validity remain: each session signs with credentials that outlive it.
    """

    def __init__(self) -> None:
        self._credentials: Credentials | None = None

    async def get_identity(self, **kwargs: object) -> AWSCredentialsIdentity:
        """The current credentials, refreshed first when they near expiry.

        Raises:
            APIError: of kind ``auth`` when no complete credentials can be loaded.
        """
        try:
            if self._credentials is None:
                self._credentials = boto3.Session().get_credentials()
            if self._credentials is None:
                raise APIError("Unable to load AWS credentials", kind="auth")
            creds = self._credentials.get_frozen_credentials()
        except BotoCoreError as e:
            raise APIError(f"Failed to load AWS credentials: {e}", kind="auth") from e
        if not creds.access_key or not creds.secret_key:
            raise APIError("AWS credentials are incomplete", kind="auth")
        return AWSCredentialsIdentity(
            access_key_id=creds.access_key,
            secret_access_key=creds.secret_key,
            session_token=creds.token or None,
        )


class RealtimeModel(llm.RealtimeModel):
    """High-level entry point that conforms to the LiveKit RealtimeModel interface.

    The object is very light-weight-– it mainly stores default inference options and
    spawns a RealtimeSession when session() is invoked.
    """

    def __init__(
        self,
        *,
        model: REALTIME_MODELS | str = "amazon.nova-2-sonic-v1:0",
        modalities: MODALITIES = "mixed",
        voice: NotGivenOr[SONIC1_VOICES | SONIC2_VOICES | str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        region: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: TURN_DETECTION = "MEDIUM",
        generate_reply_timeout: float = 10.0,
    ):
        """Instantiate a new RealtimeModel.

        Args:
            model (REALTIME_MODELS | str): Bedrock model ID for realtime inference. Defaults to "amazon.nova-2-sonic-v1:0".
            modalities (MODALITIES): Input/output mode. "audio" for audio-only (Sonic 1.0), "mixed" for audio + text input (Sonic 2.0). Defaults to "mixed".
            voice (SONIC1_VOICES | SONIC2_VOICES | str | NotGiven): Voice id for TTS output. Defaults to "tiffany".
            temperature (float | NotGiven): Sampling temperature (0-1). Defaults to DEFAULT_TEMPERATURE.
            top_p (float | NotGiven): Nucleus sampling probability mass. Defaults to DEFAULT_TOP_P.
            max_tokens (int | NotGiven): Upper bound for tokens emitted by the model. Defaults to DEFAULT_MAX_TOKENS.
            tool_choice (llm.ToolChoice | None | NotGiven): Strategy for tool invocation ("auto", "required", or explicit function).
            region (str | NotGiven): AWS region of the Bedrock runtime endpoint.
            turn_detection (TURN_DETECTION): Turn-taking sensitivity. HIGH detects pauses quickly, LOW waits longer. Defaults to MEDIUM.
            generate_reply_timeout (float): Timeout in seconds for generate_reply() calls. Defaults to 10.0.
        """  # noqa: E501
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=True,
                audio_output=True,
                manual_function_calls=False,
                per_response_tool_choice=False,
            )
        )
        self._model = model
        self._generate_reply_timeout = generate_reply_timeout
        self._credentials_resolver = Boto3CredentialsResolver()
        # note: temperature and top_p do not follow industry standards and are defined slightly differently for Sonic  # noqa: E501
        # temperature ranges from 0.0 to 1.0, where 0.0 is the most random and 1.0 is the most deterministic  # noqa: E501
        # top_p ranges from 0.0 to 1.0, where 0.0 is the most random and 1.0 is the most deterministic  # noqa: E501
        self.temperature = temperature
        self.top_p = top_p
        self._opts = _RealtimeOptions(
            voice=voice if is_given(voice) else "tiffany",
            temperature=temperature if is_given(temperature) else DEFAULT_TEMPERATURE,
            top_p=top_p if is_given(top_p) else DEFAULT_TOP_P,
            max_tokens=max_tokens if is_given(max_tokens) else DEFAULT_MAX_TOKENS,
            tool_choice=tool_choice or None,
            region=region if is_given(region) else "us-east-1",
            turn_detection=turn_detection,
            modalities=modalities,
        )
        self._sessions = weakref.WeakSet[RealtimeSession]()

    @classmethod
    def with_nova_sonic_1(
        cls,
        *,
        voice: NotGivenOr[SONIC1_VOICES | str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        region: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: TURN_DETECTION = "MEDIUM",
        generate_reply_timeout: float = 10.0,
    ) -> RealtimeModel:
        """Create a RealtimeModel configured for Nova Sonic 1.0 (audio-only).

        Args:
            voice (SONIC1_VOICES | str | NotGiven): Voice id for TTS output. Import SONIC1_VOICES from livekit.plugins.aws.experimental.realtime for supported values. Defaults to "tiffany".
            temperature (float | NotGiven): Sampling temperature (0-1). Defaults to DEFAULT_TEMPERATURE.
            top_p (float | NotGiven): Nucleus sampling probability mass. Defaults to DEFAULT_TOP_P.
            max_tokens (int | NotGiven): Upper bound for tokens emitted. Defaults to DEFAULT_MAX_TOKENS.
            tool_choice (llm.ToolChoice | None | NotGiven): Strategy for tool invocation.
            region (str | NotGiven): AWS region. Defaults to "us-east-1".
            turn_detection (TURN_DETECTION): Turn-taking sensitivity. Defaults to "MEDIUM".
            generate_reply_timeout (float): Timeout for generate_reply() calls. Defaults to 10.0.

        Returns:
            RealtimeModel: Configured for Nova Sonic 1.0 with audio-only modalities.

        Example:
            model = RealtimeModel.with_nova_sonic_1(voice="matthew", tool_choice="auto")
        """
        return cls(
            model="amazon.nova-sonic-v1:0",
            modalities="audio",
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            region=region,
            turn_detection=turn_detection,
            generate_reply_timeout=generate_reply_timeout,
        )

    @classmethod
    def with_nova_sonic_2(
        cls,
        *,
        voice: NotGivenOr[SONIC2_VOICES | str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        region: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: TURN_DETECTION = "MEDIUM",
        generate_reply_timeout: float = 10.0,
    ) -> RealtimeModel:
        """Create a RealtimeModel configured for Nova Sonic 2.0 (audio + text input).

        Args:
            voice (SONIC2_VOICES | str | NotGiven): Voice id for TTS output. Import SONIC2_VOICES from livekit.plugins.aws.experimental.realtime for supported values. Defaults to "tiffany".
            temperature (float | NotGiven): Sampling temperature (0-1). Defaults to DEFAULT_TEMPERATURE.
            top_p (float | NotGiven): Nucleus sampling probability mass. Defaults to DEFAULT_TOP_P.
            max_tokens (int | NotGiven): Upper bound for tokens emitted. Defaults to DEFAULT_MAX_TOKENS.
            tool_choice (llm.ToolChoice | None | NotGiven): Strategy for tool invocation.
            region (str | NotGiven): AWS region. Defaults to "us-east-1".
            turn_detection (TURN_DETECTION): Turn-taking sensitivity. Defaults to "MEDIUM".
            generate_reply_timeout (float): Timeout for generate_reply() calls. Defaults to 10.0.

        Returns:
            RealtimeModel: Configured for Nova Sonic 2.0 with mixed modalities (audio + text input).

        Example:
            model = RealtimeModel.with_nova_sonic_2(voice="tiffany", max_tokens=10_000)
        """
        return cls(
            model="amazon.nova-2-sonic-v1:0",
            modalities="mixed",
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            region=region,
            turn_detection=turn_detection,
            generate_reply_timeout=generate_reply_timeout,
        )

    @classmethod
    def with_nova_sonic_2_5(
        cls,
        *,
        voice: NotGivenOr[SONIC2_VOICES | str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        region: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: TURN_DETECTION = "MEDIUM",
        generate_reply_timeout: float = 10.0,
    ) -> RealtimeModel:
        """Create a RealtimeModel configured for Nova Sonic 2.5 (audio + text input).

        Args:
            voice (SONIC2_VOICES | str | NotGiven): Voice id for TTS output. Import SONIC2_VOICES from livekit.plugins.aws.experimental.realtime for supported values. Defaults to "tiffany".
            temperature (float | NotGiven): Sampling temperature (0-1). Defaults to DEFAULT_TEMPERATURE.
            top_p (float | NotGiven): Nucleus sampling probability mass. Defaults to DEFAULT_TOP_P.
            max_tokens (int | NotGiven): Upper bound for tokens emitted. Defaults to DEFAULT_MAX_TOKENS.
            tool_choice (llm.ToolChoice | None | NotGiven): Strategy for tool invocation.
            region (str | NotGiven): AWS region. Defaults to "us-east-1".
            turn_detection (TURN_DETECTION): Turn-taking sensitivity. Defaults to "MEDIUM".
            generate_reply_timeout (float): Timeout for generate_reply() calls. Defaults to 10.0.

        Returns:
            RealtimeModel: Configured for Nova Sonic 2.5 with mixed modalities (audio + text input).

        Example:
            model = RealtimeModel.with_nova_sonic_2_5(voice="tiffany", max_tokens=10_000)
        """
        return cls(
            model="amazon.nova-2-5-sonic",
            modalities="mixed",
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            region=region,
            turn_detection=turn_detection,
            generate_reply_timeout=generate_reply_timeout,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def modalities(self) -> MODALITIES:
        """Input/output mode: "audio" for audio-only, "mixed" for audio + text input."""
        return self._opts.modalities

    @property
    def provider(self) -> str:
        return "Amazon"

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        """Return a new RealtimeSession bound to this model instance."""
        # disabling server-side turn detection is unsupported (can_disable_turn_detection=False)
        sess = RealtimeSession(
            self, options=self._opts, generate_reply_timeout=self._generate_reply_timeout
        )
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        """Close all active sessions."""
        pass


class RealtimeSession(  # noqa: F811
    llm.RealtimeSession[Literal["bedrock_server_event_received", "bedrock_client_event_queued"]]
):
    """Bidirectional streaming session against the Nova Sonic Bedrock runtime.

    The session owns two asynchronous tasks:

    1. _process_audio_input – pushes user mic audio and tool results to Bedrock.
    2. _process_responses – receives server events from Bedrock and converts them into
       LiveKit abstractions such as llm.MessageGeneration.

    A set of helper handlers (_handle_*) transform the low-level Bedrock
    JSON payloads into higher-level application events and keep
    _ResponseGeneration state in sync.
    """

    def __init__(
        self,
        realtime_model: RealtimeModel,
        *,
        options: _RealtimeOptions,
        generate_reply_timeout: float,
    ) -> None:
        """Create and wire-up a new realtime session.

        Args:
            realtime_model (RealtimeModel): Parent model instance that stores static
                inference options and the Smithy Bedrock client configuration.
        """
        super().__init__(
            realtime_model,
            input_sample_rate=DEFAULT_INPUT_SAMPLE_RATE,
            input_num_channels=DEFAULT_CHANNELS,
        )
        self._realtime_model: RealtimeModel = realtime_model
        self._opts = options
        self._generate_reply_timeout = generate_reply_timeout
        self._event_builder = seb(
            prompt_name=str(uuid.uuid4()),
            audio_content_name=str(uuid.uuid4()),
            model=self._realtime_model.model,
        )
        self._bstream = utils.media.AudioByteStream(
            DEFAULT_INPUT_SAMPLE_RATE, DEFAULT_CHANNELS, samples_per_channel=DEFAULT_CHUNK_SIZE
        )

        self._response_task = None
        self._audio_input_task = None
        self._stream_response = None
        self._bedrock_client = None
        self._pending_tools: set[str] = set()
        self._no_pending_tools = asyncio.Event()  # set while _pending_tools is empty
        self._no_pending_tools.set()
        self._is_sess_active = asyncio.Event()
        self._chat_ctx = llm.ChatContext.empty()
        self._tools = llm.ToolContext.empty()
        self._tool_results_ch = utils.aio.Chan[dict[str, str]]()
        # CRITICAL: Initialize futures as None for lazy creation
        # Creating futures in __init__ causes race conditions during session restart.
        # Futures are created in initialize_streams() when the event loop is guaranteed to exist.
        self._tools_ready: asyncio.Future[bool] | None = None
        self._instructions_ready: asyncio.Future[bool] | None = None
        self._chat_ctx_ready: asyncio.Future[bool] | None = None
        self._instructions = DEFAULT_SYSTEM_PROMPT
        self._audio_input_chan = utils.aio.Chan[bytes]()
        self._current_generation: _ResponseGeneration | None = None
        # Session recycling: proactively restart before credential expiry or 8-min limit
        self._session_recycle_task: asyncio.Task[None] | None = None
        self._tasks = utils.aio.TaskSet()
        self._last_audio_output_time: float = 0.0  # monotonic time of the last assistant audio
        # set while the assistant is not speaking: an idle session recycles at once
        self._audio_end_turn = asyncio.Event()
        self._audio_end_turn.set()
        self._pending_generation_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        self._sent_message_ids: set[str] = set()
        self._audio_message_ids: set[str] = set()
        self._no_gen_content_roles: dict[
            str, str
        ] = {}  # contentId → role for events without generation
        self._current_user_content_id: str | None = None  # track current user utterance
        # Signalled after await_output() returns (HTTP 200 received).
        self._stream_ready = asyncio.Event()
        # Signalled once the audio input's contentStart is sent. Interactive text waits for it:
        # Nova Sonic requires active audio to generate, and the two contentStarts must not overlap.
        self._audio_input_started = asyncio.Event()

        # Gate that pauses audio frame transmission while a text block is in-flight.
        # This prevents the race condition where an audioInput frame arrives at the
        # server between TEXT contentStart and TEXT contentEnd, which triggers:
        # "ValidationException: Chat history should be sent completely before streaming audio."
        # Uses a Lock instead of an Event so that concurrent text sends (e.g.,
        # update_chat_ctx + generate_reply firing at the same time) serialize
        # properly — audio only resumes after ALL text blocks have completed.
        self._text_block_lock = asyncio.Lock()

        self._event_handlers = {
            "completion_start": self._handle_completion_start_event,
            "audio_output_content_start": self._handle_audio_output_content_start_event,
            "audio_output_content": self._handle_audio_output_content_event,
            "audio_output_content_end": self._handle_audio_output_content_end_event,
            "text_output_content_start": self._handle_text_output_content_start_event,
            "text_output_content": self._handle_text_output_content_event,
            "text_output_content_end": self._handle_text_output_content_end_event,
            "tool_output_content_start": self._handle_tool_output_content_start_event,
            "tool_output_content": self._handle_tool_output_content_event,
            "tool_output_content_end": self._handle_tool_output_content_end_event,
            "completion_end": self._handle_completion_end_event,
            "usage": self._handle_usage_event,
            "other_event": self._handle_other_event,
        }
        self._turn_tracker = TurnTracker(
            cast(Callable[[str, Any], None], self.emit),
            self.emit_generation_event,
        )

        # Create main task to manage session lifecycle
        self._main_atask = self._tasks.spawn(
            self.initialize_streams(), name="RealtimeSession.initialize_streams"
        )

    async def _initialize_client(self) -> None:
        """Create the duplex Bedrock client with SDK retries disabled."""
        config = await AsyncBedrockRuntimeConfig.resolve(
            endpoint_uri=f"https://bedrock-runtime.{self._opts.region}.amazonaws.com",
            region=self._opts.region,
            aws_credentials_identity_resolver=self._realtime_model._credentials_resolver,
            user_agent_extra="x-client-framework:livekit-plugins-aws[realtime]",
            transport=AWSCRTHTTPClient(),
            retry_strategy=StandardRetryStrategy(max_attempts=1),
        )
        self._bedrock_client = AsyncBedrockRuntimeClient(config=config)

    def _start_session_recycle_timer(self) -> None:
        """Start the session recycling timer."""
        # a recycle restarts the timer from inside the timer's own task
        if self._session_recycle_task is not asyncio.current_task():
            if self._session_recycle_task is not None:
                self._session_recycle_task.cancel()

        # within the validity botocore keeps on refreshable credentials (over 10 minutes)
        duration = MAX_SESSION_DURATION_SECONDS

        self._session_recycle_task = self._tasks.spawn(
            self._session_recycle_timer(duration), name="RealtimeSession._session_recycle_timer"
        )

    async def _session_recycle_timer(self, duration: float) -> None:
        """Background task that triggers session recycling after duration seconds."""
        try:
            logger.info(f"[SESSION] Recycle timer started, will fire in {duration:.0f}s")
            await asyncio.sleep(duration)

            if not self._is_sess_active.is_set():
                logger.debug("[SESSION] Session no longer active, skipping recycle")
                return

            logger.info(
                f"[SESSION] Session duration limit reached ({duration:.0f}s), initiating recycle"
            )

            # Step 1: Wait for assistant to finish speaking (AUDIO contentEnd with END_TURN)
            if not self._audio_end_turn.is_set():
                logger.info(
                    "[SESSION] Waiting for assistant to finish speaking (AUDIO END_TURN)..."
                )
                await self._audio_end_turn.wait()
                logger.debug("[SESSION] Assistant finished speaking")

            # Step 2: Wait for the assistant's audio to stay quiet
            logger.debug("[SESSION] Waiting for audio to fully stop...")
            while (
                quiet := time.monotonic() - self._last_audio_output_time
            ) < RECYCLE_QUIET_PERIOD_SECONDS:
                await asyncio.sleep(RECYCLE_QUIET_PERIOD_SECONDS - quiet)

            await self._graceful_session_recycle()

        except asyncio.CancelledError:
            logger.debug("[SESSION] Recycle timer cancelled")
            raise
        except Exception as e:
            logger.error(f"[SESSION] Error in recycle timer: {e}")

    async def _graceful_session_recycle(self) -> None:
        """Gracefully recycle the session, preserving conversation state."""
        logger.info("[SESSION] Starting graceful session recycle")

        # Step 1: Drain any pending tool results
        logger.debug("[SESSION] Draining pending tool results...")
        while True:
            try:
                tool_result = self._tool_results_ch.recv_nowait()
                logger.debug(f"[TOOL] Draining pending result: {tool_result['tool_use_id']}")
                await self._send_tool_events(tool_result["tool_use_id"], tool_result["tool_result"])
            except utils.aio.ChanEmpty:
                logger.debug("[SESSION] No more pending tool results")
                break
            except Exception as e:
                logger.warning(f"[SESSION] Error draining tool result: {e}")
                break

        # Step 2: End the prompt, so Nova Sonic ends the open generation and the output stream
        logger.info("[SESSION] Sending close events to Nova Sonic...")
        if self._stream_response:
            for event in self._event_builder.create_prompt_end_block():
                await self._send_raw_event(event)
        if self._current_generation and self._current_generation.done_fut:
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._current_generation.done_fut),
                    timeout=RECYCLE_COMPLETION_END_TIMEOUT_SECONDS,
                )
                logger.debug("[SESSION] Generation completed (completionEnd received)")
            except TimeoutError:
                logger.warning("[SESSION] Timeout waiting for completionEnd, proceeding anyway")
                self._close_current_generation()

        # Step 3: Signal tasks to stop, and let the receive loop end with the stream
        self._is_sess_active.clear()
        await self._drain_response_task()

        # Step 4: Cancel audio input task (blocked on channel, won't exit naturally)
        if self._audio_input_task is not None:
            await utils.aio.cancel_and_wait(self._audio_input_task)

        # Step 5: Close the stream
        if self._stream_response:
            try:
                await self._stream_response.input_stream.close()
            except Exception as e:
                logger.debug(f"[SESSION] Error closing stream (expected): {e}")

        # Step 6: Reset state for new session
        self._stream_response = None
        self._bedrock_client = None
        self._event_builder = seb(
            prompt_name=str(uuid.uuid4()),
            audio_content_name=str(uuid.uuid4()),
            model=self._realtime_model.model,
        )
        self._tool_results_ch = utils.aio.Chan[dict[str, str]]()
        self._audio_input_chan = utils.aio.Chan[bytes]()
        logger.debug("[SESSION] Created fresh tool results and audio input channels")
        self._audio_end_turn.set()
        self._stream_ready.clear()
        self._audio_input_started.clear()

        # Step 7: Start new session with preserved state
        await self.initialize_streams(is_restart=True)

        logger.info("[SESSION] Session recycled successfully")

    async def _drain_response_task(self) -> None:
        """Let the receive loop end with its closing stream; cancel it if it does not."""
        task = self._response_task
        if task is None or task.done():
            return
        if not self._stream_ready.is_set():
            # still waiting for the stream to open: no output will end it
            await utils.aio.cancel_and_wait(task)
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=RESPONSE_DRAIN_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.debug("[SESSION] Response task timeout, cancelling...")
            await utils.aio.cancel_and_wait(task)
        except Exception:
            pass  # the receive loop reported its own failure

    async def _send_raw_event(self, event_json: str) -> None:
        """Low-level helper that serialises event_json and forwards it to the bidirectional stream.

        Args:
            event_json (str): The JSON payload (already in Bedrock wire format) to queue.

        Raises:
            Exception: Propagates any failures returned by the Bedrock runtime client.
        """
        if not self._stream_response:
            logger.warning("stream not initialized; dropping event (this should never occur)")
            return

        # Log the full JSON being sent (skip audio events to avoid log spam)
        if '"audioInput"' not in event_json:
            logger.debug("[SEND] sending event", extra={"lk.pii.event_json": event_json})

        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )

        try:
            await self._stream_response.input_stream.send(event)
        except Exception as e:
            logger.exception("Error sending event")
            err_msg = getattr(e, "message", str(e))
            request_id = None
            try:
                request_id = err_msg.split(" ")[0].split("=")[1]
            except Exception:
                pass

            self.emit(
                "error",
                llm.RealtimeModelError(
                    timestamp=time.monotonic(),
                    label=self._realtime_model.label,
                    error=APIStatusError(
                        message=err_msg,
                        status_code=500,
                        request_id=request_id,
                        body=e,
                    ),
                    recoverable=False,
                ),
            )
            raise

    def _serialize_tool_config(self) -> ToolConfiguration | None:
        """Convert self.tools into the JSON structure expected by Sonic.

        Returns:
            ToolConfiguration | None: None when no tools are present, otherwise a complete config block.
        """  # noqa: E501
        tool_cfg = None
        if self.tools.function_tools:
            tools: list[Tool] = []
            for name, f in self.tools.function_tools.items():
                if isinstance(f, llm.FunctionTool):
                    description = f.info.description
                    input_schema = llm.utils.build_legacy_openai_schema(f, internally_tagged=True)[
                        "parameters"
                    ]
                else:
                    info = f.info
                    description = info.raw_schema.get("description")
                    raw_schema = info.raw_schema
                    # Safely access parameters with fallback
                    input_schema = raw_schema.get(
                        "parameters",
                        raw_schema.get("input_schema", {"type": "object", "properties": {}}),
                    )

                tool = Tool(
                    toolSpec=ToolSpec(
                        name=name,
                        description=description or "No description provided",
                        # Pydantic's static constructor uses the alias despite populate_by_name.
                        inputSchema=ToolInputSchema(json_=json.dumps(input_schema)),  # pyright: ignore[reportCallIssue]
                    )
                )
                tools.append(tool)
            tool_choice = self._tool_choice_adapter(self._opts.tool_choice)
            logger.debug(f"TOOL CHOICE: {tool_choice}")
            tool_cfg = ToolConfiguration(tools=tools, toolChoice=tool_choice)

        return tool_cfg

    async def initialize_streams(self, is_restart: bool = False) -> None:
        """Open the Bedrock bidirectional stream and spawn background worker tasks.

        This coroutine is idempotent and can be invoked again when recoverable
        errors (e.g. timeout, throttling) require a fresh session.

        Args:
            is_restart (bool, optional): Marks whether we are re-initialising an
                existing session after an error. Defaults to False.
        """
        try:
            if not self._bedrock_client:
                logger.info("Creating Bedrock client")
                await self._initialize_client()
            assert self._bedrock_client is not None, "bedrock_client is None"

            logger.info("Initializing Bedrock stream")
            t0 = time.perf_counter()
            self._stream_response = (
                await self._bedrock_client.invoke_model_with_bidirectional_stream(
                    InvokeModelWithBidirectionalStreamOperationInput(
                        model_id=self._realtime_model.model
                    )
                )
            )
            self._report_connection_acquired(time.perf_counter() - t0)

            if not is_restart:
                # Lazy-initialize futures if needed
                if self._tools_ready is None:
                    self._tools_ready = asyncio.get_running_loop().create_future()
                if self._instructions_ready is None:
                    self._instructions_ready = asyncio.get_running_loop().create_future()
                if self._chat_ctx_ready is None:
                    self._chat_ctx_ready = asyncio.get_running_loop().create_future()

                pending_events: list[asyncio.Future[bool]] = []
                if not self.tools.function_tools:
                    pending_events.append(self._tools_ready)
                if not self._instructions_ready.done():
                    pending_events.append(self._instructions_ready)
                if not self._chat_ctx_ready.done():
                    pending_events.append(self._chat_ctx_ready)

                # note: can't know during sess init whether tools were not added
                # or if they were added haven't yet been updated
                # therefore in the case there are no tools, we wait the entire timeout
                try:
                    if pending_events:
                        await asyncio.wait_for(asyncio.gather(*pending_events), timeout=0.5)
                except TimeoutError:
                    if self._tools_ready and not self._tools_ready.done():
                        logger.warning("Tools not ready after 500ms, continuing without them")

                    if self._instructions_ready and not self._instructions_ready.done():
                        logger.warning(
                            "Instructions not received after 500ms, proceeding with default instructions"  # noqa: E501
                        )
                    if self._chat_ctx_ready and not self._chat_ctx_ready.done():
                        logger.warning(
                            "Chat context not received after 500ms, proceeding with empty chat context"  # noqa: E501
                        )

            logger.info(f"Initializing Bedrock session with realtime options: {self._opts}")
            # there is a 40-message limit on the chat context
            if len(self._chat_ctx.items) > MAX_MESSAGES:
                logger.warning(
                    f"Chat context has {len(self._chat_ctx.items)} messages, truncating to {MAX_MESSAGES}"  # noqa: E501
                )
                self._chat_ctx.truncate(max_items=MAX_MESSAGES)

            # On restart, ensure chat history starts with USER (Nova Sonic requirement)
            restart_ctx = self._chat_ctx
            if is_restart and self._chat_ctx.items:
                first_item = self._chat_ctx.items[0]
                if first_item.type == "message" and first_item.role == "assistant":
                    restart_ctx = self._chat_ctx.copy()
                    dummy_msg = llm.ChatMessage(role="user", content=["[Resuming conversation]"])
                    restart_ctx.items.insert(0, dummy_msg)
                    logger.debug("[SESSION] Added dummy USER message to start of chat history")

            # On restart, if the last message is from the user, send it as
            # interactive text instead of non-interactive history. This triggers
            # Nova Sonic to generate a response, since non-interactive history
            # alone won't prompt the model to act.
            interactive_user_text: str | None = None
            if is_restart and restart_ctx.items:
                last_item = restart_ctx.items[-1]
                if (
                    last_item.type == "message"
                    and last_item.role == "user"
                    and last_item.raw_text_content
                    and last_item.raw_text_content.strip()
                ):
                    interactive_user_text = last_item.raw_text_content.strip()
                    restart_ctx = restart_ctx.copy()
                    restart_ctx.items.pop()
                    logger.debug(
                        "[SESSION] Popped last user message for interactive send",
                        extra={"lk.pii.user_text": f"{interactive_user_text[:60]}..."},
                    )

            init_events, history_events = self._event_builder.create_prompt_start_block(
                voice_id=self._opts.voice,
                sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE,  # type: ignore
                system_content=self._instructions,
                chat_ctx=restart_ctx,
                tool_configuration=self._serialize_tool_config(),
                max_tokens=self._opts.max_tokens,
                top_p=self._opts.top_p,
                temperature=self._opts.temperature,
                endpointing_sensitivity=self._opts.turn_detection,
            )

            # Step 1: Send session init events (session start, prompt start, system prompt)
            for event in init_events:
                await self._send_raw_event(event)

            # Start session recycling timer
            self._start_session_recycle_timer()

            # Step 2: Send history events with small delays between them
            for event in history_events:
                await self._send_raw_event(event)
                await asyncio.sleep(0.01)

            # Step 3: Start response reader first (calls await_output, sets _stream_ready)
            self._response_task = self._tasks.spawn(
                self._process_responses(), name="RealtimeSession._process_responses"
            )

            # Step 4: Start audio input (waits for _stream_ready before sending audio_content_start)
            self._audio_input_task = self._tasks.spawn(
                self._process_audio_input(), name="RealtimeSession._process_audio_input"
            )
            self._is_sess_active.set()

            # Step 5: If we popped a user message from history, send it as
            # interactive text now to trigger Nova Sonic to respond.
            if interactive_user_text:
                await self._audio_input_started.wait()
                logger.debug(
                    "[SESSION] Sending interactive user text",
                    extra={"lk.pii.user_text": f"{interactive_user_text[:60]}..."},
                )
                await self._send_text_message(interactive_user_text, interactive=True)

            logger.debug("Stream initialized successfully")
        except Exception as e:
            logger.debug(f"Failed to initialize stream: {str(e)}")
            raise

    def emit_generation_event(self) -> None:
        """Publish a llm.GenerationCreatedEvent to external subscribers.

        This can be called multiple times for the same generation:
        - Once from _create_response_generation() when a NEW generation is created
        - Once from TurnTracker when TOOL_OUTPUT_CONTENT_START or ASSISTANT_SPEC_START arrives

        The TurnTracker emission is critical for tool calls - it happens at the right moment
        for the framework to start listening before the tool call is emitted.
        """
        if self._current_generation is None:
            logger.debug("[GEN] emit_generation_event called but no generation exists - ignoring")
            return

        # Log whether this is first or re-emission for tool call
        if self._current_generation.emitted:
            logger.debug(
                f"[GEN] EMITTING generation_created (re-emit for tool call) for response_id={self._current_generation.response_id}"
            )
        else:
            logger.debug(
                f"[GEN] EMITTING generation_created for response_id={self._current_generation.response_id}"
            )

        self._current_generation.emitted = True
        generation_ev = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
            response_id=self._current_generation.response_id,
        )
        self.emit("generation_created", generation_ev)

        # Resolve pending generate_reply future if exists
        if self._pending_generation_fut and not self._pending_generation_fut.done():
            self._pending_generation_fut.set_result(generation_ev)
            self._pending_generation_fut = None

    async def _handle_event(self, raw_event: object) -> None:
        """Validate a Bedrock payload before dispatching its known fields."""
        try:
            event_data = SonicOutputEvent.model_validate(raw_event)
        except PayloadValidationError as error:
            raise APIError(
                "invalid Nova Sonic event", kind="protocol", detail=json.dumps(raw_event)
            ) from error
        event_type = self._event_builder.get_event_type(event_data)
        event_handler = self._event_handlers.get(event_type)
        if event_handler:
            await event_handler(event_data)
            self._turn_tracker.feed(event_data)
        else:
            logger.warning(f"No event handler found for event type: {event_type}")

    async def _handle_completion_start_event(self, event_data: SonicOutputEvent) -> None:
        """Handle completionStart - create new generation for this completion cycle."""
        log_event_data(event_data)
        self._create_response_generation()

    def _create_response_generation(self) -> None:
        """Instantiate _ResponseGeneration and emit the GenerationCreated event.

        Can be called multiple times - will reuse existing generation but ensure
        message structure exists.
        """
        generation_created = False
        if self._current_generation is None:
            completion_id = "unknown"  # Will be set from events
            response_id = str(uuid.uuid4())

            logger.debug(f"[GEN] Creating NEW generation, response_id={response_id}")
            self._current_generation = _ResponseGeneration(
                completion_id=completion_id,
                message_ch=utils.aio.Chan(),
                function_ch=utils.aio.Chan(),
                response_id=response_id,
                done_fut=asyncio.get_running_loop().create_future(),
            )
            generation_created = True
        else:
            logger.debug(
                f"[GEN] Generation already exists: response_id={self._current_generation.response_id}, emitted={self._current_generation.emitted}"
            )

        # Always ensure message structure exists (even if generation already exists)
        if self._current_generation.message_gen is None:
            logger.debug(
                f"[GEN] Creating message structure for response_id={self._current_generation.response_id}"
            )
            msg_gen = _MessageGeneration(
                message_id=self._current_generation.response_id,
                text_ch=utils.aio.Chan(),
                audio_ch=utils.aio.Chan(),
            )
            msg_modalities = asyncio.Future[list[Literal["text", "audio"]]]()
            msg_modalities.set_result(
                ["audio", "text"] if self._realtime_model.capabilities.audio_output else ["text"]
            )

            self._current_generation.message_gen = msg_gen
            self._current_generation.message_ch.send_nowait(
                llm.MessageGeneration(
                    message_id=msg_gen.message_id,
                    text_stream=msg_gen.text_ch,
                    audio_stream=msg_gen.audio_ch,
                    modalities=msg_modalities,
                )
            )
        else:
            logger.debug(
                f"[GEN] Message structure already exists for response_id={self._current_generation.response_id}"
            )

        # Only emit generation event if we created a new generation
        if generation_created:
            logger.debug("[GEN] New generation created - calling emit_generation_event()")
            self.emit_generation_event()

    # will be completely ignoring post-ASR text events
    async def _handle_text_output_content_start_event(self, event_data: SonicOutputEvent) -> None:
        """Handle text_output_content_start - track content type."""
        payload = event_data.event.contentStart
        assert payload is not None
        log_event_data(event_data)

        role = payload.role

        # SPECULATIVE text blocks are incremental previews within the same response.
        # Don't close the generation here — keep one generation open per response.
        # Close happens on END_TURN, barge-in, tool call, or completionEnd.
        if role == "ASSISTANT":
            additional_fields = payload.additionalModelFields
            if "SPECULATIVE" in additional_fields:
                logger.debug("[GEN] ASSISTANT SPECULATIVE text received")
                self._create_response_generation()  # reuses existing if present

        # CRITICAL: Check if generation exists before accessing
        # Barge-in can set _current_generation to None between the creation above and here.
        # Without this check, we crash on interruptions.
        if self._current_generation is None:
            # Track role for events that arrive without a generation
            content_id = payload.contentId
            if content_id:
                self._no_gen_content_roles[content_id] = role
            logger.debug("No generation exists - ignoring content_start event")
            return

        content_id = payload.contentId

        # Track what type of content this is
        if role == "USER":
            self._current_generation.content_id_map[content_id] = "USER_ASR"
        elif role == "ASSISTANT":
            additional_fields = payload.additionalModelFields
            if "SPECULATIVE" in additional_fields:
                self._current_generation.content_id_map[content_id] = "ASSISTANT_TEXT"
            elif "FINAL" in additional_fields:
                self._current_generation.content_id_map[content_id] = "ASSISTANT_FINAL"

    async def _handle_text_output_content_event(self, event_data: SonicOutputEvent) -> None:
        """Stream partial text tokens into the current generation."""
        payload = event_data.event.textOutput
        assert payload is not None
        log_event_data(event_data)

        if self._current_generation is None:
            # No active generation. This happens for USER ASR and ASSISTANT FINAL
            # text arriving between turns.
            content_id = payload.contentId
            text_content = payload.content

            if text_content == BARGE_IN_SIGNAL:
                return

            role = self._no_gen_content_roles.get(content_id, "USER")
            if role == "USER":
                self._update_chat_ctx(role="user", text_content=text_content, content_id=content_id)
            elif role == "ASSISTANT":
                self._update_chat_ctx(role="assistant", text_content=text_content)
            return

        content_id = payload.contentId
        text_content = payload.content

        # Nova Sonic's automatic barge-in detection
        if text_content == BARGE_IN_SIGNAL:
            idx = self._chat_ctx.find_insertion_index(created_at=time.time()) - 1
            if idx >= 0 and (item := self._chat_ctx.items[idx]).type == "message":
                item.interrupted = True
                logger.debug("Barge-in detected - marked message as interrupted")

            # Close generation on barge-in unless tools are pending
            if not self._pending_tools:
                self._close_current_generation()
            else:
                logger.debug(f"Keeping generation open - {len(self._pending_tools)} pending tools")
            return

        content_type = self._current_generation.content_id_map.get(content_id)

        if content_type == "USER_ASR":
            logger.debug("input transcription updated", extra={"lk.pii.text_content": text_content})
            self._update_chat_ctx(role="user", text_content=text_content, content_id=content_id)

        elif content_type == "ASSISTANT_TEXT":
            # Set first token timestamp if not already set
            if self._current_generation.first_token_timestamp is None:
                self._current_generation.first_token_timestamp = time.time()

            # Stream text to LiveKit
            if self._current_generation.message_gen:
                self._current_generation.message_gen.text_ch.send_nowait(text_content)
            self._update_chat_ctx(role="assistant", text_content=text_content)

    def _update_chat_ctx(
        self, role: llm.ChatRole, text_content: str, content_id: str | None = None
    ) -> None:
        """
        Update the chat context with the latest ASR text while guarding against model limitations:
            a) 40 total messages limit
            b) 1kB message size limit
        """
        logger.debug(
            "Updating chat context", extra={"role": role, "lk.pii.text_content": text_content}
        )

        # Start a new message when the user contentId changes (new utterance)
        force_new = False
        if role == "user" and content_id is not None:
            if self._current_user_content_id != content_id:
                force_new = True
                self._current_user_content_id = content_id

        if len(self._chat_ctx.items) == 0 or force_new:
            msg = self._chat_ctx.add_message(role=role, content=text_content)
            if role == "user":
                self._audio_message_ids.add(msg.id)
            if len(self._chat_ctx.items) > MAX_MESSAGES:
                self._chat_ctx.truncate(max_items=MAX_MESSAGES)
        else:
            prev_utterance = self._chat_ctx.items[-1]
            if prev_utterance.type == "message" and prev_utterance.role == role:
                if isinstance(prev_content := prev_utterance.content[0], str) and (
                    len(prev_content.encode("utf-8")) + len(text_content.encode("utf-8"))
                    < MAX_MESSAGE_SIZE
                ):
                    prev_utterance.content[0] = "\n".join([prev_content, text_content])
                else:
                    msg = self._chat_ctx.add_message(role=role, content=text_content)
                    if role == "user":
                        self._audio_message_ids.add(msg.id)
                    if len(self._chat_ctx.items) > MAX_MESSAGES:
                        self._chat_ctx.truncate(max_items=MAX_MESSAGES)
            else:
                msg = self._chat_ctx.add_message(role=role, content=text_content)
                if role == "user":
                    self._audio_message_ids.add(msg.id)
                if len(self._chat_ctx.items) > MAX_MESSAGES:
                    self._chat_ctx.truncate(max_items=MAX_MESSAGES)

    # cannot rely on this event for user b/c stopReason=PARTIAL_TURN always for user
    async def _handle_text_output_content_end_event(self, event_data: SonicOutputEvent) -> None:
        """Handle text content end."""
        log_event_data(event_data)

    async def _handle_tool_output_content_start_event(self, event_data: SonicOutputEvent) -> None:
        """Track tool content start."""
        payload = event_data.event.contentStart
        assert payload is not None
        log_event_data(event_data)

        # Ensure generation exists
        self._create_response_generation()

        if self._current_generation is None:
            return

        content_id = payload.contentId
        self._current_generation.content_id_map[content_id] = "TOOL"

    async def _handle_tool_output_content_event(self, event_data: SonicOutputEvent) -> None:
        """Execute the referenced tool locally and queue results."""
        payload = event_data.event.toolUse
        assert payload is not None
        log_event_data(event_data)

        if self._current_generation is None:
            logger.warning("tool_output_content received without active generation")
            return

        tool_use_id = payload.toolUseId
        tool_name = payload.toolName
        args = payload.content

        # Nova Sonic sometimes double-encodes tool arguments: the outer JSON parse
        # yields a string whose contents are themselves a JSON object string
        # (e.g. "\"{\\\"order_id\\\":\\\"1234\\\"}\"").
        # Only peel one layer when the inner string is a JSON object so that
        # legitimate string-valued schemas (e.g. content="hello") are preserved.
        try:
            parsed = json.loads(args)
            if isinstance(parsed, str):
                try:
                    inner = json.loads(parsed)
                    if isinstance(inner, dict):
                        args = parsed
                except (json.JSONDecodeError, TypeError):
                    pass  # inner string is a plain value, leave args untouched
        except (json.JSONDecodeError, TypeError):
            pass

        # Emit function call to LiveKit framework
        self._current_generation.function_ch.send_nowait(
            llm.FunctionCall(call_id=tool_use_id, name=tool_name, arguments=args)
        )
        self._pending_tools.add(tool_use_id)
        self._no_pending_tools.clear()
        logger.debug(f"Tool call emitted: {tool_name} (id={tool_use_id})")

        # CRITICAL: Close generation after tool call emission
        # The LiveKit framework expects the generation to close so it can call update_chat_ctx()
        # with the tool results. A new generation will be created when Nova Sonic sends the next
        # ASSISTANT SPECULATIVE text event with the tool response.
        logger.debug("Closing generation to allow tool result delivery")
        self._close_current_generation()

    async def _handle_tool_output_content_end_event(self, event_data: SonicOutputEvent) -> None:
        log_event_data(event_data)

    async def _handle_audio_output_content_start_event(self, event_data: SonicOutputEvent) -> None:
        """Track audio content start."""
        payload = event_data.event.contentStart
        assert payload is not None
        self._audio_end_turn.clear()  # the assistant is speaking: a recycle waits for its end
        if self._current_generation is not None:
            log_event_data(event_data)
            content_id = payload.contentId
            self._current_generation.content_id_map[content_id] = "ASSISTANT_AUDIO"

    async def _handle_audio_output_content_event(self, event_data: SonicOutputEvent) -> None:
        """Decode base64 audio from Bedrock and forward it to the audio stream."""
        payload = event_data.event.audioOutput
        assert payload is not None
        if self._current_generation is None or self._current_generation.message_gen is None:
            return

        content_id = payload.contentId
        content_type = self._current_generation.content_id_map.get(content_id)

        if content_type == "ASSISTANT_AUDIO":
            audio_content = payload.content
            audio_bytes = base64.b64decode(audio_content)
            self._current_generation.message_gen.audio_ch.send_nowait(
                rtc.AudioFrame(
                    data=audio_bytes,
                    sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE,
                    num_channels=DEFAULT_CHANNELS,
                    samples_per_channel=len(audio_bytes) // 2,
                )
            )
            # Track when we last received audio output (for session recycling)
            self._last_audio_output_time = time.monotonic()

    async def _handle_audio_output_content_end_event(self, event_data: SonicOutputEvent) -> None:
        """Handle audio content end - track END_TURN for session recycling."""
        log_event_data(event_data)

        # Check if this is END_TURN (assistant finished speaking)
        content_end = event_data.event.contentEnd
        self._audio_end_turn.set()  # an interrupted turn ends the speech too
        if content_end is not None and content_end.stopReason == "END_TURN":
            logger.debug("[SESSION] AUDIO END_TURN received - assistant finished speaking")
            self._close_current_generation()

    def _close_current_generation(self) -> None:
        """Helper that closes all channels of the active generation."""
        if self._current_generation is None:
            return

        response_id = self._current_generation.response_id
        wasemitted = self._current_generation.emitted

        # Set completed timestamp
        if self._current_generation.completed_timestamp is None:
            self._current_generation.completed_timestamp = time.time()

        # Close message channels
        if self._current_generation.message_gen:
            if not self._current_generation.message_gen.audio_ch.closed:
                self._current_generation.message_gen.audio_ch.close()
            if not self._current_generation.message_gen.text_ch.closed:
                self._current_generation.message_gen.text_ch.close()

        # Close generation channels
        if not self._current_generation.message_ch.closed:
            self._current_generation.message_ch.close()
        if not self._current_generation.function_ch.closed:
            self._current_generation.function_ch.close()

        # Resolve done_fut to signal generation is complete (for session recycling)
        if self._current_generation.done_fut and not self._current_generation.done_fut.done():
            self._current_generation.done_fut.set_result(None)

        logger.debug(f"[GEN] CLOSED generation response_id={response_id}, wasemitted={wasemitted}")
        self._current_generation = None

    async def _handle_completion_end_event(self, event_data: SonicOutputEvent) -> None:
        """Handle completionEnd - close the generation for this completion cycle."""
        log_event_data(event_data)

        # Close generation if still open
        if self._current_generation:
            logger.debug("completionEnd received, closing generation")
            self._close_current_generation()

    async def _handle_other_event(self, event_data: SonicOutputEvent) -> None:
        log_event_data(event_data)

    async def _handle_usage_event(self, event_data: SonicOutputEvent) -> None:
        payload = event_data.event.usageEvent
        assert payload is not None
        # log_event_data(event_data)
        input_tokens = payload.details.delta.input
        output_tokens = payload.details.delta.output

        # Calculate metrics from timestamps
        duration = 0.0
        ttft: float | None = None
        tokens_per_second = 0.0

        if self._current_generation is not None:
            created_ts = self._current_generation.created_timestamp
            first_token_ts = self._current_generation.first_token_timestamp
            completed_ts = self._current_generation.completed_timestamp

            # Calculate TTFT (time to first token)
            if first_token_ts is not None:
                ttft = first_token_ts - created_ts

            # Calculate duration (total time from creation to completion)
            if completed_ts is not None:
                duration = completed_ts - created_ts

            # Calculate tokens per second
            total_tokens = (
                input_tokens.speechTokens
                + input_tokens.textTokens
                + output_tokens.speechTokens
                + output_tokens.textTokens
            )
            if duration > 0:
                tokens_per_second = total_tokens / duration

        metrics = RealtimeModelMetrics(
            label=self._realtime_model.label,
            request_id=payload.completionId,
            timestamp=time.monotonic(),
            duration=duration,
            ttft=ttft,
            cancelled=False,
            tokens=TokenUsage(
                input_tokens=input_tokens.speechTokens + input_tokens.textTokens,
                output_tokens=output_tokens.speechTokens + output_tokens.textTokens,
                input_text_tokens=input_tokens.textTokens,
                input_audio_tokens=input_tokens.speechTokens,
                output_text_tokens=output_tokens.textTokens,
                output_audio_tokens=output_tokens.speechTokens,
            ),
            tokens_per_second=tokens_per_second,
            metadata=Metadata(
                model_name=self._realtime_model.model, model_provider=self._realtime_model.provider
            ),
        )
        self.emit("metrics_collected", metrics)

    async def _process_responses(self) -> None:
        """Background task that drains Bedrock's output stream and feeds the event handlers."""
        try:
            await self._is_sess_active.wait()
            assert self._stream_response is not None, "stream_response is None"

            _, output_stream = await self._stream_response.await_output()
            # Stream is now fully bidirectional (HTTP 200 received).
            # Unblock interactive text sends so they land after audio is flowing.
            self._stream_ready.set()
            while self._is_sess_active.is_set():
                # and not self.stream_response.output_stream.closed:
                try:
                    result = await output_stream.receive()
                    if result is None:
                        # Stream closed, exit gracefully
                        logger.debug("[SESSION] Stream returned None, exiting")
                        break
                    if not isinstance(result, InvokeModelWithBidirectionalStreamOutputChunk):
                        raise APIError(
                            "unexpected Nova Sonic output event",
                            kind="protocol",
                            detail=type(result).__name__,
                        )
                    if result.value.bytes_:
                        response_data = result.value.bytes_.decode("utf-8")
                        try:
                            json_data = json.loads(response_data)
                            # logger.debug(f"Received event: {json_data}")
                            await self._handle_event(json_data)
                        except json.JSONDecodeError:
                            logger.warning(
                                "JSON decode error",
                                extra={"lk.pii.response_data": f"{response_data}"},
                            )
                    else:
                        logger.warning("No response received")
                except concurrent.futures.InvalidStateError:
                    # Future was cancelled during shutdown - expected when AWS CRT
                    # tries to deliver data to cancelled futures
                    logger.debug(
                        "[SESSION] Future cancelled during receive (expected during shutdown)"
                    )
                    break
                except AttributeError as ae:
                    # Result is None during shutdown
                    if "'NoneType' object has no attribute" in str(ae):
                        logger.debug(
                            "[SESSION] Stream closed during receive (expected during shutdown)"
                        )
                        break
                    raise
                except asyncio.CancelledError:
                    logger.info("Response processing task cancelled")
                    self._close_current_generation()
                    raise
                except ValidationException as ve:
                    # Some Bedrock ValidationException messages represent transient stream
                    # failures. Recover by restarting the Sonic session instead of tearing
                    # down the LiveKit session.
                    if _is_recoverable_validation_error(ve):
                        logger.warning(f"Validation error: {ve}\nAttempting to recover...")
                        await self._restart_session(ve)
                        break
                    elif "Tool Response parsing error" in ve.message:
                        # Tool parsing errors are recoverable - log and continue
                        logger.warning(f"Tool response parsing error (recoverable): {ve}")

                        # Close current generation to unblock the model
                        if self._current_generation:
                            logger.debug("Closing generation due to tool parsing error")
                            self._close_current_generation()

                        # Clear pending tools since they failed
                        if self._pending_tools:
                            logger.debug(f"Clearing {len(self._pending_tools)} pending tools")
                            self._pending_tools.clear()
                            self._no_pending_tools.set()

                        self.emit(
                            "error",
                            llm.RealtimeModelError(
                                timestamp=time.monotonic(),
                                label=self._realtime_model.label,
                                error=APIStatusError(
                                    message=ve.message,
                                    status_code=400,
                                    request_id="",
                                    body=ve,
                                ),
                                recoverable=True,
                            ),
                        )
                        # Don't raise - continue processing
                    else:
                        logger.error(f"Validation error: {ve}")
                        self.emit(
                            "error",
                            llm.RealtimeModelError(
                                timestamp=time.monotonic(),
                                label=self._realtime_model.label,
                                error=APIStatusError(
                                    message=ve.message,
                                    status_code=400,
                                    request_id="",
                                    body=ve,
                                ),
                                recoverable=False,
                            ),
                        )
                        raise
                except (
                    ThrottlingException,
                    ModelNotReadyException,
                    ModelErrorException,
                    ModelStreamErrorException,
                    InvalidEventBytes,
                ) as re:
                    logger.warning(
                        f"Retryable error: {re}\nAttempting to recover...", exc_info=True
                    )
                    await self._restart_session(re)
                    break
                except ModelTimeoutException as mte:
                    logger.warning(
                        f"Model timeout error: {mte}\nAttempting to recover...", exc_info=True
                    )
                    await self._restart_session(mte)
                    break
                except ValueError as val_err:
                    if "I/O operation on closed file." == val_err.args[0]:
                        logger.info("initiating graceful shutdown of session")
                        break
                    raise
                except OSError:
                    logger.info("stream already closed, exiting")
                    break
                except Exception as e:
                    err_msg = getattr(e, "message", str(e))
                    request_id = None
                    try:
                        request_id = err_msg.split(" ")[0].split("=")[1]
                    except Exception:
                        pass

                    self.emit(
                        "error",
                        llm.RealtimeModelError(
                            timestamp=time.monotonic(),
                            label=self._realtime_model.label,
                            error=APIStatusError(
                                f"Nova Sonic response processing error ({type(e).__name__})",
                                status_code=500,
                                request_id=request_id,
                                body=e,
                                detail=err_msg,
                            ),
                            recoverable=False,
                        ),
                    )
                    raise

        finally:
            logger.info("main output response stream processing task exiting")
            # A retry can replace this task before the old task reaches its
            # cleanup block. Keep the session active for the replacement task.
            if asyncio.current_task() is self._response_task:
                self._is_sess_active.clear()

    async def _restart_session(self, ex: Exception) -> None:
        # Get restart attempts from current generation, or 0 if no generation
        restart_attempts = (
            self._current_generation.restart_attempts if self._current_generation else 0
        )

        if restart_attempts >= DEFAULT_MAX_SESSION_RESTART_ATTEMPTS:
            logger.error("Max restart attempts reached for this turn, exiting")
            err_msg = getattr(ex, "message", str(ex))
            request_id = None
            try:
                request_id = err_msg.split(" ")[0].split("=")[1]
            except Exception:
                pass
            self.emit(
                "error",
                llm.RealtimeModelError(
                    timestamp=time.monotonic(),
                    label=self._realtime_model.label,
                    error=APIStatusError(
                        "Max restart attempts exceeded",
                        status_code=500,
                        request_id=request_id,
                        body=ex,
                        detail=err_msg,
                    ),
                    recoverable=False,
                ),
            )
            self._is_sess_active.clear()
            return

        # Increment restart counter for current generation
        if self._current_generation:
            self._current_generation.restart_attempts += 1
            restart_attempts = self._current_generation.restart_attempts
        else:
            restart_attempts = 1

        self._is_sess_active.clear()
        delay = 2 ** (restart_attempts - 1) - 1
        await asyncio.sleep(min(delay, DEFAULT_MAX_SESSION_RESTART_DELAY))
        await self.initialize_streams(is_restart=True)
        logger.info(
            f"Turn restarted successfully ({restart_attempts}/{DEFAULT_MAX_SESSION_RESTART_ATTEMPTS})"
        )

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    async def update_instructions(self, instructions: str) -> None:
        """Injects the system prompt at the start of the session."""
        self._instructions = instructions
        if self._instructions_ready is None:
            self._instructions_ready = asyncio.get_running_loop().create_future()
        if not self._instructions_ready.done():
            self._instructions_ready.set_result(True)
        logger.debug("Instructions updated", extra={"lk.pii.instructions": instructions})

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """Inject chat history and handle incremental user messages."""
        if self._chat_ctx_ready is None:
            self._chat_ctx_ready = asyncio.get_running_loop().create_future()

        chat_ctx = chat_ctx.copy(
            exclude_handoff=True,
            exclude_instructions=True,
            exclude_empty_message=True,
            exclude_config_update=True,
        )

        # Initial context setup (once)
        if not self._chat_ctx_ready.done():
            self._chat_ctx = chat_ctx.copy()
            # Nova Sonic requires history to start with a user message.
            # During handoff, the context may begin with an orphaned assistant
            # greeting (from generate_reply with instructions). Strip it.
            if (
                self._chat_ctx.items
                and self._chat_ctx.items[0].type == "message"
                and self._chat_ctx.items[0].role == "assistant"
            ):
                removed = self._chat_ctx.items.pop(0)
                logger.debug("Stripped leading assistant message from context: %s", removed.id)
            # Mark all initial context messages as already sent so the loop below
            # doesn't re-send them as interactive=true text. These messages are already
            # sent as non-interactive history via create_prompt_start_block during
            # initialize_streams.
            for item in chat_ctx.items:
                if item.type == "message":
                    self._sent_message_ids.add(item.id)
            logger.debug(
                "Chat context updated", extra={"lk.pii.chat_ctx_items": self._chat_ctx.items}
            )
            self._chat_ctx_ready.set_result(True)

        # Process items in context
        for item in chat_ctx.items:
            # Handle tool results
            if item.type == "function_call_output":
                if item.call_id not in self._pending_tools:
                    continue

                logger.debug("function call output received", extra={"lk.pii.item": item})
                self._pending_tools.discard(item.call_id)
                if not self._pending_tools:
                    self._no_pending_tools.set()

                if not item.reply_required:
                    logger.warning(
                        "a tool result wants no reply, but Nova Sonic will answer it anyway. "
                        "Sending it regardless, since an unanswered tool use keeps the turn open.",
                        extra={"function": item.name, "call_id": item.call_id},
                    )

                # Format tool result as proper JSON
                if item.is_error:
                    tool_result = json.dumps({"error": str(item.output)})
                else:
                    tool_result = item.output

                self._tool_results_ch.send_nowait(
                    {
                        "tool_use_id": item.call_id,
                        "tool_result": tool_result,
                    }
                )
                continue

            # Handle new user messages (Nova 2.0 text input)
            # Only send if it's NOT an audio transcription (audio messages are tracked in _audio_message_ids)
            if (
                item.type == "message"
                and item.role == "user"
                and item.id not in self._sent_message_ids
            ):
                # Check if this is an audio message (already transcribed by Nova)
                if item.id not in self._audio_message_ids:
                    if item.raw_text_content and item.raw_text_content.strip():
                        logger.debug(
                            "Sending user message as interactive text",
                            extra={"lk.pii.text_content": item.raw_text_content},
                        )
                        # Send interactive text to Nova Sonic (triggers generation)
                        # This is the flow for generate_reply(user_input=...) from the framework
                        fut = asyncio.Future[llm.GenerationCreatedEvent]()
                        self._pending_generation_fut = fut

                        text = item.raw_text_content

                        async def _send_user_text(
                            text: str = text, fut: asyncio.Future[llm.GenerationCreatedEvent] = fut
                        ) -> None:
                            try:
                                await self._audio_input_started.wait()
                                await self._send_text_message(text, interactive=True)
                            except Exception as e:
                                if not fut.done():
                                    fut.set_exception(e)
                                if self._pending_generation_fut is fut:
                                    self._pending_generation_fut = None

                        self._tasks.spawn(_send_user_text(), name="RealtimeSession._send_user_text")

                    self._sent_message_ids.add(item.id)
                    self._chat_ctx.items.append(item)
                else:
                    logger.debug(
                        "Skipping user message (already in context from audio)",
                        extra={"lk.pii.text_content": item.raw_text_content},
                    )
                    self._sent_message_ids.add(item.id)

    async def _send_tool_events(self, tool_use_id: str, tool_result: str) -> None:
        """Send tool_result back to Bedrock, grouped under tool_use_id."""
        tool_content_name = str(uuid.uuid4())
        tool_events = self._event_builder.create_tool_content_block(
            content_name=tool_content_name,
            tool_use_id=tool_use_id,
            content=tool_result,
        )
        for event in tool_events:
            await self._send_raw_event(event)
            # logger.debug(f"Sent tool event: {event}")

    def _tool_choice_adapter(
        self, tool_choice: llm.ToolChoice | None
    ) -> dict[str, dict[str, str]] | None:
        """Translate the LiveKit ToolChoice enum into Sonic's JSON schema."""
        if tool_choice == "auto":
            return {"auto": {}}
        elif tool_choice == "required":
            return {"any": {}}
        elif isinstance(tool_choice, dict) and tool_choice["type"] == "function":
            return {"tool": {"name": tool_choice["function"]["name"]}}
        else:
            return None

    # note: return value from tool functions registered to Sonic must be Structured Output (a dict that is JSON serializable)  # noqa: E501
    async def update_tools(self, tools: list[llm.Tool]) -> None:
        """Replace the active tool set with tools.

        Nova Sonic requires tools to be declared at session start.
        If the session is already active, we schedule a session recycle
        so the new tool set is sent in the next prompt start block.
        The recycle is deferred to avoid conflicts with in-flight tool
        results that are still being delivered to the current session.
        """
        old_tools = set(self._tools.function_tools)
        self._tools = llm.ToolContext(tools)
        new_tools = set(self._tools.function_tools)

        # the first update, empty or not, is what the session start waits for
        if self._tools_ready is None:
            self._tools_ready = asyncio.get_running_loop().create_future()
        if not self._tools_ready.done():
            self._tools_ready.set_result(True)
            logger.debug("Tool list has been injected (initial)")
            return

        # If tools actually changed and session is active, schedule a deferred recycle.
        # We defer because update_tools is often called from within a tool execution
        # callback, and the tool result is still being delivered to the current session.
        if old_tools != new_tools and self._is_sess_active.is_set():
            logger.info(
                f"[SESSION] Tools changed (added={new_tools - old_tools}, "
                f"removed={old_tools - new_tools}), scheduling deferred session recycle"
            )
            self._tasks.spawn(
                self._deferred_tool_recycle(), name="RealtimeSession._deferred_tool_recycle"
            )
        else:
            logger.debug("Tool list updated locally")

    async def _deferred_tool_recycle(self) -> None:
        """Wait for the tool calls in flight to be answered, then recycle."""
        try:
            async with asyncio.timeout(TOOL_RESULTS_RECYCLE_TIMEOUT_SECONDS):
                await self._no_pending_tools.wait()
        except TimeoutError:
            logger.warning("[SESSION] Tool results still pending, recycling without them")

        if not self._is_sess_active.is_set():
            logger.debug("[SESSION] Session no longer active, skipping tool recycle")
            return

        logger.info("[SESSION] Recycling session for updated tools")
        # results that arrive later answer the old stream's tool uses: ignore them
        self._pending_tools.clear()
        self._no_pending_tools.set()
        # the recycle sends the results already queued before it closes the stream
        await self._graceful_session_recycle()

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        """Live update of inference options is not supported by Sonic yet."""
        logger.warning(
            "updating inference configuration options is not yet supported by Nova Sonic's Realtime API"  # noqa: E501
        )

    async def _process_audio_input(self) -> None:
        """Background task that feeds audio and tool results into the Bedrock stream."""
        # Wait for the bidirectional stream to be fully established (HTTP 200)
        # before sending audio_content_start_event. Without this, under load
        # Bedrock may not have finished processing chat history, causing:
        # ValidationException: "Chat history should be sent completely before streaming audio."
        if not self._stream_ready.is_set():
            # the response reader opens the stream: if it ends first, nothing will
            assert self._response_task is not None
            async with utils.aio.TaskSet() as tasks:
                await tasks.first_completed(
                    self._stream_ready.wait(), asyncio.shield(self._response_task)
                )
            if not self._stream_ready.is_set():
                return
        await self._send_raw_event(self._event_builder.create_audio_content_start_event())
        self._audio_input_started.set()
        logger.info("Starting audio input processing loop")

        await self._is_sess_active.wait()

        try:
            async with utils.aio.TaskSet() as tasks:
                audio = tasks.spawn(self._audio_input_chan.recv())
                tool = tasks.spawn(self._tool_results_ch.recv())
                while self._is_sess_active.is_set():
                    try:
                        # shielded: the read that did not finish keeps its place for the next round
                        await tasks.first_completed(asyncio.shield(audio), asyncio.shield(tool))
                    except utils.aio.ChanClosed:
                        logger.warning("audio input or tool results channel closed")
                        break
                    try:
                        if audio.done():
                            audio_bytes = audio.result()
                            audio = tasks.spawn(self._audio_input_chan.recv())
                            # Wait for any in-flight text block to complete before sending the
                            # frame, so audioInput never lands between TEXT contentStart and
                            # contentEnd. The lock serializes with _send_text_message.
                            async with self._text_block_lock:
                                blob = base64.b64encode(audio_bytes)
                                audio_event = self._event_builder.create_audio_input_event(
                                    audio_content=blob.decode("utf-8"),
                                )
                                await self._send_raw_event(audio_event)
                        if tool.done():
                            val = tool.result()
                            tool = tasks.spawn(self._tool_results_ch.recv())
                            tool_result = val["tool_result"]
                            try:
                                tool_result = json.loads(tool_result)
                            except json.JSONDecodeError:
                                tool_result = json.dumps({"tool_result": tool_result})

                            logger.debug(
                                "Sending tool result", extra={"lk.pii.tool_result": tool_result}
                            )
                            await self._send_tool_events(val["tool_use_id"], tool_result)
                    except Exception:
                        logger.exception("Error processing audio")
        except asyncio.CancelledError:
            logger.info("Audio processing loop cancelled")
            self._audio_input_chan.close()
            self._tool_results_ch.close()
            raise

    # for debugging purposes only
    def _log_significant_audio(self, audio_bytes: bytes) -> None:
        """Utility that prints a debug message when the audio chunk has non-trivial RMS energy."""
        squared_sum = sum(sample**2 for sample in audio_bytes)
        if (squared_sum / len(audio_bytes)) ** 0.5 > 200:
            if lk_bedrock_debug:
                log_message("Enqueuing significant audio chunk", AnsiColors.BLUE)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Enqueue an incoming mic rtc.AudioFrame for transcription."""
        if not self._audio_input_chan.closed:
            for nf in self._bstream.push(frame.data.tobytes()):
                audio_bytes = bytes(nf.data)
                self._log_significant_audio(audio_bytes)
                self._audio_input_chan.send_nowait(audio_bytes)
        else:
            logger.warning("audio input channel closed, skipping audio")

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """Generate a reply from the model.

        This method is called by the LiveKit framework's VoiceSession.generate_reply() and
        AgentActivity._realtime_reply_task(). The framework handles user_input by adding it
        to the chat context via update_chat_ctx() before calling this method.

        Flow for user_input:
            1. Framework receives generate_reply with user_input parameter
            2. Framework adds user message to chat context
            3. Framework calls update_chat_ctx() (which sends the message to Nova Sonic)
            4. Framework calls this method no parameters
            5. This method trigger Nova Sonic's response based on the last context message add

        Flow for instructions:
            1. Framework receives generate_reply with instructions parameter
            2. Framework calls this method instructions parameter
            3. This method sends instructions as a prompt to Nova Sonic and triggers a response.

        If both parameters are sent, the same flow will strip the user_input out of the initial call
        and send the instructions on to this method.

        For Nova Sonic 2.0 and any supporting model:
            - Sends instructions as interactive text if provided
            - Triggers model response generation

        For Nova Sonic 1.0:
            - Not supported (no text input capability)
            - Logs warning and returns empty future

        Args:
            instructions (NotGivenOr[str]): Additional instructions to guide the response.
                These are sent as system-level prompts to influence how the model responds.
                User input should be added via update_chat_ctx(), not passed here.

        Returns:
            asyncio.Future[llm.GenerationCreatedEvent]: Future that resolves when generation starts.
                Raises RealtimeError on timeout (default: 10s).

        Note:
            User messages flow through VoiceSession.generate_reply(user_input=...) →
            update_chat_ctx() which sends interactive text to Nova Sonic.
            This method handles the instructions parameter for system-level prompts.
        """
        if is_given(tools):
            logger.warning(
                "per-response tools is not supported by AWS Nova Sonic Realtime API, ignoring"
            )
        # Check if generate_reply is supported (requires mixed modalities)
        if self._realtime_model.modalities != "mixed":
            logger.warning(
                "generate_reply() is not supported by this model (requires mixed modalities). "
                "Skipping generate_reply call. Use modalities='mixed' or Nova Sonic 2.0 "
                "to enable this feature."
            )

            # Return a completed future with empty streams so the caller doesn't hang
            async def _empty_message_stream() -> AsyncIterator[llm.MessageGeneration]:
                return
                yield  # Make it an async generator

            async def _empty_function_stream() -> AsyncIterator[llm.FunctionCall]:
                return
                yield  # Make it an async generator

            fut = asyncio.Future[llm.GenerationCreatedEvent]()
            fut.set_result(
                llm.GenerationCreatedEvent(
                    message_stream=_empty_message_stream(),
                    function_stream=_empty_function_stream(),
                    user_initiated=True,
                )
            )
            return fut

        # Nova 2.0: Only send if instructions provided
        if is_given(instructions):
            logger.info(
                "generate_reply: sending instructions",
                extra={"lk.pii.instructions": instructions},
            )

            # Create future that will be resolved when generation starts
            fut = asyncio.Future[llm.GenerationCreatedEvent]()
            self._pending_generation_fut = fut

            # Send text message asynchronously
            async def _send_text() -> None:
                try:
                    await self._audio_input_started.wait()
                    await self._send_text_message(instructions, interactive=True)
                except Exception as e:
                    if not fut.done():
                        fut.set_exception(e)
                    if self._pending_generation_fut is fut:
                        self._pending_generation_fut = None

            send_task = self._tasks.spawn(_send_text(), name="RealtimeSession._send_text")

            # Set timeout from model configuration
            def _on_timeout() -> None:
                if not fut.done():
                    fut.set_exception(
                        llm.RealtimeError("generate_reply timed out waiting for generation")
                    )
                    if self._pending_generation_fut is fut:
                        self._pending_generation_fut = None

            timeout_handle = asyncio.get_running_loop().call_later(
                self._generate_reply_timeout, _on_timeout
            )

            def _on_fut_done(f: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
                timeout_handle.cancel()
                if self._pending_generation_fut is fut:
                    self._pending_generation_fut = None
                if f.cancelled() or f.exception() is not None:
                    # cancelled, timed out or closed before the text was sent: drop the send
                    send_task.cancel()

            fut.add_done_callback(_on_fut_done)

            return fut

        # No instructions: Return pending generation if exists, otherwise create empty future that never resolves
        # (Framework will timeout naturally if no generation happens)
        if self._pending_generation_fut is not None:
            logger.debug("generate_reply: no instructions, returning existing pending generation")
            return self._pending_generation_fut

        logger.debug(
            "generate_reply: no instructions and no pending generation, returning empty future"
        )
        return asyncio.Future[llm.GenerationCreatedEvent]()

    async def _send_text_message(self, text: str, interactive: bool = True) -> None:
        """Internal method to send text message to Nova Sonic 2.0.

        Args:
            text (str): The text message to send to the model.
            interactive (bool): If True, triggers generation. If False, adds to context only.
        """
        # Generate unique content_name for this message (required for multi-turn)
        content_name = str(uuid.uuid4())

        # Choose appropriate event builder based on interactive flag
        if interactive:
            event = self._event_builder.create_text_content_start_event_interactive(
                content_name=content_name, role="USER"
            )
        else:
            event = self._event_builder.create_text_content_start_event(
                content_name=content_name, role="USER"
            )

        # Gate audio frames while the text block is in-flight.
        # This ensures TEXT contentEnd arrives before the next audioInput frame,
        # preventing the "Chat history should be sent completely before streaming audio"
        # ValidationException from Nova Sonic.
        async with self._text_block_lock:
            # Send event sequence: contentStart → textInput → contentEnd
            # Use try/finally to guarantee contentEnd is sent even if the task is
            # cancelled mid-send. Without this, a cancelled reply leaves the text
            # block open server-side, causing all subsequent audio to be rejected.
            content_started = False
            try:
                await self._send_raw_event(event)
                content_started = True
                await self._send_raw_event(
                    self._event_builder.create_text_content_event(content_name, text)
                )
                await self._send_raw_event(
                    self._event_builder.create_content_end_event(content_name)
                )
                content_started = False
            except (asyncio.CancelledError, Exception):
                if content_started:
                    # Best-effort: close the text block so the server doesn't
                    # reject all future audio frames for this session.
                    try:
                        await self._send_raw_event(
                            self._event_builder.create_content_end_event(content_name)
                        )
                    except Exception:
                        pass
                raise

        logger.info(
            "Sent text message",
            extra={"interactive": interactive, "lk.pii.text": text[:50]},
        )

    def commit_audio(self) -> None:
        logger.warning("commit_audio is not supported by Nova Sonic's Realtime API")

    def clear_audio(self) -> None:
        logger.warning("clear_audio is not supported by Nova Sonic's Realtime API")

    def push_video(self, frame: rtc.VideoFrame) -> None:
        logger.warning("video is not supported by Nova Sonic's Realtime API")

    def interrupt(self) -> None:
        """Nova Sonic handles interruption automatically via barge-in detection.

        Unlike OpenAI's client-initiated interrupt, Nova Sonic automatically detects
        when the user starts speaking while the model is generating audio. When this
        happens, the model:
        1. Immediately stops generating speech
        2. Switches to listening mode
        3. Sends a text event with content: { "interrupted" : true }

        The plugin already handles this event (see _handle_text_output_content_event).
        No client action is needed - interruption works automatically.

        See AWS docs: https://docs.aws.amazon.com/nova/latest/userguide/output-events.html
        """
        logger.info(
            "Nova Sonic handles interruption automatically via barge-in detection. "
            "The model detects when users start speaking and stops generation automatically."
        )

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        logger.warning("truncate is not supported by Nova Sonic's Realtime API")

    async def aclose(self) -> None:
        """Gracefully shut down the realtime session and release network resources."""
        logger.info("attempting to shutdown agent session")

        # Cancel any pending generation futures
        if self._pending_generation_fut and not self._pending_generation_fut.done():
            self._pending_generation_fut.set_exception(
                llm.RealtimeError("Session closed while waiting for generation")
            )
            self._pending_generation_fut = None

        # a recycle in flight would reopen the stream
        if self._session_recycle_task is not None:
            await utils.aio.cancel_and_wait(self._session_recycle_task)

        if self._is_sess_active.is_set():
            for event in self._event_builder.create_prompt_end_block():
                await self._send_raw_event(event)
            # allow event loops to fall out naturally
            # otherwise, the smithy layer will raise an InvalidStateError during cancellation
            self._is_sess_active.clear()
            if self._stream_response and (output_stream := self._stream_response.output_stream):
                await output_stream.close()
            await self._drain_response_task()

        # the audio input task must stop before the input stream closes
        await self._tasks.aclose()
        if self._stream_response:
            await self._stream_response.input_stream.close()

        logger.debug(
            "chat context at session end", extra={"lk.pii.chat_ctx_items": self._chat_ctx.items}
        )
        logger.info("Session end")
