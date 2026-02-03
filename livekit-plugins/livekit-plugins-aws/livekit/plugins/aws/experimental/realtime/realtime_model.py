# mypy: disable-error-code=unused-ignore

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import os
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, cast

import boto3
from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
    ModelErrorException,
    ModelNotReadyException,
    ModelStreamErrorException,
    ModelTimeoutException,
    ThrottlingException,
    ValidationException,
)
from smithy_aws_core.identity import AWSCredentialsIdentity
from smithy_aws_event_stream.exceptions import InvalidEventBytes
from smithy_core.aio.interfaces.identity import IdentityResolver

from livekit import rtc
from livekit.agents import (
    APIStatusError,
    llm,
    utils,
)
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins.aws.experimental.realtime.turn_tracker import _TurnTracker

from ...log import logger
from .events import (
    SonicEventBuilder as seb,
    Tool,
    ToolConfiguration,
    ToolInputSchema,
    ToolSpec,
)
from .pretty_printer import AnsiColors, log_event_data, log_message
from .types import MODALITIES, REALTIME_MODELS, SONIC1_VOICES, SONIC2_VOICES, TURN_DETECTION

DEFAULT_INPUT_SAMPLE_RATE = 16000
DEFAULT_OUTPUT_SAMPLE_RATE = 24000
DEFAULT_SAMPLE_SIZE_BITS = 16
DEFAULT_CHANNELS = 1
DEFAULT_CHUNK_SIZE = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 1024
MAX_MESSAGE_SIZE = 1024
MAX_MESSAGES = 40
DEFAULT_MAX_SESSION_RESTART_ATTEMPTS = 3
DEFAULT_MAX_SESSION_RESTART_DELAY = 10
# Session recycling: restart before 8-min AWS limit or credential expiry
# Override with LK_SESSION_MAX_DURATION env var for testing (e.g., "60" for 1 minute)
MAX_SESSION_DURATION_SECONDS = int(os.getenv("LK_SESSION_MAX_DURATION", 6 * 60))
CREDENTIAL_EXPIRY_BUFFER_SECONDS = 3 * 60  # Restart 3 min before credential expiry
BARGE_IN_SIGNAL = '{ "interrupted" : true }\n'  # Nova Sonic's barge-in detection signal
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

# Shared credentials resolver instance to preserve cache across all sessions
_shared_credentials_resolver: Boto3CredentialsResolver | None = None


def _get_credentials_resolver() -> Boto3CredentialsResolver:
    """Get or create the shared credentials resolver instance.

    This ensures credential caching works across all RealtimeSession instances.
    """
    global _shared_credentials_resolver
    if _shared_credentials_resolver is None:
        _shared_credentials_resolver = Boto3CredentialsResolver()
    return _shared_credentials_resolver


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
        _created_timestamp (float): Wall-clock time when the generation record was created.
        _first_token_timestamp (float | None): Wall-clock time of first token emission.
        _completed_timestamp (float | None): Wall-clock time when the turn fully completed.
        _restart_attempts (int): Number of restart attempts for this specific completion.
    """  # noqa: E501

    completion_id: str
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    response_id: str
    message_gen: _MessageGeneration | None = None
    content_id_map: dict[str, str] = field(default_factory=dict)
    _created_timestamp: float = field(default_factory=time.time)
    _first_token_timestamp: float | None = None
    _completed_timestamp: float | None = None
    _restart_attempts: int = 0
    _done_fut: asyncio.Future[None] | None = None  # Resolved when generation completes
    _emitted: bool = False  # Track if generation_created event was emitted


class Boto3CredentialsResolver(IdentityResolver):  # type: ignore[misc]
    """IdentityResolver implementation that sources AWS credentials from boto3.

    The resolver delegates to the default boto3.Session() credential chain which
    checks environment variables, shared credentials files, EC2 instance profiles, etc.
    The credentials are then wrapped in an AWSCredentialsIdentity so they can be
    passed into Bedrock runtime clients.
    """

    def __init__(self) -> None:
        self.session = boto3.Session()  # type: ignore[attr-defined]
        self._cached_identity: AWSCredentialsIdentity | None = None
        self._cached_expiry: float | None = None

    async def get_identity(self, **kwargs: Any) -> AWSCredentialsIdentity:
        """Asynchronously resolve AWS credentials.

        This method is invoked by the Bedrock runtime client whenever a new request needs to be
        signed.  It converts the static or temporary credentials returned by boto3
        into an AWSCredentialsIdentity instance.

        Returns:
            AWSCredentialsIdentity: Identity containing the
            AWS access key, secret key and optional session token.

        Raises:
            ValueError: If no credentials could be found by boto3.
        """
        # Return cached credentials if available
        # Session recycling will close the connection and get fresh credentials before these expire
        if self._cached_identity:
            return self._cached_identity

        try:
            logger.debug("[CREDS] Attempting to load AWS credentials")
            credentials = self.session.get_credentials()
            if not credentials:
                logger.error("[CREDS] Unable to load AWS credentials")
                raise ValueError("Unable to load AWS credentials")

            creds = credentials.get_frozen_credentials()

            # Ensure credentials are valid
            if not creds.access_key or not creds.secret_key:
                logger.error("AWS credentials are incomplete")
                raise ValueError("AWS credentials are incomplete")

            logger.debug(
                f"[CREDS] AWS credentials loaded successfully. AWS_ACCESS_KEY_ID: {creds.access_key[:4]}***"
            )

            # Get expiration time if available (for temporary credentials)
            expiry_time = getattr(credentials, "_expiry_time", None)

            identity = AWSCredentialsIdentity(
                access_key_id=creds.access_key,
                secret_access_key=creds.secret_key,
                session_token=creds.token if creds.token else None,
                expiration=expiry_time,
            )

            # Cache the identity and expiry
            self._cached_identity = identity
            if expiry_time:
                # Session will restart 3 minutes before expiration
                self._cached_expiry = expiry_time.timestamp() - 180
                logger.debug(
                    f"[CREDS] Cached credentials with expiry. "
                    f"expiry_time={expiry_time}, restart_before={self._cached_expiry}"
                )
            else:
                # Static credentials don't have an inherent expiration attribute, cache indefinitely
                self._cached_expiry = None
                logger.debug("[CREDS] Cached static credentials (no expiry)")

            return identity
        except Exception as e:
            logger.error(f"[CREDS] Failed to load AWS credentials: {str(e)}")
            raise ValueError(f"Failed to load AWS credentials: {str(e)}")  # noqa: B904

    def get_credential_expiry_time(self) -> float | None:
        """Get the credential expiry timestamp synchronously.

        This loads credentials if not cached and returns the expiry time.
        Used for calculating session duration before the async stream starts.

        Returns:
            float | None: Unix timestamp when credentials expire, or None for static credentials.
        """
        try:
            session = boto3.Session()  # type: ignore[attr-defined]
            credentials = session.get_credentials()
            if not credentials:
                return None

            expiry_time = getattr(credentials, "_expiry_time", None)
            if expiry_time:
                return float(expiry_time.timestamp())
            return None
        except Exception as e:
            logger.warning(f"[CREDS] Failed to get credential expiry: {e}")
            return None


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
            )
        )
        self._model = model
        self._generate_reply_timeout = generate_reply_timeout
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

    def session(self) -> RealtimeSession:
        """Return a new RealtimeSession bound to this model instance."""
        sess = RealtimeSession(self)
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

    def __init__(self, realtime_model: RealtimeModel) -> None:
        """Create and wire-up a new realtime session.

        Args:
            realtime_model (RealtimeModel): Parent model instance that stores static
                inference options and the Smithy Bedrock client configuration.
        """
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._event_builder = seb(
            prompt_name=str(uuid.uuid4()),
            audio_content_name=str(uuid.uuid4()),
        )
        self._input_resampler: rtc.AudioResampler | None = None
        self._bstream = utils.audio.AudioByteStream(
            DEFAULT_INPUT_SAMPLE_RATE, DEFAULT_CHANNELS, samples_per_channel=DEFAULT_CHUNK_SIZE
        )

        self._response_task = None
        self._audio_input_task = None
        self._stream_response = None
        self._bedrock_client = None
        self._pending_tools: set[str] = set()
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
        self._session_start_time: float | None = None
        self._session_recycle_task: asyncio.Task[None] | None = None
        self._last_audio_output_time: float = 0.0  # Track when assistant last produced audio
        self._audio_end_turn_received: bool = False  # Track when assistant finishes speaking
        self._pending_generation_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        self._sent_message_ids: set[str] = set()
        self._audio_message_ids: set[str] = set()

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
        self._turn_tracker = _TurnTracker(
            cast(Callable[[str, Any], None], self.emit),
            cast(Callable[[], None], self.emit_generation_event),
        )

        # Create main task to manage session lifecycle
        self._main_atask = asyncio.create_task(
            self.initialize_streams(), name="RealtimeSession.initialize_streams"
        )

    @utils.log_exceptions(logger=logger)
    def _initialize_client(self) -> None:
        """Instantiate the Bedrock runtime client"""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self._realtime_model._opts.region}.amazonaws.com",
            region=self._realtime_model._opts.region,
            aws_credentials_identity_resolver=_get_credentials_resolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="bedrock")},
            user_agent_extra="x-client-framework:livekit-plugins-aws[realtime]",
        )
        self._bedrock_client = BedrockRuntimeClient(config=config)

    def _calculate_session_duration(self) -> float:
        """Calculate session duration based on credential expiry and AWS 8-min limit."""
        resolver = _get_credentials_resolver()
        credential_expiry = resolver.get_credential_expiry_time()

        if credential_expiry is None:
            # Static credentials - just use the max session duration
            logger.info(
                f"[SESSION] Static credentials, using max duration: {MAX_SESSION_DURATION_SECONDS}s"
            )
            return MAX_SESSION_DURATION_SECONDS

        # Calculate time until we should restart (before credential expiry)
        now = time.time()
        time_until_cred_expiry = credential_expiry - now - CREDENTIAL_EXPIRY_BUFFER_SECONDS

        # Use the minimum of session limit and credential expiry
        duration = min(MAX_SESSION_DURATION_SECONDS, time_until_cred_expiry)

        if duration < 30:
            logger.warning(
                f"[SESSION] Very short session duration: {duration:.0f}s. "
                f"Credentials may expire soon."
            )
            duration = max(duration, 10)  # At least 10 seconds

        logger.info(
            f"[SESSION] Session will recycle in {duration:.0f}s "
            f"(max={MAX_SESSION_DURATION_SECONDS}s, time_until_cred_expiry={time_until_cred_expiry:.0f}s)"
        )

        return duration

    def _start_session_recycle_timer(self) -> None:
        """Start the session recycling timer."""
        if self._session_recycle_task and not self._session_recycle_task.done():
            self._session_recycle_task.cancel()

        duration = self._calculate_session_duration()

        self._session_recycle_task = asyncio.create_task(
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
            if not self._audio_end_turn_received:
                logger.info(
                    "[SESSION] Waiting for assistant to finish speaking (AUDIO END_TURN)..."
                )
                while not self._audio_end_turn_received:
                    await asyncio.sleep(0.1)
                logger.debug("[SESSION] Assistant finished speaking")

            # Step 2: Wait for audio to fully stop (no new audio for 1 second)
            logger.debug("[SESSION] Waiting for audio to fully stop...")
            last_audio_time = self._last_audio_output_time
            while True:
                await asyncio.sleep(0.1)
                if self._last_audio_output_time == last_audio_time:
                    await asyncio.sleep(0.9)
                    if self._last_audio_output_time == last_audio_time:
                        logger.debug("[SESSION] No new audio for 1s, proceeding with recycle")
                        break
                else:
                    logger.debug("[SESSION] New audio detected, continuing to wait...")
                    last_audio_time = self._last_audio_output_time

            # Step 3: Send close events to trigger completionEnd from Nova Sonic
            # This must happen BEFORE cancelling tasks so response task can receive completionEnd
            logger.info("[SESSION] Sending close events to Nova Sonic...")
            if self._stream_response:
                for event in self._event_builder.create_prompt_end_block():
                    await self._send_raw_event(event)

            # Step 4: Wait for completionEnd and let _done_fut resolve
            if self._current_generation and self._current_generation._done_fut:
                try:
                    await asyncio.wait_for(self._current_generation._done_fut, timeout=2.0)
                    logger.debug("[SESSION] Generation completed (completionEnd received)")
                except asyncio.TimeoutError:
                    logger.warning("[SESSION] Timeout waiting for completionEnd, proceeding anyway")
                    self._close_current_generation()

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
            except utils.aio.channel.ChanEmpty:
                logger.debug("[SESSION] No more pending tool results")
                break
            except Exception as e:
                logger.warning(f"[SESSION] Error draining tool result: {e}")
                break

        # Step 2: Signal tasks to stop
        self._is_sess_active.clear()

        # Step 3: Wait for response task to exit naturally, then cancel if needed
        if self._response_task and not self._response_task.done():
            try:
                # TODO: Even waiting for 30 seconds this never just happens.
                # See if we can figure out how to make this more graceful
                await asyncio.wait_for(self._response_task, timeout=1.0)
            except asyncio.TimeoutError:
                logger.debug("[SESSION] Response task timeout, cancelling...")
                self._response_task.cancel()
                try:
                    await self._response_task
                except asyncio.CancelledError:
                    pass

        # Step 4: Cancel audio input task (blocked on channel, won't exit naturally)
        if self._audio_input_task and not self._audio_input_task.done():
            self._audio_input_task.cancel()
            try:
                await self._audio_input_task
            except asyncio.CancelledError:
                pass

        # Step 5: Close the stream (close events already sent in _session_recycle_timer)
        if self._stream_response:
            try:
                if not self._stream_response.input_stream.closed:
                    await self._stream_response.input_stream.close()
            except Exception as e:
                logger.debug(f"[SESSION] Error closing stream (expected): {e}")

        # Step 6: Reset state for new session
        self._stream_response = None
        self._bedrock_client = None
        self._event_builder = seb(
            prompt_name=str(uuid.uuid4()),
            audio_content_name=str(uuid.uuid4()),
        )
        self._tool_results_ch = utils.aio.Chan[dict[str, str]]()
        logger.debug("[SESSION] Created fresh tool results channel")
        self._audio_end_turn_received = False

        # Step 7: Start new session with preserved state
        await self.initialize_streams(is_restart=True)

        logger.info("[SESSION] Session recycled successfully")

    @utils.log_exceptions(logger=logger)
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
            logger.debug(f"[SEND] {event_json}")

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
                    label=self._realtime_model._label,
                    error=APIStatusError(
                        message=err_msg,
                        status_code=500,
                        request_id=request_id,
                        body=e,
                        retryable=False,
                    ),
                    recoverable=False,
                ),
            )
            raise

    def _serialize_tool_config(self) -> ToolConfiguration | None:
        """Convert self.tools into the JSON structure expected by Sonic.

        If any tools are registered, the method also harmonises temperature and
        top_p defaults to Sonic's recommended greedy values (1.0).

        Returns:
            ToolConfiguration | None: None when no tools are present, otherwise a complete config block.
        """  # noqa: E501
        tool_cfg = None
        if self.tools.function_tools:
            tools = []
            for name, f in self.tools.function_tools.items():
                if isinstance(f, llm.FunctionTool):
                    description = f.info.description
                    input_schema = llm.utils.build_legacy_openai_schema(f, internally_tagged=True)[
                        "parameters"
                    ]
                elif isinstance(f, llm.RawFunctionTool):
                    info = f.info
                    description = info.raw_schema.get("description")
                    raw_schema = info.raw_schema
                    # Safely access parameters with fallback
                    input_schema = raw_schema.get(
                        "parameters",
                        raw_schema.get("input_schema", {"type": "object", "properties": {}}),
                    )
                else:
                    continue

                tool = Tool(
                    toolSpec=ToolSpec(
                        name=name,
                        description=description or "No description provided",
                        inputSchema=ToolInputSchema(json_=json.dumps(input_schema)),  # type: ignore
                    )
                )
                tools.append(tool)
            tool_choice = self._tool_choice_adapter(self._realtime_model._opts.tool_choice)
            logger.debug(f"TOOL CHOICE: {tool_choice}")
            tool_cfg = ToolConfiguration(tools=tools, toolChoice=tool_choice)

            # recommended to set greedy inference configs for tool calls
            if not is_given(self._realtime_model.top_p):
                self._realtime_model._opts.top_p = 1.0
            if not is_given(self._realtime_model.temperature):
                self._realtime_model._opts.temperature = 1.0
        return tool_cfg

    @utils.log_exceptions(logger=logger)
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
                self._initialize_client()
            assert self._bedrock_client is not None, "bedrock_client is None"

            logger.info("Initializing Bedrock stream")
            self._stream_response = (
                await self._bedrock_client.invoke_model_with_bidirectional_stream(
                    InvokeModelWithBidirectionalStreamOperationInput(
                        model_id=self._realtime_model.model
                    )
                )
            )

            if not is_restart:
                # Lazy-initialize futures if needed
                if self._tools_ready is None:
                    self._tools_ready = asyncio.get_running_loop().create_future()
                if self._instructions_ready is None:
                    self._instructions_ready = asyncio.get_running_loop().create_future()
                if self._chat_ctx_ready is None:
                    self._chat_ctx_ready = asyncio.get_running_loop().create_future()

                pending_events: list[asyncio.Future] = []
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
                except asyncio.TimeoutError:
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

            logger.info(
                f"Initializing Bedrock session with realtime options: {self._realtime_model._opts}"
            )
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

            init_events = self._event_builder.create_prompt_start_block(
                voice_id=self._realtime_model._opts.voice,
                sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE,  # type: ignore
                system_content=self._instructions,
                chat_ctx=restart_ctx,
                tool_configuration=self._serialize_tool_config(),
                max_tokens=self._realtime_model._opts.max_tokens,
                top_p=self._realtime_model._opts.top_p,
                temperature=self._realtime_model._opts.temperature,
                endpointing_sensitivity=self._realtime_model._opts.turn_detection,
            )

            for event in init_events:
                await self._send_raw_event(event)
                logger.debug(f"Sent event: {event}")

            # Always create audio input task (even on restart)
            self._audio_input_task = asyncio.create_task(
                self._process_audio_input(), name="RealtimeSession._process_audio_input"
            )

            self._response_task = asyncio.create_task(
                self._process_responses(), name="RealtimeSession._process_responses"
            )
            self._is_sess_active.set()

            # Start session recycling timer
            self._session_start_time = time.time()
            self._start_session_recycle_timer()

            logger.debug("Stream initialized successfully")
        except Exception as e:
            logger.debug(f"Failed to initialize stream: {str(e)}")
            raise
        return self

    @utils.log_exceptions(logger=logger)
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
        if self._current_generation._emitted:
            logger.debug(
                f"[GEN] EMITTING generation_created (re-emit for tool call) for response_id={self._current_generation.response_id}"
            )
        else:
            logger.debug(
                f"[GEN] EMITTING generation_created for response_id={self._current_generation.response_id}"
            )

        self._current_generation._emitted = True
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

    @utils.log_exceptions(logger=logger)
    async def _handle_event(self, event_data: dict) -> None:
        """Dispatch a raw Bedrock event to the corresponding _handle_* method."""
        event_type = self._event_builder.get_event_type(event_data)
        event_handler = self._event_handlers.get(event_type)
        if event_handler:
            await event_handler(event_data)
            self._turn_tracker.feed(event_data)
        else:
            logger.warning(f"No event handler found for event type: {event_type}")

    async def _handle_completion_start_event(self, event_data: dict) -> None:
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
                _done_fut=asyncio.get_running_loop().create_future(),
            )
            generation_created = True
        else:
            logger.debug(
                f"[GEN] Generation already exists: response_id={self._current_generation.response_id}, emitted={self._current_generation._emitted}"
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
    async def _handle_text_output_content_start_event(self, event_data: dict) -> None:
        """Handle text_output_content_start - track content type."""
        log_event_data(event_data)

        role = event_data["event"]["contentStart"]["role"]

        # CRITICAL: Create NEW generation for each ASSISTANT SPECULATIVE response
        # Nova Sonic sends ASSISTANT SPECULATIVE for each new assistant turn, including after tool calls.
        # Without this, audio frames get routed to the wrong generation and don't play.
        if role == "ASSISTANT":
            additional_fields = event_data["event"]["contentStart"].get("additionalModelFields", "")
            if "SPECULATIVE" in additional_fields:
                # This is a new assistant response - close previous and create new
                logger.debug("[GEN] ASSISTANT SPECULATIVE text received")
                if self._current_generation is not None:
                    logger.debug(
                        f"[GEN] Closing previous generation (response_id={self._current_generation.response_id}) for new SPECULATIVE"
                    )
                    self._close_current_generation()
                self._create_response_generation()
        else:
            # For USER and FINAL, just ensure generation exists
            self._create_response_generation()

        # CRITICAL: Check if generation exists before accessing
        # Barge-in can set _current_generation to None between the creation above and here.
        # Without this check, we crash on interruptions.
        if self._current_generation is None:
            logger.debug("No generation exists - ignoring content_start event")
            return

        content_id = event_data["event"]["contentStart"]["contentId"]

        # Track what type of content this is
        if role == "USER":
            self._current_generation.content_id_map[content_id] = "USER_ASR"
        elif role == "ASSISTANT":
            additional_fields = event_data["event"]["contentStart"].get("additionalModelFields", "")
            if "SPECULATIVE" in additional_fields:
                self._current_generation.content_id_map[content_id] = "ASSISTANT_TEXT"
            elif "FINAL" in additional_fields:
                self._current_generation.content_id_map[content_id] = "ASSISTANT_FINAL"

    async def _handle_text_output_content_event(self, event_data: dict) -> None:
        """Stream partial text tokens into the current generation."""
        log_event_data(event_data)

        if self._current_generation is None:
            logger.debug("No generation exists - ignoring text_output event")
            return

        content_id = event_data["event"]["textOutput"]["contentId"]
        text_content = f"{event_data['event']['textOutput']['content']}\n"

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
            logger.debug(f"INPUT TRANSCRIPTION UPDATED: {text_content}")
            self._update_chat_ctx(role="user", text_content=text_content)

        elif content_type == "ASSISTANT_TEXT":
            # Set first token timestamp if not already set
            if self._current_generation._first_token_timestamp is None:
                self._current_generation._first_token_timestamp = time.time()

            # Stream text to LiveKit
            if self._current_generation.message_gen:
                self._current_generation.message_gen.text_ch.send_nowait(text_content)
            self._update_chat_ctx(role="assistant", text_content=text_content)

    def _update_chat_ctx(self, role: llm.ChatRole, text_content: str) -> None:
        """
        Update the chat context with the latest ASR text while guarding against model limitations:
            a) 40 total messages limit
            b) 1kB message size limit
        """
        logger.debug(f"Updating chat context with role: {role} and text_content: {text_content}")
        if len(self._chat_ctx.items) == 0:
            msg = self._chat_ctx.add_message(role=role, content=text_content)
            if role == "user":
                self._audio_message_ids.add(msg.id)
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
    async def _handle_text_output_content_end_event(self, event_data: dict) -> None:
        """Handle text content end - log but don't close generation yet."""
        # Nova Sonic sends multiple content blocks within one completion
        # Don't close generation here - wait for completionEnd or audio_output_content_end
        log_event_data(event_data)

    async def _handle_tool_output_content_start_event(self, event_data: dict) -> None:
        """Track tool content start."""
        log_event_data(event_data)

        # Ensure generation exists
        self._create_response_generation()

        if self._current_generation is None:
            return

        content_id = event_data["event"]["contentStart"]["contentId"]
        self._current_generation.content_id_map[content_id] = "TOOL"

    async def _handle_tool_output_content_event(self, event_data: dict) -> None:
        """Execute the referenced tool locally and queue results."""
        log_event_data(event_data)

        if self._current_generation is None:
            logger.warning("tool_output_content received without active generation")
            return

        tool_use_id = event_data["event"]["toolUse"]["toolUseId"]
        tool_name = event_data["event"]["toolUse"]["toolName"]
        args = event_data["event"]["toolUse"]["content"]

        # Emit function call to LiveKit framework
        self._current_generation.function_ch.send_nowait(
            llm.FunctionCall(call_id=tool_use_id, name=tool_name, arguments=args)
        )
        self._pending_tools.add(tool_use_id)
        logger.debug(f"Tool call emitted: {tool_name} (id={tool_use_id})")

        # CRITICAL: Close generation after tool call emission
        # The LiveKit framework expects the generation to close so it can call update_chat_ctx()
        # with the tool results. A new generation will be created when Nova Sonic sends the next
        # ASSISTANT SPECULATIVE text event with the tool response.
        logger.debug("Closing generation to allow tool result delivery")
        self._close_current_generation()

    async def _handle_tool_output_content_end_event(self, event_data: dict) -> None:
        log_event_data(event_data)

    async def _handle_audio_output_content_start_event(self, event_data: dict) -> None:
        """Track audio content start."""
        if self._current_generation is not None:
            log_event_data(event_data)
            content_id = event_data["event"]["contentStart"]["contentId"]
            self._current_generation.content_id_map[content_id] = "ASSISTANT_AUDIO"

    async def _handle_audio_output_content_event(self, event_data: dict) -> None:
        """Decode base64 audio from Bedrock and forward it to the audio stream."""
        if self._current_generation is None or self._current_generation.message_gen is None:
            return

        content_id = event_data["event"]["audioOutput"]["contentId"]
        content_type = self._current_generation.content_id_map.get(content_id)

        if content_type == "ASSISTANT_AUDIO":
            audio_content = event_data["event"]["audioOutput"]["content"]
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
            self._last_audio_output_time = time.time()

    async def _handle_audio_output_content_end_event(self, event_data: dict) -> None:
        """Handle audio content end - track END_TURN for session recycling."""
        log_event_data(event_data)

        # Check if this is END_TURN (assistant finished speaking)
        stop_reason = event_data.get("event", {}).get("contentEnd", {}).get("stopReason")
        if stop_reason == "END_TURN":
            self._audio_end_turn_received = True
            logger.debug("[SESSION] AUDIO END_TURN received - assistant finished speaking")

        # Nova Sonic uses one completion for entire session
        # Don't close generation here - wait for new completionStart or session end

    def _close_current_generation(self) -> None:
        """Helper that closes all channels of the active generation."""
        if self._current_generation is None:
            return

        response_id = self._current_generation.response_id
        was_emitted = self._current_generation._emitted

        # Set completed timestamp
        if self._current_generation._completed_timestamp is None:
            self._current_generation._completed_timestamp = time.time()

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

        # Resolve _done_fut to signal generation is complete (for session recycling)
        if self._current_generation._done_fut and not self._current_generation._done_fut.done():
            self._current_generation._done_fut.set_result(None)

        logger.debug(
            f"[GEN] CLOSED generation response_id={response_id}, was_emitted={was_emitted}"
        )
        self._current_generation = None

    async def _handle_completion_end_event(self, event_data: dict) -> None:
        """Handle completionEnd - close the generation for this completion cycle."""
        log_event_data(event_data)

        # Close generation if still open
        if self._current_generation:
            logger.debug("completionEnd received, closing generation")
            self._close_current_generation()

    async def _handle_other_event(self, event_data: dict) -> None:
        log_event_data(event_data)

    async def _handle_usage_event(self, event_data: dict) -> None:
        # log_event_data(event_data)
        input_tokens = event_data["event"]["usageEvent"]["details"]["delta"]["input"]
        output_tokens = event_data["event"]["usageEvent"]["details"]["delta"]["output"]

        # Calculate metrics from timestamps
        duration = 0.0
        ttft = 0.0
        tokens_per_second = 0.0

        if self._current_generation is not None:
            created_ts = self._current_generation._created_timestamp
            first_token_ts = self._current_generation._first_token_timestamp
            completed_ts = self._current_generation._completed_timestamp

            # Calculate TTFT (time to first token)
            if first_token_ts is not None and isinstance(created_ts, (int, float)):
                ttft = first_token_ts - created_ts

            # Calculate duration (total time from creation to completion)
            if completed_ts is not None and isinstance(created_ts, (int, float)):
                duration = completed_ts - created_ts

            # Calculate tokens per second
            total_tokens = (
                input_tokens["speechTokens"]
                + input_tokens["textTokens"]
                + output_tokens["speechTokens"]
                + output_tokens["textTokens"]
            )
            if duration > 0:
                tokens_per_second = total_tokens / duration

        metrics = RealtimeModelMetrics(
            label=self._realtime_model.label,
            request_id=event_data["event"]["usageEvent"]["completionId"],
            timestamp=time.monotonic(),
            duration=duration,
            ttft=ttft,
            cancelled=False,
            input_tokens=input_tokens["speechTokens"] + input_tokens["textTokens"],
            output_tokens=output_tokens["speechTokens"] + output_tokens["textTokens"],
            total_tokens=input_tokens["speechTokens"]
            + input_tokens["textTokens"]
            + output_tokens["speechTokens"]
            + output_tokens["textTokens"],
            tokens_per_second=tokens_per_second,
            input_token_details=RealtimeModelMetrics.InputTokenDetails(
                text_tokens=input_tokens["textTokens"],
                audio_tokens=input_tokens["speechTokens"],
                image_tokens=0,
                cached_tokens=0,
                cached_tokens_details=None,
            ),
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                text_tokens=output_tokens["textTokens"],
                audio_tokens=output_tokens["speechTokens"],
                image_tokens=0,
            ),
            metadata=Metadata(
                model_name=self._realtime_model.model, model_provider=self._realtime_model.provider
            ),
        )
        self.emit("metrics_collected", metrics)

    @utils.log_exceptions(logger=logger)
    async def _process_responses(self) -> None:
        """Background task that drains Bedrock's output stream and feeds the event handlers."""
        try:
            await self._is_sess_active.wait()
            assert self._stream_response is not None, "stream_response is None"

            # note: may need another signal here to block input task until bedrock is ready
            # TODO: save this as a field so we're not re-awaiting it every time
            _, output_stream = await self._stream_response.await_output()
            while self._is_sess_active.is_set():
                # and not self.stream_response.output_stream.closed:
                try:
                    result = await output_stream.receive()
                    if result is None:
                        # Stream closed, exit gracefully
                        logger.debug("[SESSION] Stream returned None, exiting")
                        break
                    if result.value and result.value.bytes_:
                        try:
                            response_data = result.value.bytes_.decode("utf-8")
                            json_data = json.loads(response_data)
                            # logger.debug(f"Received event: {json_data}")
                            await self._handle_event(json_data)
                        except json.JSONDecodeError:
                            logger.warning(f"JSON decode error: {response_data}")
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
                    # there is a 3min no-activity (e.g. silence) timeout on the stream, after which the stream is closed  # noqa: E501
                    if (
                        "InternalErrorCode=531::RST_STREAM closed stream. HTTP/2 error code: NO_ERROR"  # noqa: E501
                        in ve.message
                    ):
                        logger.warning(f"Validation error: {ve}\nAttempting to recover...")
                        await self._restart_session(ve)
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

                        self.emit(
                            "error",
                            llm.RealtimeModelError(
                                timestamp=time.monotonic(),
                                label=self._realtime_model._label,
                                error=APIStatusError(
                                    message=ve.message,
                                    status_code=400,
                                    request_id="",
                                    body=ve,
                                    retryable=False,
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
                                label=self._realtime_model._label,
                                error=APIStatusError(
                                    message=ve.message,
                                    status_code=400,
                                    request_id="",
                                    body=ve,
                                    retryable=False,
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
                    logger.error(f"Response processing error: {err_msg} (type: {type(e)})")
                    request_id = None
                    try:
                        request_id = err_msg.split(" ")[0].split("=")[1]
                    except Exception:
                        pass

                    self.emit(
                        "error",
                        llm.RealtimeModelError(
                            timestamp=time.monotonic(),
                            label=self._realtime_model._label,
                            error=APIStatusError(
                                message=err_msg,
                                status_code=500,
                                request_id=request_id,
                                body=e,
                                retryable=False,
                            ),
                            recoverable=False,
                        ),
                    )
                    raise

        finally:
            logger.info("main output response stream processing task exiting")
            self._is_sess_active.clear()

    async def _restart_session(self, ex: Exception) -> None:
        # Get restart attempts from current generation, or 0 if no generation
        restart_attempts = (
            self._current_generation._restart_attempts if self._current_generation else 0
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
                    label=self._realtime_model._label,
                    error=APIStatusError(
                        message=f"Max restart attempts exceeded: {err_msg}",
                        status_code=500,
                        request_id=request_id,
                        body=ex,
                        retryable=False,
                    ),
                    recoverable=False,
                ),
            )
            self._is_sess_active.clear()
            return

        # Increment restart counter for current generation
        if self._current_generation:
            self._current_generation._restart_attempts += 1
            restart_attempts = self._current_generation._restart_attempts
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
        logger.debug(f"Instructions updated: {instructions}")

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """Inject chat history and handle incremental user messages."""
        if self._chat_ctx_ready is None:
            self._chat_ctx_ready = asyncio.get_running_loop().create_future()

        chat_ctx = chat_ctx.copy(
            exclude_handoff=True, exclude_instructions=True, exclude_empty_message=True
        )

        # Initial context setup (once)
        if not self._chat_ctx_ready.done():
            self._chat_ctx = chat_ctx.copy()
            logger.debug(f"Chat context updated: {self._chat_ctx.items}")
            self._chat_ctx_ready.set_result(True)

        # Process items in context
        for item in chat_ctx.items:
            # Handle tool results
            if item.type == "function_call_output":
                if item.call_id not in self._pending_tools:
                    continue

                logger.debug(f"function call output: {item}")
                self._pending_tools.discard(item.call_id)

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
                    if item.text_content:
                        logger.debug(
                            f"Sending user message as interactive text: {item.text_content}"
                        )
                        # Send interactive text to Nova Sonic (triggers generation)
                        # This is the flow for generate_reply(user_input=...) from the framework
                        fut = asyncio.Future[llm.GenerationCreatedEvent]()
                        self._pending_generation_fut = fut

                        text = item.text_content

                        async def _send_user_text(
                            text: str = text, fut: asyncio.Future = fut
                        ) -> None:
                            try:
                                # Wait for session to be fully initialized before sending
                                await self._is_sess_active.wait()
                                await self._send_text_message(text, interactive=True)
                            except Exception as e:
                                if not fut.done():
                                    fut.set_exception(e)
                                if self._pending_generation_fut is fut:
                                    self._pending_generation_fut = None

                        asyncio.create_task(_send_user_text())

                    self._sent_message_ids.add(item.id)
                    self._chat_ctx.items.append(item)
                else:
                    logger.debug(
                        f"Skipping user message (already in context from audio): {item.text_content}"
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
        """Replace the active tool set with tools and notify Sonic if necessary."""
        logger.debug(f"Updating tools: {tools}")
        self._tools = llm.ToolContext(tools)
        if self._tools.function_tools:
            if self._tools_ready is None:
                self._tools_ready = asyncio.get_running_loop().create_future()
            if not self._tools_ready.done():
                self._tools_ready.set_result(True)
            logger.debug("Tool list has been injected")

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        """Live update of inference options is not supported by Sonic yet."""
        logger.warning(
            "updating inference configuration options is not yet supported by Nova Sonic's Realtime API"  # noqa: E501
        )

    @utils.log_exceptions(logger=logger)
    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        """Ensure mic audio matches Sonic's required sample rate & channels."""
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != DEFAULT_INPUT_SAMPLE_RATE or frame.num_channels != DEFAULT_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=DEFAULT_INPUT_SAMPLE_RATE,
                num_channels=DEFAULT_CHANNELS,
            )

        if self._input_resampler:
            # flush the resampler when the input source is changed
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    @utils.log_exceptions(logger=logger)
    async def _process_audio_input(self) -> None:
        """Background task that feeds audio and tool results into the Bedrock stream."""
        await self._send_raw_event(self._event_builder.create_audio_content_start_event())
        logger.info("Starting audio input processing loop")

        # Create tasks for both channels so we can wait on either
        audio_task = asyncio.create_task(self._audio_input_chan.recv())
        tool_task = asyncio.create_task(self._tool_results_ch.recv())
        pending = {audio_task, tool_task}

        while self._is_sess_active.is_set():
            try:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    if task == audio_task:
                        try:
                            audio_bytes = cast(bytes, task.result())
                            blob = base64.b64encode(audio_bytes)
                            audio_event = self._event_builder.create_audio_input_event(
                                audio_content=blob.decode("utf-8"),
                            )
                            await self._send_raw_event(audio_event)
                            # Create new task for next audio
                            audio_task = asyncio.create_task(self._audio_input_chan.recv())
                            pending.add(audio_task)
                        except utils.aio.channel.ChanClosed:
                            logger.warning("audio input channel closed")
                            break

                    elif task == tool_task:
                        try:
                            val = cast(dict[str, str], task.result())
                            tool_result = val["tool_result"]
                            tool_use_id = val["tool_use_id"]
                            if not isinstance(tool_result, str):
                                tool_result = json.dumps(tool_result)
                            else:
                                try:
                                    tool_result = json.loads(tool_result)
                                except json.JSONDecodeError:
                                    try:
                                        tool_result = json.dumps({"tool_result": tool_result})
                                    except Exception:
                                        logger.exception("Failed to parse tool result")

                            logger.debug(f"Sending tool result: {tool_result}")
                            await self._send_tool_events(tool_use_id, tool_result)
                            # Create new task for next tool result
                            tool_task = asyncio.create_task(self._tool_results_ch.recv())
                            pending.add(tool_task)
                        except utils.aio.channel.ChanClosed:
                            logger.warning("tool results channel closed")
                            break

            except asyncio.CancelledError:
                logger.info("Audio processing loop cancelled")
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                self._audio_input_chan.close()
                self._tool_results_ch.close()
                raise
            except Exception:
                logger.exception("Error processing audio")

    # for debugging purposes only
    def _log_significant_audio(self, audio_bytes: bytes) -> None:
        """Utility that prints a debug message when the audio chunk has non-trivial RMS energy."""
        squared_sum = sum(sample**2 for sample in audio_bytes)
        if (squared_sum / len(audio_bytes)) ** 0.5 > 200:
            if lk_bedrock_debug:
                log_message("Enqueuing significant audio chunk", AnsiColors.BLUE)

    @utils.log_exceptions(logger=logger)
    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Enqueue an incoming mic rtc.AudioFrame for transcription."""
        if not self._audio_input_chan.closed:
            # logger.debug(f"Raw audio received: samples={len(frame.data)} rate={frame.sample_rate} channels={frame.num_channels}")  # noqa: E501
            for f in self._resample_audio(frame):
                # logger.debug(f"Resampled audio: samples={len(frame.data)} rate={frame.sample_rate} channels={frame.num_channels}")  # noqa: E501

                for nf in self._bstream.write(f.data.tobytes()):
                    self._log_significant_audio(nf.data)
                    self._audio_input_chan.send_nowait(nf.data)
        else:
            logger.warning("audio input channel closed, skipping audio")

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """Generate a reply from the model.

        This method is called by the LiveKit framework's AgentSession.generate_reply() and
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
            User messages flow through AgentSession.generate_reply(user_input=...) →
            update_chat_ctx() which sends interactive text to Nova Sonic.
            This method handles the instructions parameter for system-level prompts.
        """
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
            logger.info(f"generate_reply: sending instructions='{instructions}'")

            # Create future that will be resolved when generation starts
            fut = asyncio.Future[llm.GenerationCreatedEvent]()
            self._pending_generation_fut = fut

            # Send text message asynchronously
            async def _send_text() -> None:
                try:
                    # Wait for session to be fully initialized before sending
                    await self._is_sess_active.wait()
                    await self._send_text_message(instructions, interactive=True)
                except Exception as e:
                    if not fut.done():
                        fut.set_exception(e)
                    if self._pending_generation_fut is fut:
                        self._pending_generation_fut = None

            asyncio.create_task(_send_text())

            # Set timeout from model configuration
            def _on_timeout() -> None:
                if not fut.done():
                    fut.set_exception(
                        llm.RealtimeError("generate_reply timed out waiting for generation")
                    )
                    if self._pending_generation_fut is fut:
                        self._pending_generation_fut = None

            timeout_handle = asyncio.get_running_loop().call_later(
                self._realtime_model._generate_reply_timeout, _on_timeout
            )
            fut.add_done_callback(lambda _: timeout_handle.cancel())

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

        # Send event sequence: contentStart → textInput → contentEnd
        await self._send_raw_event(event)
        await self._send_raw_event(
            self._event_builder.create_text_content_event(content_name, text)
        )
        await self._send_raw_event(self._event_builder.create_content_end_event(content_name))
        logger.info(
            f"Sent text message (interactive={interactive}): {text[:50]}{'...' if len(text) > 50 else ''}"
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

    @utils.log_exceptions(logger=logger)
    async def aclose(self) -> None:
        """Gracefully shut down the realtime session and release network resources."""
        logger.info("attempting to shutdown agent session")
        if not self._is_sess_active.is_set():
            logger.info("agent session already inactive")
            return

        # Cancel any pending generation futures
        if self._pending_generation_fut and not self._pending_generation_fut.done():
            self._pending_generation_fut.set_exception(
                llm.RealtimeError("Session closed while waiting for generation")
            )
            self._pending_generation_fut = None

        for event in self._event_builder.create_prompt_end_block():
            await self._send_raw_event(event)
        # allow event loops to fall out naturally
        # otherwise, the smithy layer will raise an InvalidStateError during cancellation
        self._is_sess_active.clear()

        if self._stream_response and not self._stream_response.output_stream.closed:
            await self._stream_response.output_stream.close()

        # note: even after the self.is_active flag is flipped and the output stream is closed,
        # there is a future inside output_stream.receive() at the AWS-CRT C layer that blocks
        # resulting in an error after cancellation
        # however, it's mostly cosmetic-- the event loop will still exit
        # TODO: fix this nit
        tasks: list[asyncio.Task[Any]] = []

        # Cancel session recycle timer
        if self._session_recycle_task and not self._session_recycle_task.done():
            self._session_recycle_task.cancel()
            try:
                await self._session_recycle_task
            except asyncio.CancelledError:
                pass

        if self._response_task:
            try:
                await asyncio.wait_for(self._response_task, timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("shutdown of output event loop timed out-- cancelling")
                self._response_task.cancel()
            tasks.append(self._response_task)

        # must cancel the audio input task before closing the input stream
        if self._audio_input_task and not self._audio_input_task.done():
            self._audio_input_task.cancel()
            tasks.append(self._audio_input_task)
        if self._stream_response and not self._stream_response.input_stream.closed:
            await self._stream_response.input_stream.close()

        # cancel main task to prevent pending task warnings
        if self._main_atask and not self._main_atask.done():
            self._main_atask.cancel()
            tasks.append(self._main_atask)

        await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug(f"CHAT CONTEXT: {self._chat_ctx.items}")
        logger.info("Session end")
