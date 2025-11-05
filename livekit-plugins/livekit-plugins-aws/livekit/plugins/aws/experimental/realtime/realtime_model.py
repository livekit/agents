# mypy: disable-error-code=unused-ignore

from __future__ import annotations

import ast
import asyncio
import base64
import json
import os
import time
import uuid
import weakref
from collections.abc import Iterator
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
    VOICE_ID,
    SonicEventBuilder as seb,
    Tool,
    ToolConfiguration,
    ToolInputSchema,
    ToolSpec,
)
from .pretty_printer import AnsiColors, log_event_data, log_message

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
DEFAULT_SYSTEM_PROMPT = (
    "Your name is Sonic. You are a friend and eagerly helpful assistant."
    "The user and you will engage in a spoken dialog exchanging the transcripts of a natural real-time conversation."  # noqa: E501
    "Keep your responses short and concise unless the user asks you to elaborate or you are explicitly asked to be verbose and chatty."  # noqa: E501
    "Do not repeat yourself. Do not ask the user to repeat themselves."
    "Do ask the user to confirm or clarify their response if you are not sure what they mean."
    "If after asking the user for clarification you still do not understand, be honest and tell them that you do not understand."  # noqa: E501
    "Do not make up information or make assumptions. If you do not know the answer, tell the user that you do not know the answer."  # noqa: E501
    "If the user makes a request of you that you cannot fulfill, tell them why you cannot fulfill it."  # noqa: E501
    "When making tool calls, inform the user that you are using a tool to generate the response."
    "Avoid formatted lists or numbering and keep your output as a spoken transcript to be acted out."  # noqa: E501
    "Be appropriately emotive when responding to the user. Use American English as the language for your responses."  # noqa: E501
)

lk_bedrock_debug = int(os.getenv("LK_BEDROCK_DEBUG", 0))


@dataclass
class _RealtimeOptions:
    """Configuration container for a Sonic realtime session.

    Attributes:
        voice (VOICE_ID): Voice identifier used for TTS output.
        temperature (float): Sampling temperature controlling randomness; 1.0 is most deterministic.
        top_p (float): Nucleus sampling parameter; 0.0 considers all tokens.
        max_tokens (int): Maximum number of tokens the model may generate in a single response.
        tool_choice (llm.ToolChoice | None): Strategy that dictates how the model should invoke tools.
        region (str): AWS region hosting the Bedrock Sonic model endpoint.
    """  # noqa: E501

    voice: VOICE_ID
    temperature: float
    top_p: float
    max_tokens: int
    tool_choice: llm.ToolChoice | None
    region: str


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


class Boto3CredentialsResolver(IdentityResolver):  # type: ignore[misc]
    """IdentityResolver implementation that sources AWS credentials from boto3.

    The resolver delegates to the default boto3.Session() credential chain which
    checks environment variables, shared credentials files, EC2 instance profiles, etc.
    The credentials are then wrapped in an AWSCredentialsIdentity so they can be
    passed into Bedrock runtime clients.
    """

    def __init__(self) -> None:
        self.session = boto3.Session()  # type: ignore[attr-defined]

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
        try:
            logger.debug("Attempting to load AWS credentials")
            credentials = self.session.get_credentials()
            if not credentials:
                logger.error("Unable to load AWS credentials")
                raise ValueError("Unable to load AWS credentials")

            creds = credentials.get_frozen_credentials()
            logger.debug(
                f"AWS credentials loaded successfully. AWS_ACCESS_KEY_ID: {creds.access_key[:4]}***"
            )

            identity = AWSCredentialsIdentity(
                access_key_id=creds.access_key,
                secret_access_key=creds.secret_key,
                session_token=creds.token if creds.token else None,
                expiration=None,
            )
            return identity
        except Exception as e:
            logger.error(f"Failed to load AWS credentials: {str(e)}")
            raise ValueError(f"Failed to load AWS credentials: {str(e)}")  # noqa: B904


class RealtimeModel(llm.RealtimeModel):
    """High-level entry point that conforms to the LiveKit RealtimeModel interface.

    The object is very light-weight-– it mainly stores default inference options and
    spawns a RealtimeSession when session() is invoked.
    """

    def __init__(
        self,
        *,
        voice: NotGivenOr[VOICE_ID] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        region: NotGivenOr[str] = NOT_GIVEN,
    ):
        """Instantiate a new RealtimeModel.

        Args:
            voice (VOICE_ID | NotGiven): Preferred voice id for Sonic TTS output. Falls back to "tiffany".
            temperature (float | NotGiven): Sampling temperature (0-1). Defaults to DEFAULT_TEMPERATURE.
            top_p (float | NotGiven): Nucleus sampling probability mass. Defaults to DEFAULT_TOP_P.
            max_tokens (int | NotGiven): Upper bound for tokens emitted by the model. Defaults to DEFAULT_MAX_TOKENS.
            tool_choice (llm.ToolChoice | None | NotGiven): Strategy for tool invocation ("auto", "required", or explicit function).
            region (str | NotGiven): AWS region of the Bedrock runtime endpoint.
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
        self.model_id = "amazon.nova-sonic-v1:0"
        # note: temperature and top_p do not follow industry standards and are defined slightly differently for Sonic  # noqa: E501
        # temperature ranges from 0.0 to 1.0, where 0.0 is the most random and 1.0 is the most deterministic  # noqa: E501
        # top_p ranges from 0.0 to 1.0, where 0.0 is the most random and 1.0 is the most deterministic  # noqa: E501
        self.temperature = temperature
        self.top_p = top_p
        self._opts = _RealtimeOptions(
            voice=cast(VOICE_ID, voice) if is_given(voice) else "tiffany",
            temperature=temperature if is_given(temperature) else DEFAULT_TEMPERATURE,
            top_p=top_p if is_given(top_p) else DEFAULT_TOP_P,
            max_tokens=max_tokens if is_given(max_tokens) else DEFAULT_MAX_TOKENS,
            tool_choice=tool_choice or None,
            region=region if is_given(region) else "us-east-1",
        )
        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return self.model_id

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
            aws_credentials_identity_resolver=Boto3CredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()},
            user_agent_extra="x-client-framework:livekit-plugins-aws[realtime]",
        )
        self._bedrock_client = BedrockRuntimeClient(config=config)

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
                if llm.tool_context.is_function_tool(f):
                    description = llm.tool_context.get_function_info(f).description
                    input_schema = llm.utils.build_legacy_openai_schema(f, internally_tagged=True)[
                        "parameters"
                    ]
                elif llm.tool_context.is_raw_function_tool(f):
                    description = llm.tool_context.get_raw_function_info(f).raw_schema.get(
                        "description"
                    )
                    input_schema = llm.tool_context.get_raw_function_info(f).raw_schema[
                        "parameters"
                    ]
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
                        model_id=self._realtime_model.model_id
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
            init_events = self._event_builder.create_prompt_start_block(
                voice_id=self._realtime_model._opts.voice,
                sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE,  # type: ignore
                system_content=self._instructions,
                chat_ctx=self.chat_ctx,
                tool_configuration=self._serialize_tool_config(),
                max_tokens=self._realtime_model._opts.max_tokens,
                top_p=self._realtime_model._opts.top_p,
                temperature=self._realtime_model._opts.temperature,
            )

            for event in init_events:
                await self._send_raw_event(event)
                logger.debug(f"Sent event: {event}")

            if not is_restart:
                self._audio_input_task = asyncio.create_task(
                    self._process_audio_input(), name="RealtimeSession._process_audio_input"
                )

            self._response_task = asyncio.create_task(
                self._process_responses(), name="RealtimeSession._process_responses"
            )
            self._is_sess_active.set()
            logger.debug("Stream initialized successfully")
        except Exception as e:
            logger.debug(f"Failed to initialize stream: {str(e)}")
            raise
        return self

    @utils.log_exceptions(logger=logger)
    def emit_generation_event(self) -> None:
        """Publish a llm.GenerationCreatedEvent to external subscribers."""
        if self._current_generation is None:
            logger.debug("emit_generation_event called but no generation exists - ignoring")
            return

        logger.debug("Emitting generation event")
        generation_ev = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
            response_id=self._current_generation.response_id,
        )
        self.emit("generation_created", generation_ev)

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

            logger.debug(f"Creating new generation, response_id={response_id}")
            self._current_generation = _ResponseGeneration(
                completion_id=completion_id,
                message_ch=utils.aio.Chan(),
                function_ch=utils.aio.Chan(),
                response_id=response_id,
            )
            generation_created = True
        else:
            logger.debug(
                f"Generation already exists: response_id={self._current_generation.response_id}"
            )

        # Always ensure message structure exists (even if generation already exists)
        if self._current_generation.message_gen is None:
            logger.debug(
                f"Creating message structure for response_id={self._current_generation.response_id}"
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
                f"Message structure already exists for response_id={self._current_generation.response_id}"
            )

        # Only emit generation event if we created a new generation
        if generation_created:
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
                logger.debug("ASSISTANT SPECULATIVE text - creating new generation")
                if self._current_generation is not None:
                    logger.debug("Closing previous generation for new assistant response")
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
        if text_content == '{ "interrupted" : true }\n':
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
            self._chat_ctx.add_message(role=role, content=text_content)
        else:
            prev_utterance = self._chat_ctx.items[-1]
            if prev_utterance.type == "message" and prev_utterance.role == role:
                if isinstance(prev_content := prev_utterance.content[0], str) and (
                    len(prev_content.encode("utf-8")) + len(text_content.encode("utf-8"))
                    < MAX_MESSAGE_SIZE
                ):
                    prev_utterance.content[0] = "\n".join([prev_content, text_content])
                else:
                    self._chat_ctx.add_message(role=role, content=text_content)
                    if len(self._chat_ctx.items) > MAX_MESSAGES:
                        self._chat_ctx.truncate(max_items=MAX_MESSAGES)
            else:
                self._chat_ctx.add_message(role=role, content=text_content)
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

    async def _handle_audio_output_content_end_event(self, event_data: dict) -> None:
        """Handle audio content end - log but don't close generation."""
        log_event_data(event_data)
        # Nova Sonic uses one completion for entire session
        # Don't close generation here - wait for new completionStart or session end

    def _close_current_generation(self) -> None:
        """Helper that closes all channels of the active generation."""
        if self._current_generation is None:
            return

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

        logger.debug(
            f"Closed generation for completion_id={self._current_generation.completion_id}"
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
                ) as re:
                    logger.warning(f"Retryable error: {re}\nAttempting to recover...")
                    await self._restart_session(re)
                    break
                except ModelTimeoutException as mte:
                    logger.warning(f"Model timeout error: {mte}\nAttempting to recover...")
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
        """Inject an initial chat history once during the very first session startup."""
        # sometimes fires randomly
        # add a guard here to only allow chat_ctx to be updated on
        # the very first session initialization
        if self._chat_ctx_ready is None:
            self._chat_ctx_ready = asyncio.get_running_loop().create_future()

        if not self._chat_ctx_ready.done():
            self._chat_ctx = chat_ctx.copy()
            logger.debug(f"Chat context updated: {self._chat_ctx.items}")
            self._chat_ctx_ready.set_result(True)

        # for each function tool, send the result to aws
        logger.debug(
            f"update_chat_ctx called with {len(chat_ctx.items)} items, pending_tools: {self._pending_tools}"
        )
        for item in chat_ctx.items:
            if item.type != "function_call_output":
                continue

            logger.debug(
                f"Found function_call_output: call_id={item.call_id}, in_pending={item.call_id in self._pending_tools}"
            )

            if item.call_id not in self._pending_tools:
                continue

            logger.debug(f"function call output: {item}")
            self._pending_tools.discard(item.call_id)
            self._tool_results_ch.send_nowait(
                {
                    "tool_use_id": item.call_id,
                    "tool_result": item.output
                    if not item.is_error
                    else f"{{'error': '{item.output}'}}",
                }
            )

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
    async def update_tools(self, tools: list[llm.FunctionTool | llm.RawFunctionTool | Any]) -> None:
        """Replace the active tool set with tools and notify Sonic if necessary."""
        logger.debug(f"Updating tools: {tools}")
        retained_tools: list[llm.FunctionTool | llm.RawFunctionTool] = []

        for tool in tools:
            retained_tools.append(tool)
        self._tools = llm.ToolContext(retained_tools)
        if retained_tools:
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
                                    json.loads(tool_result)
                                except json.JSONDecodeError:
                                    try:
                                        tool_result = json.dumps(ast.literal_eval(tool_result))
                                    except Exception:
                                        pass

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
        logger.warning("unprompted generation is not supported by Nova Sonic's Realtime API")
        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        fut.set_exception(
            llm.RealtimeError("unprompted generation is not supported by Nova Sonic's Realtime API")
        )
        return fut

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
