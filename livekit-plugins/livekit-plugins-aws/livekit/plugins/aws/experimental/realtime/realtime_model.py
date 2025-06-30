from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

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
    ModelTimeoutException,
    ThrottlingException,
    ValidationException,
)
from smithy_aws_core.identity import AWSCredentialsIdentity
from smithy_core.aio.interfaces.identity import IdentityResolver

from livekit import rtc
from livekit.agents import (
    APIStatusError,
    ToolError,
    llm,
    utils,
)
from livekit.agents.llm.realtime import RealtimeSession
from livekit.agents.metrics import RealtimeModelMetrics
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
    """Book-keeping dataclass tracking the lifecycle of a Sonic turn.

    This object is created whenever we receive a *completion_start* event from the model
    and is disposed of once the assistant turn finishes (e.g. *END_TURN*).

    Attributes:
        message_ch (utils.aio.Chan[llm.MessageGeneration]): Multiplexed stream for all assistant messages.
        function_ch (utils.aio.Chan[llm.FunctionCall]): Stream that emits function tool calls.
        input_id (str): Synthetic message id for the user input of the current turn.
        response_id (str): Synthetic message id for the assistant reply of the current turn.
        messages (dict[str, _MessageGeneration]): Map of message_id -> per-message stream containers.
        user_messages (dict[str, str]): Map Bedrock content_id -> input_id.
        speculative_messages (dict[str, str]): Map Bedrock content_id -> response_id (assistant side).
        tool_messages (dict[str, str]): Map Bedrock content_id -> response_id for tool calls.
        output_text (str): Accumulated assistant text (only used for metrics / debugging).
        _created_timestamp (str): ISO-8601 timestamp when the generation record was created.
        _first_token_timestamp (float | None): Wall-clock time of first token emission.
        _completed_timestamp (float | None): Wall-clock time when the turn fully completed.
    """  # noqa: E501

    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    input_id: str  # corresponds to user's portion of the turn
    response_id: str  # corresponds to agent's portion of the turn
    messages: dict[str, _MessageGeneration] = field(default_factory=dict)
    user_messages: dict[str, str] = field(default_factory=dict)
    speculative_messages: dict[str, str] = field(default_factory=dict)
    tool_messages: dict[str, str] = field(default_factory=dict)
    output_text: str = ""  # agent ASR text
    _created_timestamp: str = field(default_factory=datetime.now().isoformat())
    _first_token_timestamp: float | None = None
    _completed_timestamp: float | None = None


class Boto3CredentialsResolver(IdentityResolver):
    """IdentityResolver implementation that sources AWS credentials from boto3.

    The resolver delegates to the default boto3.Session() credential chain which
    checks environment variables, shared credentials files, EC2 instance profiles, etc.
    The credentials are then wrapped in an AWSCredentialsIdentity so they can be
    passed into Bedrock runtime clients.
    """

    def __init__(self):
        self.session = boto3.Session()

    async def get_identity(self, **kwargs):
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
            )
        )
        self.model_id = "amazon.nova-sonic-v1:0"
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
        )
        self._sessions = weakref.WeakSet[RealtimeSession]()

    def session(self) -> RealtimeSession:
        """Return a new RealtimeSession bound to this model instance."""
        sess = RealtimeSession(self)

        # note: this is a hack to get the session to initialize itself
        # TODO: change how RealtimeSession is initialized by creating a single task main_atask that spawns subtasks  # noqa: E501
        asyncio.create_task(sess.initialize_streams())
        self._sessions.add(sess)
        return sess

        # stub b/c RealtimeSession.aclose() is invoked directly
        async def aclose(self) -> None:
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
        self._realtime_model = realtime_model
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
        self._is_sess_active = asyncio.Event()
        self._chat_ctx = llm.ChatContext.empty()
        self._tools = llm.ToolContext.empty()
        self._tool_type_map = {}
        self._tool_results_ch = utils.aio.Chan[dict[str, str]]()
        self._tools_ready = asyncio.get_running_loop().create_future()
        self._instructions_ready = asyncio.get_running_loop().create_future()
        self._chat_ctx_ready = asyncio.get_running_loop().create_future()
        self._instructions = DEFAULT_SYSTEM_PROMPT
        self._audio_input_chan = utils.aio.Chan[bytes]()
        self._current_generation: _ResponseGeneration | None = None

        # note: currently tracks session restart attempts across all sessions
        # TODO: track restart attempts per turn
        self._session_restart_attempts = 0

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
            self.emit, streams_provider=self._current_generation_streams
        )

    def _current_generation_streams(
        self,
    ) -> tuple[utils.aio.Chan[llm.MessageGeneration], utils.aio.Chan[llm.FunctionCall]]:
        return (self._current_generation.message_ch, self._current_generation.function_ch)

    @utils.log_exceptions(logger=logger)
    def _initialize_client(self):
        """Instantiate the Bedrock runtime client"""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self._realtime_model._opts.region}.amazonaws.com",
            region=self._realtime_model._opts.region,
            aws_credentials_identity_resolver=Boto3CredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()},
        )
        self._bedrock_client = BedrockRuntimeClient(config=config)

    @utils.log_exceptions(logger=logger)
    async def _send_raw_event(self, event_json):
        """Low-level helper that serialises event_json and forwards it to the bidirectional stream.

        Args:
            event_json (dict | str): The JSON payload (already in Bedrock wire format) to queue.

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
                    self._tool_type_map[name] = "FunctionTool"
                else:
                    description = llm.tool_context.get_raw_function_info(f).raw_schema.get(
                        "description"
                    )
                    input_schema = llm.tool_context.get_raw_function_info(f).raw_schema[
                        "parameters"
                    ]
                    self._tool_type_map[name] = "RawFunctionTool"

                tool = Tool(
                    toolSpec=ToolSpec(
                        name=name,
                        description=description,
                        inputSchema=ToolInputSchema(json_=json.dumps(input_schema)),
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
    async def initialize_streams(self, is_restart: bool = False):
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

            logger.info("Initializing Bedrock stream")
            self._stream_response = (
                await self._bedrock_client.invoke_model_with_bidirectional_stream(
                    InvokeModelWithBidirectionalStreamOperationInput(
                        model_id=self._realtime_model.model_id
                    )
                )
            )

            if not is_restart:
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
                    if not self._tools_ready.done():
                        logger.warning("Tools not ready after 500ms, continuing without them")

                    if not self._instructions_ready.done():
                        logger.warning(
                            "Instructions not received after 500ms, proceeding with default instructions"  # noqa: E501
                        )
                    if not self._chat_ctx_ready.done():
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
                sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE,
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
            self._is_sess_active.set_exception(e)
            logger.debug(f"Failed to initialize stream: {str(e)}")
            raise
        return self

    @utils.log_exceptions(logger=logger)
    def _emit_generation_event(self) -> None:
        """Publish a llm.GenerationCreatedEvent to external subscribers."""
        logger.debug("Emitting generation event")
        generation_ev = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
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
        log_event_data(event_data)
        self._create_response_generation()

    def _create_response_generation(self) -> None:
        """Instantiate _ResponseGeneration and emit the GenerationCreated event."""
        if self._current_generation is None:
            self._current_generation = _ResponseGeneration(
                message_ch=utils.aio.Chan(),
                function_ch=utils.aio.Chan(),
                input_id=str(uuid.uuid4()),
                response_id=str(uuid.uuid4()),
                messages={},
                user_messages={},
                speculative_messages={},
                _created_timestamp=datetime.now().isoformat(),
            )
            msg_gen = _MessageGeneration(
                message_id=self._current_generation.response_id,
                text_ch=utils.aio.Chan(),
                audio_ch=utils.aio.Chan(),
            )
            self._current_generation.message_ch.send_nowait(
                llm.MessageGeneration(
                    message_id=msg_gen.message_id,
                    text_stream=msg_gen.text_ch,
                    audio_stream=msg_gen.audio_ch,
                )
            )
            self._current_generation.messages[self._current_generation.response_id] = msg_gen

    # will be completely ignoring post-ASR text events
    async def _handle_text_output_content_start_event(self, event_data: dict) -> None:
        """Handle text_output_content_start for both user and assistant roles."""
        log_event_data(event_data)
        role = event_data["event"]["contentStart"]["role"]

        # note: does not work if you emit llm.GCE too early (for some reason)
        if role == "USER":
            self._create_response_generation()
            content_id = event_data["event"]["contentStart"]["contentId"]
            self._current_generation.user_messages[content_id] = self._current_generation.input_id

        elif (
            role == "ASSISTANT"
            and "SPECULATIVE" in event_data["event"]["contentStart"]["additionalModelFields"]
        ):
            text_content_id = event_data["event"]["contentStart"]["contentId"]
            self._current_generation.speculative_messages[text_content_id] = (
                self._current_generation.response_id
            )

    async def _handle_text_output_content_event(self, event_data: dict) -> None:
        """Stream partial text tokens into the current _MessageGeneration."""
        log_event_data(event_data)
        text_content_id = event_data["event"]["textOutput"]["contentId"]
        text_content = f"{event_data['event']['textOutput']['content']}\n"

        # currently only agent can be interrupted
        if text_content == '{ "interrupted" : true }\n':
            # the interrupted flag is not being set correctly in chat_ctx
            # this is b/c audio playback is desynced from text transcription
            # TODO: fix this; possibly via a playback timer
            idx = self._chat_ctx.find_insertion_index(created_at=time.time()) - 1
            logger.debug(
                f"BARGE-IN DETECTED using idx: {idx} and chat_msg: {self._chat_ctx.items[idx]}"
            )
            self._chat_ctx.items[idx].interrupted = True
            self._close_current_generation()
            return

        # ignore events until turn starts
        if self._current_generation is not None:
            # TODO: rename event to llm.InputTranscriptionUpdated
            if (
                self._current_generation.user_messages.get(text_content_id)
                == self._current_generation.input_id
            ):
                logger.debug(f"INPUT TRANSCRIPTION UPDATED: {text_content}")
                # note: user ASR text is slightly different than what is sent to LiveKit (newline vs whitespace)  # noqa: E501
                # TODO: fix this
                self._update_chat_ctx(role="user", text_content=text_content)

            elif (
                self._current_generation.speculative_messages.get(text_content_id)
                == self._current_generation.response_id
            ):
                curr_gen = self._current_generation.messages[self._current_generation.response_id]
                curr_gen.text_ch.send_nowait(text_content)
                # note: this update is per utterance, not per turn
                self._update_chat_ctx(role="assistant", text_content=text_content)

    def _update_chat_ctx(self, role: str, text_content: str) -> None:
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
            if prev_utterance.role == role:
                if (
                    len(prev_utterance.content[0].encode("utf-8"))
                    + len(text_content.encode("utf-8"))
                    < MAX_MESSAGE_SIZE
                ):
                    prev_utterance.content[0] = "\n".join([prev_utterance.content[0], text_content])
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
        """Mark the assistant message closed when Bedrock signals END_TURN."""
        stop_reason = event_data["event"]["contentEnd"]["stopReason"]
        text_content_id = event_data["event"]["contentEnd"]["contentId"]
        if (
            self._current_generation
            is not None  # means that first utterance in the turn was an interrupt
            and self._current_generation.speculative_messages.get(text_content_id)
            == self._current_generation.response_id
            and stop_reason == "END_TURN"
        ):
            log_event_data(event_data)
            self._close_current_generation()

    async def _handle_tool_output_content_start_event(self, event_data: dict) -> None:
        """Track mapping content_id -> response_id for upcoming tool use."""
        log_event_data(event_data)
        tool_use_content_id = event_data["event"]["contentStart"]["contentId"]
        self._current_generation.tool_messages[tool_use_content_id] = (
            self._current_generation.response_id
        )

    # note: tool calls are synchronous for now
    async def _handle_tool_output_content_event(self, event_data: dict) -> None:
        """Execute the referenced tool locally and forward results back to Bedrock."""
        log_event_data(event_data)
        tool_use_content_id = event_data["event"]["toolUse"]["contentId"]
        tool_use_id = event_data["event"]["toolUse"]["toolUseId"]
        tool_name = event_data["event"]["toolUse"]["toolName"]
        if (
            self._current_generation.tool_messages.get(tool_use_content_id)
            == self._current_generation.response_id
        ):
            args = event_data["event"]["toolUse"]["content"]
            self._current_generation.function_ch.send_nowait(
                llm.FunctionCall(
                    call_id=tool_use_id,
                    name=tool_name,
                    arguments=args,
                )
            )

            # note: may need to inject RunContext here...
            tool_type = self._tool_type_map[tool_name]
            if tool_type == "FunctionTool":
                tool_result = await self.tools.function_tools[tool_name](**json.loads(args))
            elif tool_type == "RawFunctionTool":
                tool_result = await self.tools.function_tools[tool_name](json.loads(args))
            else:
                raise ValueError(f"Unknown tool type: {tool_type}")
            logger.debug(f"TOOL ARGS: {args}\nTOOL RESULT: {tool_result}")

            # Sonic only accepts Structured Output for tool results
            # therefore, must JSON stringify ToolError
            if isinstance(tool_result, ToolError):
                logger.warning(f"TOOL ERROR: {tool_name} {tool_result.message}")
                tool_result = {"error": tool_result.message}
            self._tool_results_ch.send_nowait(
                {
                    "tool_use_id": tool_use_id,
                    "tool_result": tool_result,
                }
            )

    async def _handle_tool_output_content_end_event(self, event_data: dict) -> None:
        log_event_data(event_data)

    async def _handle_audio_output_content_start_event(self, event_data: dict) -> None:
        """Associate the upcoming audio chunk with the active assistant message."""
        if self._current_generation is not None:
            log_event_data(event_data)
            audio_content_id = event_data["event"]["contentStart"]["contentId"]
            self._current_generation.speculative_messages[audio_content_id] = (
                self._current_generation.response_id
            )

    async def _handle_audio_output_content_event(self, event_data: dict) -> None:
        """Decode base64 audio from Bedrock and forward it to the audio stream."""
        if (
            self._current_generation is not None
            and self._current_generation.speculative_messages.get(
                event_data["event"]["audioOutput"]["contentId"]
            )
            == self._current_generation.response_id
        ):
            audio_content = event_data["event"]["audioOutput"]["content"]
            audio_bytes = base64.b64decode(audio_content)
            curr_gen = self._current_generation.messages[self._current_generation.response_id]
            curr_gen.audio_ch.send_nowait(
                rtc.AudioFrame(
                    data=audio_bytes,
                    sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE,
                    num_channels=DEFAULT_CHANNELS,
                    samples_per_channel=len(audio_bytes) // 2,
                )
            )

    async def _handle_audio_output_content_end_event(self, event_data: dict) -> None:
        """Close the assistant message streams once Bedrock finishes audio for the turn."""
        if (
            self._current_generation is not None
            and event_data["event"]["contentEnd"]["stopReason"] == "END_TURN"
            and self._current_generation.speculative_messages.get(
                event_data["event"]["contentEnd"]["contentId"]
            )
            == self._current_generation.response_id
        ):
            log_event_data(event_data)
            self._close_current_generation()

    def _close_current_generation(self) -> None:
        """Helper that closes all channels of the active _ResponseGeneration."""
        if self._current_generation is not None:
            if self._current_generation.response_id in self._current_generation.messages:
                curr_gen = self._current_generation.messages[self._current_generation.response_id]
                if not curr_gen.audio_ch.closed:
                    curr_gen.audio_ch.close()
                if not curr_gen.text_ch.closed:
                    curr_gen.text_ch.close()

            if not self._current_generation.message_ch.closed:
                self._current_generation.message_ch.close()
            if not self._current_generation.function_ch.closed:
                self._current_generation.function_ch.close()

            self._current_generation = None

    async def _handle_completion_end_event(self, event_data: dict) -> None:
        log_event_data(event_data)

    async def _handle_other_event(self, event_data: dict) -> None:
        log_event_data(event_data)

    async def _handle_usage_event(self, event_data: dict) -> None:
        # log_event_data(event_data)
        # TODO: implement duration and ttft
        input_tokens = event_data["event"]["usageEvent"]["details"]["delta"]["input"]
        output_tokens = event_data["event"]["usageEvent"]["details"]["delta"]["output"]
        # Q: should we be counting per turn or utterance?
        metrics = RealtimeModelMetrics(
            label=self._realtime_model._label,
            # TODO: pass in the correct request_id
            request_id=event_data["event"]["usageEvent"]["completionId"],
            timestamp=time.monotonic(),
            duration=0,
            ttft=0,
            cancelled=False,
            input_tokens=input_tokens["speechTokens"] + input_tokens["textTokens"],
            output_tokens=output_tokens["speechTokens"] + output_tokens["textTokens"],
            total_tokens=input_tokens["speechTokens"]
            + input_tokens["textTokens"]
            + output_tokens["speechTokens"]
            + output_tokens["textTokens"],
            # need duration to calculate this
            tokens_per_second=0,
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
        )
        self.emit("metrics_collected", metrics)

    @utils.log_exceptions(logger=logger)
    async def _process_responses(self):
        """Background task that drains Bedrock's output stream and feeds the event handlers."""
        try:
            await self._is_sess_active.wait()

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
                        request_id = ve.split(" ")[0].split("=")[1]
                        self.emit(
                            "error",
                            llm.RealtimeModelError(
                                timestamp=time.monotonic(),
                                label=self._realtime_model._label,
                                error=APIStatusError(
                                    message=ve.message,
                                    status_code=400,
                                    request_id=request_id,
                                    body=ve,
                                    retryable=False,
                                ),
                                recoverable=False,
                            ),
                        )
                        raise
                except (ThrottlingException, ModelNotReadyException, ModelErrorException) as re:
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
                                message=e.message,
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
        if self._session_restart_attempts >= DEFAULT_MAX_SESSION_RESTART_ATTEMPTS:
            logger.error("Max session restart attempts reached, exiting")
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
        self._session_restart_attempts += 1
        self._is_sess_active.clear()
        delay = 2 ** (self._session_restart_attempts - 1) - 1
        await asyncio.sleep(min(delay, DEFAULT_MAX_SESSION_RESTART_DELAY))
        await self.initialize_streams(is_restart=True)
        logger.info(
            f"Session restarted successfully ({self._session_restart_attempts}/{DEFAULT_MAX_SESSION_RESTART_ATTEMPTS})"  # noqa: E501
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
        self._instructions_ready.set_result(True)
        logger.debug(f"Instructions updated: {instructions}")

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """Inject an initial chat history once during the very first session startup."""
        # sometimes fires randomly
        # add a guard here to only allow chat_ctx to be updated on
        # the very first session initialization
        if not self._chat_ctx_ready.done():
            self._chat_ctx = chat_ctx.copy()
            logger.debug(f"Chat context updated: {self._chat_ctx.items}")
            self._chat_ctx_ready.set_result(True)

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

    def _tool_choice_adapter(self, tool_choice: llm.ToolChoice) -> dict[str, dict[str, str]] | None:
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
    async def _process_audio_input(self):
        """Background task that feeds audio and tool results into the Bedrock stream."""
        await self._send_raw_event(self._event_builder.create_audio_content_start_event())
        logger.info("Starting audio input processing loop")
        while self._is_sess_active.is_set():
            try:
                # note: could potentially pull this out into a separate task
                try:
                    val = self._tool_results_ch.recv_nowait()
                    tool_result = val["tool_result"]
                    tool_use_id = val["tool_use_id"]
                    await self._send_tool_events(tool_use_id, tool_result)

                except utils.aio.channel.ChanEmpty:
                    pass
                except utils.aio.channel.ChanClosed:
                    logger.warning(
                        "tool results channel closed, exiting audio input processing loop"
                    )
                    break

                try:
                    audio_bytes = await self._audio_input_chan.recv()
                    blob = base64.b64encode(audio_bytes)
                    audio_event = self._event_builder.create_audio_input_event(
                        audio_content=blob.decode("utf-8"),
                    )

                    await self._send_raw_event(audio_event)
                except utils.aio.channel.ChanEmpty:
                    pass
                except utils.aio.channel.ChanClosed:
                    logger.warning(
                        "audio input channel closed, exiting audio input processing loop"
                    )
                    break

            except asyncio.CancelledError:
                logger.info("Audio processing loop cancelled")
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

    def commit_audio(self) -> None:
        logger.warning("commit_audio is not supported by Nova Sonic's Realtime API")

    def clear_audio(self) -> None:
        logger.warning("clear_audio is not supported by Nova Sonic's Realtime API")

    def push_video(self, frame: rtc.VideoFrame) -> None:
        logger.warning("video is not supported by Nova Sonic's Realtime API")

    def interrupt(self) -> None:
        logger.warning("interrupt is not supported by Nova Sonic's Realtime API")

    def truncate(self, *, message_id: str, audio_end_ms: int) -> None:
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
        if self._response_task:
            try:
                await asyncio.wait_for(self._response_task, timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("shutdown of output event loop timed out-- cancelling")
                self._response_task.cancel()

        # must cancel the audio input task before closing the input stream
        if self._audio_input_task and not self._audio_input_task.done():
            self._audio_input_task.cancel()
        if self._stream_response and not self._stream_response.input_stream.closed:
            await self._stream_response.input_stream.close()

        await asyncio.gather(self._response_task, self._audio_input_task, return_exceptions=True)
        logger.debug(f"CHAT CONTEXT: {self._chat_ctx.items}")
        logger.info("Session end")
