from __future__ import annotations

import asyncio
import base64
from datetime import datetime
import json
import time
import traceback
import uuid
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Dict
from livekit import rtc
from livekit.agents import (
    llm,
    utils,
    ToolError,
    APIError,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
)
from livekit.agents.llm.realtime import RealtimeSession
from livekit.agents.llm.tool_context import (
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart,
    ValidationException,
    ModelTimeoutException,
    ThrottlingException,
    ModelNotReadyException,
    ModelErrorException,
)
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from livekit.plugins.aws.experimental.realtime.turn_tracker import _TurnTracker
from smithy_aws_core.identity import AWSCredentialsIdentity
from smithy_core.aio.interfaces.identity import IdentityResolver
import boto3
from ...log import logger
from .events import (
    SonicEventBuilder as seb,
    VOICE_ID,
    ToolConfiguration,
    Tool,
    ToolSpec,
    ToolInputSchema,
    Event,
)
from .pretty_printer import log_event_data, log_message, AnsiColors


INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
SAMPLE_SIZE_BITS = 16
CHANNELS = 1
CHUNK_SIZE = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 1024

DEFAULT_SYSTEM_PROMPT = (
    "Your name is Sonic. You are a friend and eagerly helpful assistant."
    "The user and you will engage in a spoken dialog exchanging the transcripts of a natural real-time conversation."
    "Keep your responses short and concise unless the user asks you to elaborate or you are explicitly asked to be verbose and chatty."
    "Do not repeat yourself. Do not ask the user to repeat themselves."
    "Do ask the user to confirm or clarify their response if you are not sure what they mean."
    "If after asking the user for clarification you still do not understand, be honest and tell them that you do not understand."
    "Do not make up information or make assumptions. If you do not know the answer, tell the user that you do not know the answer."
    "If the user makes a request of you that you cannot fulfill, tell them why you cannot fulfill it."
    "When making tool calls, inform the user that you are using a tool to generate the response."
    "Avoid formatted lists or numbering and keep your output as a spoken transcript to be acted out."
    "Be appropriately emotive when responding to the user. Use American English as the language for your responses."
)


@dataclass
class _RealtimeOptions:
    voice: VOICE_ID
    temperature: float
    top_p: float
    max_tokens: int
    tool_choice: llm.ToolChoice | None
    region: str


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]


@dataclass
class _ResponseGeneration:
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
    def __init__(self):
        self.session = boto3.Session()

    async def get_identity(self, **kwargs):
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
            raise ValueError(f"Failed to load AWS credentials: {str(e)}")


class RealtimeModel(llm.RealtimeModel):
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
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=True,
            )
        )
        self.model_id = "amazon.nova-sonic-v1:0"
        # note: temperature and top_p do not follow industry standards and are defined slightly differently for Sonic
        # temperature ranges from 0.0 to 1.0, where 0.0 is the most random and 1.0 is the most deterministic
        # top_p ranges from 0.0 to 1.0, where 0.0 is the most random and 1.0 is the most deterministic
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
        sess = RealtimeSession(self)

        # note: this is a hack to get the session to initialize itself
        # TODO: change how RealtimeSession is initialized by creating a single task main_atask that spawns subtasks
        asyncio.create_task(sess.initialize_streams())
        self._sessions.add(sess)
        return sess

        # stub b/c RealtimeSession.aclose() is invoked directly
        async def aclose(self) -> None:
            pass


class RealtimeSession(
    llm.RealtimeSession[Literal["bedrock_server_event_received", "bedrock_client_event_queued"]]
):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model = realtime_model
        self._event_builder = seb(
            prompt_name=str(uuid.uuid4()),
            audio_content_name=str(uuid.uuid4()),
        )
        self._input_resampler: rtc.AudioResampler | None = None
        self._bstream = utils.audio.AudioByteStream(
            INPUT_SAMPLE_RATE, CHANNELS, samples_per_channel=CHUNK_SIZE
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
        if not self._stream_response:
            logger.warning("stream not initialized; dropping event (this should never occur)")
            return

        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )

        try:
            await self._stream_response.input_stream.send(event)
        except Exception as e:
            logger.error(f"Error sending event: {str(e)}")
            traceback.print_exc()
            raise

    def _serialize_tool_config(self) -> ToolConfiguration | None:
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

        for tool in tool_cfg.tools:
            logger.debug(f"TOOL CONFIGURATION: {tool.toolSpec.inputSchema}")
        return tool_cfg

    @utils.log_exceptions(logger=logger)
    async def initialize_streams(self, is_restart: bool = False):
        try:
            if not self._bedrock_client:
                logger.debug("Creating Bedrock client")
                self._initialize_client()

            logger.debug("Initializing Bedrock stream")
            self._stream_response = (
                await self._bedrock_client.invoke_model_with_bidirectional_stream(
                    InvokeModelWithBidirectionalStreamOperationInput(
                        model_id=self._realtime_model.model_id
                    )
                )
            )

            if is_restart:
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
                        await asyncio.wait_for(asyncio.gather(*pending_events), timeout=1.0)
                except asyncio.TimeoutError:
                    if not self._tools_ready.done():
                        logger.warning("Tools not ready after 1sec, continuing without them")

                    if not self._instructions_ready.done():
                        logger.warning(
                            "Instructions not received after 1sec, proceeding with default instructions"
                        )
                    if not self._chat_ctx_ready.done():
                        logger.warning(
                            "Chat context not received after 1sec, proceeding with empty chat context"
                        )

            logger.debug(
                f"Initializing Bedrock session with realtime options: {self._realtime_model._opts}"
            )
            init_events = self._event_builder.create_prompt_start_block(
                voice_id=self._realtime_model._opts.voice,
                sample_rate=OUTPUT_SAMPLE_RATE,
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
        logger.debug(f"Emitting generation event")
        generation_ev = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
        )
        self.emit("generation_created", generation_ev)
        # logger.debug(f"Emitted generation event: {generation_ev}")

    @utils.log_exceptions(logger=logger)
    async def _handle_event(self, event_data: dict) -> None:
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
                # note: user ASR text is slightly different than what is sent to LiveKit (newline vs whitespace)
                # TODO: fix this
                self._chat_ctx.add_message(role="user", content=text_content)

            elif (
                self._current_generation.speculative_messages.get(text_content_id)
                == self._current_generation.response_id
            ):
                logger.debug(
                    f"SENDING TEXT CONTENT TO MESSAGE CH: {text_content} with response_id: {self._current_generation.response_id} and content_id: {text_content_id}"
                )

                curr_gen = self._current_generation.messages[self._current_generation.response_id]
                curr_gen.text_ch.send_nowait(text_content)
                self._chat_ctx.add_message(role="assistant", content=text_content)

    # cannot rely on this event for user b/c stopReason=PARTIAL_TURN always for user
    async def _handle_text_output_content_end_event(self, event_data: dict) -> None:
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
        log_event_data(event_data)
        tool_use_content_id = event_data["event"]["contentStart"]["contentId"]
        self._current_generation.tool_messages[tool_use_content_id] = (
            self._current_generation.response_id
        )

    # note: tool calls are synchronous for now
    async def _handle_tool_output_content_event(self, event_data: dict) -> None:
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
            # need to invoke tool function here
            logger.debug(f"TOOL ARGS: {args}")
            # note: may need to inject RunContext here...
            tool_type = self._tool_type_map[tool_name]
            if tool_type == "FunctionTool":
                tool_result = await self.tools.function_tools[tool_name](**json.loads(args))
            elif tool_type == "RawFunctionTool":
                tool_result = await self.tools.function_tools[tool_name](json.loads(args))
            else:
                raise ValueError(f"Unknown tool type: {tool_type}")
            logger.debug(f"TOOL RESULT: {tool_result}")

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
        if self._current_generation is not None:
            log_event_data(event_data)
            audio_content_id = event_data["event"]["contentStart"]["contentId"]
            self._current_generation.speculative_messages[audio_content_id] = (
                self._current_generation.response_id
            )

    async def _handle_audio_output_content_event(self, event_data: dict) -> None:
        if (
            self._current_generation is not None
            and self._current_generation.speculative_messages.get(
                event_data["event"]["audioOutput"]["contentId"]
            )
            == self._current_generation.response_id
        ):
            audio_content = event_data["event"]["audioOutput"]["content"]
            audio_bytes = base64.b64decode(audio_content)
            # logger.debug(f"SENDING AUDIO CONTENT TO MESSAGE CH: {len(audio_bytes)} bytes")
            curr_gen = self._current_generation.messages[self._current_generation.response_id]
            curr_gen.audio_ch.send_nowait(
                rtc.AudioFrame(
                    data=audio_bytes,
                    sample_rate=OUTPUT_SAMPLE_RATE,
                    num_channels=CHANNELS,
                    samples_per_channel=len(audio_bytes) // 2,
                )
            )

    async def _handle_audio_output_content_end_event(self, event_data: dict) -> None:
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
        pass

    @utils.log_exceptions(logger=logger)
    async def _process_responses(self):
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
                            await self._handle_event(json_data)
                        except json.JSONDecodeError:
                            logger.warning(f"JSON decode error: {response_data}")
                    else:
                        logger.debug("No response received")
                except asyncio.InvalidStateError:
                    logger.debug("Response processing task invalid state error")
                    break
                except asyncio.CancelledError:
                    logger.debug("Response processing task cancelled")
                    self._close_current_generation()
                    raise
                except ValidationException as ve:
                    # there is a 3min no-activity (e.g. silence) timeout on the stream, after which the stream is closed
                    if (
                        "InternalErrorCode=531::RST_STREAM closed stream. HTTP/2 error code: NO_ERROR"
                        in ve.message
                    ):
                        logger.error(f"Validation error: {ve}\nAttempting to recover...")
                        self._is_sess_active.clear()
                        await self.initialize_streams(is_restart=True)
                        logger.info("Session restarted successfully")

                    else:
                        logger.error(f"Validation error: {ve}")
                        break
                except (ThrottlingException, ModelNotReadyException, ModelErrorException) as re:
                    logger.error(f"Retryable error: {re}")
                    self._is_sess_active.clear()
                    await self.initialize_streams(is_restart=True)
                    logger.info("Session restarted successfully")

                    break
                except ModelTimeoutException as mte:
                    logger.warning(f"Model timeout error: {mte}\nAttempting to recover...")
                    self._is_sess_active.clear()
                    await self.initialize_streams(is_restart=True)
                    logger.info("Session restarted successfully")

                    break
                except Exception as e:
                    logger.debug(f"error type: {type(e)}")
                    logger.error(f"Response processing error: {e}")
                    break
                finally:
                    pass
        finally:
            self._is_sess_active.clear()

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    async def update_instructions(self, instructions: str) -> None:
        self._instructions = instructions
        self._instructions_ready.set_result(True)
        logger.debug(f"Instructions updated: {instructions}")

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        # sometimes fires randomly
        # add a guard here to only allow chat_ctx to be updated on
        # the very first session initialization
        if not self._chat_ctx_ready.done():
            self._chat_ctx = chat_ctx.copy()
            logger.debug(f"Chat context updated: {self._chat_ctx.items}")
            self._chat_ctx_ready.set_result(True)

    async def _send_tool_events(self, tool_use_id: str, tool_result: str) -> None:
        tool_content_name = str(uuid.uuid4())
        tool_events = self._event_builder.create_tool_content_block(
            content_name=tool_content_name,
            tool_use_id=tool_use_id,
            content=tool_result,
        )
        for event in tool_events:
            await self._send_raw_event(event)
            # logger.debug(f"Sent tool event: {event}")

    def _tool_choice_adapter(self, tool_choice: llm.ToolChoice) -> Dict[str, Dict[str, str]] | None:
        if tool_choice == "auto":
            return {"auto": {}}
        elif tool_choice == "required":
            return {"any": {}}
        elif isinstance(tool_choice, dict) and tool_choice["type"] == "function":
            return {"tool": {"name": tool_choice["function"]["name"]}}
        else:
            return None

    # note: return value from tool functions registered to Sonic must be Structured Output (aka a dict that is JSON serializable)
    async def update_tools(self, tools: list[llm.FunctionTool | llm.RawFunctionTool | Any]) -> None:
        logger.debug(f"Updating tools: {tools}")
        retained_tools: list[llm.FunctionTool | llm.RawFunctionTool] = []

        for tool in tools:
            retained_tools.append(tool)
        self._tools = llm.ToolContext(retained_tools)
        if retained_tools:
            self._tools_ready.set_result(True)
            logger.debug("Tool list has been injected")

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        logger.warning(
            "updating inference configuration options is not yet supported by Nova Sonic's Realtime API"
        )

    @utils.log_exceptions(logger=logger)
    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != INPUT_SAMPLE_RATE or frame.num_channels != CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=INPUT_SAMPLE_RATE,
                num_channels=CHANNELS,
            )

        if self._input_resampler:
            # flush the resampler when the input source is changed
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    @utils.log_exceptions(logger=logger)
    async def _process_audio_input(self):
        await self._send_raw_event(self._event_builder.create_audio_content_start_event())
        logger.debug("Starting audio input processing loop")
        while self._is_sess_active.is_set():
            try:
                # note: could potentially pull this out into a separate task
                try:
                    # logger.debug("attempting to recv tool result")
                    val = self._tool_results_ch.recv_nowait()
                    logger.debug(f"TOOL RESULT: {val}")
                    tool_result = val["tool_result"]
                    tool_use_id = val["tool_use_id"]
                    await self._send_tool_events(tool_use_id, tool_result)

                except utils.aio.channel.ChanEmpty:
                    # logger.debug("No tool results received")
                    pass
                except utils.aio.channel.ChanClosed:
                    logger.warning(
                        "tool results channel closed, exiting audio input processing loop"
                    )
                    break

                try:
                    # logger.debug("attempting to recv audio data")
                    audio_bytes = await self._audio_input_chan.recv()
                    blob = base64.b64encode(audio_bytes)
                    # logger.debug(f"Sending audio data to Bedrock: size={len(audio_bytes)} bytes")
                    audio_event = self._event_builder.create_audio_input_event(
                        audio_content=blob.decode("utf-8"),
                    )

                    await self._send_raw_event(audio_event)
                    # logger.debug("audio event sent to Bedrock")
                except utils.aio.channel.ChanEmpty:
                    # logger.debug("No audio data received")
                    pass
                except utils.aio.channel.ChanClosed:
                    logger.warning(
                        "audio input channel closed, exiting audio input processing loop"
                    )
                    break

            except asyncio.CancelledError:
                logger.debug("Audio processing loop cancelled")
                self._audio_input_chan.close()
                self._tool_results_ch.close()
                raise
            except Exception as e:
                logger.debug(f"Error processing audio: {e}")
                traceback.print_exc()

    # for debugging purposes only
    def _log_significant_audio(self, audio_bytes: bytes) -> None:
        squared_sum = sum(sample**2 for sample in audio_bytes)
        if (squared_sum / len(audio_bytes)) ** 0.5 > 200:
            log_message("Enqueuing significant audio chunk", AnsiColors.BLUE)

    @utils.log_exceptions(logger=logger)
    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if not self._audio_input_chan.closed:
            # logger.debug(f"Raw audio received: samples={len(frame.data)} rate={frame.sample_rate} channels={frame.num_channels}")
            for f in self._resample_audio(frame):
                # logger.debug(f"Resampled audio: samples={len(frame.data)} rate={frame.sample_rate} channels={frame.num_channels}")

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
        logger.debug("aclose invoked")
        if not self._is_sess_active.is_set():
            logger.debug("already inactive within aclose")
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
                logger.debug("shutdown of output event loop timed out-- cancelling")
                self._response_task.cancel()

        # must cancel the audio input task before closing the input stream
        if self._audio_input_task and not self._audio_input_task.done():
            self._audio_input_task.cancel()
        if self._stream_response and not self._stream_response.input_stream.closed:
            await self._stream_response.input_stream.close()

        await asyncio.gather(self._response_task, self._audio_input_task, return_exceptions=True)

        logger.debug("Session end")
        logger.debug(f"CHAT CONTEXT: {self._chat_ctx.items}")
