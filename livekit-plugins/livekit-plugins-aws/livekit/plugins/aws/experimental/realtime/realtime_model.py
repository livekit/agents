from __future__ import annotations

import asyncio
import base64
import json
import traceback
import uuid
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal
from livekit.agents.llm.chat_context import ChatContext
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.llm.realtime import RealtimeSession
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart,
)
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from smithy_aws_core.identity import AWSCredentialsIdentity
from smithy_core.aio.interfaces.identity import IdentityResolver
from smithy_core.exceptions import SmithyException
import boto3
from ...log import logger
from .events import SonicEventBuilder as seb


INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
SAMPLE_SIZE_BITS = 16
CHANNELS = 1
CHUNK_SIZE = 512
DEFAULT_TEMPERATURE = 0.7


@dataclass
class _RealtimeOptions:
    model: str
    voice: str
    temperature: float
    tool_choice: llm.ToolChoice | None


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


@dataclass
class _CreateResponseHandle:
    instructions: NotGivenOr[str]
    done_fut: asyncio.Future[llm.GenerationCreatedEvent]
    timeout: asyncio.TimerHandle | None = None

    def timeout_start(self) -> None:
        if self.timeout or self.done_fut is None or self.done_fut.done():
            return

        def _on_timeout() -> None:
            if not self.done_fut.done():
                self.done_fut.set_exception(llm.RealtimeError("generate_reply timed out."))

        self.timeout = asyncio.get_event_loop().call_later(5.0, _on_timeout)
        self.done_fut.add_done_callback(lambda _: self.timeout.cancel())


class Boto3CredentialsResolver(IdentityResolver):
    def __init__(self):
        self.session = boto3.Session()

    async def get_identity(self, **kwargs):
        try:
            credentials = self.session.get_credentials()
            logger.debug("Attempting to load AWS credentials")
            if not credentials:
                logger.error("Unable to load AWS credentials")
                raise SmithyException("Unable to load AWS credentials")

            creds = credentials.get_frozen_credentials()
            logger.debug(
                f"AWS credentials loaded successfully. Access key ID: {creds.access_key[:4]}***"
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
            raise SmithyException(f"Failed to load AWS credentials: {str(e)}")


class RealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        model: str = "amazon.nova-sonic-v1:0",
        voice: str = "matthew",
        temperature: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        region: str = "us-east-1",
    ):
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=False,
                user_transcription=False,
                auto_tool_reply_generation=False,
            )
        )
        self.voice = voice
        self.model = model
        self.region = region
        self._opts = _RealtimeOptions(
            model=model,
            voice=voice,
            temperature=temperature if is_given(temperature) else DEFAULT_TEMPERATURE,
            tool_choice=tool_choice or None,
        )
        self._sessions = weakref.WeakSet[RealtimeSession]()

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(self)
        sess._initialization_task = asyncio.get_event_loop().create_future()
        asyncio.create_task(sess._initialize_stream())
        self._sessions.add(sess)
        return sess

    # Q: why can this be a stub? shouldn't it proxy to the session's impl?
    async def aclose(self) -> None:
        pass


class RealtimeSession(
    llm.RealtimeSession[Literal["bedrock_server_event_received", "bedrock_client_event_queued"]]
):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model = realtime_model
        self._input_resampler: rtc.AudioResampler | None = None
        self._bstream = utils.audio.AudioByteStream(
            INPUT_SAMPLE_RATE, CHANNELS, samples_per_channel=CHUNK_SIZE
        )

        self.response_task = None
        self.stream_response = None
        self.is_active = False
        self._initialization_task = None
        self.bedrock_client = None
        self.audio_input_queue = utils.aio.Chan()
        self.audio_output_queue = utils.aio.Chan()
        self.output_queue = utils.aio.Chan()
        self.dummy_stream = utils.aio.Chan()

        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())

    @DeprecationWarning
    @classmethod
    async def create(cls, realtime_model: RealtimeModel) -> "RealtimeSession":
        session = cls(realtime_model)
        await session.initialize_stream()
        return session

    @utils.log_exceptions(logger=logger)
    def _initialize_client(self):
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self._realtime_model.region}.amazonaws.com",
            region=self._realtime_model.region,
            aws_credentials_identity_resolver=Boto3CredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()},
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)

    @utils.log_exceptions(logger=logger)
    async def _initialize_stream(self):
        try:
            logger.debug(
                f"Initializing Bedrock stream with model: {self._realtime_model._opts.model}"
            )
            if not self.bedrock_client:
                logger.debug("Creating Bedrock client")
                self._initialize_client()
            self.stream_response = await self.bedrock_client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(
                    model_id=self._realtime_model._opts.model
                )
            )
            self.is_active = True
            default_system_prompt = (
                "You are a friend. The user and you will engage in a spoken dialog exchanging the transcripts of a natural real-time conversation."
                "When reading order numbers, please read each digit individually, separated by pauses. For example, order #1234 should be read as 'order number one-two-three-four' rather than 'order number one thousand two hundred thirty-four'."
            )

            init_events = [
                seb.create_session_start_event(),
                seb.create_prompt_start_event(
                    prompt_name=self.prompt_name,
                    voice_id=self._realtime_model.voice,
                    sample_rate=OUTPUT_SAMPLE_RATE,
                ),
                seb.create_text_content_start_event(
                    prompt_name=self.prompt_name,
                    content_name=self.content_name,
                    role="SYSTEM",
                ),
                seb.create_text_input_event(
                    prompt_name=self.prompt_name,
                    content_name=self.content_name,
                    content=default_system_prompt,
                ),
                seb.create_content_end_event(
                    prompt_name=self.prompt_name,
                    content_name=self.content_name,
                ),
            ]

            for event in init_events:
                await self.send_raw_event(event)
                # Small delay between init events
                await asyncio.sleep(0.1)

            # Start listening for responses
            self.response_task = asyncio.create_task(
                self._process_responses(), name="RealtimeSession._process_responses"
            )

            # Start processing audio input
            asyncio.create_task(
                self._process_audio_input(), name="RealtimeSession._process_audio_input"
            )

            if not self._initialization_task.done():
                self._initialization_task.set_result(self)

            # Wait a bit to ensure everything is set up
            await asyncio.sleep(0.1)

            logger.debug("Stream initialized successfully")
        except Exception as e:
            self.is_active = False
            self._initialization_task.set_exception(e)
            logger.debug(f"Failed to initialize stream: {str(e)}")
            raise
        return self

    # can be used in places that need to explicitly wait for stream initialization
    @utils.log_exceptions(logger=logger)
    async def initialize_stream(self):
        if not self.bedrock_client:
            self._initialize_client()

        if self.is_active:
            return self

        if self._initialization_task is not None:
            return await self._initialization_task

        return await self._initialize_stream()

    @utils.log_exceptions(logger=logger)
    async def send_raw_event(self, event_json):
        if not self.stream_response or not self.is_active:
            logger.debug("Stream not initialized or closed")
            return

        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )

        try:
            await self.stream_response.input_stream.send(event)
            ev_json = json.loads(event_json)
            if "event" in ev_json and "audioOutput" in ev_json["event"]:
                del ev_json["event"]["audioOutput"]["content"]
            if "event" in ev_json and "audioInput" in ev_json["event"]:
                del ev_json["event"]["audioInput"]["content"]
            logger.debug(f"Sent event: {json.dumps(ev_json, indent=2)}")
        except Exception as e:
            logger.debug(f"Error sending event: {str(e)}")
            traceback.print_exc()

    def _create_text_content_start_event(self, prompt_name, content_name, role):
        return {
            "event": {
                "contentStart": {
                    "promptName": prompt_name,
                    "contentName": content_name,
                    "type": "TEXT",
                    "role": role,
                    "interactive": True,
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        }

    def _create_text_input_event(self, prompt_name, content_name, content):
        return {
            "event": {
                "textInput": {
                    "promptName": prompt_name,
                    "contentName": content_name,
                    "content": content,
                }
            }
        }

    def _create_tool_content_start_event(self, prompt_name, content_name, tool_use_id):
        return {
            "event": {
                "contentStart": {
                    "promptName": prompt_name,
                    "contentName": content_name,
                    "interactive": False,
                    "type": "TOOL",
                    "role": "TOOL",
                    "toolResultInputConfiguration": {
                        "toolUseId": tool_use_id,
                        "type": "TEXT",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    },
                }
            }
        }

    @utils.log_exceptions(logger=logger)
    async def _process_responses(self):
        """Process incoming responses from Bedrock."""
        try:
            await self.initialize_stream()
            while self.is_active:
                try:
                    output = await self.stream_response.await_output()
                    result = await output[1].receive()
                    if result.value and result.value.bytes_:
                        try:
                            response_data = result.value.bytes_.decode("utf-8")
                            json_data = json.loads(response_data)

                            # Handle different response types
                            if "event" in json_data:
                                if json_data["event"].get("contentStart") and json_data["event"].get("contentStart").get("type") == "AUDIO":
                                    logger.debug(
                                        f"Output audio start detected: {json.dumps(json_data, indent=2)}"
                                    )
                                if json_data["event"].get("contentEnd") and json_data["event"].get("contentEnd").get("type") == "AUDIO":
                                    logger.debug(
                                        f"Output audio end detected: {json.dumps(json_data, indent=2)}"
                                    )
                                    logger.debug("Will emit generation event")
                                    self._emit_generation_event()
                                    logger.debug("Emitted generation event")
                                elif "audioOutput" not in json_data["event"] and 'usageEvent' not in json_data['event']:
                                    logger.debug(
                                        f"Output event detected: {json.dumps(json_data, indent=2)}"
                                    )
                                elif "audioOutput" in json_data["event"]:
                                    audio_content = json_data["event"]["audioOutput"]["content"]
                                    audio_bytes = base64.b64decode(audio_content)
                                    samples = len(audio_bytes) // 2
                                    logger.debug(f"AUDIO OUTPUT LENGTH: {samples}")

                                    # must transform into rtc.AudioFrame b/c that's what
                                    # the AgentSession expects
                                    await self.audio_output_queue.send(
                                        rtc.AudioFrame(
                                            data=audio_bytes,
                                            sample_rate=OUTPUT_SAMPLE_RATE,
                                            num_channels=CHANNELS,
                                            samples_per_channel=samples,
                                        )
                                    )
                                    del json_data["event"]["audioOutput"]["content"]
                                    logger.debug(
                                        f"Output audio detected: {json.dumps(json_data, indent=2)}"
                                    )
    
                            # Put the response in the output queue for other components
                            await self.output_queue.send(json_data)
                        except json.JSONDecodeError:
                            await self.output_queue.send({"raw_data": response_data})
                except StopAsyncIteration:
                    # Stream has ended
                    break
                except Exception as e:
                    # Handle ValidationException properly
                    if "ValidationException" in str(e):
                        error_message = str(e)
                        logger.debug(f"Validation error: {error_message}")
                    else:
                        logger.debug(f"Error receiving response: {e}")
                    break

        except Exception as e:
            logger.debug(f"Response processing error: {e}")
        finally:
            self.is_active = False

    @utils.log_exceptions(logger=logger)
    def _emit_generation_event(self) -> None:
        logger.debug(f"Emitting generation event")
        message_generation = llm.MessageGeneration(
            message_id=str(uuid.uuid4()),
            text_stream=self.dummy_stream,
            audio_stream=self.audio_output_queue,
        )

        # temp async generator
        # required in order to fulfill contract that
        # llm.GenerationCreatedEvent.message_stream requires Chan[llm.MessageGeneration]
        async def message_stream_gen():
            yield message_generation

        generation_ev = llm.GenerationCreatedEvent(
            message_stream=message_stream_gen(),
            function_stream=self.dummy_stream,
            user_initiated=False,
        )
        self.emit("generation_created", generation_ev)
        logger.debug(f"Emitted generation event: {generation_ev}")

    def chat_ctx(self) -> ChatContext:
        return ChatContext()

    def tools(self) -> llm.ToolContext:
        return llm.ToolContext()

    async def update_instructions(self, instructions: str) -> None:
        pass

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        # can raise RealtimeError on Timeout
        pass

    async def update_tools(self, tools: list[llm.FunctionTool | llm.RawFunctionTool | Any]) -> None:
        pass

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        pass

    @utils.log_exceptions(logger=logger)
    async def send_audio_content_start_event(self):
        """Send a content start event to the Bedrock stream."""
        await self.send_raw_event(seb.create_audio_content_start_event(
            prompt_name=self.prompt_name,
            content_name=self.audio_content_name,
        ))

    @utils.log_exceptions(logger=logger)
    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                # input audio changed to a different sample rate
                self._input_resampler = None

        # set resampler if resampling is needed
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
    def add_audio_chunk(self, audio_bytes):
        """Add an audio chunk to the queue."""
        squared_sum = sum(sample**2 for sample in audio_bytes)
        if (squared_sum / len(audio_bytes)) ** 0.5 > 200:
            logger.debug(f"Enqueuing significant audio chunk")
        logger.debug(f"Enqueuing audio chunk: {len(audio_bytes)} bytes")
        self.audio_input_queue.send_nowait(
            {
                "audio_bytes": audio_bytes,
                "prompt_name": self.prompt_name,
                "content_name": self.audio_content_name,
            }
        )

    @utils.log_exceptions(logger=logger)
    def push_audio(self, frame: rtc.AudioFrame) -> None:
        logger.debug(f"Raw audio received: samples={len(frame.data)} rate={frame.sample_rate} channels={frame.num_channels}")
        for f in self._resample_audio(frame):
            logger.debug(f"Resampled audio: samples={len(frame.data)} rate={frame.sample_rate} channels={frame.num_channels}")
            data = f.data.tobytes()
            for nf in self._bstream.write(data):
                self.add_audio_chunk(nf.data)
                # logger.debug(f"Audio chunk added: size={len(nf.data)}")

    @utils.log_exceptions(logger=logger)
    async def _process_audio_input(self):
        """Process audio input from the queue and send to Bedrock."""
        await self.send_audio_content_start_event()
        logger.debug("Starting audio input processing loop")
        while self.is_active:
            try:
                # Get audio data from the queue
                logger.debug("Waiting for audio data from queue...")
                data = await self.audio_input_queue.recv()

                audio_bytes = data.get("audio_bytes")
                if not audio_bytes:
                    logger.debug("No audio bytes received")
                    continue

                # Base64 encode the audio data
                blob = base64.b64encode(audio_bytes)
                logger.debug(f"Sending audio data to Bedrock: size={len(audio_bytes)} bytes")
                audio_event = seb.create_audio_input_event(
                    prompt_name=self.prompt_name,
                    content_name=self.audio_content_name,
                    audio_content=blob.decode("utf-8"),
                )

                # Send the event
                await self.send_raw_event(audio_event)
                logger.debug("Audio event sent to Bedrock")

            except asyncio.CancelledError:
                logger.debug("Audio processing loop cancelled")
                break
            except Exception as e:
                logger.debug(f"Error processing audio: {e}")
                traceback.print_exc()

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        # can raise RealtimeError on Timeout
        pass

    # commit the input audio buffer to the server
    def commit_audio(self) -> None:
        pass

    # clear the input audio buffer to the server
    def clear_audio(self) -> None:
        pass

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    # cancel the current generation (do nothing if no generation is in progress)
    def interrupt(self) -> None:
        pass

    # message_id is the ID of the message to truncate (inside the ChatCtx)
    def truncate(self, *, message_id: str, audio_end_ms: int) -> None:
        pass

    @utils.log_exceptions(logger=logger)
    async def aclose(self) -> None:
        """Close the stream properly."""
        if not self.is_active:
            return

        self.is_active = False
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()

        await self.send_audio_content_end_event()
        await self.send_prompt_end_event()
        await self.send_session_end_event()

        if self.stream_response:
            await self.stream_response.input_stream.close()

    async def send_audio_content_end_event(self):
        """Send a content end event to the Bedrock stream."""
        if not self.is_active:
            logger.debug("Stream is not active")
            return

        await self.send_raw_event(seb.create_content_end_event(
            prompt_name=self.prompt_name,
            content_name=self.audio_content_name,
        ))
        logger.debug("Audio ended")

    async def send_prompt_end_event(self):
        """Close the stream and clean up resources."""
        if not self.is_active:
            logger.debug("Stream is not active")
            return

        await self.send_raw_event(seb.create_prompt_end_event(
            prompt_name=self.prompt_name,
        ))
        logger.debug("Prompt ended")

    async def send_session_end_event(self):
        """Send a session end event to the Bedrock stream."""
        if not self.is_active:
            logger.debug("Stream is not active")
            return

        await self.send_raw_event(seb.create_session_end_event())
        self.is_active = False
        logger.debug("Session ended")
