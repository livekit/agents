from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import time
import wave
import weakref
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from azure.ai.voicelive.aio import connect
from azure.ai.voicelive.models import (
    AudioInputTranscriptionOptions,
    AzureStandardVoice,
    FunctionTool,
    InputAudioFormat,
    Modality,
    OutputAudioFormat,
    RequestSession,
    ServerEventType,
    TurnDetection,
)
from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import DefaultAzureCredential
from livekit import rtc
from livekit.agents import APIConnectionError, APIError, llm, utils
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ..log import logger
from .utils import (
    DEFAULT_INPUT_AUDIO_FORMAT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODALITIES,
    DEFAULT_OUTPUT_AUDIO_FORMAT,
    DEFAULT_TEMPERATURE,
    livekit_item_to_azure_item,
    livekit_tool_to_azure_tool,
    to_audio_transcription,
    to_turn_detection,
)

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
BYTES_PER_SAMPLE = NUM_CHANNELS * 2  # 2 bytes per sample for PCM16
DEFAULT_VOICE = "en-US-AvaMultilingualNeural"  # Multilingual voice for multi-language support


@dataclass
class _RealtimeOptions:
    endpoint: str
    model: str
    voice: str | AzureStandardVoice
    input_audio_transcription: AudioInputTranscriptionOptions | None
    tool_choice: llm.ToolChoice | None
    turn_detection: TurnDetection | None
    input_audio_format: InputAudioFormat
    output_audio_format: OutputAudioFormat
    modalities: Sequence[Modality | str]
    temperature: float
    max_output_tokens: int
    api_key: str | None
    use_default_credential: bool
    conn_options: APIConnectOptions
    save_audio_per_turn: bool


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    modalities: asyncio.Future[list[Literal["text", "audio"]]]
    audio_transcript: str = ""


@dataclass
class _FunctionCallGeneration:
    """Tracks a function call as its arguments are streamed in."""

    item_id: str
    call_id: str
    name: str
    arguments: str = ""  # Accumulated via delta events


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    messages: dict[str, _MessageGeneration]
    function_calls: dict[str, _FunctionCallGeneration]  # Track function calls being streamed
    _done_fut: asyncio.Future[None]
    _created_timestamp: float
    _first_token_timestamp: float | None = None
    _audio_data: bytearray = field(default_factory=bytearray)  # Store audio data for saving to file
    _response_id: str = ""  # Response ID for file naming


class RealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str | None = None,
        voice: str = DEFAULT_VOICE,
        input_audio_transcription: NotGivenOr[AudioInputTranscriptionOptions | None] = NOT_GIVEN,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int] = NOT_GIVEN,
        api_key: str | None = None,
        use_default_credential: bool = False,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        save_audio_per_turn: bool = False,
    ) -> None:
        """
        Initialize Azure Voice Live Realtime model.

        Args:
            endpoint: Azure Voice Live endpoint URL (wss://...). If None, reads from AZURE_VOICELIVE_ENDPOINT.
            model: Model name. If None, reads from AZURE_VOICELIVE_MODEL (default: "gpt-realtime").
            voice: Voice for audio responses (default: "en-US-AvaMultilingualNeural").
            input_audio_transcription: Configuration for input audio transcription. If NOT_GIVEN,
                uses default config (whisper-1). Set to None to disable transcription.
                Use AudioInputTranscriptionOptions to configure model and language.
            modalities: List of modalities to enable (default: ["text", "audio"]).
            turn_detection: Turn detection configuration. Accepts ServerVad, AzureSemanticVad,
                AzureSemanticVadEn, or AzureSemanticVadMultilingual (default: ServerVad with threshold=0.5).
            tool_choice: Tool selection policy.
            temperature: Sampling temperature (default: 0.8).
            max_output_tokens: Maximum output tokens (default: 4096).
            api_key: Azure API key. If None, reads from AZURE_VOICELIVE_API_KEY.
            use_default_credential: Use DefaultAzureCredential for auth instead of API key.
            conn_options: Connection retry and timeout options.
            save_audio_per_turn: Save audio to file for each turn (default: False).

        Example:
            ```python
            from livekit.plugins.azure.realtime import RealtimeModel
            from azure.ai.voicelive.models import AudioInputTranscriptionOptions, ServerVad

            # English-only session (recommended for reliable language detection)
            model = RealtimeModel(
                endpoint=os.getenv("AZURE_VOICELIVE_ENDPOINT"),
                api_key=os.getenv("AZURE_VOICELIVE_API_KEY"),
                voice="en-US-AvaNeural",
                input_audio_transcription=AudioInputTranscriptionOptions(
                    model="whisper-1",
                    language="en-US",  # Constrains transcription to English
                ),
                turn_detection=ServerVad(threshold=0.5, silence_duration_ms=500),
            )

            # Multi-language session with auto-detection
            model = RealtimeModel(
                endpoint=os.getenv("AZURE_VOICELIVE_ENDPOINT"),
                api_key=os.getenv("AZURE_VOICELIVE_API_KEY"),
                voice="en-US-AvaMultilingualNeural",
                input_audio_transcription=AudioInputTranscriptionOptions(
                    model="whisper-1",
                    language="en,zh,ja",  # Allow English, Chinese, and Japanese
                ),
            )
            ```
        """
        modalities_list: Sequence[Modality | str] = (
            [Modality.TEXT if m == "text" else Modality.AUDIO for m in modalities]
            if is_given(modalities)
            else DEFAULT_MODALITIES
        )
        turn_detection_val = to_turn_detection(turn_detection)

        logger.info(f"[AZURE_INIT] model: {model}")
        logger.info(f"[AZURE_INIT] turn_detection parameter: {turn_detection}")
        logger.info(
            f"[AZURE_INIT] turn_detection_val after to_turn_detection: {turn_detection_val}"
        )
        logger.info(
            f"[AZURE_INIT] turn_detection capability will be: {turn_detection_val is not None}"
        )

        input_audio_transcription_val = to_audio_transcription(input_audio_transcription)

        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=turn_detection_val is not None,
                user_transcription=input_audio_transcription_val is not None,
                auto_tool_reply_generation=False,  # Tool responses handled via generate_reply
                audio_output=Modality.AUDIO in modalities_list,
                manual_function_calls=True,
            )
        )

        # Get endpoint from environment if not provided
        endpoint_val = endpoint or os.environ.get("AZURE_VOICELIVE_ENDPOINT")
        if not endpoint_val:
            raise ValueError(
                "Azure Voice Live endpoint must be provided via 'endpoint' parameter "
                "or AZURE_VOICELIVE_ENDPOINT environment variable"
            )

        # Get model from environment if not provided
        model_val = model or os.environ.get("AZURE_VOICELIVE_MODEL") or "gpt-realtime"

        # Get API key if not using default credential
        api_key_val = api_key
        if not use_default_credential:
            api_key_val = api_key or os.environ.get("AZURE_VOICELIVE_API_KEY")
            if not api_key_val:
                raise ValueError(
                    "Azure Voice Live API key must be provided via 'api_key' parameter "
                    "or AZURE_VOICELIVE_API_KEY environment variable, "
                    "or set use_default_credential=True"
                )

        tool_choice_val: llm.ToolChoice | None = (
            cast(llm.ToolChoice, tool_choice) if is_given(tool_choice) else None
        )
        self._opts = _RealtimeOptions(
            endpoint=endpoint_val,
            model=model_val,
            voice=voice,
            input_audio_transcription=input_audio_transcription_val,
            tool_choice=tool_choice_val,
            turn_detection=turn_detection_val,
            input_audio_format=DEFAULT_INPUT_AUDIO_FORMAT,
            output_audio_format=DEFAULT_OUTPUT_AUDIO_FORMAT,
            modalities=modalities_list,
            temperature=temperature if is_given(temperature) else DEFAULT_TEMPERATURE,
            max_output_tokens=max_output_tokens
            if is_given(max_output_tokens)
            else DEFAULT_MAX_OUTPUT_TOKENS,
            api_key=api_key_val,
            use_default_credential=use_default_credential,
            conn_options=conn_options,
            save_audio_per_turn=save_audio_per_turn,
        )

        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "azure-voicelive"

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        pass


class RealtimeSession(
    llm.RealtimeSession[Literal["azure_server_event_received", "azure_client_event_sent"]]
):
    """
    Azure Voice Live Realtime API session.

    Manages WebSocket connection to Azure Voice Live and handles:
    - Audio streaming (input/output)
    - Text generation
    - Function calling
    - Turn detection (VAD)
    - Session management

    Emits additional events:
    - azure_server_event_received: Raw server events from Azure
    - azure_client_event_sent: Raw client messages sent to Azure
    """

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._tools = llm.ToolContext.empty()
        self._instructions: str | None = None

        self._connection = None
        self._connection_ready = asyncio.Event()
        self._main_atask = asyncio.create_task(
            self._main_task(), name="AzureRealtimeSession._main_task"
        )

        self._current_generation: _ResponseGeneration | None = None
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()

        self._update_chat_ctx_lock = asyncio.Lock()
        self._update_fnc_ctx_lock = asyncio.Lock()

        # Audio buffering for input
        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )
        self._input_resampler: rtc.AudioResampler | None = None
        self._pushed_duration_s: float = 0

        # Metrics
        self._response_id: str | None = None
        self._session_id: str | None = None

        # Track user-initiated responses (from generate_reply calls)
        # Maps client_event_id -> Future that resolves when response.created arrives
        self._response_created_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._remote_chat_ctx.to_chat_ctx()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools

    @property
    def tools_ctx(self) -> llm.ToolContext:
        return self._tools

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        """Main task that manages the Azure Voice Live WebSocket connection."""
        num_retries: int = 0
        max_retries = self._realtime_model._opts.conn_options.max_retry

        while True:
            try:
                await self._run_connection()
                break
            except APIError as e:
                # Server disconnect (idle timeout) - always reconnect without counting retries
                if "Server closed connection" in str(e):
                    self._emit_error(e, recoverable=True)
                    # Brief delay before reconnecting
                    await asyncio.sleep(0.5)
                    continue

                if max_retries == 0 or not e.retryable:
                    self._emit_error(e, recoverable=False)
                    raise
                elif num_retries >= max_retries:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"Azure Voice Live connection failed after {num_retries} attempts"
                    ) from e
                else:
                    self._emit_error(e, recoverable=True)
                    retry_interval = self._realtime_model._opts.conn_options._interval_for_retry(
                        num_retries
                    )
                    logger.warning(
                        f"Azure Voice Live connection failed, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={"attempt": num_retries, "max_retries": max_retries},
                    )
                    await asyncio.sleep(retry_interval)
                num_retries += 1
            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

    async def _run_connection(self) -> None:
        """Establish connection and process events."""
        # Create credential
        credential: DefaultAzureCredential | AzureKeyCredential
        if self._realtime_model._opts.use_default_credential:
            credential = DefaultAzureCredential()
        else:
            api_key = self._realtime_model._opts.api_key
            assert api_key is not None, "API key must be set when not using default credential"
            credential = AzureKeyCredential(api_key)

        try:
            async with connect(
                endpoint=self._realtime_model._opts.endpoint,
                credential=credential,
                model=self._realtime_model._opts.model,
            ) as conn:
                self._connection = conn

                # Configure session
                await self._configure_session(conn)

                # Process events
                async for event in conn:
                    await self._handle_event(event)

                # Server closed connection - trigger reconnect
                print("\033[43m\033[30m [AZURE] Server disconnected, reconnecting... \033[0m")
                logger.warning("Event loop ended - connection closed by server, will reconnect")
                raise APIError("Server closed connection", retryable=True)

        except APIError:
            # Re-raise APIError to trigger retry logic in _main_task
            raise
        except Exception as e:
            logger.error(f"Azure Voice Live connection error: {e}", exc_info=True)
            raise APIConnectionError(f"Azure Voice Live connection error: {e}") from e
        finally:
            self._connection = None
            self._connection_ready.clear()

    async def _configure_session(self, conn: Any) -> None:
        """Configure the Azure Voice Live session with initial settings."""
        tools_list: list[FunctionTool] = []
        if self._tools:
            for tool in self._tools.flatten():
                tools_list.append(livekit_tool_to_azure_tool(tool))

        # Wrap voice name in AzureStandardVoice if it's an Azure voice name
        voice_config: str | AzureStandardVoice = self._realtime_model._opts.voice
        input_audio_transcription = self._realtime_model._opts.input_audio_transcription

        # Extract language from input_audio_transcription for voice locale
        language = input_audio_transcription.language if input_audio_transcription else None

        if isinstance(voice_config, str):
            # Check if it's an Azure voice name (contains hyphen like "en-US-AvaNeural")
            if "-" in voice_config and voice_config not in [
                "alloy",
                "ash",
                "ballad",
                "coral",
                "echo",
                "sage",
                "shimmer",
                "verse",
                "marin",
                "cedar",
            ]:
                # Create AzureStandardVoice with locale if language is specified
                voice_config = AzureStandardVoice(
                    name=voice_config,
                    locale=language if language else None,
                )

        session_config = RequestSession(
            modalities=list(self._realtime_model._opts.modalities),
            instructions=self._instructions or "You are a helpful assistant.",
            voice=voice_config,
            input_audio_format=self._realtime_model._opts.input_audio_format,
            output_audio_format=self._realtime_model._opts.output_audio_format,
            turn_detection=self._realtime_model._opts.turn_detection,
            input_audio_transcription=input_audio_transcription,
            tools=tools_list if tools_list else None,  # type: ignore[arg-type]
            temperature=self._realtime_model._opts.temperature,
            max_response_output_tokens=self._realtime_model._opts.max_output_tokens,
        )

        await conn.session.update(session=session_config)

    async def _handle_event(self, event: Any) -> None:
        """Handle events from Azure Voice Live."""
        self.emit("azure_server_event_received", event)

        event_type = event.type

        # Log all events for debugging (except high-frequency audio events)
        if event_type not in (
            ServerEventType.RESPONSE_AUDIO_DELTA,
            ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DELTA,
        ):
            logger.info(f"[EVENT] Received: {event_type}")

        if event_type == ServerEventType.SESSION_UPDATED:
            self._session_id = getattr(event.session, "id", None)
            logger.info(f"[SESSION] Azure session updated: {self._session_id}")

            # Log audio configuration for debugging
            session = event.session
            if hasattr(session, "output_audio_format"):
                logger.info(f"Output audio format: {session.output_audio_format}")
            if hasattr(session, "input_audio_format"):
                logger.info(f"Input audio format: {session.input_audio_format}")

            self._connection_ready.set()
            self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

        elif event_type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
            logger.info("[VAD] Speech started detected")
            self.emit("input_speech_started", llm.InputSpeechStartedEvent())
            if self._current_generation:
                await self._cancel_response()

        elif event_type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
            logger.info("[VAD] Speech stopped detected")
            self.emit(
                "input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=True)
            )

        elif event_type == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
            item_id = getattr(event, "item_id", None) or ""
            transcript = getattr(event, "transcript", "")
            self.emit(
                "input_audio_transcription_completed",
                llm.InputTranscriptionCompleted(
                    item_id=item_id,
                    transcript=transcript,
                    is_final=True,
                ),
            )

        elif event_type == ServerEventType.RESPONSE_CREATED:
            await self._handle_response_created(event)

        elif event_type == ServerEventType.RESPONSE_OUTPUT_ITEM_ADDED:
            await self._handle_output_item_added(event)

        elif event_type == ServerEventType.RESPONSE_CONTENT_PART_ADDED:
            await self._handle_content_part_added(event)

        elif event_type == ServerEventType.RESPONSE_AUDIO_DELTA:
            await self._handle_audio_delta(event)

        elif event_type == ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DELTA:
            await self._handle_text_delta(event)

        elif event_type == ServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA:
            await self._handle_function_call_arguments_delta(event)

        elif event_type == ServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
            await self._handle_function_call_arguments_done(event)

        elif event_type == ServerEventType.RESPONSE_DONE:
            await self._handle_response_done(event)

        elif event_type == ServerEventType.ERROR:
            error_msg = getattr(event, "error", {}).get("message", "Unknown error")
            # Suppress "no active response" errors - these are harmless race conditions
            # that occur when a cancellation request arrives after a response completed
            error_lower = error_msg.lower()
            if "no active response" in error_lower or "response_cancel_not_active" in error_lower:
                logger.debug(f"Azure Voice Live (suppressed): {error_msg}")
                return
            logger.error(f"Azure Voice Live error: {error_msg}")
            self._emit_error(APIError(error_msg), recoverable=True)

    def _close_generation(self, gen: _ResponseGeneration) -> None:
        """Close all channels and futures for a generation.

        This is used both when a new response starts (to clean up any existing generation)
        and when a response completes normally.
        """
        for msg_gen in gen.messages.values():
            if not msg_gen.modalities.done():
                msg_gen.modalities.set_result([])
            msg_gen.text_ch.close()
            msg_gen.audio_ch.close()

        gen.message_ch.close()
        gen.function_ch.close()

        if not gen._done_fut.done():
            gen._done_fut.set_result(None)

    async def _handle_response_created(self, event: Any) -> None:
        """Handle response.created event."""
        response_id = getattr(event.response, "id", None)
        self._response_id = response_id

        # Clean up any existing generation before starting a new one
        if self._current_generation is not None:
            logger.warning(
                f"New response {response_id} started while previous generation was still active, "
                "closing previous generation channels"
            )
            self._close_generation(self._current_generation)
            self._current_generation = None

        gen = _ResponseGeneration(
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            messages={},
            function_calls={},  # Track function calls being streamed
            _done_fut=asyncio.Future(),
            _created_timestamp=time.time(),
        )
        self._current_generation = gen

        # Create the generation event
        generation_ev = llm.GenerationCreatedEvent(
            message_stream=gen.message_ch,
            function_stream=gen.function_ch,
            user_initiated=False,  # Default to server-initiated
            response_id=response_id,
        )

        # Check if this response was triggered by generate_reply (user-initiated)
        # Note: Azure SDK doesn't currently support metadata in response.create
        # So we can't match by client_event_id like OpenAI does
        # For now, check if there's ANY pending generate_reply future
        # This is a simplified approach that works when generate_reply is called sequentially
        if self._response_created_futures:
            # Pop the oldest pending future (FIFO)
            # In practice, there should only be one at a time
            event_id, fut = next(iter(self._response_created_futures.items()))
            self._response_created_futures.pop(event_id)

            if not fut.done():
                generation_ev.user_initiated = True
                fut.set_result(generation_ev)
                logger.info(
                    f"[RESPONSE_CREATED] User-initiated response {response_id}, resolved future for event_id: {event_id}"
                )
            else:
                logger.warning(
                    f"[RESPONSE_CREATED] Future for event_id {event_id} was already done"
                )

        logger.info(
            f"[RESPONSE_CREATED] Emitting generation_created event for response {response_id}, user_initiated={generation_ev.user_initiated}"
        )
        self.emit("generation_created", generation_ev)

    async def _handle_output_item_added(self, event: Any) -> None:
        """Handle response.output_item.added event."""
        if not self._current_generation:
            return

        item = getattr(event, "item", None)
        if not item:
            return

        item_id = getattr(item, "id", utils.shortuuid("msg_"))
        item_type = getattr(item, "type", None)

        logger.info(f"[OUTPUT_ITEM_ADDED] item_type: {item_type}, item_id: {item_id}")
        if item_type == "function_call":
            # Log the raw item for debugging
            logger.info(f"[OUTPUT_ITEM_ADDED] Raw function_call item: {item}")

        if item_type == "message":
            msg_gen = _MessageGeneration(
                message_id=item_id,
                text_ch=utils.aio.Chan[str](),
                audio_ch=utils.aio.Chan[rtc.AudioFrame](maxsize=25),  # Buffer ~500ms of audio
                modalities=asyncio.Future(),
            )
            self._current_generation.messages[item_id] = msg_gen

            logger.info(f"Sending MessageGeneration to message_ch for item {item_id}")
            self._current_generation.message_ch.send_nowait(
                llm.MessageGeneration(
                    message_id=item_id,
                    text_stream=msg_gen.text_ch,
                    audio_stream=msg_gen.audio_ch,
                    modalities=msg_gen.modalities,
                )
            )
            logger.info(f"MessageGeneration sent successfully for item {item_id}")

        elif item_type == "function_call":
            call_id = getattr(item, "call_id", "")
            name = getattr(item, "name", "")

            # Don't emit yet - arguments will be streamed via delta events
            # Store the function call info and wait for arguments
            fnc_call_gen = _FunctionCallGeneration(
                item_id=item_id,
                call_id=call_id,
                name=name,
                arguments="",  # Will be accumulated
            )
            self._current_generation.function_calls[item_id] = fnc_call_gen

            logger.info(
                f"Function call started - name: {name}, call_id: {call_id}, waiting for arguments..."
            )

    async def _handle_content_part_added(self, event: Any) -> None:
        """Handle response.content_part.added event."""
        if not self._current_generation:
            return

        item_id = getattr(event, "item_id", None)
        part = getattr(event, "part", None)

        if not item_id or not part or item_id not in self._current_generation.messages:
            return

        msg_gen = self._current_generation.messages[item_id]
        part_type = getattr(part, "type", None)

        # Set modalities - use contextlib.suppress to avoid InvalidStateError if already set
        result_modalities: list[Literal["text", "audio"]]
        if part_type == "audio":
            result_modalities = ["audio", "text"]
        elif part_type == "text":
            result_modalities = ["text"]
        else:
            return

        logger.info(
            f"Setting modalities for item {item_id}: {result_modalities}, part_type: {part_type}, modalities_done: {msg_gen.modalities.done()}"
        )

        with contextlib.suppress(asyncio.InvalidStateError):
            msg_gen.modalities.set_result(result_modalities)
            logger.info(f"Modalities set successfully for item {item_id}")

    async def _handle_audio_delta(self, event: Any) -> None:
        """Handle response.audio.delta event."""
        if not self._current_generation:
            return  # Skip logging in hot path

        item_id = getattr(event, "item_id", None)
        delta = getattr(event, "delta", None)

        if not delta or not item_id:
            return

        # Cache message generation lookup
        msg_gen = self._current_generation.messages.get(item_id)
        if not msg_gen:
            return

        if self._current_generation._first_token_timestamp is None:
            self._current_generation._first_token_timestamp = time.time()

        try:
            # Log first frame only
            if not hasattr(self, "_logged_first_audio"):
                self._logged_first_audio = True
                logger.info(f"First audio delta: {len(delta)} bytes, PCM16 format")

            # Conditional audio saving - only when enabled
            if self._realtime_model._opts.save_audio_per_turn:
                self._current_generation._audio_data.extend(delta)

            # Create AudioFrame directly from raw bytes
            frame = rtc.AudioFrame(
                data=delta,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(delta) // BYTES_PER_SAMPLE,
            )

            msg_gen.audio_ch.send_nowait(frame)

        except Exception as e:
            logger.error(f"Failed to process audio delta: {e}")

    async def _handle_text_delta(self, event: Any) -> None:
        """Handle response.audio_transcript.delta or response.text.delta event."""
        if not self._current_generation:
            return

        item_id = getattr(event, "item_id", None)
        delta = getattr(event, "delta", None)

        if not delta or not item_id or item_id not in self._current_generation.messages:
            return

        if self._current_generation._first_token_timestamp is None:
            self._current_generation._first_token_timestamp = time.time()

        msg_gen = self._current_generation.messages[item_id]
        msg_gen.text_ch.send_nowait(delta)
        msg_gen.audio_transcript += delta

    async def _handle_function_call_arguments_delta(self, event: Any) -> None:
        """Handle response.function_call_arguments.delta event."""
        if not self._current_generation:
            logger.warning("Received function call arguments delta but no current generation")
            return

        item_id = getattr(event, "item_id", None)
        delta = getattr(event, "delta", None)

        if not item_id or item_id not in self._current_generation.function_calls:
            logger.warning(
                f"Skipping function call arguments delta: item_id={item_id}, "
                f"in_function_calls={item_id in self._current_generation.function_calls if item_id else False}"
            )
            return

        if self._current_generation._first_token_timestamp is None:
            self._current_generation._first_token_timestamp = time.time()

        # Accumulate the arguments
        fnc_call_gen = self._current_generation.function_calls[item_id]
        fnc_call_gen.arguments += delta or ""

        logger.debug(
            f"Function call arguments delta for {fnc_call_gen.name}: +{len(delta or '')} chars, "
            f"total: {len(fnc_call_gen.arguments)} chars"
        )

    async def _handle_function_call_arguments_done(self, event: Any) -> None:
        """Handle response.function_call_arguments.done event."""
        if not self._current_generation:
            logger.warning("Received function call arguments done but no current generation")
            return

        item_id = getattr(event, "item_id", None)

        if not item_id or item_id not in self._current_generation.function_calls:
            logger.warning(f"Skipping function call arguments done: item_id={item_id}")
            return

        # Get the complete function call
        fnc_call_gen = self._current_generation.function_calls[item_id]

        # Emit the complete FunctionCall
        function_call = llm.FunctionCall(
            id=fnc_call_gen.item_id,
            call_id=fnc_call_gen.call_id,
            name=fnc_call_gen.name,
            arguments=fnc_call_gen.arguments,
        )

        self._current_generation.function_ch.send_nowait(function_call)

        logger.info(
            f"Function call complete - name: {fnc_call_gen.name}, "
            f"call_id: {fnc_call_gen.call_id}, "
            f"arguments length: {len(fnc_call_gen.arguments)} chars"
        )
        logger.debug(f"Function call arguments: {fnc_call_gen.arguments}")

    async def _handle_response_done(self, event: Any) -> None:
        """Handle response.done event."""
        if not self._current_generation:
            return

        # Log response status and content info
        response = getattr(event, "response", None)
        if response:
            status = getattr(response, "status", "unknown")
            status_details = getattr(response, "status_details", None)
            output = getattr(response, "output", [])
            logger.info(
                f"[RESPONSE_DONE] status={status}, status_details={status_details}, output_items={len(output) if output else 0}"
            )

            # Log each output item
            for i, item in enumerate(output or []):
                item_type = getattr(item, "type", "unknown")
                item_status = getattr(item, "status", "unknown")
                content = getattr(item, "content", [])
                logger.info(
                    f"[RESPONSE_DONE] output[{i}]: type={item_type}, status={item_status}, content_parts={len(content) if content else 0}"
                )

                # Log content parts
                for j, part in enumerate(content or []):
                    part_type = getattr(part, "type", "unknown")
                    has_audio = hasattr(part, "audio") and part.audio
                    has_transcript = hasattr(part, "transcript") and part.transcript
                    logger.info(
                        f"[RESPONSE_DONE] content[{j}]: type={part_type}, has_audio={has_audio}, has_transcript={has_transcript}"
                    )

        audio_bytes_received = len(self._current_generation._audio_data)
        logger.info(f"[RESPONSE_DONE] Total audio bytes received: {audio_bytes_received}")

        # Save audio response to file if enabled
        if (
            self._realtime_model._opts.save_audio_per_turn
            and len(self._current_generation._audio_data) > 0
        ):
            try:
                os.makedirs("audio_debug", exist_ok=True)
                timestamp = int(time.time() * 1000)
                response_id = self._response_id or "unknown"
                filename = f"audio_debug/response_{response_id}_{timestamp}.wav"

                with wave.open(filename, "wb") as wav_file:
                    wav_file.setnchannels(NUM_CHANNELS)
                    wav_file.setsampwidth(2)  # 2 bytes for PCM16
                    wav_file.setframerate(SAMPLE_RATE)
                    wav_file.writeframes(bytes(self._current_generation._audio_data))

                logger.info(
                    f"Saved audio response to {filename} ({len(self._current_generation._audio_data)} bytes)"
                )
            except Exception as e:
                logger.error(f"Failed to save audio file: {e}")

        # Close all channels to signal end of response
        # The RESPONSE_DONE event from Azure signals that all content is complete
        logger.info(f"Closing channels for response {self._response_id}")
        self._close_generation(self._current_generation)

        # Emit metrics
        if self._response_id:
            ttft = -1.0
            if self._current_generation._first_token_timestamp:
                ttft = (
                    self._current_generation._first_token_timestamp
                    - self._current_generation._created_timestamp
                )

            duration = time.time() - self._current_generation._created_timestamp

            # Extract usage data from the response event
            usage = getattr(event.response, "usage", None)
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            input_audio_tokens = 0
            input_text_tokens = 0
            output_audio_tokens = 0
            output_text_tokens = 0

            if usage:
                input_tokens = getattr(usage, "input_tokens", 0)
                output_tokens = getattr(usage, "output_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)

                # Get detailed token counts
                input_token_details = getattr(usage, "input_token_details", None)
                if input_token_details:
                    input_audio_tokens = getattr(input_token_details, "audio_tokens", 0)
                    input_text_tokens = getattr(input_token_details, "text_tokens", 0)

                output_token_details = getattr(usage, "output_token_details", None)
                if output_token_details:
                    output_audio_tokens = getattr(output_token_details, "audio_tokens", 0)
                    output_text_tokens = getattr(output_token_details, "text_tokens", 0)

            # Calculate tokens per second
            tokens_per_second = output_tokens / duration if duration > 0 else 0

            self.emit(
                "metrics_collected",
                RealtimeModelMetrics(
                    timestamp=time.time(),
                    request_id=self._response_id,
                    ttft=ttft,
                    duration=duration,
                    cancelled=False,
                    label=self.realtime_model.label,
                    error=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    tokens_per_second=tokens_per_second,
                    input_token_details=RealtimeModelMetrics.InputTokenDetails(
                        audio_tokens=input_audio_tokens,
                        text_tokens=input_text_tokens,
                        image_tokens=0,
                        cached_tokens=0,
                        cached_tokens_details=None,
                    ),
                    output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                        text_tokens=output_text_tokens,
                        audio_tokens=output_audio_tokens,
                        image_tokens=0,
                    ),
                ),
            )

        self._current_generation = None
        self._response_id = None

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Push audio frame to Azure Voice Live."""
        if not self._connection:
            if not getattr(self, "_logged_no_connection", False):
                logger.warning("[PUSH_AUDIO] No connection, dropping audio")
                self._logged_no_connection = True
            return
        elif not self._connection_ready.is_set():
            if not getattr(self, "_logged_not_ready", False):
                logger.warning("[PUSH_AUDIO] Connection not ready, dropping audio")
                self._logged_not_ready = True
            return
        else:
            # Reset the logging flags when connection is good
            self._logged_no_connection = False
            self._logged_not_ready = False

        # Resample audio to target sample rate and channel count if needed
        for resampled_frame in self._resample_audio(frame):
            # Push resampled audio through byte stream for chunking
            for audio_frame in self._bstream.push(resampled_frame.data):
                audio_b64 = base64.b64encode(audio_frame.data).decode("utf-8")
                asyncio.create_task(self._push_audio_async(audio_b64))

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        """Resample audio to target sample rate and channel count if needed."""
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                # input audio changed to a different sample rate
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        if self._input_resampler:
            # TODO(long): flush the resampler when the input source is changed
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    async def _push_audio_async(self, audio_b64: str) -> None:
        """Async helper to push audio to connection."""
        # Double-check connection is still valid
        if not self._connection or not self._connection_ready.is_set():
            return

        try:
            await self._connection.input_audio_buffer.append(audio=audio_b64)
        except Exception as e:
            # Silently ignore errors from closed connection
            if "closing transport" not in str(e).lower():
                logger.error(f"Failed to push audio: {e}")

    def push_video(self, frame: rtc.VideoFrame) -> None:
        """Push video frame (not supported by Azure Voice Live)."""
        logger.warning("push_video() is not supported by Azure Voice Live")

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        """Update session options."""
        if is_given(tool_choice):
            self._realtime_model._opts.tool_choice = cast(llm.ToolChoice | None, tool_choice)

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """Generate a reply from the model.

        Returns a Future that resolves to GenerationCreatedEvent when the response.created
        event is received from Azure with matching client_event_id.
        """
        # Generate unique event ID to track this response
        event_id = utils.shortuuid("response_create_")
        fut = asyncio.Future[llm.GenerationCreatedEvent]()

        # Store the future so we can resolve it when response.created arrives
        self._response_created_futures[event_id] = fut

        # Send response.create request with event_id in metadata
        asyncio.create_task(self._run_generate_reply(event_id, instructions))

        # Set up timeout
        def _on_timeout() -> None:
            if fut and not fut.done():
                fut.set_exception(llm.RealtimeError("generate_reply timed out."))
                self._response_created_futures.pop(event_id, None)

        handle = asyncio.get_running_loop().call_later(5.0, _on_timeout)
        fut.add_done_callback(lambda _: handle.cancel())

        return fut

    async def _run_generate_reply(self, event_id: str, instructions: NotGivenOr[str]) -> None:
        """Helper to run generate_reply asynchronously."""
        try:
            # Wait for connection to be ready (with timeout)
            await asyncio.wait_for(self._connection_ready.wait(), timeout=10.0)

            if not self._connection:
                if fut := self._response_created_futures.pop(event_id, None):
                    fut.set_exception(APIError("No active connection"))
                return

            # TODO: Azure SDK doesn't support metadata in response.create yet
            # For now, just call response.create without metadata
            # When Azure SDK adds metadata support, add: metadata={"client_event_id": event_id}
            await self._connection.response.create(
                additional_instructions=instructions if is_given(instructions) else None,
            )

            logger.info(f"[GENERATE_REPLY] Sent response.create with event_id: {event_id}")
            # The generation_created event will be emitted in _handle_response_created
            # and will resolve the future if metadata matches
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for connection to be ready")
            if fut := self._response_created_futures.pop(event_id, None):
                fut.set_exception(APIError("Timeout waiting for connection"))
        except Exception as e:
            logger.error(f"Failed to generate reply: {e}")
            if fut := self._response_created_futures.pop(event_id, None):
                fut.set_exception(e)

    def interrupt(self) -> None:
        """Interrupt the current response."""
        asyncio.create_task(self._cancel_response())

    async def _cancel_response(self) -> None:
        """Cancel the current response."""
        if not self._connection or not self._current_generation:
            return

        try:
            await self._connection.response.cancel()
        except Exception as e:
            # Suppress "no active response" errors - these are harmless race conditions
            error_str = str(e).lower()
            if (
                "no active response" not in error_str
                and "response_cancel_not_active" not in error_str
            ):
                logger.error(f"Failed to cancel response: {e}")

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Truncate conversation history (not supported by Azure Voice Live)."""
        logger.warning("truncate() is not supported by Azure Voice Live")

    async def update_instructions(self, instructions: str) -> None:
        """Update system instructions."""
        self._instructions = instructions
        if self._connection:
            try:
                session_config = RequestSession(instructions=instructions)
                await self._connection.session.update(session=session_config)
            except Exception as e:
                logger.error(f"Failed to update instructions: {e}")

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """Update chat context by sending conversation items to Azure."""
        async with self._update_chat_ctx_lock:
            if not self._connection:
                logger.warning("No active connection, cannot update chat context")
                return

            remote_ctx = self._remote_chat_ctx.to_chat_ctx()
            diff_ops = llm.utils.compute_chat_ctx_diff(remote_ctx, chat_ctx)

            logger.info(
                f"[UPDATE_CHAT_CTX] Diff computed - to_create: {len(diff_ops.to_create)}, to_remove: {len(diff_ops.to_remove)}, to_update: {len(diff_ops.to_update)}"
            )

            # Send new items to Azure
            for previous_msg_id, msg_id in diff_ops.to_create:
                chat_item = chat_ctx.get_by_id(msg_id)
                if not chat_item:
                    logger.warning(f"Item {msg_id} not found in chat_ctx")
                    continue

                logger.info(f"[UPDATE_CHAT_CTX] Creating item type={chat_item.type}, id={msg_id}")

                # Convert to Azure format and send
                try:
                    azure_item = livekit_item_to_azure_item(chat_item)
                    await self._connection.conversation.item.create(item=azure_item)

                    # Update remote context to track what we've sent
                    self._remote_chat_ctx.insert(previous_msg_id, chat_item)
                    logger.info(f"[UPDATE_CHAT_CTX] Successfully created item {msg_id}")
                except Exception as e:
                    logger.error(f"Failed to create conversation item {msg_id}: {e}")

            # Note: We don't automatically trigger response.create() here
            # The framework will call generate_reply() when appropriate based on auto_tool_reply_generation setting

            # Note: We don't support deletion or updates for now
            # Azure manages the conversation history internally
            if diff_ops.to_remove:
                logger.debug(f"Ignoring {len(diff_ops.to_remove)} items to remove (not supported)")
            if diff_ops.to_update:
                logger.debug(f"Ignoring {len(diff_ops.to_update)} items to update (not supported)")

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        """Update available tools."""
        async with self._update_fnc_ctx_lock:
            self._tools = llm.ToolContext(tools)

            if self._connection:
                try:
                    tools_list = [livekit_tool_to_azure_tool(t) for t in tools]
                    session_config = RequestSession(tools=tools_list if tools_list else None)
                    await self._connection.session.update(session=session_config)
                except Exception as e:
                    logger.error(f"Failed to update tools: {e}")

    def commit_audio(self) -> None:
        """Commit the audio buffer."""
        if self._connection:
            asyncio.create_task(self._commit_audio_async())

    async def _commit_audio_async(self) -> None:
        """Async helper for commit_audio."""
        if self._connection:
            try:
                await self._connection.input_audio_buffer.commit()
            except Exception as e:
                logger.error(f"Failed to commit audio: {e}")

    def clear_audio(self) -> None:
        """Clear the audio buffer."""
        if self._connection:
            asyncio.create_task(self._clear_audio_async())

    async def _clear_audio_async(self) -> None:
        """Async helper for clear_audio."""
        if self._connection:
            try:
                await self._connection.input_audio_buffer.clear()
            except Exception as e:
                logger.error(f"Failed to clear audio: {e}")

    def commit_user_turn(self) -> None:
        logger.warning("commit_user_turn is not supported by Azure Realtime API.")

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        """Emit an error event."""
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self.realtime_model.label,
                error=error,
                recoverable=recoverable,
            ),
        )

    async def aclose(self) -> None:
        """Close the session."""
        self._main_atask.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_atask
