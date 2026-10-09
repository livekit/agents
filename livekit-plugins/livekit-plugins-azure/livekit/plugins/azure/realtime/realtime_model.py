from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import time
import weakref
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, cast

from azure.ai.voicelive.aio import VoiceLiveConnection, connect
from azure.ai.voicelive.models import (
    AudioInputTranscriptionOptions,
    AzureStandardVoice,
    ClientEvent,
    ClientEventConversationItemCreate,
    ClientEventConversationItemDelete,
    ClientEventInputAudioBufferAppend,
    ClientEventInputAudioBufferClear,
    ClientEventInputAudioBufferCommit,
    ClientEventResponseCancel,
    ClientEventResponseCreate,
    ClientEventSessionUpdate,
    InputAudioFormat,
    Modality,
    OutputAudioFormat,
    RequestSession,
    ResponseCreateParams,
    ServerEvent,
    ServerEventConversationItemCreated,
    ServerEventConversationItemDeleted,
    ServerEventConversationItemInputAudioTranscriptionCompleted,
    ServerEventError,
    ServerEventInputAudioBufferCommitted,
    ServerEventInputAudioBufferSpeechStarted,
    ServerEventInputAudioBufferSpeechStopped,
    ServerEventResponseAudioDelta,
    ServerEventResponseAudioTranscriptDelta,
    ServerEventResponseContentPartAdded,
    ServerEventResponseCreated,
    ServerEventResponseDone,
    ServerEventResponseFunctionCallArgumentsDelta,
    ServerEventResponseFunctionCallArgumentsDone,
    ServerEventResponseOutputItemAdded,
    ServerEventResponseTextDelta,
    ServerEventSessionUpdated,
    TurnDetection,
)
from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import DefaultAzureCredential
from livekit import rtc
from livekit.agents import APIConnectionError, APIError, llm, utils
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.generation import remove_instructions

from ..log import logger
from .utils import (
    DEFAULT_INPUT_AUDIO_FORMAT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODALITIES,
    DEFAULT_OUTPUT_AUDIO_FORMAT,
    DEFAULT_TEMPERATURE,
    AzureConversationItem,
    azure_item_to_livekit_item,
    livekit_item_to_azure_item,
    livekit_tools_to_azure_tools,
    to_audio_transcription,
    to_azure_response_tool_choice,
    to_azure_tool_choice,
    to_turn_detection,
)

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
BYTES_PER_SAMPLE = NUM_CHANNELS * 2  # 2 bytes per sample for PCM16
DEFAULT_VOICE = "en-US-AvaMultilingualNeural"  # Multilingual voice for multi-language support

_OPENAI_VOICES = frozenset(
    {"alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"}
)

# how long generate_reply waits for its response.created, including a pending (re)connection
_GENERATE_REPLY_TIMEOUT = 10.0
# how long update_chat_ctx waits for Azure to confirm the items it created
_UPDATE_CHAT_CTX_TIMEOUT = 10.0
# the input audio buffer can only be committed with at least 100ms of audio
_MIN_COMMIT_SAMPLES = SAMPLE_RATE // 10
# how many 100ms chunks of uncommitted audio (30s) a lost connection hands over to the next one
_MAX_RESENT_AUDIO_CHUNKS = 300


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
    response_id: str | None
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    created_timestamp: float
    messages: dict[str, _MessageGeneration] = field(default_factory=dict)
    function_calls: dict[str, _FunctionCallGeneration] = field(default_factory=dict)
    first_token_timestamp: float | None = None


@dataclass
class _DiscardedGeneration:
    """A response whose generate_reply timed out or was cancelled before it was created.

    The response is cancelled when it arrives, its events are dropped and its items deleted.
    """

    response_id: str | None


_ConfirmableEvent = (
    ClientEventConversationItemCreate
    | ClientEventResponseCreate
    | ClientEventInputAudioBufferCommit
)


@dataclass
class _ConnectionRequests:
    """What was sent on a single connection, the Azure conversation of which ends with it."""

    # item and response creations and audio commits Azure hasn't confirmed yet, by event id in
    # send order. They are sent again on the next connection when this one closes first
    unconfirmed: dict[str, _ConfirmableEvent] = field(default_factory=dict)
    # the event that created each item of this conversation, including the replayed ones
    item_events: dict[str, str] = field(default_factory=dict)
    # replayed conversation items by event id, so a rejected one leaves the mirror
    replay_events: dict[str, str] = field(default_factory=dict)
    # requests of a previous connection that couldn't be sent again before this one closed
    unsent: list[ClientEvent] = field(default_factory=list)
    # samples appended to the input audio buffer since it was last committed or cleared
    input_audio_samples: int = 0
    # the most recent audio appended since the input audio buffer was last committed or
    # cleared, the buffer of the next connection starts with it when this one closes first
    input_audio: deque[ClientEventInputAudioBufferAppend] = field(
        default_factory=lambda: deque(maxlen=_MAX_RESENT_AUDIO_CHUNKS)
    )
    # the audio of each unconfirmed commit by event id, sent again ahead of it
    committed_audio: dict[str, list[ClientEventInputAudioBufferAppend]] = field(
        default_factory=dict
    )
    # whether the conversation was replayed (or didn't need to be) and the requests sent again
    replayed: bool = False


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
    ) -> None:
        """
        Initialize Azure Voice Live Realtime model.

        Requires the optional dependencies: ``pip install 'livekit-plugins-azure[realtime]'``.

        Args:
            endpoint: Azure Voice Live endpoint URL. If None, reads from AZURE_VOICE_LIVE_ENDPOINT.
            model: Model name. If None, reads from AZURE_VOICE_LIVE_MODEL (default: "gpt-realtime").
            voice: Voice for audio responses (default: "en-US-AvaMultilingualNeural").
            input_audio_transcription: Configuration for input audio transcription. If NOT_GIVEN,
                uses azure-speech for the non-multimodal (text) models such as gpt-4.1 and for
                phi4-mm-realtime, which don't support whisper-1, and whisper-1 for the other
                models such as gpt-realtime. Set to None to disable transcription. Use
                AudioInputTranscriptionOptions to configure model and language, OpenAI
                transcription models such as whisper-1 take a single language, azure-speech up
                to 10 languages ("en-US,zh-CN").
            modalities: List of modalities to enable (default: ["text", "audio"]).
            turn_detection: Turn detection configuration. Accepts ServerVad, AzureSemanticVad,
                AzureSemanticVadEn, or AzureSemanticVadMultilingual (default: ServerVad with
                threshold=0.5).
            tool_choice: Tool selection policy (default: "auto").
            temperature: Sampling temperature (default: 0.8).
            max_output_tokens: Maximum output tokens (default: 4096).
            api_key: Azure API key. If None, reads from AZURE_VOICE_LIVE_API_KEY.
            use_default_credential: Use DefaultAzureCredential for auth instead of API key.
            conn_options: Connection retry and timeout options.

        Example:
            ```python
            from livekit.plugins.azure.realtime import RealtimeModel
            from azure.ai.voicelive.models import AudioInputTranscriptionOptions, ServerVad

            # English-only session (recommended for reliable language detection)
            model = RealtimeModel(
                endpoint=os.getenv("AZURE_VOICE_LIVE_ENDPOINT"),
                api_key=os.getenv("AZURE_VOICE_LIVE_API_KEY"),
                model="gpt-realtime",
                voice="en-US-AvaNeural",
                input_audio_transcription=AudioInputTranscriptionOptions(
                    model="whisper-1",
                    language="en-US",  # Constrains transcription to English
                ),
                turn_detection=ServerVad(threshold=0.5, silence_duration_ms=500),
            )

            # Multi-language session, azure-speech takes up to 10 languages, the first is primary
            model = RealtimeModel(
                endpoint=os.getenv("AZURE_VOICE_LIVE_ENDPOINT"),
                api_key=os.getenv("AZURE_VOICE_LIVE_API_KEY"),
                model="gpt-4.1",
                voice="en-US-AvaMultilingualNeural",
                input_audio_transcription=AudioInputTranscriptionOptions(
                    model="azure-speech",
                    language="en-US,zh-CN,ja-JP",  # Allow English, Chinese, and Japanese
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
        # Get model from environment if not provided, the transcription default depends on it
        model_val = model or os.environ.get("AZURE_VOICE_LIVE_MODEL") or "gpt-realtime"
        input_audio_transcription_val = to_audio_transcription(
            input_audio_transcription, model=model_val
        )

        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=turn_detection_val is not None,
                user_transcription=input_audio_transcription_val is not None,
                auto_tool_reply_generation=False,  # Tool responses handled via generate_reply
                audio_output=Modality.AUDIO in modalities_list,
                manual_function_calls=True,
                mutable_chat_context=False,
                mutable_instructions=True,
                mutable_tools=True,
                per_response_tool_choice=True,
            )
        )

        # Get endpoint from environment if not provided
        endpoint_val = endpoint or os.environ.get("AZURE_VOICE_LIVE_ENDPOINT")
        if not endpoint_val:
            raise ValueError(
                "Azure Voice Live endpoint must be provided via 'endpoint' parameter "
                "or AZURE_VOICE_LIVE_ENDPOINT environment variable"
            )

        # Get API key if not using default credential
        api_key_val = api_key
        if not use_default_credential:
            api_key_val = api_key or os.environ.get("AZURE_VOICE_LIVE_API_KEY")
            if not api_key_val:
                raise ValueError(
                    "Azure Voice Live API key must be provided via 'api_key' parameter "
                    "or AZURE_VOICE_LIVE_API_KEY environment variable, "
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
        )

        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "azure-voicelive"

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        # manual turn-taking is unsupported (can_disable_turn_detection=False)
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
    - azure_client_event_sent: Raw client events sent to Azure
    """

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        # per-session copy, so update_options only affects this session
        self._opts = replace(realtime_model._opts)
        self._tools = llm.ToolContext.empty()
        self._instructions: str | None = None

        # every client event goes through this channel, so they reach Azure in call order
        self._msg_ch = utils.aio.Chan[ClientEvent]()
        self._credential: DefaultAzureCredential | None = None
        # set once Azure confirmed the configuration and the replayed conversation of the
        # current connection, a connection that fails later was healthy until then
        self._session_confirmed = False
        self._connection_established = False
        # the requests of the current connection, and those a lost one left unconfirmed
        self._requests: _ConnectionRequests | None = None
        self._resend: list[ClientEvent] = []

        self._current_generation: _ResponseGeneration | _DiscardedGeneration | None = None
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()
        # items of discarded responses, deleted from the Azure conversation
        self._dropped_item_ids: set[str] = set()
        # requests whose errors are expected and only logged
        self._ignored_error_event_ids: set[str] = set()

        # generate_reply requests by client event id, resolved by the matching response.created
        self._response_created_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}
        # sent generate_reply requests that timed out or were cancelled before their response
        self._discarded_event_ids: set[str] = set()

        # conversation.item.create requests of update_chat_ctx, by item id and by event id
        self._item_create_futures: dict[str, asyncio.Future[None]] = {}
        self._item_create_events: dict[str, str] = {}
        self._pending_items: dict[str, llm.ChatItem] = {}

        self._update_chat_ctx_lock = asyncio.Lock()
        self._update_fnc_ctx_lock = asyncio.Lock()

        # Audio buffering for input, sent in 100ms chunks
        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )
        self._input_resampler: rtc.AudioResampler | None = None
        self._video_warned = False

        self._main_atask = asyncio.create_task(
            self._main_task(), name="AzureRealtimeSession._main_task"
        )

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._remote_chat_ctx.to_chat_ctx()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools

    @property
    def tools_ctx(self) -> llm.ToolContext:
        return self._tools

    def _send(self, event: ClientEvent) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        """Main task that manages the Azure Voice Live WebSocket connection."""
        num_retries: int = 0
        max_retries = self._opts.conn_options.max_retry
        reconnecting = False

        try:
            while not self._msg_ch.closed:
                self._session_confirmed = False
                self._connection_established = False
                try:
                    await self._run_connection(reconnecting=reconnecting)
                except APIError as e:
                    if self._connection_established:
                        # the connection was healthy before it dropped (e.g. an idle timeout)
                        num_retries = 0

                    if max_retries == 0 or not e.retryable:
                        self._emit_error(e, recoverable=False)
                        logger.error("Azure Voice Live connection failed", exc_info=e)
                        return

                    if num_retries >= max_retries:
                        self._emit_error(
                            APIConnectionError(
                                f"Azure Voice Live connection failed after {num_retries} attempts"
                            ),
                            recoverable=False,
                        )
                        logger.error(
                            f"Azure Voice Live connection failed after {num_retries} attempts",
                            exc_info=e,
                        )
                        return

                    self._emit_error(e, recoverable=True)
                    retry_interval = self._opts.conn_options._interval_for_retry(num_retries)
                    logger.warning(
                        f"Azure Voice Live connection failed, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={"attempt": num_retries, "max_retries": max_retries},
                    )
                    await asyncio.sleep(retry_interval)
                    num_retries += 1
                except Exception as e:
                    self._emit_error(e, recoverable=False)
                    logger.exception("Azure Voice Live session failed")
                    return

                reconnecting = True
        finally:
            # nothing can be sent anymore: settle everything still waiting on the connection
            self._msg_ch.close()
            self._close_current_generation()
            self._fail_pending_requests("Azure Voice Live session closed")
            if self._credential is not None:
                with contextlib.suppress(Exception):
                    await self._credential.close()
                self._credential = None

    def _get_credential(self) -> AzureKeyCredential | DefaultAzureCredential:
        if self._opts.use_default_credential:
            # one credential per session, so its token cache survives reconnections
            if self._credential is None:
                self._credential = DefaultAzureCredential()
            return self._credential

        assert self._opts.api_key is not None, "API key must be set when not using credentials"
        return AzureKeyCredential(self._opts.api_key)

    async def _run_connection(self, *, reconnecting: bool) -> None:
        """Connect, configure the session, and exchange events until the connection ends."""
        async with contextlib.AsyncExitStack() as stack:
            try:
                conn = await asyncio.wait_for(
                    stack.enter_async_context(
                        connect(
                            endpoint=self._opts.endpoint,
                            credential=self._get_credential(),
                            model=self._opts.model,
                        )
                    ),
                    self._opts.conn_options.timeout,
                )
            except asyncio.TimeoutError as e:
                raise APIConnectionError("Azure Voice Live connection timed out") from e
            except Exception as e:
                raise APIConnectionError("failed to connect to Azure Voice Live") from e

            requests = _ConnectionRequests()
            self._requests = requests
            # receive from the start, so the replay's confirmations never back up the socket
            tasks = [
                asyncio.create_task(self._recv_task(conn), name="AzureRealtimeSession._recv_task")
            ]
            try:
                try:
                    await self._send_direct(
                        conn, ClientEventSessionUpdate(session=self._create_session_config())
                    )
                    if reconnecting:
                        await self._replay_conversation(conn, requests)
                except Exception as e:
                    raise APIConnectionError(
                        "failed to configure the Azure Voice Live session"
                    ) from e

                if reconnecting:
                    self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

                requests.replayed = True
                self._check_established()
                tasks.append(
                    asyncio.create_task(
                        self._send_task(conn, requests), name="AzureRealtimeSession._send_task"
                    )
                )
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task.cancelled():
                        raise APIConnectionError("Azure Voice Live connection was cancelled")
                    task.result()
            finally:
                await utils.aio.cancel_and_wait(*tasks)
                self._on_connection_closed(requests)

    async def _send_task(self, conn: VoiceLiveConnection, requests: _ConnectionRequests) -> None:
        async for event in self._msg_ch:
            await self._send_event(conn, requests, event)

    async def _send_event(
        self, conn: VoiceLiveConnection, requests: _ConnectionRequests, event: ClientEvent
    ) -> None:
        if isinstance(event, ClientEventResponseCreate):
            event_id = event.event_id or ""
            reply_fut = self._response_created_futures.get(event_id)
            if reply_fut is None or reply_fut.done():
                # generate_reply timed out or was cancelled before it could be sent
                self._response_created_futures.pop(event_id, None)
                self._discarded_event_ids.discard(event_id)
                return
            requests.unconfirmed[event_id] = event
        elif isinstance(event, ClientEventConversationItemCreate) and event.item and event.item.id:
            item_id = event.item.id
            if item_id in requests.item_events:
                # created, or being created, in this conversation (e.g. replayed after a
                # reconnection): the waiter is settled when that creation is confirmed or rejected
                item_fut = self._item_create_futures.get(item_id)
                if self._remote_chat_ctx.get(item_id) is not None and item_fut is not None:
                    self._item_create_futures.pop(item_id, None)
                    if not item_fut.done():
                        item_fut.set_result(None)
                return
            event_id = event.event_id or ""
            requests.item_events[item_id] = event_id
            requests.unconfirmed[event_id] = event
        elif isinstance(event, ClientEventInputAudioBufferAppend):
            requests.input_audio_samples += _decoded_size(event.audio) // BYTES_PER_SAMPLE
            requests.input_audio.append(event)
        elif isinstance(event, ClientEventInputAudioBufferCommit):
            if not requests.input_audio_samples:
                # nothing to commit, e.g. the turn was committed already
                return
            if requests.input_audio_samples < _MIN_COMMIT_SAMPLES:
                # Azure rejects committing less than 100ms: the turn is dropped, rather than
                # left in the buffer to merge with the next one
                event = ClientEventInputAudioBufferClear()
            else:
                event_id = event.event_id or utils.shortuuid("commit_")
                event.event_id = event_id
                requests.unconfirmed[event_id] = event
                requests.committed_audio[event_id] = list(requests.input_audio)
            requests.input_audio_samples = 0
            requests.input_audio.clear()
        elif isinstance(event, ClientEventInputAudioBufferClear):
            requests.input_audio_samples = 0
            requests.input_audio.clear()

        try:
            await self._send_direct(conn, event)
        except Exception as e:
            raise APIConnectionError("failed to send an event to Azure Voice Live") from e

    async def _send_direct(self, conn: VoiceLiveConnection, event: ClientEvent) -> None:
        await conn.send(event)
        self.emit("azure_client_event_sent", event)

    async def _recv_task(self, conn: VoiceLiveConnection) -> None:
        try:
            async for event in conn:
                self._handle_server_event(event)
        except Exception as e:
            raise APIConnectionError("failed to receive events from Azure Voice Live") from e

        if not self._msg_ch.closed:
            raise APIConnectionError("Azure Voice Live connection closed unexpectedly")

    def _on_connection_closed(self, requests: _ConnectionRequests) -> None:
        """Settle what the closed connection can no longer complete."""
        self._requests = None
        self._close_current_generation(None if self._msg_ch.closed else "connection lost")
        # the deletions of dropped items went down with the conversation
        self._dropped_item_ids.clear()
        self._ignored_error_event_ids.clear()

        # a new connection starts a new, empty conversation: what Azure didn't confirm is sent
        # again once the conversation is replayed, the waiting callers are none the wiser
        unconfirmed: list[ClientEvent] = []
        for event_id, event in requests.unconfirmed.items():
            if isinstance(event, ClientEventResponseCreate):
                # a response of this conversation can't arrive anymore
                self._discarded_event_ids.discard(event_id)
                fut = self._response_created_futures.get(event_id)
                if fut is not None and not fut.done():
                    unconfirmed.append(event)
            elif isinstance(event, ClientEventInputAudioBufferCommit):
                # the turn went down with the input audio buffer, it's committed again
                unconfirmed.extend(requests.committed_audio.get(event_id, ()))
                unconfirmed.append(event)
            elif event.item and event.item.id and self._remote_chat_ctx.get(event.item.id) is None:
                unconfirmed.append(event)

        # so is the audio of the turn in progress, it doesn't belong to a conversation yet
        unconfirmed.extend(requests.input_audio)
        self._resend = unconfirmed + requests.unsent + self._resend

    def _fail_pending_requests(self, reason: str) -> None:
        self._resend = []
        error = llm.RealtimeError(reason)
        for response_fut in self._response_created_futures.values():
            if not response_fut.done():
                response_fut.set_exception(error)
        self._response_created_futures.clear()
        self._discarded_event_ids.clear()

        for item_fut in self._item_create_futures.values():
            if not item_fut.done():
                item_fut.set_exception(error)
        self._item_create_futures.clear()
        self._item_create_events.clear()
        self._pending_items.clear()

    async def _replay_conversation(
        self, conn: VoiceLiveConnection, requests: _ConnectionRequests
    ) -> None:
        """Re-create the conversation on a new connection, which starts empty.

        Replayed items are mirrored right away, the conversation.item.created events Azure answers
        them with are then no-ops. Requests the lost connection left unconfirmed follow them, with
        the audio its input audio buffer held.
        """
        chat_ctx = self.chat_ctx.copy(
            exclude_empty_message=True,
            exclude_handoff=True,
            exclude_config_update=True,
        )
        replayed = llm.remote_chat_context.RemoteChatContext()
        azure_items: list[AzureConversationItem] = []
        for item in chat_ctx.items:
            # a function call cut off before its arguments were complete
            if item.type == "function_call" and not item.arguments:
                continue
            try:
                azure_items.append(livekit_item_to_azure_item(item))
            except ValueError:
                continue
            replayed.insert(replayed.tail_id, item)

        # the mirror is the conversation every connection starts from, a failed replay is redone
        self._remote_chat_ctx = replayed
        for azure_item in azure_items:
            event_id = utils.shortuuid("replay_")
            if azure_item.id:
                requests.item_events[azure_item.id] = event_id
                requests.replay_events[event_id] = azure_item.id
            await self._send_direct(
                conn, ClientEventConversationItemCreate(event_id=event_id, item=azure_item)
            )

        resend, self._resend = self._resend, []
        for i, event in enumerate(resend):
            try:
                await self._send_event(conn, requests, event)
            except BaseException:
                # a request that fails to send is still unconfirmed, the rest is sent next time
                requests.unsent = resend[i + 1 :]
                raise

    def _voice_config(self) -> str | AzureStandardVoice:
        voice = self._opts.voice
        # Azure voice names contain a hyphen, e.g. "en-US-AvaNeural"
        if isinstance(voice, str) and "-" in voice and voice not in _OPENAI_VOICES:
            transcription = self._opts.input_audio_transcription
            language = transcription.language if transcription else None
            # a voice can only be pinned to a single locale, a language list stays auto-detected
            locale = language if language and "," not in language else None
            return AzureStandardVoice(name=voice, locale=locale)
        return voice

    def _create_session_config(self) -> RequestSession:
        """Configure the Azure Voice Live session with the current settings."""
        session = RequestSession(
            modalities=list(self._opts.modalities),
            instructions=(
                self._instructions
                if self._instructions is not None
                else "You are a helpful assistant."
            ),
            voice=self._voice_config(),
            input_audio_format=self._opts.input_audio_format,
            output_audio_format=self._opts.output_audio_format,
            turn_detection=self._opts.turn_detection,
            input_audio_transcription=self._opts.input_audio_transcription,
            tool_choice=to_azure_tool_choice(self._opts.tool_choice),
            temperature=self._opts.temperature,
            max_response_output_tokens=self._opts.max_output_tokens,
        )
        if tools := livekit_tools_to_azure_tools(self._tools.flatten()):
            session.tools = tools
        return session

    def _output_modalities(self) -> list[Literal["text", "audio"]]:
        return ["audio", "text"] if self._realtime_model.capabilities.audio_output else ["text"]

    def _handle_server_event(self, event: ServerEvent) -> None:
        """Handle events from Azure Voice Live."""
        self.emit("azure_server_event_received", event)

        try:
            if isinstance(event, ServerEventSessionUpdated):
                self._handle_session_updated(event)
            elif isinstance(event, ServerEventInputAudioBufferSpeechStarted):
                self._handle_input_speech_started(event)
            elif isinstance(event, ServerEventInputAudioBufferSpeechStopped):
                self._handle_input_speech_stopped(event)
            elif isinstance(event, ServerEventInputAudioBufferCommitted):
                self._handle_input_audio_buffer_committed(event)
            elif isinstance(event, ServerEventConversationItemCreated):
                self._handle_conversation_item_created(event)
            elif isinstance(event, ServerEventConversationItemDeleted):
                self._handle_conversation_item_deleted(event)
            elif isinstance(event, ServerEventConversationItemInputAudioTranscriptionCompleted):
                self._handle_input_audio_transcription_completed(event)
            elif isinstance(event, ServerEventResponseCreated):
                self._handle_response_created(event)
            elif isinstance(event, ServerEventResponseOutputItemAdded):
                self._handle_output_item_added(event)
            elif isinstance(event, ServerEventResponseContentPartAdded):
                self._handle_content_part_added(event)
            elif isinstance(event, ServerEventResponseAudioDelta):
                self._handle_audio_delta(event)
            elif isinstance(event, ServerEventResponseAudioTranscriptDelta):
                self._handle_text_delta(event, is_transcript=True)
            elif isinstance(event, ServerEventResponseTextDelta):
                # text-only responses (modalities=["text"])
                self._handle_text_delta(event, is_transcript=False)
            elif isinstance(event, ServerEventResponseFunctionCallArgumentsDelta):
                self._handle_function_call_arguments_delta(event)
            elif isinstance(event, ServerEventResponseFunctionCallArgumentsDone):
                self._handle_function_call_arguments_done(event)
            elif isinstance(event, ServerEventResponseDone):
                self._handle_response_done(event)
            elif isinstance(event, ServerEventError):
                self._handle_error(event)
        except Exception:
            logger.exception(
                "failed to handle Azure Voice Live event", extra={"event_type": event.type}
            )

    def _check_established(self) -> None:
        requests = self._requests
        if (
            requests is not None
            and requests.replayed
            and not requests.replay_events
            and self._session_confirmed
        ):
            self._connection_established = True

    def _handle_session_updated(self, event: ServerEventSessionUpdated) -> None:
        # a session.updated only confirms a configuration change, it never means a reconnection
        self._session_confirmed = True
        self._check_established()
        logger.debug(
            "Azure Voice Live session updated",
            extra={"session_id": getattr(event.session, "id", None)},
        )

    def _handle_input_speech_started(self, _: ServerEventInputAudioBufferSpeechStarted) -> None:
        # interrupting the reply is left to the agent, which knows if it can be interrupted
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _handle_input_speech_stopped(self, _: ServerEventInputAudioBufferSpeechStopped) -> None:
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(
                user_transcription_enabled=self._opts.input_audio_transcription is not None
            ),
        )

    def _handle_input_audio_buffer_committed(self, _: ServerEventInputAudioBufferCommitted) -> None:
        if (requests := self._requests) is None:
            return

        # confirmed: the turn doesn't need to be committed again after a reconnection. Commits
        # are confirmed in order, without their event id
        commit_id = next(iter(requests.committed_audio), None)
        if commit_id is not None:
            requests.committed_audio.pop(commit_id)
            requests.unconfirmed.pop(commit_id, None)
        else:
            # committed by the turn detection of Azure
            requests.input_audio.clear()

    def _handle_conversation_item_created(self, event: ServerEventConversationItemCreated) -> None:
        item = event.item
        if item is None or not item.id:
            return

        item_id = item.id
        created_by_client = False
        if (requests := self._requests) is not None:
            created_by_client = item_id in requests.item_events
            if requests.replay_events.pop(requests.item_events.get(item_id, ""), None):
                self._check_established()
            # confirmed: the item doesn't need to be sent again after a reconnection
            requests.unconfirmed.pop(requests.item_events.get(item_id, ""), None)

        # items of update_chat_ctx are mirrored as the caller built them
        lk_item: llm.ChatItem | None = self._pending_items.pop(item_id, None)
        if lk_item is None:
            try:
                lk_item = azure_item_to_livekit_item(item)
            except ValueError:
                logger.debug(
                    "ignoring an Azure Voice Live conversation item",
                    extra={"item_type": item.type},
                )

        if (
            not created_by_client
            and isinstance(self._current_generation, _DiscardedGeneration)
            and lk_item is not None
            and (
                lk_item.type == "function_call"
                or (lk_item.type == "message" and lk_item.role == "assistant")
            )
        ):
            # an output of the discarded response, created ahead of its output_item.added
            self._drop_item(item_id)
            return

        if (
            lk_item is not None
            and item_id not in self._dropped_item_ids
            and self._remote_chat_ctx.get(item_id) is None
        ):
            previous_item_id = event.previous_item_id
            if previous_item_id is None or self._remote_chat_ctx.get(previous_item_id) is None:
                # Azure appends an item it isn't told where to insert
                previous_item_id = self._remote_chat_ctx.tail_id

            self._remote_chat_ctx.insert(previous_item_id, lk_item)
            self.emit(
                "remote_item_added",
                llm.RemoteItemAddedEvent(previous_item_id=previous_item_id, item=lk_item),
            )

        if (fut := self._item_create_futures.pop(item_id, None)) and not fut.done():
            fut.set_result(None)

    def _handle_conversation_item_deleted(self, event: ServerEventConversationItemDeleted) -> None:
        self._dropped_item_ids.discard(event.item_id)
        if event.item_id and self._remote_chat_ctx.get(event.item_id) is not None:
            self._remote_chat_ctx.delete(event.item_id)

    def _handle_input_audio_transcription_completed(
        self, event: ServerEventConversationItemInputAudioTranscriptionCompleted
    ) -> None:
        item_id = event.item_id or ""
        transcript = event.transcript or ""

        remote_item = self._remote_chat_ctx.get(item_id)
        if (
            transcript
            and remote_item is not None
            and isinstance(remote_item.item, llm.ChatMessage)
            and transcript not in remote_item.item.content
        ):
            remote_item.item.content.append(transcript)

        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(item_id=item_id, transcript=transcript, is_final=True),
        )

    def _close_generation(self, generation: _ResponseGeneration) -> None:
        """Close all channels and futures of a generation, so its consumers never hang."""
        for message in generation.messages.values():
            if not message.modalities.done():
                message.modalities.set_result(self._output_modalities())
            message.text_ch.close()
            message.audio_ch.close()

        generation.message_ch.close()
        generation.function_ch.close()

    def _close_current_generation(self, reason: str | None = None) -> None:
        generation = self._current_generation
        self._current_generation = None
        if isinstance(generation, _ResponseGeneration):
            self._close_generation(generation)
            if reason:
                logger.warning(f"in-progress Azure Voice Live generation closed due to {reason}")

    def _generation_for(self, response_id: str | None) -> _ResponseGeneration | None:
        """The current generation, if the event of `response_id` belongs to it."""
        generation = self._current_generation
        if not isinstance(generation, _ResponseGeneration):
            return None
        if response_id and generation.response_id and response_id != generation.response_id:
            return None
        return generation

    def _drop_item(self, item_id: str) -> None:
        """Remove an item of a discarded response, nobody heard it, from the conversation."""
        if item_id in self._dropped_item_ids:
            # being deleted already, e.g. its conversation.item.created came first
            return
        self._dropped_item_ids.add(item_id)
        if self._remote_chat_ctx.get(item_id) is not None:
            self._remote_chat_ctx.delete(item_id)

        event_id = utils.shortuuid("drop_item_")
        # e.g. the cancelled response never kept it
        self._ignored_error_event_ids.add(event_id)
        self._send(ClientEventConversationItemDelete(item_id=item_id, event_id=event_id))

    def _handle_response_created(self, event: ServerEventResponseCreated) -> None:
        response = event.response
        response_id = response.id if response else None
        metadata = response.metadata if response else None
        client_event_id = metadata.get("client_event_id") if isinstance(metadata, dict) else None

        # never leave a previous generation open, e.g. when its response.done is still in flight
        self._close_current_generation()

        if client_event_id and self._requests is not None:
            # confirmed: the request doesn't need to be sent again after a reconnection
            self._requests.unconfirmed.pop(client_event_id, None)

        # generate_reply tags its request with metadata that Azure echoes back, so a response
        # created by server-side turn detection can't be mistaken for the requested one
        fut = self._response_created_futures.pop(client_event_id, None) if client_event_id else None
        if client_event_id and (
            client_event_id in self._discarded_event_ids or (fut is not None and fut.done())
        ):
            # its generate_reply timed out or was cancelled, nobody is waiting for it anymore
            self._discarded_event_ids.discard(client_event_id)
            self._send(ClientEventResponseCancel(response_id=response_id))
            self._current_generation = _DiscardedGeneration(response_id=response_id)
            logger.warning(
                "discarding an Azure Voice Live response created after its generate_reply "
                "timed out or was cancelled"
            )
            return

        generation = _ResponseGeneration(
            response_id=response_id,
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            created_timestamp=time.time(),
        )
        self._current_generation = generation

        generation_ev = llm.GenerationCreatedEvent(
            message_stream=generation.message_ch,
            function_stream=generation.function_ch,
            user_initiated=fut is not None,
            response_id=response_id,
        )
        if fut is not None:
            fut.set_result(generation_ev)

        self.emit("generation_created", generation_ev)

    def _handle_output_item_added(self, event: ServerEventResponseOutputItemAdded) -> None:
        item = event.item
        if item is None or not item.id:
            return

        discarded = self._current_generation
        if isinstance(discarded, _DiscardedGeneration) and (
            not event.response_id
            or not discarded.response_id
            or event.response_id == discarded.response_id
        ):
            self._drop_item(item.id)
            return

        generation = self._generation_for(event.response_id)
        if generation is None:
            return

        if item.type == "message":
            message = _MessageGeneration(
                message_id=item.id,
                text_ch=utils.aio.Chan[str](),
                # unbounded: the agent can start reading a reply only once it may be played
                audio_ch=utils.aio.Chan[rtc.AudioFrame](),
                modalities=asyncio.Future[list[Literal["text", "audio"]]](),
            )
            if not self._realtime_model.capabilities.audio_output:
                message.audio_ch.close()
                message.modalities.set_result(["text"])

            generation.messages[item.id] = message
            generation.message_ch.send_nowait(
                llm.MessageGeneration(
                    message_id=item.id,
                    text_stream=message.text_ch,
                    audio_stream=message.audio_ch,
                    modalities=message.modalities,
                )
            )

        elif item.type == "function_call":
            # emitted once its arguments are complete, see _handle_function_call_arguments_done
            generation.function_calls[item.id] = _FunctionCallGeneration(
                item_id=item.id,
                call_id=getattr(item, "call_id", None) or "",
                name=getattr(item, "name", None) or "",
            )

    def _handle_content_part_added(self, event: ServerEventResponseContentPartAdded) -> None:
        generation = self._generation_for(event.response_id)
        if generation is None or not (message := generation.messages.get(event.item_id)):
            return

        part_type = getattr(event.part, "type", None)
        modalities: list[Literal["text", "audio"]]
        if part_type == "audio":
            modalities = ["audio", "text"]
        elif part_type == "text":
            modalities = ["text"]
        else:
            return

        with contextlib.suppress(asyncio.InvalidStateError):
            message.modalities.set_result(modalities)

    def _handle_audio_delta(self, event: ServerEventResponseAudioDelta) -> None:
        generation = self._generation_for(event.response_id)
        if generation is None or not (message := generation.messages.get(event.item_id)):
            return

        # the SDK decodes the base64 payload, accept a raw string in case it doesn't
        data = event.delta
        if isinstance(data, str):
            data = base64.b64decode(data)
        if not data or message.audio_ch.closed:
            return

        if generation.first_token_timestamp is None:
            generation.first_token_timestamp = time.time()

        with contextlib.suppress(asyncio.InvalidStateError):
            message.modalities.set_result(["audio", "text"])

        message.audio_ch.send_nowait(
            rtc.AudioFrame(
                data=data,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(data) // BYTES_PER_SAMPLE,
            )
        )

    def _handle_text_delta(
        self,
        event: ServerEventResponseAudioTranscriptDelta | ServerEventResponseTextDelta,
        *,
        is_transcript: bool,
    ) -> None:
        generation = self._generation_for(event.response_id)
        if generation is None or not (message := generation.messages.get(event.item_id)):
            return

        delta = event.delta
        if not delta or message.text_ch.closed:
            return

        # transcripts trail their audio, while text deltas are model output on their own,
        # including the text fallback of an audio session
        if not is_transcript and generation.first_token_timestamp is None:
            generation.first_token_timestamp = time.time()

        message.text_ch.send_nowait(delta)
        message.audio_transcript += delta

    def _handle_function_call_arguments_delta(
        self, event: ServerEventResponseFunctionCallArgumentsDelta
    ) -> None:
        generation = self._generation_for(event.response_id)
        if generation is None or not (
            function_call := generation.function_calls.get(event.item_id)
        ):
            return

        if generation.first_token_timestamp is None:
            generation.first_token_timestamp = time.time()

        function_call.arguments += event.delta or ""

    def _handle_function_call_arguments_done(
        self, event: ServerEventResponseFunctionCallArgumentsDone
    ) -> None:
        generation = self._generation_for(event.response_id)
        if generation is None or not event.item_id:
            return

        pending = generation.function_calls.pop(event.item_id, None)
        call_id = event.call_id or (pending.call_id if pending else "")
        name = event.name or (pending.name if pending else "")
        arguments = (
            event.arguments
            if event.arguments is not None
            else (pending.arguments if pending else "")
        )
        if not call_id or not name:
            logger.warning(
                "ignoring an Azure Voice Live function call without a name or call id",
                extra={"item_id": event.item_id},
            )
            return

        remote_item = self._remote_chat_ctx.get(event.item_id)
        if remote_item is not None and isinstance(remote_item.item, llm.FunctionCall):
            remote_item.item.arguments = arguments

        generation.function_ch.send_nowait(
            llm.FunctionCall(id=event.item_id, call_id=call_id, name=name, arguments=arguments)
        )
        logger.debug(
            "Azure Voice Live function call completed",
            extra={"function": name, "call_id": call_id},
        )

    def _handle_response_done(self, event: ServerEventResponseDone) -> None:
        response = event.response
        response_id = response.id if response else None
        generation = self._current_generation

        if isinstance(generation, _DiscardedGeneration):
            if (
                not response_id
                or not generation.response_id
                or response_id == generation.response_id
            ):
                self._current_generation = None
            return

        if generation is None or (
            response_id and generation.response_id and response_id != generation.response_id
        ):
            return

        # mirror what the assistant said, so reconnections replay it
        for item_id, message in generation.messages.items():
            transcript = message.audio_transcript
            remote_item = self._remote_chat_ctx.get(item_id)
            if (
                transcript
                and remote_item is not None
                and isinstance(remote_item.item, llm.ChatMessage)
                and transcript not in remote_item.item.content
            ):
                remote_item.item.content.append(transcript)

        self._close_generation(generation)
        self._current_generation = None

        status = getattr(response, "status", None)
        self._emit_response_metrics(generation, event)

        if status == "failed":
            error = getattr(getattr(response, "status_details", None), "error", None)
            self._emit_error(
                APIError("Azure Voice Live response failed", body=error, retryable=True),
                recoverable=True,
            )
        elif status in ("cancelled", "incomplete"):
            logger.debug(
                f"Azure Voice Live response {status}",
                extra={
                    "response_id": response_id,
                    "reason": getattr(getattr(response, "status_details", None), "reason", None),
                },
            )

    def _emit_response_metrics(
        self, generation: _ResponseGeneration, event: ServerEventResponseDone
    ) -> None:
        response = event.response
        usage = getattr(response, "usage", None)
        input_details = getattr(usage, "input_token_details", None)
        output_details = getattr(usage, "output_token_details", None)

        created_timestamp = generation.created_timestamp
        ttft = (
            generation.first_token_timestamp - created_timestamp
            if generation.first_token_timestamp
            else -1
        )
        duration = time.time() - created_timestamp
        output_tokens = _token_count(usage, "output_tokens")

        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                timestamp=created_timestamp,
                request_id=getattr(response, "id", None) or "",
                ttft=ttft,
                duration=duration,
                cancelled=getattr(response, "status", None) == "cancelled",
                label=self._realtime_model.label,
                input_tokens=_token_count(usage, "input_tokens"),
                output_tokens=output_tokens,
                total_tokens=_token_count(usage, "total_tokens"),
                tokens_per_second=output_tokens / duration if duration > 0 else 0,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(
                    audio_tokens=_token_count(input_details, "audio_tokens"),
                    text_tokens=_token_count(input_details, "text_tokens"),
                    image_tokens=_token_count(input_details, "image_tokens"),
                    cached_tokens=_token_count(input_details, "cached_tokens"),
                    cached_tokens_details=None,
                ),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                    text_tokens=_token_count(output_details, "text_tokens"),
                    audio_tokens=_token_count(output_details, "audio_tokens"),
                    image_tokens=0,
                ),
                metadata=Metadata(
                    model_name=self._realtime_model.model,
                    model_provider=self._realtime_model.provider,
                ),
            ),
        )

    def _handle_error(self, event: ServerEventError) -> None:
        error = event.error
        message = getattr(error, "message", None) or "unknown error"
        code = getattr(error, "code", None)
        event_id = getattr(error, "event_id", None)

        if event_id:
            if (requests := self._requests) is not None:
                if (replayed_id := requests.replay_events.pop(event_id, None)) is not None:
                    requests.item_events.pop(replayed_id, None)
                    if self._remote_chat_ctx.get(replayed_id) is not None:
                        self._remote_chat_ctx.delete(replayed_id)
                    logger.warning(
                        f"Azure Voice Live rejected a replayed conversation item: {message}",
                        extra={"item_id": replayed_id, "code": code},
                    )
                    self._check_established()
                    return

                # a rejected request is settled, it's never sent again after a reconnection
                rejected = requests.unconfirmed.pop(event_id, None)
                requests.committed_audio.pop(event_id, None)
                if (
                    isinstance(rejected, ClientEventConversationItemCreate)
                    and rejected.item
                    and rejected.item.id
                ):
                    rejected_id = rejected.item.id
                    # the item doesn't exist, so it can be created again
                    requests.item_events.pop(rejected_id, None)
                    self._item_create_events.pop(event_id, None)
                    self._pending_items.pop(rejected_id, None)
                    # fails whoever waits for the item now, e.g. a retry joining this creation
                    item_fut = self._item_create_futures.pop(rejected_id, None)
                    if item_fut is not None and not item_fut.done():
                        item_fut.set_exception(llm.RealtimeError(message, code=code))
                    return

            if event_id in self._ignored_error_event_ids:
                self._ignored_error_event_ids.discard(event_id)
                logger.debug(f"Azure Voice Live (ignored): {message}")
                return

            # a rejected conversation.item.create fails its update_chat_ctx
            if (item_id := self._item_create_events.pop(event_id, None)) is not None:
                self._pending_items.pop(item_id, None)
                if (fut := self._item_create_futures.pop(item_id, None)) and not fut.done():
                    fut.set_exception(llm.RealtimeError(message, code=code))
                return

            # a rejected response.create never gets a response.created
            if (reply_fut := self._response_created_futures.pop(event_id, None)) and not (
                reply_fut.done()
            ):
                reply_fut.set_exception(llm.RealtimeError(message, code=code))

        # cancelling a response that already ended is a harmless race
        lowered = message.lower()
        if code == "response_cancel_not_active" or "no active response" in lowered:
            logger.debug(f"Azure Voice Live (suppressed): {message}")
            return

        logger.error(f"Azure Voice Live error: {message}", extra={"code": code})
        self._emit_error(
            APIError(f"Azure Voice Live error: {message}", body=error, retryable=True),
            recoverable=True,
        )

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Push audio frame to Azure Voice Live."""
        for resampled_frame in self._resample_audio(frame):
            for audio_frame in self._bstream.push(resampled_frame.data.tobytes()):
                self._send_audio(audio_frame)

    def _send_audio(self, frame: rtc.AudioFrame) -> None:
        self._send(
            ClientEventInputAudioBufferAppend(
                audio=base64.b64encode(frame.data).decode("utf-8"),
            )
        )

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

    def push_video(self, frame: rtc.VideoFrame) -> None:
        """Push video frame (not supported by Azure Voice Live)."""
        if not self._video_warned:
            self._video_warned = True
            logger.warning("push_video() is not supported by Azure Voice Live")

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        """Update session options and send session.update to Azure server."""
        if not is_given(tool_choice) or tool_choice == self._opts.tool_choice:
            return

        self._opts.tool_choice = tool_choice
        self._send(
            ClientEventSessionUpdate(
                session=RequestSession(tool_choice=to_azure_tool_choice(tool_choice))
            )
        )

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """Generate a reply from the model.

        Returns a Future that resolves to GenerationCreatedEvent when the response.created
        event carrying this request's client_event_id is received from Azure.
        """
        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        if self._msg_ch.closed:
            fut.set_exception(llm.RealtimeError("Azure Voice Live session is closed"))
            return fut

        event_id = utils.shortuuid("response_create_")
        params = ResponseCreateParams(metadata={"client_event_id": event_id})
        if is_given(tool_choice):
            params.tool_choice = to_azure_response_tool_choice(tool_choice)
        if is_given(tools):
            params.tools = livekit_tools_to_azure_tools(tools)

        self._response_created_futures[event_id] = fut
        self._send(
            ClientEventResponseCreate(
                event_id=event_id,
                response=params,
                additional_instructions=instructions if is_given(instructions) else None,
            )
        )

        def _on_timeout() -> None:
            if not fut.done():
                # a sent request is cancelled once Azure creates it, see _handle_response_created
                if self._response_created_futures.pop(event_id, None) is not None:
                    self._discarded_event_ids.add(event_id)
                fut.set_exception(llm.RealtimeError("generate_reply timed out."))

        # one deadline covers waiting for the connection, sending, and response.created
        handle = asyncio.get_running_loop().call_later(_GENERATE_REPLY_TIMEOUT, _on_timeout)

        def _on_done(_: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            handle.cancel()
            # still registered: the caller cancelled it before its response.created arrived
            if self._response_created_futures.pop(event_id, None) is not None:
                self._discarded_event_ids.add(event_id)

        fut.add_done_callback(_on_done)
        return fut

    def interrupt(self) -> None:
        """Interrupt the current response."""
        if isinstance(self._current_generation, _ResponseGeneration) or (
            self._response_created_futures
        ):
            self._send(ClientEventResponseCancel())

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
        self._send(ClientEventSessionUpdate(session=RequestSession(instructions=instructions)))

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """Create the new items of `chat_ctx` in the Azure conversation.

        Raises:
            llm.RealtimeError: if Azure rejects an item or doesn't confirm it in time.
        """
        async with self._update_chat_ctx_lock:
            # Filter out internal framework items that Azure doesn't understand
            chat_ctx = chat_ctx.copy(
                exclude_handoff=True,
                exclude_config_update=True,
            )
            # Remove instruction messages (already sent via the session configuration)
            remove_instructions(chat_ctx)

            remote_ctx = self._remote_chat_ctx.to_chat_ctx()
            known_ids = {item.id for item in remote_ctx.items}
            # empty messages are only placeholders, unless they already exist remotely
            chat_ctx = llm.ChatContext(
                [
                    item
                    for item in chat_ctx.items
                    if item.type != "message" or item.content or item.id in known_ids
                ]
            )
            diff_ops = llm.utils.compute_chat_ctx_diff(remote_ctx, chat_ctx)

            # Azure manages the conversation history internally, only additions are synced
            if diff_ops.to_remove or diff_ops.to_update:
                logger.debug(
                    "Azure Voice Live ignores removed and updated chat items",
                    extra={
                        "to_remove": len(diff_ops.to_remove),
                        "to_update": len(diff_ops.to_update),
                    },
                )

            if not diff_ops.to_create:
                return

            if self._msg_ch.closed:
                raise llm.RealtimeError("Azure Voice Live session is closed")

            futs: list[asyncio.Future[None]] = []
            event_ids: list[str] = []
            for previous_item_id, item_id in diff_ops.to_create:
                chat_item = chat_ctx.get_by_id(item_id)
                assert chat_item is not None
                try:
                    azure_item = livekit_item_to_azure_item(chat_item)
                except ValueError:
                    logger.warning(
                        "skipping a chat item Azure Voice Live doesn't support",
                        extra={"item_type": chat_item.type},
                    )
                    continue

                event_id = utils.shortuuid("chat_ctx_create_")
                fut = asyncio.Future[None]()
                self._item_create_futures[item_id] = fut
                self._item_create_events[event_id] = item_id
                self._pending_items[item_id] = chat_item
                futs.append(fut)
                event_ids.append(event_id)

                self._send(
                    ClientEventConversationItemCreate(
                        event_id=event_id,
                        # without an anchor Azure appends the item
                        previous_item_id=previous_item_id
                        if previous_item_id in known_ids
                        else None,
                        item=azure_item,
                    )
                )
                known_ids.add(item_id)

            if not futs:
                return

            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*futs, return_exceptions=True),
                    timeout=_UPDATE_CHAT_CTX_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise llm.RealtimeError("update_chat_ctx timed out.") from None
            finally:
                for event_id in event_ids:
                    if (pending_id := self._item_create_events.pop(event_id, None)) is not None:
                        self._item_create_futures.pop(pending_id, None)
                        self._pending_items.pop(pending_id, None)

            if errors := [r for r in results if isinstance(r, BaseException)]:
                raise llm.RealtimeError(
                    f"Azure Voice Live rejected {len(errors)} of {len(results)} chat items: "
                    f"{errors[0]}"
                )

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        """Update available tools."""
        async with self._update_fnc_ctx_lock:
            self._tools = llm.ToolContext(tools)
            # an empty list clears the tools of the session
            self._send(
                ClientEventSessionUpdate(
                    session=RequestSession(
                        tools=livekit_tools_to_azure_tools(self._tools.flatten())
                    )
                )
            )

    def commit_audio(self) -> None:
        """Commit the audio buffer.

        Azure can't commit less than 100ms of audio, such a short turn is cleared instead.
        """
        # the buffered tail belongs to this turn, send it before the commit
        for audio_frame in self._bstream.flush():
            self._send_audio(audio_frame)

        # checked against the buffer of the connection when it's sent
        self._send(ClientEventInputAudioBufferCommit(event_id=utils.shortuuid("commit_")))

    def clear_audio(self) -> None:
        """Clear the audio buffer."""
        self._bstream.clear()
        self._send(ClientEventInputAudioBufferClear())

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
        self._msg_ch.close()
        await utils.aio.cancel_and_wait(self._main_atask)

        # the main task may have been cancelled before it started, and couldn't clean up
        self._close_current_generation()
        self._fail_pending_requests("Azure Voice Live session closed")
        if self._credential is not None:
            with contextlib.suppress(Exception):
                await self._credential.close()
            self._credential = None


def _token_count(obj: object, name: str) -> int:
    value = getattr(obj, name, None)
    return value if isinstance(value, int) else 0


def _decoded_size(data: str) -> int:
    """Size of base64 encoded `data` once decoded."""
    return len(data) * 3 // 4 - (2 if data.endswith("==") else 1 if data.endswith("=") else 0)
