from __future__ import annotations
import asyncio
import contextlib
import copy
import json
import logging
import os
import time
import uuid
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    Annotated,
)
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import aiohttp
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from livekit import agents, rtc
from livekit.agents import (
    APIConnectionError,
    APIError,
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    llm,
    utils,
)
from livekit.agents.llm.tool_context import (
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
    function_tool,
)
from livekit.agents.types import (
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    DEFAULT_API_CONNECT_OPTIONS,
)
from livekit.agents.utils import is_given

DEFAULT_MODEL = "accounts/fireworks/models/qwen2p5-72b-instruct"
FIREWORKS_BASE_URL = "wss://audio-agent.link.fireworks.ai/v1"
AGENT_PATH = "/audio/agent"
STT_SAMPLE_RATE = 16000
STT_NUM_CHANNELS = 1
TTS_SAMPLE_RATE = 48000
TTS_NUM_CHANNELS = 1
TTS_FRAME_DURATION_MS = 20
BYTES_PER_SAMPLE = 2  # 16-bit PCM
TARGET_SAMPLES_PER_CHANNEL = TTS_SAMPLE_RATE * TTS_FRAME_DURATION_MS // 1000
TARGET_CHUNK_SIZE_BYTES = TARGET_SAMPLES_PER_CHANNEL * TTS_NUM_CHANNELS * BYTES_PER_SAMPLE

DEFAULT_TEMPERATURE = 0.7


class AudioProcessingConfig(BaseModel):
    high_pass_filter: Optional[Dict[str, bool]] = None
    noise_suppression: Optional[Dict[str, Union[bool, int, str]]] = None
    gain_controller2: Optional[Dict[str, Any]] = None
    echo_cancellation: Optional[Dict[str, bool]] = None


class AudioConfig(BaseModel):
    audio_processing_config: AudioProcessingConfig = Field(default_factory=AudioProcessingConfig)
    frame_ms: int = 10
    profiling_enabled: bool = False
    cng_dbfs: Optional[float] = None
    aec_config: Optional[Dict] = None


class IntentConfig(BaseModel):
    model_name: str = DEFAULT_MODEL
    temperature: float = 0.5
    min_delay: float = 0.75
    max_interrupt_delay: float = 3.0
    max_follow_up_delay: float = 10.0
    timeout: float = 30.0


class ToolConfig(BaseModel):
    system_prompt: Optional[str] = None
    tools: List[Dict] = Field(default_factory=list)


class AnswerConfig(BaseModel):
    model_name: str = DEFAULT_MODEL
    system_prompt: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    timeout: float = 30.0
    tool_config: Optional[ToolConfig] = None


class TtsVoice(str, Enum):
    FW_EN_US_ABIGAIL = "fw_en_US_abigail"
    FW_EN_US_ALEXANDER = "fw_en_US_alexander"
    FW_EN_US_AMELIA = "fw_en_US_amelia"
    FW_EN_US_AVA = "fw_en_US_ava"
    FW_EN_US_BENJAMIN = "fw_en_US_benjamin"
    FW_EN_US_CALEB = "fw_en_US_caleb"
    FW_EN_US_CHARLOTTE = "fw_en_US_charlotte"
    FW_EN_US_CHLOE = "fw_en_US_chloe"
    FW_EN_US_DANIEL = "fw_en_US_daniel"
    FW_EN_US_DAVE = "fw_en_US_dave"
    FW_EN_US_ELLA = "fw_en_US_ella"
    FW_EN_US_EMILY = "fw_en_US_emily"
    FW_EN_US_ETHAN = "fw_en_US_ethan"
    FW_EN_US_EVELYN = "fw_en_US_evelyn"
    FW_EN_US_GABRIEL = "fw_en_US_gabriel"
    FW_EN_US_GRACE = "fw_en_US_grace"
    FW_EN_US_HARPER = "fw_en_US_harper"
    FW_EN_US_HENRY = "fw_en_US_henry"
    FW_EN_US_ISABELLA = "fw_en_US_isabella"
    FW_EN_US_JACK = "fw_en_US_jack"
    FW_EN_US_JACKSON = "fw_en_US_jackson"
    FW_EN_US_JORDAN = "fw_en_US_jordan"
    FW_EN_US_LARRY = "fw_en_US_larry"
    FW_EN_US_LILY = "fw_en_US_lily"
    FW_EN_US_LOGAN = "fw_en_US_logan"
    FW_EN_US_LUCAS = "fw_en_US_lucas"
    FW_EN_US_MADISON = "fw_en_US_madison"
    FW_EN_US_MATTHEW = "fw_en_US_matthew"
    FW_EN_US_MIA = "fw_en_US_mia"
    FW_EN_US_MORGAN = "fw_en_US_morgan"
    FW_EN_US_OLIVIA = "fw_en_US_olivia"
    FW_EN_US_OWEN = "fw_en_US_owen"
    FW_EN_US_SAMUEL = "fw_en_US_samuel"
    FW_EN_US_SEBASTIAN = "fw_en_US_sebastian"
    FW_EN_US_SONIA = "fw_en_US_sonia"
    FW_EN_US_SOPHIA = "fw_en_US_sophia"
    FW_EN_US_TAYLOR = "fw_en_US_taylor"
    FW_EN_US_TRISH = "fw_en_US_trish"
    FW_EN_US_VICTORIA = "fw_en_US_victoria"
    FW_EN_US_WILLIAM = "fw_en_US_william"
    AF = "af"
    AF_ALLOY = "af_alloy"
    AF_AOEDE = "af_aoede"
    AF_BELLA = "af_bella"
    AF_HEART = "af_heart"
    AF_JESSICA = "af_jessica"
    AF_KORE = "af_kore"
    AF_NICOLE = "af_nicole"
    AF_NOVA = "af_nova"
    AF_RIVER = "af_river"
    AF_SARAH = "af_sarah"
    AF_SKY = "af_sky"
    AM_ADAM = "am_adam"
    AM_ECHO = "am_echo"
    AM_ERIC = "am_eric"
    AM_FENRIR = "am_fenrir"
    AM_LIAM = "am_liam"
    AM_MICHAEL = "am_michael"
    AM_ONYX = "am_onyx"
    AM_PUCK = "am_puck"
    AM_SANTA = "am_santa"
    BF_ALICE = "bf_alice"
    BF_EMMA = "bf_emma"
    BF_ISABELLA = "bf_isabella"
    BF_LILY = "bf_lily"
    BM_DANIEL = "bm_daniel"
    BM_FABLE = "bm_fable"
    BM_GEORGE = "bm_george"
    BM_LEWIS = "bm_lewis"
    EM_ALEX = "em_alex"
    EM_SANTA = "em_santa"
    EF_DORA = "ef_dora"
    FF_SIWIS = "ff_siwis"
    HF_ALPHA = "hf_alpha"
    HF_BETA = "hf_beta"
    HM_OMEGA = "hm_omega"
    HM_PSI = "hm_psi"
    IF_SARA = "if_sara"
    IM_NICOLA = "im_nicola"
    PF_DORA = "pf_dora"
    PM_ALEX = "pm_alex"
    PM_SANTA = "pm_santa"
    ZF_XIAOBEI = "zf_xiaobei"
    ZF_XIAONI = "zf_xiaoni"
    ZF_XIAOXIAO = "zf_xiaoxiao"
    ZF_XIAOYI = "zf_xiaoyi"
    ZM_YUNJIAN = "zm_yunjian"
    ZM_YUNXI = "zm_yunxi"
    ZM_YUNXIA = "zm_yunxia"
    ZM_YUNYANG = "zm_yunyang"


class TtsConfig(BaseModel):
    voice: TtsVoice = TtsVoice.FW_EN_US_ETHAN
    speed: float = 1.15


class FunctionCall(BaseModel):
    name: str
    arguments: Optional[str] = None


class ToolCallType(str, Enum):
    FUNCTION = "function"


class ToolCall(BaseModel):
    id: str
    type: ToolCallType = ToolCallType.FUNCTION
    function: FunctionCall


class OpeningBehavior(str, Enum):
    NO_GREETING = "no_greeting"
    GREETING = "greeting"


class AgentStateConfigure(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object: Literal["agent.state.configure"] = "agent.state.configure"
    config_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audio: AudioConfig = Field(default_factory=AudioConfig)
    intent: IntentConfig = Field(default_factory=IntentConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    opening_behavior: OpeningBehavior = OpeningBehavior.NO_GREETING
    agent_greeting: str = "Hello! How may I help you?"


class AgentStateConfigured(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object: Literal["agent.state.configured"]
    config_id: str


class AgentOutputWaiting(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object: Literal["agent.output.waiting"]


class AgentOutputGenerating(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object: Literal["agent.output.generating"]


class AgentOutputTranscript(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object: Literal["agent.output.transcript"]
    transcript: str


class AgentOutputDeltaMetadata(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object: Literal["agent.output.delta.metadata"]
    output_id: str
    delta_id: str
    text: str
    audio_sample_rate: int
    audio_duration: float
    chunk_index: int
    is_last_chunk: bool = False


class AgentOutputDone(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object: Literal["agent.output.done"]
    output_id: str
    text: str


class AgentOutputToolCall(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object: Literal["agent.output.tool_call"] = "agent.output.tool_call"
    tool_calls: List[ToolCall]


class Error(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    error: Error


class AgentOutputError(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object: Literal["agent.output.error"] = "agent.output.error"
    error: ErrorResponse


class TraceEvent(str, Enum):
    ASR_START = "asr_start"
    ASR_END = "asr_end"
    INTENT_START = "intent_start"
    INTENT_FIRST_TOKEN = "intent_first_token"
    INTENT_END = "intent_end"
    ANSWER_START = "answer_start"
    ANSWER_FIRST_TOKEN = "answer_first_token"
    ANSWER_END = "answer_end"
    TTS_START = "tts_start"
    TTS_FIRST_TOKEN = "tts_first_token"
    TTS_END = "tts_end"


class AgentOutputTrace(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object: Literal["agent.output.trace"] = "agent.output.trace"
    trace_id: str
    timeline: List[Tuple[float, TraceEvent]] = Field(default_factory=list)


class AudioProfile(BaseModel):
    sample_rate: int
    mic_pre: str
    mic_post: str
    speaker: str


class AgentOutputProfile(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object: Literal["agent.output.profile"] = "agent.output.profile"
    audio: Optional[AudioProfile] = None


AgentEgressTypes = Union[
    AgentStateConfigured,
    AgentOutputWaiting,
    AgentOutputGenerating,
    AgentOutputTranscript,
    AgentOutputDeltaMetadata,
    AgentOutputDone,
    AgentOutputToolCall,
    AgentOutputTrace,
    AgentOutputProfile,
    AgentOutputError,
]
AgentEgress = Annotated[AgentEgressTypes, Field(discriminator="object")]


class AgentInputToolResult(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object: Literal["agent.input.tool_result"] = "agent.input.tool_result"
    tool_results: Dict[str, Dict]


class AgentInputTrace(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    object: Literal["agent.input.trace"] = "agent.input.trace"
    trace_id: str


lk_fw_debug = int(os.getenv("LK_FIREWORKS_DEBUG", 0))
logger = logging.getLogger("fireworks_voice_agent")
if lk_fw_debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


@dataclass
class _RealtimeOptions:
    # General
    base_url: NotGivenOr[str] = NOT_GIVEN
    api_key: NotGivenOr[str] = NOT_GIVEN
    account_id: NotGivenOr[str] = NOT_GIVEN
    conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    max_session_duration: NotGivenOr[float] = NOT_GIVEN
    opening_behavior: NotGivenOr[str] = NOT_GIVEN
    agent_greeting: NotGivenOr[str] = NOT_GIVEN

    # STT
    high_pass_filter: NotGivenOr[bool] = NOT_GIVEN
    noise_suppression: NotGivenOr[bool] = NOT_GIVEN
    noise_suppression_level: NotGivenOr[int] = NOT_GIVEN
    gain_controller2: NotGivenOr[bool] = NOT_GIVEN
    echo_cancellation: NotGivenOr[bool] = NOT_GIVEN

    # Intent & Timing
    min_delay: NotGivenOr[float] = NOT_GIVEN
    max_interrupt_delay: NotGivenOr[float] = NOT_GIVEN
    max_follow_up_delay: NotGivenOr[float] = NOT_GIVEN

    # LLM
    model: NotGivenOr[str] = NOT_GIVEN
    max_tokens: NotGivenOr[int] = NOT_GIVEN
    temperature: NotGivenOr[float] = NOT_GIVEN
    top_p: NotGivenOr[float] = NOT_GIVEN
    tools: NotGivenOr[List[Dict]] = NOT_GIVEN
    tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN

    # TTS
    voice: NotGivenOr[str] = NOT_GIVEN
    speed: NotGivenOr[float] = NOT_GIVEN

    # Performance & Debug
    tracing: NotGivenOr[bool] = NOT_GIVEN


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    message: _MessageGeneration
    pending_audio_metadata: dict[str, AgentOutputDeltaMetadata]
    _done_fut: asyncio.Future[None]
    _created_timestamp: float
    full_text: str = ""
    _first_token_timestamp: float | None = None


DEFAULT_MAX_SESSION_DURATION = 20 * 60  # 20 minutes


class RealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        *,
        # General
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        account_id: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float] = NOT_GIVEN,
        opening_behavior: NotGivenOr[str] = NOT_GIVEN,
        agent_greeting: NotGivenOr[str] = NOT_GIVEN,
        # STT
        high_pass_filter: NotGivenOr[bool] = NOT_GIVEN,
        noise_suppression: NotGivenOr[bool] = NOT_GIVEN,
        noise_suppression_level: NotGivenOr[int] = NOT_GIVEN,
        gain_controller2: NotGivenOr[bool] = NOT_GIVEN,
        echo_cancellation: NotGivenOr[bool] = NOT_GIVEN,
        # Intent & Timing
        min_delay: NotGivenOr[float] = NOT_GIVEN,
        max_interrupt_delay: NotGivenOr[float] = NOT_GIVEN,
        max_follow_up_delay: NotGivenOr[float] = NOT_GIVEN,
        # LLM
        model: NotGivenOr[str] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        tools: NotGivenOr[List[Dict]] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        # TTS
        voice: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        # Performance & Debug
        tracing: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=True,
            )
        )
        api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError(
                "The api_key client option must be set either by passing api_key "
                "to the client or by setting the FIREWORKS_API_KEY environment variable"
            )
        if is_given(base_url):
            base_url_val = base_url
        else:
            base_url_val = FIREWORKS_BASE_URL

        self._opts = _RealtimeOptions(
            base_url=base_url_val,
            api_key=api_key,
            account_id=account_id,
            conn_options=conn_options,
            max_session_duration=(
                max_session_duration if is_given(max_session_duration) else DEFAULT_MAX_SESSION_DURATION
            ),
            high_pass_filter=high_pass_filter,
            noise_suppression=noise_suppression,
            noise_suppression_level=noise_suppression_level,
            gain_controller2=gain_controller2,
            echo_cancellation=echo_cancellation,
            min_delay=min_delay,
            max_interrupt_delay=max_interrupt_delay,
            max_follow_up_delay=max_follow_up_delay,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            voice=voice,
            speed=speed,
            opening_behavior=opening_behavior,
            agent_greeting=agent_greeting,
            tracing=tracing,
        )
        self._http_session = http_session
        self._sessions = weakref.WeakSet[RealtimeSession]()

    def update_options(
        self,
        *,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:

        if is_given(temperature) and self._opts.temperature != temperature:
            self._opts.temperature = temperature

        if is_given(max_tokens) and self._opts.max_tokens != max_tokens:
            self._opts.max_tokens = max_tokens

        if is_given(top_p) and self._opts.top_p != top_p:
            self._opts.top_p = top_p

        if is_given(voice) and self._opts.voice != voice:
            self._opts.voice = voice

        if is_given(speed) and self._opts.speed != speed:
            self._opts.speed = speed

        for sess in self._sessions:
            sess.update_options(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                voice=voice,
                speed=speed,
            )

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            self._http_session = utils.http_context.http_session()
        return self._http_session

    def session(self) -> "RealtimeSession":
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None: ...


def process_base_url(
    url: str,
) -> str:
    if url.startswith("http"):
        url = url.replace("http", "ws", 1)
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    new_query = urlencode(query_params, doseq=True)
    return urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, "", new_query, ""))


class RealtimeSession(llm.RealtimeSession[Literal["agent_interrupted",]]):
    """
    A session for the Fireworks Voice Agent Platform.
    This class is responsible for sending/receiving events to/from the platform.
    """

    _MODEL_MAP = {
        "agent.state.configured": AgentStateConfigured,
        "agent.output.waiting": AgentOutputWaiting,
        "agent.output.generating": AgentOutputGenerating,
        "agent.output.transcript": AgentOutputTranscript,
        "agent.output.delta.metadata": AgentOutputDeltaMetadata,
        "agent.output.done": AgentOutputDone,
        "agent.output.tool_call": AgentOutputToolCall,
        "agent.output.error": AgentOutputError,
        "agent.output.trace": AgentOutputTrace,
        "agent.output.profile": AgentOutputProfile,
    }

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._tools = llm.ToolContext.empty()
        self._msg_ch = utils.aio.Chan[Union[BaseModel, bytes]]()
        self._input_resampler: rtc.AudioResampler | None = None
        self._instructions: str | None = None
        self._main_atask = asyncio.create_task(self._main_task(), name="RealtimeSession._main_task")
        self.send_event(self._build_agent_state_configuration())

        self._current_generation: _ResponseGeneration | None = None
        self._chat_ctx = llm.ChatContext.empty()
        self._output_audio_buffer: bytearray = bytearray()
        self._agent_is_replying: bool = False
        self._last_user_transcript: str = ""

        self._generation_lock = asyncio.Lock()
        self._tools_lock = asyncio.Lock()
        self._chat_ctx_lock = asyncio.Lock()

    def send_event(self, event: Union[BaseModel, bytes]) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        num_retries: int = 0
        max_retries = self._realtime_model._opts.conn_options.max_retry

        async def _reconnect() -> None:
            logger.debug("reconnecting to Fireworks Agent API")
            try:
                config = self._build_agent_state_configuration()
                self.send_event(config)
            except Exception as e:
                raise APIConnectionError(
                    message="Failed to send configuration to Fireworks Agent API during session re-connection",
                ) from e

            logger.debug("reconnected to Fireworks Agent API")
            self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

        reconnecting = False
        while not self._msg_ch.closed:
            ws_conn = await self._create_ws_conn()

            try:
                if reconnecting:
                    await _reconnect()
                    num_retries = 0
                await self._run_ws(ws_conn)

            except APIError as e:
                if max_retries == 0 or not e.retryable:
                    self._emit_error(e, recoverable=False)
                    raise
                elif num_retries == max_retries:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"Fireworks Agent API connection failed after {num_retries} attempts",
                    ) from e
                else:
                    self._emit_error(e, recoverable=True)

                    retry_interval = self._realtime_model._opts.conn_options._interval_for_retry(num_retries)
                    logger.warning(
                        f"Fireworks Agent API connection failed, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={"attempt": num_retries, "max_retries": max_retries},
                    )
                    await asyncio.sleep(retry_interval)
                num_retries += 1

            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

            reconnecting = True

        self._close_current_generation()

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        headers = {"User-Agent": "LiveKit Agents"}
        if self._realtime_model._opts.api_key:
            headers["Authorization"] = f"Bearer {self._realtime_model._opts.api_key}"
        if self._realtime_model._opts.account_id:
            headers["x-fireworks-account-id"] = self._realtime_model._opts.account_id

        url = process_base_url(self._realtime_model._opts.base_url).rstrip("/") + AGENT_PATH
        if lk_fw_debug:
            logger.debug(f"connecting to Fireworks Agent API: {url}")
        return await self._realtime_model._ensure_http_session().ws_connect(
            url=url,
            headers=headers,
            timeout=aiohttp.ClientWSTimeout(ws_close=self._realtime_model._opts.conn_options.timeout),
        )

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        closing = False

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal closing
            async for msg in self._msg_ch:
                try:
                    if isinstance(msg, BaseModel):
                        json_str = msg.model_dump_json(by_alias=True, exclude_none=True)
                        await ws_conn.send_str(json_str)
                        if lk_fw_debug:
                            logger.debug(f">>> {json_str}")
                    elif isinstance(msg, bytes):
                        await ws_conn.send_bytes(msg)
                        if lk_fw_debug:
                            logger.debug(f">>> sent {len(msg)} bytes of audio data")
                except Exception:
                    break
            closing = True
            await ws_conn.close()

        @utils.log_exceptions(logger=logger)
        async def _recv_task() -> None:
            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:
                        return
                    raise APIConnectionError(message="Fireworks S2S connection closed unexpectedly")

                if msg.type == aiohttp.WSMsgType.TEXT:
                    event_data = json.loads(msg.data)
                    if lk_fw_debug:
                        logger.debug(f"<<< {event_data}")
                    try:
                        obj = event_data.get("object")
                        if obj in self._MODEL_MAP:
                            event = self._MODEL_MAP[obj].model_validate(event_data)
                            await self._handle_fireworks_event(event)
                        else:
                            logger.info(
                                f"Received unknown event object type: {obj}",
                                extra={"event": event_data},
                            )
                    except ValidationError as e:
                        logger.error(
                            "Failed to validate or handle incoming event",
                            exc_info=e,
                            extra={"event": event_data},
                        )
                    except Exception:
                        logger.exception("failed to handle event", extra={"event": event_data})
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await self._handle_fireworks_binary(msg.data)

        tasks = [
            asyncio.create_task(_recv_task(), name="_recv_task"),
            asyncio.create_task(_send_task(), name="_send_task"),
        ]
        wait_reconnect_task: asyncio.Task | None = None
        if self._realtime_model._opts.max_session_duration is not None:
            wait_reconnect_task = asyncio.create_task(
                asyncio.sleep(self._realtime_model._opts.max_session_duration),
                name="_timeout_task",
            )
            tasks.append(wait_reconnect_task)
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # propagate exceptions from completed tasks
            for task in done:
                if task != wait_reconnect_task:
                    task.result()

            if wait_reconnect_task and wait_reconnect_task in done and self._current_generation:
                # wait for the current generation to complete before reconnecting
                await self._current_generation._done_fut

        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws_conn.close()

    def _build_agent_state_configuration(self) -> AgentStateConfigure:
        opts = self._realtime_model._opts

        # Build dictionaries for each configuration object
        apc_kwargs: Dict[str, Any] = {}
        if is_given(opts.high_pass_filter):
            apc_kwargs["high_pass_filter"] = {"enabled": opts.high_pass_filter}
        if is_given(opts.noise_suppression):
            level = opts.noise_suppression_level if is_given(opts.noise_suppression_level) else 2
            apc_kwargs["noise_suppression"] = {"enabled": opts.noise_suppression, "level": level}
        if is_given(opts.gain_controller2):
            if opts.gain_controller2:
                apc_kwargs["gain_controller2"] = {
                    "enabled": True,
                    "adaptive_digital": {
                        "enabled": True,
                        "headroom_db": 6,
                        "max_gain_db": 45,
                        "max_gain_change_db_per_second": 9,
                    },
                }
            else:
                apc_kwargs["gain_controller2"] = {"enabled": False}
        if is_given(opts.echo_cancellation):
            apc_kwargs["echo_cancellation"] = {"enabled": opts.echo_cancellation}

        audio_kwargs: Dict[str, Any] = {"audio_processing_config": apc_kwargs}

        intent_kwargs: Dict[str, Any] = {}
        if is_given(opts.model):
            intent_kwargs["model_name"] = opts.model
        if is_given(opts.min_delay):
            intent_kwargs["min_delay"] = opts.min_delay
        if is_given(opts.max_interrupt_delay):
            intent_kwargs["max_interrupt_delay"] = opts.max_interrupt_delay
        if is_given(opts.max_follow_up_delay):
            intent_kwargs["max_follow_up_delay"] = opts.max_follow_up_delay

        answer_kwargs: Dict[str, Any] = {}
        if is_given(opts.model):
            answer_kwargs["model_name"] = opts.model
        if self._instructions:
            answer_kwargs["system_prompt"] = self._instructions
        if is_given(opts.max_tokens):
            answer_kwargs["max_tokens"] = opts.max_tokens
        if is_given(opts.temperature):
            answer_kwargs["temperature"] = opts.temperature
        if is_given(opts.top_p):
            answer_kwargs["top_p"] = opts.top_p

        fw_tools = []
        for tool in self._tools.function_tools.values():
            if is_function_tool(tool):
                fw_tools.append(llm.utils.build_legacy_openai_schema(tool))
            elif is_raw_function_tool(tool):
                fw_tools.append(get_raw_function_info(tool).raw_schema)
        if fw_tools:
            tool_config_kwargs = {"tools": fw_tools}
            if self._instructions:
                tool_config_kwargs["system_prompt"] = self._instructions
            answer_kwargs["tool_config"] = tool_config_kwargs

        tts_kwargs: Dict[str, Any] = {}
        if is_given(opts.voice):
            try:
                tts_kwargs["voice"] = TtsVoice(opts.voice)
            except ValueError:
                logger.warning(f"Invalid TtsVoice value: {opts.voice}, using default.")
        if is_given(opts.speed):
            tts_kwargs["speed"] = opts.speed

        config_kwargs: Dict[str, Any] = {}
        if is_given(opts.opening_behavior):
            try:
                config_kwargs["opening_behavior"] = OpeningBehavior(opts.opening_behavior)
            except ValueError:
                logger.warning(f"Invalid OpeningBehavior value: {opts.opening_behavior}, using default.")
        if is_given(opts.agent_greeting):
            config_kwargs["agent_greeting"] = opts.agent_greeting

        return AgentStateConfigure(
            audio=AudioConfig.model_validate(audio_kwargs),
            intent=IntentConfig.model_validate(intent_kwargs),
            answer=AnswerConfig.model_validate(answer_kwargs),
            tts=TtsConfig.model_validate(tts_kwargs),
            **config_kwargs,
        )

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    def update_options(
        self,
        *,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Update the agent's configuration mid-session."""
        if is_given(temperature):
            self._realtime_model._opts.temperature = temperature
        if is_given(max_tokens):
            self._realtime_model._opts.max_tokens = max_tokens
        if is_given(top_p):
            self._realtime_model._opts.top_p = top_p
        if is_given(voice):
            self._realtime_model._opts.voice = voice
        if is_given(speed):
            self._realtime_model._opts.speed = speed
        self.send_event(self._build_agent_state_configuration())

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """Update the chat context for this session."""
        async with self._chat_ctx_lock:
            diff_ops = llm.utils.compute_chat_ctx_diff(self._chat_ctx, chat_ctx)

            if diff_ops.to_remove:
                logger.warning("Fireworks does not support removing messages from context")

            append_ctx = llm.ChatContext.empty()
            for _, item_id in diff_ops.to_create:
                item = chat_ctx.get_by_id(item_id)
                if item:
                    append_ctx.items.append(item)

            if tool_results := self._get_tool_results(append_ctx):
                self.send_event(tool_results)

            self._chat_ctx = chat_ctx.copy()
            logger.debug(f"Updated chat context with {len(chat_ctx.items)} items")

    def _get_tool_results(
        self,
        chat_ctx: llm.ChatContext,
    ) -> AgentInputToolResult | None:
        """
        Check for function call outputs in the chat context and create an AgentInputToolResult.
        """
        tool_results_dict: Dict[str, Dict] = {}
        for msg in chat_ctx.items:
            if msg.type == "function_call_output" and msg.call_id:
                tool_results_dict[msg.call_id] = {"output": msg.output}

        if not tool_results_dict:
            return None

        return AgentInputToolResult(tool_results=tool_results_dict)

    async def update_tools(self, tools: list[llm.FunctionTool | llm.RawFunctionTool]) -> None:
        async with self._tools_lock:
            self._tools = llm.ToolContext(tools)
            config = self._build_agent_state_configuration()
            logger.info("Sending updated tool configuration to Fireworks.")
            self.send_event(config)

    async def update_instructions(self, instructions: str) -> None:
        if instructions != self._instructions:
            self._instructions = instructions
            config = self._build_agent_state_configuration()
            logger.info("Sending updated instructions to Fireworks.")
            self.send_event(config)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        for f in self._resample_audio(frame):
            data = f.data.tobytes()
            self.send_event(data)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def clear_audio(self) -> None:
        pass

    def commit_audio(self) -> None:
        pass

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        logger.debug("generate_reply called, but it is a no-op for Fireworks Agent.")
        fut = asyncio.Future()
        fut.set_result(
            llm.GenerationCreatedEvent(
                message_stream=utils.aio.Chan(),
                function_stream=utils.aio.Chan(),
                user_initiated=False,
            )
        )
        return fut

    def interrupt(self) -> None:
        self._close_current_generation()

    def truncate(self, *, message_id: str, audio_end_ms: int, audio_transcript: NotGivenOr[str] = NOT_GIVEN) -> None:
        logger.debug(f"Truncate is not supported by Fireworks Agent.")
        pass

    async def aclose(self) -> None:
        self._msg_ch.close()
        if self._main_atask:
            await self._main_atask

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != STT_SAMPLE_RATE or frame.num_channels != STT_NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=STT_SAMPLE_RATE,
                num_channels=STT_NUM_CHANNELS,
            )

        if self._input_resampler:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def _emit_generation_event(self) -> None:
        if self._current_generation is None:
            logger.warning("Tried to emit a generation event, but no generation is in progress.")
            return

        generation_ev = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
        )

        self.emit("generation_created", generation_ev)

    def _emit_input_speech_started(self) -> None:
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _emit_input_speech_stopped(self) -> None:
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(user_transcription_enabled=True),
        )

    def _emit_function_call(self, tool_call: ToolCall) -> None:
        """Emits a function call to the function_ch."""
        if self._current_generation is None:
            logger.warning("Tried to emit a function call, but no generation is in progress.")
            return

        self._current_generation.function_ch.send_nowait(
            llm.FunctionCall(
                call_id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments or "",
            )
        )

    def emit_input_audio_transcription_completed(self, transcript: str, is_final: bool) -> None:
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=str(uuid.uuid4()),
                transcript=transcript,
                is_final=is_final,
            ),
        )

    def _handle_new_generation(self) -> None:
        """
        Cleans up any existing generation and initializes a new one.
        This is called when the agent is about to start speaking.
        """
        if self._current_generation:
            logger.info("A new generation is starting while another is in progress. Closing the previous one.")
            self._close_current_generation()

        self._agent_is_replying = True

        # Create the message generation object that will be streamed to the user
        item_generation = _MessageGeneration(
            message_id=str(uuid.uuid4()),
            text_ch=utils.aio.Chan(),
            audio_ch=utils.aio.Chan(),
        )

        # Create the main response generation object that holds all state for this turn
        self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan(),
            function_ch=utils.aio.Chan(),
            message=item_generation,
            _created_timestamp=time.time(),
            _done_fut=asyncio.Future(),
            pending_audio_metadata={},
        )

        # Immediately send the new message generation object into the message channel
        self._current_generation.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=item_generation.message_id,
                text_stream=item_generation.text_ch,
                audio_stream=item_generation.audio_ch,
            )
        )

        # Immediately emit the generation_created event with the streams
        self._emit_generation_event()

    def _handle_delta_metadata(self, event: AgentOutputDeltaMetadata) -> None:
        """Handles receiving a metadata chunk for a part of the agent's response."""
        if not self._current_generation:
            logger.info("Received delta metadata without an active generation, ignoring.")
            return

        # This is the first token of the response, record the timestamp for TTFT
        if self._current_generation._first_token_timestamp is None:
            self._current_generation._first_token_timestamp = time.time()

        logger.debug(f"Handling AgentOutputDeltaMetadata for output_id: {event.output_id}, delta_id: {event.delta_id}")

        unique_key = f"{event.delta_id}"
        self._current_generation.pending_audio_metadata[unique_key] = event

        if event.chunk_index == 0:
            item_gen = self._current_generation.message
            item_gen.text_ch.send_nowait(event.text)
            self._current_generation.full_text += event.text

    async def _handle_fireworks_event(self, event: AgentEgress):
        logger.info(f"Received Fireworks event: {type(event).__name__}")
        async with self._generation_lock:
            if isinstance(event, AgentOutputTranscript):
                if lk_fw_debug:
                    logger.debug(
                        f"Handling AgentOutputTranscript: '{event.transcript}', is_agent_replying: {self._agent_is_replying}"
                    )
                if self._agent_is_replying:
                    if lk_fw_debug:
                        logger.debug("User interrupted agent, calling interrupt.")
                    self._emit_input_speech_started()
                    if lk_fw_debug:
                        logger.debug(f"is_agent_replying: {self._agent_is_replying}")

                self._last_user_transcript = event.transcript
                if lk_fw_debug:
                    logger.debug(f"Handling AgentOutputTranscript: '{event.transcript}'")

                self.emit_input_audio_transcription_completed(event.transcript, is_final=False)
                if lk_fw_debug:
                    logger.debug(f"I send a temporary complete event")

            elif isinstance(event, AgentOutputWaiting):
                if self._current_generation:
                    logger.debug("Ignoring 'waiting' event while a generation is in progress.")
                    return

            elif isinstance(event, AgentOutputGenerating):
                if self._last_user_transcript:
                    self._emit_input_speech_stopped()
                    self.emit_input_audio_transcription_completed(self._last_user_transcript, is_final=True)
                    self._chat_ctx.add_message(role="user", content=self._last_user_transcript)
                    if lk_fw_debug:
                        logger.debug("I send a final complete event")
                    self._last_user_transcript = ""

                self._handle_new_generation()
                if lk_fw_debug:
                    logger.debug(f"Generation created, is_agent_replying: {self._agent_is_replying}")

            elif isinstance(event, AgentOutputDeltaMetadata):
                self._handle_delta_metadata(event)

            elif isinstance(event, AgentOutputDone):
                logger.debug(f"Handling AgentOutputDone for output_id: {event.output_id}")
                if not self._current_generation:
                    logger.info("Could not find generation. Ignoring Done event.")
                    return

                logger.debug("Output done, closing entire generation.")
                self._close_current_generation()
                self._agent_is_replying = False

            elif isinstance(event, AgentOutputToolCall):
                logger.info(f"Received tool call request from Fireworks, executing directly: {event.tool_calls}")
                for call in event.tool_calls:
                    self._chat_ctx.insert(
                        llm.FunctionCall(
                            call_id=call.id,
                            name=call.function.name,
                            arguments=call.function.arguments or "",
                        )
                    )
                asyncio.create_task(self._execute_tool_calls(event.tool_calls))

            elif isinstance(event, AgentOutputError):
                self._emit_error(
                    APIError(message=event.error.error.message, body=event.error.model_dump()),
                    recoverable=False,
                )
            elif isinstance(event, AgentOutputTrace):
                self.emit("trace_updated", event)
            elif isinstance(event, AgentOutputProfile):
                self.emit("audio_profile_received", event)

    async def _execute_tool_calls(self, tool_calls: List[ToolCall]):
        tool_results_dict: Dict[str, Dict] = {}
        for call in tool_calls:
            tool = self._tools.function_tools.get(call.function.name)
            if not tool:
                logger.warning(f"Tool '{call.function.name}' not found.")
                output_str = f"Error: Tool '{call.function.name}' not found."
                self._chat_ctx.insert(
                    llm.FunctionCallOutput(
                        name=call.function.name,
                        call_id=call.id,
                        output=output_str,
                        is_error=True,
                    )
                )
                tool_results_dict[call.id] = {"output": output_str}
                continue

            try:
                kwargs = {}
                if call.function.arguments:
                    kwargs = json.loads(call.function.arguments)

                logger.info(f"Executing tool '{call.function.name}' with args: {kwargs}")
                output = await tool(**kwargs)
                output_str = json.dumps(output) if not isinstance(output, str) else output
                self._chat_ctx.insert(
                    llm.FunctionCallOutput(
                        name=call.function.name,
                        call_id=call.id,
                        output=output_str,
                        is_error=False,
                    )
                )
                tool_results_dict[call.id] = {"output": output}
                logger.info(f"Tool '{call.function.name}' executed with result: {output}")

            except Exception as e:
                logger.exception(f"Error executing tool '{call.function.name}'")
                output_str = f"Error: {str(e)}"
                self._chat_ctx.insert(
                    llm.FunctionCallOutput(
                        name=call.function.name,
                        call_id=call.id,
                        output=output_str,
                        is_error=True,
                    )
                )
                tool_results_dict[call.id] = {"output": output_str}

        if not tool_results_dict:
            return

        result_event = AgentInputToolResult(tool_results=tool_results_dict)
        logger.info(f"Sending tool results to Fireworks: {result_event}")
        self.send_event(result_event)

    async def _handle_fireworks_binary(self, data: bytes):
        async with self._generation_lock:
            if not self._current_generation:
                logger.info("Received audio data without an active generation, ignoring.")
                return

            if lk_fw_debug:
                logger.debug(f"Handling binary audio data of size {len(data)} bytes.")

            if not self._current_generation.pending_audio_metadata:
                logger.warning("Received audio data but no pending metadata found.")
                return

            first_delta_id = next(iter(self._current_generation.pending_audio_metadata))
            last_meta = self._current_generation.pending_audio_metadata.pop(first_delta_id)
            logger.debug(f"Matched audio data with metadata for delta_id: {first_delta_id}")

            item_gen = self._current_generation.message

            self._output_audio_buffer.extend(data)

            while len(self._output_audio_buffer) >= TARGET_CHUNK_SIZE_BYTES:
                chunk_data = self._output_audio_buffer[:TARGET_CHUNK_SIZE_BYTES]
                del self._output_audio_buffer[:TARGET_CHUNK_SIZE_BYTES]

                chunk_frame = rtc.AudioFrame(
                    data=bytes(chunk_data),
                    sample_rate=last_meta.audio_sample_rate,
                    num_channels=TTS_NUM_CHANNELS,
                    samples_per_channel=TARGET_SAMPLES_PER_CHANNEL,
                )

                try:
                    item_gen.audio_ch.send_nowait(chunk_frame)
                except Exception as e:
                    logger.error(f"Channel closed: {item_gen.audio_ch.closed}")
                    raise

    def _close_current_generation(self) -> None:
        if not self._current_generation:
            return

        if self._current_generation.full_text:
            self._chat_ctx.add_message(
                role="assistant",
                content=self._current_generation.full_text,
            )

        self._agent_is_replying = False

        if self._current_generation and not self._current_generation.message.audio_ch.closed:
            item_gen = self._current_generation.message

            def _send_frame(data: bytes, description: str) -> bool:
                """Creates and sends an audio frame, returns False on failure."""
                frame = rtc.AudioFrame(
                    data=data,
                    sample_rate=TTS_SAMPLE_RATE,
                    num_channels=TTS_NUM_CHANNELS,
                    samples_per_channel=TARGET_SAMPLES_PER_CHANNEL,
                )
                try:
                    item_gen.audio_ch.send_nowait(frame)
                    return True
                except Exception as e:
                    logger.error(f"Failed to send {description} frame: {e}")
                    return False

            if self._output_audio_buffer:
                remaining_data = bytearray(self._output_audio_buffer)
                padding_needed = TARGET_CHUNK_SIZE_BYTES - len(remaining_data)
                if padding_needed > 0:
                    remaining_data.extend(b"\x00" * padding_needed)
                if lk_fw_debug:
                    logger.debug(
                        f"Flushing remaining audio buffer of size {len(self._output_audio_buffer)} "
                        f"and sending padded frame of size {len(remaining_data)}"
                    )
                _send_frame(bytes(remaining_data), "final audio")

            # Send a few silent frames to signal the end of the utterance clearly
            logger.info("Sending 5 extra frames of silence to mark end of utterance.")
            silent_frame_data = b"\x00" * TARGET_CHUNK_SIZE_BYTES
            for _ in range(5):
                if not _send_frame(silent_frame_data, "silent"):
                    break  # Stop if channel is closed or an error occurs

        self._output_audio_buffer.clear()
        if not self._current_generation.message.text_ch.closed:
            self._current_generation.message.text_ch.close()
        if not self._current_generation.message.audio_ch.closed:
            self._current_generation.message.audio_ch.close()
        self._current_generation.function_ch.close()
        self._current_generation.message_ch.close()
        self._current_generation.pending_audio_metadata.clear()
        if (msg := self._chat_ctx.get_by_id(self._current_generation.message.message_id)) and isinstance(
            msg, llm.ChatMessage
        ):
            msg.content = [self._current_generation.full_text]
        with contextlib.suppress(asyncio.InvalidStateError):
            self._current_generation._done_fut.set_result(None)
        self._current_generation = None
        if lk_fw_debug:
            logger.info("Current generation cleared.")

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model._label,
                error=error,
                recoverable=recoverable,
            ),
        )

    def start_trace(self) -> str:
        """Start a performance trace and return the trace_id."""
        trace_id = str(uuid.uuid4())
        trace_event = AgentInputTrace(trace_id=trace_id)
        self.send_event(trace_event)
        logger.debug(f"Starting trace with id: {trace_id}")
        return trace_id
