# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/worker/worker.rs
#     rust/crates/lk-agents-abi/src/worker/job.rs
#     rust/crates/lk-agents-abi/src/voice/workflows.rs
# and the `#[host(export)]` / `#[host(object)]` items in `EXPORT_SOURCES`.
#
# The stub of every `_core` class and function the binding generates rather than
# writes. `_core.pyi` re-exports this file.
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from typing import Any

from ..types import NotGivenOr

class JobInfo:
    """The scalar half of a `livekit.agent.Job`, as handed to the user's `request_fnc`."""
    @property
    def id(self) -> str: ...
    @property
    def dispatch_id(self) -> str: ...
    @property
    def type(self) -> int: ...
    @property
    def room(self) -> str | None: ...
    @property
    def room_sid(self) -> str | None: ...
    @property
    def agent_name(self) -> str: ...
    @property
    def metadata(self) -> str: ...
    @property
    def deployment(self) -> str: ...
    @property
    def enable_recording(self) -> bool: ...
    @property
    def enable_redaction(self) -> bool: ...
    @property
    def proto(self) -> bytes: ...


class JobRequest:
    """A job offered to the host. The host returns a [`JobDecision`]."""
    @property
    def job(self) -> JobInfo: ...
    @property
    def resuming(self) -> bool: ...


class RunningJobInfo:
    """`RunningJobInfo`: what a launched job is handed."""
    @property
    def accept_arguments(self) -> AcceptArguments: ...
    @property
    def job(self) -> JobInfo: ...
    @property
    def url(self) -> str: ...
    @property
    def token(self) -> str: ...
    @property
    def worker_id(self) -> str: ...
    @property
    def fake_job(self) -> bool: ...
    @property
    def received_at(self) -> float: ...
    @property
    def accepted_at(self) -> float: ...
    @property
    def assigned_at(self) -> float: ...


class AcceptArguments:
    """Accepted participant fields; application attributes retain their own string keys."""
    @property
    def name(self) -> str: ...
    @property
    def identity(self) -> str: ...
    @property
    def metadata(self) -> str: ...
    @property
    def attributes(self) -> list[tuple[str, str]]: ...


class ProcessPoolOptions:
    """Process-pool settings resolved before the host launch callback."""

    def __init__(
        self,
        *,
        num_idle_processes: int,
        initialize_timeout: float,
        close_timeout: float,
        session_end_timeout: float,
        memory_warn_mb: float,
        memory_limit_mb: float,
        http_proxy: str | None,
    ) -> None: ...
    @property
    def num_idle_processes(self) -> int: ...
    @property
    def initialize_timeout(self) -> float: ...
    @property
    def close_timeout(self) -> float: ...
    @property
    def session_end_timeout(self) -> float: ...
    @property
    def memory_warn_mb(self) -> float: ...
    @property
    def memory_limit_mb(self) -> float: ...
    @property
    def http_proxy(self) -> str | None: ...


class ProcessStartOptions:
    """Resolved server settings handed to the process launch adapter."""

    def __init__(
        self,
        *,
        devmode: bool,
        job_processes: bool,
        env: list[tuple[str, str]],
        job_executor_type: str,
        multiprocessing_context: str,
        pool: ProcessPoolOptions,
    ) -> None: ...
    @property
    def devmode(self) -> bool: ...
    @property
    def job_processes(self) -> bool: ...
    @property
    def env(self) -> list[tuple[str, str]]: ...
    @property
    def job_executor_type(self) -> str: ...
    @property
    def multiprocessing_context(self) -> str: ...
    @property
    def pool(self) -> ProcessPoolOptions: ...


class StartJobFields:
    """The process's one job, shared by the generated host boundary and the IPC wire protocol."""

    def __init__(
        self,
        *,
        job: bytes,
        name: str,
        identity: str,
        metadata: str,
        url: str,
        token: str,
        worker_id: str,
        fake_job: bool,
        received_at: float,
        accepted_at: float,
        assigned_at: float,
        launched_at: float,
        traceparent: str | None = None,
    ) -> None: ...
    @property
    def job(self) -> bytes: ...
    @property
    def name(self) -> str: ...
    @property
    def identity(self) -> str: ...
    @property
    def metadata(self) -> str: ...
    @property
    def url(self) -> str: ...
    @property
    def token(self) -> str: ...
    @property
    def worker_id(self) -> str: ...
    @property
    def fake_job(self) -> bool: ...
    @property
    def received_at(self) -> float: ...
    @property
    def accepted_at(self) -> float: ...
    @property
    def assigned_at(self) -> float: ...
    @property
    def launched_at(self) -> float: ...
    @property
    def traceparent(self) -> str | None: ...


class GetAddressResult:
    """GetAddressResult(address: 'str')"""

    def __init__(self, *, address: str) -> None: ...
    @property
    def address(self) -> str: ...


class GetEmailResult:
    """GetEmailResult(email_address: 'str')"""

    def __init__(self, *, email_address: str) -> None: ...
    @property
    def email_address(self) -> str: ...


class GetPhoneNumberResult:
    """GetPhoneNumberResult(phone_number: 'str')"""

    def __init__(self, *, phone_number: str) -> None: ...
    @property
    def phone_number(self) -> str: ...


class GetNameResult:
    """GetNameResult(first_name: 'str | None' = None, middle_name: 'str | None' = None, last_name: 'str | None' = None)"""

    def __init__(
        self,
        *,
        first_name: str | None = None,
        middle_name: str | None = None,
        last_name: str | None = None,
    ) -> None: ...
    @property
    def first_name(self) -> str | None: ...
    @property
    def middle_name(self) -> str | None: ...
    @property
    def last_name(self) -> str | None: ...


class GetCardNumberResult:
    """GetCardNumberResult(issuer: 'str', card_number: 'str')"""

    def __init__(self, *, issuer: str, card_number: str) -> None: ...
    @property
    def issuer(self) -> str: ...
    @property
    def card_number(self) -> str: ...


class GetSecurityCodeResult:
    """GetSecurityCodeResult(security_code: 'str')"""

    def __init__(self, *, security_code: str) -> None: ...
    @property
    def security_code(self) -> str: ...


class GetExpirationDateResult:
    """GetExpirationDateResult(date: 'str')"""

    def __init__(self, *, date: str) -> None: ...
    @property
    def date(self) -> str: ...


class GetCreditCardResult:
    """GetCreditCardResult(cardholder_name: 'str', issuer: 'str', card_number: 'str', security_code: 'str', expiration_date: 'str')"""

    def __init__(
        self,
        *,
        cardholder_name: str,
        issuer: str,
        card_number: str,
        security_code: str,
        expiration_date: str,
    ) -> None: ...
    @property
    def cardholder_name(self) -> str: ...
    @property
    def issuer(self) -> str: ...
    @property
    def card_number(self) -> str: ...
    @property
    def security_code(self) -> str: ...
    @property
    def expiration_date(self) -> str: ...


def confirm_duplicate_description() -> str: ...
def validate_card_number(card_number: str) -> bool: ...
def card_expired(month: int, year: int, today: Any) -> bool: ...
def normalize_date_of_birth(year: int, month: int, day: int, today: Any) -> Any: ...
def normalize_email_address(email: str) -> str: ...
def normalize_phone_number(phone_number: str) -> str: ...
def tool_call_output(outcome: Any) -> tuple[str, bool, bool]: ...
HSR_ABORT: int
HSR_REFUSED: int
HSR_INTERNAL: int
HSR_TIMEOUT: int
HSR_PROTOCOL: int
def resolve_keyterm_detection(config: Any | None = None) -> dict[str, Any]: ...
def amd_resolve_max_endpointing_delay(provided: float | None, activity: float | None) -> float: ...
def amd_constants() -> dict[str, Any]: ...
def resolve_recording_options(record: Any) -> Any: ...
def rt_server_turn_detection(
    *,
    caps_turn_detection: bool,
    can_disable_turn_detection: bool,
    vad: bool,
    turn_detection: str | None,
    streaming_detector: bool,
    interruption_given: bool,
) -> bool: ...
def resolve_interruption_detector(
    *,
    stt_aligned_transcript: bool | None,
    stt_streaming: bool | None,
    realtime: bool,
    rt_server_turn_detection: bool,
    has_vad: bool,
    turn_detection: str | None,
    allow_interruptions: bool,
    agent_mode: Any | None,
    session_mode: Any | None,
    hosted: bool,
    dev_mode: bool,
) -> dict[str, Any]: ...
def speech_priorities() -> tuple[int, int, int]: ...
def interruption_timeout() -> float: ...
def default_input_modality() -> str: ...
def dtmf_code(symbol: str) -> int: ...
def format_dtmf(symbols: list[str]) -> str: ...
WARM_TRANSFER_IDENTITY: str
TWILIO_RINGING_TIMEOUT: float
END_CALL_DESCRIPTION: str
def default_aec_warmup() -> float: ...
def plan_session_start(
    *,
    requested: Any | None = None,
    job_default: bool = False,
    has_job_context: bool = False,
    another_session_is_primary: bool = False,
    session_host: bool | None = None,
    text_only: bool = False,
) -> dict[str, Any]: ...
def session_state_option_fields() -> list[tuple[str, bool]]: ...
def session_messages_topic() -> str: ...
def frame(payload: bytes) -> bytes: ...
def chat_message_raw_text_content(parts: list[str]) -> str | None: ...
def chat_ctx_copy_plan(
    facts: list[tuple[str, str, str | None, int, str | None, float, str | None]],
    *,
    exclude_function_call: bool = False,
    exclude_instructions: bool = False,
    exclude_empty_message: bool = False,
    exclude_handoff: bool = False,
    exclude_config_update: bool = False,
    tools: list[str] | None = None,
) -> list[int]: ...
def chat_ctx_truncate_plan(
    facts: list[tuple[str, str, str | None, int, str | None, float, str | None]],
    max_items: int,
) -> list[int] | None: ...
def chat_ctx_merge_plan(
    mine: list[tuple[str, str, str | None, int, str | None, float, str | None]],
    theirs: list[tuple[str, str, str | None, int, str | None, float, str | None]],
    *,
    exclude_function_call: bool = False,
    exclude_instructions: bool = False,
    exclude_config_update: bool = False,
) -> list[tuple[int, int]]: ...
def api_connect_options_check(
    max_retry: int,
    retry_interval: float,
    pool_timeout: float | None = None,
    connect_timeout: float | None = None,
    write_timeout: float | None = None,
    read_timeout: float | None = None,
    total_timeout: float | None = None,
) -> None: ...
def api_retry_interval(retry_interval: float, num_retries: int) -> float: ...
def _core_resolve_api_connect_options(provider: Any, call: Any | None = None) -> Any: ...
def instructions_message_id() -> str: ...
def remove_instructions(item_ids: list[str]) -> list[int]: ...
def rtc_wire_names() -> tuple[list[str], list[str], list[str], list[str], list[str]]: ...
def video_queue_frames() -> int: ...
def validate_synthesis(
    text: str,
    duration: float,
    segments: tuple[int, int] | None,
) -> str | None: ...
def filter_emoji(chunk: str) -> str: ...
def hyphenate_word(word: str) -> list[str]: ...
def chat_chunk_has_response(chunk: Any) -> bool: ...
def _core_mcp_timeout_check(field: str, seconds: float) -> None: ...


class Sound:
    def __init__(
        self,
        path: str | None,
        stream: Any | None = None,
        volume: float = 1.0,
        probability: float = 1.0,
        fade_in: float = 0.0,
        fade_out: float = 0.0,
    ) -> None: ...


class SessionOptions:
    """`AgentSession(**kwargs)`'s options: `kwargs` crosses once as a raw dict, and the resolved
    configuration comes back as the dict `session.options` views.
    """

    def __init__(self, kwargs: Any) -> None: ...
    def update(self, update: Any) -> None: ...
    def set_aec_warmup(self, duration: float | None) -> None: ...
    def resolved(self, legs: Any) -> dict[str, Any]: ...
    @property
    def turn_detection_explicit(self) -> bool: ...


class SpeechHandle:
    def __init__(self, *, speech_id: str = ..., allow_interruptions: bool = True) -> None: ...
    @property
    def host(self) -> Any | None: ...
    @host.setter
    def host(self, host: Any) -> None: ...
    @property
    def id(self) -> str: ...
    @property
    def num_steps(self) -> int: ...
    @property
    def chat_items(self) -> list[ChatItem]: ...
    @property
    def scheduled(self) -> bool: ...
    @property
    def interrupted(self) -> bool: ...
    def done(self) -> bool: ...
    @property
    def allow_interruptions(self) -> bool: ...
    @allow_interruptions.setter
    def allow_interruptions(self, value: bool) -> None: ...
    def exception(self) -> Any | None: ...
    def interrupt(self, *, force: bool = False) -> None: ...
    def hold_interruptions(self) -> None: ...
    def release_interruptions(self) -> None: ...
    async def wait_for_playout(self, *, origin: str | None = None) -> None: ...
    async def wait_for_generation(self, step: int | None = None) -> None: ...
    async def wait_for_interrupted(self) -> None: ...
    async def wait_until_created(self) -> None: ...


class GetDOBResult:
    """GetDOBResult(date_of_birth: 'date', time_of_birth: 'time | None' = None)"""

    def __init__(self, *, date_of_birth: Any, time_of_birth: Any | None = None) -> None: ...
    @property
    def date_of_birth(self) -> Any: ...
    @property
    def time_of_birth(self) -> Any | None: ...
    def format_long(self) -> str: ...
    def format_time(self) -> str | None: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...


class SessionClock:
    def __repr__(self) -> str: ...


class VoiceSession:
    """`voice/agent_session.py`'s `AgentSession`, over the driver: assembled at construction, opened at
    `start()`, closed at `aclose()`; every other method is one command. `frozen`, so there is no
    borrow flag: `aclose()` can be awaited while the event drain calls back into the same object.
    """

    def __init__(
        self,
        room: Any,
        *,
        instructions: Any | None = None,
        dynamic_instructions: Any | None = None,
        agent: str = ...,
        agent_host: Any | None = None,
        chat_ctx: list[ChatItem] | None = None,
        storage: Any | None = None,
        stt: Any | None = None,
        vad: Any | None = None,
        llm: Any | None = None,
        realtime: Any | None = None,
        tts: Any | None = None,
        on_event: Any | None = None,
        hooks: Any | None = None,
        participant_identity: str | None = None,
        auto_gain_control: bool = True,
        input_sample_rate: int = 24000,
        output_sample_rate: int = 24000,
        options: SessionOptions | None = None,
        agent_turn_handling: Any | None = None,
        agent_min_consecutive_speech_delay: float | None = None,
        agent_use_tts_aligned_transcript: bool | None = None,
        turn_detector: Any | None = None,
        interruption_detector: Any | None = None,
        text_output: bool = True,
        room_io_config: Any | None = None,
        tools: list[Any] | None = None,
        tools_host: Any | None = None,
        voice_agent: Any | None = None,
        transcription_node: Any | None = None,
        realtime_audio_output_node: Any | None = None,
        llm_node: Any | None = None,
        tts_node: Any | None = None,
        stt_node: Any | None = None,
        host_audio_input: Any | None = None,
        host_audio_output: Any | None = None,
        host_video_input: Any | None = None,
        video_sampler: tuple[float, float] | None = None,
        host_video_sampler: bool = False,
        host_transcription_output: Any | None = None,
        using_default_vad: bool = False,
        hook_bound: float | None = None,
        max_unrecoverable_errors: int = 3,
        stt_context: Any | None = None,
        keyterm_llm: Any | None = None,
        keyterm_tools: Any | None = None,
        agent_expressive: NotGivenOr[Any] = ...,
        session_id: str | None = None,
        remote_session: Any | None = None,
        console: Any | None = None,
        job: Any | None = None,
    ) -> None: ...
    @property
    def usage(self) -> list[dict[str, Any]]: ...
    @property
    def session_clock(self) -> SessionClock: ...
    @property
    def audio_input(self) -> RoomAudioInput | None: ...
    @property
    def video_input(self) -> RoomVideoInput | None: ...
    def attach_recorder(self, recorder: Any | None) -> None: ...
    async def wait_until_ready(self) -> None: ...
    def update_models(
        self,
        *,
        llm: NotGivenOr[Any | None] = ...,
        tts: NotGivenOr[Any | None] = ...,
        stt: NotGivenOr[Any | None] = ...,
        vad: NotGivenOr[Any | None] = ...,
        llm_node: Any | None = None,
        tts_node: Any | None = None,
        stt_node: Any | None = None,
    ) -> None: ...
    async def update_agent(
        self,
        *,
        instructions: Any,
        dynamic_instructions: Any | None = None,
        agent: str,
        agent_host: Any | None = None,
        tools: list[Any] | None = None,
        tools_host: Any | None = None,
        voice_agent: Any,
        chat_ctx: list[ChatItem] | None = None,
        llm: NotGivenOr[Any | None] = ...,
        tts: NotGivenOr[Any | None] = ...,
        stt: NotGivenOr[Any | None] = ...,
        vad: NotGivenOr[Any | None] = ...,
        turn_handling: Any | None = None,
        min_consecutive_speech_delay: float | None = None,
        use_tts_aligned_transcript: bool | None = None,
        expressive: NotGivenOr[Any] = ...,
        llm_node: Any | None = None,
        tts_node: Any | None = None,
        stt_node: Any | None = None,
        transcription_node: Any | None = None,
        realtime_audio_output_node: Any | None = None,
    ) -> bool: ...
    async def run_voice_task(
        self,
        *,
        instructions: Any,
        dynamic_instructions: Any | None = None,
        agent: str,
        agent_host: Any | None = None,
        tools: list[Any] | None = None,
        tools_host: Any | None = None,
        voice_agent: Any,
        chat_ctx: list[ChatItem] | None = None,
        llm: NotGivenOr[Any | None] = ...,
        tts: NotGivenOr[Any | None] = ...,
        stt: NotGivenOr[Any | None] = ...,
        vad: NotGivenOr[Any | None] = ...,
        turn_handling: Any | None = None,
        min_consecutive_speech_delay: float | None = None,
        use_tts_aligned_transcript: bool | None = None,
        expressive: NotGivenOr[Any] = ...,
        llm_node: Any | None = None,
        tts_node: Any | None = None,
        stt_node: Any | None = None,
        transcription_node: Any | None = None,
        realtime_audio_output_node: Any | None = None,
        origin: str | None = None,
        opens_with_reply: bool = False,
    ) -> Any: ...
    def set_audio_input(self, source: Any | None) -> None: ...
    def set_video_input(self, source: Any | None) -> None: ...
    def set_audio_output(self, sink: Any | None) -> None: ...
    def set_transcription_output(self, sink: Any | None) -> None: ...
    def update_tools(
        self,
        tools: list[Any] | None = None,
        *,
        tools_host: Any | None = None,
        voice_agent: Any | None = None,
    ) -> None: ...
    def update_turn_options(self, update: Any) -> None: ...
    def set_aec_warmup(self, duration: float | None) -> None: ...
    def resolved_options(self) -> dict[str, Any] | None: ...
    def push_video(self, content: ChatContent) -> bool: ...
    def set_run_output(
        self,
        speech: SpeechHandle,
        *,
        value: Any | None = None,
        error: Any | None = None,
    ) -> None: ...
    def generate_reply(
        self,
        *,
        user_input: str | None = None,
        instructions: str | None = None,
        allow_interruptions: bool | None = None,
        tool_choice: Any | None = None,
        chat_ctx: list[ChatItem] | None = None,
        tools: list[Any] | None = None,
        tools_host: Any | None = None,
        voice_agent: Any | None = None,
        on_enter: bool = False,
        origin: str | None = None,
    ) -> SpeechHandle: ...
    def ivr_start(
        self,
        max_silence: float = 5.0,
        tools: list[Any] | None = None,
        *,
        tools_host: Any | None = None,
    ) -> None: ...
    def attach_amd(
        self,
        cfg: Any,
        llm: Any,
        tools: Any | None = None,
        stt: Any | None = None,
        participant_identity: str | None = None,
    ) -> None: ...
    async def aclose(self, *, reason: str = "user_initiated", drain: bool = False) -> None: ...
    def shutdown(self, *, drain: bool = True) -> None: ...
    @property
    def linked_participant(self) -> str | None: ...
    async def set_participant(self, identity: str | None) -> None: ...
    async def unset_participant(self) -> None: ...
    def filler_start(
        self,
        call_id: str,
        speech_id: str,
        *,
        delay: float,
        interval: float | None = None,
        max_steps: int | None = None,
        text: str | None = None,
        source: Any | None = None,
    ) -> int: ...
    def filler_stop(self, id: int) -> None: ...
    def clear_user_turn(self) -> None: ...
    def update_instructions(self, instructions: Any) -> None: ...
    def set_io_enabled(self, leg: str, enabled: bool) -> None: ...
    def tool_update(self, call_id: str, report: Any) -> None: ...
    def update_chat_ctx(self, items: list[ChatItem]) -> None: ...
    async def interrupt(self, *, force: bool = False) -> None: ...
    async def drain(self) -> str | None: ...
    async def dtmf_start(
        self,
        *,
        num_digits: int,
        ask_for_confirmation: bool = False,
        dtmf_input_timeout: float = 4.0,
        dtmf_stop_event: str = "#",
    ) -> dict[str, Any]: ...
    def dtmf_reply_running(self) -> bool: ...
    def end_call(self, speech: SpeechHandle) -> None: ...
    def dtmf_digit(self, digit: str) -> None: ...
    def dtmf_stop(self) -> None: ...
    async def wait_for_idle(self, *, hold: int | None = None) -> None: ...
    async def wait_for_idle_and_hold(self, *, origin: str | None = None) -> int: ...
    def release_idle_hold(self, hold: int) -> None: ...
    def claim_user_turn(self) -> None: ...
    def release_user_turn(self) -> None: ...
    def run(
        self,
        *,
        user_input: str | None,
        expects_output: bool,
        tool_retries: int,
        output_retries: int,
        retry_instructions: str,
        origin: str | None = None,
        host: Any | None = None,
    ) -> None: ...
    def say(
        self,
        text: str,
        *,
        allow_interruptions: bool | None = None,
        add_to_chat_ctx: bool = True,
        origin: str | None = None,
    ) -> SpeechHandle: ...
    def say_stream(
        self,
        text: Any,
        *,
        allow_interruptions: bool | None = None,
        add_to_chat_ctx: bool = True,
        origin: str | None = None,
    ) -> SpeechHandle: ...
    def say_audio(
        self,
        text: str,
        audio: Any,
        *,
        allow_interruptions: bool | None = None,
        add_to_chat_ctx: bool = True,
        origin: str | None = None,
    ) -> SpeechHandle: ...
    def host_metrics(self, record: Any) -> None: ...
    def ivr_final(self, text: str) -> None: ...
    def ivr_close(self) -> None: ...
    def amd_listening(self) -> bool: ...
    def amd_execute(self) -> None: ...
    def amd_close(self) -> None: ...
    def keyterms(self) -> list[str] | None: ...
    def agent_chat_ctx(self) -> list[ChatItem]: ...
    def set_static_keyterms(self, terms: list[str]) -> None: ...
    async def commit_user_turn(
        self,
        *,
        transcript_timeout: float = 2.0,
        stt_flush_duration: float = 2.0,
        skip_reply: bool = False,
    ) -> str: ...


class ModelUsageCollector:
    """`metrics.ModelUsageCollector` -- the live per-model rollup."""

    def __init__(self) -> None: ...
    def collect(self, metrics: Any) -> None: ...
    def flatten(self) -> list[dict[str, Any]]: ...


class SentencePacer:
    def __init__(self, minimum_audio: float, maximum_text: int) -> None: ...
    def push(self, text: str) -> None: ...
    def end(self) -> None: ...
    def poll(self, now: float, audio: float) -> tuple[str | None, float, float | None]: ...


class FrameDecoder:
    def __init__(self) -> None: ...


class ChatContent:
    """One element of `ChatMessage.content`: `str | ImageContent | AudioContent`, built by the host one
    constructor per arm so Rust never reads attributes or dispatches on `.type`.
    """

    @staticmethod
    def text(text: str) -> ChatContent: ...
    @staticmethod
    def image_url(
        *,
        id: str,
        url: str,
        inference_width: int | None = None,
        inference_height: int | None = None,
        inference_detail: str = "auto",
        mime_type: str | None = None,
    ) -> ChatContent: ...
    @staticmethod
    def image_frame(
        *,
        id: str,
        frame: Any,
        inference_width: int | None = None,
        inference_height: int | None = None,
        inference_detail: str = "auto",
        mime_type: str | None = None,
    ) -> ChatContent: ...
    @staticmethod
    def audio(*, frames: Any, transcript: str | None = None) -> ChatContent: ...
    @property
    def kind(self) -> str: ...
    @property
    def value(self) -> Any: ...


class ChatItem:
    """One row of conversation history: the discriminated union of `chat_context.py:407`, built by the
    host one constructor per variant. `created_at` is a wall clock; ids are always host-supplied.
    """

    @staticmethod
    def message(
        *,
        id: str,
        role: str,
        content: list[ChatContent],
        created_at: float,
        interrupted: bool = False,
        transcript_confidence: float | None = None,
        extra: dict[str, Any] | None = None,
        metrics: Any | None = None,
    ) -> ChatItem: ...
    @staticmethod
    def function_call(
        *,
        id: str,
        call_id: str,
        arguments: str,
        name: str,
        created_at: float,
        extra: dict[str, Any] | None = None,
        group_id: str | None = None,
    ) -> ChatItem: ...
    @staticmethod
    def function_call_output(
        *,
        id: str,
        call_id: str,
        output: str,
        is_error: bool,
        created_at: float,
        name: str | None = None,
        reply_required: bool = True,
    ) -> ChatItem: ...
    @staticmethod
    def agent_handoff(
        *,
        id: str,
        new_agent_id: str,
        created_at: float,
        old_agent_id: str | None = None,
    ) -> ChatItem: ...
    @staticmethod
    def agent_config_update(
        *,
        id: str,
        created_at: float,
        instructions: str | None = None,
        tools_added: list[str] | None = None,
        tools_removed: list[str] | None = None,
        tools: Any | None = None,
    ) -> ChatItem: ...
    @property
    def type(self) -> str: ...
    @property
    def tools(self) -> Any | None: ...
    @property
    def id(self) -> str: ...
    @property
    def created_at(self) -> float: ...
    def fields(self) -> dict[str, Any]: ...


class PlaybackTracker:
    """A host sink's side (`io.AudioOutput._core_tracker()`): it reports its playback events here, in
    its own `time.time()` stamps and float seconds.
    """

    def __init__(self) -> None: ...
    def note_flush(self) -> None: ...
    def reset(self) -> None: ...
    async def wait_for_playout(self) -> tuple[float, bool, str | None]: ...


class CoreTextOutput:
    async def capture_text(self, text: str) -> None: ...
    async def capture_timed(
        self,
        text: str,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> None: ...
    def flush(self) -> None: ...


class TrackPublishOptions:
    """The host's view: what `_core_rtc.LocalParticipant._core_publish_audio_track` builds its
    `rtc.AudioSource`, `LocalAudioTrack` and `rtc.TrackPublishOptions` from.
    """

    @property
    def name(self) -> str: ...
    @property
    def source(self) -> str: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def num_channels(self) -> int: ...
    @property
    def queue_size_ms(self) -> int: ...


class AudioStreamOptions:
    """The host's view: what `_core_rtc.TrackPublication._core_open_audio_stream` opens its
    `rtc.AudioStream` (and, with `auto_gain_control`, its `AudioProcessingModule`) with.
    """

    @property
    def participant_identity(self) -> str: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def num_channels(self) -> int: ...
    @property
    def frame_size_ms(self) -> int | None: ...
    @property
    def queue_capacity(self) -> int: ...
    @property
    def noise_cancellation(self) -> Any | None: ...
    @property
    def auto_gain_control(self) -> bool: ...


class VideoFrameRef:
    """What a host `io.VideoInput` answers `_core_next_frame()` with: its frame (interned; only the key
    crosses back) and the size it read off it.
    """

    def __init__(self, frame: Any, width: int, height: int, timestamp_us: int = 0) -> None: ...


class StreamTextOptions:
    """The host's view: `_core_rtc.LocalParticipant._core_stream_text` spreads it into
    `stream_text(..)`, leaving `sender_identity` out when it is `None` (an explicit `None` would
    publish the user's transcript as the agent).
    """

    @property
    def topic(self) -> str: ...
    @property
    def sender_identity(self) -> str | None: ...
    @property
    def attributes(self) -> dict[str, str]: ...


class StreamBytesOptions:
    """The host's view: `_core_rtc.LocalParticipant._core_stream_bytes` spreads it into
    `stream_bytes(..)`.
    """

    @property
    def name(self) -> str: ...
    @property
    def topic(self) -> str: ...
    @property
    def destination_identities(self) -> list[str]: ...
    @property
    def attributes(self) -> dict[str, str]: ...


class Transcription:
    """The host's view: `_core_rtc.LocalParticipant._core_publish_transcription` builds the
    `rtc.Transcription` and its `rtc.TranscriptionSegment`s (keyword dicts, `final` spelled plainly).
    """

    @property
    def participant_identity(self) -> str: ...
    @property
    def track_sid(self) -> str: ...
    @property
    def segments(self) -> list[dict[str, Any]]: ...


class PlaybackInfo:
    """What a host audio source's `PlayoutLedger` hands back, and the core reads back unchanged."""

    @property
    def played(self) -> float: ...
    @property
    def discarded(self) -> float: ...


class AudioOutputCapabilities:
    """What a host `io.AudioOutput` answers its `_core_capabilities()` with."""

    def __init__(self, pause: bool) -> None: ...
    @property
    def pause(self) -> bool: ...


class RoomEventBus:
    @property
    def subscriber_count(self) -> int: ...


class ByteStreamSender:
    def open(self, info: Any, sender: str, chunks: Any) -> bool: ...


class TextStreamSender:
    def open(self, info: Any, sender: str, chunks: Any) -> bool: ...


class RpcHandler:
    def call(self, caller_identity: str, payload: str) -> str: ...


class RoomAudioInput:
    @property
    def reading(self) -> str | None: ...
    async def next_frame(self) -> Any | None: ...
    async def aclose(self) -> None: ...


class RoomVideoInput:
    @property
    def reading(self) -> str | None: ...
    async def next_frame(self) -> Any | None: ...
    async def aclose(self) -> None: ...


class _core_RoomSession:
    """Runtime capability connecting a session to its room, clock, and owned I/O."""



class ChatRequest:
    """The request as a host `LLM._core_chat` reads it: a `_core.ChatRequest` handle over a clone. The
    host objects cross as their side-table keys (`_core.resolve_host_ref`), and the context as its
    `ChatContext.to_dict()` form, read only when the request carries no host context of its own.
    """

    @property
    def chat_ctx_host(self) -> int | None: ...
    @property
    def chat_ctx(self) -> Any: ...
    @property
    def tool_names(self) -> list[str]: ...
    @property
    def function_schemas(self) -> list[dict[str, Any]]: ...
    @property
    def tools_host(self) -> int | None: ...
    @property
    def tools_host_extra(self) -> list[int]: ...
    @property
    def parallel_tool_calls(self) -> bool | None: ...
    @property
    def tool_choice(self) -> Any | None: ...
    @property
    def extra(self) -> dict[str, Any] | None: ...
    @property
    def inference_class(self) -> str | None: ...


class MarkdownFilter:
    def __init__(self) -> None: ...
    def push(self, chunk: str) -> list[str]: ...
    def flush(self) -> list[str]: ...


class TranscriptSynchronizer:
    """`_core.TranscriptSynchronizer`: the node every native session builds, handed to a host that
    assembles a synchroniser by hand (the console, a custom chain), so pacing is one implementation
    on one clock.
    """

    def __init__(
        self,
        next_in_chain_audio: Any,
        next_in_chain_text: Any | None = None,
        *,
        speed: float = 1.0,
    ) -> None: ...
    @property
    def text_output(self) -> CoreTextOutput: ...
    @property
    def enabled(self) -> bool: ...
    def set_enabled(self, enabled: bool) -> None: ...
    def rotate_segment(self) -> None: ...
    async def aclose(self) -> None: ...


class SessionClosing(RuntimeError):
    """`wait_for_idle()`/`wait_for_idle_and_hold()` refused: the session is closing."""


class StaleHandle(RuntimeError):
    """A handle whose originating host no longer owns a live entry."""
