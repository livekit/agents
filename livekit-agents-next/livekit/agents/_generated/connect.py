# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/types/connect.rs

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from inspect import get_annotations
from typing import Final

from .session_wire import DEFAULT_MAX_UNRECOVERABLE_ERRORS

# Record declarations:
#     rust/crates/lk-agents-abi/src/voice/session_options.rs


@dataclass(frozen=True)
class APIConnectOptions:
    """Per-operation connection policy: retries, their interval and the limit of each wait.

    The five phase limits are seconds, finite and non-negative, `None` disabling one: `pool` waits
    for transport capacity, `connect` for a usable connection (DNS and TLS included), `write`
    without progress while sending, `read` without progress while a response is expected, `total`
    the whole operation (admission, retries and fallback). Each phase defaults to 10 s; `total`
    is off.
    """

    max_retry: int = 3
    """Attempts after the first."""
    retry_interval: float = 2.0
    """Seconds between retries; the first retry waits 0.1 s."""
    pool_timeout: float | None = 10.0
    connect_timeout: float | None = 10.0
    write_timeout: float | None = 10.0
    read_timeout: float | None = 10.0
    total_timeout: float | None = None

    def __post_init__(self) -> None:
        from .. import _bindings

        _bindings.api_connect_options_validate(self)

    def interval_for_retry(self, num_retries: int) -> float:
        from .. import _bindings

        return _bindings.api_connect_options_interval(self, num_retries)

    def total_deadline(self) -> AbstractAsyncContextManager[None]:
        from .. import _bindings

        return _bindings.api_connect_options_total_deadline(self)


@dataclass
class SessionConnectOptions:
    """`SessionConnectOptions`: each leg's retry budget and timeout; not given (`None`) is the
    provider's own policy, else the defaults.
    """

    stt_conn_options: APIConnectOptions | None = None
    llm_conn_options: APIConnectOptions | None = None
    tts_conn_options: APIConnectOptions | None = None
    max_unrecoverable_errors: int = DEFAULT_MAX_UNRECOVERABLE_ERRORS
    """Maximum number of consecutive unrecoverable errors from STT, LLM, or TTS."""


APIConnectOptions.__annotations__ = get_annotations(APIConnectOptions, eval_str=True)
APIConnectOptions.__module__ = "livekit.agents.types"
SessionConnectOptions.__annotations__ = get_annotations(SessionConnectOptions, eval_str=True)
SessionConnectOptions.__module__ = "livekit.agents.voice.agent_session"


DEFAULT_API_CONNECT_OPTIONS: Final = APIConnectOptions(
    max_retry=3,
    retry_interval=2.0,
    pool_timeout=10.0,
    connect_timeout=10.0,
    write_timeout=10.0,
    read_timeout=10.0,
    total_timeout=None,
)
