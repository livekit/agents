# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

from livekit.agents.tts import TTS

from .._core import TtsStream
from .connect import APIConnectOptions
from .inference_credentials import Credentials
from .inference_models import EotSelection
from .inference_tts import FallbackModels
from .stt import STTCapabilities
from .tts import TTSCapabilities

def inference_api_key(explicit: str | None = None) -> str:
    """`explicit`, else `LIVEKIT_INFERENCE_API_KEY`, else the server's `LIVEKIT_API_KEY`; `""`
    when nothing resolves. A set but empty `LIVEKIT_INFERENCE_API_KEY` shadows the server's.
    """

def inference_api_secret(explicit: str | None = None) -> str:
    """The same ladder for the secret, over `LIVEKIT_INFERENCE_API_SECRET`."""

def inference_url() -> str:
    """A nonempty `LIVEKIT_INFERENCE_URL`, then staging if the server's `LIVEKIT_URL` names the
    staging cloud, then production.
    """

def inference_credentials(
    base_url: str | None = None,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> Credentials:
    """The gateway's URL (a nonempty `base_url`, else [`Self::inference_url`]) and key pair (the
    ladders above); a `ValueError` when the key or the secret does not resolve.
    """

def inference_identity_headers(
    user_agent: str,
    inference_class: str | None = None,
    room_sid: str | None = None,
    job_id: str | None = None,
    agent_sid: str | None = None,
    job_headers: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """An inference request's identification headers, in order: `user_agent`, the priority of a
    nonempty `inference_class`, and in a job (`job_headers` given) its nonempty room, job and
    agent ids, `LIVEKIT_WORKER_TOKEN` when set, the job's `job_headers` (overriding any of
    these) and a fresh session id.
    """

def inference_quota_usage(headers: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The gateway's quota dimensions among a response's header pairs (names case-insensitive);
    an empty value is not reported.
    """

def local_models_init(vad: bool, eot: bool) -> bool:
    """Loads the native models' weights; `false` when no provider is linked."""

def tts_markup_provider_key(model: str) -> str:
    """The speech-markup dialect the model's provider reads."""

def min_reasoning_effort(model: str) -> str | None:
    """The lowest reasoning effort the model supports, or `None` if it has no such control."""

def amd_auto_select_models(livekit_url: str, api_key: str, api_secret: str) -> bool:
    """Whether AMD may fill in its own LLM and STT from the gateway (`detector.py:184-194`)."""

def amd_is_evaluated_model(model: str, kind: str) -> bool:
    """Whether `model` is in AMD's evaluated `"llm"` or `"stt"` table."""

def llm_unsupported_params(
    model: str,
    *,
    has_tools: bool = False,
    reasoning_effort: str | None = None,
    reasoning: str | None = None,
) -> list[str]:
    """The parameters the given model refuses. `has_tools` is `bool(tools)`; `reasoning_effort`
    is the request's `reasoning_effort` and `reasoning` the effort of its Responses
    `reasoning`.
    """

def eot_select(
    version: str | None,
    hosted: bool,
    dev: bool,
    url: str,
    key: str,
    secret: str,
    local_fallback: bool,
) -> EotSelection:
    """The turn detector a host constructs."""

def stt_fallback_capabilities(entries: Sequence[STTCapabilities]) -> STTCapabilities:
    """`stt.FallbackAdapter`'s capabilities, merged from its members' before any chain is built."""

def tts_fallback_capabilities(entries: Sequence[TTSCapabilities]) -> TTSCapabilities:
    """`tts.FallbackAdapter`'s capabilities, merged from its members' before any chain is built."""

class InferenceAvatar:
    """One avatar provisioned through LiveKit Inference: at most one provider session per instance,
    created under one operation identity, held until `aclose()` terminates it.
    """
    def __new__(
        cls,
        model: str,
        *,
        avatar_identity: str | None = None,
        avatar_name: str | None = None,
        extra_kwargs: dict[str, object] | None = None,
        base_url: str,
        api_key: str,
        api_secret: str,
        conn_options: APIConnectOptions,
    ) -> Self:
        """`model` is `"provider"` or `"provider/<avatar_id>"`. The identity defaults to
        `"<provider>-inference-avatar"`, the name to the identity. Starts nothing.
        """
    @property
    def provider(self) -> str: ...
    @property
    def avatar_identity(self) -> str: ...
    @property
    def session_id(self) -> str | None:
        """The gateway-generated avatar session id, once created."""
    @property
    def provider_session_id(self) -> str | None:
        """The provider's own session id while the session is held."""
    async def create(
        self,
        *,
        room_name: str,
        room_sid: str,
        livekit_url: str | None = None,
        agent_identity: str,
        headers: list[tuple[str, str]],
    ) -> int:
        """Creates the provider session for the agent `agent_identity` in `room_name` and returns the
        audio sample rate the avatar takes. A second call while one runs, or once a session was
        created, is refused. Cancelling the call stops creating; a session created anyway is
        terminated. `livekit_url` defaults to `LIVEKIT_URL`; `headers` are `get_inference_headers()`.
        """
    async def aclose(self, headers: list[tuple[str, str]]) -> None:
        """Stops creating and terminates every held session so it stops billing now. Returns once
        that cleanup settled or its deadline passed (then logged); a failed terminate keeps the
        session held, so a second call retries.
        """

class InferenceTts:
    """`inference/tts.py`'s WebSocket, over `lk-agents-inference::tts::GatewayTts`."""
    def __new__(
        cls,
        model: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        encoding: str | None = None,
        sample_rate: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        extra_kwargs: dict[str, object] | None = None,
        fallback: FallbackModels | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> Self:
        """A client of `model` (``"provider/model[:voice]"``) on the gateway. Starts nothing. The
        gateway's URL and key pair resolve from the given values, the environment and the server's
        settings.
        """
    @property
    def model(self) -> str: ...
    @property
    def capabilities(self) -> TTSCapabilities:
        """Streaming, its alignment read from the live model and extra kwargs."""
    @property
    def num_channels(self) -> int: ...
    @property
    def conn_options(self) -> APIConnectOptions:
        """The policy this client was given, which a session or run that supplies none keeps."""
    @property
    def sample_rate(self) -> int: ...
    def prewarm(self, *, headers: list[tuple[str, str]] | None = None) -> None: ...
    def identified(
        self,
        headers: list[tuple[str, str]],
        reporter: TTS[Any] | None = None,
    ) -> InferenceTts: ...
    def update_options(
        self,
        *,
        voice: str | None = None,
        model: str | None = None,
        language: str | None = None,
        extra_kwargs: dict[str, object] | None = None,
    ) -> None:
        """Read by the next synthesis; ``extra_kwargs`` merges into the persistent kwargs."""
    async def stream(
        self,
        *,
        conn_options: APIConnectOptions | None = None,
        expressive: bool = False,
    ) -> TtsStream:
        """A synthesis of the text pushed; its connection carries this view's job."""
