# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from livekit.agents._core import SessionClock
from livekit.agents.tts import TTS
from livekit.agents.types import APIConnectOptions

from .._core import InferenceTtsStream
from .inference_stt import FallbackModel
from .inference_tts import FallbackModel as HostFallbackModel
from .realtime import RealtimeCapabilities
from .stt import STTCapabilities
from .tts import TTSCapabilities
from .turn_events import TurnDetectorModel

def inference_resolve_api_key(explicit: str | None = None, server: str | None = None) -> str:
    """`explicit`, else `LIVEKIT_INFERENCE_API_KEY`, else `server` (the running job's server
    key), else `LIVEKIT_API_KEY`; a set but empty `LIVEKIT_INFERENCE_API_KEY` shadows the rest.
    """

def inference_resolve_api_secret(explicit: str | None = None, server: str | None = None) -> str:
    """The same ladder for the secret."""

def inference_quota_usage(headers: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """The gateway's quota dimensions among a response's header pairs (names case-insensitive);
    an empty value is not reported.
    """

def inference_default_url(server_url: str | None = None) -> str:
    """`LIVEKIT_INFERENCE_URL`, then staging if the server URL names the staging cloud, then
    production. `server_url` is the running job's (`AgentServer(ws_url=)`), else `LIVEKIT_URL`.
    """

def local_models_native() -> bool:
    """Whether the build links a native provider of the local models."""

def local_models_init(vad: bool, eot: bool) -> bool:
    """Loads the native models' weights; `false` when no provider is linked."""

def interruption_min_frames(min_interruption_duration: float) -> int:
    """The frames an interruption must span: `min_interruption_duration` at the detector's rate."""

def interruption_estimate_probability(probabilities: Sequence[float]) -> float:
    """The interruption probability a window of frame probabilities reports."""

def tts_parse_model_string(model: str) -> tuple[str, str | None]:
    """`"provider/model:voice"` split into the model and its voice."""

def tts_has_aligned_transcript(model: str, extra_kwargs: dict[str, object]) -> bool:
    """Whether the model streams word timings for its audio under these options."""

def tts_markup_provider_key(model: str) -> str:
    """The speech-markup dialect the model's provider reads."""

def stt_parse_model_string(model: str) -> tuple[str, str | None]:
    """`"provider/model:language"` split into the model and its language."""

def stt_diarization_enabled(extra_kwargs: dict[str, object] | None = None) -> bool:
    """Whether these options ask the provider for speaker diarization."""

def stt_supports_agent_context_carryover(model: str | None = None) -> bool:
    """Whether the model carries the agent's context across recognitions."""

def stt_aligned_transcript_for_model(model: str | None = None) -> bool:
    """Whether the model reports word timings by default."""

def min_reasoning_effort(model: str) -> str | None:
    """The lowest reasoning effort the model supports, or `None` if it has no such control."""

def amd_auto_select_models(livekit_url: str, api_key: str, api_secret: str) -> bool:
    """Whether AMD may fill in its own LLM and STT from the gateway (`detector.py:184-194`)."""

def amd_is_evaluated_model(model: str, kind: str) -> bool:
    """Whether `model` is in AMD's evaluated `"llm"` or `"stt"` table."""

def amd_category_is_human(category: str) -> bool:
    """`AMDPredictionEvent.is_human` for a category's wire value."""

def amd_category_is_machine(category: str) -> bool:
    """`AMDPredictionEvent.is_machine` for a category's wire value."""

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
) -> tuple[TurnDetectorModel, bool, list[str]]:
    """The turn detector a host constructs: `(model, cloud, warnings)`."""

def stt_resolve_vad_for_model(model: str | None = None, *, has_vad: bool = False) -> str:
    """`_resolve_vad_for_model`: what the host does with the caller's VAD for this model."""

def stt_normalize_fallback_str(model: str) -> FallbackModel:
    """`_normalize_fallback`'s string arm for an STT chain."""

def tts_normalize_fallback_str(model: str) -> HostFallbackModel:
    """`_normalize_fallback`'s string arm for a TTS chain."""

def stt_capabilities(
    model: str | None = None,
    *,
    fallback: Sequence[str] | None = None,
    extra_kwargs: dict[str, object] | None = None,
) -> dict[str, object]:
    """`inference.STT`'s capabilities. Word alignment holds only when the model and every
    `fallback` model align: any turn may be served by a fallback, and a false claim disables
    the fast VAD interruption path.
    """

def stt_fallback_capabilities(entries: Sequence[STTCapabilities]) -> dict[str, object]:
    """`stt.FallbackAdapter`'s capabilities, merged from its members' before any chain is built."""

def tts_fallback_capabilities(entries: Sequence[TTSCapabilities]) -> dict[str, object]:
    """`tts.FallbackAdapter`'s capabilities, merged from its members' before any chain is built."""

def merge_realtime_capabilities(caps: Sequence[RealtimeCapabilities]) -> dict[str, object]:
    """The realtime fallback's capabilities: `audio_output` and `turn_detection` must agree (a
    `ValueError` naming the capability and both values), the rest are ANDed.
    """

class InferenceAvatar:
    """One avatar provisioned through LiveKit Inference: at most one provider session per instance,
    created under one operation identity, held until `aclose()` terminates it.
    """
    def __init__(
        self,
        model: str,
        *,
        avatar_identity: str | None = None,
        avatar_name: str | None = None,
        extra_kwargs: dict[str, object] | None = None,
        base_url: str,
        api_key: str,
        api_secret: str,
        conn_options: APIConnectOptions,
    ) -> None:
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
        headers: Sequence[tuple[str, str]],
    ) -> int:
        """Creates the provider session for the agent `agent_identity` in `room_name` and returns the
        audio sample rate the avatar takes. A second call while one runs, or once a session was
        created, is refused. Cancelling the call stops creating; a session created anyway is
        terminated. `livekit_url` defaults to `LIVEKIT_URL`; `headers` are `get_inference_headers()`.
        """
    async def aclose(self, headers: Sequence[tuple[str, str]]) -> None:
        """Stops creating and terminates every held session so it stops billing now. Returns once
        that cleanup settled or its deadline passed (then logged); a failed terminate keeps the
        session held, so a second call retries.
        """

class InferenceTts:
    """`inference/tts.py`'s WebSocket, over `lk-agents-inference::tts::GatewayTts`."""
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str,
        api_secret: str,
        conn_options: APIConnectOptions,
        voice: str | None = None,
        language: str | None = None,
        encoding: str = "pcm_s16le",
        sample_rate: int = 24000,
        extra_kwargs: dict[str, object] | None = None,
        fallback: list[HostFallbackModel] | None = None,
    ) -> None: ...
    def set_expressive(self, expressive: bool) -> None: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def aligned_transcript(self) -> bool: ...
    def bind_clock(self, clock: SessionClock) -> None: ...
    def prewarm(self, *, headers: Sequence[tuple[str, str]] | None = None) -> None: ...
    def identified(
        self,
        headers: Sequence[tuple[str, str]],
        reporter: TTS[Any] | None = None,
    ) -> InferenceTts: ...
    def update_options(
        self,
        *,
        model: str | None = None,
        voice: str | None = None,
        language: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> None: ...
    async def stream(
        self,
        *,
        headers: Sequence[tuple[str, str]] | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> InferenceTtsStream: ...
