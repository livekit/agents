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
from .inference_tts import FallbackModel

def inference_resolve_api_key(explicit: str | None = None) -> str:
    """`explicit`, else `LIVEKIT_INFERENCE_API_KEY`, else `LIVEKIT_API_KEY`; a set but empty
    `LIVEKIT_INFERENCE_API_KEY` shadows `LIVEKIT_API_KEY` (`os.getenv(A, os.getenv(B, ""))`).
    """

def inference_resolve_api_secret(explicit: str | None = None) -> str:
    """The same ladder for the secret."""

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
    """`inference/tts.py`'s WebSocket, over `lk-agents-inference::tts::NativeTts`."""
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
        fallback: list[FallbackModel] | None = None,
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
