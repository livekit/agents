# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from livekit.agents import VAD
from livekit.agents._core_view import AvailabilityObserver
from livekit.agents.stt import STT
from livekit.agents.types import APIConnectOptions
from livekit.rtc import AudioFrame

from .inference_stt import FallbackModel

class FallbackStt:
    """`stt.FallbackAdapter`'s engine as the host builds it; the public class stays Python."""
    def __init__(
        self,
        stts: Sequence[STT[Any]],
        *,
        attempt_timeout: float = 10.0,
        max_retry_per_stt: int = 1,
        retry_interval: float = 5.0,
        on_availability: AvailabilityObserver | None = None,
    ) -> None:
        """`stts` in order, each already streaming (the host wraps a recognize-only entry in its
        `StreamAdapter`); the keywords are `FallbackAdapter.__init__`'s.
        """
    @property
    def active_index(self) -> int:
        """Which entry most recently served, as a position in the host's list."""
    @property
    def capabilities(self) -> dict[str, object]:
        """The merged capabilities, not the primary's: a caller must not wait for interim results a
        secondary never sends.
        """
    async def aclose(self) -> None:
        """Every recovery probe in flight closes its stream, awaited (bounded)."""
    async def recognize(
        self,
        frames: Sequence[AudioFrame],
        language: str | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> dict[str, object]:
        """`_recognize_impl`: the batch path, over whichever entry answers."""
    async def stream(
        self,
        language: str | None = None,
        start_time_offset: float | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> FallbackRecognizeStream:
        """Audio in, tee'd to the serving stream and every recovery probe; events out."""

class FallbackRecognizeStream:
    def push(self, item: AudioFrame) -> None: ...
    def flush(self) -> None: ...
    def end_input(self) -> None: ...
    async def recv(self) -> dict[str, object] | None: ...
    async def aclose(self) -> None: ...

class InferenceStt:
    """The gateway STT behind `inference.STT`. Its options, keyterms and open streams are shared by
    every view (`identified`); each view's connections carry its own job's identification.
    """
    def __init__(
        self,
        *,
        model: str | None,
        language: str | None,
        encoding: str,
        sample_rate: int,
        base_url: str,
        api_key: str,
        api_secret: str,
        extra_kwargs: dict[str, object],
        fallback: list[FallbackModel] | None = None,
        conn_options: APIConnectOptions,
        vad: VAD[Any] | None = None,
    ) -> None:
        """A client of `model` on the gateway at `base_url`. Starts nothing. `vad`'s end of speech
        finalizes the transcript (Speechmatics only).
        """
    def set_vad(self, vad: VAD[Any] | None = None) -> None:
        """The model's VAD as the host resolved it on `update_options(model=)`; `None` drops it.
        Later streams use it.
        """
    def identified(
        self,
        headers: Sequence[tuple[str, str]],
        reporter: STT[Any] | None = None,
    ) -> InferenceStt:
        """This client as a session of the job `headers` (`get_inference_headers()`) identifies holds
        it: the same options, keyterms and open streams, its connections carrying that job.
        `reporter` is the host `stt.STT` whose `error` events report this view's failures.
        """
    def update_options(
        self,
        *,
        model: str | None = None,
        language: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        """Read by later streams, and sent to the open ones as a mid-stream `session.update`."""
    def update_keyterms(self, keyterms: Sequence[str]) -> None:
        """The session's keyterms: a warning once for a model without keyterm prompting, else every
        open stream applies them (now if the user is silent, at end of speech if not).
        """
    def push_conversation_item(self, role: str, text: str) -> None:
        """An assistant turn's text, its tail, biases recognition as `agent_context` on a model that
        carries it.
        """
    def stream(
        self,
        *,
        language: str | None = None,
        start_time_offset: float | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> InferenceSttStream:
        """Opens a stream; its pump opens the socket. `start_time_offset` shifts its timestamps onto
        the caller's input timeline.
        """

class InferenceSttStream:
    def push(self, item: AudioFrame) -> None: ...
    def flush(self) -> None: ...
    def end_input(self) -> None: ...
    def update_options(
        self,
        *,
        model: str | None = None,
        language: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        """A mid-stream `session.update`, for the providers that take one (AssemblyAI, Deepgram Flux)."""
    async def recv(self) -> dict[str, object] | None: ...
    async def aclose(self) -> None: ...

class SttStreamAdapter:
    """`stt.StreamAdapter`'s engine as the host builds it; the public class stays Python."""
    def __init__(self, stt: STT[Any], vad: VAD[Any]) -> None:
        """The wrapped STT and VAD; each stream binds its own recognition policy."""
    async def recognize(
        self,
        frames: Sequence[AudioFrame],
        language: str | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> dict[str, object]:
        """`_recognize_impl`: the wrapped STT's batch path."""
    async def stream(
        self,
        language: str | None = None,
        start_time_offset: float | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> SttAdapterStream:
        """Audio into the VAD, its utterances recognized."""

class SttAdapterStream:
    def push(self, item: AudioFrame) -> None: ...
    def flush(self) -> None: ...
    def end_input(self) -> None: ...
    async def recv(self) -> dict[str, object] | None: ...
    async def aclose(self) -> None: ...
