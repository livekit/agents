# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Self

from livekit.agents import VAD
from livekit.agents._bindings import FallbackListener
from livekit.agents.stt import STT
from livekit.local_inference import VAD as HostVAD

from .connect import APIConnectOptions
from .inference_stt import FallbackModels
from .rtc_frames import AudioFrame
from .stt import SpeechEvent, STTCapabilities

class FallbackStt:
    """`stt.FallbackAdapter`'s engine as the host builds it; the public class stays Python."""
    def __new__(
        cls,
        stt: Sequence[STT[Any]],
        *,
        vad: VAD[Any] | None = None,
        attempt_timeout: float = 10.0,
        max_retry_per_stt: int = 1,
        retry_interval: float = 5.0,
        on_availability: FallbackListener | None = None,
    ) -> Self:
        """`stt` in order; a recognize-only one streams over `vad` (`stt.StreamAdapter`)."""
    @property
    def active_index(self) -> int:
        """Which entry most recently served, as a position in the host's list."""
    @property
    def capabilities(self) -> STTCapabilities:
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
    ) -> SpeechEvent:
        """`_recognize_impl`: the batch path, over whichever entry answers."""
    async def stream(
        self,
        *,
        language: str | None = None,
        start_time_offset: float | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> FallbackRecognizeStream:
        """Audio in, tee'd to the serving stream and every recovery probe; events out."""

class FallbackRecognizeStream:
    def push(self, item: AudioFrame) -> None: ...
    def flush(self) -> None: ...
    def end_input(self) -> None: ...
    async def recv(self) -> SpeechEvent | None: ...
    async def aclose(self) -> None: ...

class InferenceStt:
    """The gateway STT behind `inference.STT`. Its options, keyterms and open streams are shared by
    every view (`identified`); each view's connections carry its own job's identification. As the
    session's STT it implements the keyterm and conversation hooks.
    """
    def __new__(
        cls,
        vad_models: Callable[[], HostVAD],
        model: str | None = None,
        *,
        language: str | None = None,
        base_url: str | None = None,
        encoding: str | None = None,
        sample_rate: int | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        extra_kwargs: dict[str, object] | None = None,
        fallback: FallbackModels | None = None,
        conn_options: APIConnectOptions | None = None,
        vad: VAD[Any] | None = None,
    ) -> Self:
        """A client of `model` (``"provider/model[:language]"``) on the gateway. Starts nothing. The
        gateway's URL and key pair resolve from the given values, the environment and the
        server's settings. `vad`'s end of speech finalizes the transcript of a model that
        endpoints client-side (Speechmatics, OpenAI), which gets a default one from
        `vad_models` when none is given; a model that endpoints server-side ignores it.
        """
    @property
    def model(self) -> str:
        """`"unknown"` without a model."""
    @property
    def capabilities(self) -> STTCapabilities:
        """Recomputed from the model, its fallbacks and the extra kwargs."""
    @property
    def conn_options(self) -> APIConnectOptions:
        """The policy this client was given, which a session or run that supplies none keeps."""
    def identified(
        self,
        headers: list[tuple[str, str]],
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
        """Read by later streams, and sent to the open ones as a mid-stream `session.update`. A new
        model resolves its VAD again.
        """
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
    async def recv(self) -> SpeechEvent | None: ...
    async def aclose(self) -> None: ...

class SttStreamAdapter:
    """`stt.StreamAdapter`'s engine as the host builds it; the public class stays Python."""
    def __new__(cls, *, stt: STT[Any], vad: VAD[Any]) -> Self:
        """The wrapped STT and VAD; each stream binds its own recognition policy."""
    @property
    def capabilities(self) -> STTCapabilities:
        """Streaming, with the wrapped STT's keyterms and chat context support."""
    async def recognize(
        self,
        frames: Sequence[AudioFrame],
        language: str | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> SpeechEvent:
        """`_recognize_impl`: the wrapped STT's batch path."""
    async def stream(
        self,
        *,
        language: str | None = None,
        start_time_offset: float | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> SttAdapterStream:
        """Audio into the VAD, its utterances recognized."""

class SttAdapterStream:
    def push(self, item: AudioFrame) -> None: ...
    def flush(self) -> None: ...
    def end_input(self) -> None: ...
    async def recv(self) -> SpeechEvent | None: ...
    async def aclose(self) -> None: ...
