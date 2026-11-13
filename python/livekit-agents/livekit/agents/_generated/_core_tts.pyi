# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

from livekit.agents.tts import TTS, AudioEmitter

from .connect import APIConnectOptions
from .expressive import SpeechSteeringOptions
from .stream_pacer import SentenceStreamPacer
from .tts import (
    MarkupInfo,
    MarkupSplit,
    SegmentCount,
    SentenceTokenizerLeg,
    SynthesizedAudio,
    TTSCapabilities,
)

def chat_message_text_content(parts: Sequence[str], assistant: bool) -> str | None: ...

def supported_nonverbals(provider: str) -> MarkupInfo: ...

def strip_expr_markup(text: str) -> str: ...

def split_expr_markup(text: str) -> MarkupSplit:
    """Strip only the `<expr/>` dialect, returning the clean text and its markers; the
    provider-native tags stay in the words.
    """

def to_xml(
    tag_name: str,
    content: str | None = None,
    attrs: Sequence[tuple[str, str]] | None = None,
) -> str: ...

class Markup:
    """The markup dialect a TTS speaks, for the expressive pipeline: what the LLM is taught to write
    and how it is normalized and lowered before synthesis.

    Plugins opt in by overriding `_provider_key`.
    """
    def __new__(cls, provider_key: str) -> Self: ...
    def _provider_key(self) -> str:
        """The provider's key into the markup tables, or `""` for no markup."""
    def llm_instructions(
        self,
        *,
        speech_steering: SpeechSteeringOptions | None = None,
    ) -> str | None:
        """The instructions teaching the LLM this voice's markup, `None` without markup."""

class TtsStreamAdapter:
    """`tts.StreamAdapter`'s engine: a non-streaming TTS as a streaming one, each chunk of its input
    synthesized on its own.
    """
    def __new__(cls, *, tts: TTS[Any]) -> Self:
        """`tts` synthesizes each chunk of the text pushed."""
    @property
    def capabilities(self) -> TTSCapabilities:
        """Streaming, with word alignment."""
    @property
    def sample_rate(self) -> int: ...
    @property
    def num_channels(self) -> int: ...

class TtsStream:
    """One synthesis, `SynthesizeStream`'s shape: text in through `push_text`, `flush` and
    `end_input`, audio out through `recv`. `aclose` cancels it; it does not end the input.
    """
    def __new__(
        cls,
        *,
        tts: TTS[Any],
        text: str | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> Self:
        """A synthesis by `tts` of `text`, or, without it, of the text pushed. Its text goes through
        the TTS's stage (split, paced, lowered to its syntax) and its attempts are retried within
        `conn_options`; the first opens when its audio is first read.
        """
    def push_text(self, text: str) -> None: ...
    def flush(self) -> None: ...
    def end_input(self) -> None: ...
    async def recv(self) -> SynthesizedAudio | None: ...
    async def drain_into(self, emitter: AudioEmitter, *, stream: bool) -> None:
        """Every audio it synthesizes into `emitter`, until it ends: its frames, the words they
        cover and the usage reported, in one segment when `stream`.
        """
    async def aclose(self) -> None: ...

def validate_synthesis(text: str, duration: float, segments: SegmentCount | None) -> str | None:
    """Why a synthesis of ``text`` that pushed ``duration`` seconds of audio (and, streamed,
    ``segments``) failed, or ``None``.
    """

class TtsOptionsCell:
    """`_core.TtsOptionsCell`: the options a `tts.TTS` owns (`tts.options`), shared with every
    synthesis over it. A synthesis reads them when its stage opens; one already open keeps them.
    """
    def __new__(
        cls,
        sentence_tokenizer: object,
        leg: SentenceTokenizerLeg,
        capabilities: TTSCapabilities,
    ) -> Self:
        """Split by `sentence_tokenizer` (through `leg`, its core leg), paced as a TTS with
        `capabilities` is by default.
        """
    @property
    def sentence_tokenizer(self) -> object:
        """The tokenizer as it was given."""
    @property
    def pacing(self) -> SentenceStreamPacer | None: ...
    def set(
        self,
        sentence_tokenizer: object,
        leg: SentenceTokenizerLeg,
        pacing: SentenceStreamPacer | None,
    ) -> None:
        """Replace both at once."""
