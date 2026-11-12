# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

from livekit.agents.tokenize import SentenceStream, SentenceTokenizer
from livekit.agents.tts import TTS, AudioEmitter

from .connect import APIConnectOptions
from .expressive import SpeechSteeringOptions
from .stream_pacer import SentenceStreamPacer, TextPacing
from .tts import (
    ExpressiveTag,
    MarkupInfo,
    MarkupSplit,
    SegmentCount,
    SynthesizedAudio,
    TTSCapabilities,
)

def chat_message_text_content(parts: Sequence[str], assistant: bool) -> str | None: ...

def supported_nonverbals(provider: str) -> MarkupInfo: ...

def split_all_markup(text: str) -> MarkupSplit:
    """Strip every provider's expressive XML markup; brackets are prose and stay."""

def strip_all_markup(text: str) -> str: ...

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
    def normalize(self, text: str) -> str:
        """Fix common LLM markup mistakes (e.g. unclosed self-closing tags)."""
    def convert(self, text: str) -> str:
        """Lower the framework's markup to the provider's native syntax."""

class TranscriptMarkupStripper:
    """Stateful, provider-agnostic markup stripper for one transcript segment."""
    def __new__(cls) -> Self: ...
    def push(self, text: str) -> str: ...
    def flush(self) -> str: ...
    @property
    def tags(self) -> list[ExpressiveTag]: ...
    def expression_attribute(self) -> dict[str, str] | None: ...

class TtsStreamAdapter:
    """`tts.StreamAdapter`'s engine: a non-streaming TTS as a streaming one, the text split into
    sentences each synthesized on its own.
    """
    def __new__(
        cls,
        *,
        tts: TTS[Any],
        sentence_tokenizer: SentenceTokenizer | None = None,
        text_pacing: TextPacing | None = None,
    ) -> Self:
        """`tts` synthesizes each sentence of the text pushed; `sentence_tokenizer` splits it
        (`tokenize.SentenceTokenizer`'s by default); `text_pacing` releases sentences as the audio plays out.
        """
    @property
    def capabilities(self) -> TTSCapabilities:
        """Streaming, with word alignment."""
    @property
    def sample_rate(self) -> int: ...
    @property
    def num_channels(self) -> int: ...
    async def stream(
        self,
        *,
        conn_options: APIConnectOptions | None = None,
        expressive: bool = False,
    ) -> TtsStream:
        """One segment of the text pushed, each sentence retried within `conn_options`. Under
        `expressive` the wrapped TTS's markup is lowered before each sentence.
        """

class _core_PacedSentences:
    """`SentenceStreamPacer.wrap()`: a host sentence stream's sentences, paced by the audio its
    emitter pushed.
    """
    def __new__(
        cls,
        sent_stream: SentenceStream,
        audio_emitter: AudioEmitter,
        *,
        pacer: SentenceStreamPacer,
    ) -> Self: ...
    async def next(self) -> str | None:
        """The next batch, `None` once the input ended and every sentence went; the host stream's
        failure is raised as it was.
        """
    async def aclose(self) -> None:
        """Stop reading and close the host stream."""

class TtsStream:
    """One synthesis, `SynthesizeStream`'s shape: text in through `push_text`, `flush` and
    `end_input`, audio out through `recv`. `aclose` cancels it; it does not end the input.
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
