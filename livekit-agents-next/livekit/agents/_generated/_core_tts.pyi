# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

from livekit.agents.tokenize import SentenceStream
from livekit.agents.tts import TTS, AudioEmitter

from .connect import APIConnectOptions
from .expressive import SpeechSteeringOptions
from .tts import ExpressiveTag, MarkupInfo, SynthesizedAudio

def chat_message_text_content(parts: Sequence[str], assistant: bool) -> str | None: ...

def supported_nonverbals(provider: str) -> MarkupInfo: ...

def split_all_markup(text: str) -> tuple[str, list[ExpressiveTag]]: ...

def strip_all_markup(text: str) -> str: ...

def strip_expr_markup(text: str) -> str: ...

def split_expr_markup(text: str) -> tuple[str, list[ExpressiveTag]]: ...

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

async def tts_adapter_stream(
    provider: TTS[Any],
    conn_options: APIConnectOptions,
    *,
    expressive: bool = False,
    sentences: SentenceStream | None = None,
    pacing: tuple[float, int] | None = None,
) -> TtsStream:
    """`provider`'s synthesis of each sentence of the text pushed, one segment, each sentence
    retried within `conn_options`. Under `expressive` the provider's markup is lowered
    before each sentence. `sentences` is the host tokenizer's stream the text is split by
    (blingfire's by default); `pacing` is `(min_remaining_audio, max_text_length)` to release
    sentences as the audio plays out (`tts.SentenceStreamPacer`).
    """

class _core_PacedSentences:
    """`SentenceStreamPacer.wrap()`: a host sentence stream's sentences, paced by the audio its
    emitter pushed. The first sentence goes at once; the next batch once the audio stopped
    growing and less than `min_remaining_audio` seconds of it is left to play, at most
    `max_text_length` characters of whole sentences joined by a space.
    """
    def __new__(
        cls,
        sentences: SentenceStream,
        audio: AudioEmitter,
        min_remaining_audio: float,
        max_text_length: int,
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
    async def aclose(self) -> None: ...

def validate_synthesis(text: str, duration: float, segments: tuple[int, int] | None) -> str | None:
    """Why a synthesis of ``text`` that pushed ``duration`` seconds of audio (and, streamed,
    ``segments`` as ``(expected, actual)``) failed, or ``None``.
    """
