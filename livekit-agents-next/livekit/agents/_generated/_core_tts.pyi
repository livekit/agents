# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

from livekit.agents.tts import TTS
from livekit.agents.types import APIConnectOptions

from .chat_items import ChatItem
from .expressive import SpeechSteeringOptions

def chat_message_text_content(parts: Sequence[str], assistant: bool) -> str | None: ...

def supported_nonverbals(provider: str) -> dict[str, list[str]]: ...

def split_all_markup(text: str) -> tuple[str, list[dict[str, object]]]: ...

def strip_all_markup(text: str) -> str: ...

def strip_expr_markup(text: str) -> str: ...

def split_expr_markup(text: str) -> tuple[str, list[dict[str, object]]]: ...

def to_xml(
    tag_name: str,
    content: str | None = None,
    attrs: Sequence[tuple[str, str]] | None = None,
) -> str: ...

def chat_history_summary_xml(summary: str) -> str: ...

def chat_ctx_summarize_plan(
    items: Sequence[ChatItem],
    keep_last_turns: int,
) -> tuple[int, list[int], str, str, float] | None: ...

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
    def tags(self) -> list[dict[str, object]]: ...
    def expression_attribute(self) -> dict[str, str] | None: ...

async def tts_adapter_stream(provider: TTS[Any], conn_options: APIConnectOptions) -> TtsStream: ...

class SentencePacer:
    """Paces a synthesis's sentences: the first at once, the rest in batches once the audio already
    synthesized is nearly played.
    """
    def __new__(cls, minimum_audio: float, maximum_text: int) -> Self: ...
    def push(self, text: str) -> None: ...
    def end(self) -> None: ...
    def poll(self, now: float, audio: float) -> tuple[str | None, float, float | None]:
        """Text to emit, remaining audio for diagnostics, and next wakeup (None means done)."""

class TtsStream:
    """One synthesis, `SynthesizeStream`'s shape: text in through `push_text`, `flush` and
    `end_input`, audio out through `recv`. `aclose` cancels it; it does not end the input.
    """
    def push_text(self, text: str) -> None: ...
    def flush(self) -> None: ...
    def end_input(self) -> None: ...
    async def recv(self) -> dict[str, object] | None: ...
    async def aclose(self) -> None: ...

def validate_synthesis(text: str, duration: float, segments: tuple[int, int] | None) -> str | None:
    """Why a synthesis of ``text`` that pushed ``duration`` seconds of audio (and, streamed,
    ``segments`` as ``(expected, actual)``) failed, or ``None``.
    """
