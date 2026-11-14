# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

from livekit.agents import ProviderAttempts
from livekit.agents.tts import TTS, AudioEmitter

from .connect import APIConnectOptions
from .expressive import SpeechSteeringOptions
from .stream_pacer import SentenceStreamPacer
from .tts import (
    MarkupInfo,
    MarkupSplit,
    OptionsTokenizer,
    SegmentCount,
    SpeechChunk,
    SynthesizedAudio,
    TTSCapabilities,
)
from .tts_segments import SegmentInput, SegmentOutput

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

class TtsSegments:
    """`_core.TtsSegments`: one attempt's segments. The input as `encode` takes it (a chunk opens a
    segment when none is open, a flush ends it), each segment's audio played in order into the
    emitter (a later segment's waits for the ones before it) and completed once, and the read
    deadline, run while a segment whose text ended is unanswered.
    """
    def __new__(cls, emitter: AudioEmitter, deadline: ProviderAttempts, label: str) -> Self:
        """`label` names the provider in a protocol failure."""
    def push(self, chunk: SpeechChunk) -> None:
        """A chunk of the input."""
    def flush(self) -> None:
        """A flush: the open segment's text ended."""
    def next(self) -> SegmentInput | None:
        """The next input `encode` takes; `None` while none is queued."""
    def deliver(self, output: SegmentOutput) -> None:
        """What `on_message` returned for a segment."""
    def complete_ended(self) -> None:
        """The provider finished the session the segments whose text ended were sent on: each is
        done.
        """
    def holds(self, segment_id: str) -> bool:
        """Whether `segment_id` is a segment not yet done: a multiplexed stream's context."""
    @property
    def open(self) -> list[str]:
        """The segments not yet done, in order."""
    async def settled(self) -> None:
        """Waits until every segment is done."""

class TtsStreamAdapter:
    """`tts.StreamAdapter`'s engine: a non-streaming TTS as a streaming one, each chunk of its input
    synthesized on its own.
    """
    def __init__(self, *, tts: TTS[Any]) -> None:
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
        `conn_options`; the first opens when its audio is first read. Its requests carry the
        identity of the job `cx` runs for.
        """
    @property
    def native(self) -> bool:
        """Whether its TTS runs natively; a host plugin's own attempts report their metrics."""
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
    def __new__(cls, sentence_tokenizer: OptionsTokenizer, capabilities: TTSCapabilities) -> Self:
        """Split by `sentence_tokenizer`, paced as a TTS with `capabilities` is by default."""
    @property
    def sentence_tokenizer(self) -> object | None:
        """The tokenizer as it was given; `None` only for the options of a TTS no host built."""
    @property
    def pacing(self) -> SentenceStreamPacer | None: ...
    def set(self, sentence_tokenizer: OptionsTokenizer, pacing: SentenceStreamPacer | None) -> None:
        """Replace both at once."""
