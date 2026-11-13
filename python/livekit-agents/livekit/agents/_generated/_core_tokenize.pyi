# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Self

from .tokenizer import ReplacedWords, TokenizerToken, TokenSpan
from .tool_context import RawFunctionDescription
from .tool_search import SearchFields, ToolProxyCall

def split_words(
    text: str,
    *,
    ignore_punctuation: bool = True,
    retain_format: bool = False,
) -> list[TokenSpan]:
    """The words of `text`, each with its character offsets: what `WordTokenizer` splits."""

def replace_words(
    text: str,
    replacements: dict[str, str],
    hold_last_word: bool = False,
) -> ReplacedWords | None: ...

class SpeechSentenceTokenizer:
    """`tokenize.SentenceTokenizer`'s engine."""
    def __init__(
        self,
        *,
        min_words: int = 4,
        clause_words: int | None = None,
        retain_format: bool = False,
        max_token_len: int | None = None,
    ) -> None: ...
    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        """The sentences of `text`, batched as the stream batches them."""
    def _core_stream(self) -> SpeechSentenceStream: ...
    @property
    def _core_max_token_len(self) -> int | None: ...

class SpeechSentenceStream:
    """`tokenize.SentenceTokenizer`'s stream: a sentence goes out once no text to come could
    change it.
    """
    def push(self, text: str) -> list[str]:
        """The sentences ``text`` completes."""
    def flush(self) -> list[str]: ...

class SpeechWordTokenizer:
    """`tokenize.WordTokenizer`'s engine."""
    def __init__(self, *, ignore_punctuation: bool = True, retain_format: bool = False) -> None: ...
    def tokenize(self, text: str, *, language: str | None = None) -> list[str]: ...
    def _core_stream(self) -> SpeechWordStream: ...

class SpeechWordStream:
    """`tokenize.WordTokenizer`'s stream: a word goes out once the next one proves it whole."""
    def push(self, text: str) -> list[str]:
        """The words ``text`` completes."""
    def flush(self) -> list[str]: ...

class TokenBuffer:
    def __new__(
        cls,
        tokenize_fnc: Callable[[str], Iterable[TokenizerToken]],
        *,
        min_token_len: int,
        min_ctx_len: int,
        max_token_len: int | None = None,
    ) -> Self: ...
    def push(self, text: str) -> list[str]:
        """The tokens ``text`` proves complete, batched."""
    def flush(self) -> list[str]: ...

def normalize_language(code: str) -> str:
    """Normalise a language code or name to BCP-47."""

def base_language(code: str) -> str:
    """Base language code (ISO 639-1 when possible).

    E.g., ``'en'`` from ``'en-US'``, ``'zh'`` from ``'cmn-Hans-CN'``.
    """

def language_region(code: str) -> str | None:
    """Region code, e.g., ``'US'`` from ``'en-US'``, or ``None``."""

def language_iso(code: str) -> str:
    """ISO 639-1 tag with region, e.g., ``'zh-CN'`` from ``'cmn-Hans-CN'``."""

def language_name(code: str) -> str | None:
    """Return the English language name (e.g. ``'english'``), or ``None`` if unknown."""

class SearchDocument:
    """One tool as the search sees it; identity stays with the host."""
    def __new__(cls, item: SearchFields) -> Self: ...
    def keyword_score(self, terms: Sequence[str]) -> float: ...

class BM25Index:
    def __new__(cls, k1: float, b: float) -> Self: ...
    def build(self, documents: Sequence[SearchDocument]) -> None: ...
    def clear(self) -> None: ...
    def score(self, doc: SearchDocument, terms: Sequence[str]) -> float: ...
    def search(self, query: str, docs: Sequence[SearchDocument], limit: int) -> list[int]: ...

def keyword_search(query: str, docs: Sequence[SearchDocument], limit: int) -> list[int]: ...

def tool_search_validate(query: str) -> None: ...

def tool_search_loaded(query: str, count: int) -> str: ...

def tool_search_schema(search_description: str, query_description: str) -> RawFunctionDescription:
    """`ToolSearchToolset`'s `tool_search` raw schema."""

def call_tool_schema(description: str) -> RawFunctionDescription:
    """`ToolProxyToolset`'s `call_tool` raw schema."""

def tool_proxy_arguments(arguments: object, known: Sequence[str]) -> ToolProxyCall: ...
