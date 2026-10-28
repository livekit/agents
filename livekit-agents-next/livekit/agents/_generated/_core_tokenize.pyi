# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from .tokenizer import TokenizerToken

def split_words(
    text: str,
    *,
    ignore_punctuation: bool = True,
    split_character: bool = False,
    retain_format: bool = False,
) -> list[tuple[str, int, int]]: ...

def basic_split_sentences(
    text: str,
    min_sentence_len: int = 20,
    *,
    retain_format: bool = False,
) -> list[tuple[str, int, int]]: ...

def replace_words(
    text: str,
    replacements: dict[str, str],
    hold_last_word: bool = False,
) -> tuple[str, int] | None: ...

def split_paragraphs(text: str) -> list[tuple[str, int, int]]: ...

def tokenize_paragraphs(text: str) -> list[str]: ...

def xml_has_unclosed_tags(text: str) -> bool: ...

def strip_thinking(
    content: str | None,
    start_tag: str,
    end_tag: str,
    buffer: str,
    end_marker: str | None,
    is_final: bool,
) -> tuple[str | None, str, str | None]: ...

def flatten_delta_text(parts: Sequence[str]) -> str | None: ...

def xml_tokenize(
    tokenize_fnc: Callable[[str], Iterable[TokenizerToken]],
    text: str,
) -> list[tuple[str, int, int]]: ...

class BasicWordTokenizer:
    """Really naive implementation of a word tokenizer: words split on whitespace, punctuation
    stripped unless asked to keep it.
    """
    def __init__(
        self,
        *,
        ignore_punctuation: bool = True,
        split_character: bool = False,
        retain_format: bool = False,
    ) -> None: ...
    def tokenize(self, text: str, *, language: str | None = None) -> list[str]: ...
    def _core_split(self, text: str) -> list[tuple[str, int, int]]: ...
    @property
    def _core_min_token_len(self) -> int: ...
    @property
    def _core_min_ctx_len(self) -> int: ...

class BasicSentenceTokenizer:
    """Really naive implementation of a sentence tokenizer: rule-based, and only English is really
    tested.
    """
    def __init__(
        self,
        *,
        language: str = "english",
        min_sentence_len: int = 20,
        stream_context_len: int = 10,
        retain_format: bool = False,
    ) -> None: ...
    def tokenize(self, text: str, *, language: str | None = None) -> list[str]: ...
    def _core_split(self, text: str) -> list[tuple[str, int, int]]: ...
    @property
    def _core_min_token_len(self) -> int: ...
    @property
    def _core_min_ctx_len(self) -> int: ...

class TokenBuffer:
    def __init__(
        self,
        tokenize_fnc: Callable[[str], Iterable[TokenizerToken]],
        *,
        min_token_len: int,
        min_ctx_len: int,
        max_token_len: int | None = None,
        xml_aware: bool = False,
    ) -> None: ...
    def push(self, text: str) -> list[str]:
        """The tokens ``text`` proves complete, batched."""
    def flush(self) -> list[str]: ...

class SearchDocument:
    """One tool as the search sees it; identity stays with the host."""
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Sequence[tuple[str, str]],
    ) -> None: ...
    def keyword_score(self, terms: Sequence[str]) -> float: ...

class BM25Index:
    def __init__(self, k1: float, b: float) -> None: ...
    def build(self, documents: Sequence[SearchDocument]) -> None: ...
    def clear(self) -> None: ...
    def score(self, doc: SearchDocument, terms: Sequence[str]) -> float: ...
    def search(self, query: str, docs: Sequence[SearchDocument], limit: int) -> list[int]: ...

def keyword_search(query: str, docs: Sequence[SearchDocument], limit: int) -> list[int]: ...

def tool_search_validate(query: str) -> None: ...

def tool_search_loaded(query: str, count: int) -> str: ...

def tool_proxy_arguments(arguments: object, known: Sequence[str]) -> tuple[str, object]: ...
