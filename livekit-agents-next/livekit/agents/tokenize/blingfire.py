from __future__ import annotations

from dataclasses import dataclass

from .. import _core
from . import BufferedSentenceStream, SentenceStream, SentenceTokenizer as _SentenceTokenizer

__all__ = [
    "SentenceTokenizer",
]


@dataclass
class _TokenizerOptions:
    min_sentence_len: int
    stream_context_len: int
    retain_format: bool
    max_token_len: int | None
    min_token_len: int | None
    xml_aware: bool


class SentenceTokenizer(_SentenceTokenizer):
    def __init__(
        self,
        *,
        min_sentence_len: int = 20,
        stream_context_len: int = 10,
        retain_format: bool = False,
        max_token_len: int | None = None,
        min_token_len: int | None = None,
        xml_aware: bool = False,
    ) -> None:
        """
        Args:
            min_sentence_len: minimum length for a span to be treated as its own
                sentence; shorter spans are merged forward into the next one.
            stream_context_len: minimum buffered text before the stream emits.
            retain_format: keep original whitespace/formatting in emitted tokens.
            max_token_len: hard cap on emitted token length; a token is flushed
                before appending a sentence that would exceed it.
            min_token_len: minimum length a token must reach before it is emitted.
                Sentences are batched together until the running token reaches this
                length, so raising it (e.g. toward ``max_token_len``) yields larger,
                fewer chunks. Defaults to ``min_sentence_len`` (per-sentence emission).
            xml_aware: treat XML markup as atomic — never split a tag across tokens
                and keep tags attached to the following sentence. Only enable when
                the input actually carries markup (e.g. expressive TTS): a stray "<"
                in plain text can otherwise hold back streaming until flush.
        """
        self._config = _TokenizerOptions(
            min_sentence_len=min_sentence_len,
            stream_context_len=stream_context_len,
            retain_format=retain_format,
            max_token_len=max_token_len,
            min_token_len=min_token_len,
            xml_aware=xml_aware,
        )

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        tokenize_fnc = self._split
        tokens = (
            _core.xml_tokenize(tokenize_fnc, text) if self._config.xml_aware else tokenize_fnc(text)
        )
        return [tok[0] for tok in tokens]

    def _split(self, text: str) -> list[tuple[str, int, int]]:
        return _core.split_sentences(
            text, self._config.min_sentence_len, retain_format=self._config.retain_format
        )

    def stream(self, *, language: str | None = None) -> SentenceStream:
        return BufferedSentenceStream(
            tokenizer=self._split,
            max_token_len=self._config.max_token_len,
            min_token_len=(
                self._config.min_token_len
                if self._config.min_token_len is not None
                else self._config.min_sentence_len
            ),
            min_ctx_len=self._config.stream_context_len,
            xml_aware=self._config.xml_aware,
        )
