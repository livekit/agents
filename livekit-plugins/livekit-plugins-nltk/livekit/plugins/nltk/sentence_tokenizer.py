from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import nltk  # type: ignore[import-untyped]  # NLTK does not ship a py.typed marker.
from livekit import agents
from livekit.agents import LanguageCode

# nltk is using the punkt tokenizer
# https://www.nltk.org/_modules/nltk/tokenize/punkt.html
# this code is using a whitespace to concatenate small sentences together
# (languages such as Chinese and Japanese are not yet supported)


# NLTK does not annotate sent_tokenize; its public API returns sentences as strings.
_sent_tokenize = cast(Callable[[str, str], list[str]], nltk.tokenize.sent_tokenize)  # pyright: ignore[reportUnknownMemberType]


@dataclass
class _TokenizerOptions:
    language: str
    min_sentence_len: int
    stream_context_len: int


class SentenceTokenizer(agents.tokenize.SentenceTokenizer):
    def __init__(
        self,
        *,
        language: str = "english",
        min_sentence_len: int = 20,
        stream_context_len: int = 10,
    ) -> None:
        super().__init__()
        self._config = _TokenizerOptions(
            language=language,
            min_sentence_len=min_sentence_len,
            stream_context_len=stream_context_len,
        )

    def _sanitize_options(self, language: str | None = None) -> _TokenizerOptions:
        config = dataclasses.replace(self._config)
        if language:
            lang = LanguageCode(language)
            config.language = lang.to_language_name() or language
        return config

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        config = self._sanitize_options(language=language)
        sentences = _sent_tokenize(text, config.language)
        new_sentences: list[str] = []
        buff = ""
        for sentence in sentences:
            buff += sentence + " "
            if len(buff) - 1 >= config.min_sentence_len:
                new_sentences.append(buff.rstrip())
                buff = ""

        if buff:
            new_sentences.append(buff.rstrip())

        return new_sentences

    def stream(self, *, language: str | None = None) -> agents.tokenize.SentenceStream:
        config = self._sanitize_options(language=language)
        return agents.tokenize.BufferedSentenceStream(
            tokenizer=lambda text: _sent_tokenize(text, config.language),
            min_token_len=self._config.min_sentence_len,
            min_ctx_len=self._config.stream_context_len,
        )
