from __future__ import annotations

from dataclasses import dataclass

from blingfire import text_to_sentences  # type: ignore
from livekit import agents

# nltk is using the punkt tokenizer
# https://www.nltk.org/_modules/nltk/tokenize/punkt.html
# this code is using a whitespace to concatenate small sentences together
# (languages such as Chinese and Japanese are not yet supported)


@dataclass
class _TokenizerOptions:
    min_sentence_len: int
    stream_context_len: int


class SentenceTokenizer(agents.tokenize.SentenceTokenizer):
    def __init__(
        self,
        *,
        min_sentence_len: int = 20,
        stream_context_len: int = 10,
    ) -> None:
        super().__init__()
        self._config = _TokenizerOptions(
            min_sentence_len=min_sentence_len,
            stream_context_len=stream_context_len,
        )

    def tokenize(self, text: str) -> list[str]:
        new_sentences = []
        buff = ""
        for sentence in self._tokenize(text):
            buff += sentence + " "
            if len(buff) - 1 >= self._config.min_sentence_len:
                new_sentences.append(buff.rstrip())
                buff = ""

        if buff:
            new_sentences.append(buff.rstrip())

        return new_sentences

    def _tokenize(self, text: str) -> list[str]:
        sentences: str = text_to_sentences(text)
        return sentences.split("\n")

    def stream(self) -> agents.tokenize.SentenceStream:
        return agents.tokenize.BufferedSentenceStream(
            tokenizer=self._tokenize,
            min_token_len=self._config.min_sentence_len,
            min_ctx_len=self._config.stream_context_len,
        )
