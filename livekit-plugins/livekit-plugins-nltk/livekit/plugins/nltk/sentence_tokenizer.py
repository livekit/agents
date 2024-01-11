from typing import List
from livekit import agents
from dataclasses import dataclass
import dataclasses
from typing import Optional
import asyncio
import nltk

# nltk is using the punkt tokenizer
# https://www.nltk.org/_modules/nltk/tokenize/punkt.html
# this code is using a whitespace to concatenate small sentences together
# (languages such as Chinese and Japanese are not yet supported)


@dataclass
class TokenizerOptions:
    language: str
    min_sentence_len: int
    stream_context_len: int


class SentenceTokenizer(agents.tokenize.SentenceTokenizer):
    def __init__(
        self,
        language: str = "en-US",
        min_sentence_len: int = 20,
        stream_context_len: int = 10,
    ) -> None:
        super().__init__()
        self._config = TokenizerOptions(
            language=language,
            min_sentence_len=min_sentence_len,
            stream_context_len=stream_context_len,
        )

    def _sanitize_options(self, language: Optional[str] = None) -> TokenizerOptions:
        config = dataclasses.replace(self._config)
        if language:
            config.language = language
        return config

    def tokenize(
        self, *, text: str, language: Optional[str] = None
    ) -> List[agents.tokenize.SegmentedSentence]:
        config = self._sanitize_options(language=language)
        sentences = nltk.tokenize.sent_tokenize(text, config.language)
        new_sentences = []
        buff = ""
        for sentence in sentences:
            buff += sentence + " "
            if len(buff) - 1 >= config.min_sentence_len:
                new_sentences.append(buff.rstrip())
                buff = ""

        if buff:
            new_sentences.append(buff.rstrip())

        return [agents.tokenize.SegmentedSentence(text=text) for text in new_sentences]

    def stream(
        self,
        *,
        language: Optional[str] = None,
    ) -> agents.tokenize.SentenceStream:
        config = self._sanitize_options(language=language)
        return SentenceStream(
            language=config.language,
            min_sentence_len=config.min_sentence_len,
            context_len=config.stream_context_len,
        )


class SentenceStream(agents.tokenize.SentenceStream):
    def __init__(
        self, *, language: str, min_sentence_len: int, context_len: int
    ) -> None:
        self._language = language
        self._context_len = context_len
        self._min_sentence_len = min_sentence_len
        self._event_queue = asyncio.Queue()

        self._incomplete_sentences = []  # <= min_sentence_len
        self._buffer = ""

    def push_text(self, text: str) -> None:
        for char in text:
            self._buffer += char

            if len(self._buffer) < self._context_len:
                continue

            sentences = nltk.tokenize.sent_tokenize(self._buffer, self._language)
            if len(sentences) < 2:
                continue

            new_sentence = sentences[0]
            self._incomplete_sentences.append(new_sentence)
            s = " ".join(self._incomplete_sentences)

            if len(s) >= self._min_sentence_len:
                self._event_queue.put_nowait(agents.tokenize.SegmentedSentence(text=s))
                self._incomplete_sentences = []

            self._buffer = self._buffer[len(new_sentence) :].lstrip()

    async def flush(self) -> None:
        # try to segment the remaining data inside self._text_buffer
        buff = " ".join(self._incomplete_sentences)
        sentences = nltk.tokenize.sent_tokenize(self._buffer, self._language)
        for sentence in sentences:
            buff += " " + sentence
            if len(buff) >= self._min_sentence_len:
                await self._event_queue.put(
                    agents.tokenize.SegmentedSentence(text=buff)
                )
                buff = ""

        if buff:
            await self._event_queue.put(agents.tokenize.SegmentedSentence(text=buff))

    async def __anext__(self) -> agents.tokenize.SegmentedSentence:
        if self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()

    def __aiter__(self) -> "SentenceStream":
        return self
