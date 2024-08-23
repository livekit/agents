from typing import Callable

from livekit.agents import tokenize


class SentenceChunker:
    def __init__(
        self,
        *,
        max_chunk_size: int = 120,
        chunk_overlap: int = 30,
        paragraph_tokenizer: Callable[
            [str], list[str]
        ] = tokenize.basic.tokenize_paragraphs,
        sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(
            ignore_punctuation=False
        ),
    ) -> None:
        self._max_chunk_size = max_chunk_size
        self._chunk_overlap = chunk_overlap
        self._paragraph_tokenizer = paragraph_tokenizer
        self._sentence_tokenizer = sentence_tokenizer
        self._word_tokenizer = word_tokenizer

    def chunk(self, *, text: str) -> list[str]:
        chunks = []

        buf_words: list[str] = []
        for paragraph in self._paragraph_tokenizer(text):
            last_buf_words: list[str] = []

            for sentence in self._sentence_tokenizer.tokenize(text=paragraph):
                for word in self._word_tokenizer.tokenize(text=sentence):
                    reconstructed = self._word_tokenizer.format_words(
                        buf_words + [word]
                    )

                    if len(reconstructed) > self._max_chunk_size:
                        while (
                            len(self._word_tokenizer.format_words(last_buf_words))
                            > self._chunk_overlap
                        ):
                            last_buf_words = last_buf_words[1:]

                        new_chunk = self._word_tokenizer.format_words(
                            last_buf_words + buf_words
                        )
                        chunks.append(new_chunk)
                        last_buf_words = buf_words
                        buf_words = []

                    buf_words.append(word)

            if buf_words:
                while (
                    len(self._word_tokenizer.format_words(last_buf_words))
                    > self._chunk_overlap
                ):
                    last_buf_words = last_buf_words[1:]

                new_chunk = self._word_tokenizer.format_words(
                    last_buf_words + buf_words
                )
                chunks.append(new_chunk)
                buf_words = []

        return chunks
