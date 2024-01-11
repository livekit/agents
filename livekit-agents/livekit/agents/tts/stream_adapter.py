from .tts import (
    TTS,
    SynthesizeStream,
    SynthesisEvent,
    SynthesizedAudio,
)
from ..tokenize import SentenceTokenizer, SentenceStream


class StreamAdapterWrapper(SynthesizeStream):
    def __init__(self, tts: TTS, sentence_stream: SentenceStream) -> None:
        super().__init__()
        self._closed = False

        self._text = ""

    def push_text(self, token: str) -> None:
        self._text += token

        # Divide the text into sentences and push them to the queue

    async def flush(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def __anext__(self) -> SynthesisEvent:
        raise StopAsyncIteration


class StreamAdapter(TTS):
    def __init__(self, tts: TTS, tokenizer: SentenceTokenizer) -> None:
        super().__init__(streaming_supported=True)
        self._tts = tts
        self._tokenizer = tokenizer

    async def synthesize(self, *, text: str) -> SynthesizedAudio:
        return await self._tts.synthesize(text=text)

    def stream(self) -> SynthesizeStream:
        return StreamAdapterWrapper(self._tts, self._tokenizer.stream())
