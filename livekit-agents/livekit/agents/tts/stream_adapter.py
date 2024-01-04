from .tts import (
    TTS,
    TranscriptionStream,
    SynthesisEvent,
    SynthesizedAudio,
    SynthesisOptions,
)


class StreamAdapter(TTS):
    def __init__(self, tts: TTS) -> None:
        super().__init__(streaming_supported=True)
        self._tts = tts

    async def synthesize(
        self, text: str, opts: SynthesisOptions = SynthesisOptions()
    ) -> SynthesizedAudio:
        return await self._tts.synthesize(text, opts)

    def stream(
        self, opts: SynthesisOptions = SynthesisOptions()
    ) -> TranscriptionStream:
        return StreamAdapterWrapper()


class StreamAdapterWrapper(TranscriptionStream):
    def __init__(self) -> None:
        super().__init__()
        self._closed = False

        self._text = ""

    def push_token(self, token: str):
        self._text += token

        # Divide the text into sentences and push them to the queue

    async def flush(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def __anext__(self) -> SynthesisEvent:
        raise StopAsyncIteration
