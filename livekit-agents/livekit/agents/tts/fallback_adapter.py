from .tts import TTS, ChunkedStream, SynthesizeStream


class FallbackAdapter(TTS):
    def __init__(self) -> None:
        pass


class FallbackChunkedStream(ChunkedStream):
    def __init__(self):
        super().__init__()


class FallbackSynthesizeStream(SynthesizeStream):
    def __init__(self):
        super().__init__()
