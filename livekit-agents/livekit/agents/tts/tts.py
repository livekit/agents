from abc import ABC, abstractmethod
from livekit import rtc
from dataclasses import dataclass


@dataclass
class SynthesizedAudio:
    text: str
    data: rtc.AudioFrame


@dataclass
class SynthesisEvent:
    audio: SynthesizedAudio


@dataclass
class SynthesisOptions:
    pass


class TranscriptionStream(ABC):
    @abstractmethod
    def push_token(self, token: str):
        pass

    @abstractmethod
    async def flush(self) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    async def __anext__(self) -> SynthesisEvent:
        pass

    def __aiter__(self) -> "TranscriptionStream":
        return self


class TTS(ABC):
    def __init__(self, *, streaming_supported=False) -> None:
        self._streaming_supported = streaming_supported

    @abstractmethod
    async def synthesize(
        self, text: str, opts: SynthesisOptions = SynthesisOptions()
    ) -> SynthesizedAudio:
        pass

    def stream(
        self, opts: SynthesisOptions = SynthesisOptions()
    ) -> TranscriptionStream:
        raise NotImplementedError(
            "streaming is not supported by this TTS, please use a different TTS or use a StreamAdapter"
        )

    @property
    def streaming_supported(self):
        return self._streaming_supported
