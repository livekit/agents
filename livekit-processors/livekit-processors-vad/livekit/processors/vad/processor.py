from livekit import agents
from .vad import VAD


class VADProcessor(agents.Processor):
    def __init__(self, silence_threshold_ms=250):
        self.vad = VAD(silence_threshold_ms=silence_threshold_ms)
        super().__init__(process=self.vad.push_frame)
