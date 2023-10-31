from dataclasses import dataclass
from processor import Processor
from livekit import rtc


class TTSProcessor(Processor):
    @dataclass
    class Event:
        audio_frames: [rtc.AudioFrame]
