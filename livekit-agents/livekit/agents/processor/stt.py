from dataclasses import dataclass
from .processor import Processor
from livekit import rtc


class STTProcessor(Processor[[rtc.AudioFrame], "STTProcessor.Event"]):
    @dataclass
    class Event:
        text: str
