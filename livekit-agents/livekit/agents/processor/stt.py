from dataclasses import dataclass
from .processor import Processor
from typing import AsyncIterator
from livekit import rtc


class STTProcessor(Processor[[rtc.AudioFrame], AsyncIterator["STTProcessor.Event"]]):
    @dataclass
    class Event:
        text: str
