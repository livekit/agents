import asyncio
from typing import AsyncIterable, TypeVar, Callable
from livekit.agents.utils.async_queue_iterator import AsyncQueueIterator
from livekit import rtc
from .processor import Processor, ProcessorEventType
from dataclasses import dataclass
from enum import Enum


TTTProcessor = Processor[str, str]


VADProcessorEventType = Enum(
    'VADProcessorEventType', ['STARTED', 'FINISHED'])


@dataclass
class VADProcessorEvent:
    type: VADProcessorEventType
    frames: [rtc.AudioFrame]


VADProcessor = Processor[rtc.AudioFrame,
                         VADProcessorEvent]

STTProcessorEventType = Enum(
    'STTProcessorEventType', ['DELTA_RESULT'])


@dataclass
class STTProcessorEvent:
    type: STTProcessorEventType
    text: str


STTProcessor = Processor[rtc.AudioFrame, STTProcessorEvent]


TTSProcessor = Processor[str, rtc.AudioFrame]
