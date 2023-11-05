import asyncio
from typing import AsyncIterable, TypeVar, Callable
from livekit.agents.utils.async_queue_iterator import AsyncQueueIterator
from livekit import rtc
from .processor import Processor, ProcessorEventType
from dataclasses import dataclass
from enum import Enum


TextToTextProcessor = Processor[str, str]


VoiceActivityDetectionProcessorEventType = Enum(
    'VoiceActivityDetectionProcessorEventType', ['STARTED', 'FINISHED'])


@dataclass
class VoiceActivityDetectionProcessorEvent:
    type: VoiceActivityDetectionProcessorEventType
    frames: [rtc.AudioFrame]


VoiceActivityDetectionProcessor = Processor[rtc.AudioFrame,
                                            VoiceActivityDetectionProcessorEvent]

SpeechToTextProcessorEventType = Enum(
    'VoiceActivityDetectionProcessorEventType', ['DELTA_RESULT'])


@dataclass
class SpeechToTextProcessorEvent:
    type: SpeechToTextProcessorEventType
    frames: [rtc.AudioFrame]


SpeechToTextProcessor = Processor[rtc.AudioFrame, SpeechToTextProcessorEvent]


TextToSpeechProcessor = Processor[str, rtc.AudioFrame]
