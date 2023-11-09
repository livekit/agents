import asyncio
from typing import AsyncIterable, TypeVar, Callable
from livekit import rtc
from .async_queue_iterator import AsyncQueueIterator
from .plugin import Plugin 
from dataclasses import dataclass
from enum import Enum


TTTPlugin = Plugin[str, str]


VADPluginResultType = Enum(
    'VADPluginEventType', ['STARTED', 'FINISHED'])


@dataclass
class VADPluginEvent:
    type: VADPluginResultType
    frames: [rtc.AudioFrame]


VADPlugin = Plugin[rtc.AudioFrame,
                   VADPluginEvent]

STTPluginEventType = Enum(
    'STTPluginEventType', ['DELTA_RESULT'])


@dataclass
class STTPluginEvent:
    type: STTPluginEventType
    text: str


STTPlugin = Plugin[rtc.AudioFrame, STTPluginEvent]


TTSPlugin = Plugin[AsyncIterable[str], AsyncIterable[rtc.AudioFrame]]
