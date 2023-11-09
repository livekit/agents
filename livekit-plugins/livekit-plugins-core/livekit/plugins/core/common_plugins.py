import asyncio
from typing import AsyncIterable, TypeVar, Callable
from livekit import rtc
from .async_queue_iterator import AsyncQueueIterator
from .plugin import Plugin 
from dataclasses import dataclass
from enum import Enum


TTTPlugin = Plugin[str, str]


VADPluginResultType = Enum(
    'VADPluginResultType', ['STARTED', 'FINISHED'])


@dataclass
class VADPluginResult:
    type: VADPluginResultType
    frames: [rtc.AudioFrame]


VADPlugin = Plugin[rtc.AudioFrame,
                   VADPluginResult]

STTPluginResultType = Enum(
    'STTPluginResultType', ['DELTA_RESULT'])


@dataclass
class STTPluginResult:
    type: STTPluginResultType
    text: str


STTPlugin = Plugin[rtc.AudioFrame, STTPluginResult]


TTSPlugin = Plugin[AsyncIterable[str], AsyncIterable[rtc.AudioFrame]]
