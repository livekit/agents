import asyncio
from typing import AsyncIterable, TypeVar, Callable
from livekit.agents.utils.async_queue_iterator import AsyncQueueIterator
from livekit import rtc
from .plugin import Plugin, PluginEventType
from dataclasses import dataclass
from enum import Enum


TTTPlugin = Plugin[str, str]


VADPluginEventType = Enum(
    'VADPluginEventType', ['STARTED', 'FINISHED'])


@dataclass
class VADPluginEvent:
    type: VADPluginEventType
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
