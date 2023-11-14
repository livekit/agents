from typing import AsyncIterable, List
from livekit import rtc
from .async_iterator_list import AsyncIteratorList
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


STTPlugin = Plugin[List[rtc.AudioFrame], AsyncIterable[STTPluginResult]]


TTSPlugin = Plugin[AsyncIterable[str], AsyncIterable[rtc.AudioFrame]]
