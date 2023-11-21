from typing import AsyncIterable, List
from livekit import rtc
from .plugin import Plugin, PluginIterator
from dataclasses import dataclass
from enum import Enum


TTTPlugin = Plugin[str, str]


VADPluginResultType = Enum(
    'VADPluginResultType', ['STARTED', 'FINISHED'])


@dataclass
class VADPluginResult:
    type: VADPluginResultType
    frames: [rtc.AudioFrame]


VADPlugin = Plugin[AsyncIterable[rtc.AudioFrame], PluginIterator[VADPluginResult]]

STTPluginResultType = Enum('STTPluginResultType', ['DELTA_RESULT'])


@dataclass
class STTPluginResult:
    type: STTPluginResultType
    text: str


STTPlugin = Plugin[List[rtc.AudioFrame], PluginIterator[STTPluginResult]]


TTSPlugin = Plugin[AsyncIterable[str], PluginIterator[rtc.AudioFrame]]
