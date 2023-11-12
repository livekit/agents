from typing import AsyncIterable
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


STTPlugin = Plugin[[rtc.AudioFrame], AsyncIterable[STTPluginResult]]


TTSPlugin = Plugin[AsyncIterable[str], AsyncIterable[rtc.AudioFrame]]


class CompleteSentencesPlugin(Plugin[AsyncIterable[str], AsyncIterable[str]]):
    def __init__(self) -> None:
        super().__init__(process=self._process, close=self._close)

    def _close(self):
        pass

    def _process(self, text_streams: AsyncIterable[AsyncIterable[str]]) -> AsyncIterable[AsyncIterable[str]]:
        async def iterator():
            async for text_stream in text_streams:
                running_sentence = ""
                async for chunk in text_stream:
                    if chunk.endswith('.') or chunk.endswith('?') or chunk.endswith('!'):
                        running_sentence += chunk
                        yield AsyncIteratorList([running_sentence])
                        running_sentence = ""
                    else:
                        running_sentence += chunk

                if len(running_sentence) > 0:
                    yield AsyncIteratorList([running_sentence])

        return iterator()
