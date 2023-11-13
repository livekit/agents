from typing import AsyncIterable
from livekit.plugins.core.plugin import Plugin
from livekit.plugins.core.async_iterator_list import AsyncIteratorList

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