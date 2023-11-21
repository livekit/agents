import asyncio
from typing import AsyncIterable
from livekit.plugins.core.plugin import Plugin, PluginIterator


class CompleteSentencesPlugin(Plugin[AsyncIterable[str], AsyncIterable[str]]):
    async def close(self):
        pass

    async def process(self, text_stream: AsyncIterable[str]) -> AsyncIterable[str]:
        res = PluginIterator[str]()

        async def generate_sentences():
            running_sentence = ""
            async for chunk in text_stream:
                if chunk.endswith('.') or chunk.endswith(
                        '?') or chunk.endswith('!'):
                    running_sentence += chunk
                    await res.put(running_sentence)
                    running_sentence = ""
                else:
                    running_sentence += chunk

            if len(running_sentence) > 0:
                await res.put(running_sentence)
            
            await res.aclose()

        asyncio.create_task(generate_sentences())
        return res
