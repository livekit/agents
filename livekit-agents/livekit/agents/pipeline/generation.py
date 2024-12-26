from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterable

from ..llm import ChatChunk, ChatContext, FunctionCallInfo, FunctionContext
from ..utils import aio
from . import io


@dataclass
class _LLMGenerationData:
    chat_ctx: ChatContext
    fnc_ctx: FunctionContext | None
    text_ch: aio.Chan[str]
    tools_ch: aio.Chan[FunctionCallInfo]


async def do_llm_inference(*, node: io.LLMNode, data: _LLMGenerationData) -> bool:
    llm_node = node(data.chat_ctx, data.fnc_ctx)
    if asyncio.iscoroutine(llm_node):
        llm_node = await llm_node

    if isinstance(llm_node, str):
        data.text_ch.send_nowait(llm_node)
        return True

    if isinstance(llm_node, AsyncIterable):
        # forward llm stream to output channels
        async for chunk in llm_node:
            # io.LLMNode can either return a string or a ChatChunk
            if isinstance(chunk, str):
                data.text_ch.send_nowait(chunk)

            elif isinstance(chunk, ChatChunk):
                if not chunk.choices:
                    continue  # this can happens if we receive the stats chunk

                delta = chunk.choices[0].delta

                if delta.tool_calls:
                    for tool in delta.tool_calls:
                        data.tools_ch.send_nowait(tool)

                if delta.content:
                    data.text_ch.send_nowait(delta.content)

        return True

    return False


@dataclass
class _TTSGenerationData:
    input_ch: AsyncIterable[str]
    audio_ch: aio.Chan[bytes]

async def do_tts_inference(*, node: io.TTSNode, data: _TTSGenerationData) -> bool:
    tts_node = node(data.input_ch)


    return False
