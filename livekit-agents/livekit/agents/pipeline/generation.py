from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterable, Protocol, Tuple, runtime_checkable
from livekit import rtc

from ..llm import ChatChunk, ChatContext, FunctionCallInfo, FunctionContext
from ..utils import aio
from . import io


@runtime_checkable
class _ACloseable(Protocol):
    async def aclose(self): ...


@dataclass
class _LLMGenerationData:
    text_ch: aio.Chan[str]
    tools_ch: aio.Chan[FunctionCallInfo]
    generated_text: str = ""  #


def do_llm_inference(
    *, node: io.LLMNode, chat_ctx: ChatContext, fnc_ctx: FunctionContext | None
) -> Tuple[asyncio.Task, _LLMGenerationData]:
    text_ch = aio.Chan()
    tools_ch = aio.Chan()

    data = _LLMGenerationData(text_ch=text_ch, tools_ch=tools_ch)

    async def _inference_task():
        llm_node = node(chat_ctx, fnc_ctx)
        if asyncio.iscoroutine(llm_node):
            llm_node = await llm_node

        if isinstance(llm_node, str):
            data.generated_text = llm_node
            text_ch.send_nowait(llm_node)
            return True

        if isinstance(llm_node, AsyncIterable):
            # forward llm stream to output channels
            try:
                async for chunk in llm_node:
                    # io.LLMNode can either return a string or a ChatChunk
                    if isinstance(chunk, str):
                        data.generated_text += chunk
                        text_ch.send_nowait(chunk)

                    elif isinstance(chunk, ChatChunk):
                        if not chunk.choices:
                            continue  # this can happens if we receive the stats chunk

                        delta = chunk.choices[0].delta

                        if delta.tool_calls:
                            for tool in delta.tool_calls:
                                tools_ch.send_nowait(tool)

                        if delta.content:
                            data.generated_text += delta.content
                            text_ch.send_nowait(delta.content)
            finally:
                if isinstance(llm_node, _ACloseable):
                    await llm_node.aclose()

            return True

        return False

    llm_task = asyncio.create_task(_inference_task())
    llm_task.add_done_callback(lambda _: text_ch.close())
    llm_task.add_done_callback(lambda _: tools_ch.close())

    return llm_task, data


@dataclass
class _TTSGenerationData:
    audio_ch: aio.Chan[rtc.AudioFrame]


def do_tts_inference(
    *, node: io.TTSNode, input: AsyncIterable[str]
) -> Tuple[asyncio.Task, _TTSGenerationData]:
    audio_ch = aio.Chan[rtc.AudioFrame]()

    async def _inference_task():
        tts_node = node(input)
        if asyncio.iscoroutine(tts_node):
            tts_node = await tts_node

        if isinstance(tts_node, AsyncIterable):
            async for audio_frame in tts_node:
                audio_ch.send_nowait(audio_frame)

            return True

        return False

    tts_task = asyncio.create_task(_inference_task())
    tts_task.add_done_callback(lambda _: audio_ch.close())

    return tts_task, _TTSGenerationData(audio_ch=audio_ch)
