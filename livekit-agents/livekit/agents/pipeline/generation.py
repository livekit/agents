from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import (
    AsyncIterable,
    Protocol,
    Tuple,
    runtime_checkable,
)

from livekit import rtc

from .. import utils
from ..llm import (
    ChatChunk,
    ChatContext,
    FunctionContext,
)
from ..log import logger
from ..utils import aio
from . import io


@runtime_checkable
class _ACloseable(Protocol):
    async def aclose(self): ...


@dataclass
class _LLMGenerationData:
    text_ch: aio.Chan[str]
    tools_ch: aio.Chan[FunctionCallInfo]
    generated_text: str = ""
    generated_tools: list[FunctionCallInfo] = field(default_factory=list)


def perform_llm_inference(
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
                                data.generated_tools.append(tool)
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


def perform_tts_inference(
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


@dataclass
class _TextOutput:
    text: str


def perform_text_forwarding(
    *, text_output: io.TextSink, llm_output: AsyncIterable[str]
) -> tuple[asyncio.Task, _TextOutput]:
    out = _TextOutput(text="")
    task = asyncio.create_task(_text_forwarding_task(text_output, llm_output, out))
    return task, out


@utils.log_exceptions(logger=logger)
async def _text_forwarding_task(
    text_output: io.TextSink, llm_output: AsyncIterable[str], out: _TextOutput
) -> None:
    try:
        async for delta in llm_output:
            out.text += delta
            await text_output.capture_text(delta)
    finally:
        text_output.flush()


@dataclass
class _AudioOutput:
    audio: list[rtc.AudioFrame]


def perform_audio_forwarding(
    *,
    audio_output: io.AudioSink,
    tts_output: AsyncIterable[rtc.AudioFrame],
) -> tuple[asyncio.Task, _AudioOutput]:
    out = _AudioOutput(audio=[])
    task = asyncio.create_task(_audio_forwarding_task(audio_output, tts_output, out))
    return task, out


@utils.log_exceptions(logger=logger)
async def _audio_forwarding_task(
    audio_output: io.AudioSink,
    tts_output: AsyncIterable[rtc.AudioFrame],
    out: _AudioOutput,
) -> None:
    try:
        async for frame in tts_output:
            out.audio.append(frame)
            await audio_output.capture_frame(frame)
    finally:
        audio_output.flush()
