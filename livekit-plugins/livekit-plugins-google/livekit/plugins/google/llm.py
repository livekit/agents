# Copyright 2024 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, AsyncGenerator, Awaitable, Dict, List

import google.generativeai as genai
from google.generativeai.types import (
    ContentDict,
    ContentsType,
    content_types,
    generation_types,
)
from google.generativeai.types.answer_types import FinishReason
from livekit.agents import llm

from .log import logger
from .models import ChatModels
from .utils import build_genai_message, create_gemini_function_info


class GoogleLLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "gemini-1.5-flash",
        api_key: str | None = None,
    ) -> None:
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model)

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
    ) -> LLMStream:
        tools: List[content_types.FunctionLibraryType] = []
        if fnc_ctx and len(fnc_ctx.ai_functions) > 0:
            for fnc in fnc_ctx.ai_functions.values():
                tools.append(fnc.callable)
        contents = _build_content(chat_ctx)
        response = self.model.generate_content_async(contents, stream=True, tools=tools)
        return LLMStream(genai_stream=response, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        *,
        genai_stream: Awaitable[generation_types.AsyncGenerateContentResponse],
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
    ) -> None:
        super().__init__(chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)
        self._awaitable_genai_stream = genai_stream
        self._genai_iter: (
            AsyncGenerator[generation_types.GenerateContentResponse, None] | None
        ) = None

        # current function call that we're waiting for full completion (args are streamed)
        self._fnc_name: str | None = None
        self._fnc_arguments: Dict[str, Any] | None = None

    async def aclose(self) -> None:
        return await super().aclose()

    async def __anext__(self) -> llm.ChatChunk:
        if self._genai_iter is None:
            self._genai_iter = (await self._awaitable_genai_stream).__aiter__()

        chunk = await self._genai_iter.__anext__()
        for candidate in chunk.candidates:
            chat_chunk = self._parse_candidate(candidate)
            if chat_chunk is not None:
                return chat_chunk

    def _parse_candidate(self, candidate) -> llm.ChatChunk | None:
        if candidate.finish_reason == FinishReason.RECITATION:
            # Recitation is when the model starts to recite the training data. This
            # seems to be a condition that causes termination of generation due to
            # copyright infringement concerns.
            # TODO(RWS): Figure out how to rerun generation when this happens.
            logger.warn(
                "Gemini model stopped generation because it started to "
                "recite training data."
            )
            raise StopAsyncIteration()

        for part in candidate.content.parts:
            if part.function_call:
                tool = part.function_call
                if tool.name:
                    self._fnc_name = tool.name
                self._fnc_arguments = dict(tool.args)
                call_chunk = self._try_run_function(part)

                if call_chunk is not None:
                    return call_chunk
            elif part.text:
                return llm.ChatChunk(
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(
                                content=candidate.content.parts[0].text,
                                role="assistant",
                            ),
                            index=0,
                        )
                    ]
                )

    def _try_run_function(self, part) -> llm.ChatChunk | None:
        if not self._fnc_ctx:
            logger.warning(
                "Gemini stream tried to run function without function context"
            )
            return None

        if self._fnc_name is None or self._fnc_arguments is None:
            logger.warning(
                "Gemini stream tried to call a function but arguments and fnc_name are not set"
            )
            return None

        fnc_info = create_gemini_function_info(
            self._fnc_ctx, "", self._fnc_name, self._fnc_arguments
        )
        self._fnc_name = self._fnc_arguments = None
        self._function_calls_info.append(fnc_info)

        return llm.ChatChunk(
            choices=[
                llm.Choice(
                    delta=llm.ChoiceDelta(role="assistant", tool_calls=[fnc_info]),
                    index=0,
                )
            ]
        )


def _build_content(chat_ctx: llm.ChatContext) -> ContentsType:
    contents: List[str] = []
    for msg in chat_ctx.messages:
        role = "model" if msg.role == "assistant" else msg.role
        contents += [ContentDict(role=role, parts=[content_types.to_part(msg.content)])]
    return contents


def _build_genai_context(chat_ctx: llm.ChatContext, cache_key: Any) -> ContentsType:
    return [build_genai_message(msg, cache_key) for msg in chat_ctx.messages]  # type: ignore
