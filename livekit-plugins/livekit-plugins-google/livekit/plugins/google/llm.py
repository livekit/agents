# Copyright 2023 LiveKit, Inc.
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

import inspect
import json
from typing import Any, AsyncIterator, Awaitable, Dict, List, Optional
from uuid import uuid4 as uuid

import vertexai
from livekit.agents import llm
from vertexai.generative_models import (
    Candidate,
    FunctionCall,
    FunctionDeclaration,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    Tool,
)

from .log import logger
from .models import ChatModels

JSON_SCHEMA_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
}


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        project: Optional[str] = None,
        location: Optional[str] = None,
        model: str | ChatModels = "gemini-1.5-flash-002",
        **kwargs,
    ) -> None:
        logger.info(
            "Initializing LLM with project: %s, location: %s, model: %s",
            project,
            location,
            model,
        )
        super().__init__()
        self._project = project
        self._location = location
        self._model_name = model
        self._kwargs = kwargs
        vertexai.init(project=project, location=location)

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: Optional[llm.FunctionContext] = None,
        temperature: Optional[float] = None,
        n: int = 1,
    ) -> LLMStream:
        logger.info(
            "Starting chat with temperature: %s, candidate count: %s", temperature, n
        )
        system_instruction = _extract_system_instruction(chat_ctx)

        tools = _build_tools(fnc_ctx) if fnc_ctx and fnc_ctx.ai_functions else []

        generation_config = GenerationConfig(
            temperature=temperature,
            candidate_count=n,
            max_output_tokens=1024,
        )

        model = GenerativeModel(
            model_name=self._model_name,
            system_instruction=system_instruction,
            **self._kwargs,
        )

        contents = _build_vertex_context(chat_ctx, id(self))

        cmp = model.generate_content_async(
            contents=contents,
            generation_config=generation_config,
            tools=tools or None,
            stream=True,
        )

        return LLMStream(
            llm=self,
            vai_stream=cmp,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
        )


def _extract_system_instruction(chat_ctx: llm.ChatContext) -> Optional[List[str]]:
    system_messages = [msg.content for msg in chat_ctx.messages if msg.role == "system"]
    if system_messages:
        logger.debug("Extracted system messages.")
        return system_messages
    return None


def _build_tools(fnc_ctx: llm.FunctionContext) -> List[Tool]:
    tools = [
        Tool(
            function_declarations=[
                FunctionDeclaration(
                    name=fnc_info.name,
                    description=fnc_info.description,
                    parameters=_build_parameters(fnc_info.arguments),
                )
            ]
        )
        for fnc_info in fnc_ctx.ai_functions.values()
    ]
    return tools


def _build_parameters(arguments: Dict[str, llm.FunctionArgInfo]) -> Dict[str, Any]:
    properties = {
        arg_name: {
            "type": JSON_SCHEMA_TYPE_MAP.get(arg_info.type, "string"),
            "description": arg_info.description,
            **({"enum": arg_info.choices} if arg_info.choices else {}),
        }
        for arg_name, arg_info in arguments.items()
    }
    required = [
        arg_name
        for arg_name, arg_info in arguments.items()
        if arg_info.default is inspect.Parameter.empty
    ]

    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required

    return parameters


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        vai_stream: Awaitable[AsyncIterator[GenerationResponse]],
        chat_ctx: llm.ChatContext,
        fnc_ctx: Optional[llm.FunctionContext],
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)
        self._awaitable_vai_stream = vai_stream
        self._vai_stream: Optional[AsyncIterator[GenerationResponse]] = None
        self._function_calls_info: List[llm.FunctionCallInfo] = []

    async def _main_task(self) -> None:
        if not self._vai_stream:
            self._vai_stream = await self._awaitable_vai_stream
        try:
            async for response in self._vai_stream:
                for idx, candidate in enumerate(response.candidates):
                    chat_chunk = self._parse_candidate(candidate, idx)
                    if chat_chunk:
                        self._event_ch.send_nowait(chat_chunk)
        except Exception as e:
            logger.error("Error during main task: %s", e)
            raise

    def _parse_candidate(
        self, candidate: Candidate, index: int
    ) -> Optional[llm.ChatChunk]:
        choice_delta = llm.ChoiceDelta(role="assistant")

        if candidate.function_calls:
            for fnc_call in candidate.function_calls:
                function_call = self._create_function_call_info(fnc_call)
                if function_call:
                    self._function_calls_info.append(function_call)
            choice_delta.tool_calls = self._function_calls_info
            return llm.ChatChunk(
                request_id=uuid(),
                choices=[
                    llm.Choice(
                        delta=choice_delta,
                        index=index,
                    )
                ],
            )

        if candidate.text:
            choice_delta.content = candidate.text
            return llm.ChatChunk(
                request_id=uuid(),
                choices=[
                    llm.Choice(
                        delta=choice_delta,
                        index=index,
                    )
                ],
            )

        return None

    def _create_function_call_info(
        self, function_call: FunctionCall
    ) -> Optional[llm.FunctionCallInfo]:
        if not self._fnc_ctx:
            return None
        fnc_name = function_call.name
        raw_arguments = json.dumps(function_call.args)

        fnc_info = self._fnc_ctx.ai_functions.get(fnc_name)
        if not fnc_info:
            logger.error("Function '%s' not found in function context", fnc_name)
            raise ValueError(f"Function '{fnc_name}' not found in function context")

        parsed_arguments = function_call.args
        sanitized_arguments = {}

        for arg_name, arg_info in fnc_info.arguments.items():
            if arg_name in parsed_arguments:
                arg_value = parsed_arguments[arg_name]
                sanitized_arguments[arg_name] = _sanitize_value(arg_value, arg_info)
            elif arg_info.default is inspect.Parameter.empty:
                logger.error(
                    "Missing required argument '%s' for function '%s'",
                    arg_name,
                    fnc_name,
                )
                raise ValueError(
                    f"Missing required argument '{arg_name}' for function '{fnc_name}'"
                )
            else:
                sanitized_arguments[arg_name] = arg_info.default

        return llm.FunctionCallInfo(
            tool_call_id=f"{fnc_name}_{uuid()}",
            function_info=fnc_info,
            raw_arguments=raw_arguments,
            arguments=sanitized_arguments,
        )


def _build_vertex_context(chat_ctx: llm.ChatContext, cache_key: Any) -> List[dict]:
    contents = []
    current_entry: Optional[dict] = None

    for msg in chat_ctx.messages:
        content = _build_content(msg, cache_key)
        if not content:
            continue

        role = content["role"]
        parts = content["parts"]

        if role == "model" and msg.role == "system":
            contents.append(content)
            continue

        if current_entry is None or role != current_entry["role"]:
            if current_entry:
                contents.append(current_entry)
            current_entry = {"role": role, "parts": parts}
        else:
            current_entry["parts"].extend(parts)

    if current_entry:
        contents.append(current_entry)

    logger.debug("Built vertex context.")
    return contents


def _build_content(msg: llm.ChatMessage, cache_key: Any) -> Optional[dict]:
    role = msg.role
    if role in {"system", "assistant"}:
        role = "model"

    parts = []

    if role == "user":
        if isinstance(msg.content, str) and msg.content.strip():
            parts.append({"text": msg.content})
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, str) and item.strip():
                    parts.append({"text": item})

    elif role == "model":
        if msg.tool_calls:
            for fnc in msg.tool_calls:
                function_call = {"name": fnc.function_info.name, "args": fnc.arguments}
                parts.append({"function_call": function_call})
        elif isinstance(msg.content, str) and msg.content.strip():
            parts.append({"text": msg.content})

    elif role == "tool":
        role = "user"
        function_response = {"name": msg.name, "response": {"content": msg.content}}
        parts.append({"function_response": function_response})

    return {"role": role, "parts": parts} if parts else None


def _sanitize_value(value: Any, arg_info: llm.FunctionArgInfo) -> Any:
    expected_type = arg_info.type

    if not isinstance(value, expected_type) and not (
        expected_type is float and isinstance(value, int)
    ):
        raise ValueError(
            f"Expected argument '{arg_info.name}' to be of type {expected_type.__name__}"
        )

    if arg_info.choices and value not in arg_info.choices:
        raise ValueError(
            f"Value '{value}' for argument '{arg_info.name}' is not in allowed choices {arg_info.choices}"
        )

    return value
