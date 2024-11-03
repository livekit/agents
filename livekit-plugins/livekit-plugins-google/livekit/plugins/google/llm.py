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
import typing
from typing import Any, AsyncIterator, Awaitable, Dict, List, Optional
from uuid import uuid4 as uuid

import vertexai
from livekit.agents import llm
from vertexai.generative_models import (
    Candidate,
    Content,
    FunctionCall,
    FunctionDeclaration,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    Part,
    Tool,
)

from .log import logger
from .models import ChatModels


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        project: str | None = None,
        location: str | None = None,
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
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
    ) -> "LLMStream":
        logger.info(
            "Starting chat with temperature: %s, candidate count: %s", temperature, n
        )
        system_instruction = _extract_system_instruction(chat_ctx)

        tools = []
        if fnc_ctx and fnc_ctx.ai_functions:
            tools = _build_tools(fnc_ctx)

        generation_config = GenerationConfig(
            temperature=temperature,
            candidate_count=n or 1,
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
            tools=tools if tools else None,
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
    logger.debug("System messages: %s", system_messages)
    return system_messages if system_messages else None


def _build_tools(fnc_ctx: llm.FunctionContext) -> List[Tool]:
    tools = []
    for fnc_info in fnc_ctx.ai_functions.values():
        func_decl = FunctionDeclaration(
            name=fnc_info.name,
            description=fnc_info.description,
            parameters=_build_parameters(fnc_info.arguments),
        )
        tool = Tool(function_declarations=[func_decl])
        tools.append(tool)
    # logger.debug("Built tools: %s", tools)
    return tools


def _build_parameters(arguments: Dict[str, llm.FunctionArgInfo]) -> Dict[str, Any]:
    properties = {}
    required = []
    for arg_name, arg_info in arguments.items():
        param = {
            "type": _python_type_to_jsonschema_type(arg_info.type),
            "description": arg_info.description,
        }
        if arg_info.choices:
            param["enum"] = arg_info.choices
        properties[arg_name] = param
        if arg_info.default is inspect.Parameter.empty:
            required.append(arg_name)
    parameters = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required
    # logger.debug("Built parameters: %s", parameters)
    return parameters


def _python_type_to_jsonschema_type(py_type: type) -> str:
    if py_type is str:
        return "string"
    elif py_type is int:
        return "integer"
    elif py_type is float:
        return "number"
    elif py_type is bool:
        return "boolean"
    elif typing.get_origin(py_type) is list:
        return "array"
    else:
        raise ValueError(f"Unsupported type {py_type}")


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        vai_stream: Awaitable[AsyncIterator[GenerationResponse]],
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)
        self._awaitable_vai_stream = vai_stream
        self._vai_stream: AsyncIterator[GenerationResponse] | None = None

        self._function_calls_info: List[llm.FunctionCallInfo] = []

    async def _main_task(self) -> None:
        if not self._vai_stream:
            self._vai_stream = await self._awaitable_vai_stream
        try:
            async for response in self._vai_stream:
                for idx, candidate in enumerate(response.candidates):
                    chat_chunk = self._parse_candidate(candidate, idx)
                    # logger.debug("Chat chunk: %s", chat_chunk)
                    if chat_chunk is not None:
                        self._event_ch.send_nowait(chat_chunk)

        except Exception as e:
            logger.error("Error during main task: %s", e)
            raise e

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
            logger.debug("Function calls: %s", self._function_calls_info)
            logger.debug("index: %s", index)
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

    def _create_function_call_info(
        self, function_call: FunctionCall
    ) -> Optional[llm.FunctionCallInfo]:
        if not self._fnc_ctx:
            return None
        fnc_name = function_call.name
        raw_arguments = json.dumps(function_call.args)
        if fnc_name not in self._fnc_ctx.ai_functions:
            logger.error("Function '%s' not found in function context", fnc_name)
            raise ValueError(f"Function '{fnc_name}' not found in function context")
        fnc_info = self._fnc_ctx.ai_functions[fnc_name]
        parsed_arguments = function_call.args

        sanitized_arguments = {}
        for arg_name, arg_info in fnc_info.arguments.items():
            if arg_name in parsed_arguments:
                arg_value = parsed_arguments[arg_name]
                sanitized_value = _sanitize_value(arg_value, arg_info)
                sanitized_arguments[arg_name] = sanitized_value
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


def _build_vertex_context(chat_ctx: llm.ChatContext, cache_key: Any) -> List[Content]:
    contents: List[Content] = []
    for msg in chat_ctx.messages:
        content = _build_content(msg, cache_key)
        if content:
            contents.append(content)
    logger.debug("Built vertex context: %s", contents)
    return contents


def _build_content(msg: llm.ChatMessage, cache_key: Any) -> Content:
    parts: List[Part] = []
    role = msg.role
    if role == "system" or role == "assistant":
        role = "model"

    if role == "user":
        if isinstance(msg.content, str) and msg.content.strip():
            parts.append(Part.from_text(msg.content))
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, str) and item.strip():
                    parts.append(Part.from_text(item))
                elif isinstance(item, llm.ChatImage):
                    pass  # Handle images if necessary
    elif role == "model":
        if msg.tool_calls:
            logger.debug("Tool calls: %s", msg)
            for fnc in msg.tool_calls:
                # Construct the function call part
                function_call = FunctionCall(
                    name=fnc.function_info.name, args=fnc.arguments
                )
                raw_function_call = function_call._raw_message
                part = Part()
                part._raw_part.function_call = raw_function_call
                parts.append(part)
            pass
        elif isinstance(msg.content, str) and msg.content.strip():
            parts.append(Part.from_text(msg.content))
    elif role == "tool":
        # The tool's response should be represented as a function response
        logger.debug("Tool response: %s", msg)
        role = "model"
        parts.append(
            Part.from_function_response(
                name=msg.name,
                response={"content": msg.content},
            )
        )

    return Content(parts=parts, role=role) if parts else None


def _sanitize_value(value: Any, arg_info: llm.FunctionArgInfo) -> Any:
    expected_type = arg_info.type
    if expected_type is str and not isinstance(value, str):
        raise ValueError(f"Expected argument '{arg_info.name}' to be of type str")
    elif expected_type is int and not isinstance(value, int):
        raise ValueError(f"Expected argument '{arg_info.name}' to be of type int")
    elif expected_type is float and not isinstance(value, (int, float)):
        raise ValueError(f"Expected argument '{arg_info.name}' to be of type float")
    elif expected_type is bool and not isinstance(value, bool):
        raise ValueError(f"Expected argument '{arg_info.name}' to be of type bool")
    elif typing.get_origin(expected_type) is list and not isinstance(value, list):
        raise ValueError(f"Expected argument '{arg_info.name}' to be of type list")
    elif expected_type not in {str, int, float, bool, list}:
        raise ValueError(
            f"Unsupported argument type '{expected_type}' for argument '{arg_info.name}'"
        )

    if arg_info.choices and value not in arg_info.choices:
        raise ValueError(
            f"Value '{value}' for argument '{arg_info.name}' is not in allowed choices {arg_info.choices}"
        )

    return value
