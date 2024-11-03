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

import asyncio
import inspect
import json
import typing
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part, FunctionDeclaration, Tool, GenerationResponse, FunctionCall, ToolConfig, Candidate

from livekit.agents import llm
from .models import ChatModels
from .log import logger

class LLM(llm.LLM):
    def __init__(
        self,
        *,
        project: str | None = None,
        location: str | None = None,
        model: str | ChatModels = "gemini-1.5-flash-002",
        **kwargs,
    ) -> None:
        logger.debug(
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
        logger.debug(
            "Starting chat with temperature: %s, candidate count: %s", temperature, n
        )
        system_instruction = _extract_system_instruction(chat_ctx)
        logger.debug("Extracted system instruction: %s", system_instruction)

        tools = []
        tool_config = None
        if fnc_ctx and len(fnc_ctx.ai_functions) > 0:
            tools = _build_tools(fnc_ctx)
            logger.debug("Built tools: %s", tools)
            
        generation_config = GenerationConfig(
            temperature=temperature,
            candidate_count=n or 1,
            max_output_tokens=1024,
        )
        logger.debug("Configured generation settings: %s", generation_config)

        model = GenerativeModel(
            model_name=self._model_name,
            system_instruction=system_instruction,
            **self._kwargs,
        )
        logger.debug("Created GenerativeModel instance")

        contents = _build_vertex_context(chat_ctx, id(self))

        response_stream_awaitable = model.generate_content_async(
            contents=contents,
            generation_config=generation_config,
            tools=tools if tools else None,
            tool_config=tool_config,
            stream=True,
        )

        return LLMStream(
            llm=self,
            response_stream_awaitable=response_stream_awaitable,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            generation_config=generation_config,
        )


def _extract_system_instruction(chat_ctx: llm.ChatContext) -> Optional[List[str]]:
    system_messages = [msg.content for msg in chat_ctx.messages if msg.role == "system"]
    logger.debug("System messages extracted: %s", system_messages)
    return system_messages if system_messages else None


def _build_tools(fnc_ctx: llm.FunctionContext) -> List[Tool]:
    logger.debug("Building tools from function context")
    tools = []
    for fnc_info in fnc_ctx.ai_functions.values():
        func_decl = FunctionDeclaration(
            name=fnc_info.name,
            description=fnc_info.description,
            parameters=_build_parameters(fnc_info.arguments),
        )
        tool = Tool(function_declarations=[func_decl])
        logger.debug("Created tool for function %s", fnc_info.name)
        tools.append(tool)
    return tools


def _build_parameters(arguments: Dict[str, llm.FunctionArgInfo]) -> Dict[str, Any]:
    properties = {}
    required = []
    logger.debug("Building parameters from arguments")
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
        logger.debug("Parameter %s built with details: %s", arg_name, param)
    parameters = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required
    logger.debug("Final parameters object: %s", parameters)
    return parameters


def _python_type_to_jsonschema_type(py_type: type) -> str:
    logger.debug("Mapping Python type %s to JSON schema type", py_type)
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
        response_stream_awaitable: Awaitable[AsyncIterator[GenerationResponse]],
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None,
        generation_config: GenerationConfig,
    ) -> None:
        logger.debug("Initializing LLMStream")
        super().__init__(llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)
        self._response_stream_awaitable = response_stream_awaitable
        self._response_stream: AsyncIterator[GenerationResponse] | None = None
        self._generation_config = generation_config
        self._function_calls_info: List[llm.FunctionCallInfo] = []

    async def _main_task(self) -> None:
        try:
            if not self._response_stream:
                self._response_stream = await self._response_stream_awaitable

            logger.debug("response stream: %s", self._response_stream)
            async for response in self._response_stream:
                for idx, candidate in enumerate(response.candidates):
                    chat_chunk = self._parse_candidate(candidate, idx)
                    if chat_chunk is not None:
                        self._event_ch.send_nowait(chat_chunk)
                        logger.debug("Sent chat chunk: %s", chat_chunk)
                
                
                
        except Exception as e:
            logger.error("Error during main task: %s", e)
            raise e

    def _parse_candidate(self, candidate: Candidate, index: int) -> Optional[llm.ChatChunk]:
        logger.debug("Parsing candidate index %d", index)
        logger.debug("Candidate : %s", candidate)
        content = candidate.content
        text_parts = []
        function_calls = []

        for part in content.parts:
            part_type = part._raw_part._pb.WhichOneof("data")
            logger.debug("Part type: %s", part_type)
            if part_type == "text":
                text_parts.append(part.text)
                logger.debug("Appended text part: %s", part.text)
            elif part_type == "function_call":
                function_call_info = self._create_function_call_info(part.function_call)
                if function_call_info:
                    self._function_calls_info.append(function_call_info)
                    function_calls.append(function_call_info)
                    logger.debug("Appended function call: %s", function_call_info)
            else:
                logger.warning("Unhandled part type: %s", part_type)

        text = ''.join(text_parts).strip()
        choice_delta = llm.ChoiceDelta(role="assistant")

        if text:
            choice_delta.content = text

        if function_calls:
            choice_delta.tool_calls = function_calls

        if not text and not function_calls:
            logger.warning("No content or function calls found for candidate index %d", index)
            return None

        return llm.ChatChunk(
            request_id=candidate.index,
            choices=[
                llm.Choice(
                    delta=choice_delta,
                    index=index,
                )
            ],
        )


    def _create_function_call_info(self, function_call: FunctionCall) -> Optional[llm.FunctionCallInfo]:
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
                logger.error("Missing required argument '%s' for function '%s'", arg_name, fnc_name)
                raise ValueError(f"Missing required argument '{arg_name}' for function '{fnc_name}'")
            else:
                sanitized_arguments[arg_name] = arg_info.default
        logger.debug("Created function call info for '%s': %s", fnc_name, sanitized_arguments)
        return llm.FunctionCallInfo(
            tool_call_id=fnc_name,
            function_info=fnc_info,
            raw_arguments=raw_arguments,
            arguments=sanitized_arguments,
        )

def _build_vertex_context(
    chat_ctx: llm.ChatContext, cache_key: Any
) -> List[Content]:
    contents: List[Content] = []
    for msg in chat_ctx.messages:
        content = _build_content(msg, cache_key)
        if content:
            contents.append(content)
    logger.debug("Built vertex context: %s", contents)
    return contents

def _build_content(
    msg: llm.ChatMessage, cache_key: Any
) -> Content:
    parts: List[Part] = []
    content: Content = None
    role = msg.role
    if role == "system" or role == "assistant":
        role = "model"
    
    if role == "user" or role == "model":
        if isinstance(msg.content, str) and msg.content.strip():
            parts.append(Part.from_text(msg.content))
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, str) and item.strip():
                    parts.append(Part.from_text(item))
                elif isinstance(item, llm.ChatImage):
                    pass  # Handle images 
        if msg.tool_calls is not None:
            for fnc in msg.tool_calls:
                # Part doesn't provide a way to add function calls
                parts.append(
                    Part.from_text(f"function {fnc.function_info.name} call: {fnc.arguments}")
                )
    elif role == "tool":
        role = "model"
        parts.append(
            # Part.from_function_response(
            #     name=msg.name,
            #     response= {"content": msg.content},
            # )
            # gives: error during main task: 400 please ensure that function response turn comes immediately after a function call turn. and the number of function response parts should be equal to number of function call parts of the function call turn.
            # mostly because we are passing whole context 
            Part.from_text(f"function {msg.name} response: {msg.content}")
        )
    if parts:
        content = Content(parts=parts, role=role)
        logger.debug("Built content for message: %s", content)
    else:
        logger.warning("No content found for message: %s", msg)
    
    return content

def _sanitize_value(value: Any, arg_info: llm.FunctionArgInfo) -> Any:
    expected_type = arg_info.type
    if expected_type is str:
        if not isinstance(value, str):
            logger.error("Expected argument '%s' to be of type str", arg_info.name)
            raise ValueError(f"Expected argument '{arg_info.name}' to be of type str")
    elif expected_type is int:
        if not isinstance(value, int):
            logger.error("Expected argument '%s' to be of type int", arg_info.name)
            raise ValueError(f"Expected argument '{arg_info.name}' to be of type int")
    elif expected_type is float:
        if not isinstance(value, (int, float)):
            logger.error("Expected argument '%s' to be of type float", arg_info.name)
            raise ValueError(f"Expected argument '{arg_info.name}' to be of type float")
    elif expected_type is bool:
        if not isinstance(value, bool):
            logger.error("Expected argument '%s' to be of type bool", arg_info.name)
            raise ValueError(f"Expected argument '{arg_info.name}' to be of type bool")
    elif typing.get_origin(expected_type) is list:
        if not isinstance(value, list):
            logger.error("Expected argument '%s' to be of type list", arg_info.name)
            raise ValueError(f"Expected argument '{arg_info.name}' to be of type list")
    else:
        logger.error("Unsupported argument type '%s' for argument '%s'", expected_type, arg_info.name)
        raise ValueError(f"Unsupported argument type '{expected_type}' for argument '{arg_info.name}'")

    if arg_info.choices and value not in arg_info.choices:
        logger.error("Value '%s' for argument '%s' is not in allowed choices %s", value, arg_info.name, arg_info.choices)
        raise ValueError(f"Value '{value}' for argument '{arg_info.name}' is not in allowed choices {arg_info.choices}")

    return value
