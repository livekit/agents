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

import functools
import inspect
import json
import tempfile
import typing
from typing import Any, Dict

import google.generativeai as genai
import requests
from google.generativeai.types import ContentDict, File, StrictContentType
from google.generativeai.types.content_types import to_part
from livekit.agents import llm, utils
from livekit.agents.llm import function_context
from livekit.agents.llm._oai_api import _sanitize_primitive

from livekit import rtc


def build_genai_message(msg: llm.ChatMessage, cache_key: Any) -> StrictContentType:
    genai_msg: dict[str, Any] = {"role": msg.role}

    role = "model" if msg.role == "assistant" else msg.role
    if role != "model" and role != "user":
        raise ValueError(
            "Only 'model' and 'user' roles are supported with Google Gemini."
        )

    parts = []

    # add content if provided
    if isinstance(msg.content, str):
        parts = [to_part(msg.content)]
    elif isinstance(msg.content, list):
        for cnt in msg.content:
            if isinstance(cnt, str):
                parts += [to_part(str)]
            elif isinstance(cnt, llm.ChatImage):
                parts += [to_part(_build_genai_image_content(cnt, cache_key))]

    # NOTE(RWS): Not sure if this if this happens with Gemini.
    # # make sure to provide when function has been called inside the context
    # # (+ raw_arguments)
    # if msg.tool_calls is not None:
    #     tool_calls: list[dict[str, Any]] = []
    #     genai_msg["tool_calls"] = tool_calls
    #     for fnc in msg.tool_calls:
    #         tool_calls.append(
    #             {
    #                 "id": fnc.tool_call_id,
    #                 "type": "function",
    #                 "function": {
    #                     "name": fnc.function_info.name,
    #                     "arguments": fnc.raw_arguments,
    #                 },
    #             }
    #         )

    # # tool_call_id is set when the message is a response/result to a function call
    # # (content is a string in this case)
    # if msg.tool_call_id:
    #     genai_msg["tool_call_id"] = msg.tool_call_id

    genai_msg = ContentDict(role=role, parts=parts)
    return genai_msg


@functools.lru_cache
def _build_genai_image_from_url(url: str) -> File:
    # TODO(RWS): Handle case where file is removed from server due to staleness.
    response = requests.get(url)
    if response.status_code == 200:
        content_type = response.headers.get("Content-Type", "").split(";")[0]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(response.content)
            uploaded_file = genai.upload_file(temp_file.name, mime_type=content_type)
            return uploaded_file

    raise ValueError(f"Failed to download image from {url}")


def _build_genai_image_content(image: llm.ChatImage, cache_key: Any) -> File:
    if isinstance(image.image, str):  # image url
        genai_image = _build_genai_image_from_url(image.image)
        return genai_image
    elif isinstance(image.image, rtc.VideoFrame):  # VideoFrame
        if cache_key not in image._cache:
            # inside our internal implementation, we allow to put extra metadata to
            # each ChatImage (avoid to reencode each time we do a chatcompletion request)
            opts = utils.images.EncodeOptions()
            if image.inference_width and image.inference_height:
                opts.resize_options = utils.images.ResizeOptions(
                    width=image.inference_width,
                    height=image.inference_height,
                    strategy="center_aspect_fit",
                )
            encoded_data = utils.images.encode(image.image, opts)
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(encoded_data)
                genai_image = genai.upload_file(temp_file.name, mime_type=opts.format)
                image._cache[cache_key] = genai_image

        return image._cache[cache_key]

    raise ValueError(f"unknown image type {type(image.image)}")


def build_gemini_function_description(
    fnc_info: function_context.FunctionInfo,
) -> dict[str, Any]:
    def build_gemini_property(arg_info: function_context.FunctionArgInfo):
        def type2str(t: type) -> str:
            if t is str:
                return "string"
            elif t in (int, float):
                return "number"
            elif t is bool:
                return "boolean"

            raise ValueError(f"unsupported type {t} for ai_property")

        p: dict[str, Any] = {}

        if arg_info.description:
            p["description"] = arg_info.description

        if typing.get_origin(arg_info.type) is list:
            inner_type = typing.get_args(arg_info.type)[0]
            p["type"] = "array"
            p["items"] = {}
            p["items"]["type"] = type2str(inner_type)

            if arg_info.choices:
                p["items"]["enum"] = arg_info.choices
        else:
            p["type"] = type2str(arg_info.type)
            if arg_info.choices:
                p["enum"] = arg_info.choices

        return p

    properties_info: dict[str, dict[str, Any]] = {}
    required_properties: list[str] = []

    for arg_info in fnc_info.arguments.values():
        if arg_info.default is inspect.Parameter.empty:
            required_properties.append(arg_info.name)

        properties_info[arg_info.name] = build_gemini_property(arg_info)

    return {
        "name": fnc_info.name,
        "description": fnc_info.description,
        "parameters": {
            "type": "object",
            "properties": properties_info,
            "required": required_properties,
        },
    }


def create_gemini_function_info(
    fnc_ctx: function_context.FunctionContext,
    tool_call_id: str,
    fnc_name: str,
    arguments: Dict[str, Any],
) -> function_context.FunctionCallInfo:
    if fnc_name not in fnc_ctx.ai_functions:
        raise ValueError(f"AI function {fnc_name} not found")

    fnc_info = fnc_ctx.ai_functions[fnc_name]

    # Ensure all necessary arguments are present and of the correct type.
    sanitized_arguments: dict[str, Any] = {}
    for arg_info in fnc_info.arguments.values():
        if arg_info.name not in arguments:
            if arg_info.default is inspect.Parameter.empty:
                raise ValueError(
                    f"AI function {fnc_name} missing required argument {arg_info.name}"
                )
            continue

        arg_value = arguments[arg_info.name]
        if typing.get_origin(arg_info.type) is not None:
            if not isinstance(arg_value, list):
                raise ValueError(
                    f"AI function {fnc_name} argument {arg_info.name} should be a list"
                )

            inner_type = typing.get_args(arg_info.type)[0]
            sanitized_value = [
                _sanitize_primitive(
                    value=v, expected_type=inner_type, choices=arg_info.choices
                )
                for v in arg_value
            ]
        else:
            sanitized_value = _sanitize_primitive(
                value=arg_value, expected_type=arg_info.type, choices=arg_info.choices
            )

        sanitized_arguments[arg_info.name] = sanitized_value

    return function_context.FunctionCallInfo(
        tool_call_id=tool_call_id,
        raw_arguments=json.dumps(arguments),
        function_info=fnc_info,
        arguments=sanitized_arguments,
    )
