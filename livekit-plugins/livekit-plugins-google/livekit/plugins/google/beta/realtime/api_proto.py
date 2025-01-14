from __future__ import annotations

import inspect
import json
from typing import Any, Dict, List, Literal, Sequence, Union

from livekit.agents import llm

from google.genai import types

__all__ = [
    "ClientEvents",
    "LiveAPIModels",
    "ResponseModality",
    "Voice",
    "_build_gemini_ctx",
    "_build_tools",
]

LiveAPIModels = Literal["gemini-2.0-flash-exp"]

Voice = Literal["Puck", "Charon", "Kore", "Fenrir", "Aoede"]
ResponseModality = Literal["AUDIO", "TEXT"]


ClientEvents = Union[
    types.ContentListUnion,
    types.ContentListUnionDict,
    types.LiveClientContentOrDict,
    types.LiveClientRealtimeInput,
    types.LiveClientRealtimeInputOrDict,
    types.LiveClientToolResponseOrDict,
    types.FunctionResponseOrDict,
    Sequence[types.FunctionResponseOrDict],
]


JSON_SCHEMA_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def _build_parameters(arguments: Dict[str, Any]) -> types.SchemaDict:
    properties: Dict[str, types.SchemaDict] = {}
    required: List[str] = []

    for arg_name, arg_info in arguments.items():
        py_type = arg_info.type
        if py_type not in JSON_SCHEMA_TYPE_MAP:
            raise ValueError(f"Unsupported type: {py_type}")

        prop: types.SchemaDict = {
            "type": JSON_SCHEMA_TYPE_MAP[py_type],
            "description": arg_info.description,
        }

        if arg_info.choices:
            prop["enum"] = arg_info.choices

        properties[arg_name] = prop

        if arg_info.default is inspect.Parameter.empty:
            required.append(arg_name)

    parameters: types.SchemaDict = {"type": "object", "properties": properties}

    if required:
        parameters["required"] = required

    return parameters


def _build_tools(fnc_ctx: Any) -> List[types.FunctionDeclarationDict]:
    function_declarations: List[types.FunctionDeclarationDict] = []
    for fnc_info in fnc_ctx.ai_functions.values():
        parameters = _build_parameters(fnc_info.arguments)

        func_decl: types.FunctionDeclarationDict = {
            "name": fnc_info.name,
            "description": fnc_info.description,
            "parameters": parameters,
        }

        function_declarations.append(func_decl)

    return function_declarations


def _build_gemini_ctx(chat_ctx: llm.ChatContext) -> List[types.Content]:
    content = None
    turns = []

    for msg in chat_ctx.messages:
        role = None
        if msg.role == "assistant":
            role = "model"
        elif msg.role in {"system", "user"}:
            role = "user"
        elif msg.role == "tool":
            continue

        if content and content.role == role:
            if isinstance(msg.content, str):
                content.parts.append(types.Part(text=msg.content))
            elif isinstance(msg.content, dict):
                content.parts.append(types.Part(text=json.dumps(msg.content)))
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, str):
                        content.parts.append(types.Part(text=item))
        else:
            content = types.Content(
                parts=[types.Part(text=msg.content)],
                role=role,
            )
            turns.append(content)

    return turns
