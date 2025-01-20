from __future__ import annotations

from typing import Literal, Sequence, Union

from google.genai import types

from ..._utils import _build_tools

LiveAPIModels = Literal["gemini-2.0-flash-exp"]

Voice = Literal["Puck", "Charon", "Kore", "Fenrir", "Aoede"]

__all__ = ["_build_tools", "ClientEvents"]

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
