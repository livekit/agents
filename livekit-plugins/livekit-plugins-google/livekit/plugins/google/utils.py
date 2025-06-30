from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from pydantic import TypeAdapter

from google.genai import types
from livekit.agents import llm
from livekit.agents.llm import utils as llm_utils
from livekit.agents.llm.tool_context import (
    FunctionTool,
    RawFunctionTool,
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)

from .log import logger
from .tools import _LLMTool

__all__ = ["to_fnc_ctx"]


def to_fnc_ctx(fncs: list[FunctionTool | RawFunctionTool]) -> list[types.FunctionDeclaration]:
    tools: list[types.FunctionDeclaration] = []
    for fnc in fncs:
        if is_raw_function_tool(fnc):
            info = get_raw_function_info(fnc)
            tools.append(
                types.FunctionDeclaration(
                    name=info.name,
                    description=info.raw_schema.get("description", ""),
                    parameters_json_schema=info.raw_schema.get("parameters", {}),
                )
            )

        elif is_function_tool(fnc):
            tools.append(_build_gemini_fnc(fnc))

    return tools


def create_tools_config(
    *,
    function_tools: list[types.FunctionDeclaration] | None = None,
    gemini_tools: list[_LLMTool] | None = None,
) -> list[types.Tool]:
    tools: list[types.Tool] = []

    if function_tools:
        tools.append(types.Tool(function_declarations=function_tools))

    if gemini_tools:
        for tool in gemini_tools:
            if isinstance(tool, types.GoogleSearchRetrieval):
                tools.append(types.Tool(google_search_retrieval=tool))
            elif isinstance(tool, types.ToolCodeExecution):
                tools.append(types.Tool(code_execution=tool))
            elif isinstance(tool, types.GoogleSearch):
                tools.append(types.Tool(google_search=tool))
            elif isinstance(tool, types.UrlContext):
                tools.append(types.Tool(url_context=tool))
            elif isinstance(tool, types.GoogleMaps):
                tools.append(types.Tool(google_maps=tool))
            else:
                logger.warning(f"Warning: Received unhandled tool type: {type(tool)}")
                continue

    if len(tools) > 1:
        # https://github.com/google/adk-python/issues/53#issuecomment-2799538041
        logger.warning(
            "Multiple kinds of tools are not supported in Gemini. Only the first tool will be used."
        )
        tools = tools[:1]

    return tools


def get_tool_results_for_realtime(
    chat_ctx: llm.ChatContext, *, vertexai: bool = False
) -> types.LiveClientToolResponse | None:
    function_responses: list[types.FunctionResponse] = []
    for msg in chat_ctx.items:
        if msg.type == "function_call_output":
            res = types.FunctionResponse(
                name=msg.name,
                response={"output": msg.output},
            )
            if not vertexai:
                # vertexai does not support id in FunctionResponse
                # see: https://github.com/googleapis/python-genai/blob/85e00bc/google/genai/_live_converters.py#L1435
                res.id = msg.call_id
            function_responses.append(res)
    return (
        types.LiveClientToolResponse(function_responses=function_responses)
        if function_responses
        else None
    )


def _build_gemini_fnc(function_tool: FunctionTool) -> types.FunctionDeclaration:
    fnc = llm.utils.build_legacy_openai_schema(function_tool, internally_tagged=True)
    json_schema = _GeminiJsonSchema(fnc["parameters"]).simplify()
    return types.FunctionDeclaration(
        name=fnc["name"],
        description=fnc["description"],
        parameters=types.Schema.model_validate(json_schema) if json_schema else None,
    )


def to_response_format(response_format: type | dict) -> types.SchemaUnion:
    _, json_schema_type = llm_utils.to_response_format_param(response_format)
    if isinstance(json_schema_type, TypeAdapter):
        schema = json_schema_type.json_schema()
    else:
        schema = json_schema_type.model_json_schema()

    return _GeminiJsonSchema(schema).simplify()


class _GeminiJsonSchema:
    """
    Transforms the JSON Schema from Pydantic to be suitable for Gemini.
    based on pydantic-ai implementation
    https://github.com/pydantic/pydantic-ai/blob/085a9542a7360b7e388ce575323ce189b397d7ad/pydantic_ai_slim/pydantic_ai/models/gemini.py#L809
    """

    # Type mapping from JSON Schema to Gemini Schema
    TYPE_MAPPING: dict[str, types.Type] = {
        "string": types.Type.STRING,
        "number": types.Type.NUMBER,
        "integer": types.Type.INTEGER,
        "boolean": types.Type.BOOLEAN,
        "array": types.Type.ARRAY,
        "object": types.Type.OBJECT,
    }

    def __init__(self, schema: dict[str, Any]):
        self.schema = deepcopy(schema)
        self.defs = self.schema.pop("$defs", {})

    def simplify(self) -> dict[str, Any] | None:
        self._simplify(self.schema, refs_stack=())
        # If the schema is an OBJECT with no properties, return None.
        if self.schema.get("type") == types.Type.OBJECT and not self.schema.get("properties"):
            return None
        return self.schema

    def _simplify(self, schema: dict[str, Any], refs_stack: tuple[str, ...]) -> None:
        schema.pop("title", None)
        schema.pop("default", None)
        schema.pop("additionalProperties", None)
        schema.pop("$schema", None)

        if (const := schema.pop("const", None)) is not None:
            # Gemini doesn't support const, but it does support enum with a single value
            schema["enum"] = [const]

        schema.pop("discriminator", None)
        schema.pop("examples", None)

        if ref := schema.pop("$ref", None):
            key = re.sub(r"^#/\$defs/", "", ref)
            if key in refs_stack:
                raise ValueError("Recursive `$ref`s in JSON Schema are not supported by Gemini")
            refs_stack += (key,)
            schema_def = self.defs[key]
            self._simplify(schema_def, refs_stack)
            schema.update(schema_def)
            return

        # Convert type value to Gemini format
        if "type" in schema and schema["type"] != "null":
            json_type = schema["type"]
            if json_type in self.TYPE_MAPPING:
                schema["type"] = self.TYPE_MAPPING[json_type]
            elif isinstance(json_type, types.Type):
                schema["type"] = json_type
            else:
                raise ValueError(f"Unsupported type in JSON Schema: {json_type}")

        # Map field names that differ between JSON Schema and Gemini
        self._map_field_names(schema)

        # Handle anyOf - map to any_of
        if any_of := schema.pop("anyOf", None):
            if any_of:
                mapped_any_of = []
                has_null = False
                non_null_schema = None

                for item_schema in any_of:
                    self._simplify(item_schema, refs_stack)
                    if item_schema == {"type": "null"}:
                        has_null = True
                    else:
                        non_null_schema = item_schema
                        mapped_any_of.append(item_schema)

                if has_null and len(any_of) == 2 and non_null_schema:
                    schema.update(non_null_schema)
                    schema["nullable"] = True
                else:
                    schema["any_of"] = mapped_any_of

        type_ = schema.get("type")

        if type_ == types.Type.OBJECT:
            self._object(schema, refs_stack)
        elif type_ == types.Type.ARRAY:
            self._array(schema, refs_stack)

    def _map_field_names(self, schema: dict[str, Any]) -> None:
        """Map JSON Schema field names to Gemini Schema field names."""
        mappings = {
            "minLength": "min_length",
            "maxLength": "max_length",
            "minItems": "min_items",
            "maxItems": "max_items",
            "minProperties": "min_properties",
            "maxProperties": "max_properties",
        }

        for json_name, gemini_name in mappings.items():
            if json_name in schema:
                schema[gemini_name] = schema.pop(json_name)

    def _object(self, schema: dict[str, Any], refs_stack: tuple[str, ...]) -> None:
        if properties := schema.get("properties"):
            for value in properties.values():
                self._simplify(value, refs_stack)

    def _array(self, schema: dict[str, Any], refs_stack: tuple[str, ...]) -> None:
        if prefix_items := schema.get("prefixItems"):
            for prefix_item in prefix_items:
                self._simplify(prefix_item, refs_stack)

        if items_schema := schema.get("items"):
            self._simplify(items_schema, refs_stack)
