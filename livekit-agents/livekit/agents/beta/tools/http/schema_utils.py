from __future__ import annotations

import re
from typing import Any, Callable

from ....llm.tool_context import ToolError
from .config import HTTPToolParam


def extract_url_params(url: str) -> set[str]:
    """Extract parameter names from URL template like /tasks/:id."""
    return set(re.findall(r":(\w+)", url))


def normalize_parameters_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    """Normalize parameters schema, converting string descriptions to objects."""
    if not parameters or "properties" not in parameters:
        return parameters

    normalized_properties: dict[str, Any] = {}
    for key, value in parameters["properties"].items():
        if isinstance(value, str):
            normalized_properties[key] = {
                "type": "string",
                "description": value,
            }
        else:
            normalized_properties[key] = value

    return {
        **parameters,
        "properties": normalized_properties,
    }


def schema_from_params(params: list[HTTPToolParam]) -> dict[str, Any]:
    """Build a JSON schema from HTTPToolParam entries."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    seen_names: set[str] = set()

    for param in params:
        if param.name in seen_names:
            raise ValueError(f"Duplicate parameter name: {param.name}")
        seen_names.add(param.name)

        field_schema: dict[str, Any] = {
            "type": param.type,
            "description": param.description,
        }

        if param.enum is not None:
            if param.type != "string":
                raise ValueError(
                    f"Parameter '{param.name}' uses enum but is declared as {param.type}. "
                    "Enums are only supported for string parameters."
                )
            field_schema["enum"] = param.enum

        properties[param.name] = field_schema

        if param.required:
            required.append(param.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _validate_argument_type(name: str, value: Any, schema: dict[str, Any]) -> None:
    """Validate argument type based on JSON schema."""

    def _is_number(val: Any) -> bool:
        # bool is a subclass of int; exclude it from number checks
        return isinstance(val, (int, float)) and not isinstance(val, bool)

    type_checkers: dict[str, Callable[[Any], bool]] = {
        "string": lambda v: isinstance(v, str),
        "number": _is_number,
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "null": lambda v: v is None,
    }

    expected_type = schema.get("type")
    if expected_type is None:
        return

    if isinstance(expected_type, list):
        valid = False
        for typ in expected_type:
            checker = type_checkers.get(typ)
            if checker is None:
                valid = True
                break
            if checker(value):
                valid = True
                break
    else:
        checker = type_checkers.get(expected_type)
        if checker is None:
            return
        valid = checker(value)

    if not valid:
        raise ToolError(f"Invalid type for '{name}'. Expected {expected_type}.")

    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        raise ToolError(
            f"Invalid value for '{name}'. Expected one of: {', '.join(map(str, enum_values))}."
        )


def sanitize_arguments(raw_arguments: dict | None, schema: dict[str, Any]) -> dict[str, Any]:
    """Filter and validate arguments based on schema definition."""
    if raw_arguments is None:
        raw_arguments = {}
    if not isinstance(raw_arguments, dict):
        raise ToolError("Invalid arguments - expected an object.")

    properties = schema.get("properties") or {}
    required_fields = schema.get("required", []) or []

    if not properties:
        sanitized = dict(raw_arguments)
    else:
        sanitized = {}
        for key, value in raw_arguments.items():
            if key not in properties:
                continue
            _validate_argument_type(key, value, properties[key])
            sanitized[key] = value

    missing = [field for field in required_fields if field not in sanitized]
    if missing:
        raise ToolError(f"Missing required parameters: {', '.join(missing)}.")

    return sanitized
