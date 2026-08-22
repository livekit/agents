from __future__ import annotations

import json
from typing import Any

from livekit.agents import llm
from livekit.agents.llm.utils import function_arguments_to_pydantic_model


def parse_tools(tools: list[llm.FunctionTool | llm.RawFunctionTool]) -> list[dict[str, Any]]:
    """Prepare tools for sending to Ultravox. https://docs.ultravox.ai/essentials/tools#creating-your-first-custom-tool"""

    results: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, llm.RawFunctionTool):
            raw_fnc_info = tool.info
            name = raw_fnc_info.name
            description = raw_fnc_info.raw_schema.get("description", None)
            parameters = raw_fnc_info.raw_schema.get("parameters", {})
        elif isinstance(tool, llm.FunctionTool):
            fnc_info = tool.info
            model = function_arguments_to_pydantic_model(tool)
            name = fnc_info.name
            description = fnc_info.description
            parameters = model.model_json_schema()

        def _extract_type(prop: dict[str, Any]) -> str:
            """Best-effort guess of a parameter's primitive type."""
            if "type" in prop:
                assert isinstance(prop["type"], str)
                return prop["type"]
            if "enum" in prop:
                return "string"
            if "items" in prop:
                return "array"
            for key in ("anyOf", "oneOf"):
                if key in prop:
                    for variant in prop[key]:
                        if isinstance(variant, dict) and "type" in variant:
                            assert isinstance(variant["type"], str)
                            return variant["type"]
            # Fallback to string
            return "string"

        results.append(
            {
                "temporaryTool": {
                    "modelToolName": name,
                    "description": description,
                    "dynamicParameters": [
                        {
                            "name": pn,
                            "location": "PARAMETER_LOCATION_BODY",
                            "schema": (
                                p
                                if "type" in p
                                else {  # fallback minimal schema for enum/anyOf etc.
                                    "type": _extract_type(p),
                                    **(
                                        {"description": p.get("description")}
                                        if p.get("description")
                                        else {}
                                    ),
                                }
                            ),
                            "required": pn in parameters.get("required", []),
                        }
                        for pn, p in parameters["properties"].items()
                    ],
                    "client": {},
                }
            }
        )
    return results


def coerce_parameters_to_schema(
    tools_ctx: llm.ToolContext,
    tool_name: str,
    params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Coerce Ultravox tool parameters to match the declared tool schema.

    This mitigates cases where Ultravox delivers stringified values (e.g., "10", "true",
    "[\"x\"]") by converting them to the correct JSON types expected by
    downstream tool executors (e.g., MCP servers).
    """

    def _best_effort_coerce(raw: dict[str, Any]) -> dict[str, Any]:
        def try_parse_json(s: str) -> Any:
            try:
                return json.loads(s)
            except Exception:
                return s

        out: dict[str, Any] = {}
        for k, v in (raw or {}).items():
            if isinstance(v, str):
                vv = v.strip()
                lower = vv.lower()
                if lower in ("true", "false"):
                    out[k] = lower == "true"
                elif lower == "null":
                    out[k] = None
                elif vv and (vv[0] in "[{" and vv[-1] in "]}"):
                    out[k] = try_parse_json(vv)
                else:
                    try:
                        out[k] = float(vv) if "." in vv or "e" in lower else int(vv)
                    except Exception:
                        out[k] = v
            else:
                out[k] = v
        return out

    def _coerce_value(value: Any, schema: dict[str, Any]) -> Any:
        t = schema.get("type")
        if t is None:
            for key in ("anyOf", "oneOf", "allOf"):
                if key in schema and isinstance(schema[key], list) and schema[key]:
                    for variant in schema[key]:
                        if isinstance(variant, dict):
                            try:
                                vt = variant.get("type")
                                if (
                                    isinstance(vt, (str, list))
                                    and isinstance(value, str)
                                    and value.strip().lower() in ("null", "none")
                                ):
                                    allowed = [vt] if isinstance(vt, str) else list(vt)
                                    if "null" in allowed:
                                        return None
                                return _coerce_value(value, variant)
                            except Exception:
                                continue
                    break
            return value

        # Normalize types and detect nullability
        allowed_types: list[str] = [t] if isinstance(t, str) else list(t)
        is_nullable = "null" in allowed_types

        # If schema allows null and value is a string "null"/"None", convert to None
        if is_nullable and isinstance(value, str) and value.strip().lower() in ("null", "none"):
            return None

        # Pick the effective type to coerce to (prefer first non-null)
        t = next((tt for tt in allowed_types if tt != "null"), allowed_types[0])

        if t == "null":
            return None if value in (None, "null") else value

        if t == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("true", "1", "yes", "y", "on"):
                    return True
                if v in ("false", "0", "no", "n", "off"):
                    return False
            return value

        if t in ("number", "integer"):
            if isinstance(value, (int, float)):
                return int(value) if t == "integer" else float(value)
            if isinstance(value, str):
                v = value.strip()
                vl = v.lower()
                if vl == "nan":
                    return float("nan")
                if vl in ("inf", "+inf", "infinity", "+infinity"):
                    return float("inf")
                if vl in ("-inf", "-infinity"):
                    return float("-inf")
                try:
                    num = float(v) if "." in v or "e" in vl else int(v)
                    return int(num) if t == "integer" else float(num)
                except Exception:
                    return value
            return value

        if t == "array":
            if isinstance(value, list):
                items_schema = schema.get("items")
                if isinstance(items_schema, dict):
                    return [_coerce_value(it, items_schema) for it in value]
                return value
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        items_schema = schema.get("items")
                        if isinstance(items_schema, dict):
                            return [_coerce_value(it, items_schema) for it in parsed]
                        return parsed
                except Exception:
                    return [value]
            return [value]

        if t == "object":
            if isinstance(value, dict):
                props = schema.get("properties") or {}
                coerced = {}
                for k, v in value.items():
                    if k in props and isinstance(props[k], dict):
                        coerced[k] = _coerce_value(v, props[k])
                    else:
                        coerced[k] = v
                return coerced
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        return _coerce_value(parsed, schema)
                except Exception:
                    return value
            return value

        if t == "string" and isinstance(value, str):
            if schema.get("enum") is None:
                v = value.strip()
                if (v.startswith("{") and v.endswith("}")) or (
                    v.startswith("[") and v.endswith("]")
                ):
                    try:
                        return json.loads(v)
                    except Exception:
                        pass
        return value

    params = params or {}
    tool = tools_ctx.function_tools.get(tool_name)
    if not tool:
        return _best_effort_coerce(params)

    schema: dict[str, Any] | None = None
    if is_function_tool(tool):
        try:
            model = function_arguments_to_pydantic_model(tool)
            schema = model.model_json_schema()
        except Exception:
            schema = None
    elif is_raw_function_tool(tool):
        try:
            info = get_raw_function_info(tool)
            schema = info.raw_schema.get("parameters")
        except Exception:
            schema = None

    if not schema:
        return _best_effort_coerce(params)

    properties = schema.get("properties") or {}
    out: dict[str, Any] = {}
    for k, v in params.items():
        prop_schema = properties.get(k)
        if isinstance(prop_schema, dict):
            out[k] = _coerce_value(v, prop_schema)
        else:
            out[k] = v

    required = schema.get("required") or []
    for k in required:
        if k in out and isinstance(out[k], str) and out[k].strip().lower() == "null":
            out[k] = None

    return out
