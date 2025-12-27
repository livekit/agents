from __future__ import annotations

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
