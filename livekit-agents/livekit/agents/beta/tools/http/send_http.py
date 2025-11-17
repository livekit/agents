from __future__ import annotations

import asyncio

from typing import Any

import aiohttp

from .... import FunctionTool
from ....llm.tool_context import ToolError, function_tool
from ....log import logger
from ....voice import RunContext
from ....utils.http_context import http_session
from .config import HTTPResponse, HTTPToolConfig, HTTPToolRequest
from .execution_utils import announce_execution
from .schema_utils import (
    extract_url_params,
    normalize_parameters_schema,
    sanitize_arguments,
    schema_from_params,
)


async def run_http_request(
    request: HTTPToolRequest,
    arguments: dict[str, Any],
) -> HTTPResponse:
    """Make HTTP request with given parameters.

    Args:
        request: HTTP request configuration
        arguments: Parameters to send

    Returns:
        HTTPResponse with success/failure details
    """
    method = request.method.upper()
    request_timeout = aiohttp.ClientTimeout(total=request.timeout_ms / 1000.0)

    # Extract URL template parameters and partition arguments
    url_param_names = extract_url_params(request.url_template)
    url_params = {k: v for k, v in arguments.items() if k in url_param_names}
    body_params = {k: v for k, v in arguments.items() if k not in url_param_names}

    # Substitute URL template parameters
    templated_url = request.url_template
    for param_name, param_value in url_params.items():
        templated_url = templated_url.replace(f":{param_name}", str(param_value))

    session = http_session()

    try:
        if method == "GET":
            resp_ctx = session.get(
                templated_url, params=body_params, headers=request.headers, timeout=request_timeout
            )
        elif method == "POST":
            resp_ctx = session.post(
                templated_url, json=body_params, headers=request.headers, timeout=request_timeout
            )
        elif method == "PUT":
            resp_ctx = session.put(
                templated_url, json=body_params, headers=request.headers, timeout=request_timeout
            )
        elif method == "PATCH":
            resp_ctx = session.patch(
                templated_url, json=body_params, headers=request.headers, timeout=request_timeout
            )
        elif method == "DELETE":
            resp_ctx = session.delete(
                templated_url, params=body_params, headers=request.headers, timeout=request_timeout
            )
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        async with resp_ctx as resp:
            body = await resp.text()
            success = 200 <= resp.status < 300

            response = HTTPResponse(
                success=success,
                status_code=resp.status,
                body=body,
                headers=dict(resp.headers),
                error=None if success else f"HTTP {resp.status}",
            )

            if not success:
                logger.warning(
                    "HTTP request returned error status",
                    extra={
                        "url": request.url_template,
                        "method": method,
                        "status_code": resp.status,
                        "response_size": len(body),
                    },
                )

            return response

    except asyncio.TimeoutError as e:
        logger.warning(
            "HTTP request timeout",
            extra={
                "url": request.url_template,
                "method": method,
                "timeout_ms": request.timeout_ms,
            },
        )
        raise ToolError(f"Request timeout after {request.timeout_ms}ms") from e
    except aiohttp.ClientError as e:
        logger.error(
            "HTTP request client error",
            extra={
                "url": request.url_template,
                "method": method,
                "error": str(e),
            },
        )
        raise ToolError(f"HTTP request failed: {str(e)}") from e
    except Exception as e:
        logger.exception(
            "unexpected error during HTTP request",
            extra={
                "url": request.url_template,
                "method": method,
            },
        )
        raise ToolError(f"Unexpected error: {str(e)}") from e


def create_http_tool(config: HTTPToolConfig) -> FunctionTool:
    """Create an HTTP request tool.

    The tool makes HTTP requests to a specified endpoint with configurable
    parameters, headers, and response handling.

    Args:
        config: Configuration for the HTTP tool

    Returns:
        FunctionTool that can be used in Agent.tools

    Raises:
        ValueError: If config is invalid (missing URL, invalid parameters schema)

    Example:
        ```python
        from livekit.agents.beta.tools.http import create_http_tool, HTTPToolConfig, HTTPToolParam

        # Minimal usage
        tool = create_http_tool(
            HTTPToolConfig(
                name="status",
                url="https://api.example.com/status",
            )
        )

        # With JSON schema
        config = HTTPToolConfig(
            name="create_ticket",
            description="Create a support ticket",
            url="https://api.example.com/tickets",
            method="POST",
            headers={"Authorization": "Bearer token"},
            json_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Ticket title"},
                    "priority": {"type": "string", "enum": ["low", "high"]}
                },
                "required": ["title"]
            }
        )
        tool = create_http_tool(config)

        # With URL templating using HTTPToolParam definitions
        update_task_tool = create_http_tool(
            HTTPToolConfig(
                name="update_task",
                url="https://api.example.com/tasks/:id",
                method="PUT",
                params=[
                    HTTPToolParam(name="id", description="the task id", required=True),  # as URL param
                    HTTPToolParam(name="title", description="new title"),  # as body param
                ],
            )
        )

        # With normalizer for custom response handling
        config_with_norm = HTTPToolConfig(
            name="get_weather",
            url="https://api.weather.com/current",
            method="GET",
            json_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            },
            output_normalizer=lambda resp: f"Temp: {resp.body}" if resp.success else None
        )
        weather_tool = create_http_tool(config_with_norm)

        # Use in agent
        agent = Agent(tools=[tool, update_task_tool, weather_tool])
        ```
    """
    if not config.url:
        logger.error("HTTP tool creation failed: missing URL")
        raise ValueError("URL is required in HTTPToolConfig")

    supported_methods = {"GET", "POST", "PUT", "PATCH", "DELETE"}
    method_upper = config.method.upper()
    if method_upper not in supported_methods:
        logger.error(
            "HTTP tool creation failed: unsupported method",
            extra={"method": config.method, "supported_methods": list(supported_methods)},
        )
        raise ValueError(
            f"Unsupported HTTP method: {config.method}. "
            f"Supported methods: {', '.join(sorted(supported_methods))}"
        )

    if config.json_schema is not None and config.params:
        raise ValueError("Cannot set both 'json_schema' and 'params' in HTTPToolConfig")

    if config.params is not None:
        parameters_schema = schema_from_params(config.params)
    else:
        parameters_schema = config.json_schema or {"type": "object", "properties": {}, "required": []}

    if not isinstance(parameters_schema, dict) or "type" not in parameters_schema:
        logger.error(
            "HTTP tool creation failed: invalid parameters schema",
            extra={"schema_type": type(parameters_schema).__name__},
        )
        raise ValueError("Invalid parameters schema - must be a dict with 'type' field")

    # Normalize parameters schema (convert string descriptions to full schema objects)
    parameters_schema = normalize_parameters_schema(parameters_schema)

    # Validate URL template parameters
    url_params = extract_url_params(config.url)
    if url_params:
        schema_properties = parameters_schema.get("properties", {})
        missing_params = url_params - set(schema_properties.keys())
        if missing_params:
            logger.error(
                "HTTP tool creation failed: URL template parameters missing from schema",
                extra={
                    "url": config.url,
                    "missing_params": list(missing_params),
                    "url_params": list(url_params),
                },
            )
            raise ValueError(
                f"URL template parameters missing from schema: {', '.join(sorted(missing_params))}"
            )

    if not config.name:
        raise ValueError("Tool name is required in HTTPToolConfig")

    tool_name = config.name
    tool_description = config.description

    logger.info(
        "creating HTTP tool",
        extra={
            "tool_name": tool_name,
            "url": config.url,
            "method": config.method,
            "timeout_ms": config.timeout_ms,
            "has_custom_headers": config.headers is not None,
            "has_execution_message": config.execution_message is not None,
            "has_output_normalizer": config.output_normalizer is not None,
            "uses_param_list": config.params is not None,
        },
    )

    raw_schema = {
        "name": tool_name,
        "parameters": parameters_schema,
    }
    if tool_description:
        raw_schema["description"] = tool_description

    request = HTTPToolRequest(
        url_template=config.url,
        method=config.method,
        headers=config.headers,
        timeout_ms=config.timeout_ms,
    )

    async def http_handler(raw_arguments: dict, context: RunContext) -> str | None:
        if config.execution_message:
            await announce_execution(tool_name, config.execution_message, context.session)

        if config.method.upper() != "GET":
            context.disallow_interruptions()
            # TODO (nikita): maybe we should restore it after the tool call is done?

        arguments = sanitize_arguments(raw_arguments, parameters_schema)

        response = await run_http_request(
            request=request,
            arguments=arguments,
        )

        if not response.success:
            error_body = response.body[:500] if len(response.body) > 500 else response.body
            raise ToolError(f"HTTP {response.status_code}: {error_body}")

        if config.output_normalizer:
            return config.output_normalizer(response)

        return response.body

    return function_tool(
        http_handler,
        raw_schema=raw_schema,
    )
