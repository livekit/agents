import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import aiohttp

from .... import FunctionTool
from ....llm.tool_context import ToolError, function_tool
from ....log import logger
from ....voice import AgentSession, RunContext


@dataclass
class HTTPResponse:
    """Structured response from HTTP request.

    Attributes:
        success: Whether the request was successful (2xx status)
        status_code: HTTP status code
        body: Response body as string
        headers: Response headers
        error: Error message (if failed)
    """

    success: bool
    status_code: int
    body: str
    headers: dict[str, str]
    error: str | None = None


@dataclass
class HTTPToolConfig:
    """Configuration for HTTP tool customization.

    Attributes:
        url: Endpoint URL (required)
        name: Custom tool name (optional)
        description: Custom tool description (optional)
        method: HTTP method (default: "POST")
        headers: Optional dict of headers to send with request
        timeout_ms: Request timeout in milliseconds (default: 30000)
        parameters: JSON schema dict for tool parameters (defaults to empty object schema)
        execution_message: Optional message to announce before executing (spoken by agent)
        output_normalizer: Optional function to transform HTTPResponse to string or None
            - Return string (including ""): agent uses it in next response
            - Return None: agent stays silent (good for background calls)
    """

    url: str
    name: str | None = None
    description: str | None = None
    method: str = "POST"
    headers: dict[str, str] | None = None
    timeout_ms: int = 30000
    parameters: dict[str, Any] | None = None
    execution_message: str | None = None
    output_normalizer: Callable[[HTTPResponse], str | None] | None = None


async def _announce_execution(tool_name: str, message: str, session: AgentSession) -> None:
    """Announce execution message via agent session.

    Args:
        tool_name: Name of the tool being executed
        message: Message to announce
        session: AgentSession from RunContext
    """
    try:
        await session.generate_reply(
            instructions=f"You are running {tool_name} tool (do not announce the tool name) for user and should announce it using this instruction: {message}",
            allow_interruptions=False,
        )
    except Exception as e:
        logger.debug(
            "failed to announce execution for tool",
            extra={"tool_name": tool_name, "error": str(e)},
        )


async def _make_http_request(
    url: str,
    method: str,
    headers: dict[str, str] | None,
    timeout_ms: int,
    arguments: dict[str, Any],
) -> HTTPResponse:
    """Make HTTP request with given parameters.

    Args:
        url: Endpoint URL
        method: HTTP method
        headers: Request headers
        timeout_ms: Timeout in milliseconds
        arguments: Parameters to send

    Returns:
        HTTPResponse with success/failure details
    """
    method = method.upper()
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000.0)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            if method == "GET":
                resp_ctx = session.get(url, params=arguments, headers=headers)
            elif method == "POST":
                resp_ctx = session.post(url, json=arguments, headers=headers)
            elif method == "PUT":
                resp_ctx = session.put(url, json=arguments, headers=headers)
            elif method == "PATCH":
                resp_ctx = session.patch(url, json=arguments, headers=headers)
            else:  # DELETE
                resp_ctx = session.delete(url, params=arguments, headers=headers)

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
                            "url": url,
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
                    "url": url,
                    "method": method,
                    "timeout_ms": timeout_ms,
                },
            )
            raise ToolError(f"Request timeout after {timeout_ms}ms") from e
        except aiohttp.ClientError as e:
            logger.error(
                "HTTP request client error",
                extra={
                    "url": url,
                    "method": method,
                    "error": str(e),
                },
            )
            raise ToolError(f"HTTP request failed: {str(e)}") from e
        except Exception as e:
            logger.exception(
                "unexpected error during HTTP request",
                extra={
                    "url": url,
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
        from livekit.agents.beta.tools.http import create_http_tool, HTTPToolConfig

        # Minimal usage
        tool = create_http_tool(HTTPToolConfig(
            url="https://api.example.com/status"
        ))

        # With parameters
        config = HTTPToolConfig(
            name="create_ticket",
            description="Create a support ticket",
            url="https://api.example.com/tickets",
            method="POST",
            headers={"Authorization": "Bearer token"},
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Ticket title"},
                    "priority": {"type": "string", "enum": ["low", "high"]}
                },
                "required": ["title"]
            }
        )
        tool = create_http_tool(config)

        # With normalizer for custom response handling
        config_with_norm = HTTPToolConfig(
            name="get_weather",
            url="https://api.weather.com/current",
            method="GET",
            parameters={
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
        agent = Agent(tools=[tool, weather_tool])
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

    parameters_schema = config.parameters or {"type": "object", "properties": {}, "required": []}

    if not isinstance(parameters_schema, dict) or "type" not in parameters_schema:
        logger.error(
            "HTTP tool creation failed: invalid parameters schema",
            extra={"schema_type": type(parameters_schema).__name__},
        )
        raise ValueError("Invalid parameters schema - must be a dict with 'type' field")

    tool_name = config.name or "http_request"
    tool_description = config.description or f"Make {config.method} request to {config.url}"

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
        },
    )

    raw_schema = {
        "name": tool_name,
        "description": tool_description,
        "parameters": parameters_schema,
    }

    async def http_handler(raw_arguments: dict, context: RunContext) -> str | None:
        if config.execution_message:
            await _announce_execution(tool_name, config.execution_message, context.session)

        response = await _make_http_request(
            url=config.url,
            method=config.method,
            headers=config.headers,
            timeout_ms=config.timeout_ms,
            arguments=raw_arguments,
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
