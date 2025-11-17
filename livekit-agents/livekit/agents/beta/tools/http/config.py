from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


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
class HTTPToolParam:
    """Typed parameter definition for simple HTTP tool schemas.

    Attributes:
        name: Parameter name (must be unique)
        description: Human-readable description
        type: Basic JSON schema type (string, number, integer, boolean)
        enum: Optional list of allowed string values (only valid for string type)
        required: Whether this parameter is required
    """

    name: str
    description: str
    type: Literal["string", "number", "integer", "boolean"] = "string"
    enum: list[str] | None = None
    required: bool = False


@dataclass
class HTTPToolRequest:
    """Simple request configuration used by the HTTP utility function."""

    url_template: str
    method: str = "GET"
    headers: dict[str, str] | None = None
    timeout_ms: int = 30000


@dataclass
class HTTPToolConfig:
    """Configuration for HTTP tool customization.

    Attributes:
        url: Endpoint URL (required). Supports URL templating with :param syntax.
            Example: "https://api.example.com/users/:user_id/posts/:post_id"
            Parameters matching :param names will be substituted into the URL,
            while remaining parameters will be sent as body/query parameters.
        name: Custom tool name (optional)
        description: Custom tool description (optional)
        method: HTTP method (default: "POST")
        headers: Optional dict of headers to send with request
        timeout_ms: Request timeout in milliseconds (default: 30000)
        json_schema: raw JSON schema dict for tool parameters (defaults to empty object schema).
            Supports shorthand syntax: "param": "description" is equivalent to
            "param": {"type": "string", "description": "description"}
        params: Optional list of HTTPToolParam for simple typed definitions.
            Cannot be combined with the `json_schema` dict. When provided, the
            tool will synthesize the JSON schema automatically.
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
    json_schema: dict[str, Any] | None = None
    params: list[HTTPToolParam] | None = None
    execution_message: str | None = None
    output_normalizer: Callable[[HTTPResponse], str | None] | None = None

