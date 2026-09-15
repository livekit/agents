# Copyright 2026 LiveKit, Inc.
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

"""Vertex AI Model Garden (AI Platform) LLM integration.

Targets self-deployed Model Garden endpoints that expose an OpenAI-compatible
chat completions API at:

    {ENDPOINT_DNS}/{API_VERSION}/projects/{PROJECT}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}/chat/completions

The dedicated endpoint DNS comes from ``Endpoint.dedicated_endpoint_dns`` on a
deployed Model Garden endpoint (for example
``mg-endpoint-<id>.us-central1-<project_number>.prediction.vertexai.goog``).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass
from typing import Any, Literal

import httpx
import openai
from openai.types.chat import ChatCompletionToolChoiceOptionParam, completion_create_params

import google.auth
import google.auth.credentials
import google.auth.transport.requests
from livekit.agents import llm
from livekit.agents.inference.llm import LLMStream
from livekit.agents.llm import ToolChoice, utils as llm_utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

ApiVersion = Literal["v1", "v1beta1"]


class _GoogleBearerAuth(httpx.Auth):
    """httpx auth handler that injects a Google OAuth bearer token.

    Accepts either a static token (useful for short-lived testing) or
    ``google.auth.credentials.Credentials``, which are refreshed lazily
    when their access token is missing or expired. A custom callable can
    also be supplied for cases where the caller manages tokens itself.
    """

    requires_request_body = False
    requires_response_body = False

    def __init__(
        self,
        *,
        credentials: google.auth.credentials.Credentials | None = None,
        static_token: str | None = None,
        token_provider: Callable[[], str] | None = None,
    ) -> None:
        if not credentials and not static_token and not token_provider:
            raise ValueError("one of credentials, static_token, or token_provider must be provided")
        self._credentials = credentials
        self._static_token = static_token
        self._token_provider = token_provider
        self._refresh_request = (
            google.auth.transport.requests.Request() if credentials is not None else None
        )

    def _current_token(self) -> str:
        if self._token_provider is not None:
            return self._token_provider()
        if self._credentials is not None:
            # Credentials.valid is False when token is missing or expired.
            if not self._credentials.valid:
                assert self._refresh_request is not None
                self._credentials.refresh(self._refresh_request)  # type: ignore[no-untyped-call]
            return self._credentials.token or ""
        return self._static_token or ""

    def sync_auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self._current_token()}"
        yield request

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        request.headers["Authorization"] = f"Bearer {self._current_token()}"
        yield request


@dataclass
class _AIPlatformOptions:
    model: str
    temperature: NotGivenOr[float]
    top_p: NotGivenOr[float]
    max_completion_tokens: NotGivenOr[int]
    parallel_tool_calls: NotGivenOr[bool]
    tool_choice: NotGivenOr[ToolChoice]
    extra_body: NotGivenOr[dict[str, Any]]
    extra_headers: NotGivenOr[dict[str, str]]
    extra_query: NotGivenOr[dict[str, str]]


class AIPlatformLLM(llm.LLM):
    """LLM that talks to a self-deployed Vertex AI Model Garden chat-completions endpoint.

    Example:
        ```python
        llm = AIPlatformLLM(
            endpoint_url="https://mg-endpoint-<id>.us-central1-<projnum>.prediction.vertexai.goog",
            project="my-project",
            endpoint_id="12345678-abcd-1234-abcd-1234567890ab",
            location="us-central1",
            model="google/gemma-4-31b-it",
        )
        ```
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        project: str,
        endpoint_id: str,
        location: str = "us-central1",
        model: str = "gemma",
        access_token: NotGivenOr[str] = NOT_GIVEN,
        credentials: google.auth.credentials.Credentials | None = None,
        token_provider: Callable[[], str] | None = None,
        api_version: ApiVersion = "v1beta1",
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_body: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        extra_headers: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        extra_query: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        strict_tool_schema: bool = True,
        client: openai.AsyncClient | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        """Create a new AIPlatformLLM.

        Args:
            endpoint_url: Base DNS for the dedicated Model Garden endpoint
                (no path component), e.g.
                ``https://mg-endpoint-<id>.us-central1-<projnum>.prediction.vertexai.goog``.
            project: Google Cloud project (id or number) that owns the endpoint.
            endpoint_id: The numeric or UUID endpoint id.
            location: GCP region (defaults to ``us-central1``).
            model: Model name passed in the chat completions request body.
            access_token: Optional static OAuth access token. If omitted and
                ``credentials``/``token_provider`` are not given, falls back to
                ``google.auth.default(scopes=["…/cloud-platform"])``.
            credentials: ``google.auth.credentials.Credentials`` instance. The
                auth handler will refresh it on demand.
            token_provider: Callable returning a fresh bearer token. Takes
                precedence over ``credentials`` and ``access_token``.
            api_version: ``v1`` or ``v1beta1``. Defaults to ``v1beta1``, which
                is currently the public-documented version for
                ``projects.locations.endpoints.chat.completions``.
            strict_tool_schema: When ``True`` (default), emits OpenAI-style
                strict JSON-schema function descriptions. Set to ``False`` for
                self-deployed OSS models that don't accept strict schemas.
            client: Pre-built ``openai.AsyncClient``. When provided, all
                auth/base-url construction is bypassed and the caller is
                responsible for those concerns.
        """
        super().__init__()

        self._opts = _AIPlatformOptions(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            extra_body=extra_body,
            extra_headers=extra_headers,
            extra_query=extra_query,
        )
        self._strict_tool_schema = strict_tool_schema

        if client is not None:
            self._owns_client = False
            self._client = client
        else:
            resolved_credentials = credentials
            resolved_token = access_token if is_given(access_token) else None
            if token_provider is None and resolved_credentials is None and resolved_token is None:
                # Fall back to application default credentials.
                resolved_credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )

            auth = _GoogleBearerAuth(
                credentials=resolved_credentials,
                static_token=resolved_token,
                token_provider=token_provider,
            )

            base_url = (
                f"{endpoint_url.rstrip('/')}"
                f"/{api_version}/projects/{project}"
                f"/locations/{location}/endpoints/{endpoint_id}"
            )

            self._owns_client = True
            self._client = openai.AsyncClient(
                api_key="ignored-auth-comes-from-httpx-auth",
                base_url=base_url,
                max_retries=0,
                http_client=httpx.AsyncClient(
                    auth=auth,
                    timeout=timeout
                    if timeout is not None
                    else httpx.Timeout(connect=10.0, read=10.0, write=10.0, pool=5.0),
                    follow_redirects=True,
                    limits=httpx.Limits(
                        max_connections=50,
                        max_keepalive_connections=50,
                        keepalive_expiry=120,
                    ),
                ),
            )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Vertex AI Model Garden"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[
            completion_create_params.ResponseFormat | type[llm_utils.ResponseFormatT]
        ] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        extra: dict[str, Any] = {}
        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        if is_given(self._opts.extra_body):
            extra["extra_body"] = self._opts.extra_body
        if is_given(self._opts.extra_headers):
            extra["extra_headers"] = self._opts.extra_headers
        if is_given(self._opts.extra_query):
            extra["extra_query"] = self._opts.extra_query

        if is_given(self._opts.temperature):
            extra["temperature"] = self._opts.temperature
        if is_given(self._opts.top_p):
            extra["top_p"] = self._opts.top_p
        if is_given(self._opts.max_completion_tokens):
            extra["max_completion_tokens"] = self._opts.max_completion_tokens

        parallel_tool_calls = (
            parallel_tool_calls if is_given(parallel_tool_calls) else self._opts.parallel_tool_calls
        )
        if is_given(parallel_tool_calls):
            extra["parallel_tool_calls"] = parallel_tool_calls

        tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice
        if is_given(tool_choice):
            oai_tool_choice: ChatCompletionToolChoiceOptionParam
            if isinstance(tool_choice, dict):
                oai_tool_choice = {
                    "type": "function",
                    "function": {"name": tool_choice["function"]["name"]},
                }
                extra["tool_choice"] = oai_tool_choice
            elif tool_choice in ("auto", "required", "none"):
                extra["tool_choice"] = tool_choice

        if is_given(response_format):
            extra["response_format"] = llm_utils.to_openai_response_format(response_format)  # type: ignore[arg-type]

        return LLMStream(
            self,
            model=self._opts.model,
            provider=None,
            inference_class=None,
            strict_tool_schema=self._strict_tool_schema,
            client=self._client,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
            provider_fmt="openai",
        )


__all__ = ["AIPlatformLLM"]
