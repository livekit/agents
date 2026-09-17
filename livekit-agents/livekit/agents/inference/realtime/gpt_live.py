from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import urlencode, urlparse, urlunparse

import aiohttp

from livekit.agents import llm
from livekit.agents.inference._utils import (
    HEADER_INFERENCE_PROVIDER,
    InferenceClass,
    create_access_token,
    get_default_inference_url,
    get_inference_headers,
    resolve_credentials,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ...llm._realtime import gpt_live_types as types
from ...llm._realtime.gpt_live import (
    DEFAULT_VOICE,
    GPTLiveModel as _GPTLiveModel,
    GPTLiveSession as _GPTLiveSession,
    GPTLiveVoices,
    ResponsesDelegationOptions,
    _ResponsesDelegationOptionsBase,
)
from ...llm._realtime.openai_tools import OpenAITool


class GPTLiveResponsesDelegationOptions(_ResponsesDelegationOptionsBase, total=False):
    """Responses options supported through LiveKit Inference.

    A key left unset is not sent, and the service's own default applies.
    Only the default service tier is available through LiveKit Inference.
    """

    service_tier: Literal["default"]


_GATEWAY_FATAL_ERROR_CODES = frozenset(
    {
        "invalid_event",
        "session_start_required",
        "invalid_session",
        "invalid_model",
        "invalid_delegated_model",
        "unsupported_delegated_model",
        "unsupported_server_tool",
        "unsupported_service_tier",
    }
)


@dataclass
class _GPTLiveOptions:
    provider: str | None
    api_key: str
    api_secret: str
    inference_class: InferenceClass | None


class GPTLiveModel(_GPTLiveModel):
    """Native GPT-Live through LiveKit Inference.

    Responses delegation supports function tools and the default service tier.
    Separately priced OpenAI-hosted tools and service tiers are not available.
    """

    def __init__(
        self,
        model: str,
        *,
        provider: str | None = None,
        voice: GPTLiveVoices | str | dict[str, Any] = DEFAULT_VOICE,
        delegation: types.DelegationTarget = "responses",
        responses_options: NotGivenOr[GPTLiveResponsesDelegationOptions] = NOT_GIVEN,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        inference_class: InferenceClass | None = None,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        if "/" not in model:
            raise ValueError("model must be provider-prefixed, for example 'openai/gpt-live-1'")
        if (
            is_given(responses_options)
            and "service_tier" in responses_options
            and responses_options["service_tier"] != "default"
        ):
            raise ValueError("LiveKit Inference GPT-Live supports only service_tier='default'")

        resolved_api_key, resolved_api_secret = resolve_credentials(api_key, api_secret)

        super().__init__(
            model=model,
            voice=voice,
            delegation=delegation,
            responses_options=cast(NotGivenOr[ResponsesDelegationOptions], responses_options),
            api_key="livekit-inference",
            base_url=base_url or get_default_inference_url(),
            http_session=http_session,
            max_session_duration=max_session_duration,
            conn_options=conn_options,
        )
        self._inference_opts = _GPTLiveOptions(
            provider=provider,
            api_key=resolved_api_key,
            api_secret=resolved_api_secret,
            inference_class=inference_class,
        )
        self._provider_label = "LiveKit Inference GPT-Live"

    @property
    def provider(self) -> str:
        return "livekit"

    def session(self) -> GPTLiveSession:
        return GPTLiveSession(self)


class GPTLiveSession(_GPTLiveSession):
    def __init__(self, duplex_model: GPTLiveModel) -> None:
        self._inference_model = duplex_model
        super().__init__(duplex_model)

    def _create_ws_url_and_headers(self) -> tuple[str, dict[str, str]]:
        url, _ = super()._create_ws_url_and_headers()
        parsed = urlparse(url)
        url = urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                "",
                urlencode({"model": self._opts.model}),
                "",
            )
        )

        opts = self._inference_model._inference_opts
        headers = get_inference_headers(inference_class=opts.inference_class)
        headers["Authorization"] = f"Bearer {create_access_token(opts.api_key, opts.api_secret)}"
        if opts.provider:
            headers[HEADER_INFERENCE_PROVIDER] = opts.provider
        return url, headers

    async def _update_tools(self, tools: list[llm.Tool]) -> None:
        provider_tools = llm.ToolContext(tools).provider_tools
        if any(isinstance(tool, OpenAITool) for tool in provider_tools):
            raise llm.RealtimeError(
                "LiveKit Inference GPT-Live does not support OpenAI-hosted tools"
            )
        await super()._update_tools(tools)

    def _is_fatal_error(self, error: types.ErrorBody) -> bool:
        code = error.code or error.type or ""
        return (
            not self._session_started_fut.done() and code in _GATEWAY_FATAL_ERROR_CODES
        ) or super()._is_fatal_error(error)


# Compatibility names retained for the former plugin API.
InferenceGPTLiveModel = GPTLiveModel
InferenceGPTLiveSession = GPTLiveSession
InferenceResponsesDelegationOptions = GPTLiveResponsesDelegationOptions
