from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlencode, urlparse, urlunparse

import aiohttp

from livekit.agents.inference._utils import (
    HEADER_INFERENCE_PROVIDER,
    create_access_token,
    get_default_inference_url,
    get_inference_headers,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from . import gpt_live_types as types
from .gpt_live_model import (
    DEFAULT_VOICE,
    GPTLiveModel,
    GPTLiveSession,
    GPTLiveVoices,
    _ResponsesDelegationOptionsBase,
)

InferenceClass = Literal["priority", "standard", "low"]


class InferenceResponsesDelegationOptions(_ResponsesDelegationOptionsBase, total=False):
    """Responses options supported through LiveKit Inference."""

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
class _InferenceGPTLiveOptions:
    provider: str | None
    api_key: str
    api_secret: str
    inference_class: InferenceClass | None


class InferenceGPTLiveModel(GPTLiveModel):
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
        responses_options: NotGivenOr[InferenceResponsesDelegationOptions] = NOT_GIVEN,
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

        resolved_api_key = api_key or os.getenv(
            "LIVEKIT_INFERENCE_API_KEY", os.getenv("LIVEKIT_API_KEY", "")
        )
        if not resolved_api_key:
            raise ValueError(
                "api_key is required, either as argument or set LIVEKIT_API_KEY environmental variable"
            )

        resolved_api_secret = api_secret or os.getenv(
            "LIVEKIT_INFERENCE_API_SECRET", os.getenv("LIVEKIT_API_SECRET", "")
        )
        if not resolved_api_secret:
            raise ValueError(
                "api_secret is required, either as argument or set LIVEKIT_API_SECRET environmental variable"
            )

        super().__init__(
            model=model,
            voice=voice,
            delegation=delegation,
            responses_options=responses_options,
            api_key="livekit-inference",
            base_url=base_url or get_default_inference_url(),
            http_session=http_session,
            max_session_duration=max_session_duration,
            conn_options=conn_options,
        )
        self._inference_opts = _InferenceGPTLiveOptions(
            provider=provider,
            api_key=resolved_api_key,
            api_secret=resolved_api_secret,
            inference_class=inference_class,
        )
        self._provider_label = "LiveKit Inference GPT-Live"

    @property
    def provider(self) -> str:
        return "livekit"

    def session(self) -> InferenceGPTLiveSession:
        return InferenceGPTLiveSession(self)


class InferenceGPTLiveSession(GPTLiveSession):
    def __init__(self, duplex_model: InferenceGPTLiveModel) -> None:
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

    def _is_fatal_error(self, error: types.ErrorBody) -> bool:
        code = error.code or error.type or ""
        return (
            self._session_id is None and code in _GATEWAY_FATAL_ERROR_CODES
        ) or super()._is_fatal_error(error)
