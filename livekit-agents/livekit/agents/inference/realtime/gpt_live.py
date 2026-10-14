from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

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
from livekit.agents.utils.misc import is_given

from ...llm._realtime import gpt_live_types as types
from ...llm._realtime.gpt_live import (
    DEFAULT_VOICE,
    GPTLiveModel as _GPTLiveModel,
    GPTLiveSession as _GPTLiveSession,
    GPTLiveVoices,
    ResponsesDelegationOptions,
)
from .._realtime_models import OpenAIGPTLiveModels

DEFAULT_MODEL = "openai/gpt-live-1"

# the gateway refuses a session.start it cannot run; reconnecting resends the same one
_FATAL_ERROR_CODES = frozenset(
    {
        "invalid_model",
        "unsupported_server_tool",
        "unsupported_service_tier",
        "invalid_delegated_model",
        "unsupported_delegated_model",
    }
)


@dataclass
class _InferenceOptions:
    provider: str | None
    api_key: str
    api_secret: str
    inference_class: InferenceClass | None


class GPTLiveModel(_GPTLiveModel):
    """OpenAI GPT-Live full-duplex voice model, served through LiveKit Inference."""

    def __init__(
        self,
        model: OpenAIGPTLiveModels | str = DEFAULT_MODEL,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        inference_class: InferenceClass | None = None,
        voice: GPTLiveVoices | str | dict[str, Any] = DEFAULT_VOICE,
        delegation: types.DelegationTarget = "responses",
        responses_options: NotGivenOr[ResponsesDelegationOptions] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            model: Provider-prefixed GPT-Live model, ``openai/gpt-live-1`` by default.
            provider: Host that must serve the model, e.g. ``"openai"``. The session is
                refused if no host in the region matches.
            inference_class: Has no effect on GPT-Live sessions.
            voice: Output voice, as in :class:`livekit.plugins.openai.realtime.GPTLiveModel`.
            delegation: Where delegated work goes, as in
                :class:`livekit.plugins.openai.realtime.GPTLiveModel`.
            responses_options: The backend Responses model under ``delegation="responses"``.
                Its ``model`` must be an OpenAI LLM LiveKit Inference serves, and only the
                ``default`` ``service_tier`` is billed, so no other is accepted.
            http_session: Optional shared HTTP session.
            max_session_duration: Seconds before the connection is recycled.
            conn_options: Retry/backoff and connection settings.
        """
        if "/" not in model:
            raise ValueError("model must be provider-prefixed, for example 'openai/gpt-live-1'")
        if is_given(responses_options) and responses_options.get("service_tier") not in (
            None,
            "default",
        ):
            raise ValueError("LiveKit Inference runs GPT-Live delegation on the default tier only")

        resolved_api_key, resolved_api_secret = resolve_credentials(api_key, api_secret)

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
        self._inference_opts = _InferenceOptions(
            provider=provider,
            api_key=resolved_api_key,
            api_secret=resolved_api_secret,
            inference_class=inference_class,
        )
        self._provider_label = "LiveKit Inference GPT-Live"

    @classmethod
    def from_model_string(cls, model: str) -> GPTLiveModel:
        """Create a GPTLiveModel instance from a model string"""
        return cls(model)

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
        opts = self._inference_model._inference_opts
        url, _ = super()._create_ws_url_and_headers()
        url = f"{url}?{urlencode({'model': self._opts.model})}"
        headers = get_inference_headers(inference_class=opts.inference_class)
        # signed per connection, so a reconnect never presents an expired token
        headers["Authorization"] = f"Bearer {create_access_token(opts.api_key, opts.api_secret)}"
        if opts.provider:
            headers[HEADER_INFERENCE_PROVIDER] = opts.provider
        return url, headers

    async def _update_tools(self, tools: list[llm.Tool]) -> None:
        hosted = sorted(
            tool.id
            for tool in llm.ToolContext(tools).flatten()
            if isinstance(tool, llm.ProviderTool)
        )
        if hosted:
            # hosted tools bill separately from the delegated tokens the gateway meters
            raise llm.RealtimeError(
                f"LiveKit Inference does not run hosted tools on GPT-Live, so the model can never "
                f"call {hosted}. Use function tools, or the OpenAI plugin's GPTLiveModel directly."
            )
        await super()._update_tools(tools)

    def _is_fatal_error(self, error: types.ErrorBody) -> bool:
        return (error.code or "") in _FATAL_ERROR_CODES or super()._is_fatal_error(error)
