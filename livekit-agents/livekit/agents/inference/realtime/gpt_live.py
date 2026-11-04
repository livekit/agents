from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict, overload
from urllib.parse import urlencode

import aiohttp
from openai.types.responses import ResponseTextConfigParam
from openai.types.shared_params import Reasoning

from livekit.agents import llm
from livekit.agents.inference._utils import (
    HEADER_INFERENCE_PROVIDER,
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
)
from .._realtime_models import DuplexModels, OpenAIDuplexModels

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


class OpenAIResponsesOptions(TypedDict, total=False):
    """The backend Responses model GPT-Live delegates work to, under ``delegation="responses"``.

    A key left unset is not sent, and the service's own default applies.
    """

    model: str
    """An OpenAI LLM LiveKit Inference serves; ``gpt-5.6-luna`` when unset."""
    instructions: str
    """Instructions for the backend model, distinct from the voice model's."""
    tool_choice: llm.ToolChoice | None
    parallel_tool_calls: bool
    reasoning: Reasoning
    """Responses reasoning settings, for example ``{"effort": "medium"}``."""
    text: ResponseTextConfigParam
    """Responses text settings, for example ``{"verbosity": "low"}``."""
    max_output_tokens: int
    """Upper bound on the tokens one backend response may generate; at least 16."""


class OpenAIDuplexOptions(TypedDict, total=False):
    delegation: Literal["responses", "client"]
    """Where delegated work goes, fixed for the life of the session. ``responses`` (the default)
    runs it on a backend model, so ``@function_tool`` works as usual; ``client`` hands it to the
    application as a ``delegation_created`` event."""
    responses_options: OpenAIResponsesOptions


@dataclass
class _InferenceOptions:
    provider: str | None
    api_key: str
    api_secret: str


class DuplexModel(_GPTLiveModel):
    """Full-duplex voice model, served through LiveKit Inference."""

    @overload
    def __init__(
        self,
        model: OpenAIDuplexModels,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        voice: NotGivenOr[str] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[OpenAIDuplexOptions] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        pass

    @overload
    def __init__(
        self,
        model: str,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        voice: NotGivenOr[str] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        pass

    def __init__(
        self,
        model: DuplexModels | str,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        voice: NotGivenOr[str] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[OpenAIDuplexOptions | dict[str, Any]] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            model: Provider-prefixed duplex model, for example ``openai/gpt-live-1``.
            provider: Host that must serve the model, e.g. ``"openai"``. The session is
                refused if no host in the region matches.
            voice: Output voice, ``marin`` by default for GPT-Live.
            extra_kwargs: Model-specific options; :class:`OpenAIDuplexOptions` for GPT-Live.
                Delegated work runs on the default service tier only.
            http_session: Optional shared HTTP session.
            max_session_duration: Seconds before the connection is recycled.
            conn_options: Retry/backoff and connection settings.
        """
        if "/" not in model:
            raise ValueError("model must be provider-prefixed, for example 'openai/gpt-live-1'")
        options: dict[str, Any] = dict(extra_kwargs) if is_given(extra_kwargs) else {}
        if unknown := options.keys() - OpenAIDuplexOptions.__optional_keys__:
            raise ValueError(f"unsupported extra_kwargs for {model}: {sorted(unknown)}")
        responses_options = options.get("responses_options")
        if responses_options and responses_options.get("service_tier") not in (None, "default"):
            raise ValueError("LiveKit Inference runs GPT-Live delegation on the default tier only")

        resolved_api_key, resolved_api_secret = resolve_credentials(api_key, api_secret)

        super().__init__(
            model=model,
            voice=voice if is_given(voice) else DEFAULT_VOICE,
            delegation=options.get("delegation", "responses"),
            responses_options=responses_options if responses_options is not None else NOT_GIVEN,
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
        )
        self._provider_label = "LiveKit Inference Duplex"

    @classmethod
    def from_model_string(cls, model: str) -> DuplexModel:
        """Create a DuplexModel instance from a model string"""
        return cls(model)

    @property
    def provider(self) -> str:
        return "livekit"

    def session(self) -> DuplexSession:
        return DuplexSession(self)


class DuplexSession(_GPTLiveSession):
    def __init__(self, duplex_model: DuplexModel) -> None:
        self._inference_model = duplex_model
        super().__init__(duplex_model)

    def _create_ws_url_and_headers(self) -> tuple[str, dict[str, str]]:
        opts = self._inference_model._inference_opts
        url, _ = super()._create_ws_url_and_headers()
        url = f"{url}?{urlencode({'model': self._opts.model})}"
        headers = get_inference_headers()
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
