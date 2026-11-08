# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core
from ..types import NOT_GIVEN
from ..utils import is_given
from .llm import LLM as BaseLLM

if TYPE_CHECKING:
    from ..types import NotGivenOr
    from .inference_llm import LLMModels
    from .inference_models import InferenceClass
    from .llm_provider_options import ChatCompletionOptions, ProviderEndpoint


class LLM(BaseLLM):
    """`inference.LLM`: the LiveKit Inference gateway's LLM, or an OpenAI-compatible provider's."""

    if TYPE_CHECKING:
        def _core_native(self) -> _core.InferenceLlm: ...

    def __init__(
        self,
        model: LLMModels | str,
        *,
        provider: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        inference_class: NotGivenOr[InferenceClass] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[ChatCompletionOptions | dict[str, Any]] = NOT_GIVEN,
        endpoint: NotGivenOr[ProviderEndpoint] = NOT_GIVEN,
    ) -> None:
        """A LiveKit Inference LLM.

        Args:
            model: The model, ``"provider/model"``.
            provider: The provider the gateway routes to; the gateway picks by default.
            base_url: The gateway URL; ``LIVEKIT_INFERENCE_URL`` or the cloud gateway by default.
            api_key: ``LIVEKIT_INFERENCE_API_KEY`` or ``LIVEKIT_API_KEY`` by default.
            api_secret: ``LIVEKIT_INFERENCE_API_SECRET`` or ``LIVEKIT_API_SECRET`` by default.
            inference_class: The scheduling class of its requests.
            extra_kwargs: Request options sent with every completion.
            endpoint: A provider plugin's own endpoint in place of the gateway.
        """
        super().__init__()
        self._native = _core.InferenceLlm(
            model,
            provider=provider if is_given(provider) else None,
            base_url=base_url if is_given(base_url) else None,
            api_key=api_key if is_given(api_key) else None,
            api_secret=api_secret if is_given(api_secret) else None,
            inference_class=inference_class if is_given(inference_class) else None,
            extra_kwargs=dict(extra_kwargs) if is_given(extra_kwargs) else None,
            endpoint=endpoint if is_given(endpoint) else None,
        )

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, LLM, "chat")
        return self if replaced else self._core_native()

    @property
    def model(self) -> str:
        return self._native.model

    def update_options(
        self,
        *,
        model: NotGivenOr[LLMModels | str] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[ChatCompletionOptions | dict[str, Any]] = NOT_GIVEN,
    ) -> None:
        """Read by the next ``chat()``. ``extra_kwargs`` replaces the persistent kwargs."""
        self._native.update_options(
            model=model if is_given(model) else None,
            extra_kwargs=dict(extra_kwargs) if is_given(extra_kwargs) else None,
        )
