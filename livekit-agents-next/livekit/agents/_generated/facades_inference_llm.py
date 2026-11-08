# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core
from ..types import NOT_GIVEN
from ..utils import is_given

if TYPE_CHECKING:
    from ..types import NotGivenOr
    from .inference_llm import LLMModels
    from .llm_provider_options import ChatCompletionOptions


class LLM:
    """`inference.LLM`'s client, built by its constructor."""

    if TYPE_CHECKING:
        _native: _core.InferenceLlm

        def _core_native(self) -> _core.InferenceLlm: ...

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
