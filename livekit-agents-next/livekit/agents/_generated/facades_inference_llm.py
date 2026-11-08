# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from .. import _bindings, _core
from ..types import NOT_GIVEN
from ..utils import is_given

if TYPE_CHECKING:
    from ..types import NotGivenOr
    from .inference_llm import LLMModels
    from .llm_provider_options import ChatCompletionOptions


class LLM:
    """`inference.LLM`'s client, built on first use, after a fork, from the LLM's attributes."""

    if TYPE_CHECKING:
        _model: str
        _provider: str | None
        _base_url: str
        _api_key: str
        _api_secret: str | None
        _label: str
        _inference_class: str | None
        _strict_tool_schema: bool
        _gzip_body: bool
        _msgpack_body: bool
        _extra_kwargs: dict[str, object]

        def _core_native(self) -> _core.InferenceLlm: ...

    @functools.cached_property
    def _native(self) -> _core.InferenceLlm:
        return _core.InferenceLlm(
            self._model,
            provider=self._provider,
            base_url=self._base_url,
            api_key=self._api_key,
            api_secret=self._api_secret,
            label=self._label,
            inference_class=self._inference_class,
            strict_tool_schema=self._strict_tool_schema,
            gzip_body=self._gzip_body,
            msgpack_body=self._msgpack_body,
            extra_kwargs=self._extra_kwargs,
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
