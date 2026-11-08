# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from .. import _bindings, _core

if TYPE_CHECKING:
    from .connect import APIConnectOptions
    from .inference_tts import FallbackModel
    from .language import LanguageCode


class TTS:
    """`inference.TTS`'s client, built on first use, after a fork, from the TTS's attributes."""

    if TYPE_CHECKING:
        _model: str
        _base_url: str
        _api_key: str
        _api_secret: str
        _conn_options: APIConnectOptions
        _voice: str | None
        _language: LanguageCode | None
        _encoding: str
        _sample_rate: int
        _extra_kwargs: dict[str, Any]
        _fallback: list[FallbackModel]

        def _core_native(self) -> _core.InferenceTts: ...

    @functools.cached_property
    def _native(self) -> _core.InferenceTts:
        return _core.InferenceTts(
            self._model,
            base_url=self._base_url,
            api_key=self._api_key,
            api_secret=self._api_secret,
            conn_options=self._conn_options,
            voice=self._voice,
            language=self._language,
            encoding=self._encoding,
            sample_rate=self._sample_rate,
            extra_kwargs=self._extra_kwargs,
            fallback=self._fallback,
        )

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, TTS, "synthesize", "stream")
        return self if replaced else self._core_native()
