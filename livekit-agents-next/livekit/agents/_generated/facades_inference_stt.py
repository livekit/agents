# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from .. import _bindings, _core

if TYPE_CHECKING:
    from livekit.agents import VAD

    from .connect import APIConnectOptions
    from .inference_stt import FallbackModel
    from .language import LanguageCode


class STT:
    """`inference.STT`'s client, built on first use, after a fork, from the STT's attributes."""

    if TYPE_CHECKING:
        _model: str | None
        _language: LanguageCode | None
        _encoding: str
        _sample_rate: int
        _base_url: str
        _api_key: str
        _api_secret: str
        _extra_kwargs: dict[str, Any]
        _fallback: list[FallbackModel]
        _conn_options: APIConnectOptions
        _vad: VAD[Any] | None

        def _core_native(self) -> _core.InferenceStt: ...

    @functools.cached_property
    def _native(self) -> _core.InferenceStt:
        return _core.InferenceStt(
            model=self._model,
            language=self._language,
            encoding=self._encoding,
            sample_rate=self._sample_rate,
            base_url=self._base_url,
            api_key=self._api_key,
            api_secret=self._api_secret,
            extra_kwargs=self._extra_kwargs,
            fallback=self._fallback,
            conn_options=self._conn_options,
            vad=self._vad,
        )

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, STT, "stream")
        return self if replaced else self._core_native()
