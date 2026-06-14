"""Per-language ``unlikely`` thresholds for the mini detector."""

from __future__ import annotations

from typing import Literal, cast

from ..._exceptions import APIError
from ...language import LanguageCode
from ...types import NOT_GIVEN, NotGivenOr
from ...utils.misc import is_given

LOCAL_LANGUAGES: dict[str, float] = {
    "ar": 0.3500,
    "de": 0.2450,
    "en": 0.3600,
    "es": 0.3500,
    "fr": 0.2850,
    "hi": 0.3050,
    "id": 0.3450,
    "it": 0.2300,
    "ja": 0.2950,
    "ko": 0.4000,
    "nl": 0.2000,
    "pt": 0.3200,
    "tr": 0.2550,
    "zh": 0.3550,
}

TurnDetectorModels = Literal["turn-detector-v1", "turn-detector-v1-mini"]
TurnDetectorVersions = Literal["v1", "v1-mini"]


def _normalize_overrides(
    overrides: NotGivenOr[float | dict[LanguageCode | str, float]],
) -> NotGivenOr[float | dict[str, float]]:
    if not is_given(overrides) or not isinstance(overrides, dict):
        return overrides
    return {LanguageCode(k).language: float(v) for k, v in overrides.items()}


class ThresholdOptions:
    def __init__(
        self,
        model: TurnDetectorModels,
        overrides: NotGivenOr[float | dict[LanguageCode | str, float]] = NOT_GIVEN,
    ) -> None:
        self._model = model
        self._overrides = _normalize_overrides(overrides)

        # server/shipped defaults
        self._server_thresholds: dict[str, float] | None = None
        self._server_default: float | None = None
        if model == "turn-detector-v1-mini":
            self._server_thresholds = dict(LOCAL_LANGUAGES)
            self._server_default = LOCAL_LANGUAGES["en"]

        # materialized values
        self._thresholds: dict[str, float] = {}
        self._default: float | None = None

        # backchannel thresholds: server-provided only (the local mini model
        # produces no backchannel probability), no overrides, no fallback.
        self._bc_thresholds: dict[str, float] = {}
        self._bc_default: float | None = None

        self._resolve()

    @property
    def model(self) -> TurnDetectorModels:
        return self._model

    @property
    def overrides(self) -> NotGivenOr[float | dict[str, float]]:
        return self._overrides

    @property
    def thresholds(self) -> dict[str, float]:
        return self._thresholds

    @property
    def default_threshold(self) -> float | None:
        return self._default

    def lookup(self, language: LanguageCode | None) -> float | None:
        lang_key = language.language if language else "en"
        return self._thresholds.get(lang_key, self.default_threshold)

    def lookup_backchannel(self, language: LanguageCode | None) -> float | None:
        """Backchannel threshold for a language, or ``None`` when backchannel is
        disabled (no server thresholds provided, or a non-positive value)."""
        if not self._bc_thresholds and not self._bc_default:
            return None
        lang_key = language.language if language else "en"
        threshold = self._bc_thresholds.get(lang_key, self._bc_default)
        return threshold if threshold and threshold > 0 else None

    def supports(self, language: LanguageCode | None) -> bool:
        pending = self._model == "turn-detector-v1" and self._server_thresholds is None
        return pending or self.lookup(language) is not None

    def update_overrides(
        self, overrides: NotGivenOr[float | dict[LanguageCode | str, float]]
    ) -> None:
        self._overrides = _normalize_overrides(overrides)
        self._resolve()

    def _update_defaults(
        self,
        server_thresholds: dict[str, float],
        server_default: float,
        backchannel_thresholds: dict[str, float] | None = None,
        backchannel_default: float = 0.0,
    ) -> None:
        if not server_thresholds or server_default <= 0:
            raise APIError(
                "turn detector session created without usable default thresholds",
                retryable=False,
            )

        self._server_thresholds = {
            LanguageCode(lang).language: round(value, 4)
            for lang, value in server_thresholds.items()
        }
        self._server_default = round(server_default, 4)

        # backchannel defaults are optional; an absent/empty map keeps backchannel disabled
        self._bc_thresholds = {
            LanguageCode(lang).language: round(value, 4)
            for lang, value in (backchannel_thresholds or {}).items()
        }
        self._bc_default = round(backchannel_default, 4) if backchannel_default > 0 else None

        self._resolve()

    def _to_local_fallback(self) -> None:
        if self._model == "turn-detector-v1-mini":
            return

        rescaled: dict[str, float] | None = None
        if server := self._server_thresholds:
            effective = {lang: self.lookup(LanguageCode(lang)) for lang in server}
            rescaled = {
                lang: LOCAL_LANGUAGES[lang] * (active_t / server[lang])
                for lang, active_t in effective.items()
                if active_t is not None and lang in LOCAL_LANGUAGES and server[lang] != 0
            }

        self._model = "turn-detector-v1-mini"
        self._server_thresholds = dict(LOCAL_LANGUAGES)
        self._server_default = LOCAL_LANGUAGES["en"]
        # the mini model produces no backchannel probability
        self._bc_thresholds = {}
        self._bc_default = None
        self._resolve()

        if rescaled is not None:
            self._thresholds = rescaled
            self._default = self.lookup(LanguageCode("en"))

    def _resolve(self) -> None:
        scalar_override = is_given(self._overrides) and not isinstance(self._overrides, dict)
        if self._server_thresholds is None or self._server_default is None:
            # cloud defaults not received yet; only a scalar override resolves up front
            self._thresholds = {}
            self._default = float(cast(float, self._overrides)) if scalar_override else None
            return

        if not is_given(self._overrides):
            self._thresholds, self._default = dict(self._server_thresholds), self._server_default
            return

        if scalar_override:
            self._thresholds, self._default = {}, float(cast(float, self._overrides))
            return

        override = cast("dict[str, float]", self._overrides)
        self._thresholds = {**self._server_thresholds, **override}
        self._default = self._server_default
