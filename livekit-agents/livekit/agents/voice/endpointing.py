from __future__ import annotations

from typing import TypeVar

from ..core import endpointing as core
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from .turn import EndpointingOptions

_T = TypeVar("_T")


def _or_none(value: NotGivenOr[_T]) -> _T | None:
    return value if is_given(value) else None


class BaseEndpointing(core.FixedEndpointing):
    def update_options(
        self, *, min_delay: NotGivenOr[float] = NOT_GIVEN, max_delay: NotGivenOr[float] = NOT_GIVEN
    ) -> None:
        self._handle(
            core.OptionsUpdated(min_delay=_or_none(min_delay), max_delay=_or_none(max_delay))
        )

    def on_start_of_speech(self, started_at: float, overlapping: bool = False) -> None:
        self._handle(core.UserSpeechStarted(at=started_at, overlapping=overlapping))

    def on_end_of_speech(self, ended_at: float, interruption: NotGivenOr[bool] = NOT_GIVEN) -> None:
        # any given value is a verdict, so a falsy one (including None) is a non-interruption
        verdict = bool(interruption) if is_given(interruption) else None
        self._handle(core.UserSpeechEnded(at=ended_at, interruption=verdict))

    def on_start_of_agent_speech(self, started_at: float) -> None:
        self._handle(core.AgentSpeechStarted(at=started_at))

    def on_end_of_agent_speech(self, ended_at: float) -> None:
        self._handle(core.AgentSpeechEnded(at=ended_at))

    def _handle(self, event: core.EndpointingEvent) -> None:
        for output in self.handle(event):
            _log(output)


class DynamicEndpointing(BaseEndpointing, core.DynamicEndpointing):
    def update_options(
        self,
        *,
        min_delay: NotGivenOr[float] = NOT_GIVEN,
        max_delay: NotGivenOr[float] = NOT_GIVEN,
        alpha: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        self._handle(
            core.OptionsUpdated(
                min_delay=_or_none(min_delay),
                max_delay=_or_none(max_delay),
                alpha=_or_none(alpha),
            )
        )


def _log(output: core.EndpointingOutput) -> None:
    if isinstance(output, core.MinDelayUpdated):
        extra: dict[str, str | float] = {"reason": output.reason, "pause": output.pause}
        if output.interruption_delay is not None:
            extra["interruption_delay"] = output.interruption_delay
        if output.turn_delay is not None:
            extra["turn_delay"] = output.turn_delay
        extra["max_delay"] = output.max_delay
        extra["min_delay"] = output.new
        logger.debug(
            "min endpointing delay updated: %s -> %s", output.prev, output.new, extra=extra
        )
    elif isinstance(output, core.UtteranceEndAdjusted):
        logger.trace("utterance ended at adjusted: %s", output.at)
    elif isinstance(output, core.NonInterruptionOverridden):
        logger.trace(
            "overriding non-interruption verdict: user speech started within %.3fs of "
            "agent speech "
            "(within grace period of %.3fs)",
            output.gap,
            core.AGENT_SPEECH_LEADING_SILENCE_GRACE_PERIOD,
        )


def create_endpointing(options: EndpointingOptions) -> BaseEndpointing:
    match options.get("mode", "fixed"):
        case "dynamic":
            return DynamicEndpointing(
                min_delay=options["min_delay"],
                max_delay=options["max_delay"],
                alpha=options["alpha"],
            )
        case _:
            return BaseEndpointing(
                min_delay=options["min_delay"],
                max_delay=options["max_delay"],
            )
