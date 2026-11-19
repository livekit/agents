from __future__ import annotations

import asyncio
import copy
import dataclasses
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from opentelemetry import trace

from .._exceptions import APIConnectionError, APIError
from ..log import logger
from ..telemetry import trace_types, tracer
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from ..utils import aio
from .chat_context import ChatContext
from .decisions import (
    Decision,
    DecisionCapabilities,
    DecisionModel,
    DecisionResponse,
    _format_transcript,
    _validate_decisions,
)


@dataclass
class DecisionAvailabilityChangedEvent:
    model: DecisionModel
    available: bool


def _name(model: DecisionModel) -> str:
    # models of one class share a label, so the label alone does not say which one failed
    return f"{model.label} ({model.model})"


@dataclass
class _ModelStatus:
    available: bool = True
    recovering_task: asyncio.Task[None] | None = None


class DecisionModelFallbackAdapter(DecisionModel[Literal["decision_availability_changed"]]):
    """Sends a batch of decisions to the next decision model when the current one fails.

    Models are tried in the given order. A model whose request fails is skipped by later
    requests until a background copy of a request succeeds on it. If every model is marked
    unavailable, all of them are tried again.

    Only a failed request moves on to the next model. A response that answers some decisions
    and reports others in ``errors`` is returned as it is.

    Probabilities and scores are not calibrated across models, so a threshold tuned for one
    model can behave differently on another. ``DecisionResponse.model`` and
    ``DecisionResponse.provider`` name the model that answered.

    Example::

        model = llm.DecisionModelFallbackAdapter(
            [
                inference.DecisionModel("openai/gpt-6-luna-decisions"),
                microsoft_ai.DecisionModel(),
            ]
        )
        response = await model.evaluate(
            input=chat_ctx,
            decisions={"handoff": llm.Probability("The caller requests a human.")},
        )
    """

    def __init__(
        self,
        models: list[DecisionModel],
        *,
        attempt_timeout: float = 1.0,
        max_retry_per_model: int = 0,
        retry_interval: float = 0.5,
    ) -> None:
        """
        Args:
            models: Decision models in priority order.
            attempt_timeout: Timeout for each request to a model, in seconds. It applies to
                every model and replaces a model's own default. Set it from the latency of
                the slowest model: a healthy model that answers later than this is marked
                unavailable. Keep it below any deadline the caller sets on ``evaluate``: a
                model is only marked unavailable when its request fails, not when the caller
                cancels first.
            max_retry_per_model: Retries on the same model before moving on. Defaults to 0,
                which moves on after the first failure.
            retry_interval: Seconds between retries on the same model.

        Raises:
            ValueError: If no model is given, or the models share no decision kind.
        """
        if not models:
            raise ValueError("at least one decision model must be provided")

        # a decision has to be answerable by whichever model ends up serving it
        kinds = [
            kind
            for kind in models[0].capabilities.decision_kinds
            if all(kind in model.capabilities.decision_kinds for model in models[1:])
        ]
        if not kinds:
            raise ValueError("the decision models share no decision kind")
        super().__init__(capabilities=DecisionCapabilities(decision_kinds=kinds))

        self._models = list(models)
        self._attempt_timeout = attempt_timeout
        self._max_retry_per_model = max_retry_per_model
        self._retry_interval = retry_interval
        self._status = [_ModelStatus() for _ in models]
        self._closed = False

        for model in models:
            model.on("metrics_collected", self._on_metrics_collected)

    def _next_instance(self) -> DecisionModel:
        """The first available model, or the primary if all are down."""
        for model, status in zip(self._models, self._status, strict=True):
            if status.available:
                return model
        return self._models[0]

    @property
    def model(self) -> str:
        """The model expected to answer the next request."""
        return self._next_instance().model

    @property
    def provider(self) -> str:
        """The provider expected to answer the next request."""
        return self._next_instance().provider

    @tracer.start_as_current_span("decision_fallback_adapter")
    async def evaluate(
        self,
        *,
        input: str | ChatContext,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> DecisionResponse:
        """Evaluate a batch of decisions on the first model that answers.

        Args:
            input: Text to evaluate, sent as-is, or a conversation converted to a
                transcript of user and assistant text.
            decisions: Decisions keyed by name.
            conn_options: The adapter's ``attempt_timeout``, ``max_retry_per_model`` and
                ``retry_interval`` replace the retry and timeout settings given here.

        Raises:
            APIConnectionError: If every model fails.
        """
        definitions = copy.deepcopy(dict(decisions))
        _validate_decisions(definitions, capabilities=self.capabilities)
        if not definitions:
            # no request is made, so this must not count as a model answering
            return DecisionResponse(results={}, model=self.model, provider=self.provider)

        # every model, and a later recovery check, reads the conversation as it is now
        text = input if isinstance(input, str) else _format_transcript(input)
        return await self._evaluate_impl(
            input=text,
            decisions=definitions,
            conn_options=dataclasses.replace(
                conn_options,
                max_retry=self._max_retry_per_model,
                timeout=self._attempt_timeout,
                retry_interval=self._retry_interval,
            ),
        )

    async def _evaluate_impl(
        self,
        *,
        input: str,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions,
    ) -> DecisionResponse:
        started = time.perf_counter()
        span = trace.get_current_span()

        all_failed = not any(status.available for status in self._status)
        if all_failed:
            logger.error("all decision models are unavailable, retrying..")

        for i, (model, status) in enumerate(zip(self._models, self._status, strict=True)):
            if status.available or all_failed:
                try:
                    # each model emits its own metrics and validates its own response
                    response = await model.evaluate(
                        input=input, decisions=decisions, conn_options=conn_options
                    )
                except APIError as e:
                    logger.warning(
                        "%s failed, switching to next decision model: %s", _name(model), e
                    )
                except Exception:
                    logger.exception(
                        f"{_name(model)} unexpected error, switching to next decision model"
                    )
                else:
                    self._set_available(model, status, True)
                    span.set_attributes(
                        {
                            trace_types.ATTR_FALLBACK_LABEL: model.label,
                            trace_types.ATTR_FALLBACK_INDEX: i,
                        }
                    )
                    return response
                self._set_available(model, status, False)

            self._try_recovery(
                model, status, input=input, decisions=decisions, conn_options=conn_options
            )

        raise APIConnectionError(
            f"all decision models failed ({[_name(model) for model in self._models]}) "
            f"after {time.perf_counter() - started:.2f} seconds"
        )

    def _set_available(self, model: DecisionModel, status: _ModelStatus, available: bool) -> None:
        if status.available == available:
            return
        status.available = available
        self.emit(
            "decision_availability_changed",
            DecisionAvailabilityChangedEvent(model=model, available=available),
        )

    def _try_recovery(
        self,
        model: DecisionModel,
        status: _ModelStatus,
        *,
        input: str,
        decisions: Mapping[str, Decision],
        conn_options: APIConnectOptions,
    ) -> None:
        # a request still running at aclose must not start a check that outlives the adapter
        if self._closed:
            return
        if status.recovering_task is not None and not status.recovering_task.done():
            return

        async def _recover() -> None:
            # the answer is discarded; the span marks the request as a check, not an attempt
            with tracer.start_as_current_span(
                "decision_fallback_adapter.recovery",
                attributes={trace_types.ATTR_FALLBACK_RECOVERY: True},
            ):
                try:
                    await model.evaluate(
                        input=input, decisions=decisions, conn_options=conn_options
                    )
                except Exception as e:
                    logger.debug("%s recovery attempt failed: %s", _name(model), e)
                    return
            logger.info(f"{_name(model)} recovered")
            self._set_available(model, status, True)

        status.recovering_task = asyncio.create_task(_recover())

    async def aclose(self) -> None:
        self._closed = True
        await aio.cancel_and_wait(
            *(s.recovering_task for s in self._status if s.recovering_task is not None)
        )
        for model in self._models:
            model.off("metrics_collected", self._on_metrics_collected)

    def _on_metrics_collected(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)
