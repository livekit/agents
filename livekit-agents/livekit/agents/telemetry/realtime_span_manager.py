import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar, Union

from opentelemetry import trace

from ..log import logger
from . import trace_types
from .traces import tracer

if TYPE_CHECKING:
    from ..metrics import RealtimeModelMetrics


@dataclass
class RealtimeSpanContext:
    """Span context for realtime metrics attribution."""

    realtime_turn: trace.Span
    created_at: float = field(default_factory=time.time)


def _is_openai_realtime_model(metrics: "RealtimeModelMetrics") -> bool:
    """Check if RealtimeModelMetrics comes from OpenAI realtime model.

    :param metrics: The RealtimeModelMetrics event.
    :return: True if metrics are from OpenAI realtime model, False otherwise.
    """
    return "openai" in metrics.label.lower()


K = TypeVar("K")
V = TypeVar("V")


class BoundedSpanDict(Generic[K, V]):
    """A bounded dictionary for span references that auto-evicts oldest entries
    to provide extra protection against memory leaks if we did not correctly clean up
    old references to spans.

    Based on the OrderedDict LRU pattern from Python's collections documentation.
    """

    def __init__(self, maxsize: int = 100):
        """Initialize bounded span dictionary.

        :param maxsize: Maximum number of span references to keep.
        """
        self.cache: OrderedDict[K, V] = OrderedDict()
        self.maxsize = maxsize

    def __setitem__(self, key: K, value: V) -> None:
        """Store span reference, evicting oldest if needed."""
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            oldest_key, _ = self.cache.popitem(last=False)

    def get(self, key: K, default: Union[V, None] = None) -> Union[V, None]:
        """Get span reference by key."""
        return self.cache.get(key, default)

    def pop(self, key: K, default: Union[V, None] = None) -> Union[V, None]:
        """Remove and return span reference."""
        return self.cache.pop(key, default)

    def __len__(self) -> int:
        """Return number of span references."""
        return len(self.cache)

    def clear(self) -> None:
        """Remove all span references."""
        self.cache.clear()

    def keys(self) -> list[K]:
        """Return list of all keys."""
        return list(self.cache.keys())


class RealtimeSpanManager:
    """Manages OpenTelemetry span attribution for realtime model metrics.

    Tracks references to realtime_assistant_turn spans to ensure metrics are
    attached to the appropriate active span, preventing timing and scope mismatches.

    This class stores references to existing spans for metrics attribution purposes.
    It sets metrics as attributes on existing spans, or creates new spans dedicated
    to metrics, but is not used to create sub-spans within existing spans.
    """

    def __init__(self, maxsize: int = 100):
        """Initialize the span manager.

        Args:
            maxsize: Maximum number of span contexts to keep in memory
        """
        self._realtime_spans: BoundedSpanDict[str, RealtimeSpanContext] = BoundedSpanDict(
            maxsize=maxsize
        )
        self._model_name: str = "unknown-realtime-model"

    @property
    def model_name(self) -> str:
        """Get the stored model name."""
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        """Set the model name for use in metrics attribution."""
        self._model_name = value

    def track_span(self, response_id: str, span: trace.Span) -> None:
        """Track a realtime_assistant_turn span for metrics attribution.

        Stores a reference to an existing span to enable later metric attribution.

        Args:
            response_id: The response ID from the generation event
            span: The existing realtime_assistant_turn span to track
        """
        self._realtime_spans[response_id] = RealtimeSpanContext(realtime_turn=span)

    def _find_best_active_span(self, context: RealtimeSpanContext) -> Union[trace.Span, None]:
        """Find the most appropriate active span for metrics attribution.

        Args:
            context: The realtime span context

        Returns:
            The realtime_turn span if it's currently recording, None otherwise
        """
        # Check if the realtime turn span is active
        if context.realtime_turn.is_recording():
            return context.realtime_turn

        return None

    def _create_metrics_span(
        self, context: RealtimeSpanContext, metrics: "RealtimeModelMetrics"
    ) -> None:
        """Create a dedicated child span for orphaned metrics.

        Args:
            context: The realtime span context
            metrics: The metrics to attach
        """
        # Create a dedicated child span for orphaned metrics
        with trace.use_span(context.realtime_turn):
            metrics_span = tracer.start_span(
                "realtime_metrics",
                attributes={
                    "request_id": metrics.request_id,
                    "lk.realtime.metrics_type": "orphaned",
                },
            )
            try:
                self._set_metrics_on_span(metrics_span, metrics)
            finally:
                metrics_span.end()

    def _set_metrics_on_span(self, target_span: trace.Span, ev: "RealtimeModelMetrics") -> None:
        """Set realtime model metrics on the given span.

        Args:
            target_span: The span to attach metrics to
            ev: The metrics event
        """
        # Set the model name to ensure Langfuse treats this as a generation (not just a span)
        # LangFuse requires generation observations to be able to assign cost/token counts
        if self._model_name:
            target_span.set_attribute(trace_types.ATTR_GEN_AI_REQUEST_MODEL, self._model_name)

        # Add the full metrics as JSON (following LLM pattern)
        target_span.set_attribute(trace_types.ATTR_REALTIME_MODEL_METRICS, ev.model_dump_json())

        target_span.set_attributes(
            {
                trace_types.ATTR_GEN_AI_USAGE_INPUT_TOKENS: ev.input_tokens,
                trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TOKENS: ev.output_tokens,
                trace_types.ATTR_GEN_AI_USAGE_INPUT_TEXT_TOKENS: ev.input_token_details.text_tokens,
                trace_types.ATTR_GEN_AI_USAGE_INPUT_AUDIO_TOKENS: ev.input_token_details.audio_tokens,
                trace_types.ATTR_GEN_AI_USAGE_INPUT_CACHED_TOKENS: ev.input_token_details.cached_tokens,
                trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TEXT_TOKENS: ev.output_token_details.text_tokens,
                trace_types.ATTR_GEN_AI_USAGE_OUTPUT_AUDIO_TOKENS: ev.output_token_details.audio_tokens,
            }
        )

    def attach_realtime_metrics_to_span(self, ev: "RealtimeModelMetrics") -> None:
        """Attach realtime model metrics to the most appropriate active OpenTelemetry span.

        Uses span attribution to find the active realtime_assistant_turn span:
        1. Active realtime_turn span
        2. Create fallback metrics span if span has ended

        TODO: expand this code to non OpenAI realtime providers,
        One issue for AWS realtime model would be, at time of writing,
        the ev.request_id is not the same key as the relevant trace

        Args:
            ev: The RealtimeModelMetrics event to process
        """
        if not (ev.type == "realtime_model_metrics" and _is_openai_realtime_model(ev)):
            return

        # Check if we have a stored span context for this request_id
        span_context = self._realtime_spans.get(ev.request_id)
        # logger.debug(
        #     "Realtime Model Metrics telemetry starting",
        #     extra={
        #         "span_contexts_count": len(self._realtime_spans),
        #         "ev": ev.model_dump(mode="json"),
        #     },
        # )
        context_found = span_context is not None

        try:
            if span_context:
                # Find the most appropriate active span for metrics attribution
                target_span = self._find_best_active_span(span_context)

                if target_span:
                    # logger.debug(
                    #     "Adding RealtimeModelMetrics to active span: %s",
                    #     target_span.name,  # type: ignore[attr-defined]
                    #     extra={
                    #         "request_id": ev.request_id,
                    #         "target_span": target_span.name,  # type: ignore[attr-defined]
                    #         "target_span_is_active": target_span.is_recording(),
                    #     },
                    # )

                    # Use the target span as the current context to ensure proper trace attribution
                    with trace.use_span(target_span):
                        self._set_metrics_on_span(target_span, ev)

                else:
                    # Realtime turn span has ended, create a dedicated metrics span
                    # It is possible for the realtime_assistant_turn span to end before we collect its metrics
                    # and as spans are immutable in OpenTelemetry, we make a new span to dump the metrics.

                    # logger.debug(
                    #     "All spans have ended, creating fallback metrics span",
                    #     extra={
                    #         "request_id": ev.request_id,
                    #         "realtime_turn_active": span_context.realtime_turn.is_recording(),
                    #     },
                    # )
                    self._create_metrics_span(span_context, ev)
            else:
                logger.warning(
                    "OpenTelemetry Issue: No span context found for request_id: indicative of a bug",
                    extra={
                        "request_id": ev.request_id,
                        "available_spans": list(self._realtime_spans.keys()),
                    },
                )
        finally:
            # Always clean up the span context after processing metrics to prevent memory leaks
            # This ensures cleanup happens even if there's an exception during metric processing
            if context_found:
                self._realtime_spans.pop(ev.request_id, None)

    def clear(self) -> None:
        """Clear all span contexts and clean up references to prevent memory leaks."""
        # Clear the cache
        self._realtime_spans.clear()
