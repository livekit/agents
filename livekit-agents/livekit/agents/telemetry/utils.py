import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar, Union

from opentelemetry import trace

from . import trace_types, tracer

if TYPE_CHECKING:
    from ..metrics import RealtimeModelMetrics


@dataclass
class RealtimeSpanContext:
    """Enhanced span context for hierarchical metrics attribution."""

    realtime_turn: trace.Span
    agent_speaking: Union[trace.Span, None] = None
    function_tools: list[trace.Span] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


def _is_openai_realtime_model(metrics: "RealtimeModelMetrics") -> bool:
    """Check if RealtimeModelMetrics comes from OpenAI realtime model.

    :param metrics: The RealtimeModelMetrics event.
    :return: True if metrics are from OpenAI realtime model, False otherwise.
    """
    return "openai" in metrics.label.lower()


class RealtimeSpanManager:
    """Manages OpenTelemetry span attribution for realtime model metrics.

    Provides hierarchical span attribution to ensure metrics are attached to the most
    appropriate active span, preventing timing and scope mismatches.
    """

    def __init__(self, maxsize: int = 100):
        """Initialize the span manager.

        Args:
            maxsize: Maximum number of span contexts to keep in memory
        """
        self._realtime_spans: BoundedSpanDict[str, RealtimeSpanContext] = BoundedSpanDict(
            maxsize=maxsize
        )

    def register_realtime_turn(self, response_id: str, span: trace.Span) -> None:
        """Register a realtime_assistant_turn span.

        Args:
            response_id: The response ID from the generation event
            span: The realtime_assistant_turn span
        """
        self._realtime_spans[response_id] = RealtimeSpanContext(realtime_turn=span)

    def register_agent_speaking_span(self, span: trace.Span) -> None:
        """Register an agent_speaking span with active realtime contexts.

        Args:
            span: The agent_speaking span to register
        """
        # Register the speaking span with all active realtime contexts
        for context in self._realtime_spans.cache.values():
            if context.realtime_turn.is_recording():
                context.agent_speaking = span

    def register_function_tool_span(self, span: trace.Span) -> None:
        """Register a function_tool span with active realtime contexts.

        Args:
            span: The function_tool span to register
        """
        # Register the tool span with all active realtime contexts
        for context in self._realtime_spans.cache.values():
            if context.realtime_turn.is_recording():
                context.function_tools.append(span)

    def _find_best_active_span(self, context: RealtimeSpanContext) -> Union[trace.Span, None]:
        """Find the most appropriate active span for metrics attribution.

        Priority order:
        1. Active function_tool span (most specific)
        2. Active agent_speaking span
        3. Active realtime_turn span

        Args:
            context: The realtime span context

        Returns:
            The best active span, or None if no spans are active
        """
        # 1. Check for active function tool spans (most specific)
        for tool_span in context.function_tools:
            if tool_span.is_recording():
                return tool_span

        # 2. Check for active agent speaking span
        if context.agent_speaking and context.agent_speaking.is_recording():
            return context.agent_speaking

        # 3. Check for active realtime turn span
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

    def _cleanup_span_context(self, context: RealtimeSpanContext) -> None:
        """Clean up all span references in a context to prevent memory leaks.

        Args:
            context: The context to clean up
        """
        # Clear function tool span references
        context.function_tools.clear()
        # Note: We don't need to explicitly clean up the other span references
        # as they're just references that will be garbage collected

    def attach_realtime_metrics_to_span(self, ev: "RealtimeModelMetrics") -> None:
        """Attach realtime model metrics to the most appropriate active OpenTelemetry span.

        Uses hierarchical span attribution to find the best active span:
        1. Active function_tool span (most specific)
        2. Active agent_speaking span
        3. Active realtime_turn span
        4. Create fallback metrics span if all spans have ended

        TODO: expand this code to non OpenAI realtime providers,
        One issue for AWS realtime model would be, at time of writing,
        the ev.request_id is not the same key as the relevant trace

        Args:
            ev: The RealtimeModelMetrics event to process
        """
        from ..log import logger

        if not (ev.type == "realtime_model_metrics" and _is_openai_realtime_model(ev)):
            return

        # Check if we have a stored span context for this request_id
        span_context = self._realtime_spans.get(ev.request_id)
        logger.debug(
            "Realtime Model Metrics telemetry starting",
            extra={
                "span_contexts_count": len(self._realtime_spans),
                "ev": ev.model_dump(mode="json"),
            },
        )
        context_found = span_context is not None

        try:
            if span_context:
                # Find the most appropriate active span for metrics attribution
                target_span = self._find_best_active_span(span_context)

                if target_span:
                    logger.critical(
                        "Adding RealtimeModelMetrics to active span: %s",
                        target_span.name, # type: ignore[attr-defined]
                        extra={
                            "request_id": ev.request_id,
                            "target_span": target_span.name, # type: ignore[attr-defined]
                            "target_span_is_active": target_span.is_recording(),
                        },
                    )

                    # Use the target span as the current context to ensure proper trace attribution
                    with trace.use_span(target_span):
                        self._set_metrics_on_span(target_span, ev)
                else:
                    # All spans have ended, create a dedicated metrics span
                    logger.warning(
                        "All spans have ended, creating fallback metrics span",
                        extra={
                            "request_id": ev.request_id,
                            "realtime_turn_active": span_context.realtime_turn.is_recording(),
                            "agent_speaking_active": span_context.agent_speaking.is_recording()
                            if span_context.agent_speaking
                            else False,
                            "function_tools_count": len(span_context.function_tools),
                        },
                    )
                    self._create_metrics_span(span_context, ev)
            else:
                logger.warning(
                    "No span context found for request_id: indicative of a bug",
                    extra={
                        "request_id": ev.request_id,
                        "available_spans": list(self._realtime_spans.keys()),
                    },
                )
        finally:
            # Always clean up the span context after processing metrics to prevent memory leaks
            # This ensures cleanup happens even if there's an exception during metric processing
            if context_found:
                if span_context:
                    self._cleanup_span_context(span_context)
                self._realtime_spans.pop(ev.request_id, None)

    def clear(self) -> None:
        """Clear all span contexts."""
        self._realtime_spans.clear()


def record_exception(span: trace.Span, exception: Exception) -> None:
    span.record_exception(exception)
    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))
    # set the exception in span attributes in case the exception event is not rendered
    span.set_attributes(
        {
            trace_types.ATTR_EXCEPTION_TYPE: exception.__class__.__name__,
            trace_types.ATTR_EXCEPTION_MESSAGE: str(exception),
            trace_types.ATTR_EXCEPTION_TRACE: traceback.format_exc(),
        }
    )


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
