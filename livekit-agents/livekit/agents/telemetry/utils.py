import traceback
from collections import OrderedDict

from opentelemetry import trace

from . import trace_types


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


class BoundedSpanDict:
    """A bounded dictionary for span references that auto-evicts oldest entries
    to provide extra protection against memory leaks if we did not correctly clean up
    old references to spans.

    Based on the OrderedDict LRU pattern from Python's collections documentation.
    """

    def __init__(self, maxsize: int = 100):
        """Initialize bounded span dictionary.

        :param maxsize: Maximum number of span references to keep.
        """
        self.cache: OrderedDict[str, trace.Span] = OrderedDict()
        self.maxsize = maxsize

    def __setitem__(self, key: str, span: trace.Span) -> None:
        """Store span reference, evicting oldest if needed."""
        self.cache[key] = span
        if len(self.cache) > self.maxsize:
            oldest_key, _ = self.cache.popitem(last=False)

    def get(self, key: str, default: trace.Span | None = None) -> trace.Span | None:
        """Get span reference by key."""
        return self.cache.get(key, default)

    def pop(self, key: str, default: trace.Span | None = None) -> trace.Span | None:
        """Remove and return span reference."""
        return self.cache.pop(key, default)

    def __len__(self) -> int:
        """Return number of span references."""
        return len(self.cache)

    def clear(self) -> None:
        """Remove all span references."""
        self.cache.clear()

    def keys(self) -> list[str]:
        """Return list of all keys."""
        return list(self.cache.keys())