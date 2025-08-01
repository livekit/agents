import traceback

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
