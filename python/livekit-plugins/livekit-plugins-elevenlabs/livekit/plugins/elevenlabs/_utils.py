from collections.abc import Mapping

# ElevenLabs returns a trace id in this response header. It can be shared with their
# support team to debug failed requests.
TRACE_ID_HEADER = "x-trace-id"


def trace_id_from_headers(headers: Mapping[str, str] | None) -> str | None:
    """Return the ElevenLabs `x-trace-id` response header, or None when it is absent."""
    return headers.get(TRACE_ID_HEADER) if headers else None
