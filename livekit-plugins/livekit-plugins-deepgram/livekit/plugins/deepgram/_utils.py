import time
from collections.abc import Callable
from typing import Generic, TypeVar
from urllib.parse import urlencode

T = TypeVar("T")


class PeriodicCollector(Generic[T]):
    def __init__(self, callback: Callable[[T], None], *, duration: float) -> None:
        """
        Create a new periodic collector that accumulates values and calls the callback
        after the specified duration if there are values to report.

        Args:
            duration: Time in seconds between callback invocations
            callback: Function to call with accumulated value when duration expires
        """
        self._duration = duration
        self._callback = callback
        self._last_flush_time = time.monotonic()
        self._total: T | None = None

    def push(self, value: T) -> None:
        """Add a value to the accumulator"""
        if self._total is None:
            self._total = value
        else:
            self._total += value  # type: ignore
        if time.monotonic() - self._last_flush_time >= self._duration:
            self.flush()

    def flush(self) -> None:
        """Force callback to be called with current total if non-zero"""
        if self._total is not None:
            self._callback(self._total)
            self._total = None
        self._last_flush_time = time.monotonic()


def _to_deepgram_url(opts: dict, base_url: str, *, websocket: bool) -> str:
    # don't modify the original opts
    opts = opts.copy()
    if opts.get("keywords"):
        # convert keywords to a list of "keyword:intensifier"
        opts["keywords"] = [
            f"{keyword}:{intensifier}" for (keyword, intensifier) in opts["keywords"]
        ]
    if opts.get("replace"):
        # convert replace dict to a list of "term:replacement"
        # https://developers.deepgram.com/reference/speech-to-text/listen-streaming#query-replace
        opts["replace"] = [f"{term}:{replacement}" for term, replacement in opts["replace"].items()]

    # lowercase bools
    opts = {k: str(v).lower() if isinstance(v, bool) else v for k, v in opts.items()}

    if websocket and base_url.startswith("http"):
        base_url = base_url.replace("http", "ws", 1)

    elif not websocket and base_url.startswith("ws"):
        base_url = base_url.replace("ws", "http", 1)
    return f"{base_url}?{urlencode(opts, doseq=True)}"


# Encodings the LiveKit pipeline can decode, mapped to their mime type. Deepgram TTS
# also offers mulaw/alaw, but those aren't playable through the pipeline yet, so they're
# intentionally omitted here.
_ENCODING_TO_MIMETYPE: dict[str, str] = {
    "linear16": "audio/pcm",
    "mp3": "audio/mp3",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "aac": "audio/aac",
}

# The /speak REST endpoints reject a format parameter the encoding has no use for:
# `container` is only valid for the headerless encodings, and mp3/opus/aac come at a
# single sample rate that can't be requested.
_CONTAINERLESS_ENCODINGS = frozenset({"linear16", "mulaw", "alaw"})
_FIXED_SAMPLE_RATE_ENCODINGS = frozenset({"mp3", "opus", "aac"})


def _encoding_to_mimetype(encoding: str) -> str:
    try:
        return _ENCODING_TO_MIMETYPE[encoding]
    except KeyError:
        raise ValueError(
            f"unsupported Deepgram TTS encoding {encoding!r}; "
            f"supported encodings are: {', '.join(_ENCODING_TO_MIMETYPE)}"
        ) from None


def _tts_http_format_params(encoding: str, sample_rate: int) -> dict:
    """Audio format query parameters for a /speak REST request."""
    params: dict = {"encoding": encoding}
    if encoding in _CONTAINERLESS_ENCODINGS:
        params["container"] = "none"
    if encoding not in _FIXED_SAMPLE_RATE_ENCODINGS:
        params["sample_rate"] = sample_rate
    return params
