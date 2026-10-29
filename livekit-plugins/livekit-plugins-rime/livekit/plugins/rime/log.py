import logging

logger = logging.getLogger("livekit.plugins.rime")


class RedactedCause(Exception):
    """A failure's cause with its type and traceback but not its message, which can carry
    credentials, provider responses or user text."""


def redacted(error: BaseException) -> RedactedCause:
    return RedactedCause(type(error).__qualname__).with_traceback(error.__traceback__)
