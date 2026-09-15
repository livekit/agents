import logging
import sys

logger = logging.getLogger("livekit.plugins.bithuman")


def print_unsupported_python() -> None:
    """Explain an ImportError caused by the Python version, on versions bithuman has no build for."""
    if (3, 11) <= sys.version_info < (3, 14):
        return

    print(
        "the bithuman SDK is not available on Python "
        f"{sys.version_info.major}.{sys.version_info.minor}, "
        "livekit-plugins-bithuman requires Python 3.11, 3.12 or 3.13",
        flush=True,
    )
