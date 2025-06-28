import json
import logging
from .events import SonicEventBuilder

logger = logging.getLogger("livekit.plugins.aws")


# https://jakob-bagterp.github.io/colorist-for-python/ansi-escape-codes/standard-16-colors/#bright-colors
class AnsiColors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ENDC = "\033[0m"


EVENT_COLOR_MAP = {
    "audio_output_content_start": AnsiColors.GREEN,
    "audio_output_content_end": AnsiColors.GREEN,
    "text_output_content_start": AnsiColors.BLUE,
    "text_output_content_end": AnsiColors.BLUE,
    "tool_output_content_start": AnsiColors.YELLOW,
    "tool_output_content_end": AnsiColors.YELLOW,
    "text_output_content": AnsiColors.BLUE,
    "audio_output_content": AnsiColors.GREEN,
    "tool_output_content": AnsiColors.YELLOW,
    "completion_start": AnsiColors.MAGENTA,
    "completion_end": AnsiColors.MAGENTA,
    "usage": AnsiColors.CYAN,
    "other_event": AnsiColors.UNDERLINE,
}


def log_event_data(event_data: dict) -> None:
    event_type = SonicEventBuilder.get_event_type(event_data)
    color = EVENT_COLOR_MAP[event_type]
    logger.debug(
        f"{color}{event_type.upper()}: {json.dumps(event_data, indent=2)}{AnsiColors.ENDC}"
    )


def log_message(message: str, color: str) -> None:
    logger.debug(f"{color}{message}{AnsiColors.ENDC}")
