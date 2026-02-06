from enum import Enum


class DtmfEvent(str, Enum):
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    ZERO = "0"
    STAR = "*"
    POUND = "#"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


def dtmf_event_to_code(event: DtmfEvent) -> int:
    if event.value.isdigit():
        return int(event.value)
    elif event.value == "*":
        return 10
    elif event.value == "#":
        return 11
    elif event.value in ["A", "B", "C", "D"]:
        # DTMF codes 10-15 are used for letters A-D
        return ord(event.value) - ord("A") + 12
    else:
        raise ValueError(f"Invalid DTMF event: {event}")


def format_dtmf(events: list[DtmfEvent]) -> str:
    return " ".join(event.value for event in events)
