from enum import Enum

from livekit.agents.job import get_job_context
from livekit.agents.llm import function_tool


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


@function_tool
async def send_dtmf_events(
    events: list[DtmfEvent],
) -> None:
    """
    Send a list of DTMF events to the telephony provider.

    Call when:
    - User wants to send a DTMF events sequence

    Do not call when:
    - User only wants to send a single DTMF event
    """
    job_ctx = get_job_context()

    for event in events:
        try:
            code = dtmf_event_to_code(event)
            await job_ctx.room.local_participant.publish_dtmf(code=code, digit=event.value)
        except Exception as e:
            return f"Failed to send DTMF event: {event.value}. Error: {str(e)}"

    return f"Successfully sent DTMF events: {', '.join(events)}"
