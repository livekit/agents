import asyncio
from enum import Enum
from typing import Annotated, Literal

from livekit import rtc
from livekit.agents.job import get_job_context
from livekit.agents.llm import function_tool

DTMF_WAIT_TIMEOUT: float = 10


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
    return "".join(event.value for event in events)


@function_tool
async def send_dtmf_events(
    events: list[DtmfEvent],
) -> None:
    """
    Send a list of DTMF events to the telephony provider.

    Call when:
    - User wants to send DTMF events
    """
    job_ctx = get_job_context()

    for event in events:
        try:
            code = dtmf_event_to_code(event)
            await job_ctx.room.local_participant.publish_dtmf(code=code, digit=event.value)
        except Exception as e:
            return f"Failed to send DTMF event: {event.value}. Error: {str(e)}"

    return f"Successfully sent DTMF events: {', '.join(events)}"


async def collect_dtmf_inputs(
    max_digits: Annotated[
        int | None,
        "The maximum number of DTMF digits to wait for, None if unspecified.",
    ] = None,
    terminator_key: Annotated[
        Literal[DtmfEvent.STAR, DtmfEvent.POUND] | None,
        "The DTMF key to terminate the input collection, None if unspecified.",
    ] = None,
    timeout: Annotated[
        float,
        "The timeout period in seconds to wait for the DTMF inputs, None if unspecified.",
    ] = DTMF_WAIT_TIMEOUT,
) -> list[str]:
    """
    Wait for and collect DTMF (Dual-Tone Multi-Frequency) inputs from the telephony provider.

    This function listens for DTMF key presses (phone keypad inputs) and collects them into a sequence.
    It will automatically stop collecting inputs when:
    - The specified maximum number of digits is reached
    - A terminator key (like '#' or '*') is pressed
    - A timeout period expires without new input (10 seconds by default, resets with each new digit)

    Call when:
    - You need to collect multi-digit input sequences during phone calls
    - You are been prompted as an interactive voice response (IVR) system

    Make sure to actively call this tool along with the Preamble: "Please wait while we collect your input..."

    Returns:
        A list of DTMF digit strings (e.g., ['1', '2', '3', '*']) or an error message string.
    """
    if max_digits is not None and max_digits <= 0:
        raise ValueError("max_digits must be greater than 0")

    job_ctx = get_job_context()
    dtmf_inputs: list[str] = []
    dtmf_inputs_done = asyncio.Event()
    timeout_task: asyncio.Task | None = None

    def _reset_timeout() -> None:
        nonlocal timeout_task
        if timeout_task is not None and not timeout_task.done():
            timeout_task.cancel()

        async def _timeout_handler() -> None:
            await asyncio.sleep(timeout or DTMF_WAIT_TIMEOUT)
            dtmf_inputs_done.set()

        timeout_task = asyncio.create_task(_timeout_handler())

    def _on_dtmf_received(dtmf: rtc.SipDTMF) -> None:
        if dtmf_inputs_done.is_set():
            return

        if terminator_key is not None and dtmf.digit == terminator_key.value:
            dtmf_inputs_done.set()
            return

        dtmf_inputs.append(dtmf.digit)
        _reset_timeout()  # Reset timeout when new digit is received

        if max_digits is not None and len(dtmf_inputs) >= max_digits:
            dtmf_inputs_done.set()
            return

    job_ctx.room.on("sip_dtmf_received", _on_dtmf_received)
    _reset_timeout()  # Start initial timeout

    try:
        await dtmf_inputs_done.wait()
        return dtmf_inputs
    except Exception:
        raise RuntimeError("Something went wrong while waiting for DTMF inputs") from None
    finally:
        job_ctx.room.off("sip_dtmf_received", _on_dtmf_received)
        if timeout_task is not None and not timeout_task.done():
            timeout_task.cancel()
