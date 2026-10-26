from __future__ import annotations

import asyncio
from dataclasses import dataclass

from livekit import rtc

from ...llm.tool_context import ToolError
from ..workflows.utils import DtmfEvent, format_dtmf


@dataclass
class GetDtmfResult:
    user_input: str

    @classmethod
    def from_dtmf_inputs(cls, dtmf_inputs: list[DtmfEvent]) -> GetDtmfResult:
        return cls(user_input=format_dtmf(dtmf_inputs))


async def receive_dtmf(
    room: rtc.Room,
    *,
    num_digits: int,
    timeout: float = 4.0,
    stop_event: DtmfEvent = DtmfEvent.POUND,
) -> GetDtmfResult:
    """Receive keypad digits; stop at the delimiter or after an inter-digit timeout."""
    if num_digits <= 0:
        raise ValueError("num_digits must be greater than 0")
    queue: asyncio.Queue[str] = asyncio.Queue()

    def on_digit(event: rtc.SipDTMF) -> None:
        queue.put_nowait(event.digit)

    room.on("sip_dtmf_received", on_digit)
    digits: list[DtmfEvent] = []
    try:
        while True:
            try:
                digit = (
                    await asyncio.wait_for(queue.get(), timeout) if digits else await queue.get()
                )
            except asyncio.TimeoutError:
                break
            if digit == stop_event.value:
                break
            digits.append(DtmfEvent(digit))
            if len(digits) == num_digits:
                break
        if len(digits) != num_digits:
            raise ToolError(
                f"Digits input not fully received. Expect {num_digits} digits, got {len(digits)}"
            )
        return GetDtmfResult.from_dtmf_inputs(digits)
    finally:
        room.off("sip_dtmf_received", on_digit)
