import asyncio

from ... import function_tool
from ...job import get_job_context
from ..workflows.utils import DtmfEvent, dtmf_event_to_code

DEFAULT_DTMF_PUBLISH_DELAY = 0.3  # seconds to wait between sending DTMF events


@function_tool
async def send_dtmf_events(
    events: list[DtmfEvent],
) -> str:
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
            await asyncio.sleep(DEFAULT_DTMF_PUBLISH_DELAY)
        except Exception as e:
            return f"Failed to send DTMF event: {event.value}. Error: {str(e)}"

    return f"Successfully sent DTMF events: {', '.join(events)}"
